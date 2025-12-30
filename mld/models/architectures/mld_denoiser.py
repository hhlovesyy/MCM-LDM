# for 813adain_loss
import torch
import torch.nn as nn
from torch import  nn
import torch.nn.functional as F
from mld.models.architectures.tools.embeddings import (TimestepEmbedding,
                                                       Timesteps)
from mld.models.operator import PositionalEncoding
from mld.models.operator.cross_attention import (SkipTransformerEncoder_concat,
                                                 TransformerDecoder,
                                                 TransformerDecoderLayer,
                                                 TransformerEncoder,
                                                 TransformerEncoderLayer_concat,
                                                 TransformerEncoderLayer)
from mld.models.operator.position_encoding import build_position_encoding
from mld.utils.temos_utils import lengths_to_mask
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp



# trans encoder
class TransEncoder(nn.Module):

    def __init__(self, d_model=256, num_heads=4, position_embedding: str = "learned", **block_kwargs):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        seqTransEncoderLayer = nn.TransformerEncoderLayer(self.d_model, self.num_heads)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=2)
        self.global_motion_token = nn.Parameter(
                torch.randn(1, self.d_model))
        self.pe = build_position_encoding(
                self.d_model, position_embedding=position_embedding)

        self.emb_proj_st = nn.Sequential(
            nn.ReLU(), nn.Linear(3, self.d_model))


    def forward(self, x, lengths):
        # 
        if lengths is None:
            lengths = [len(feature) for feature in features]

        device = x.device

        bs, nframes, nfeats = x.shape
        mask = lengths_to_mask(lengths, device)

        # x = features
        # Embed each human poses into latent vectors
        x = self.emb_proj_st(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # Each batch has its own set of tokens
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))

        # create a bigger mask, to allow attend to emb
        dist_masks = torch.ones((bs, dist.shape[0]),
                                dtype=bool,
                                device=x.device)
        aug_mask = torch.cat((dist_masks, mask), 1)

        # adding the embedding token for all sequences
        xseq = torch.cat((dist, x), 0)

        xseq = self.pe(xseq)
        dist = self.seqTransEncoder(xseq,
                                src_key_padding_mask=~aug_mask)[:dist.shape[0]]

        return dist  # torch.Size([1, 32, 256])



class TrajectoryEncoderV2(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. 简单的 MLP 映射：把 (x,y,z) 映射到 Latent Dim
        # 也可以用 1D Conv，但 MLP 最直接
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 2. 轻量级 Transformer (可选，为了提取平滑特征)
        # 如果你想模型更强，可以保留；想更轻量，这层都可以不要，直接用 MLP
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 位置编码 (必须加，因为轨迹有时序)
        self.pe = build_position_encoding(hidden_dim, position_embedding="learned")

    def forward(self, trajectory, lengths=None, target_len=None):
        # trajectory: [Batch, Frames, 4]
        # print("lets check trajectory shape:", trajectory.shape) # lets check trajectory shape
        x = self.input_proj(trajectory) # [Batch, Frames, 256]
        # 1. PE 模块通常期望第 0 维是 Time/Frames
        #    所以我们需要先把 x 从 [B, T, D] 变成 [T, B, D]
        x = x.permute(1, 0, 2) 

        # 加位置编码
        # 注意 pe 的维度处理，这里简化写
        if self.pe is not None:
            # 假设 pe 返回 [Batch, Frames, Dim]
            x = x + self.pe(x) 
        
        x = x.permute(1, 0, 2)
        # Transformer 处理
        # 注意：这里我们**不**加 Global Token，也**不**做 Pooling
        # 我们要的就是序列对序列 (Seq2Seq)
        mask = lengths_to_mask(lengths, x.device) if lengths is not None else None
        
        # output: [Batch, Frames, 256]
        x = self.transformer(x, src_key_padding_mask=~mask)

        if target_len is not None:
            # x: [B, Frames, Dim] -> [B, Dim, Frames]
            x = x.permute(0, 2, 1)
            
            # 强制池化到 target_len (例如 7)
            x = F.adaptive_avg_pool1d(x, output_size=target_len)
            
            # 转回来 -> [B, target_len, 256] # target_len = 7
            x = x.permute(0, 2, 1)
        
        return x








# adaln-zero in dit

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
        self.adaLN_modulation_trans = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )

    def forward(self, x, c, t): # x: torch.Size([32, 13, 256])， c（这里指的是风格，即条件）: torch.Size([32, 256])， t: torch.Size([32, 256])
        shift_msa, scale_msa, gate_msa = self.adaLN_modulation(c).chunk(3, dim=1) # 每个都是torch.Size([32, 256])的tensor
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation_trans(t).chunk(3, dim=1) # 每个都是torch.Size([32, 256])的tensor
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x # torch.Size([32, 13, 256])

class DiTBlockNew(nn.Module):
    """
    Standard DiT block: Condition 'c' controls BOTH Attention and MLP.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        
        # 【修改点】: 现在的 c 要控制一切，所以输出维度从 3x 变成 6x
        # (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        # 删除了 self.adaLN_modulation_trans

    def forward(self, x, c): 
        # x: [Batch, Seq_Len, Dim]
        # c: [Batch, Dim] (Style + Time)
        
        # 一次性切分出 6 个参数
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # MSA (Self-Attention) Part
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        
        # MLP Part
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        return x

class MldDenoiser(nn.Module):

    def __init__(self,
                 ablation,
                 nfeats: int = 263,
                 condition: str = "text",
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 6,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 flip_sin_to_cos: bool = True,
                 return_intermediate_dec: bool = False,
                 position_embedding: str = "learned",
                 arch: str = "trans_enc",
                 freq_shift: int = 0,
                 guidance_scale: float = 7.5,
                 guidance_uncondp: float = 0.1,
                 text_encoded_dim: int = 256,
                 motion_encoded_dim: int = 512,
                 nclasses: int = 10,
                 train_denoiser_config: dict = None,
                 **kwargs) -> None:

        super().__init__()

        self.latent_dim = latent_dim[-1]
        text_encoded_dim = 256
        self.text_encoded_dim = 256
        self.condition = condition
        self.abl_plus = False
        self.arch = arch
        self.motion_encoded_dim = motion_encoded_dim
        self.train_denoiser_config = train_denoiser_config




        # emb proj

        # text condition
        # project time from text_encoded_dim to latent_dim
        self.time_proj = Timesteps(text_encoded_dim, flip_sin_to_cos,
                                    freq_shift)
        self.time_embedding = TimestepEmbedding(text_encoded_dim,
                                                self.latent_dim)

        self.emb_proj = nn.Sequential(
            nn.ReLU(), nn.Linear(text_encoded_dim, self.latent_dim))
        self.emb_proj_st = nn.Sequential(
            nn.ReLU(), nn.Linear(motion_encoded_dim, self.latent_dim))





        self.query_pos = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
            
        self.mem_pos = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)


        if self.train_denoiser_config.INJECTION_MODE == 'concat':  # 强力注入，我们修改升级后的版本
            self.blocks = nn.ModuleList([
                DiTBlockNew( hidden_size=self.latent_dim, num_heads=num_heads, mlp_ratio=4.0) for _ in range(num_layers)
            ])
        else:
        # DIT
            self.blocks = nn.ModuleList([
                DiTBlock( hidden_size=self.latent_dim, num_heads=num_heads, mlp_ratio=4.0) for _ in range(num_layers)
            ])
        

        # IN
        self.IN = nn.InstanceNorm1d(text_encoded_dim, affine=True)

        # transformer for content to remove style
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=4)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=1)
        self.pe_content = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)

        self.linear = nn.Linear(7*256, 6*256)

        if self.train_denoiser_config.ENCODER_TYPE == 'seq': # 新的版本，升级轨迹编码器
            self.trans_Encoder = TrajectoryEncoderV2(input_dim=4, hidden_dim=256, num_layers=2)
            self.fusion_layer = nn.Linear(self.latent_dim + 256, self.latent_dim)
        else:
            self.trans_Encoder = TransEncoder(d_model=256, num_heads=4)
        
        # 【新增】定义三个模态的 Segment Embedding
        # 维度是 [1, 1, latent_dim] 以便广播
        self.seg_emb_content = nn.Parameter(torch.zeros(1, 1, self.latent_dim))
        self.seg_emb_traj = nn.Parameter(torch.zeros(1, 1, self.latent_dim))
        self.seg_emb_sample = nn.Parameter(torch.zeros(1, 1, self.latent_dim))

        # 初始化为小的随机数
        nn.init.normal_(self.seg_emb_content, std=0.02)
        nn.init.normal_(self.seg_emb_traj, std=0.02)
        nn.init.normal_(self.seg_emb_sample, std=0.02)


    def forward(self,
                sample,
                timestep,
                encoder_hidden_states,
                lengths=None,
                **kwargs):
        # 如果是训练，这个sample是原动作加随机timestep噪声的有噪声的动作；如果是推理，这个sample最开始在t=1000的时候是纯噪声，后面是越来越干净的动作
        sample = sample.permute(1, 0, 2)  # torch.Size([7, 32, 256])，sample是加了噪声的z，原始动作加噪声，有轨迹（完完整整的原始动作）
        # print("sample.shape:::", sample.shape)  # 推理的时候是[7,1,256]
        # time_embedding：没动过
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        # print("timestep.shape???? ", timestep.shape) # torch.Size([2])
        timesteps = timestep.expand(sample.shape[1]).clone()  # torch.Size([32])，里面的值比如[10,265,985,...]
        # print("timesteps.shape!!!", timesteps.shape)
        time_emb = self.time_proj(timesteps)
        time_emb = time_emb.to(dtype=sample.dtype) # torch.Size([32, 256])
        # [1, bs, latent_dim] <= [bs, latent_dim]
        time_emb = self.time_embedding(time_emb).unsqueeze(0)  # torch.Size([1, 32, 256])

        # three conditions
        style_emb = encoder_hidden_states[1].permute(1, 0, 2)  # torch.Size([1, 32, 512])
        content_emb = encoder_hidden_states[0].permute(1, 0, 2) # torch.Size([7, 32, 256])
        trans_cond = encoder_hidden_states[-1] # torch.Size([32, 40, 4])
        
        # content        
        content_emb_latent = content_emb
        # style remover for content
        content_emb_latent = self.IN(content_emb_latent.permute(1,2,0)).permute(2,0,1) # torch.Size([7, 32, 256]),【QUESTION】这里面的IN是什么？有什么作用？self.IN = nn.InstanceNorm1d(text_encoded_dim, affine=True)
        content_emb_latent = content_emb_latent+time_emb
        content_emb_latent = self.pe_content(content_emb_latent) # torch.Size([7, 32, 256])
        content_emb_latent = self.seqTransEncoder(content_emb_latent).permute(1,0,2) # torch.Size([32, 7, 256])
        content_emb_latent = self.linear(content_emb_latent.reshape(content_emb_latent.shape[0],-1)).reshape(content_emb_latent.shape[0], 6 ,256) # torch.Size([32, 6, 256])
        content_emb_latent = content_emb_latent.permute(1,0,2) # torch.Size([6, 32, 256])
        # concatenation with sample
        # xseq = torch.cat((content_emb_latent, sample), axis=0)  # torch.Size([13, 32, 256])

        # style encoder： style的也一行没改
        style_emb_latent = self.emb_proj_st(style_emb) # torch.Size([1, 32, 256])
        style_emb_latent = time_emb + style_emb_latent
        style_emb_latent = style_emb_latent.squeeze(0) # torch.Size([32, 256])

        if self.train_denoiser_config.INJECTION_MODE == "concat":

            latent_len = sample.shape[0]  # 7
            # trajectory encoder
            trans_emb = self.trans_Encoder(trans_cond, lengths, target_len=latent_len) # torch.Size([32, 7, 256])
            trans_emb = trans_emb.permute(1, 0, 2) # [7, 32, 256]
            trans_emb = trans_emb + time_emb
            # trans_emb = trans_emb.squeeze() # torch.Size([32, 256])
            sample = self.query_pos(sample)  # torch.Size([7, 32, 256])
            seg_content = content_emb_latent + self.seg_emb_content
            seg_traj = trans_emb + self.seg_emb_traj
            seg_sample = sample + self.seg_emb_sample

            # 这里 query_pos 是类似 build_position_encoding 的正弦编码
            # sample = sample + sample_pe 
            xseq = torch.cat((seg_content, seg_traj, seg_sample), axis=0)  # torch.Size([6 + 7 + 7, 32, 256])
        else:
            xseq = torch.cat((content_emb_latent, sample), axis=0)  # torch.Size([13, 32, 256])
            trans_emb = self.trans_Encoder(trans_cond, lengths) # torch.Size([1, 32, 256])
            trans_emb = trans_emb + time_emb
            trans_emb = trans_emb.squeeze(0) # torch.Size([32, 256]) 
            xseq = self.query_pos(xseq)
        
        # to dit blocks (N, T, D)
        xseq = xseq.permute(1,0,2) # torch.Size([32, 20, 256])
        if self.train_denoiser_config.INJECTION_MODE == "concat":
            for block in self.blocks:
                xseq = block(xseq, style_emb_latent) # 回顾一下：xseq是content与z拼接后的：torch.Size([32, 13, 256])；style_emb_latent：torch.Size([32, 256])和trans_emb torch.Size([32, 256])是AdaLN的旁路输入
            sample = xseq[:,-sample.shape[0]:,:] # torch.Size([32, 7, 256])，只取后半部分，也就是z，即sample
        else:
            for block in self.blocks:
                xseq = block(xseq, style_emb_latent, trans_emb) # 回顾一下：xseq是content与z拼接后的：torch.Size([32, 13, 256])；style_emb_latent：torch.Size([32, 256])和trans_emb torch.Size([32, 256])是AdaLN的旁路输入
            sample = xseq[:,content_emb_latent.shape[0]:,:] # torch.Size([32, 7, 256])，只取后半部分，也就是z，即sample

        return (sample, )    # torch.Size([32, 7, 256])


class EmbedAction(nn.Module):

    def __init__(self,
                 num_actions,
                 latent_dim,
                 guidance_scale=7.5,
                 guidance_uncodp=0.1,
                 force_mask=False):
        super().__init__()
        self.nclasses = num_actions
        self.guidance_scale = guidance_scale
        self.action_embedding = nn.Parameter(
            torch.randn(num_actions, latent_dim))

        self.guidance_uncodp = guidance_uncodp
        self.force_mask = force_mask
        self._reset_parameters()

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        if not self.training and self.guidance_scale > 1.0:
            uncond, output = output.chunk(2)
            uncond_out = self.mask_cond(uncond, force=True)
            out = self.mask_cond(output)
            output = torch.cat((uncond_out, out))

        output = self.mask_cond(output)

        return output.unsqueeze(0)

    def mask_cond(self, output, force=False):
        bs, d = output.shape
        # classifer guidence
        if self.force_mask or force:
            return torch.zeros_like(output)
        elif self.training and self.guidance_uncodp > 0.:
            mask = torch.bernoulli(
                torch.ones(bs, device=output.device) *
                self.guidance_uncodp).view(
                    bs, 1)  # 1-> use null_cond, 0-> use real cond
            return output * (1. - mask)
        else:
            return output

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)