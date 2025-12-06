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

        return dist



class DenseTransEncoder(nn.Module):
    def __init__(self, d_model=256, target_len=6):
        super().__init__()
        self.target_len = target_len
        
        # 输入维度是 4 (RotVel, VelX, VelZ, RootY)
        # 我们用卷积层逐步提取特征并压缩长度
        
        self.net = nn.Sequential(
            # Layer 1: 提取基础运动特征
            # [B, 4, T] -> [B, 64, T]
            nn.Conv1d(4, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.BatchNorm1d(64),
            
            # Layer 2: 下采样 (Stride=2)
            # [B, 64, T] -> [B, 128, T/2]
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.BatchNorm1d(128),
            
            # Layer 3: 继续提取
            # [B, 128, T/2] -> [B, 256, T/2]
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.BatchNorm1d(256),
            
            # Layer 4: 自适应池化到目标长度
            # 在特征提取丰富之后再池化，此时池化的是"语义特征"而不是"原始速度"
            # 比如特征通道 10 代表"左转倾向"，这个特征平均后依然存在，不会被抵消
            nn.AdaptiveAvgPool1d(target_len) 
        )
        
        # 最后的投影，确保数值分布对齐
        self.out_proj = nn.Linear(256, d_model)

    def forward(self, x):
        # x: [Batch, Frames, 4]
        
        # Conv1d 需要 [Batch, Dim, Frames]
        x = x.permute(0, 2, 1) 
        
        # 卷积提取 + 压缩
        x = self.net(x) # -> [Batch, 256, 6]
        
        # 转回 Transformer 格式 [Batch, 6, 256]
        x = x.permute(0, 2, 1)
        
        x = self.out_proj(x) # torch.Size([32, 6, 256])
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
                 **kwargs) -> None:

        super().__init__()

        self.latent_dim = latent_dim[-1]
        text_encoded_dim = 256
        self.text_encoded_dim = 256
        self.condition = condition
        self.abl_plus = False
        self.arch = arch
        self.motion_encoded_dim = motion_encoded_dim





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


        # self.trans_Encoder = TransEncoder(d_model=256, num_heads=4)
        self.trans_Encoder = DenseTransEncoder(d_model=self.latent_dim) # 换成新的
      
    
    def forward(self, sample, timestep, encoder_hidden_states, lengths=None, **kwargs):
        # 1. 基础处理 (Permute, Time Emb...) 保持不变
        sample = sample.permute(1, 0, 2) 
        timesteps = timestep.expand(sample.shape[1]).clone()
        time_emb = self.time_proj(timesteps)
        time_emb = time_emb.to(dtype=sample.dtype)
        time_emb = self.time_embedding(time_emb).unsqueeze(0)

        # 提取条件
        style_emb = encoder_hidden_states[1].permute(1, 0, 2)
        content_emb = encoder_hidden_states[0].permute(1, 0, 2) 
        trans_cond = encoder_hidden_states[-1] 

        # 2. Content 处理 (生成 Clean Context)
        content_emb_latent = content_emb
        content_emb_latent = self.IN(content_emb_latent.permute(1,2,0)).permute(2,0,1)
        content_emb_latent = content_emb_latent + time_emb
        content_emb_latent = self.pe_content(content_emb_latent)
        content_emb_latent = self.seqTransEncoder(content_emb_latent).permute(1,0,2) 
        
        # 降维到 6 个 Token
        content_emb_latent = self.linear(content_emb_latent.reshape(content_emb_latent.shape[0],-1)).reshape(content_emb_latent.shape[0], 6 ,256) 
        content_emb_latent = content_emb_latent.permute(1,0,2) # [6, B, 256]

        # 3. Trajectory 处理 (生成 Clean Traj Features)
        # 这里的 Encoder 输出必须是 [B, 6, 256]
        # 请确保你的 ConvTrajectoryEncoder(target_len=6)
        trans_dense = self.trans_Encoder(trans_cond) # torch.Size([32, 6, 256])
        trans_dense = trans_dense.permute(1, 0, 2) # [6, B, 256] torch.Size([6, 32, 256])

        # ==========================================================
        # 【修正点】 把轨迹加到 Context 上 (Clean + Clean)
        # ==========================================================
        # 这是一个非常合理的特征融合：Context 既包含了姿态(Content)也包含了位移(Traj)
        context_emb = content_emb_latent + trans_dense
        
        # 4. 拼接 (In-Context Conditioning)
        # xseq = [Context(Clean), Sample(Noisy)]
        # Transformer 会让 Sample 去 attend Context
        xseq = torch.cat((context_emb, sample), axis=0) # [13, B, 256]

        # 5. Style 处理 (保持不变)
        style_emb_latent = self.emb_proj_st(style_emb)
        style_emb_latent = time_emb + style_emb_latent
        style_emb_latent = style_emb_latent.squeeze(0)

        # 6. 送入 DiT Blocks
        xseq = self.query_pos(xseq).permute(1,0,2)
        dummy_trans = torch.zeros_like(style_emb_latent)
        
        for block in self.blocks:
            xseq = block(xseq, style_emb_latent, dummy_trans)
            
        sample = xseq[:, content_emb_latent.shape[0]:, :] 
        return (sample, )


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
