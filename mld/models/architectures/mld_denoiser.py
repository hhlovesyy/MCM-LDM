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



# # 定义“场景-风格适配器”，并在前向传播时拦截 Style 特征，用 Scene 特征对其进行“物理修正”。
class SceneStyleAdapter(nn.Module):
    def __init__(self, style_dim=256, scene_dim=512):
        super().__init__()
        # 降维层
        self.scene_mapper = nn.Sequential(
            nn.Linear(scene_dim, style_dim),
            nn.SiLU(),
            nn.Linear(style_dim, style_dim)
        )
        # 门控层 (Channel-wise Gating)
        self.channel_gate = nn.Sequential(
            nn.Linear(style_dim * 2, style_dim),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(style_dim)

    def forward(self, style_feat, scene_feat):
        # style_feat: [Batch, 256]
        # scene_feat: [Batch, 1, 512]
        scene_feat = scene_feat.squeeze(1) # torch.Size([32, 512])
        
        scene_latent = self.scene_mapper(scene_feat) # [B, 256] torch.Size([32, 256])
        
        # 拼接并计算门控
        cat_feat = torch.cat([style_feat, scene_latent], dim=-1) # torch.Size([32, 512])
        gate = self.channel_gate(cat_feat) # torch.Size([32, 256])
        
        # 融合公式：Style * Gate + Scene * (1-Gate)
        refined_style = style_feat * gate + scene_latent * (1 - gate)
        
        return self.norm(refined_style)
    
class SceneStyleAdapterSimple(nn.Module):
    # 非常激进的版本，直接暴力加在一起
    def __init__(self, style_dim=256, scene_dim=512):
        super().__init__()
        
        # 1. 映射层
        self.scene_mapper = nn.Sequential(
            nn.Linear(scene_dim, style_dim),
            nn.SiLU(),
            nn.Linear(style_dim, style_dim)
        )
        
        # 2. 【激进修改】去掉 Sigmoid 门控，改用简单的 Learnable Scale
        # 这是一个可学习的缩放系数，初始化为 0.1 或 1.0，强迫网络接受信号
        self.output_scale = nn.Parameter(torch.ones(1) * 0.1) 
        
        self.norm = nn.LayerNorm(style_dim)

    def forward(self, style_feat, scene_feat):
        scene_feat = scene_feat.squeeze(1)
        scene_latent = self.scene_mapper(scene_feat)
        
        # 【核心修改】残差连接 (Residual Connection)
        # 强制公式：New = Old + alpha * Scene
        # 这样 Style 必须接受 Scene 的干扰，网络被迫去适应这个干扰
        refined_style = style_feat + self.output_scale * scene_latent
        
        return self.norm(refined_style)

class SceneStyleAdapterFiLM(nn.Module):
    def __init__(self, style_dim=256, scene_dim=512):
        super().__init__()
        
        # 【新增】先对 Scene 归一化，解决你的顾虑
        self.scene_norm = nn.LayerNorm(scene_dim) 
        
        # FiLM 生成器
        self.film_generator = nn.Sequential(
            nn.Linear(scene_dim, style_dim * 2),
            nn.SiLU(),
            nn.Linear(style_dim * 2, style_dim * 2)
        )
        self.norm = nn.LayerNorm(style_dim)

    def forward(self, style_feat, scene_feat):
        scene_feat = scene_feat.squeeze(1)
        
        # 1. 先归一化 Scene
        scene_feat = self.scene_norm(scene_feat)
        
        # 2. 生成参数
        film_params = self.film_generator(scene_feat)
        gamma, beta = film_params.chunk(2, dim=-1)
        
        # 3. 调制
        refined_style = style_feat * (1 + gamma) + beta
        return self.norm(refined_style)

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


        self.trans_Encoder = TransEncoder(d_model=256, num_heads=4)
        self.scene_adapter = SceneStyleAdapterFiLM(style_dim=self.latent_dim, scene_dim=512)


    def forward(self,
                sample,
                timestep,
                encoder_hidden_states,
                lengths=None,
                **kwargs):

        sample = sample.permute(1, 0, 2)  # torch.Size([7, 32, 256])

        # time_embedding
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timestep.expand(sample.shape[1]).clone()  # torch.Size([32])，里面的值比如[10,265,985,...]
        time_emb = self.time_proj(timesteps)
        time_emb = time_emb.to(dtype=sample.dtype) # torch.Size([32, 256])
        # [1, bs, latent_dim] <= [bs, latent_dim]
        time_emb = self.time_embedding(time_emb).unsqueeze(0)  # torch.Size([1, 32, 256])

        # three conditions
        style_emb = encoder_hidden_states[1].permute(1, 0, 2)  # torch.Size([1, 32, 512])
        content_emb = encoder_hidden_states[0].permute(1, 0, 2) # torch.Size([7, 32, 256])
        trans_cond = encoder_hidden_states[2] # torch.Size([32, 40, 3])
        scene_emb = encoder_hidden_states[3] # [B, 1, 512] (新增的)
        
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
        xseq = torch.cat((content_emb_latent, sample), axis=0)  # torch.Size([13, 32, 256])

        # style encoder
        style_emb_latent = self.emb_proj_st(style_emb) # torch.Size([1, 32, 256])
        style_emb_latent = time_emb + style_emb_latent
        style_emb_latent = style_emb_latent.squeeze() # torch.Size([32, 256])

        style_emb_final = self.scene_adapter(style_emb_latent, scene_emb) # torch.Size([32, 256])

        # trajectory encoder
        trans_emb = self.trans_Encoder(trans_cond, lengths) # torch.Size([1, 32, 256])
        trans_emb = trans_emb + time_emb
        trans_emb = trans_emb.squeeze() # torch.Size([32, 256])
        
        # to dit blocks (N, T, D)
        xseq = self.query_pos(xseq).permute(1,0,2) # torch.Size([32, 13, 256])
        for block in self.blocks:
            xseq = block(xseq, style_emb_final, trans_emb) # 回顾一下：xseq是content与z拼接后的：torch.Size([32, 13, 256])；style_emb_latent：torch.Size([32, 256])和trans_emb torch.Size([32, 256])是AdaLN的旁路输入
        sample = xseq[:,content_emb_latent.shape[0]:,:] # torch.Size([32, 7, 256])，只取后半部分，也就是z，即sample
       

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
