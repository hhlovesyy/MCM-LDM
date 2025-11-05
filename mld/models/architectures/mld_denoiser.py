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

# We need a proper Cross-Attention module. 
# We can create a simple wrapper around nn.MultiheadAttention for clarity and correct input handling.
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        # x: (Batch, Seq_len_q, Dim)
        # context: (Batch, Seq_len_kv, Dim)
        B_x, N_x, C = x.shape
        B_c, N_c, C = context.shape
        
        # Project q from x, and k,v from context
        q = self.q_proj(x).reshape(B_x, N_x, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(context).reshape(B_c, N_c, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(context).reshape(B_c, N_c, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_x, N_x, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

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
        
        # 理论上更合理的新设计：
        # 通路1：motion_style 用于调制 Self-Attention (MSA)
        self.adaLN_modulation_motion_style = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
        # 通路2：text_style 也用于调制 Self-Attention (MSA)
        self.adaLN_modulation_text_style = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
        # 通路3：trajectory 用于调制 MLP
        self.adaLN_modulation_trajectory = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
        
        # 为了代码兼容性，我们将原来的变量名重命名，指向新的正确的模块
        # 这样 MldDenoiser 类中的代码几乎不用改
        self.adaLN_modulation = self.adaLN_modulation_motion_style
        self.adaLN_modulation_text = self.adaLN_modulation_text_style
        self.adaLN_modulation_trans = self.adaLN_modulation_trajectory

        
    def forward(self, x, c_motion, c_traj, c_text): # 移除 c_text=None 默认值，因为它总被提供
        # 理论上更合理的新设计：
        
        # 1. 统一和融合风格条件 (motion_style 和 text_style) 来调制 Self-Attention (MSA)
        # a. 分别计算两种风格的调制参数
        shift_msa_motion, scale_msa_motion, gate_msa_motion = self.adaLN_modulation_motion_style(c_motion).chunk(3, dim=1)
        shift_msa_text, scale_msa_text, gate_msa_text = self.adaLN_modulation_text_style(c_text).chunk(3, dim=1)

        # b. [优雅的融合] 直接相加。
        #    这个设计的精妙之处在于，上游逻辑已经保证了：
        #    - 对于 motion-guided 样本, c_motion 是有效信号, c_text 是从零向量派生的“噪声”信号。
        #    - 对于 text-guided 样本, c_motion 是从零向量派生的“噪声”信号, c_text 是有效信号。
        #    相加后，有效信号会主导，而“噪声”信号的影响可以被网络通过训练学会忽略。
        #    这比在 forward 中用 if/else 分支更高效，对 GPU 更友好。
        shift_msa_final = shift_msa_motion + shift_msa_text
        scale_msa_final = scale_msa_motion + scale_msa_text
        gate_msa_final = gate_msa_motion + gate_msa_text
        
        # c. 应用最终的、被统一风格信号调制的 Self-Attention
        x = x + gate_msa_final.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa_final, scale_msa_final))

        # 2. 轨迹条件 (c_traj) 保持不变，继续调制 MLP 模块
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation_trajectory(c_traj).chunk(3, dim=1)
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
                 text_style_dim: int = 512,
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

        # NEW: Add a projection layer for the text style feature
        self.text_style_proj = nn.Sequential( 
            nn.ReLU(), 
            nn.Linear(text_style_dim, self.latent_dim) 
        ) 

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
        # [新代码] 为轨迹条件添加一个专门的归一化层
        self.traj_norm = nn.LayerNorm(3) # 输入特征维度是 3 (x,y,z)

        # # --- [尝试: 冻结主体，只训练 Adapter] 这个目前效果是很差的，记录在这不要再踩一样的坑---
       
    def forward(self,
                sample, # torch.Size([32, 7, 256])， 牢记32是batch_size
                timestep, # torch.Size([32])
                encoder_hidden_states,  # [cond_emb, motion_emb, trans_cond]
                lengths=None,  # len() 32
                style_text_feature=None, # NEW: Add the optional parameter
                **kwargs):

        sample = sample.permute(1, 0, 2)  # torch.Size([7, 32, 256])

        # time_embedding
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timestep.expand(sample.shape[1]).clone() # torch.Size([32]), 这句是在做什么？
        time_emb = self.time_proj(timesteps) # torch.Size([32, 256])
        time_emb = time_emb.to(dtype=sample.dtype)
        # [bs, latent_dim] => [1, bs, latent_dim]
        time_emb = self.time_embedding(time_emb).unsqueeze(0)  # torch.Size([1, 32, 256])

        # three conditions
        style_emb = encoder_hidden_states[1].permute(1, 0, 2)  # torch.Size([1, 32, 512])
        content_emb = encoder_hidden_states[0].permute(1, 0, 2)  # torch.Size([7, 32, 256])
        trans_cond = encoder_hidden_states[-1]  # torch.Size([32, 28, 3])
        
        # content        
        content_emb_latent = content_emb  # torch.Size([7, 32, 256])
        # style remover for content
        content_emb_latent = self.IN(content_emb_latent.permute(1,2,0)).permute(2,0,1)  # torch.Size([7, 32, 256])
        content_emb_latent = content_emb_latent+time_emb  # torch.Size([7, 32, 256])
        content_emb_latent = self.pe_content(content_emb_latent)  # torch.Size([7, 32, 256])
        content_emb_latent = self.seqTransEncoder(content_emb_latent).permute(1,0,2)  # torch.Size([32, 7, 256])
        content_emb_latent = self.linear(content_emb_latent.reshape(content_emb_latent.shape[0],-1)).reshape(content_emb_latent.shape[0], 6 ,256)  # torch.Size([32, 6, 256])
        content_emb_latent = content_emb_latent.permute(1,0,2)  # torch.Size([6, 32, 256])
        # concatenation with sample
        xseq = torch.cat((content_emb_latent, sample), axis=0) # torch.Size([13, 32, 256])

        # style encoder
        style_emb_latent = self.emb_proj_st(style_emb)  # style_emb_latent：torch.Size([1, 32, 256])，输入函数的style_emb：torch.Size([1, 32, 512])
        style_emb_latent = time_emb + style_emb_latent  # torch.Size([1, 32, 256])
        style_emb_latent = style_emb_latent.squeeze()  # torch.Size([32, 256])

        # trajectory encoder  本来的，但是发现在train的过程中可能会梯度爆炸，所以加了一步LN
        # trans_emb = self.trans_Encoder(trans_cond, lengths) # torch.Size([1, 32, 256])
        # [修改后]
        # a. 先对轨迹条件进行 Layer Normalization
        normalized_trans_cond = self.traj_norm(trans_cond)
        # b. 再将归一化后的结果送入编码器
        trans_emb = self.trans_Encoder(normalized_trans_cond, lengths)
        trans_emb = trans_emb + time_emb  # torch.Size([1, 32, 256])
        trans_emb = trans_emb.squeeze()  # torch.Size([32, 256])

        # --- NEW: Text Style Path ---
        text_emb_latent = None 
        if style_text_feature is not None: 
            # Project text feature to the latent dimension
            text_emb_latent = self.text_style_proj(style_text_feature.permute(1, 0, 2)) # Input (B, 1, 512) -> (1, B, 512)
            # Add time embedding, same as other conditions
            text_emb_latent = time_emb + text_emb_latent 
            text_emb_latent = text_emb_latent.squeeze(0) # Squeeze to (B, D)
        
        # to dit blocks (N, T, D)
        xseq = self.query_pos(xseq).permute(1,0,2)  # torch.Size([32, 13, 256])
        for block in self.blocks:
            xseq = block(x=xseq, 
                 c_motion=style_emb_latent, 
                 c_traj=trans_emb, 
                 c_text=text_emb_latent) 
        sample = xseq[:,content_emb_latent.shape[0]:,:]  # torch.Size([32, 7, 256])
       

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