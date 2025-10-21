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

# --- [新增] 第2步：定义我们的 CrossAttention 手术工具 ---
# 将这个类直接放在文件顶部，或者放在一个单独的文件中然后导入
# 为了简单，我们直接放在这里。
class CrossAttention(nn.Module):
    """一个标准的多头交叉注意力模块"""
    def __init__(self, query_dim, context_dim=None, n_heads=8, d_head=64, dropout=0.):
        super().__init__()
        inner_dim = d_head * n_heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.n_heads = n_heads
        self.scale = d_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, context=None, mask=None):
        context = context if context is not None else query
        
        q, k, v = self.to_q(query), self.to_k(context), self.to_v(context)
        
        q, k, v = map(
            lambda t: t.view(t.shape[0], -1, self.n_heads, t.shape[-1] // self.n_heads).permute(0, 2, 1, 3),
            (q, k, v)
        )  
        
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # 想不起来的话这两句可以问一下AI
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        output = torch.matmul(attention_probs, v)
        
        # 强制 PyTorch 重新分配内存，并将非连续存储的张量数据按其当前的逻辑顺序复制到一块新的、连续的内存区域中。
        output = output.permute(0, 2, 1, 3).contiguous().view(output.shape[0], -1, q.shape[-1] * self.n_heads)
        return self.to_out(output)

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

        # 新增cross attention
        # --- [新增] Cross-Attention 层及其 LayerNorm ---
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(
            query_dim=hidden_size, 
            n_heads=num_heads, 
            d_head=hidden_size // num_heads, # 确保 d_head 正确
            context_dim=hidden_size # 假设文本特征已被投影到 hidden_size
        )
        # -----------------------------------------------

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

    # --- [修改] forward 方法以接受 text_context ---
    def forward(self, x, c, t, text_context=None):
         # 1. Self-Attention Block (adaLN-Zero for motion style)
        #    c 是来自 MotionCLIP 的风格条件
        #    如果 c 是 None，modulate 也能正常工作（shift=0, scale=0）

        # 以下是原来的
        # shift_msa, scale_msa, gate_msa = self.adaLN_modulation(c).chunk(3, dim=1)
        # shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation_trans(t).chunk(3, dim=1)
        # x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        # return x
        shift_msa, scale_msa, gate_msa = self.adaLN_modulation(c).chunk(3, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))

        # --- [新增] Cross-Attention Block for text style ---
        if text_context is not None:
            x = x + self.cross_attn(query=self.norm_cross(x), context=text_context)
        # ----------------------------------------------------
        
        # 3. Feed-Forward Block (adaLN-Zero for trajectory)
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation_trans(t).chunk(3, dim=1)
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
                 **kwargs) -> None:

        super().__init__()

        self.latent_dim = latent_dim[-1]
        text_encoded_dim = 256
        self.text_encoded_dim = 256
        self.condition = condition
        self.abl_plus = False
        self.arch = arch
        self.motion_encoded_dim = motion_encoded_dim


        # --- [新增] 从 kwargs 中获取新配置，并设置默认值 ---
        # 这样做的好处是，如果旧的 YAML 文件没有这些配置，代码也能正常运行
        self.use_text_condition = kwargs.get('use_text_condition', False)
        clip_feature_dim = kwargs.get('clip_feature_dim', 512)
        # --------------------------------------------------


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


        # --- [新增] 文本投影层 ---
        if self.use_text_condition:
            self.text_context_proj = nn.Linear(clip_feature_dim, self.latent_dim)
        # ---------------------------


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



    def forward(self,
                sample,
                timestep,
                encoder_hidden_states,
                lengths=None,
                style_text_feature=None,
                **kwargs):

        # --- [核心修复] ---
        # 1. 统一内部处理维度为 (SeqLen, Batch, Dim)
        #    原始的 sample 形状是 (B, D, L) 或 (B, L, D)，我们需要确定
        #    根据后续操作，模型期望 sample 是 (L, B, D)
        #    VAE 的输出通常是 (B, D, L) 的 latent。我们需要把它变成 (L, B, D)
        #    所以正确的操作应该是 permute(2, 0, 1)
        #
        #    让我们假设 VAE 输出就是 (B, D, L)
        sample = sample.permute(2, 0, 1)  # (B, D, L) -> (L, B, D)
        
        # --------------------

        # time_embedding
        timesteps = timestep.expand(sample.shape[1]).clone()
        time_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        time_emb = self.time_embedding(time_emb).unsqueeze(0) # [1, bs, latent_dim]

        # --- 双模态路由逻辑 ---
        text_context = None
        # [修改] 文本特征也需要是 (Batch, Seq, Dim) 的格式给 CrossAttention
        if self.use_text_condition and style_text_feature is not None:
            # style_text_feature from CLIP is (B, D_clip)
            projected_text = self.text_context_proj(style_text_feature) # (B, latent_dim)
            text_context = projected_text.unsqueeze(1) # (B, 1, latent_dim)

        # --- 准备 adaLN 的风格和轨迹条件 (保持 bs 在前) ---
        motion_style_cond = None
        style_emb = encoder_hidden_states[1] # [1, bs, 512]
        if style_emb is not None:
            style_emb = style_emb.permute(1, 0, 2) # [bs, 1, 512]
            style_emb_latent = self.emb_proj_st(style_emb) # [bs, 1, 256]
            motion_style_cond = (time_emb.permute(1,0,2) + style_emb_latent).squeeze(1) # [bs, 256]
        else:
            batch_size = sample.shape[1]
            motion_style_cond = torch.zeros(batch_size, self.latent_dim, device=sample.device)
        
        trans_cond = encoder_hidden_states[-1] # [bs, nframes, 3]
        trans_emb = self.trans_Encoder(trans_cond, lengths) # [1, bs, 256]
        trans_emb = (time_emb + trans_emb).squeeze(0) # [bs, 256]
        
        # --- 准备内容条件 (保持 L 在前) ---
        content_emb = encoder_hidden_states[0] # [L_content, bs, 256]
        content_emb_latent = self.IN(content_emb.permute(1,2,0)).permute(2,0,1)
        content_emb_latent = content_emb_latent + time_emb
        content_emb_latent = self.pe_content(content_emb_latent)
        content_emb_latent = self.seqTransEncoder(content_emb_latent) # 输出仍是 (L_c, bs, D)
        # [修改] 下面这部分 reshape 操作非常危险，我们先简化它
        content_emb_reshaped = content_emb_latent.permute(1,0,2).reshape(content_emb_latent.shape[1], -1)
        content_emb_linear = self.linear(content_emb_reshaped)
        content_emb_final = content_emb_linear.reshape(content_emb_latent.shape[1], 6, 256).permute(1,0,2)
        
        # --- 准备 DiTBlock 的输入序列 (保持 L 在前) ---
        xseq = torch.cat((content_emb_final, sample), axis=0) # [L_c + L_s, bs, D]
        xseq = self.query_pos(xseq) # 添加位置编码，形状不变
        
        # --- [核心修复] 将数据转换为 DiTBlock 期望的 (B, L, D) ---
        xseq = xseq.permute(1, 0, 2) # (L, B, D) -> (B, L, D)

        # --- 循环调用改造后的 DiTBlock ---
        for block in self.blocks:
            xseq = block(
                xseq, 
                c=motion_style_cond,
                t=trans_emb,
                text_context=text_context
            )

        # --- [核心修复] 将 DiTBlock 的输出转换回内部处理格式 (L, B, D) ---
        xseq = xseq.permute(1, 0, 2) # (B, L, D) -> (L, B, D)

        # --- 提取去噪后的样本部分 ---
        sample_denoised = xseq[content_emb_final.shape[0]:, :, :] # (L_s, bs, D)
    
        # --- [核心修复] 将最终输出转换为模型外部期望的 (B, D, L) ---
        output = sample_denoised.permute(1, 2, 0) # (L, B, D) -> (B, D, L)
        
        return (output, )
        
        # # ---------------------------
        # content_emb = encoder_hidden_states[0].permute(1, 0, 2)
        # trans_cond = encoder_hidden_states[-1]
        
        # # content        
        # content_emb_latent = content_emb
        # # style remover for content
        # content_emb_latent = self.IN(content_emb_latent.permute(1,2,0)).permute(2,0,1)
        # content_emb_latent = content_emb_latent+time_emb
        # content_emb_latent = self.pe_content(content_emb_latent)
        # content_emb_latent = self.seqTransEncoder(content_emb_latent).permute(1,0,2)
        # content_emb_latent = self.linear(content_emb_latent.reshape(content_emb_latent.shape[0],-1)).reshape(content_emb_latent.shape[0], 6 ,256)
        # content_emb_latent = content_emb_latent.permute(1,0,2)
        # # concatenation with sample
        # xseq = torch.cat((content_emb_latent, sample), axis=0)

        # # style encoder
        # style_emb_latent = self.emb_proj_st(style_emb)
        # style_emb_latent = time_emb + style_emb_latent
        # style_emb_latent = style_emb_latent.squeeze()

        # # trajectory encoder
        # trans_emb = self.trans_Encoder(trans_cond, lengths)
        # trans_emb = trans_emb + time_emb
        # trans_emb = trans_emb.squeeze()
        
        # # to dit blocks (N, T, D)
        # xseq = self.query_pos(xseq).permute(1,0,2)
        # for block in self.blocks:
        #     xseq = block(xseq, style_emb_latent, trans_emb) 
        # sample = xseq[:,content_emb_latent.shape[0]:,:]
       

        # return (sample, )        


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
