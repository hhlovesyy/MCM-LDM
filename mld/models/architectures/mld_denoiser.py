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
            # Fallback for fixed length batches
            lengths = [x.shape[1]] * x.shape[0]

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

# --- 3. [PhysiMoS 核心] 新增 PhysicsAdapter ---
# 这是一个轻量级的适配器，用于将物理嵌入注入到 DiTBlock 中。
# 关键点：Zero Initialization。
class PhysicsAdapter(nn.Module):
    """
    [升级版] Physics-AdaLN 适配器
    输入: 物理嵌入 (Batch, Dim)
    输出: 用于调制特征的 Scale 和 Shift
    """
    def __init__(self, hidden_size, act_layer=nn.SiLU):
        super().__init__()
        self.act = act_layer()
        
        # 映射到 2 倍维度 (Scale + Shift)
        self.linear = nn.Linear(hidden_size, 2 * hidden_size) 
        
        # [Zero-Init 策略]
        # 初始化为 0，意味着 Scale=0, Shift=0
        # 实际使用时我们会让 Scale = 1 + 0 = 1 (保持原样)，Shift = 0
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, phys_emb):
        """
        x: [Batch, Seq, Dim] (特征图)
        phys_emb: [Batch, Dim] (物理参数)
        """
        # 1. 计算调制参数
        # phys_emb 经过激活后映射
        style = self.linear(self.act(phys_emb)) # [Batch, 2 * Dim]
        
        # 2. 拆分 Scale (gamma) 和 Shift (beta)
        gamma, beta = style.chunk(2, dim=1) # [Batch, Dim] each
        
        # 3. 扩展维度以便广播: [Batch, 1, Dim]
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        
        # 4. 执行 AdaLN 调制
        # 公式: x * (1 + gamma) + beta
        # 这里的 x 应该是经过 LayerNorm 之后的，我们在 Block 里处理
        return gamma, beta

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

    def forward(self, x, c, t):
        shift_msa, scale_msa, gate_msa = self.adaLN_modulation(c).chunk(3, dim=1)
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation_trans(t).chunk(3, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    

# --- 4. [PhysiMoS 修复] DiTBlock_Phys ---
# 我们继承原版 DiTBlock 的逻辑，但稍作修改以接纳物理信息。
# 这样我们可以加载原版权重（除了新增的 adapter）。
class DiTBlock_Phys(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        
        # [保留原版 AdaLN] 
        # 这样做的好处是，我们可以加载预训练权重。
        # 即使我们在推理时把 style 设为 null，我们也希望保留这个结构，
        # 或者我们可以微调这个模块来适应新的“物理风格”。
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
        self.adaLN_modulation_trans = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
        
        # [新增] 物理适配器
        self.phys_adapter_attn = PhysicsAdapter(hidden_size)
        self.phys_adapter_mlp = PhysicsAdapter(hidden_size)

    def forward(self, x, style_emb, trans_emb, phys_emb):
        """
        x: [Batch, Seq, Dim] (Batch First here for internal computation)
        style_emb: [Batch, Dim] (Original Style or Null)
        trans_emb: [Batch, Dim] (Trajectory Info)
        phys_emb: [Batch, Dim] (New Physics Info from SCPAEncoder)
        """
        # 1. 原版 Style Modulation
        shift_msa, scale_msa, gate_msa = self.adaLN_modulation(style_emb).chunk(3, dim=1)
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation_trans(trans_emb).chunk(3, dim=1)
        
        # === [DEBUG 测试] 强制屏蔽物理影响 ===
        # 我们让 scale=0, shift=0，这样 (1+0)*x + 0 = x，完全等于没加
        # p_scale1 = torch.zeros_like(x) 
        # p_shift1 = torch.zeros_like(x)
        # p_scale2 = torch.zeros_like(x)
        # p_shift2 = torch.zeros_like(x)
        
        # 注释掉原来的计算，或者保留上面那几行zeros
        # p_scale1, p_shift1 = self.phys_adapter_attn(x, phys_emb) <-- 注释掉这行
        # p_scale2, p_shift2 = self.phys_adapter_mlp(x, phys_emb)  <-- 注释掉这行
        # ===================================
        
        
        # 2. [核心修改] 物理调制参数 (Scale & Shift)
        p_scale1, p_shift1 = self.phys_adapter_attn(x, phys_emb)
        p_scale2, p_shift2 = self.phys_adapter_mlp(x, phys_emb)
        
         # === 救命稻草 ===
        # 既然模型学得太 aggressive 了，我们给它打个 0.2 折
        # 这样它对原动作的破坏力就只有原来的 20%
        # 但这同时也意味着物理效果（低头）也会变弱，需要找平衡点
        scale_factor = 0.9 
        
        # x_mod1 = x_mod1 * (1 + p_scale1 * scale_factor) + (p_shift1 * scale_factor)
        # # ...
        # x_mod2 = x_mod2 * (1 + p_scale2 * scale_factor) + (p_shift2 * scale_factor)

        # 3. 混合调制逻辑
        # 我们希望物理影响也是全局的。
        # 原版逻辑: modulate(norm(x), shift, scale)
        # 新版逻辑: 我们在原版 modulate 的基础上，再叠一层物理 modulate
        # --- Block 1: Attention ---
        x_norm1 = self.norm1(x)
        # 先应用原版 Time/Style 调制
        x_mod1 = modulate(x_norm1, shift_msa, scale_msa)
        # [新增] 再应用物理调制: x * (1 + p_scale) + p_shift
        # x_mod1 = x_mod1 * (1 + p_scale1) + p_shift1
        x_mod1 = x_mod1 * (1 + p_scale1 * scale_factor) + (p_shift1 * scale_factor)

        # 计算 Attention 并加残差 (gate_msa 控制原版权重的门控，依然保留)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_mod1)
        
        # --- Block 2: MLP ---
        x_norm2 = self.norm2(x)
        # 先应用原版 Time/Trajectory 调制
        x_mod2 = modulate(x_norm2, shift_mlp, scale_mlp)
        # [新增] 再应用物理调制
        # x_mod2 = x_mod2 * (1 + p_scale2) + p_shift2
        x_mod2 = x_mod2 * (1 + p_scale2 * scale_factor) + (p_shift2 * scale_factor)


        # 计算 MLP 并加残差
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mod2)
        
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



    def forward(self,
                sample,
                timestep,
                encoder_hidden_states,
                lengths=None,
                **kwargs):

        sample = sample.permute(1, 0, 2)

        # time_embedding
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timestep.expand(sample.shape[1]).clone()
        time_emb = self.time_proj(timesteps)
        time_emb = time_emb.to(dtype=sample.dtype)
        # [1, bs, latent_dim] <= [bs, latent_dim]
        time_emb = self.time_embedding(time_emb).unsqueeze(0)

        # three conditions
        style_emb = encoder_hidden_states[1].permute(1, 0, 2)
        content_emb = encoder_hidden_states[0].permute(1, 0, 2)
        trans_cond = encoder_hidden_states[-1]
        
        # content        
        content_emb_latent = content_emb
        # style remover for content
        content_emb_latent = self.IN(content_emb_latent.permute(1,2,0)).permute(2,0,1)
        content_emb_latent = content_emb_latent+time_emb
        content_emb_latent = self.pe_content(content_emb_latent)
        content_emb_latent = self.seqTransEncoder(content_emb_latent).permute(1,0,2)
        content_emb_latent = self.linear(content_emb_latent.reshape(content_emb_latent.shape[0],-1)).reshape(content_emb_latent.shape[0], 6 ,256)
        content_emb_latent = content_emb_latent.permute(1,0,2)
        # concatenation with sample
        xseq = torch.cat((content_emb_latent, sample), axis=0)

        # style encoder
        style_emb_latent = self.emb_proj_st(style_emb)
        style_emb_latent = time_emb + style_emb_latent
        style_emb_latent = style_emb_latent.squeeze()

        # trajectory encoder
        trans_emb = self.trans_Encoder(trans_cond, lengths)
        trans_emb = trans_emb + time_emb
        trans_emb = trans_emb.squeeze()
        
        # to dit blocks (N, T, D)
        xseq = self.query_pos(xseq).permute(1,0,2)
        for block in self.blocks:
            xseq = block(xseq, style_emb_latent, trans_emb) 
        sample = xseq[:,content_emb_latent.shape[0]:,:]
       

        return (sample, )        


class MldDenoiserNew(nn.Module):
    """
    PhysiMoS 修复版 Denoiser。
    核心策略：复用原版架构，通过 Adapter 注入物理信息。
    """
    def __init__(self,
                 nfeats: int = 263,
                 latent_dim: list = [1, 256],
                 num_layers: int = 6,
                 num_heads: int = 4,
                 position_embedding: str = "learned",
                 **kwargs) -> None:
        super().__init__()
        self.latent_dim = latent_dim[-1]
        # 1. Time Embedding (保持原版)
        self.time_proj = Timesteps(self.latent_dim, True, 0)
        self.time_embedding = TimestepEmbedding(self.latent_dim, self.latent_dim)
        # 2. Condition Projections (保持原版以加载权重)
        # 即使我们可能不用 style，保留这个层可以避免加载权重报错，或者我们可以用它把 Time 映射进去作为 Null Style。
        self.emb_proj_st = nn.Sequential(nn.ReLU(), nn.Linear(512, self.latent_dim)) # 假设原 MotionCLIP 维数是 512

        # 3. Trajectory Encoder (恢复原版)
        self.trans_Encoder = TransEncoder(d_model=self.latent_dim, num_heads=4)
        
        # 4. Positional Encodings (保持原版)
        self.query_pos = build_position_encoding(self.latent_dim, position_embedding=position_embedding)

        # 5. Blocks (使用修复版 DiTBlock_Phys)
        self.blocks = nn.ModuleList([
            DiTBlock_Phys(hidden_size=self.latent_dim, num_heads=num_heads) 
            for _ in range(num_layers)
        ])

        # 6. Style Remover / Content Process (保留原版逻辑)
        # 原版在这里做了很复杂的操作：IN -> TransEnc -> Linear
        # 我们必须保留这个逻辑，因为 Content 的 Latent Feature 分布是经过这些层调整的。
        self.IN = nn.InstanceNorm1d(256, affine=True) # text_encoded_dim 假定 256
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=4)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=1)
        self.pe_content = build_position_encoding(self.latent_dim, position_embedding=position_embedding)
        # 原版从 7*256 -> 6*256，这一步我们保留
        self.linear = nn.Linear(7*256, 6*256)
    
    # 【NOTE：这个函数跟原版的整体数据流依旧有一些不同的地方，如果有报错的话需要回来看一下】
    def forward(self, sample, timestep, encoder_hidden_states, lengths=None, **kwargs):
        """
        严格复刻原版 forward 的数据流，仅在 Block 调用时注入 Physics。
        """
        # [复刻] 1. 初始 Permute: [B, S, D] -> [S, B, D]
        sample = sample.permute(1, 0, 2) # torch.Size([7, bs, 256])

        # [复刻] 2. Time Embedding 处理
        # 这里保持 sample.shape[1] 作为 batch size，完全正确
        timesteps = timestep.expand(sample.shape[1]).clone()
        time_emb = self.time_proj(timesteps)
        time_emb = time_emb.to(dtype=sample.dtype)
        # [1, B, D]
        time_emb = self.time_embedding(time_emb).unsqueeze(0) # torch.Size([1, bs, 256])

        # [解包条件]
        # encoder_hidden_states 列表顺序由 mld.py 决定，假设为:
        # [0]: Content [S, B, D]
        # [1]: Physics [B, 1, D] (这是你的 SCPAEncoder 输出)
        # [2]: Trajectory [B, S, 3] (这是 Dataset 的 raw trajectory)
        content_emb = encoder_hidden_states[0].permute(1, 0, 2) # torch.Size([7, bs, 256])
        physics_emb = encoder_hidden_states[1].squeeze(1)       # torch.Size([bs, 256])
        trans_cond = encoder_hidden_states[2]                   # torch.Size([bs, motion_seq_len, 3])

        # [复刻] 3. Content 处理 (Style Remover logic)
        # 这部分逻辑非常绕，但必须保留，否则 Content 就废了
        content_emb_latent = content_emb
        content_emb_latent = self.IN(content_emb_latent.permute(1,2,0)).permute(2,0,1)
        content_emb_latent = content_emb_latent + time_emb # [S, B, D] + [1, B, D] -> Broadcast OK
        content_emb_latent = self.pe_content(content_emb_latent)
        content_emb_latent = self.seqTransEncoder(content_emb_latent).permute(1,0,2)
        content_emb_latent = self.linear(content_emb_latent.reshape(content_emb_latent.shape[0],-1)).reshape(content_emb_latent.shape[0], 6 ,256)
        content_emb_latent = content_emb_latent.permute(1,0,2) # torch.Size([6, bs, 256])

        # [复刻] 4. 拼接 Content + Sample
        # [S_content, B, D] + [S_sample, B, D] -> [S_tot, B, D]
        xseq = torch.cat((content_emb_latent, sample), axis=0) # torch.Size([13, bs, 256])

        # [修改] 5. 准备 Block 需要的条件
        
        # A. Trajectory (使用恢复的 TransEncoder)
        # 输出 [B, D] -> squeeze 后 [B, D] (原版逻辑)
        trans_emb = self.trans_Encoder(trans_cond, lengths)
        trans_emb = trans_emb + time_emb.squeeze(0) # [B, D] + [B, D]
        # trans_emb = trans_emb.squeeze() # torch.Size([bs, 256])
        trans_emb = trans_emb.squeeze(0) # 只压缩序列维，保留Batch维

        # B. Style 替代品
        # 原版是 style_emb + time_emb。我们没有 style_emb。
        # 策略：直接把 time_emb 作为 style 输入。
        # 这样 AdaLN 接收到的就是单纯的时间信号，这在 Diffusion 中是合法的。
        style_emb_latent = time_emb.squeeze(0) # torch.Size([bs, 256])

        # [复刻] 6. 进 Block 前的最后准备
        # 添加 PE 并 Permute 回 [B, S, D] (因为 DiTBlock 内部期望 Batch First)
        xseq = self.query_pos(xseq).permute(1,0,2)  # torch.Size([bs, 13, 256])
        
        # [核心修改] 7. 循环 DiT Blocks (注入 Physics)
        for block in self.blocks:
            # 传入: Feature, Style(Time), Trajectory, Physics
            xseq = block(xseq, style_emb_latent, trans_emb, physics_emb)
            
        # [复刻] 8. 切片与返回
        # xseq 目前是 [B, S_tot, D]。
        # 我们要切掉前面的 Content 部分。content_emb_latent.shape[0] 是 Batch (因为在上面permute过)
        # 等等，让我们看第 3 步最后：content_emb_latent.permute(1,0,2) -> [S, B, D]
        # 所以 content_emb_latent.shape[0] 是 Sequence Length。
        # 这里必须用 shape[0] 切片，对应的是 Sequence 维度。
        # 注意：xseq 是 [B, S, D]，切片 xseq[:, S_cont:, :] 是对的。 torch.Size([bs, 13, 256])
        sample = xseq[:, content_emb_latent.shape[0]:, :]  # torch.Size([bs, 7, 256])
        
        # testtest0109
        
        # 原版没有再 permute 回去吗？
        # 检查原版 return (sample, )。
        # 原版 forward 第一行 permute(1,0,2) 变成了 [S, B, D]。
        # 这里的 sample 是 [B, S, D]。
        # 通常 model 的输出应该和输入形状一致。
        # 如果输入是 [B, S, D] (Dataset loader 出来通常是这个)，那么这里返回 [B, S, D] 是对的。
        # **但是在原版代码里**：
        #   输入 `sample` (Batch First) -> permute -> Seq First
        #   Output `sample` (Batch First because of the slice)
        #   所以原版代码输入输出形状是**不一致**的吗？或者外部调用者处理了？
        #   让我们看 `mld.py` 的 `_diffusion_process`。通常计算 Loss 时需要 shape 匹配。
        #   既然你给的原版代码最后没有 permute，那我们也别加，保持原样。
        #   (如果有报错，我们在 mld.py 里修)

        return (sample, ) # torch.Size([bs, 7, 256])




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
