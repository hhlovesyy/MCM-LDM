import torch
import torch.nn as nn
import torch.nn.functional as F

# 保持你的 Mock 逻辑，方便单元测试
try:
    from mld.config import instantiate_from_config, OmegaConf
except ImportError:
    # print("Warning: Could not import from mld.config...")
    def instantiate_from_config(config):
        target_class = eval(config["target"].split('.')[-1])
        return target_class(**config.get("params", {}))
    from omegaconf import OmegaConf

class SCPAEncoderBlock(nn.Module):
    """
    SCPAEncoder 的基础构建块。
    显式实现了 Transformer Decoder Layer 的逻辑，以便能够直接访问 Attention Map。
    """
    def __init__(self, d_model, n_head, ff_dim, dropout):
        super().__init__()
        # 1. Self-Attention (对于长度为1的Query，这其实只是一个Linear变换，但为了结构完整保留)
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        
        # 2. Cross-Attention (核心: Scene Query -> Physics Key/Value)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        
        # 3. Feed Forward Network
        self.norm3 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.activation = nn.GELU()

    # def forward(self, x, memory, need_weights=False):
    #     # x: Query (Scene Context)
    #     # memory: Key/Value (Physics Params)
        
    #     # Part 1: Self Attention (with Residual)
    #     x2 = self.norm1(x)
    #     x2, _ = self.self_attn(x2, x2, x2)
    #     x = x + self.dropout(x2)
        
    #     # Part 2: Cross Attention (with Residual) & Weight Extraction
    #     x2 = self.norm2(x)
    #     x2, attn_weights = self.cross_attn(query=x2, key=memory, value=memory, need_weights=need_weights)
    #     x = x + self.dropout(x2)
        
    #     # Part 3: FFN (with Residual)
    #     x2 = self.norm3(x)
    #     x2 = self.linear2(self.dropout(self.activation(self.linear1(x2))))
    #     x = x + self.dropout(x2)
        
    #     return x, attn_weights
    # 修改 forward，增加 key_padding_mask 参数
    def forward(self, x, memory, need_weights=False, key_padding_mask=None):
        # Part 1: Self Attention (不变)
        x2 = self.norm1(x)
        x2, _ = self.self_attn(x2, x2, x2)
        x = x + self.dropout(x2)
        
        # Part 2: Cross Attention (核心修改)
        x2 = self.norm2(x)
        # 这里的 key_padding_mask 是 PyTorch MultiheadAttention 的标准参数
        # 它会自动屏蔽掉 mask 为 True 的位置
        x2, attn_weights = self.cross_attn(
            query=x2, 
            key=memory, 
            value=memory, 
            key_padding_mask=key_padding_mask, # <--- 加上这行
            need_weights=need_weights
        )
        x = x + self.dropout(x2)
        
        # Part 3: FFN (不变)
        x2 = self.norm3(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x2))))
        x = x + self.dropout(x2)
        
        return x, attn_weights

class SCPAEncoder(nn.Module):
    """
    场景语境化参数注意力编码器 (SCPAEncoder) - 稳健工程版
    """
    def __init__(self, scene_cat_dim: int, phys_params_dim: int,
                 d_model: int = 256, n_head: int = 4, n_layers: int = 1,
                 ff_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.phys_params_dim = phys_params_dim
        
        # --- 1. Embedding ---
        self.scene_embedding = nn.Linear(scene_cat_dim, d_model)
        self.phys_value_embedding_layer = nn.Linear(1, d_model)
        self.phys_type_embedding = nn.Embedding(phys_params_dim, d_model)
        
        # Learnable Query Token: 类似于 BERT 的 [CLS] 或 DETR 的 Object Query
        self.scene_query_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # --- 2. Stacking Blocks ---
        # 使用 ModuleList 显式堆叠，替代 nn.TransformerDecoder
        self.layers = nn.ModuleList([
            SCPAEncoderBlock(d_model, n_head, ff_dim, dropout) 
            for _ in range(n_layers)
        ])
        
        # --- 3. Output Norm ---
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, scene_cat: torch.Tensor, phys_params: torch.Tensor, need_weights: bool = False):
        """
        Returns:
            final_emb: (Batch, 1, d_model)
            attn_weights: List of (Batch, 1, phys_params_dim), one per layer
        """
        batch_size = phys_params.shape[0]

        # A. 构建 Memory (Physics)
        # phys_params: (B, 4) -> (B, 4, 1)
        phys_params_seq = phys_params.unsqueeze(-1)
        # Value Embedding
        phys_value_emb = self.phys_value_embedding_layer(phys_params_seq) # (B, 4, D)
        # Type Embedding
        param_indices = torch.arange(self.phys_params_dim, device=phys_params.device)
        phys_type_emb = self.phys_type_embedding(param_indices).unsqueeze(0).expand(batch_size, -1, -1)
        # Final Memory
        memory = phys_value_emb + phys_type_emb # (B, 4, D)

        # B. 构建 Initial Query (Scene)
        # scene_cat: (B, 5) -> (B, D)
        scene_emb = self.scene_embedding(scene_cat)
        # Combine with learnable token: (1, 1, D) + (B, 1, D) -> (B, 1, D)
        query = self.scene_query_token + scene_emb.unsqueeze(1)

        # C. Passing through layers
        all_attn_weights = []
        x = query
        for layer in self.layers:
            x, weights = layer(x, memory, need_weights=need_weights)
            if need_weights:
                all_attn_weights.append(weights)

        # D. Final Output
        final_emb = self.output_norm(x)
        
        # 如果只有一层，直接返回 Tensor 而不是 List，方便调用
        if need_weights:
            if len(all_attn_weights) == 1:
                return final_emb, all_attn_weights[0]
            return final_emb, all_attn_weights
        
        return final_emb
    

# class SCPAEncoder1125(nn.Module):
#     """
#     [最终版] 纯物理参数注意力编码器 (Scene-Agnostic)
#     不再接收场景标签，而是用一个可学习的全局Query去理解物理参数序列。
#     """
#     def __init__(self, phys_params_dim: int, d_model: int = 256, 
#                  n_head: int = 4, n_layers: int = 1, **kwargs): # 吸收多余的 scene_cat_dim
#         super().__init__()
        
#         self.d_model = d_model
#         self.phys_params_dim = phys_params_dim
        
#         # --- Embeddings ---
#         self.phys_value_embedding = nn.Linear(1, d_model)
#         self.phys_type_embedding = nn.Embedding(phys_params_dim, d_model)
        
#         # [核心修改] 可学习的全局令牌
#         self.physics_query_token = nn.Parameter(torch.randn(1, 1, d_model))
        
#         # --- Transformer Blocks ---
#         self.layers = nn.ModuleList([
#             SCPAEncoderBlock(d_model, n_head, 1024, 0.1) 
#             for _ in range(n_layers)
#         ])
        
#         self.output_norm = nn.LayerNorm(d_model)

#     def forward(self, phys_params: torch.Tensor, scene_cat: torch.Tensor = None, need_weights: bool = False):
#         # [核心修改] scene_cat 变为可选参数，但我们不再使用它
#         # print("in forward , phys params:", phys_params.shape)
#         batch_size = phys_params.shape[0]  # torch.Size([16, 4])

#         # A. 构建 Memory (同之前)
#         phys_params_seq = phys_params.unsqueeze(-1)  # torch.Size([16, 4, 1])
#         phys_value_emb = self.phys_value_embedding(phys_params_seq) # torch.Size([16, 4, 256])
#         param_indices = torch.arange(self.phys_params_dim, device=phys_params.device) # tensor([0, 1, 2, 3], device='cuda:0')
#         phys_type_emb = self.phys_type_embedding(param_indices).unsqueeze(0).expand(batch_size, -1, -1) # torch.Size([16, 4, 256])
#         memory = phys_value_emb + phys_type_emb

#         # B. 构建 Query (新逻辑)
#         query = self.physics_query_token.expand(batch_size, -1, -1) # torch.Size([16, 1, 256])

#         # C. 通过 Transformer
#         all_attn_weights = []
#         x = query
#         for layer in self.layers:
#             x, weights = layer(x, memory, need_weights=need_weights)  # x:torch.Size([16, 1, 256]), weights:0
#             if need_weights:
#                 all_attn_weights.append(weights)

#         # D. 输出
#         final_emb = self.output_norm(x)
        
#         weights_tuple = all_attn_weights[0] if (need_weights and all_attn_weights) else None
#         if need_weights:
#             return final_emb, weights_tuple
#         return final_emb

class SCPAEncoder1125(nn.Module):
    """
    [论文级升级版] Input-Guided Attention Encoder
    原理：使用输入物理参数动态生成 Query，并配合 Padding Mask 强制模型关注有效信号。
    """
    def __init__(self, phys_params_dim: int, d_model: int = 256, 
                 n_head: int = 4, n_layers: int = 1, **kwargs): 
        super().__init__()
        
        self.d_model = d_model
        self.phys_params_dim = phys_params_dim
        
        # --- 1. Embedding (保持不变) ---
        # 用于把具体的物理数值映射成向量 (Value)
        self.phys_value_embedding = nn.Linear(1, d_model)
        # 用于标识这是第几个参数 (Key 的一部分)
        self.phys_type_embedding = nn.Embedding(phys_params_dim, d_model)
        
        # --- 2. [核心修改点A] 动态 Query 生成器 ---
        # 以前是：self.physics_query_token = nn.Parameter(...) 
        # 现在改成：一个简单的线性层，把输入的 6维 物理参数 直接映射成 Query
        # 这样 Query 就自带了 "当前哪个参数有值" 的信息！
        self.query_generator = nn.Linear(phys_params_dim, d_model)
        
        # --- 3. Transformer Blocks (保持不变) ---
        self.layers = nn.ModuleList([
            SCPAEncoderBlock(d_model, n_head, 1024, 0.1)
            for _ in range(n_layers)
        ])
        
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, phys_params: torch.Tensor, scene_cat: torch.Tensor = None, need_weights: bool = False):
        # phys_params shape: [Batch_Size, 6] (例如: [16, 6])
        batch_size = phys_params.shape[0]
        device = phys_params.device

        # ====================================================
        # Part A: 构建 Memory (Key & Value) - 这里的书架逻辑不变
        # ====================================================
        # (B, 6) -> (B, 6, 1)
        phys_params_seq = phys_params.unsqueeze(-1) 
        
        # Value: 数值映射
        phys_value_emb = self.phys_value_embedding(phys_params_seq) # (B, 6, D)
        
        # Type: 位置编码 (告诉模型这是 WindX 还是 GapWidth)
        param_indices = torch.arange(self.phys_params_dim, device=device)
        phys_type_emb = self.phys_type_embedding(param_indices).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Memory = Value + Type (书的内容 + 书的标签)
        memory = phys_value_emb + phys_type_emb # (B, 6, D)

        # ====================================================
        # Part B: [核心修改点B] 构建 Dynamic Query (动态查询)
        # ====================================================
        # 以前 Query 是固定的，现在由 phys_params 全局信息生成
        # 含义：如果输入里有 Ceiling，Query 向量就会长得像 Ceiling，去吸附 Ceiling 的 Key
        query = self.query_generator(phys_params) # (B, D)
        query = query.unsqueeze(1) # 变成序列形式 (B, 1, D)

        # ====================================================
        # Part C: [核心修改点C] 构建 Key Padding Mask (防呆机制)
        # ====================================================
        # 原理：有些参数是 0 (比如没风的时候风是0)，我们要告诉 Attention "别看这些 0"
        # 逻辑：True 的位置会被忽略，False 的位置会被保留
        # 我们认为绝对值 < 0.0001 的就是无效输入 (Padding)
        # key_padding_mask shape: (B, 6)
        key_padding_mask = torch.abs(phys_params) < 1e-4

        # 【极其重要】防止全 0 崩溃
        # 如果某一行全是 0 (比如无风无缝隙无天花板)，Attention 会报错或输出 NaN。
        # 这种情况下，我们临时允许它看第一个位置，反正全是 0 也没影响。
        all_zero_mask = key_padding_mask.all(dim=1) # (B,)
        if all_zero_mask.any():
            # 将全0样本的第一个位置设为 False (允许关注)，防止 NaN
            key_padding_mask[all_zero_mask, 0] = False

        # ====================================================
        # Part D: 传入 Transformer
        # ====================================================
        all_attn_weights = []
        x = query
        
        for layer in self.layers:
            # 注意：这里我们传入了 key_padding_mask
            # 你需要确保你的 SCPAEncoderBlock 的 forward 函数能接收这个参数
            # 如果之前的 Block 写死了没接收，我在下面会补上 Block 的代码
            x, weights = layer(x, memory, need_weights=need_weights, key_padding_mask=key_padding_mask)
            
            if need_weights:
                all_attn_weights.append(weights)

        final_emb = self.output_norm(x)
        
        weights_tuple = all_attn_weights[0] if (need_weights and all_attn_weights) else None
        if need_weights:
            return final_emb, weights_tuple
        return final_emb
class SCPAEncoderSimple(nn.Module):
    """
    [简化版] 物理参数编码器
    不再使用 Attention 机制，而是将物理参数视为一个整体向量进行映射。
    这能强迫模型同时考虑 X, Y 和 Strength，避免出现“只关注某个轴”的情况。
    """
    def __init__(self, scene_cat_dim: int, phys_params_dim: int,
                 d_model: int = 256, **kwargs): # kwargs 吸收多余参数
        super().__init__()
        
        self.d_model = d_model
        
        # 1. 场景映射 (虽然我们现在只有1个场景，但保留结构)
        self.scene_embedding = nn.Linear(scene_cat_dim, d_model)
        
        # 2. 物理参数映射 (核心修改)
        # 输入: [Batch, 3] -> 输出: [Batch, d_model]
        self.phys_mlp = nn.Sequential(
            nn.Linear(phys_params_dim, 512),
            nn.SiLU(),
            nn.Linear(512, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        # 3. 融合层
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, scene_cat: torch.Tensor, phys_params: torch.Tensor, need_weights: bool = False):
        """
        返回:
        - final_emb: [Batch, 1, d_model] (为了兼容接口，保持 3 维)
        - weights: None (MLP 没有 Attention 权重)
        """
        # 1. 处理场景
        scene_emb = self.scene_embedding(scene_cat) # [B, D]
        
        # 2. 处理物理 (整体映射)
        phys_emb = self.phys_mlp(phys_params) # [B, D]
        
        # 3. 简单融合 (Concat + Linear)
        # 让模型自己去学习 Scene 和 Physics 的关系
        combined = torch.cat([scene_emb, phys_emb], dim=1) # [B, 2D]
        x = self.fusion(combined) # [B, D]
        
        x = self.output_norm(x)
        
        if need_weights:
            # 调整维度以适配 Denoiser 接口: [B, 1, D]
            return x.unsqueeze(1), None
        return x.unsqueeze(1)

# --- 单元测试 Main 函数 ---
if __name__ == '__main__':
    print("--- [单元测试] 启动 SCPAEncoder (已修复错误版) 测试 ---")
    
    # ... (1. 2. 3. 步骤的单元测试代码保持不变) ...
    # 1. 加载YAML配置文件。我们直接在这里定义一个字典，避免文件路径问题。
    print("\n1. 定义并加载模块配置...")
    config_dict = {
        'scpa_encoder': {
            'target': '__main__.SCPAEncoder', # 引用当前文件中的类
            'params': {
                'scene_cat_dim': 5, 'phys_params_dim': 4, 'd_model': 256,
                'n_head': 4, 'n_layers': 1, 'ff_dim': 1024, 'dropout': 0.1
            }
        }
    }
    cfg = OmegaConf.create(config_dict)
    encoder_cfg = cfg.scpa_encoder
    print("配置加载成功。")

    # 2. 从配置实例化 SCPAEncoder 模型。
    print("\n2. 实例化 SCPAEncoder 模型...")
    try:
        encoder = instantiate_from_config(encoder_cfg)
        print("模型实例化成功:")
        # print(encoder)
        encoder.eval() 
    except Exception as e:
        print(f"\n[错误] 模型实例化失败: {e}")
        exit()

    # 3. 创建模拟的输入数据。
    print("\n3. 创建模拟输入数据...")
    batch_size = 8
    scene_dim = encoder_cfg.params.scene_cat_dim
    phys_dim = encoder_cfg.params.phys_params_dim
    scene_indices = torch.randint(0, scene_dim, (batch_size,))
    dummy_scene_cat = F.one_hot(scene_indices, num_classes=scene_dim).float()
    dummy_phys_params = torch.rand(batch_size, phys_dim)
    print(f"   - 模拟场景类别 shape: {dummy_scene_cat.shape}")
    print(f"   - 模拟物理参数 shape: {dummy_phys_params.shape}")

    # 4. 执行前向传播并检查输出。
    print("\n4. 执行前向传播并检查输出...")
    try:
        # --- 测试不返回权重 ---
        with torch.no_grad():
            output_emb = encoder(dummy_scene_cat, dummy_phys_params)
        print("   - 前向传播 (不带权重) 成功。")
        print(f"   - 输出嵌入 shape: {output_emb.shape}")
        expected_shape = (batch_size, 1, encoder_cfg.params.d_model)
        assert output_emb.shape == expected_shape
        print("   - ✅ 输出形状正确。")

        # --- 测试返回权重 ---
        with torch.no_grad():
            output_emb_w, attn_w = encoder(dummy_scene_cat, dummy_phys_params, need_weights=True)
        print("   - 前向传播 (带权重) 成功。")
        # 检查主输出
        assert output_emb_w.shape == expected_shape
        # 检查权重
        print(f"   - 返回的注意力权重 shape: {attn_w.shape}")
        expected_attn_shape = (batch_size, 1, phys_dim)
        assert attn_w.shape == expected_attn_shape
        print("   - ✅ 注意力权重形状正确。")
        assert torch.allclose(attn_w.sum(dim=-1), torch.ones(batch_size, 1))
        print("   - ✅ 注意力权重已正确归一化 (Sum to 1)。")
        
    except Exception as e:
        print(f"\n[错误] 在前向传播或检查中发生错误: {e}")
        import traceback
        traceback.print_exc()
        exit()

    print("\n--- [单元测试] SCPAEncoder (已修复错误版) 已通过所有测试! ---")