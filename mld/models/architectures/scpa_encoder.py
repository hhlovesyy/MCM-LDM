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

    def forward(self, x, memory, need_weights=False):
        # x: Query (Scene Context)
        # memory: Key/Value (Physics Params)
        
        # Part 1: Self Attention (with Residual)
        x2 = self.norm1(x)
        x2, _ = self.self_attn(x2, x2, x2)
        x = x + self.dropout(x2)
        
        # Part 2: Cross Attention (with Residual) & Weight Extraction
        x2 = self.norm2(x)
        x2, attn_weights = self.cross_attn(query=x2, key=memory, value=memory, need_weights=need_weights)
        x = x + self.dropout(x2)
        
        # Part 3: FFN (with Residual)
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