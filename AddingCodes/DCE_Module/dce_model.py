# dce_model.py (或 style_classifier_model.py，如果想放一起)

import torch
import torch.nn as nn
import pytorch_lightning as pl # 如果DCE本身要作为LightningModule进行训练
import math

# PositionalEncoding (假设已定义或可导入)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000): # max_len 应足够大
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # (1, max_len, d_model)

    def forward(self, x):
        # x: (Batch, Seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DisentangledContentExtractor(nn.Module): # 或者 pl.LightningModule
    def __init__(self,
                 input_dim: int = 256,       # VAE潜变量每个token的维度 (Dim_VAE_Latent)
                 num_input_tokens: int = 7,  # VAE潜变量序列的长度 (Num_Tokens_VAE)
                 # --- DCE内部Transformer参数 ---
                 d_model: int = 256,         # Transformer模型的内部维度
                 nhead: int = 4,             # 多头注意力的头数
                 num_encoder_layers: int = 2, # Transformer编码器的层数
                 dim_feedforward: int = 512, # 前馈网络的维度
                 dropout: float = 0.1,
                 # --- 输出维度 (通常与input_dim一致) ---
                 output_dim: int = 256       # 输出特征fc中每个token的维度
                ):
        super().__init__()
        # 如果作为pl.LightningModule训练DCE，则取消下一行注释
        # self.save_hyperparameters() 

        self.input_dim = input_dim
        self.num_input_tokens = num_input_tokens
        self.d_model = d_model
        self.output_dim = output_dim

        # 1. 输入投影 (如果 input_dim != d_model)
        if self.input_dim != self.d_model:
            self.input_projection = nn.Linear(self.input_dim, self.d_model)
        else:
            self.input_projection = nn.Identity()

        # 2. Positional Encoding (只有当 num_input_tokens > 1 才真正需要)
        # 但为了代码通用性，即使是1，PositionalEncoding(d_model, max_len=1) 也能工作
        # 或者可以明确地用 nn.Identity() 替换
        if self.num_input_tokens > 1 : # 通常 > 0, 严格来说是 > 1 才需要位置信息
            self.pos_encoder = PositionalEncoding(self.d_model, dropout, max_len=self.num_input_tokens)
        else: # num_input_tokens为0或1的情况，理论上不应为0
             self.pos_encoder = nn.Identity()


        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 4. 输出投影 (如果 d_model != output_dim)
        # 通常我们希望 output_dim == input_dim 以便直接替换
        if self.d_model != self.output_dim:
            self.output_projection = nn.Linear(self.d_model, self.output_dim)
        else:
            self.output_projection = nn.Identity()

    def forward(self, z_raw, src_key_padding_mask=None):
        """
        Args:
            z_raw (Tensor): Raw latent variables from VAE encoder, after permute.
                            Shape: (Batch, num_input_tokens, input_dim)
                                   e.g., (Batch, 7, 256)
            src_key_padding_mask (Tensor, optional): Mask for z_raw if it has padding.
                                                    Shape: (Batch, num_input_tokens),
                                                    True for padded positions.
                                                    Defaults to None.
                                                    **根据我们的分析，这个可能为None，因为VAE输出固定7个有效token。**
        Returns:
            fc (Tensor): Disentangled content features.
                         Shape: (Batch, num_input_tokens, output_dim)
                                e.g., (Batch, 7, 256)
        """
        if not isinstance(z_raw, torch.Tensor):
            raise TypeError(f"Input z_raw must be a torch.Tensor, got {type(z_raw)}")
        if z_raw.ndim != 3:
            raise ValueError(f"Input z_raw must be 3-dimensional (Batch, SeqLen, FeatDim), got {z_raw.ndim}")
        if z_raw.shape[1] != self.num_input_tokens:
            raise ValueError(f"Input z_raw sequence length ({z_raw.shape[1]}) does not match "
                             f"model's num_input_tokens ({self.num_input_tokens})")
        if z_raw.shape[2] != self.input_dim:
            raise ValueError(f"Input z_raw feature dimension ({z_raw.shape[2]}) does not match "
                             f"model's input_dim ({self.input_dim})")


        # 1. Input Projection
        projected_z = self.input_projection(z_raw)  # (Batch, num_input_tokens, d_model)

        # 2. Positional Encoding
        pos_encoded_z = self.pos_encoder(projected_z) # (Batch, num_input_tokens, d_model)

        # 3. Transformer Encoder
        # 如果 src_key_padding_mask 为 None，TransformerEncoder内部不会使用它
        transformer_output = self.transformer_encoder(pos_encoded_z, src_key_padding_mask=src_key_padding_mask)
        # transformer_output: (Batch, num_input_tokens, d_model)

        # 4. Output Projection
        fc = self.output_projection(transformer_output) # (Batch, num_input_tokens, output_dim)

        return fc

# GradientReversalLayer (保持不变，这里不再重复)
# from previous_code import GradientReversalLayer
class GradReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output.neg() * ctx.lambda_val), None
class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_val=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_val = lambda_val
    def forward(self, x):
        return GradReverseFunction.apply(x, self.lambda_val)
    def update_lambda(self, new_lambda):
        self.lambda_val = new_lambda