import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
import torch.optim as optim 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class StyleClassifierTransformer(pl.LightningModule):
    def __init__(self,
                 input_feats: int,         # **现在是VAE潜变量的特征维度, e.g., 256**
                 num_input_tokens: int,    # **现在是VAE潜变量的序列长度, e.g., 7**
                 num_styles: int = 100,
                 d_model: int = 256,
                 nhead: int = 4,
                 num_encoder_layers: int = 2,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 learning_rate: float = 1e-4
                ):
        super().__init__()
        self.save_hyperparameters()

        self.input_feats = input_feats
        self.num_input_tokens = num_input_tokens # VAE潜变量的序列长度

        # 输入投影: 从 input_feats (e.g., 256) 到 d_model
        self.input_projection = nn.Linear(self.input_feats, d_model)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional Encoding: 长度是 num_input_tokens + 1 (因为加了CLS token)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=self.num_input_tokens + 1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.fc_out = nn.Linear(d_model, num_styles)

        self.criterion = nn.CrossEntropyLoss()
        # self.lr will be in self.hparams.learning_rate

    def forward(self, src_latent_seq, src_key_padding_mask=None):
        # src_latent_seq: (Batch, num_input_tokens, input_feats), e.g., (Batch, 7, 256)
        # src_key_padding_mask: (Batch, num_input_tokens), True for padded VAE tokens.
        #                     **对于我们的固定7个有效VAE token的情况，这个应该是全False或None**

        batch_size = src_latent_seq.size(0)

        projected_src = self.input_projection(src_latent_seq)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        augmented_src = torch.cat((cls_tokens, projected_src), dim=1)
        pos_encoded_src = self.pos_encoder(augmented_src)
        
        augmented_padding_mask = None
        if src_key_padding_mask is not None:
            cls_padding_false = torch.zeros(batch_size, 1, dtype=torch.bool, device=src_latent_seq.device)
            augmented_padding_mask = torch.cat((cls_padding_false, src_key_padding_mask), dim=1)
        else: # 如果原始潜变量序列没有padding，则增广序列的mask只考虑CLS（非padding）
              # 但TransformerEncoder在mask为None时表现不同，最好提供一个全False的mask
            augmented_padding_mask = torch.zeros(batch_size, self.num_input_tokens + 1, dtype=torch.bool, device=src_latent_seq.device)


        transformer_output = self.transformer_encoder(pos_encoded_src, src_key_padding_mask=augmented_padding_mask)
        cls_embedding = transformer_output[:, 0, :]
        logits = self.fc_out(cls_embedding)
        return logits
    
    def _get_datamodule_components(self):
        """Helper to safely access datamodule and its components."""
        if not self.trainer or not hasattr(self.trainer, 'datamodule') or self.trainer.datamodule is None:
            raise RuntimeError("DataModule is not available via self.trainer.datamodule. Ensure it's correctly set up.")
        
        datamodule = self.trainer.datamodule
        if not hasattr(datamodule, 'vae_model') or not callable(getattr(datamodule, 'normalize_motion_for_vae', None)):
            raise AttributeError("The DataModule (expected StyleClassifierLatentDataModule) "
                                 "is missing 'vae_model' attribute or 'normalize_motion_for_vae' method.")
        return datamodule.vae_model, datamodule.normalize_motion_for_vae, datamodule

    def _common_step_latent(self, batch, batch_idx, stage: str): # Renamed to avoid confusion
        original_motion = batch["motion"]    # 原始的、可能未归一化的动作数据
        style_labels = batch["style_labels"]
        lengths = batch["length"]
        current_batch_size = original_motion.size(0)

        # 从 DataModule 获取 VAE 和归一化方法
        vae_model, normalize_fn, datamodule_ref = self._get_datamodule_components()
        
        # **在这里调用 normalize_motion_for_vae**
        # 原始动作数据需要先移动到和VAE模型相同的设备（通常由Lightning自动处理batch数据）
        # normalize_fn 内部会处理 mean/std 的设备对齐
        normalized_motion_for_vae = normalize_fn(original_motion) # 调用DataModule的归一化方法

        # 使用冻结的VAE进行编码
        with torch.no_grad():
            # vae_model 应该已经在 DataModule.setup() 中被移动到了正确的设备
            z_raw_vae_output, _ = vae_model.encode(normalized_motion_for_vae, lengths) 
            # z_raw_vae_output: (7, Batch, 256)

        # Permute z_raw for Cstyle input: (Batch, 7, 256)
        latent_input_for_cstyle = z_raw_vae_output.permute(1, 0, 2)
        
        # VAE输出的7个token都是有效的，所以padding_mask应该是全False
        # self.hparams.num_input_tokens 应该是 7
        cstyle_input_padding_mask = torch.zeros(current_batch_size, self.hparams.num_input_tokens, 
                                                dtype=torch.bool, device=latent_input_for_cstyle.device)

        # 调用自身的forward方法，输入是潜变量
        logits = self(latent_input_for_cstyle, src_key_padding_mask=cstyle_input_padding_mask)
        loss = self.criterion(logits, style_labels)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == style_labels).float().mean()

        self.log(f'{stage}_loss', loss, on_step=(stage=="train"), on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=current_batch_size)
        self.log(f'{stage}_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=current_batch_size)
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step_latent(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step_latent(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step_latent(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # Optional: Add a learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_acc"}} 