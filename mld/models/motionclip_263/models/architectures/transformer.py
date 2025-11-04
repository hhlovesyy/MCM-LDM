import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mld.models.motionclip_263.models.cross_attention import (
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


# only for ablation / not used in the final model
class TimeEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(TimeEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, lengths):
        time = mask * 1/(lengths[..., None]-1)
        time = time[:, None] * torch.arange(time.shape[1], device=x.device)[None, :]
        time = time[:, 0].T
        # add the time encoding
        x = x + time[..., None]
        return self.dropout(x)
    

class Encoder_TRANSFORMER(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames, num_classes, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=9, num_heads=4, dropout=0.1,normalize_before: bool = False,
                 ablation=None, activation="gelu", **kargs):
        super().__init__()
        
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes
        
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation
        
        self.latent_dim = latent_dim
        
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        
        self.input_feats = self.njoints*self.nfeats

        self.muQuery = nn.Parameter(torch.randn(1, self.latent_dim))
        self.sigmaQuery = nn.Parameter(torch.randn(1, self.latent_dim))
        self.skelEmbedding = nn.Linear(self.input_feats, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        # seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
        #                                                   nhead=self.num_heads,
        #                                                   dim_feedforward=self.ff_size,
        #                                                   dropout=self.dropout,
        #                                                   activation=self.activation)
        # self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
        #                                              num_layers=self.num_layers)
        


        # add vae transformer
        encoder_layer = TransformerEncoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        encoder_norm = nn.LayerNorm(self.latent_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers,
                                              encoder_norm)

    def forward(self, batch, return_all_layers: bool = False):  # 这个类的作用是将一个可变长度的动作序列（nframes）提炼、压缩成一个固定维度的统计表示，即高斯分布的均值 mu 和对数方差 logvar。
        x, y, mask = batch["x"], batch["y"], batch["mask"] # x是torch.Size([128, 263, 1, 196])，y是torch.Size([128])，mask是torch.Size([128, 196])(mask的话，本来动作序列的部分是True，而填充的部分是False)
        bs, njoints, nfeats, nframes = x.shape  # 这里面吃进来的njoints是263，nfeats是1，太怪了，不过他俩乘一起确实也是263，后面真炸了再回来改这个逻辑
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats) # torch.Size([196,128,263])

        # embedding of the skeleton
        x = self.skelEmbedding(x)  # torch.Size([196,128,512])，这个skelEmbedding是一个线性层，把263维度映射到了latent_dim（512）维度

        # Blank Y to 0's , no classes in our model, only learned token
        y = y - y
        xseq = torch.cat((self.muQuery[y][None], self.sigmaQuery[y][None], x), axis=0)  # torch.Size([198, 128, 512])，具体的技巧Gemini有进行总结，拼接两个可学习的向量，类似于BERT的[CLS]

        # add positional encoding
        xseq = self.sequence_pos_encoder(xseq)  # torch.Size([198, 128, 512])

        # create a bigger mask, to allow attend to mu and sigma
        muandsigmaMask = torch.ones((bs, 2), dtype=bool, device=x.device)  # torch.Size([128, 2])

        maskseq = torch.cat((muandsigmaMask, mask), axis=1)  # torch.Size([128, 198])

        encoder_output = self.encoder(xseq, src_key_padding_mask=~maskseq, return_all_layers=return_all_layers)  # torch.Size([198, 128, 512])
        
        all_features = None
        if return_all_layers:
            final, all_features = encoder_output
        else:
            final = encoder_output

        mu = final[0]  # torch.Size([128, 512])
        logvar = final[1]

        return_dict = {"mu": mu, "logvar": logvar}
        if return_all_layers and all_features is not None:
            return_dict["all_features"] = all_features  # 每一层都是[198，128，512]

        return return_dict


class Decoder_TRANSFORMER(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames, num_classes, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=9, num_heads=4, dropout=0.1, activation="gelu",normalize_before: bool = False,
                 ablation=None, **kargs):
        super().__init__()

        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes
        
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation
        
        self.latent_dim = latent_dim
        
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation

        self.activation = activation
                
        self.input_feats = self.njoints*self.nfeats

        # only for ablation / not used in the final model
        if self.ablation == "zandtime":
            self.ztimelinear = nn.Linear(self.latent_dim + self.num_classes, self.latent_dim)

        self.actionBiases = nn.Parameter(torch.randn(1, self.latent_dim))

        # only for ablation / not used in the final model
        if self.ablation == "time_encoding":
            self.sequence_pos_encoder = TimeEncoding(self.dropout)
        else:
            self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        # seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
        #                                                   nhead=self.num_heads,
        #                                                   dim_feedforward=self.ff_size,
        #                                                   dropout=self.dropout,
        #                                                   activation=activation)
        # self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
        #                                              num_layers=self.num_layers)
        


        # use mld_vae decoder
        decoder_layer = TransformerDecoderLayer(
            self.latent_dim,
            num_heads,
            ff_size,
            dropout,
            activation,
            normalize_before,
        )
        decoder_norm = nn.LayerNorm(self.latent_dim)
        self.decoder = SkipTransformerDecoder(decoder_layer, num_layers,
                                                decoder_norm)
        
        self.finallayer = nn.Linear(self.latent_dim, self.input_feats)
        
    def forward(self, batch, use_text_emb=False):
        z, y, mask, lengths = batch["z"], batch["y"], batch["mask"], batch["lengths"]
        if use_text_emb:
            z = batch["clip_text_emb"]
        latent_dim = z.shape[1]
        bs, nframes = mask.shape
        njoints, nfeats = self.njoints, self.nfeats

        # only for ablation / not used in the final model
        if self.ablation == "zandtime":
            yoh = F.one_hot(y, self.num_classes)
            z = torch.cat((z, yoh), axis=1)
            z = self.ztimelinear(z)
            z = z[None]  # sequence of size 1
        else:
            # only for ablation / not used in the final model
            if self.ablation == "concat_bias":
                # sequence of size 2
                z = torch.stack((z, self.actionBiases[y]), axis=0)
            else:
                z = z[None]  # sequence of size 1  #

        timequeries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        
        # only for ablation / not used in the final model
        if self.ablation == "time_encoding":
            timequeries = self.sequence_pos_encoder(timequeries, mask, lengths)
        else:
            timequeries = self.sequence_pos_encoder(timequeries)
        
        output = self.decoder(tgt=timequeries, memory=z,
                                      tgt_key_padding_mask=~mask)
        
        output = self.finallayer(output).reshape(nframes, bs, njoints, nfeats)
        
        # zero for padded area
        output[~mask.T] = 0
        output = output.permute(1, 2, 3, 0)

        if use_text_emb:
            batch["txt_output"] = output
        else:
            batch["output"] = output
        return batch
