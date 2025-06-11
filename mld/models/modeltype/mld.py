import inspect
import os
from mld.transforms.rotation2xyz import Rotation2xyz
import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torchmetrics import MetricCollection
import time
from mld.config import instantiate_from_config
from os.path import join as pjoin
from mld.models.architectures import (
    mld_denoiser,
    mld_vae,
    t2m_motionenc,
    t2m_textenc,
)
from mld.models.losses.mld import MLDLosses
from mld.models.modeltype.base import BaseModel
from mld.utils.temos_utils import remove_padding
from mld.utils.temos_utils import lengths_to_mask
# from animate import plot_3d_motion
# for motionclip
import clip
from ..motionclip_263.utils.get_model_and_data import get_model_and_data
from ..motionclip_263.parser.visualize import parser
from ..motionclip_263.visualize.visualize import viz_clip_text, get_gpu_device
from ..motionclip_263.utils.misc import load_model_wo_clip
import yaml
def read_yaml_to_dict(yaml_path: str, ):
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value
    
# for skeleton transform
from datasets.utils.common.skeleton import Skeleton
import numpy as np
import os
from datasets.utils.common.quaternion import *
from datasets.utils.paramUtil import *
# import evaluate.utils.rotation_conversions as geometry





from .base import BaseModel
import torch.nn as nn
import math

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
        if self.num_input_tokens > 0 : # 通常 > 0, 严格来说是 > 1 才需要位置信息
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


class MLD(BaseModel):
    """
    Stage 1 vae
    Stage 2 diffusion
    """

    def __init__(self, cfg, datamodule, **kwargs):
        super().__init__()

        self.cfg = cfg

        self.stage = cfg.TRAIN.STAGE
        self.is_vae = cfg.model.vae
        self.nfeats = cfg.DATASET.NFEATS
        self.njoints = cfg.DATASET.NJOINTS
        self.debug = cfg.DEBUG
        self.latent_dim = cfg.model.latent_dim
        self.guidance_scale = cfg.model.guidance_scale
        self.guidance_uncodp = cfg.model.guidance_uncondp
        self.datamodule = datamodule

        # --- Initialize scene_embedding layer (example, place according to your structure) ---
        self.num_scene_classes = 15 # 15个场景类别,目前写死了，后面会改成可配置的
        self.scene_embedding_dim = 512
        if self.num_scene_classes > 0:
            self.scene_embedding = torch.nn.Embedding(self.num_scene_classes, self.scene_embedding_dim)
            print(f"Initialized scene embedding layer with {self.num_scene_classes} classes and dim {self.scene_embedding_dim}.")
        else:
            self.scene_embedding = None
            print("Scene embedding layer not initialized.")



        # self.text_encoder = instantiate_from_config(cfg.model.text_encoder)

        parameters = read_yaml_to_dict("configs/motionclip_config/motionclip_params_263.yaml")
        parameters["device"] = 'cuda:{}'.format(cfg["DEVICE"][0])        
        self.motionclip = get_model_and_data(parameters, split='vald')
        print("load motion clip-xyz-263")
        print("Restore weights..")
        checkpointpath = "checkpoints/motionclip_checkpoint/motionclip.pth.tar"
        state_dict = torch.load(checkpointpath, map_location=parameters["device"])
        load_model_wo_clip(self.motionclip, state_dict)

        self.mean = torch.tensor(self.datamodule.hparams.mean).to(parameters["device"])
        self.std = torch.tensor(self.datamodule.hparams.std).to(parameters["device"])

        #don't train motionclip
        self.motionclip.training = False
        for p in self.motionclip.parameters():
            p.requires_grad = False

        # --- 开始：初始化DCE模型 ---
        # todo：目前也是写死了checkpoint的路径，这个也要改
        dce_checkpoint_path = "/root/autodl-tmp/MCM-LDM/experiments/dce_checkpoints/DCE_Training_Experiment_v1/checkpoints/last.ckpt" # 硬编码DCE权重路径
        
        dce_params = {
            "input_dim": 256,
            "num_input_tokens": 7,
            "d_model": 256, 
            "nhead": 4,   
            "num_encoder_layers": 3,
            "dim_feedforward": 1024,
            "dropout": 0.1,
            "output_dim": 256
        }
        self.dce_model = DisentangledContentExtractor(**dce_params) 
        
        print(f"尝试从以下路径加载DCE模型权重: {dce_checkpoint_path}")
        if os.path.exists(dce_checkpoint_path):
            checkpoint_dce = torch.load(dce_checkpoint_path, map_location='cpu')
            
            if 'state_dict' in checkpoint_dce:
                full_state_dict = checkpoint_dce['state_dict']
                dce_model_prefix_in_ckpt = "dce." 
                
                # 创建一个新的字典，只包含DCE模型的权重
                dce_specific_state_dict = {}
                for k, v in full_state_dict.items():
                    if k.startswith(dce_model_prefix_in_ckpt):
                        # 移除前缀，得到DCE模型内部的参数名
                        clean_key = k[len(dce_model_prefix_in_ckpt):]
                        dce_specific_state_dict[clean_key] = v
                
                if not dce_specific_state_dict:
                    print(f"警告：在checkpoint中没有找到以 '{dce_model_prefix_in_ckpt}' 开头的键。尝试直接加载整个 state_dict。")
                    # 这种情况可能意味着checkpoint直接就是DCE的state_dict，或者前缀不正确
                    # 为了安全，可以尝试加载 full_state_dict，但如果它包含非DCE键，仍会报错
                    # 或者，如果确定没有前缀，可以直接用 full_state_dict
                    # dce_specific_state_dict = full_state_dict # 如果确定ckpt就是DCE的，或者没有前缀

                if dce_specific_state_dict: # 确保我们提取到了一些权重
                    try:
                        missing_keys, unexpected_keys = self.dce_model.load_state_dict(dce_specific_state_dict, strict=False) # 使用strict=False来查看不匹配的键
                        if unexpected_keys:
                            print(f"警告: 加载DCE state_dict 时遇到未预期的键: {unexpected_keys}")
                        if missing_keys:
                            print(f"警告: 加载DCE state_dict 时缺失以下键: {missing_keys}")
                        
                        if not unexpected_keys and not missing_keys:
                             print("成功加载DCE state_dict (所有键完美匹配)。")
                        elif not unexpected_keys and missing_keys:
                             print("成功加载DCE state_dict (部分键匹配，但有缺失键，这可能没问题如果它们是buffer或可选的)。")
                        else: # 有unexpected_keys，通常是个问题
                             print("DCE state_dict 加载完成，但存在未预期或缺失的键。请检查。")

                    except RuntimeError as e:
                        print(f"加载DCE specific state_dict 失败: {e}。请检查模型定义和checkpoint。")
                else:
                    print("错误：未能从checkpoint中提取DCE特定的state_dict。DCE权重未加载。")


            # (原来的其他加载逻辑，如直接加载checkpoint如果它不是Lightning格式)
            elif isinstance(checkpoint_dce, dict) and not any(k_lightning in checkpoint_dce for k_lightning in ['epoch', 'global_step']): 
                # 假设 checkpoint_dce 直接就是 DCE 模型的 state_dict
                print("Checkpoint似乎直接是state_dict，尝试加载...")
                try:
                    missing_keys, unexpected_keys = self.dce_model.load_state_dict(checkpoint_dce, strict=False)
                    if unexpected_keys: print(f"警告 (直接加载): 未预期的键: {unexpected_keys}")
                    if missing_keys: print(f"警告 (直接加载): 缺失的键: {missing_keys}")
                    if not unexpected_keys and not missing_keys: print("成功直接加载DCE state_dict。")
                    else: print("DCE state_dict (直接加载) 完成，但有不匹配的键。")
                except RuntimeError as e:
                     print(f"直接加载DCE state_dict 失败: {e}。")
            else:
                print(f"DCE checkpoint {dce_checkpoint_path} 格式未知或不包含 'state_dict'。")
        else:
            print(f"未找到DCE checkpoint文件: {dce_checkpoint_path}。DCE模型将使用随机权重初始化。")

        self.dce_model.to(self.device)
        self.dce_model.eval()
        for param in self.dce_model.parameters():
            param.requires_grad = False
        print("DCE模型已移动到设备，设置为评估模式并已冻结。")
        # --- 结束：初始化DCE模型 ---



        # 2025.5.31 添加FiLM MLP
        self.film_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.scene_embedding_dim, self.scene_embedding_dim // 2), # 或者其他中间维度
            torch.nn.ReLU(),
            torch.nn.Linear(self.scene_embedding_dim // 2, self.scene_embedding_dim * 2) # 输出 gamma 和 beta
        )
        print(f"Initialized FiLM MLP for scene modulation. Input: {self.scene_embedding_dim}, Output: {self.scene_embedding_dim * 2}")


        self.vae = instantiate_from_config(cfg.model.motion_vae)
        # Don't train the motion encoder and decoder
        if self.stage == "diffusion": 
            self.vae.training = False
            for p in self.vae.parameters():
                p.requires_grad = False

        # pass diffusion_t to denoiser
        # --- START MODIFIED BLOCK: Pass diffusion_T to Denoiser ---
        denoiser_cfg = cfg.model.denoiser

        try:
            diffusion_T = self.cfg.model.scheduler.num_train_timesteps # Based on your scheduler init
        except AttributeError:
            print("Warning: Could not automatically determine diffusion_T from config. Using default 1000.")
            diffusion_T = 1000 # Fallback, ensure this is correct

        # Add diffusion_T to the denoiser_cfg or pass as kwargs
        # If denoiser_cfg is a dict from instantiate_from_config
        if not hasattr(denoiser_cfg, 'params'): # If it's a simple class path string in cfg
             denoiser_params = {}
        else: # If it's a config object with a params dict
             denoiser_params = denoiser_cfg.get('params', {}) # Get existing params or new dict

        denoiser_params['diffusion_T'] = diffusion_T
        denoiser_params['text_encoded_dim'] = 256


        if hasattr(denoiser_cfg, 'params'):
            denoiser_cfg.params = denoiser_params
            self.denoiser = instantiate_from_config(denoiser_cfg)
        else: # If denoiser_cfg was just a path, instantiate with new params
            self.denoiser = instantiate_from_config(denoiser_cfg, **denoiser_params)

        # --- END MODIFIED BLOCK ---

        self.scheduler = instantiate_from_config(cfg.model.scheduler)
        self.noise_scheduler = instantiate_from_config(
            cfg.model.noise_scheduler)


        self._get_t2m_evaluator(cfg)

        if cfg.TRAIN.OPTIM.TYPE.lower() == "adamw":
            self.optimizer = AdamW(lr=cfg.TRAIN.OPTIM.LR,
                                   params=self.parameters())
        else:
            raise NotImplementedError(
                "Do not support other optimizer for now.")
        
        # --- Setup training parameters (freeze/unfreeze) and configure the optimizer ---
        # This call will set self.optimizer
        self._setup_training_parameters_and_optimizer()
        # --- START: 日志文件设置 ---
        # 生成带时间戳的日志文件名
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_filename = f"train_{timestamp}_infos.log"
        # --- END: 日志文件设置 ---

        with open(self.log_filename, 'a') as f_log: # 'a' for append mode
            log_and_print = lambda message: (f_log.write(message + "\n"), print(message))

            log_and_print("\n" + "="*30 + " Model Parameter Status after __init__ " + "="*30)
            total_params = 0
            trainable_params = 0
            for name, param in self.named_parameters():
                total_params += param.numel()
                status = "TRAINABLE" if param.requires_grad else "FROZEN"
                log_message = f"{name:<60} | Size: {list(param.shape)} | Requires Grad: {param.requires_grad} ({status})"
                log_and_print(log_message)
                if param.requires_grad:
                    trainable_params += param.numel()
            log_and_print(f"\nTotal model parameters: {total_params}")
            log_and_print(f"Total trainable parameters: {trainable_params}")
            log_and_print(f"Total frozen parameters: {total_params - trainable_params}")
            log_and_print("="*80 + "\n")
        # --- END: 将参数状态打印到日志文件 ---

        if cfg.LOSS.TYPE == "mld":
            self._losses = MetricCollection({
                split: MLDLosses(vae=self.is_vae, mode="xyz", cfg=cfg)
                for split in ["losses_train", "losses_test", "losses_val"]
            })
        else:
            raise NotImplementedError(
                "MotionCross model only supports mld losses.")

        self.losses = {
            key: self._losses["losses_" + key]
            for key in ["train", "test", "val"]
        }

        self.metrics_dict = cfg.METRIC.TYPE
        self.configure_metrics()

        # If we want to overide it at testing time
        self.sample_mean = False
        self.fact = None
        self.do_classifier_free_guidance = True

        self.feats2joints = datamodule.feats2joints
        self.joints2feats = datamodule.joints2feats

    # 2025.5.30 新增代码：冻结
    def _setup_training_parameters_and_optimizer(self):
        """
        Sets requires_grad for parameters based on the finetuning strategy
        and configures the optimizer. This optimizer will be assigned to self.optimizer.
        """
        cfg = self.cfg  # Assumes self.cfg is available

        # --- 定义新的微调策略变量 ---
        FINETUNE_SCENE_FILM_AND_DENOISER = True # 新策略：微调场景、FiLM和Denoiser

        if FINETUNE_SCENE_FILM_AND_DENOISER:
            print("Strategy: Finetuning scene_embedding, scene_encoder_mlp, film_mlp, AND denoiser layers.")
            trainable_component_prefixes = ['denoiser.'] # Start with denoiser
            if hasattr(self, 'scene_embedding') and self.scene_embedding is not None:
                trainable_component_prefixes.append('scene_embedding.')
            if hasattr(self, 'scene_encoder_mlp') and self.scene_encoder_mlp is not None:
                trainable_component_prefixes.append('scene_encoder_mlp.')
            if hasattr(self, 'film_mlp') and self.film_mlp is not None:
                trainable_component_prefixes.append('film_mlp.')
            
            # Freeze all parameters first
            for param in self.parameters():
                param.requires_grad = False
            
            # Unfreeze target parameters
            unfrozen_count_final = 0
            for name, param in self.named_parameters():
                for prefix in trainable_component_prefixes:
                    if name.startswith(prefix):
                        param.requires_grad = True
                        unfrozen_count_final +=1
                        # print(f"  - Parameter '{name}' is UNFROZEN for training.")
                        break # Move to next parameter once unfrozen by a prefix
            
            total_params = sum(p.numel() for p in self.parameters())
            frozen_count_final = total_params - sum(p.numel() for p in self.parameters() if p.requires_grad)

            print(f"Final parameter status: Frozen: {frozen_count_final}, Unfrozen (trainable): {unfrozen_count_final}.")
        # The existing trainable_params_list logic should work if requires_grad is set correctly.
        trainable_params_list = [p for p in self.parameters() if p.requires_grad]

        if not trainable_params_list:
            print("WARNING: No trainable parameters found!")
            self.optimizer = None 
            return

        if cfg.TRAIN.OPTIM.TYPE.lower() == "adamw":
            self.optimizer = AdamW(lr=cfg.TRAIN.OPTIM.LR, params=trainable_params_list)
            print(f"AdamW optimizer configured with LR: {cfg.TRAIN.OPTIM.LR} for the trainable parameters.")
        else:
            raise NotImplementedError("...")

    def _get_t2m_evaluator(self, cfg):
        """
        load T2M text encoder and motion encoder for evaluating
        """
        # init module
        self.t2m_textencoder = t2m_textenc.TextEncoderBiGRUCo(
            word_size=cfg.model.t2m_textencoder.dim_word,
            pos_size=cfg.model.t2m_textencoder.dim_pos_ohot,
            hidden_size=cfg.model.t2m_textencoder.dim_text_hidden,
            output_size=cfg.model.t2m_textencoder.dim_coemb_hidden,
        )

        self.t2m_moveencoder = t2m_motionenc.MovementConvEncoder(
            input_size=cfg.DATASET.NFEATS - 4,
            hidden_size=cfg.model.t2m_motionencoder.dim_move_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_move_latent,
        )

        self.t2m_motionencoder = t2m_motionenc.MotionEncoderBiGRUCo(
            input_size=cfg.model.t2m_motionencoder.dim_move_latent,
            hidden_size=cfg.model.t2m_motionencoder.dim_motion_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_motion_latent,
        )
        # load pretrianed
        dataname = cfg.TEST.DATASETS[0]
        dataname = "t2m" if dataname == "humanml3d" or dataname== "humanml3d_scene" else dataname
        t2m_checkpoint = torch.load(
            os.path.join(cfg.model.t2m_path, dataname,
                         "text_mot_match/model/finest.tar"))
        self.t2m_textencoder.load_state_dict(t2m_checkpoint["text_encoder"])
        self.t2m_moveencoder.load_state_dict(
            t2m_checkpoint["movement_encoder"])
        self.t2m_motionencoder.load_state_dict(
            t2m_checkpoint["motion_encoder"])

        # freeze params
        self.t2m_textencoder.eval()
        self.t2m_moveencoder.eval()
        self.t2m_motionencoder.eval()
        for p in self.t2m_textencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_moveencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_motionencoder.parameters():
            p.requires_grad = False


# 


    def sample_from_distribution(
        self,
        dist,
        *,
        fact=None,
        sample_mean=False,
    ) -> Tensor:
        fact = fact if fact is not None else self.fact
        sample_mean = sample_mean if sample_mean is not None else self.sample_mean

        if sample_mean:
            return dist.loc.unsqueeze(0)

        # Reparameterization trick
        if fact is None:
            return dist.rsample().unsqueeze(0)

        # Resclale the eps
        eps = dist.rsample() - dist.loc
        z = dist.loc + fact * eps

        # add latent size
        z = z.unsqueeze(0)
        return z
    
# test
    # todo: forward 函数有可能被改坏了，需要在demo_transfer阶段进行验证
    def forward(self, batch):

        lengths = batch["length"] # 38
        # style
        motion = batch["style_motion"].clone() # torch.Size([1, 263, 263])；第二次debug进来是torch.Size([1, 345, 263])
        motion[...,:3] = 0


        # content
        content_motion = batch['content_motion'] # torch.Size([1, 38, 263])
        # 手动对content做一次归一化，因为inference的时候不是从dataloader读取数据的；content需要归一化 style不能归一化（motionclip吃归一化之前的数据）
        content_motion = (content_motion - self.mean.to(content_motion.device))/self.std.to(content_motion.device) # torch.Size([1, 38, 263])

        # trajectory
        trans_motion = content_motion.clone() # shape:torch.Size([1, 38, 263])
        # 
        content_motion[...,:3] = 0


        scale = batch["tag_scale"] # 2.5
        lengths1 = [content_motion.shape[1]]* content_motion.shape[0] # 38
        
        if self.cfg.TEST.COUNT_TIME:
            self.starttime = time.time()
            
        if self.stage in ['diffusion', 'vae_diffusion']:\
            #add style text in test
            
            
            # content motion
            with torch.no_grad():
                z, dist_m = self.vae.encode(content_motion.float(), lengths1)  # z:torch.Size([7, 1, 256])
                if self.cfg.model.useDCE:
                    z_raw_permuted_for_dce = z.permute(1, 0, 2)
                    fc_from_dce = self.dce_model(z_raw_permuted_for_dce)
                    z = fc_from_dce
                    # z: (7, Batch, 256) -> (Batch, 7, 256)
                    z = z.permute(1, 0, 2) # (Batch, 7, 256)
            uncond_tokens = torch.cat([z, z], dim = 1).permute(1,0,2)  # torch.Size([2, 7, 256])
            motion_emb_content = uncond_tokens # shape:torch.Size([2, 7, 256])；一样

            current_device = motion.device
            bsz_orig_style_scene = motion.shape[0]

            # style motion
            lengths11 = [motion.shape[1]]* motion.shape[0] # 263；same

            motion_seq_for_clip = motion.unsqueeze(-1).permute(0,2,3,1) # Use a distinct name if 'motion_seq' is used later
            
            y_for_motionclip = batch.get('y', torch.zeros(motion_seq_for_clip.shape[0], dtype=int, device=current_device))
            
            # 1. Get raw conditional style f_s_motion_cond
            # This f_s_motion_cond is based on bsz_orig_style_scene
            f_s_motion_cond = self.motionclip.encoder({'x': motion_seq_for_clip.float(), # MODIFIED: used motion_seq_for_clip
                                'y': y_for_motionclip,
                                'mask': lengths_to_mask(lengths11, device=current_device)})["mu"].unsqueeze(1)
            
            # 2. Prepare unconditional style f_s_motion_uncond
            f_s_motion_uncond = torch.zeros_like(f_s_motion_cond)

            # --- Initialize f_scene_indep related tensors (will be filled if scene is available) ---
            # Assuming self.f_scene_latent_dim is defined (e.g., 512 from your debug output)
            # These are based on bsz_orig_style_scene
            # START NEW CODE BLOCK 1
            _f_scene_indep_cond_for_denoiser = torch.zeros(bsz_orig_style_scene, 1, 512, device=current_device)
            _f_scene_indep_uncond_for_denoiser = torch.zeros(bsz_orig_style_scene, 512, device=current_device)
            # END NEW CODE BLOCK 1

            # 3. Process scene features and apply FiLM (existing logic)
            if self.scene_embedding is not None and self.film_mlp is not None and 'scene_labels' in batch: # MODIFIED: Added self.scene_encoder_mlp and self.film_mlp check
                scene_labels = batch['scene_labels'].to(current_device) 
                if scene_labels.dtype != torch.long:
                    scene_labels = scene_labels.long()
                
                # Scene feature for FiLM (as per your existing code: raw embedding)
                f_scene_for_film_cond = self.scene_embedding(scene_labels) # (bsz_orig_style_scene, D_emb)

                # print("f_scene_for_film_cond shape:", f_scene_for_film_cond.shape)
                # print("f_scene_for_film_cond has NaN:", torch.isnan(f_scene_for_film_cond).any())
                # print("f_scene_for_film_cond has Inf:", torch.isinf(f_scene_for_film_cond).any())
                # --- Conditional branch for FiLM ---
                gamma_beta_cond_raw = self.film_mlp(f_scene_for_film_cond) 
                gamma_cond_from_mlp, beta_cond_from_mlp = torch.chunk(gamma_beta_cond_raw, 2, dim=-1)
                final_gamma_cond = 1.0 + torch.tanh(gamma_cond_from_mlp) * 1.0
                final_beta_cond = beta_cond_from_mlp
                gamma_cond_to_apply = final_gamma_cond.unsqueeze(1)
                beta_cond_to_apply = final_beta_cond.unsqueeze(1)
                style_feature_cond_fused = gamma_cond_to_apply * f_s_motion_cond + beta_cond_to_apply

                # --- Unconditional branch for FiLM ---
                f_scene_for_film_uncond = torch.zeros_like(f_scene_for_film_cond)
                gamma_beta_uncond_raw = self.film_mlp(f_scene_for_film_uncond)
                gamma_uncond_from_mlp, beta_uncond_from_mlp = torch.chunk(gamma_beta_uncond_raw, 2, dim=-1)
                final_gamma_uncond = 1.0 + torch.tanh(gamma_uncond_from_mlp) * 1.0
                final_beta_uncond = beta_uncond_from_mlp
                gamma_uncond_to_apply = final_gamma_uncond.unsqueeze(1)
                beta_uncond_to_apply = final_beta_uncond.unsqueeze(1)
                style_feature_uncond_fused = gamma_uncond_to_apply * f_s_motion_uncond + beta_uncond_to_apply
                
                # --- START NEW CODE BLOCK 2: Prepare f_scene_indep for denoiser ---
                # f_scene_for_film_cond is raw embedding. Pass this to scene_encoder_mlp for f_scene_indep
                _f_scene_indep_cond_processed = f_scene_for_film_cond # (bsz_orig_style_scene, D_f_scene_latent)
                _f_scene_indep_cond_for_denoiser = _f_scene_indep_cond_processed.unsqueeze(1)

                # For unconditional f_scene_indep, use zero embedding through scene_encoder_mlp
                _f_scene_indep_uncond_processed = f_scene_for_film_uncond # f_scene_for_film_uncond is zeros
                _f_scene_indep_uncond_for_denoiser = _f_scene_indep_uncond_processed.unsqueeze(1)
                # --- END NEW CODE BLOCK 2 ---

            else: # No scene embedding/module or no scene_labels in batch
                style_feature_cond_fused = f_s_motion_cond
                style_feature_uncond_fused = f_s_motion_uncond
                # _f_scene_indep_cond_for_denoiser and _f_scene_indep_uncond_for_denoiser remain zeros

            # 4. Concatenate f_s_adapted (your 'motion_emb') for CFG
            motion_emb = torch.cat([style_feature_uncond_fused, style_feature_cond_fused], dim=0)
            
            # --- START NEW CODE BLOCK 3: Concatenate f_scene_indep for CFG ---
            f_scene_indep_for_denoiser = torch.cat([_f_scene_indep_uncond_for_denoiser, _f_scene_indep_cond_for_denoiser], dim=0)
            # --- END NEW CODE BLOCK 3 ---
            
            # --- END: 集成场景嵌入 (带CFG处理) ---

            # trajectory
            trans_cond_input = trans_motion[...,:3] # (bsz_orig_style_scene, frames, 3) or (1, frames, 3)
            # Ensure trans_cond batch size matches the CFG duplicated batch size of motion_emb_content
            # motion_emb_content is [2*bsz_orig, ...], so target bsz for trans_cond is motion_emb_content.shape[0]
            # This means trans_cond_input should be repeated to match bsz_orig_style_scene if it's 1,
            # and then doubled for CFG. Your existing uncond_trans does this correctly.
            if trans_cond_input.shape[0] != bsz_orig_style_scene and bsz_orig_style_scene > 1 : # If trans_cond_input is e.g. [1, F, 3] but style/scene batch is > 1
                 trans_cond_input_repeated = trans_cond_input.repeat(bsz_orig_style_scene, 1, 1)
            else:
                 trans_cond_input_repeated = trans_cond_input

            uncond_trans = torch.cat([trans_cond_input_repeated, trans_cond_input_repeated], dim = 0) # This is f_t

            # --- MODIFIED: Add f_scene_indep_for_denoiser ---
            multi_cond_emb = [motion_emb_content, motion_emb, uncond_trans, f_scene_indep_for_denoiser] 
            # --- END MODIFIED ---

            z = self._diffusion_reverse(multi_cond_emb, lengths, scale)

        elif self.stage in ['vae']: # This part remains unchanged as it's for VAE-only stage
            motions = batch['motion']
            z, dist_m = self.vae.encode(motions, lengths)

        with torch.no_grad():
            feats_rst = self.vae.decode(z, lengths) 
        joints = self.feats2joints(feats_rst.detach().cpu()) # feats_rst: torch.Size([1, 199, 263])
        return remove_padding(joints, lengths)


    def _diffusion_reverse(self, encoder_hidden_states, lengths=None, scale=None):
        # init latents
        bsz = encoder_hidden_states[0].shape[0] # 64
        if self.do_classifier_free_guidance:
            bsz = bsz // 2  # 32

        latents = torch.randn(
            (bsz, self.latent_dim[0], self.latent_dim[-1]),
            device=encoder_hidden_states[0].device,
            dtype=torch.float,
        )  # torch.Size([32, 7, 256])

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma # torch.Size([32, 7, 256])
        # set timesteps
        self.scheduler.set_timesteps(
            self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(encoder_hidden_states[0].device) # torch.Size([50])
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta
        

        # reverse
        for i, t in enumerate(timesteps):
            # if i // 10 ==0:
            #     latent_feature.append()
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (torch.cat(
                [latents] *
                2) if self.do_classifier_free_guidance else latents) # torch.Size([64, 7, 256])
            lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance
                               else lengths)  # len(64)
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            noise_pred = self.denoiser(
                sample=latent_model_input, # torch.Size([64, 7, 256])
                timestep=t, # torch.Size([])
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths_reverse,
            )[0]
            
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=1) # 都是torch.Size([7, 32, 256])
                noise_pred = noise_pred_uncond + scale * (
                    noise_pred_text - noise_pred_uncond) # torch.Size([7, 32, 256])
                final_noise_pred_for_scheduler = noise_pred.permute(1, 0, 2) 
            latents = self.scheduler.step(final_noise_pred_for_scheduler, t, latents, # latents：torch.Size([32, 7, 256])
                                              **extra_step_kwargs).prev_sample

        latents = latents.permute(1, 0, 2)
        return latents





    def _diffusion_process(self, latents, encoder_hidden_states, lengths=None):
        """
        heavily from https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
        """
        # our latent   [batch_size, n_token=1 or 5 or 10, latent_dim=256]
        # sd  latent   [batch_size, [n_token0=64,n_token1=64], latent_dim=4]
        # [n_token, batch_size, latent_dim] -> [batch_size, n_token, latent_dim]
        latents = latents.permute(1, 0, 2) # torch.Size([32, 7, 256])

        # Sample noise that we'll add to the latents
        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(latents) # torch.Size([32, 7, 256])
        bsz = latents.shape[0] # 32
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=latents.device,
        )
        timesteps = timesteps.long() # torch.Size([32])
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents.clone(), noise,
                                                       timesteps) # torch.Size([32, 7, 256])
        # Predict the noise residual
        noise_pred = self.denoiser(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            lengths=lengths,
            return_dict=False,
        )[0]  # torch.Size([32, 7, 256])
        noise_pred = noise_pred.permute(1, 0, 2)
        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
        if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
            noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
            noise, noise_prior = torch.chunk(noise, 2, dim=0)
        else:  # 目前会进这里
            noise_pred_prior = 0
            noise_prior = 0


        n_set = {
            "noise": noise,
            "noise_prior": noise_prior,
            "noise_pred": noise_pred,
            "noise_pred_prior": noise_pred_prior,
        }

        return n_set

    def train_vae_forward(self, batch):
        feats_ref = batch["motion"]
        lengths = batch["length"]


        motion_z, dist_m = self.vae.encode(feats_ref, lengths)##########(1,128,256)/
        feats_rst = self.vae.decode(motion_z, lengths)


        # prepare for metric
        recons_z, dist_rm = self.vae.encode(feats_rst, lengths)

        # joints recover
        joints_rst = self.feats2joints(feats_rst)
        joints_ref = self.feats2joints(feats_ref)

        if dist_m is not None:
            if self.is_vae:
                # Create a centred normal distribution to compare with
                mu_ref = torch.zeros_like(dist_m.loc)
                scale_ref = torch.ones_like(dist_m.scale)
                dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
            else:
                dist_ref = dist_m

        # cut longer part over max length
        min_len = min(feats_ref.shape[1], feats_rst.shape[1])
        rs_set = {
            "m_ref": feats_ref[:, :min_len, :],
            "m_rst": feats_rst[:, :min_len, :],
            # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
            "lat_m": motion_z.permute(1, 0, 2),
            "lat_rm": recons_z.permute(1, 0, 2),
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "dist_m": dist_m,
            "dist_ref": dist_ref,
        }
        return rs_set
    
    
# train
    def train_diffusion_forward(self, batch):
        
        feats_ref = batch["motion"] # torch.Size([32, 32:帧数, 263：特征维度])
        feats_content = batch["motion"].clone() # torch.Size([32, 32:帧数, 263：特征维度])
        feats_content[...,:3] = 0.0 # torch.Size([32, 32:帧数, 263：特征维度])
        lengths = batch["length"] # list：len=32
        bsz = feats_ref.shape[0] # NEW: Get batch size for convenience
        current_device = feats_ref.device # NEW: Get current device
        
        # content condition
        with torch.no_grad():
            z, dist = self.vae.encode(feats_ref, lengths) # z: torch.Size([7, 32, 256]), dist: torch.Size([7, 32, 256]), 第一维是隐变量序列长度，第二维才是batch size
            z_content, dist = self.vae.encode(feats_content, lengths) # z_content: torch.Size([7, 32, 256]), dist: torch.Size([7, 32, 256])
            cond_emb = z_content.permute(1,0,2)   # shape: torch.Size([32, 7, 256])  
            if self.cfg.model.useDCE:
                fc_from_dce = self.dce_model(cond_emb)
                cond_emb = fc_from_dce   
        # style condition
        motion_seq = feats_ref*self.std + self.mean # shape:torch.Size([32, 32, 263]) # 反归一化，也就是说content归一化，但是style不归一化（因为motionclip吃的style，需要原始尺度，所以style需要反归一化）
        motion_seq[...,:3]=0.0 # shape:torch.Size([32, 32, 263])
        motion_seq = motion_seq.unsqueeze(-1).permute(0,2,3,1)  # torch.Size([32, 263, 1, 32])
   
        motion_seq = motion_seq.float() # torch.Size([32, 263, 1, 32])
        raw_motion_emb = self.motionclip.encoder({'x': motion_seq,
                        'y': torch.zeros(bsz, dtype=int, device=current_device), # MODIFIED: use bsz and current_device
                        'mask': lengths_to_mask(lengths, device=current_device)})["mu"] 
        raw_motion_emb = raw_motion_emb.unsqueeze(1) # torch.Size([32, 1, 512])
        
        # Apply CFG mask to raw style. This will be the input to FiLM if scene exists,
        # or the final style if no scene.
        # `motion_emb` will now represent the (potentially masked) style before FiLM or final adapted style.
        motion_emb = raw_motion_emb.clone() # Start with raw style
        mask_uncond = torch.rand(bsz, device=current_device) < self.guidance_uncodp  # MODIFIED: use bsz and current_device
        motion_emb[mask_uncond, ...] = 0 # Apply CFG mask. motion_emb is now f_s_motion (masked)
    
        motion_emb = motion_emb.float() # Ensure float
        
        # --- START: MODIFIED BLOCK FOR SCENE INTEGRATION (FiLM and f_scene_indep) ---
        
        # Initialize f_scene_indep with zeros. It will be populated if scene info is available.
        # Assuming self.f_scene_latent_dim is defined in __init__ (e.g., 256)
        f_scene_indep = torch.zeros(bsz, 1, 512, device=current_device) # torch.Size([32, 1, 512])

        if self.scene_embedding is not None and self.film_mlp is not None and 'scene_labels' in batch:
            scene_labels = batch['scene_labels'].to(current_device) 
            if scene_labels.dtype != torch.long: 
                 scene_labels = scene_labels.long()
            
            # 1. Generate f_scene from scene_labels (used for both FiLM and f_scene_indep)
            # This f_scene has dimension self.f_scene_latent_dim after scene_encoder_mlp
            embedded_scene = self.scene_embedding(scene_labels) # (bs, D_emb_raw)
            # Process through scene_encoder_mlp to get the desired dimension and representation
            # f_scene_processed = self.scene_encoder_mlp(embedded_scene) # (bs, D_f_scene_latent)
            f_scene_processed = embedded_scene

            # 2. Prepare f_scene_indep for the denoiser
            # Unsqueeze to add sequence dimension: (bs, 1, D_f_scene_latent)
            f_scene_indep_temp = f_scene_processed.unsqueeze(1)
            
            # Apply the same CFG mask to f_scene_indep if scene influence is conditional
            f_scene_indep = f_scene_indep_temp.clone()  # torch.Size([32, 1, 512])
            f_scene_indep[mask_uncond, ...] = 0 # Mask f_scene_indep

            # 3. Apply FiLM to modulate `motion_emb` (which is already masked f_s_motion)
            # FiLM MLP takes f_scene_processed (unmasked, or masked if FiLM itself is conditional)
            # Let's assume FiLM parameters are generated from unmasked scene features,
            # and the CFG effect comes from modulating an already masked `motion_emb`.
            gamma_beta_raw = self.film_mlp(f_scene_processed) # (bs, style_dim * 2)
            gamma_raw, beta_raw = torch.chunk(gamma_beta_raw, 2, dim=-1)

            # Ensure gamma/beta match the dimension of motion_emb (style_dim, e.g., 512)
            gamma = (1.0 + torch.tanh(gamma_raw) * 1.0).unsqueeze(1) # (bs, 1, style_dim)
            beta = beta_raw.unsqueeze(1) # (bs, 1, style_dim)
            
            # `motion_emb` is updated to be f_s_adapted
            motion_emb = gamma * motion_emb + beta # FiLM application
        
        # `motion_emb` is now f_s_adapted (if scene) or masked f_s_motion (if no scene)
        # `f_scene_indep` is prepared (zeros if no scene, or masked f_scene_indep if scene)
        
        # --- END: MODIFIED BLOCK FOR SCENE INTEGRATION ---

        # trans condition
        trans_cond = batch["motion"][...,:3].to(current_device) # f_t, ensure device

        # --- MODIFIED: Add f_scene_indep to multi_cond_emb ---
        # `motion_emb` is f_s_adapted torch.Size([32, 1, 512])
        # `cond_emb` is f_c  torch.Size([32, 7, 256])
        # `trans_cond` is f_t  torch.Size([32, 32, 3])
        # `f_scene_indep` is the new independent scene condition # torch.Size([32, 1, 512])
        multi_cond_emb = [cond_emb, motion_emb, trans_cond, f_scene_indep] # 1.f_c 2.f_adpated 3.f_t 4.f_scene
        # --- END MODIFIED ---

        # diffusion process return with noise and noise_pred
        n_set = self._diffusion_process(z, multi_cond_emb, lengths) # z:输入的motion
        # print(f"DEBUG MLD.train_diffusion_forward: n_set keys from _diffusion_process: {n_set.keys()}") # ADD THIS
        return {**n_set}



# 
# evaluate the reconstruction in training time
# 

    def t2m_eval(self, batch):
        texts = batch["text"]
        motions = batch["motion"].detach().clone()

        # content 
        content_motions = batch["motion"].detach().clone()
        content_motions[...,:3] = 0.0

        lengths = batch["length"]
        word_embs = batch["word_embs"].detach().clone()
        pos_ohot = batch["pos_ohot"].detach().clone()
        text_lengths = batch["text_len"].detach().clone()

        motion = batch["motion"].detach().clone()
        

        # start
        start = time.time()
        current_device = motions.device 

        # if self.trainer.datamodule.is_mm:
        #     texts = texts * self.cfg.TEST.MM_NUM_REPEATS
        #     style_texts = style_texts * self.cfg.TEST.MM_NUM_REPEATS
        #     motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
        #                                         dim=0)
        #     motion = motion.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
        #                                         dim=0)
        #     lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
        #     word_embs = word_embs.repeat_interleave(
        #         self.cfg.TEST.MM_NUM_REPEATS, dim=0)
        #     pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
        #                                           dim=0)
        #     text_lengths = text_lengths.repeat_interleave(
        #         self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.trainer.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            # 'style_texts' was in your original MLD.py, if it's not used here, it's fine.
            # If style_texts exists in batch and needs repeating:
            # if 'style_texts' in batch and batch['style_texts'] is not None:
            #    style_texts = batch['style_texts'] * self.cfg.TEST.MM_NUM_REPEATS
            
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            # Re-assign 'motion' if it's used as the style source and needs repeating
            motion = motion.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            
            lengths_list = batch["length"] # Keep original list for a moment
            lengths = []
            for l_item in lengths_list: # lengths is a list of ints
                lengths.extend([l_item] * self.cfg.TEST.MM_NUM_REPEATS) # Correct way to repeat list of ints
            word_embs = word_embs.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            text_lengths = text_lengths.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
          
            # style (f_s_adapted for denoiser, here it's raw style + CFG)
            # This 'motion' is the (potentially repeated) style source
            motion_seq = motion * self.std.to(current_device) + self.mean.to(current_device) # MODIFIED: ensure std/mean are on current_device
            motion_seq[...,:3]=0.0
            motion_seq = motion_seq.unsqueeze(-1).permute(0,2,3,1)
            motion_seq = motion_seq.float()

            # Original 'motion_emb' (style condition)
            _motion_emb_cond_part = self.motionclip.encoder({'x': motion_seq, # MODIFIED: Removed .to(device) as motion_seq is already on device
                          'y': torch.zeros(motion_seq.shape[0], dtype=int, device=current_device), # MODIFIED: use current_device
                          'mask': lengths_to_mask(lengths, device=current_device)})["mu"] # MODIFIED: use current_device
            _motion_emb_cond_part = _motion_emb_cond_part.unsqueeze(1)
            
            uncond_motion_emb = torch.zeros(_motion_emb_cond_part.shape).to(current_device) # MODIFIED: use current_device
            motion_emb = torch.cat([uncond_motion_emb, _motion_emb_cond_part], dim=0) # This is f_s_adapted (without FiLM for t2m_eval)

            # --- START: MM_NUM_REPEATS 处理场景标签 (如果使用) ---
            # This block was for handling scene_labels for MM, which is good to keep.
            # style_motion_ref = batch["motion"].detach().clone() # This was a bit redundant if 'motion' is already the style ref.
            # current_device = style_motion_ref.device # current_device already defined

            if self.trainer.datamodule.is_mm:
                if hasattr(self, 'scene_embedding') and self.scene_embedding is not None and 'scene_labels' in batch: # MODIFIED: Check for scene_embedding attribute
                    if batch.get('scene_labels') is not None: # MODIFIED: Check if scene_labels is actually in batch and not None
                        batch['scene_labels'] = batch['scene_labels'].repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            # --- END ---
           
            # content condition (f_c)
            # This 'content_motions' uses 'batch["motion"]' as base.
            # If MM_NUM_REPEATS applied to 'motions', then 'content_motions' here will also be repeated.
            # This is consistent if content and style source are the same for t2m_eval.
            # If content should NOT be repeated, then content_motions = batch_original["motion"]...
            # Assuming current 'content_motions' (derived from potentially repeated 'motions') is intended.
            _content_motions_for_vae = content_motions.detach().clone() # Use a distinct name for clarity
            _content_motions_for_vae[...,:3] = 0.0
            _content_motions_for_vae = (_content_motions_for_vae - self.mean.to(current_device)) / self.std.to(current_device) # Normalize

            with torch.no_grad():
                z_content, dist_m_content = self.vae.encode(_content_motions_for_vae.float(), lengths) 
            
            uncond_tokens_content = torch.cat([z_content, z_content], dim = 1).permute(1,0,2)
            motion_emb_content = uncond_tokens_content 

            # trans (f_t)
            # This 'trans_cond' also uses 'batch["motion"]' as base.
            _trans_cond_base = batch["motion"].detach().clone() # Start from original batch["motion"] before potential repeats
            if self.trainer.datamodule.is_mm: # If MM, then base for trans_cond should be repeated
                _trans_cond_base = _trans_cond_base.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            
            trans_cond = _trans_cond_base[...,:3] # Now trans_cond has correct batch size for CFG
            uncond_trans = torch.cat([trans_cond, trans_cond], dim = 0)

            # --- START: ADD PLACEHOLDER FOR f_scene_indep ---
            # motion_emb_content is [2*bsz_orig, ...], use its batch dim
            _bs_for_placeholder = motion_emb_content.shape[0] 
            # self.f_scene_latent_dim should be defined in __init__ (e.g., 512)
            _f_scene_indep_placeholder = torch.zeros(_bs_for_placeholder, 1, 512, device=current_device)
            # --- END: ADD PLACEHOLDER FOR f_scene_indep ---

            # --- MODIFIED: Add placeholder to multi_cond_emb ---
            multi_cond_emb = [motion_emb_content, motion_emb, uncond_trans, _f_scene_indep_placeholder]
            # --- END MODIFIED ---
            z = self._diffusion_reverse(multi_cond_emb, lengths,scale=self.guidance_scale)
        
        elif self.stage in ['vae']: # This part remains unchanged
            # Need to ensure 'motions' here is the original batch['motion'] without MM repeats if VAE stage is separate
            # However, if this is just an alternative path when diffusion is not run,
            # and 'motions' was already repeated by MM logic, then it's fine.
            # For safety, let's use batch["motion"] directly if no MM, or the repeated 'motions' if MM.
            _motions_for_vae_stage = batch["motion"].detach().clone()
            if self.trainer.datamodule.is_mm:
                _motions_for_vae_stage = motions # Use the already repeated 'motions' variable
            
            z, dist_m = self.vae.encode(_motions_for_vae_stage, lengths) # lengths should match _motions_for_vae_stage

        with torch.no_grad():
            feats_rst = self.vae.decode(z, lengths)

        # end time
        end = time.time()
        if not hasattr(self, 'times') or self.times is None: # Initialize self.times if it doesn't exist
            self.times = []
        self.times.append(end - start)

        # joints recover
        joints_rst = self.feats2joints(feats_rst)
        # Ensure 'motions' used for joints_ref has the same batch size as feats_rst
        _motions_for_joints_ref = batch["motion"].detach().clone()
        if self.trainer.datamodule.is_mm: # If MM processing was done
            # feats_rst batch size is 2 * original_bs * MM_repeats if it came through diffusion reverse
            # or original_bs * MM_repeats if it came through VAE stage after MM on 'motions'
            # The 'motions' variable here (if from top of function) is already MM_repeated.
            _motions_for_joints_ref = motions # Use the 'motions' variable that was MM_repeated.
        elif feats_rst.shape[0] != _motions_for_joints_ref.shape[0] and self.stage in ['diffusion', 'vae_diffusion']:
            # This case might occur if feats_rst is CFG-doubled but _motions_for_joints_ref is not.
            # However, decode's output z is usually not CFG-doubled.
            # Let's assume z from _diffusion_reverse is [tokens, bsz_orig_after_MM, latent_dim]
            # So feats_rst is [bsz_orig_after_MM, frames, feat_dim]
            # And _motions_for_joints_ref should also be [bsz_orig_after_MM, frames, feat_dim]
            pass # Assuming 'motions' (if MM) or batch["motion"] (if not MM) has correct batch for joints_ref

        joints_ref = self.feats2joints(_motions_for_joints_ref.to(current_device)) # MODIFIED: ensure device

        # renorm for t2m evaluators
        # Use copies for renorm to avoid in-place modification issues
        _feats_rst_for_renorm = feats_rst.clone()
        _motions_for_renorm = _motions_for_joints_ref.clone() # Use the same motion source as for joints_ref

        feats_rst = self.datamodule.renorm4t2m(_feats_rst_for_renorm)
        motions = self.datamodule.renorm4t2m(_motions_for_renorm) # Re-assign 'motions' to its renormed version

        # t2m motion encoder
        # lengths for m_lens should correspond to the 'motions' and 'feats_rst' being processed
        m_lens_for_sort = lengths # 'lengths' should be the (possibly MM_repeated) list of frame counts
        
        m_lens = torch.tensor(m_lens_for_sort, device=current_device) # MODIFIED: use current_device
        align_idx = np.argsort(m_lens.data.cpu().tolist())[::-1].copy() # MODIFIED: ensure .cpu() before .tolist()
        
        motions = motions[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx] # m_lens is now a tensor
        
        m_lens_div = torch.div(m_lens, # MODIFIED: renamed from m_lens to m_lens_div for clarity
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens_div) # MODIFIED: use m_lens_div
        
        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        # Original code re-assigns 'motion_emb'. Let's use a new name for the output of t2m_motionencoder.
        motion_emb_encoded_gt = self.t2m_motionencoder(motion_mov, m_lens_div) # MODIFIED: use m_lens_div

        # t2m text encoder
        # Ensure word_embs, pos_ohot, text_lengths are on current_device and aligned
        _word_embs_aligned = word_embs[align_idx].to(current_device)
        _pos_ohot_aligned = pos_ohot[align_idx].to(current_device)
        _text_lengths_aligned = text_lengths[align_idx].to(current_device) # This should be a tensor of ints
        if not isinstance(_text_lengths_aligned, torch.Tensor): # Ensure it's a tensor for indexing
            _text_lengths_aligned = torch.tensor(_text_lengths_aligned, device=current_device, dtype=torch.long)


        text_emb = self.t2m_textencoder(_word_embs_aligned, _pos_ohot_aligned,
                                        _text_lengths_aligned) # text_encoder expects lengths on CPU usually, check its requirements.
                                                              # If it needs CPU: _text_lengths_aligned.cpu()

        rs_set = {
            "m_ref": motions,
            "m_rst": feats_rst,
            "lat_t": text_emb,
            "lat_m": motion_emb_encoded_gt, # MODIFIED: use new variable name
            "lat_rm": recons_emb,
            "joints_ref": joints_ref[align_idx], # MODIFIED: align joints_ref
            "joints_rst": joints_rst[align_idx], # MODIFIED: align joints_rst
        }
        return rs_set


    def a2m_gt(self, batch):
        actions = batch["action"]
        actiontexts = batch["action_text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        mask = batch["mask"]

        joints_ref = self.feats2joints(motions.to('cuda'), mask.to('cuda'))

        rs_set = {
            "m_action": actions,
            "m_text": actiontexts,
            "m_ref": motions,
            "m_lens": lengths,
            "joints_ref": joints_ref,
        }
        return rs_set

    def eval_gt(self, batch, renoem=True):
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]

        # feats_rst = self.datamodule.renorm4t2m(feats_rst)
        if renoem:
            motions = self.datamodule.renorm4t2m(motions)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        word_embs = batch["word_embs"].detach()
        pos_ohot = batch["pos_ohot"].detach()
        text_lengths = batch["text_len"].detach()

        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot,
                                        text_lengths)[align_idx]

        # joints recover
        joints_ref = self.feats2joints(motions)

        rs_set = {
            "m_ref": motions,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "joints_ref": joints_ref,
        }
        return rs_set

    def allsplit_step(self, split: str, batch, batch_idx): # pytorch_lightning
        if split in ["train", "val"]:
            # print(f"DEBUG MLD.allsplit_step: split={split}, self.stage={self.stage}") # ADD THIS


            if self.stage == "vae":
                rs_set = self.train_vae_forward(batch)
                rs_set["lat_t"] = rs_set["lat_m"]



            elif self.stage == "diffusion":#
                rs_set = self.train_diffusion_forward(batch)


            elif self.stage == "vae_diffusion":
                vae_rs_set = self.train_vae_forward(batch)
                diff_rs_set = self.train_diffusion_forward(batch)
                t2m_rs_set = self.test_diffusion_forward(batch,
                                                         finetune_decoder=True)
                # merge results
                rs_set = {
                    **vae_rs_set,
                    **diff_rs_set,
                    "gen_m_rst": t2m_rs_set["m_rst"],
                    "gen_joints_rst": t2m_rs_set["joints_rst"],
                    "lat_t": t2m_rs_set["lat_t"],
                }
            else:
                raise ValueError(f"Not support this stage {self.stage}!")

            loss = self.losses[split].update(rs_set)
            if loss is None:
                raise ValueError(
                    "Loss is None, this happend with torchmetrics > 0.7")

        # Compute the metrics - currently evaluate results from text to motion
        if split in ["val", "test"]:
            # use t2m evaluators
            rs_set = self.t2m_eval(batch)

            # MultiModality evaluation sperately
            if self.trainer.datamodule.is_mm:
                metrics_dicts = ['MMMetrics']
            else:
                metrics_dicts = self.metrics_dict
            # metric = 'TemosMetric' 'TM2TMetrics'
            for metric in metrics_dicts:
                if metric == "TemosMetric":
                    phase = split if split != "val" else "eval"
                    if eval(f"self.cfg.{phase.upper()}.DATASETS")[0].lower(
                    ) not in [
                            "humanml3d",
                            "kit",
                            "humanml3d_scene"
                    ]:
                        raise TypeError(
                            "APE and AVE metrics only support humanml3d and kit datasets now"
                        )

                    getattr(self, metric).update(rs_set["joints_rst"],
                                                 rs_set["joints_ref"],
                                                 batch["length"])
                elif metric == "TM2TMetrics":
                    getattr(self, metric).update(
                        # lat_t, latent encoded from diffusion-based text
                        # lat_rm, latent encoded from reconstructed motion
                        # lat_m, latent encoded from gt motion
                        # rs_set['lat_t'], rs_set['lat_rm'], rs_set['lat_m'], batch["length"])
                        rs_set["lat_t"],
                        rs_set["lat_rm"],
                        rs_set["lat_m"],
                        batch["length"],
                    )
                elif metric == "UncondMetrics":
                    getattr(self, metric).update(
                        recmotion_embeddings=rs_set["lat_rm"],
                        gtmotion_embeddings=rs_set["lat_m"],
                        lengths=batch["length"],
                    )
                elif metric == "MRMetrics":
                    getattr(self, metric).update(rs_set["joints_rst"],
                                                 rs_set["joints_ref"],
                                                 batch["length"])
                elif metric == "MMMetrics":
                    getattr(self, metric).update(rs_set["lat_rm"].unsqueeze(0),
                                                 batch["length"])
                elif metric == "HUMANACTMetrics":
                    getattr(self, metric).update(rs_set["m_action"],
                                                 rs_set["joints_eval_rst"],
                                                 rs_set["joints_eval_ref"],
                                                 rs_set["m_lens"])
                elif metric == "UESTCMetrics":
                    # the stgcn model expects rotations only
                    getattr(self, metric).update(
                        rs_set["m_action"],
                        rs_set["m_rst"].view(*rs_set["m_rst"].shape[:-1], 6,
                                             25).permute(0, 3, 2, 1)[:, :-1],
                        rs_set["m_ref"].view(*rs_set["m_ref"].shape[:-1], 6,
                                             25).permute(0, 3, 2, 1)[:, :-1],
                        rs_set["m_lens"])
                else:
                    raise TypeError(f"Not support this metric {metric}")

        # return forward output rather than loss during test
        if split in ["test"]:
            return rs_set["joints_rst"], batch["length"]
        return loss
