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
from omegaconf import OmegaConf # 确保 OmegaConf 被导入，如果还没有的话




from .base import BaseModel
import logging

# 初始化日志记录器
logger = logging.getLogger(__name__)

class MLD(BaseModel):
    """
    Stage 1 vae
    Stage 2 diffusion
    """

    def __init__(self, cfg, datamodule, **kwargs):
        super().__init__()

        self.cfg = cfg
        self.datamodule = datamodule # [修改] 将 datamodule 保存为实例属性
        self.stage = cfg.TRAIN.STAGE
        self.is_vae = cfg.model.vae
        self.nfeats = cfg.DATASET.NFEATS
        self.njoints = cfg.DATASET.NJOINTS
        self.debug = cfg.DEBUG
        self.latent_dim = cfg.model.latent_dim
        self.guidance_scale = cfg.model.guidance_scale
        self.guidance_uncodp = cfg.model.guidance_uncondp

        # --- [新增] 加载并冻结 CLIP 模型 ---
        print("Loading CLIP model for text conditioning...")
        self.clip_model, _ = clip.load("ViT-B/32", device="cpu")
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False
        print("CLIP model loaded and frozen.")
        
        # --- [保留] 原始 MotionCLIP 的加载逻辑 ---
        parameters = read_yaml_to_dict("configs/motionclip_config/motionclip_params_263.yaml")
        parameters["device"] = 'cuda:{}'.format(cfg["DEVICE"][0])        
        self.motionclip = get_model_and_data(parameters, split='vald')
        print("Loading original MotionCLIP...")
        checkpointpath = "checkpoints/motionclip_checkpoint/motionclip.pth.tar"
        state_dict = torch.load(checkpointpath, map_location=parameters["device"])
        load_model_wo_clip(self.motionclip, state_dict)
        
        # --- [核心修复] 从 self.datamodule.norms 中获取 mean 和 std ---
        if not hasattr(self.datamodule, 'norms'):
            # 这是一个安全检查，确保 datamodule 已经被 setup
            # 在 train.py 的标准流程中，get_datasets 会调用 setup
            raise AttributeError("DataModule has not been set up yet, 'norms' attribute not found.")
            
        self.mean = torch.tensor(self.datamodule.norms['mean']).to(parameters["device"])
        self.std = torch.tensor(self.datamodule.norms['std']).to(parameters["device"])
        # -----------------------------------------------------------------

        self.motionclip.training = False
        for p in self.motionclip.parameters():
            p.requires_grad = False
        print("MotionCLIP loaded and frozen.")

        # --- [保留] VAE 和 Denoiser 的实例化 ---
        self.vae = instantiate_from_config(cfg.model.motion_vae)
        if self.stage == "diffusion":
            print("Freezing VAE parameters for diffusion training.")
            self.vae.training = False
            for p in self.vae.parameters():
                p.requires_grad = False

        self.denoiser = instantiate_from_config(cfg.model.denoiser)

        # --- [保留] 其他所有原始初始化逻辑 ---
        self.scheduler = instantiate_from_config(cfg.model.scheduler)
        self.noise_scheduler = instantiate_from_config(cfg.model.noise_scheduler)

        self._get_t2m_evaluator(cfg)

        if cfg.LOSS.TYPE == "mld":
            self._losses = MetricCollection({
                split: MLDLosses(vae=self.is_vae, mode="xyz", cfg=cfg)
                for split in ["losses_train", "losses_test", "losses_val"]
            })
        else:
            raise NotImplementedError
        self.losses = {key: self._losses["losses_" + key] for key in ["train", "test", "val"]}
        
        self.metrics_dict = cfg.METRIC.TYPE
        self.configure_metrics()

        self.sample_mean = False
        self.fact = None
        self.do_classifier_free_guidance = True

        self.feats2joints = datamodule.feats2joints
        self.joints2feats = datamodule.joints2feats

     # --- [新增] PyTorch Lightning 标准的 setup 方法 ---
    def setup(self, stage=None):
        """
        这个方法在 __init__ 之后、训练开始之前被 trainer 调用。
        此时，datamodule.setup() 已经被调用过了。
        """
        # 只有在训练或验证阶段才需要加载 mean/std
        if stage == 'fit' or stage is None:
            logger.info("Setting up mean and std in MLD module...")
            
            # 从 datamodule 获取 mean 和 std
            # 我们的 MixedDataModule 会把它们放在 hparams 中
            # 为了安全，我们检查一下
            if hasattr(self.datamodule, 'norms') and 'mean' in self.datamodule.norms:
                self.mean = torch.tensor(self.datamodule.norms['mean'], device=self.device)
                self.std = torch.tensor(self.datamodule.norms['std'], device=self.device)
                logger.info("Mean and std loaded successfully from datamodule.norms.")
            # ---------------------
            else:
                logger.warning("Could not find mean/std in datamodule.norms.")
                
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
        dataname = "t2m" if dataname == "humanml3d" else dataname
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




    def configure_optimizers(self):
        # 从配置文件中读取学习率配置
        lr_config = self.cfg.TRAIN.OPTIM.LR_CONFIG
        
        param_groups = []
        print("--- Configuring Optimizers with Differential Learning Rates ---")
        
        # 建立一个集合来跟踪已经被分配的参数，防止重复分配
        assigned_params = set()

        # 优先分配学习率最高的组 (通常是新层)
        sorted_lr_config = sorted(lr_config, key=lambda x: x['lr'], reverse=True)

        for group_config in sorted_lr_config:
            name = group_config['name']
            layers_key = group_config['layers']
            lr = group_config['lr']
            
            params_to_optimize = []
            if layers_key == "cross_attn":
                for block in self.denoiser.blocks:
                    for param in block.cross_attn.parameters():
                        if param not in assigned_params:
                            params_to_optimize.append(param)
                            assigned_params.add(param)
            elif layers_key == "denoiser":
                for param in self.denoiser.parameters():
                    if param not in assigned_params:
                        params_to_optimize.append(param)
                        assigned_params.add(param)
            
            if params_to_optimize:
                param_groups.append({'params': params_to_optimize, 'lr': lr})
                print(f"Optimizer group '{name}' configured with LR={lr} for {len(params_to_optimize)} parameters.")

        # 检查是否有任何 Denoiser 参数未被分配
        unassigned_params = [p for p in self.denoiser.parameters() if p not in assigned_params]
        if unassigned_params:
            print(f"Warning: {len(unassigned_params)} parameters in Denoiser were not assigned to any optimizer group.")

        # 使用配置文件中指定的优化器类型
        if self.cfg.TRAIN.OPTIM.TYPE.lower() == "adamw":
            return AdamW(param_groups)
        else:
            raise NotImplementedError(f"Optimizer {self.cfg.TRAIN.OPTIM.TYPE} not supported.")





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
    
    def forward(self, batch):
        """
        这是模型的推理入口。
        它现在支持两种风格迁移模式:
        1. 动作引导: 如果 batch 中包含 'style_motion'。
        2. 文本引导: 如果 batch 中包含 'style_text'。
        """
        # --- 1. 准备通用条件：内容 和 轨迹 ---
        lengths = batch["length"]
        
        # 内容动作预处理
        content_motion = batch['content_motion']
        # 注意: 推理时的数据也需要归一化
        content_motion_normalized = (content_motion - self.mean.to(content_motion.device)) / self.std.to(content_motion.device)
        
        # 轨迹条件
        trans_motion = content_motion_normalized.clone()
        
        # 移除轨迹以准备内容条件
        content_motion_no_trans = content_motion_normalized.clone()
        content_motion_no_trans[..., :3] = 0

        scale = batch.get("tag_scale", self.guidance_scale) # 使用 batch 中的 scale，如果没有则用默认值

        if self.stage in ['diffusion', 'vae_diffusion']:
            # --- 2. 准备内容条件和轨迹条件的 embedding ---
            # 这部分对于两种模式是通用的
            with torch.no_grad():
                # 内容 embedding
                z_content, _ = self.vae.encode(content_motion_no_trans.float(), lengths)
                # 为 classifier-free guidance 复制一份
                motion_emb_content = torch.cat([z_content, z_content], dim=1).permute(1, 0, 2)

                # 轨迹 embedding
                trans_cond = trans_motion[..., :3]
                uncond_trans = torch.cat([trans_cond, trans_cond], dim=0)

            # --- 3. [核心改造] 模式选择：准备风格条件 ---
            style_text_feature = None # 初始化文本特征
            motion_emb = None         # 初始化动作特征

            if 'style_text' in batch:
                # *** 分支一：文本引导模式 ***
                logger.info("Performing text-guided style transfer.")
                texts = batch['style_text']
                with torch.no_grad():
                    tokenized_text = clip.tokenize(texts, truncate=True).to(self.device)
                    # 编码文本特征，并为 classifier-free guidance 准备无条件部分
                    text_features = self.clip_model.encode_text(tokenized_text).float()
                    uncond_text_features = torch.zeros_like(text_features)
                    style_text_feature = torch.cat([uncond_text_features, text_features], dim=0)

            elif 'style_motion' in batch:
                # *** 分支二：动作引导模式 (原始逻辑) ***
                logger.info("Performing motion-guided style transfer.")
                style_motion = batch['style_motion'].clone()
                style_motion[..., :3] = 0 # 移除轨迹

                with torch.no_grad():
                    # 使用 MotionCLIP 编码
                    motion_seq = style_motion.unsqueeze(-1).permute(0, 2, 3, 1)
                    motion_emb_raw = self.motionclip.encoder({
                        'x': motion_seq.float(),
                        'y': torch.zeros(motion_seq.shape[0], dtype=int, device=motion_seq.device),
                        'mask': lengths_to_mask(lengths, device=motion_seq.device)
                    })["mu"]
                    motion_emb_raw = motion_emb_raw.unsqueeze(1)
                    
                    # 准备无条件部分
                    uncond_motion_emb = torch.zeros_like(motion_emb_raw)
                    motion_emb = torch.cat([uncond_motion_emb, motion_emb_raw], dim=0)

            else:
                raise ValueError("Inference batch must contain either 'style_text' or 'style_motion'.")

            # --- 4. 组装并调用反向扩散过程 ---
            multi_cond_emb = [motion_emb_content, motion_emb, uncond_trans]
            
            # [关键] 将文本特征也传入
            z = self._diffusion_reverse(
                encoder_hidden_states=multi_cond_emb, 
                lengths=lengths, 
                scale=scale,
                style_text_feature=style_text_feature # 传入新参数
            )

        elif self.stage in ['vae']:
            # VAE 模式的推理逻辑保持不变
            motions = batch['motion']
            z, dist_m = self.vae.encode(motions, lengths)
        else:
            raise ValueError(f"Stage {self.stage} not supported for inference.")

        # --- 5. VAE 解码并返回结果 (通用) ---
        with torch.no_grad():
            feats_rst = self.vae.decode(z, lengths)

        joints = self.feats2joints(feats_rst.detach().cpu())
        return remove_padding(joints, lengths)
    


    def _diffusion_reverse(self, encoder_hidden_states, lengths=None, scale=None, style_text_feature=None):
        # init latents
        bsz = encoder_hidden_states[0].shape[0]
        if self.do_classifier_free_guidance:
            bsz = bsz // 2

        latents = torch.randn(
            (bsz, self.latent_dim[0], self.latent_dim[-1]),
            device=encoder_hidden_states[0].device,
            dtype=torch.float,
        )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(
            self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(encoder_hidden_states[0].device)
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
                2) if self.do_classifier_free_guidance else latents)
            lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance
                               else lengths)
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            # [新增] 如果有文本特征，也需要为 classifier-free guidance 准备
            style_text_feature_input = style_text_feature
            if self.do_classifier_free_guidance and style_text_feature is not None:
                # 我们的 style_text_feature 在 forward 中已经准备好了 cond/uncond 对
                style_text_feature_input = style_text_feature
            
            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths_reverse,
                style_text_feature=style_text_feature_input, # [关键] 传入
            )[0]
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + scale * (
                    noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample

        latents = latents.permute(1, 0, 2)
        return latents





    def _diffusion_process(self, latents, encoder_hidden_states, lengths=None, style_text_feature=None):
        """
        heavily from https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
        """
        # 1. [保留] latents (1, B, D) -> (B, L, D), L=latent_len (e.g., 7)
        latents = latents.permute(1, 0, 2)
        
        # 2. [保留] noise 的形状与 latents 一致: (B, L, D)
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        # 3. [保留] 加噪
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=latents.device,
        ).long()
        noisy_latents = self.noise_scheduler.add_noise(latents.clone(), noise, timesteps)
        
        # 4. [保留] 准备 denoiser 输入: (B, L, D) -> (B, D, L)
        noisy_latents_for_denoiser = noisy_latents.permute(0, 2, 1)

        # 5. [保留] 调用 denoiser，其输出 noise_pred_raw 的形状是 (B, D, L)
        noise_pred_raw = self.denoiser(
            sample=noisy_latents_for_denoiser,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            lengths=lengths,
            style_text_feature=style_text_feature,
            return_dict=False,
        )[0]

        # --- [最终修复] ---
        # 6. 将 denoiser 的输出转置回来，以匹配 noise 的形状
        # (B, D, L) -> (B, L, D)
        noise_pred = noise_pred_raw.permute(0, 2, 1)
        # --------------------
        
        # 7. [保留] prior loss 逻辑
        if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
            noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
            noise, noise_prior = torch.chunk(noise, 2, dim=0)
        else:
            noise_pred_prior = 0
            noise_prior = 0

        # 现在 noise 和 noise_pred 的形状完全相同，可以计算 MSE loss
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
    # def train_diffusion_forward(self, batch):
    #     feats_ref = batch["motion"]
    #     feats_content = batch["motion"].clone()
    #     feats_content[...,:3] = 0.0
    #     lengths = batch["length"]
        
    #     # content condition
    #     with torch.no_grad():
    #         z, dist = self.vae.encode(feats_ref, lengths)
    #         z_content, dist = self.vae.encode(feats_content, lengths)
    #         cond_emb = z_content.permute(1,0,2)            
    #     # style condition
    #     motion_seq = feats_ref*self.std + self.mean
    #     motion_seq[...,:3]=0.0
    #     motion_seq = motion_seq.unsqueeze(-1).permute(0,2,3,1)
    #     motion_emb = self.motionclip.encoder({'x': motion_seq,
    #                     'y': torch.zeros(motion_seq.shape[0], dtype=int, device='cuda:{}'.format(self.cfg["DEVICE"][0])),
    #                     'mask': lengths_to_mask(lengths, device='cuda:{}'.format(self.cfg["DEVICE"][0]))})["mu"]
    #     motion_emb = motion_emb.unsqueeze(1)
    #     mask_uncond = torch.rand(motion_emb.shape[0]) < self.guidance_uncodp
    #     motion_emb[mask_uncond, ...] = 0
        


    #     # trans condition
    #     trans_cond = batch["motion"][...,:3]

    #     # three condition
    #     multi_cond_emb = [cond_emb, motion_emb, trans_cond]


    #     # diffusion process return with noise and noise_pred
    #     n_set = self._diffusion_process(z, multi_cond_emb, lengths)
    #     return {**n_set}

    def train_diffusion_forward(self, batch):
        feats_ref = batch["motion"] # Shape: (B, L, D)
        feats_content = batch["motion"].clone()
        feats_content[...,:3] = 0.0
        lengths = batch["length"]
        texts = batch["text"] # [新增] 获取文本

        # --- [最终修复] 在进入 VAE 前进行维度转置 ---
        feats_ref_for_vae = feats_ref.permute(0, 2, 1) # (B, L, D) -> (B, D, L)
        feats_content_for_vae = feats_content.permute(0, 2, 1)

        # 1. [保留] 原始的内容条件获取逻辑
        with torch.no_grad():
            z, dist = self.vae.encode(feats_ref_for_vae, lengths)
            z_content, dist = self.vae.encode(feats_content_for_vae, lengths)
            cond_emb = z_content.permute(1,0,2)            

        # 2. [保留] 原始的 MotionCLIP 风格条件获取逻辑
        # 即使我们主要用文本，也保留它，以防未来需要双模态训练
        motion_seq = feats_ref*self.std + self.mean
        motion_seq[...,:3]=0.0
        motion_seq = motion_seq.unsqueeze(-1).permute(0,2,3,1)
        motion_emb = self.motionclip.encoder({'x': motion_seq,
                        'y': torch.zeros(motion_seq.shape[0], dtype=int, device=self.device),
                        'mask': lengths_to_mask(lengths, device=self.device)})["mu"]
        motion_emb = motion_emb.unsqueeze(1)
        mask_uncond = torch.rand(motion_emb.shape[0]) < self.guidance_uncodp
        motion_emb[mask_uncond, ...] = 0

        # 3. [保留] 原始的轨迹条件获取逻辑
        trans_cond = batch["motion"][...,:3]

        # 4. [保留] 原始的多条件组合
        multi_cond_emb = [cond_emb, motion_emb, trans_cond]

        # 5. [新增] 获取我们的新文本条件
        with torch.no_grad():
            tokens = clip.tokenize(texts, truncate=True).to(self.device)
            text_features = self.clip_model.encode_text(tokens).float() # Shape: (B, D_clip)

        # 6. [核心] 调用 _diffusion_process，传入所有原始条件，并“附加”我们的文本条件
        n_set = self._diffusion_process(z, multi_cond_emb, lengths, style_text_feature=text_features)
        
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

        if self.trainer.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            style_texts = style_texts * self.cfg.TEST.MM_NUM_REPEATS
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            motion = motion.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                  dim=0)
            text_lengths = text_lengths.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
          
            # style 
            motion_seq = motion*self.std + self.mean
            motion_seq[...,:3]=0.0
            motion_seq = motion_seq.unsqueeze(-1).permute(0,2,3,1)

            motion_emb = self.motionclip.encoder({'x': motion_seq,
                          'y': torch.zeros(motion_seq.shape[0], dtype=int, device='cuda:{}'.format(self.cfg["DEVICE"][0])),
                          'mask': lengths_to_mask(lengths, device='cuda:{}'.format(self.cfg["DEVICE"][0]))})["mu"]
            motion_emb = motion_emb.unsqueeze(1)
            # uncond set feature = 0
            uncond_motion_emb = torch.zeros(motion_emb.shape).to('cuda:{}'.format(self.cfg["DEVICE"][0]))
            motion_emb = torch.cat([uncond_motion_emb, motion_emb], dim=0)

            # content condition
            with torch.no_grad():
                z, dist_m = self.vae.encode(content_motions, lengths)
            uncond_tokens = torch.cat([z, z], dim = 1).permute(1,0,2)
            motion_emb_content = uncond_tokens

            # trans
            trans_cond = batch["motion"][...,:3]
            uncond_trans = torch.cat([trans_cond, trans_cond], dim = 0)

            multi_cond_emb = [motion_emb_content, motion_emb, uncond_trans]
            z = self._diffusion_reverse(multi_cond_emb, lengths,scale=self.guidance_scale)
        elif self.stage in ['vae']:
            z, dist_m = self.vae.encode(motions, lengths)

        with torch.no_grad():
            feats_rst = self.vae.decode(z, lengths)


        # end time
        end = time.time()
        self.times.append(end - start)

        # joints recover
        joints_rst = self.feats2joints(feats_rst)
        joints_ref = self.feats2joints(motions)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        motions = self.datamodule.renorm4t2m(motions)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot,
                                        text_lengths)[align_idx]

        rs_set = {
            "m_ref": motions,
            "m_rst": feats_rst,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "lat_rm": recons_emb,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
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

    def allsplit_step(self, split: str, batch, batch_idx):
        # [核心修复]
        # 无论是训练还是验证，我们都执行相同的核心去噪任务。
        # 推理评估 (t2m_eval) 只在 on_validation_epoch_end 或测试时进行。
        if split in ["train", "val"]:
            
            # --- 复用 train_diffusion_forward 的逻辑 ---
            # 1. 准备 VAE 目标和内容条件
            feats_ref = batch["motion"]
            lengths = batch["length"]
            with torch.no_grad():
                z, _ = self.vae.encode(feats_ref, lengths)
                feats_content = feats_ref.clone(); feats_content[..., :3] = 0.0
                z_content, _ = self.vae.encode(feats_content, lengths)
                cond_emb = z_content # Denoiser 内部会处理 permute

            # 2. 准备文本风格条件
            #    对于验证集 (来自 HumanML3D)，文本就是内容描述
            texts = batch["text"]
            with torch.no_grad():
                tokenized_text = clip.tokenize(texts, truncate=True).to(self.device)
                text_features = self.clip_model.encode_text(tokenized_text).float()
            
            # 3. 准备轨迹条件
            trans_cond = feats_ref[..., :3]
            
            # 4. 组装 Denoiser 输入
            multi_cond_emb = [cond_emb, None, trans_cond]

            # 5. 调用 diffusion 流程
            #    注意：z 是 (T, N, D)
            n_set = self._diffusion_process(
                latents=z, 
                encoder_hidden_states=multi_cond_emb, 
                lengths=lengths,
                style_text_feature=text_features
            )
            rs_set = {**n_set}
            # --- 逻辑复用结束 ---

            # 计算并记录损失 (与原始逻辑相同)
            loss = self.losses[split].update(rs_set)
            if loss is None:
                raise ValueError("Loss is None.")

        # [修改] 将 t2m_eval 从这里移出，它应该在 epoch 结束时调用
        # Compute the metrics - currently evaluate results from text to motion
        if split in ["val", "test"]:
            # 在每个 step，我们只计算 loss。评估指标在 epoch end 计算。
            # 为了让代码跑通，我们暂时注释掉这部分
            pass
            # rs_set = self.t2m_eval(batch)
            # ... (后续的 metric update) ...
        # if split in ["train", "val"]:



        #     if self.stage == "vae":
        #         rs_set = self.train_vae_forward(batch)
        #         rs_set["lat_t"] = rs_set["lat_m"]



        #     elif self.stage == "diffusion":#
        #         rs_set = self.train_diffusion_forward(batch)


        #     elif self.stage == "vae_diffusion":
        #         vae_rs_set = self.train_vae_forward(batch)
        #         diff_rs_set = self.train_diffusion_forward(batch)
        #         t2m_rs_set = self.test_diffusion_forward(batch,
        #                                                  finetune_decoder=True)
        #         # merge results
        #         rs_set = {
        #             **vae_rs_set,
        #             **diff_rs_set,
        #             "gen_m_rst": t2m_rs_set["m_rst"],
        #             "gen_joints_rst": t2m_rs_set["joints_rst"],
        #             "lat_t": t2m_rs_set["lat_t"],
        #         }
        #     else:
        #         raise ValueError(f"Not support this stage {self.stage}!")

        #     loss = self.losses[split].update(rs_set)
        #     if loss is None:
        #         raise ValueError(
        #             "Loss is None, this happend with torchmetrics > 0.7")

        # # Compute the metrics - currently evaluate results from text to motion
        # if split in ["val", "test"]:
        #     # use t2m evaluators
        #     rs_set = self.t2m_eval(batch)

        #     # MultiModality evaluation sperately
        #     if self.trainer.datamodule.is_mm:
        #         metrics_dicts = ['MMMetrics']
        #     else:
        #         metrics_dicts = self.metrics_dict
        #     # metric = 'TemosMetric' 'TM2TMetrics'
        #     for metric in metrics_dicts:
        #         if metric == "TemosMetric":
        #             phase = split if split != "val" else "eval"
        #             if eval(f"self.cfg.{phase.upper()}.DATASETS")[0].lower(
        #             ) not in [
        #                     "humanml3d",
        #                     "kit",
        #             ]:
        #                 raise TypeError(
        #                     "APE and AVE metrics only support humanml3d and kit datasets now"
        #                 )

        #             getattr(self, metric).update(rs_set["joints_rst"],
        #                                          rs_set["joints_ref"],
        #                                          batch["length"])
        #         elif metric == "TM2TMetrics":
        #             getattr(self, metric).update(
        #                 # lat_t, latent encoded from diffusion-based text
        #                 # lat_rm, latent encoded from reconstructed motion
        #                 # lat_m, latent encoded from gt motion
        #                 # rs_set['lat_t'], rs_set['lat_rm'], rs_set['lat_m'], batch["length"])
        #                 rs_set["lat_t"],
        #                 rs_set["lat_rm"],
        #                 rs_set["lat_m"],
        #                 batch["length"],
        #             )
        #         elif metric == "UncondMetrics":
        #             getattr(self, metric).update(
        #                 recmotion_embeddings=rs_set["lat_rm"],
        #                 gtmotion_embeddings=rs_set["lat_m"],
        #                 lengths=batch["length"],
        #             )
        #         elif metric == "MRMetrics":
        #             getattr(self, metric).update(rs_set["joints_rst"],
        #                                          rs_set["joints_ref"],
        #                                          batch["length"])
        #         elif metric == "MMMetrics":
        #             getattr(self, metric).update(rs_set["lat_rm"].unsqueeze(0),
        #                                          batch["length"])
        #         elif metric == "HUMANACTMetrics":
        #             getattr(self, metric).update(rs_set["m_action"],
        #                                          rs_set["joints_eval_rst"],
        #                                          rs_set["joints_eval_ref"],
        #                                          rs_set["m_lens"])
        #         elif metric == "UESTCMetrics":
        #             # the stgcn model expects rotations only
        #             getattr(self, metric).update(
        #                 rs_set["m_action"],
        #                 rs_set["m_rst"].view(*rs_set["m_rst"].shape[:-1], 6,
        #                                      25).permute(0, 3, 2, 1)[:, :-1],
        #                 rs_set["m_ref"].view(*rs_set["m_ref"].shape[:-1], 6,
        #                                      25).permute(0, 3, 2, 1)[:, :-1],
        #                 rs_set["m_lens"])
        #         else:
        #             raise TypeError(f"Not support this metric {metric}")

        # return forward output rather than loss during test
        if split in ["test"]:
            return rs_set["joints_rst"], batch["length"]
        return loss
