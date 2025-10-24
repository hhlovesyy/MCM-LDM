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
        # self.predict_epsilon = cfg.TRAIN.ABLATION.PREDICT_EPSILON
        self.nfeats = cfg.DATASET.NFEATS
        self.njoints = cfg.DATASET.NJOINTS
        self.debug = cfg.DEBUG
        self.latent_dim = cfg.model.latent_dim
        self.guidance_scale = cfg.model.guidance_scale
        self.guidance_uncodp = cfg.model.guidance_uncondp
        self.datamodule = datamodule

        self.mld_device = 'cuda:{}'.format(cfg["DEVICE"][0]) # NEW: 定义一个 device 方便后续使用


        # self.text_encoder = instantiate_from_config(cfg.model.text_encoder)

        parameters = read_yaml_to_dict("configs/motionclip_config/motionclip_params_263.yaml")
        parameters["device"] = 'cuda:{}'.format(cfg["DEVICE"][0])        
        self.motionclip = get_model_and_data(parameters, split='vald')
        print("load motion clip-xyz-263")
        print("Restore weights..")
        checkpointpath = "checkpoints/motionclip_checkpoint/motionclip.pth.tar"
        state_dict = torch.load(checkpointpath, map_location=parameters["device"])
        load_model_wo_clip(self.motionclip, state_dict)

        self.mean = torch.tensor(self.datamodule.norms['mean']).to(self.mld_device)
        self.std = torch.tensor(self.datamodule.norms['std']).to(self.mld_device)

        #don't train motionclip
        self.motionclip.training = False
        for p in self.motionclip.parameters():
            p.requires_grad = False

        # NEW: 加载并冻结 CLIP 文本编码器
        self.clip_model, _ = clip.load("ViT-B/32", device=self.mld_device)
        self.clip_model.training = False
        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.vae = instantiate_from_config(cfg.model.motion_vae)
        # Don't train the motion encoder and decoder
        if self.stage == "diffusion":
            self.vae.training = False
            for p in self.vae.parameters():
                p.requires_grad = False

        # Pass the new text_style_dim parameter which is 512 for ViT-B/32
        cfg.model.denoiser.text_style_dim = 512 # NEW:
        self.denoiser = instantiate_from_config(cfg.model.denoiser)

        self.scheduler = instantiate_from_config(cfg.model.scheduler)
        self.noise_scheduler = instantiate_from_config(
            cfg.model.noise_scheduler)


        self._get_t2m_evaluator(cfg)

        # MODIFIED: 配置差分学习率优化器
        if cfg.TRAIN.OPTIM.TYPE.lower() == "adamw":
            new_params = []
            base_params = []
            for name, param in self.denoiser.named_parameters():
                if 'cross_attn' in name or 'norm_cross_attn' in name or 'adaLN_modulation_text' in name or 'text_style_proj' in name:
                    new_params.append(param)
                else:
                    base_params.append(param)
            
            # 确保您在配置文件中定义了 LR_NEW
            lr_new = cfg.TRAIN.OPTIM.get('LR_NEW', cfg.TRAIN.OPTIM.LR * 10)

            self.optimizer = AdamW([
                {'params': base_params, 'lr': cfg.TRAIN.OPTIM.LR},
                {'params': new_params, 'lr': lr_new}
            ])
            print(f"Optimizer configured with base LR: {cfg.TRAIN.OPTIM.LR} and new layer LR: {lr_new}")
        else:
            raise NotImplementedError("Do not support other optimizer for now.")

        if cfg.LOSS.TYPE == "mld":
            self._losses = MetricCollection({
                split: MLDLosses(vae=self.is_vae, mode="xyz", cfg=cfg)
                for split in ["losses_train", "losses_test", "losses_val"]
            })
        else:
            raise NotImplementedError("MotionCross model only supports mld losses.")

        self.losses = {key: self._losses["losses_" + key] for key in ["train", "test", "val"]}
        self.metrics_dict = cfg.METRIC.TYPE
        self.configure_metrics()

        self.sample_mean = False
        self.fact = None
        self.do_classifier_free_guidance = True
        self.feats2joints = datamodule.feats2joints
        self.joints2feats = datamodule.joints2feats
    
    # # NEW: Implement configure_optimizers for fine-grained LR control
    # def configure_optimizers(self):
    #     # Separate parameters into two groups
    #     newly_added_params = []
    #     original_params = []

    #     for name, param in self.denoiser.named_parameters():
    #         if param.requires_grad:
    #             if 'cross_attn' in name or 'norm_cross_attn' in name or 'adaLN_modulation_text' in name or 'text_style_proj' in name:
    #                 newly_added_params.append(param)
    #             else:
    #                 original_params.append(param)

    #     # Create parameter groups for the optimizer
    #     param_groups = [
    #         {'params': original_params, 'lr': self.cfg.TRAIN.OPTIM.LR}, # Original params with base LR
    #         {'params': newly_added_params, 'lr': self.cfg.TRAIN.OPTIM.LR_NEW} # New params with a higher LR
    #     ]

    #     if self.cfg.TRAIN.OPTIM.TYPE.lower() == "adamw":
    #         optimizer = AdamW(param_groups)
    #     else:
    #         raise NotImplementedError("Only AdamW optimizer is supported for differential learning rates.")
        
    #     return optimizer

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
    def forward(self, batch):

        lengths = batch["length"]
        bs = len(lengths) # NEW: Get batch size for later use
        # style
        # motion = batch["style_motion"].clone()
        # motion[...,:3] = 0


        # content
        content_motion = batch['content_motion']
        content_motion = (content_motion - self.mean.to(content_motion.device))/self.std.to(content_motion.device)

        # trajectory
        trans_motion = content_motion.clone()
        # 
        content_motion[...,:3] = 0


        scale = batch.get("tag_scale", self.guidance_scale) # 如果 batch 中没提供，则使用默认值
        lengths1 = [content_motion.shape[1]]* content_motion.shape[0]
        
        if self.cfg.TEST.COUNT_TIME:
            self.starttime = time.time()
            
        if self.stage in ['diffusion', 'vae_diffusion']:\
            #add style text in test
            # NEW: 初始化 style_text_feature 为 None
            style_text_feature = None
            
            # content motion
            with torch.no_grad():
                z, dist_m = self.vae.encode(content_motion.float(), lengths1)
            uncond_tokens = torch.cat([z, z], dim = 1).permute(1,0,2)
            motion_emb_content = uncond_tokens

            # trajectory
            trans_cond = trans_motion[...,:3]
            uncond_trans = torch.cat([trans_cond, trans_cond], dim = 0)

            # --- 3. [核心修复] 智能判断模式并构建风格条件 ---
        
            # 模式一: 文本引导
            if "style_text" in batch and batch["style_text"] is not None:
                texts = batch["style_text"]
                with torch.no_grad():
                    text_tokens = clip.tokenize(texts).to(self.mld_device)
                    text_features = self.clip_model.encode_text(text_tokens).float()
                
                uncond_text_features = torch.zeros_like(text_features)
                style_text_feature = torch.cat([uncond_text_features, text_features], dim=0).unsqueeze(1)
                
                # 创建一个全零的 motion_emb 占位符
                # motion_emb = torch.zeros(bs * 2, 1, self.motionclip.motion_encoder.output_size).to(self.mld_device)
                motion_emb_dim = 512
                motion_emb = torch.zeros(bs * 2, 1, motion_emb_dim).to(self.mld_device)
                text_style_cond = style_text_feature
            
            # 模式二: 动作引导
            elif "style_motion" in batch:
                style_motion = batch["style_motion"].clone()
                style_motion[..., :3] = 0
                
                # [新代码 - 关键] 动作风格也需要归一化！
                # style_motion = (style_motion - self.mean.to(style_motion.device)) / self.std.to(style_motion.device)
                
                style_lengths = [style_motion.shape[1]] * bs
                motion_seq = style_motion.unsqueeze(-1).permute(0, 2, 3, 1)

                motion_emb_features = self.motionclip.encoder({
                    'x': motion_seq.float(),
                    'y': torch.zeros(bs, dtype=int, device=motion_seq.device),
                    'mask': lengths_to_mask(style_lengths, device=motion_seq.device)
                })["mu"]

                # [修改] 正确构建 CFG 条件: [uncond, cond]
                uncond_motion_emb = torch.zeros_like(motion_emb_features)
                motion_emb = torch.cat([uncond_motion_emb, motion_emb_features], dim=0).unsqueeze(1)

                # [修改] 创建一个全零的 text_style_cond 占位符
                text_style_cond = torch.zeros(bs * 2, 1, 512, device=self.mld_device)
            
            else:
                raise ValueError("Inference batch must contain either 'style_text' or 'style_motion'")

            # three conditions
            multi_cond_emb = [motion_emb_content, motion_emb, uncond_trans]


            z = self._diffusion_reverse(multi_cond_emb, lengths, scale, style_text_feature=text_style_cond)

        elif self.stage in ['vae']:
            motions = batch['motion']
            z, dist_m = self.vae.encode(motions, lengths)

        with torch.no_grad():
            feats_rst = self.vae.decode(z, lengths)
            # feats_rst[...,:3] = trans_motion[...,:3] # if copy trajectory
            if trans_motion is not None:
                feats_rst[..., :3] = trans_motion[..., :3]

        # [新代码 - 关键] 反归一化，得到正确尺度的动作数据
        # feats_rst = feats_rst * self.std.to(feats_rst.device) + self.mean.to(feats_rst.device)
        # 这里其实是要反归一化的，feats2joints里面有做，理论上违反了单一职责的原则，但为了不破坏原有代码结构，只能这样做了
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
            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths_reverse,
                style_text_feature=style_text_feature # <-- 唯一的改动
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
        # 直观地说，函数形参地latents是torch.Size([7, 32, 256])，32是batch_size
        # 你也解释一下这里的32和7是什么，n_token和latent_dim分别代表什么，这个7应该是token，256应该是latent_dim，整个latents的含义似乎是batch里面的原始动作经过vae编码后的latent表示
        # [n_token, batch_size, latent_dim] -> [batch_size, n_token, latent_dim]
        latents = latents.permute(1, 0, 2)  # torch.Size([32, 7, 256])

        # Sample noise that we'll add to the latents
        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(latents)  # torch.Size([32, 7, 256])
        bsz = latents.shape[0]  # 32
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=latents.device,
        )  # shape:torch.Size([32])
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents.clone(), noise,
                                                       timesteps)  # torch.Size([32, 7, 256])
        # Predict the noise residual
        noise_pred = self.denoiser(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            lengths=lengths,
            style_text_feature=style_text_feature, # <-- 唯一的改动
            return_dict=False,
        )[0]  # torch.Size([32, 7, 256])
        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
        if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
            noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
            noise, noise_prior = torch.chunk(noise, 2, dim=0)
        else:  # 默认应该是走到这个逻辑里面
            noise_pred_prior = 0
            noise_prior = 0


        n_set = {
            "noise": noise,  # torch.Size([32, 7, 256])
            "noise_prior": noise_prior,  # 0
            "noise_pred": noise_pred,  # torch.Size([32, 7, 256])
            "noise_pred_prior": noise_pred_prior, # 0
        }

        # [新代码] 如果是训练模式，额外传递 is_text_guided_mask
        if self.training and style_text_feature is not None:
            is_text_guided_mask = torch.any(style_text_feature.squeeze(1) != 0, dim=-1)
            n_set["is_text_guided_mask"] = is_text_guided_mask

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
        feats_ref = batch["motion"]  # shape:torch.Size([32, 28, 263]): 应该是[batchsize, motion_len, nfeats]
        feats_content = batch["motion"].clone()  # shape:torch.Size([32, 28, 263]):
        feats_content[...,:3] = 0.0  # shape:torch.Size([32, 28, 263]):
        lengths = batch["length"]  # len(): 32
        
        # content condition
        with torch.no_grad():
            z, dist = self.vae.encode(feats_ref, lengths)  # z: torch.Size([7, 32, 256]), dist(batch_shape): torch.Size([7, 32, 256])
            z_content, dist = self.vae.encode(feats_content, lengths) # z_content: torch.Size([7, 32, 256]), dist:batch_shape:torch.Size([7, 32, 256])
        cond_emb = z_content.permute(1,0,2)  # torch.Size([32, 7, 256])    

        # trans condition
        trans_cond = batch["motion"][...,:3]  # torch.Size([32, 28, 3]) 这里面的3推测应该是全局的xyz属性，表示轨迹       
        
        # --- [核心修复] 实现基于 is_text_guided 的精确条件开关 ---

        # 1. 从 batch 中解包出我们的关键标志
        is_text_guided = batch['is_text_guided']  # (bs,) e.g. [F, F, T, T, F, ...]

        # 2. 初始化两个风格条件张量，全部为零
        # 动作风格 (fs) 的 embedding 维度是 512
        motion_style_cond = torch.zeros(feats_ref.shape[0], 1, 512, device=self.mld_device, dtype=torch.float)
        # 文本风格 (text_style) 的 embedding 维度也是 512 (来自 CLIP ViT-B/32)
        text_style_cond = torch.zeros(feats_ref.shape[0], 1, 512, device=self.mld_device, dtype=torch.float)

        # 3. 根据标志，为每个样本填充正确的风格条件

        # 找到所有需要“动作引导”的样本的索引
        motion_indices = ~is_text_guided
        if motion_indices.any():
            # a. 只为这些样本提取动作风格
            motion_seq = feats_ref[motion_indices] * self.std + self.mean
            motion_seq[..., :3] = 0.0
            motion_seq = motion_seq.unsqueeze(-1).permute(0, 2, 3, 1)
            
            lengths_motion = [lengths[i] for i, flag in enumerate(motion_indices) if flag]

            motion_emb = self.motionclip.encoder({
                'x': motion_seq,
                'y': torch.zeros(motion_seq.shape[0], dtype=int, device=self.mld_device),
                'mask': lengths_to_mask(lengths_motion, device=self.mld_device)
            })["mu"].unsqueeze(1) # (N_motion, 1, 512)
            
            # b. 应用 CFG dropout
            mask_uncond = torch.rand(motion_emb.shape[0], device=self.mld_device) < self.guidance_uncodp
            motion_emb[mask_uncond, ...] = 0
            
            # c. 将计算出的 embedding 填充回主张量的对应位置
            motion_style_cond[motion_indices] = motion_emb

        # 找到所有需要“文本引导”的样本的索引
        text_indices = is_text_guided
        if text_indices.any():
            # a. 只为这些样本提取文本
            texts = [batch['text'][i] for i, flag in enumerate(text_indices) if flag]
            
            # b. 编码文本
            with torch.no_grad():
                text_tokens = clip.tokenize(texts).to(self.mld_device)
                text_features = self.clip_model.encode_text(text_tokens).float().unsqueeze(1) # (N_text, 1, 512)

            # c. 应用 CFG dropout
            mask_uncond = torch.rand(text_features.shape[0], device=self.mld_device) < self.guidance_uncodp
            text_features[mask_uncond, ...] = 0

            # d. 将计算出的 embedding 填充回主张量的对应位置
            text_style_cond[text_indices] = text_features
            
        # 4. [核心] 打包所有条件
        multi_cond_emb = [cond_emb, motion_style_cond, trans_cond]
        # 注意：我们将 text_style_cond 作为一个独立的参数传入，这与你之前的设计保持一致
        n_set = self._diffusion_process(z, multi_cond_emb, lengths, style_text_feature=text_style_cond)

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
        # --- 第一部分：计算损失 ---
        loss_val = None
        if split in ["train", "val"]:
            if self.stage == "diffusion":
                rs_set = None 
                
                if split == "train":
                    # [训练逻辑] - 调用我们修复好的、健壮的双模态函数
                    rs_set = self.train_diffusion_forward(batch)
                
                elif split == "val":
                    # [验证逻辑] - [最终修复]
                    # 为了 100% 保证验证流程的维度正确性，我们在这里
                    # 只调用原始的、单模态的训练逻辑。
                    feats_ref = batch["motion"]
                    feats_content = batch["motion"].clone()
                    feats_content[...,:3] = 0.0
                    lengths = batch["length"]
                    
                    with torch.no_grad():
                        z, _ = self.vae.encode(feats_ref, lengths)
                        z_content, _ = self.vae.encode(feats_content, lengths)
                        # VAE 输出 (T, bs, D), Denoiser 需要 (bs, T, D) 作为 list 元素
                        cond_emb = z_content.permute(1,0,2)

                    motion_seq = feats_ref*self.std + self.mean
                    motion_seq[...,:3]=0.0
                    motion_seq = motion_seq.unsqueeze(-1).permute(0,2,3,1)
                    
                    motion_emb = self.motionclip.encoder({'x': motion_seq,
                                    'y': torch.zeros(motion_seq.shape[0], dtype=int, device=self.mld_device),
                                    'mask': lengths_to_mask(lengths, device=self.mld_device)})["mu"].unsqueeze(1)
                    
                    # 在验证时，我们不应用 CFG dropout
                    # mask_uncond = torch.rand(motion_emb.shape[0]) < self.guidance_uncodp
                    # motion_emb[mask_uncond, ...] = 0

                    trans_cond = batch["motion"][...,:3]
                    
                    multi_cond_emb = [cond_emb, motion_emb, trans_cond]

                    # 在验证时，我们不传递任何文本特征
                    rs_set = self._diffusion_process(z, multi_cond_emb, lengths, style_text_feature=None)

            # ... (后续的 vae 和 loss 计算逻辑保持不变)
            elif self.stage == "vae":
                 rs_set = self.train_vae_forward(batch)
                 rs_set["lat_t"] = rs_set["lat_m"]
            else:
                raise ValueError(f"Unsupported stage for loss calculation: {self.stage}!")

            # --- [最终日志修复] ---
            # 计算损失，并用 loss_val 变量接收
            loss_val = self.losses[split].update(rs_set)
            if loss_val is None:
                raise ValueError("Loss is None from torchmetrics.")

            # [核心修复] 智能地记录日志，无论返回值是字典还是张量
            if isinstance(loss_val, dict):
                # 如果是字典 (通常在 train 时)，遍历并记录所有键
                for key, value in loss_val.items():
                    self.log(f"{split}/{key}", value, prog_bar=(key == 'total'), on_step=(split == 'train'), on_epoch=True)
            elif isinstance(loss_val, torch.Tensor):
                # 如果是单个张量 (通常在 val 时)，只记录总损失
                self.log(f"{split}/loss", loss_val, prog_bar=True, on_step=(split == 'train'), on_epoch=True)
            # --- 修复结束 ---

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
        # return loss
        # 返回总损失给优化器
        if isinstance(loss_val, dict):
            return loss_val['total']
        
        # 如果 loss_val 是张量，直接返回它
        return loss_val