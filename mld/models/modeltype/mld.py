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

import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType

class TextAdapter(nn.Module):
    def __init__(self, input_dim=512, output_dim=512, hidden_dim=512, num_layers=3):
        super().__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.GELU()
        ]
        # 允许构建更深的 MLP
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


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

        self.mld_device = 'cuda:{}'.format(cfg["DEVICE"][0]) 

        # self.text_encoder = instantiate_from_config(cfg.model.text_encoder)

        parameters = read_yaml_to_dict("configs/motionclip_config/motionclip_params_263.yaml")
        parameters["device"] = 'cuda:{}'.format(cfg["DEVICE"][0])        
        self.motionclip = get_model_and_data(parameters, split='vald') # 这个现在是”学生“
        print("load motion clip-xyz-263")
        print("Restore weights..")
        checkpointpath = "checkpoints/motionclip_checkpoint/motionclip.pth.tar"
        state_dict = torch.load(checkpointpath, map_location=parameters["device"])
        load_model_wo_clip(self.motionclip, state_dict)

        self.mean = torch.tensor(self.datamodule.norms['mean']).to(self.mld_device)
        self.std = torch.tensor(self.datamodule.norms['std']).to(self.mld_device)

        self.motionclip.train()  # 这个是学生网络
        for p in self.motionclip.parameters():
            p.requires_grad = True
        print("INFO: [C-Plan] Trainable 'motionclip' (student) initialized.")

        # 2025.10.28 创建一个teacher motionclip，冻结
        self.motionclip_teacher = get_model_and_data(parameters, split='vald')
        state_dict = torch.load(checkpointpath, map_location=parameters["device"])
        load_model_wo_clip(self.motionclip_teacher, state_dict)
        self.motionclip_teacher.eval() # 设置为评估模式
        for p in self.motionclip_teacher.parameters():
            p.requires_grad = False
        print("INFO: [C-Plan] Frozen 'motionclip_teacher' initialized.")


        # # NOTE: 加载并冻结 CLIP 文本编码器，目前我们冻结CLIP，后面可以考虑解冻，因为CLIP是文本图像域对齐做的，而我们是”动作“的风格描述，CLIP模型可能理解里本身不够
        # self.clip_model, _ = clip.load("ViT-B/32", device=self.mld_device)
        # self.clip_model.eval()
        # for name, p in self.clip_model.named_parameters():
        #     p.requires_grad = False

        # 接LoRA
        # ===> START: 替换代码 <===
        self.clip_model, _ = clip.load("ViT-B/32", device=self.mld_device)
        # # b. [临时诊断代码] 打印出模型的结构，然后退出
        # print("\n--- CLIP Model Structure ---")
        # print(self.clip_model)
        # print("----------------------------\n")
        # print("[DEBUG] Exiting after printing model structure.")
        # exit()

        # [核心修改] 使用 PEFT (LoRA) 来微调 CLIP 的文本编码器

        # a. 首先，冻结 CLIP 的所有参数。这是 LoRA 的前提。
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # b. 定义 LoRA 配置
        #    TaskType.FEATURE_EXTRACTION 表示我们只用它来提取特征
        #    target_modules 指定了我们要在哪种类型的层上应用 LoRA。
        #    对于 CLIP 的 Transformer，q_proj 和 v_proj 是最常见的选择。
        lora_config = LoraConfig(
            r=16, # LoRA 的秩，r 越大，可训练参数越多，能力越强，但越容易过拟合。8 或 16 是常用值。
            lora_alpha=16, # LoRA 的缩放因子
            # target_modules=["q_proj", "v_proj"], # 只在 attention 的 q 和 v 投影层上加 LoRA
            # [核心修复] 将目标模块指向 MLP 内部的线性层,因为attention模块一般是黑箱，peft检索不到，而MLP模块是明确的
            target_modules=["c_fc", "c_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        # c. 将 LoRA 配置应用到 CLIP 模型上
        #    get_peft_model 会自动找到所有 target_modules 并为它们加上 LoRA 层
        self.clip_model = get_peft_model(self.clip_model, lora_config)

        # d. 打印出可训练的参数，以确认 LoRA 是否成功应用
        print("\n--- LoRA Applied to CLIP Model ---")
        self.clip_model.print_trainable_parameters()
        print("---------------------------------\n")

        # 我们依然需要将模型设置为 eval() 模式，以关闭 Dropout 等层。
        # LoRA 的训练不受 model.eval() 的影响。
        self.clip_model.eval()
        # ===> END: 替换代码 <===

        # 验证一下 ln_final 是否被冻结
        print(f"CLIP's ln_final.weight.requires_grad: {self.clip_model.ln_final.weight.requires_grad}") # 应该输出 False

        self.vae = instantiate_from_config(cfg.model.motion_vae)
        # Don't train the motion encoder and decoder
        if self.stage == "diffusion":
            self.vae.training = False
            for p in self.vae.parameters():
                p.requires_grad = False

        # Pass the new text_style_dim parameter which is 512 for ViT-B/32
        cfg.model.denoiser.text_style_dim = 512 
        self.denoiser = instantiate_from_config(cfg.model.denoiser)

        self.scheduler = instantiate_from_config(cfg.model.scheduler)
        self.noise_scheduler = instantiate_from_config(
            cfg.model.noise_scheduler)


        self._get_t2m_evaluator(cfg)

        # NOTE: 注意看一下能否和losses那里的对上，逻辑是否正确，可以结合后面的一起关注
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

        self.text_adapter = TextAdapter(input_dim=512, output_dim=512, num_layers=3).to(self.mld_device)
        # [最终解决方案] 添加两个 LayerNorm 层，用于强制对齐尺度
        self.text_emb_norm = nn.LayerNorm(512).to(self.mld_device)
        self.motion_emb_norm = nn.LayerNorm(512).to(self.mld_device)
        print("INFO: TextAdapter (MLP Mapping Network) has been added.")
    

    def configure_optimizers(self):  # overriding pl.LightningModule method
        if self.cfg.TRAIN.OPTIM.TYPE.lower() == "adamw":
            print("Configuring differential learning rates in `configure_optimizers`...")
            denoiser_base_params = []
            denoiser_new_params = []
            
            for name, param in self.denoiser.named_parameters():
                if param.requires_grad:
                    if 'adaLN_modulation_text' in name or 'text_style_proj' in name:
                        denoiser_new_params.append(param)
                    else:
                        denoiser_base_params.append(param)
            
            lr_base = self.cfg.TRAIN.OPTIM.LR
            lr_new = self.cfg.TRAIN.OPTIM.get('LR_NEW', lr_base * 5)
            lr_finetune = self.cfg.TRAIN.OPTIM.get('LR_FINETUNE', lr_base * 0.1)
            lr_finetune_clip = lr_base * 0.01
            lr_finetune_motionclip = lr_base * 0.1
            # peft 库在将 LoRA 应用到 clip_model 后，会自动将 CLIP 的原始参数的 requires_grad 属性设置为 False，而只将新增的 LoRA 参数的 requires_grad 保持为 True。
            clip_trainable_params = list(filter(lambda p: p.requires_grad, self.clip_model.parameters()))
            param_groups = [
                {'params': denoiser_base_params, 'lr': lr_base, 'name': 'denoiser_base'},
                {'params': denoiser_new_params, 'lr': lr_new, 'name': 'denoiser_new'},
                {'params': self.text_adapter.parameters(), 'lr': lr_new, 'name': 'text_adapter'},
                {'params': clip_trainable_params, 'lr': lr_finetune_clip, 'name': 'clip_lora'},
                # {'params': filter(lambda p: p.requires_grad, self.clip_model.parameters()), 'lr': lr_finetune_clip, 'name': 'clip_finetune'},
                {'params': filter(lambda p: p.requires_grad, self.motionclip.parameters()), 'lr': lr_finetune_motionclip, 'name': 'motionclip_finetune'}
            ]

            optimizer = AdamW(param_groups)

            for group in optimizer.param_groups:
                print(f"  - Group '{group['name']}': {len(group['params'])} params, LR: {group['lr']}")
            
            return optimizer
        else:
            raise NotImplementedError("Do not support other optimizer for now.")

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
        """
        This is the inference function.
        It should perfectly mirror the logic of train_diffusion_forward,
        but with Classifier-Free Guidance (CFG) explicitly constructed.
        (Final Corrected Version v2)
        """
        # --- 1. 准备 Content 和 Trajectory ---
        content_motion = batch['content_motion']
        content_motion_normalized = (content_motion - self.mean.to(content_motion.device)) / self.std.to(content_motion.device)

        trans_motion_unnormalized = content_motion_normalized.clone()
        content_motion_normalized[..., :3] = 0.0

        # [修改] 我们将 content_lengths 和 style_lengths 分开处理
        content_lengths = batch["length"]
        bs = len(content_lengths)

        scale = batch.get("tag_scale", self.guidance_scale)

        with torch.no_grad():
            z_content, _ = self.vae.encode(content_motion_normalized.float(), content_lengths)
            cond_emb = torch.cat([z_content, z_content], dim=1).permute(1, 0, 2)

        trans_cond = trans_motion_unnormalized[..., :3]
        uncond_trans = torch.cat([trans_cond, trans_cond], dim=0)

        # --- 2. [核心修改] 构建与训练时完全一致的风格条件 ---
        is_text_mode = "style_text" in batch and batch["style_text"] is not None
        is_motion_mode = "style_motion" in batch

        if not is_text_mode and not is_motion_mode:
            raise ValueError("Inference batch must contain either 'style_text' or 'style_motion'")

        motion_style_cond = torch.zeros(bs * 2, 1, 512, device=self.mld_device, dtype=torch.float)
        text_style_cond = torch.zeros(bs * 2, 1, 512, device=self.mld_device, dtype=torch.float)

        if is_text_mode:
            texts = batch["style_text"]
            with torch.no_grad():
                text_tokens = clip.tokenize(texts).to(self.mld_device)
                with torch.cuda.amp.autocast(enabled=False):
                    raw_clip_features = self.clip_model.encode_text(text_tokens).float()
                adapted_features = self.text_adapter(raw_clip_features)
                normalized_text_emb = self.text_emb_norm(adapted_features)
                text_features = normalized_text_emb
                
            uncond_text_features = torch.zeros_like(text_features)
            text_style_cond = torch.cat([uncond_text_features, text_features], dim=0).unsqueeze(1)

        elif is_motion_mode:
            style_motion = batch["style_motion"]
            
            # 1. 为 style_motion 单独计算其长度
            style_lengths = [style_motion.shape[1]] * style_motion.shape[0]

            # 2. MotionClip 期望接收未经归一化的数据
            style_motion_raw = style_motion.clone()
            style_motion_raw[..., :3] = 0.0
            
            motion_seq = style_motion_raw.unsqueeze(-1).permute(0, 2, 3, 1)

            with torch.no_grad():
                    motion_emb_features = self.motionclip.encoder({
                    'x': motion_seq.float(),
                    'y': torch.zeros(bs, dtype=int, device=self.mld_device),
                    'mask': lengths_to_mask(style_lengths, device=self.mld_device)
                })["mu"]

            uncond_motion_emb = torch.zeros_like(motion_emb_features)
            motion_style_cond = torch.cat([uncond_motion_emb, motion_emb_features], dim=0).unsqueeze(1)

        multi_cond_emb = [cond_emb, motion_style_cond, uncond_trans]

        z = self._diffusion_reverse(
            encoder_hidden_states=multi_cond_emb, 
            lengths=content_lengths, 
            scale=scale, 
            style_text_feature=text_style_cond
        )

        # --- 4. VAE 解码并返回结果 ---
        with torch.no_grad():
            feats_rst = self.vae.decode(z, content_lengths) # 解码的长度也以 content 为准
            if trans_motion_unnormalized is not None:
                feats_rst[..., :3] = trans_motion_unnormalized[..., :3]

        joints = self.feats2joints(feats_rst.detach().cpu())

        return remove_padding(joints, content_lengths)


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
        # 这个7应该是token，256应该是latent_dim，整个latents的含义似乎是batch里面的原始动作经过vae编码后的latent表示
        # [n_token, batch_size, latent_dim] -> [batch_size, n_token, latent_dim]
        latents = latents.permute(1, 0, 2)  # torch.Size([32, 7, 256])

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
            style_text_feature=style_text_feature, 
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

        # 新增了style reconstruction相关的loss计算
        if self.training and self.cfg.LOSS.get('LAMBDA_STYLE_RECON', 0.0) > 0.0:
            print("Computing style reconstruction loss components...C1 Plan!")
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
        feats_ref = batch["motion"]  # shape:torch.Size([batch_size, motion_len, 263]): 应该是[batchsize, motion_len, nfeats]
        feats_content = batch["motion"].clone()  # shape:torch.Size([batch_size, motion_len, 263]):
        feats_content[...,:3] = 0.0  # shape:torch.Size([batch_size, motion_len, 263]):  feats_content会干掉轨迹
        lengths = batch["length"]  # len(): batch_size，一个list，里面是各个样本的长度，比如192，88，60等
        bs = feats_ref.shape[0]

        # 1. 从 batch 中解包出我们的关键标志
        is_text_guided = batch['is_text_guided']  # torch.Size([batch_size])，前面全是false，后面全是true，毕竟是直接拼接的
        # print(f"Batch Size: {bs}, Text-Guided Samples: {is_text_guided.sum().item()}")

        # content condition
        with torch.no_grad():
            z, dist = self.vae.encode(feats_ref, lengths)  # z: torch.Size([7, batch_size, 256]), dist(batch_shape): torch.Size([7, batch_size, 256])
            z_content, dist = self.vae.encode(feats_content, lengths) # z_content: torch.Size([7, batch_size, 256]), dist:batch_shape:torch.Size([7, batch_size, 256])
        
        cond_emb = z_content.permute(1,0,2)  # torch.Size([batch_size, 7, 256])    
        
        # trans condition
        trans_cond = batch["motion"][...,:3]  # torch.Size([batch_size, motion_len, 3]) 这里面的3推测应该是全局的xyz属性，表示轨迹       
        
        # --- [核心修复] 实现基于 is_text_guided 的精确条件开关 ---

        # 2. 初始化两个风格条件张量，全部为零
        # 动作风格 (fs) 的 embedding 维度是 512
        motion_style_cond = torch.zeros(feats_ref.shape[0], 1, 512, device=self.mld_device, dtype=torch.float) # torch.Size([batch_size, 1, 512])，里面全是0
        # 文本风格 (text_style) 的 embedding 维度也是 512 (来自 CLIP ViT-B/32)
        text_style_cond = torch.zeros(feats_ref.shape[0], 1, 512, device=self.mld_device, dtype=torch.float)  # torch.Size([batch_size, 1, 512])，里面全是0

        # 3. 根据标志，为每个样本填充正确的风格条件
        # 初始化用于 align_loss 的张量
        text_emb_for_align = None
        motion_emb_for_align = None
        # 找到所有需要“动作引导”的样本的索引
        motion_indices = ~is_text_guided # torch.Size([batch_size]),前一半都是True，后一半都是False
        if motion_indices.any(): # 这样的话肯定能进了
            # a. 只为这些样本提取动作风格
            motion_seq = feats_ref[motion_indices] * self.std + self.mean  # torch.Size([batch_size / 2, motion_len, 263])，motion_length对于一个batch来说应该填充到一样长了
            motion_seq[..., :3] = 0.0
            motion_seq = motion_seq.unsqueeze(-1).permute(0, 2, 3, 1) # torch.Size([batch_size / 2, 263, 1, motion_len])
            # 【QUESTION】这里有问题了！我发现motion_seq里面基本都是1和0，我需要把这个dump到一个文件里看看，一会你帮我补充一下代码
            lengths_motion = [lengths[i] for i, flag in enumerate(motion_indices) if flag]  # len(): batch_size / 2

            motion_emb = self.motionclip.encoder({ # 【QUESTION】这里使用学生网络，可以学习是不是？
                'x': motion_seq,
                'y': torch.zeros(motion_seq.shape[0], dtype=int, device=self.mld_device),
                'mask': lengths_to_mask(lengths_motion, device=self.mld_device)
            })["mu"].unsqueeze(1) # torch.Size([batch_size / 2, 1, 512])
            
            # b. 应用 CFG dropout
            mask_uncond = torch.rand(motion_emb.shape[0], device=self.mld_device) < self.guidance_uncodp  # torch.Size([batch_size/2])，【QUESTION】几乎都是False，True非常少，应该没什么问题？
            motion_emb[mask_uncond, ...] = 0  # 【QUESTION】这种写法是说mask_uncond是True的部分会变成0，其他部分不变么
            
            # c. 将计算出的 embedding 填充回主张量的对应位置，这个逻辑对么？
            motion_style_cond[motion_indices] = motion_emb  # torch.Size([batch_size, 1, 512])
            # 【QUESTION】debug的时候我发现motion_style_cond里面有两个字段，一个是T，一个是data，其中T看起来每个样本后一半都是0，但是data比较正常一些？前一半样本有值，后一半样本为0，这个data和T分别指的是什么？
        # 找到所有需要“文本引导”的样本的索引
        text_indices = is_text_guided  # 前一半都是False，后一半都是True
        if text_indices.any():
            # a. 只为这些样本提取文本
            texts = [batch['text'][i] for i, flag in enumerate(text_indices) if flag] # 这是一个list，长度是batch_size / 2，里面每个元素是一个style的文本，比如chicken，old
            
            # b. 编码文本，得到干净的 embedding
            text_tokens = clip.tokenize(texts).to(self.mld_device)  # torch.Size([batch_size / 2, 77])
            with torch.cuda.amp.autocast(enabled=False):
                raw_clip_features = self.clip_model.encode_text(text_tokens).float()  # torch.Size([batch_size / 2, 512])
            
            # c. 在应用 dropout 之前，先用干净的 embedding 准备 align_loss 的数据
            adapted_features = self.text_adapter(raw_clip_features)  # torch.Size([batch_size / 2, 512])
            # d. [核心修复] 对其进行 Layer Normalization，强制将其尺度拉到标准范围
            normalized_text_emb = self.text_emb_norm(adapted_features)
        
            text_features = normalized_text_emb.unsqueeze(1)  # torch.Size([64, 1, 512])
            # #【QUESTION】debug了一下，text_features和前面的motion_emb尺度有点不同，这个会有影响么？要不要debug出来看看
            # d. 现在，对 text_features 应用 dropout，用于 denoiser 条件
            mask_uncond = torch.rand(text_features.shape[0], device=self.mld_device) < self.guidance_uncodp  # 基本都是False，偶尔有的是True
            text_features[mask_uncond, ...] = 0

            # e. 将最终的、可能经过 dropout 的 embedding 填充到 text_style_cond 中
            text_style_cond[text_indices] = text_features  # torch.Size([batch_size, 1, 512])，如果看data不看T的话，前一半样本全是0，后一半样本有值

            # f. GT (Ground Truth) embedding 的提取逻辑
            with torch.no_grad():
                motion_seq_for_text = feats_ref[text_indices] * self.std + self.mean  # 【QUESTION】到这里，我觉得应该仔细看看归一化的逻辑了，MotionCLIP应该吃的是未经归一化的数据，你能解释一下所有的归一化逻辑么？看样子我们只归一化text style的这部分，这对么？前面的归一化你也要看一下
                motion_seq_for_text[..., :3] = 0.0  # torch.Size([batch_size / 2, 196, 263])
                motion_seq_for_text = motion_seq_for_text.unsqueeze(-1).permute(0, 2, 3, 1)  # 【QUESTION】torch.Size([64, 263, 1, 196]),依旧发现这里面有很多0和很多1，这正常么？需要debug一下吗
                lengths_text = [lengths[i] for i, flag in enumerate(text_indices) if flag]  # len(): batch_size / 2
                
                gt_motion_emb_for_text = self.motionclip_teacher.encoder({
                    'x': motion_seq_for_text,
                    'y': torch.zeros(motion_seq_for_text.shape[0], dtype=int, device=self.mld_device),
                    'mask': lengths_to_mask(lengths_text, device=self.mld_device)
                })["mu"]  # torch.Size([batch_size / 2, 512])
                # i. [核心修复] 同样对 GT motion embedding 进行 Layer Normalization
                normalized_gt_motion_emb = self.motion_emb_norm(gt_motion_emb_for_text)
            
            # 将用于对齐的两个 embedding 保存下来
            text_emb_for_align = text_features.squeeze(1) # (batch_size / 2, 512)
            motion_emb_for_align = normalized_gt_motion_emb # (batch_size / 2, 512)
            

        # 4. [核心] 打包所有条件，cond_emb：torch.Size([batch_size, 7, 256])，motion_style_cond：torch.Size([batch_size, 1, 512])， trans_cond：torch.Size([batch_size, 196, 3])
        multi_cond_emb = [cond_emb, motion_style_cond, trans_cond]
        
        n_set = self._diffusion_process(z, multi_cond_emb, lengths, style_text_feature=text_style_cond) # 回顾：style_text_feature的shape是torch.Size([batch_size, 1, 512])
        # [核心修改] 将用于 align_loss 的张量加入返回字典
        if text_emb_for_align is not None:  # torch.Size([batch_size/2, 512])
            n_set['text_style_emb'] = text_emb_for_align
            n_set['motion_style_emb_for_text'] = motion_emb_for_align # torch.Size([batch_size/2, 512])
        
        # import os
        # # 只在训练的第一次迭代时执行
        # if self.global_step == 0:
        #     print("\n[DEBUG] DUMPING TENSORS FOR THE FIRST BATCH...")
            
        #     # 准备保存路径
        #     dump_dir = "debug_dumps"
        #     os.makedirs(dump_dir, exist_ok=True)
        #     dump_path = os.path.join(dump_dir, "tensors_first_batch.pt")

        #     # 收集我们想检查的 tensors
        #     tensors_to_dump = {
        #         "motion_emb": motion_emb.detach().cpu(),
        #         "gt_motion_emb_for_text": gt_motion_emb_for_text.detach().cpu(),
        #         "text_features_for_denoiser": text_features.detach().cpu(), # 用于 denoiser 的
        #         "text_emb_for_align": text_emb_for_align.detach().cpu(), # 用于 align_loss 的
        #         "motion_seq_unnormalized": motion_seq.detach().cpu(),
        #         "motion_seq_for_text_unnormalized": motion_seq_for_text.detach().cpu()
        #     }
            
        #     # 保存到文件
        #     torch.save(tensors_to_dump, dump_path)
        #     print(f"[DEBUG] Tensors have been dumped to: {dump_path}")
        #     print("[DEBUG] Exiting for analysis.")
            
        #     # 退出程序
        #     exit()
        
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
        if split in ["train", "val"]:
            if self.stage == "diffusion":
                rs_set = self.train_diffusion_forward(batch)

            elif self.stage == "vae":
                 rs_set = self.train_vae_forward(batch)
                 rs_set["lat_t"] = rs_set["lat_m"]
            else:
                raise ValueError(f"Unsupported stage for loss calculation: {self.stage}!")

            total_loss, loss_dict = self.losses[split].update(rs_set)
            if loss_dict is None:
                if self.trainer.sanity_checking and total_loss is None:
                    return None
                raise ValueError("Loss dictionary is None from torchmetrics.")
            
            for key, value in loss_dict.items():
                self.log(f"{split}/{key}_loss", value, prog_bar=(key == 'total'), on_step=(split == 'train'), on_epoch=True, batch_size=self.cfg.TRAIN.BATCH_SIZE)

        # NOTE: 本来的用来评估的部分的代码，先注释掉，我们先把逻辑跑起来
                
        # # Compute the metrics - currently evaluate results from text to motion
        # if split in ["val", "test"]:
        #     # 1. 从混合批次中，筛选出只属于 HumanML3D 的那部分数据
        #     #    因为 t2m_eval 函数是专门为 HumanML3D 的数据格式设计的。
            
        #     # a. 找到 HumanML3D 样本的布尔掩码
        #     is_humanml3d_mask = [s == "humanml3d" for s in batch["source"]]
            
        #     # b. 如果当前批次中没有任何 HumanML3D 样本，则无法进行 t2m 评估，直接跳过
        #     if not any(is_humanml3d_mask):
        #         print(f"\n[INFO] Skipping metrics calculation for a '{split}' batch containing only Style100 samples.\n")
        #         # 对于 test 阶段，需要返回一个符合预期的空结果
        #         if split == "test": return [], []
        #         # 对于 val 阶段，直接返回之前计算好的损失值即可
        #         return total_loss

        #     # c. 创建一个新的 "纯净" 的批次字典，只包含 HumanML3D 的数据
        #     humanml3d_batch = {}
        #     for key, value in batch.items():
        #         if isinstance(value, torch.Tensor):
        #             # 使用布尔掩码索引 Tensor
        #             humanml3d_batch[key] = value[torch.tensor(is_humanml3d_mask, device=value.device)]
        #         elif isinstance(value, list):
        #             humanml3d_batch[key] = [v for v, flag in zip(value, is_humanml3d_mask) if flag]
        #         else:
        #             humanml3d_batch[key] = value
                    
        #     # 2. 将这个“纯净”的 HumanML3D 批次送入评估函数
        #     #    如果筛选后批次为空，也需要跳过
        #     if len(humanml3d_batch['motion']) == 0:
        #         if split == "test": return [], []
        #         return total_loss

            # rs_set = self.t2m_eval(humanml3d_batch)

            # # MultiModality evaluation sperately
            # if self.trainer.datamodule.is_mm:
            #     metrics_dicts = ['MMMetrics']
            # else:
            #     metrics_dicts = self.metrics_dict
            # # metric = 'TemosMetric' 'TM2TMetrics'
            # for metric in metrics_dicts:
            #     if metric == "TemosMetric":
            #         phase = split if split != "val" else "eval"
            #         if eval(f"self.cfg.{phase.upper()}.DATASETS")[0].lower(
            #         ) not in [
            #                 "humanml3d",
            #                 "kit",
            #         ]:
            #             raise TypeError(
            #                 "APE and AVE metrics only support humanml3d and kit datasets now"
            #             )

            #         getattr(self, metric).update(rs_set["joints_rst"],
            #                                      rs_set["joints_ref"],
            #                                      batch["length"])
            #     elif metric == "TM2TMetrics":
            #         getattr(self, metric).update(
            #             # lat_t, latent encoded from diffusion-based text
            #             # lat_rm, latent encoded from reconstructed motion
            #             # lat_m, latent encoded from gt motion
            #             # rs_set['lat_t'], rs_set['lat_rm'], rs_set['lat_m'], batch["length"])
            #             rs_set["lat_t"],
            #             rs_set["lat_rm"],
            #             rs_set["lat_m"],
            #             batch["length"],
            #         )
            #     elif metric == "UncondMetrics":
            #         getattr(self, metric).update(
            #             recmotion_embeddings=rs_set["lat_rm"],
            #             gtmotion_embeddings=rs_set["lat_m"],
            #             lengths=batch["length"],
            #         )
            #     elif metric == "MRMetrics":
            #         getattr(self, metric).update(rs_set["joints_rst"],
            #                                      rs_set["joints_ref"],
            #                                      batch["length"])
            #     elif metric == "MMMetrics":
            #         getattr(self, metric).update(rs_set["lat_rm"].unsqueeze(0),
            #                                      batch["length"])
            #     elif metric == "HUMANACTMetrics":
            #         getattr(self, metric).update(rs_set["m_action"],
            #                                      rs_set["joints_eval_rst"],
            #                                      rs_set["joints_eval_ref"],
            #                                      rs_set["m_lens"])
            #     elif metric == "UESTCMetrics":
            #         # the stgcn model expects rotations only
            #         getattr(self, metric).update(
            #             rs_set["m_action"],
            #             rs_set["m_rst"].view(*rs_set["m_rst"].shape[:-1], 6,
            #                                  25).permute(0, 3, 2, 1)[:, :-1],
            #             rs_set["m_ref"].view(*rs_set["m_ref"].shape[:-1], 6,
            #                                  25).permute(0, 3, 2, 1)[:, :-1],
            #             rs_set["m_lens"])
            #     else:
            #         raise TypeError(f"Not support this metric {metric}")

        if split in ["test"]:
            return rs_set["joints_rst"], batch["length"]
        
        return total_loss