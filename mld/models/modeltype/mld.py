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
        self.motionclip.train()
        for p in self.motionclip.parameters():
            p.requires_grad = True

        # NEW: 加载并冻结 CLIP 文本编码器
        self.clip_model, _ = clip.load("ViT-B/32", device=self.mld_device)
        # self.clip_model.train()
        # print("Finetuning CLIP model. Parameters to be trained:")
        # for name, p in self.clip_model.named_parameters():
        #     # 我们只解冻最后一个 Transformer 块 (resblocks.11) 的 MLP 部分
        #     # 以及文本投影矩阵 (text_projection)，这是 CLIP 的一个重要部分
        #     # 我们继续冻结所有的 LayerNorm (ln_) 和自注意力模块 (attn)。
        #     if ("transformer.resblocks.11.mlp" in name) or ("text_projection" in name):
        #         p.requires_grad = True
        #         print(f"  - {name}")
        #     else:
        #         p.requires_grad = False
        self.clip_model.eval()
        for name, p in self.clip_model.named_parameters():
            p.requires_grad = False

        # 验证一下 ln_final 是否被冻结
        print(f"CLIP's ln_final.weight.requires_grad: {self.clip_model.ln_final.weight.requires_grad}") # 应该输出 False

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
    
    def configure_optimizers(self):
        # 核心思路：这是 PyTorch Lightning 配置优化器的标准方法。
        # 我们将之前在 __init__ 中的所有逻辑都搬到这里。
        # 当 Trainer 启动时，它会自动调用这个方法，并获取返回的优化器。
        # 这样做，可以确保 AMP 插件能够正确地“包装”和管理我们的优化器。

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
            param_groups = [
                {'params': denoiser_base_params, 'lr': lr_base, 'name': 'denoiser_base'},
                {'params': denoiser_new_params, 'lr': lr_new, 'name': 'denoiser_new'},
                {'params': filter(lambda p: p.requires_grad, self.clip_model.parameters()), 'lr': lr_finetune_clip, 'name': 'clip_finetune'},
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
                    text_features = self.clip_model.encode_text(text_tokens).float()
            
            uncond_text_features = torch.zeros_like(text_features)
            text_style_cond = torch.cat([uncond_text_features, text_features], dim=0).unsqueeze(1)

        elif is_motion_mode:
            style_motion = batch["style_motion"]
            
            # --- [最终修复] ---
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
                    # 3. 使用正确的 style_lengths 来生成掩码！
                    'mask': lengths_to_mask(style_lengths, device=self.mld_device)
                })["mu"]
            # --- 修复结束 ---

            uncond_motion_emb = torch.zeros_like(motion_emb_features)
            motion_style_cond = torch.cat([uncond_motion_emb, motion_emb_features], dim=0).unsqueeze(1)

        # --- 3. [核心修改] 打包所有条件并调用扩散逆过程 ---
        multi_cond_emb = [cond_emb, motion_style_cond, uncond_trans]

        z = self._diffusion_reverse(
            encoder_hidden_states=multi_cond_emb, 
            lengths=content_lengths, # 逆过程的长度以 content 为准
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
        # print("\n" + "="*20 + " NEW BATCH DIAGNOSIS " + "="*20)
        feats_ref = batch["motion"]  # shape:torch.Size([batch_size, motion_len, 263]): 应该是[batchsize, motion_len, nfeats]
        feats_content = batch["motion"].clone()  # shape:torch.Size([batch_size, motion_len, 263]):
        feats_content[...,:3] = 0.0  # shape:torch.Size([batch_size, motion_len, 263]):
        lengths = batch["length"]  # len(): batch_size
        bs = feats_ref.shape[0]

        # 1. 从 batch 中解包出我们的关键标志
        is_text_guided = batch['is_text_guided']  # torch.Size([batch_size])，前面全是false，后面全是true，毕竟是直接拼接的
        # print(f"Batch Size: {bs}, Text-Guided Samples: {is_text_guided.sum().item()}")

        # 检查输入数据本身是否有 nan
        # if torch.isnan(feats_ref).any():
        #     print("!!!!!! FATAL: NaN detected in the initial 'batch[\"motion\"]' itself! Problem is in the dataset/dataloader.")
        #     raise RuntimeError("NaN in initial batch data.")
        # print("Step 0: Initial batch data is clean (no NaN).")
        
        # print("\n--- Step 1: VAE Encoding for Content Condition ---")
        # content condition
        with torch.no_grad():
            z, dist = self.vae.encode(feats_ref, lengths)  # z: torch.Size([7, batch_size, 256]), dist(batch_shape): torch.Size([7, batch_size, 256])
            z_content, dist = self.vae.encode(feats_content, lengths) # z_content: torch.Size([7, batch_size, 256]), dist:batch_shape:torch.Size([7, batch_size, 256])
        
        # if torch.isnan(z_content).any():
        #     print("!!!!!! FATAL: NaN detected in 'z_content' immediately after VAE encoding!")
        #     # 找出是哪个样本出了问题
        #     nan_indices = torch.any(torch.isnan(z_content), dim=(0, 2)).nonzero(as_tuple=True)[0]
        #     print(f"Indices in batch with NaN content: {nan_indices.tolist()}")
        #     for idx in nan_indices:
        #         print(f" -> Sample at index {idx.item()} is from {'100Style' if is_text_guided[idx] else 'HumanML3D'}")
        #     raise RuntimeError("NaN from VAE encoder.")
        # print("Step 1: VAE content encoding is clean.")
        
        cond_emb = z_content.permute(1,0,2)  # torch.Size([batch_size, 7, 256])    
        
        # trans condition
        trans_cond = batch["motion"][...,:3]  # torch.Size([batch_size, motion_len, 3]) 这里面的3推测应该是全局的xyz属性，表示轨迹       
        
        # --- [核心修复] 实现基于 is_text_guided 的精确条件开关 ---

        # 2. 初始化两个风格条件张量，全部为零
        # 动作风格 (fs) 的 embedding 维度是 512
        motion_style_cond = torch.zeros(feats_ref.shape[0], 1, 512, device=self.mld_device, dtype=torch.float) # torch.Size([batch_size, 1, 512])
        # 文本风格 (text_style) 的 embedding 维度也是 512 (来自 CLIP ViT-B/32)
        text_style_cond = torch.zeros(feats_ref.shape[0], 1, 512, device=self.mld_device, dtype=torch.float)  # torch.Size([batch_size, 1, 512])

        # 3. 根据标志，为每个样本填充正确的风格条件
        # 初始化用于 align_loss 的张量
        text_emb_for_align = None
        motion_emb_for_align = None
        # 找到所有需要“动作引导”的样本的索引
        motion_indices = ~is_text_guided # torch.Size([batch_size]),前一半都是True，后一半都是False
        if motion_indices.any():
            # a. 只为这些样本提取动作风格
            motion_seq = feats_ref[motion_indices] * self.std + self.mean  # torch.Size([batch_size / 2, motion_len, 263])
            motion_seq[..., :3] = 0.0
            motion_seq = motion_seq.unsqueeze(-1).permute(0, 2, 3, 1) # torch.Size([batch_size / 2, 263, 1, motion_len])
            
            lengths_motion = [lengths[i] for i, flag in enumerate(motion_indices) if flag]  # len(): batch_size / 2

            motion_emb = self.motionclip.encoder({
                'x': motion_seq,
                'y': torch.zeros(motion_seq.shape[0], dtype=int, device=self.mld_device),
                'mask': lengths_to_mask(lengths_motion, device=self.mld_device)
            })["mu"].unsqueeze(1) # torch.Size([batch_size / 2, 1, 512])
            
            # if torch.isnan(motion_emb).any():
            #     print("!!!!!! FATAL: NaN detected from MotionClip encoder!， motion features！")
            #     raise RuntimeError("NaN from MotionClip.")
            # print("Step 2a: MotionClip encoding is clean.")

            # b. 应用 CFG dropout
            mask_uncond = torch.rand(motion_emb.shape[0], device=self.mld_device) < self.guidance_uncodp  # torch.Size([batch_size/2])
            motion_emb[mask_uncond, ...] = 0
            
            # c. 将计算出的 embedding 填充回主张量的对应位置
            motion_style_cond[motion_indices] = motion_emb  # torch.Size([batch_size, 1, 512])，每个样本前一半有值，后一半为0，这对吗？？？会不会对原来的MCM-LDM造成比较大的干扰？这个逻辑模型能学到东西吗？

        # 找到所有需要“文本引导”的样本的索引
        text_indices = is_text_guided  # 前一半都是False，后一半都是True
        if text_indices.any():
            # a. 只为这些样本提取文本
            texts = [batch['text'][i] for i, flag in enumerate(text_indices) if flag]
            
            # b. 编码文本，得到干净的 embedding
            text_tokens = clip.tokenize(texts).to(self.mld_device)
            with torch.cuda.amp.autocast(enabled=False):
                text_features = self.clip_model.encode_text(text_tokens).float().unsqueeze(1)

            # c. 在应用 dropout 之前，先用干净的 embedding 准备 align_loss 的数据
            text_emb_for_align = text_features.squeeze(1).clone()
            
            # d. 现在，对 text_features 应用 dropout，用于 denoiser 条件
            mask_uncond = torch.rand(text_features.shape[0], device=self.mld_device) < self.guidance_uncodp
            text_features[mask_uncond, ...] = 0

            # e. 将最终的、可能经过 dropout 的 embedding 填充到 text_style_cond 中
            text_style_cond[text_indices] = text_features

            # f. GT (Ground Truth) embedding 的提取逻辑
            with torch.no_grad():
                motion_seq_for_text = feats_ref[text_indices] * self.std + self.mean
                motion_seq_for_text[..., :3] = 0.0
                motion_seq_for_text = motion_seq_for_text.unsqueeze(-1).permute(0, 2, 3, 1)
                lengths_text = [lengths[i] for i, flag in enumerate(text_indices) if flag]
                
                self.motionclip.eval()
                # 将提取出的 GT embedding 赋值给一个临时变量
                gt_motion_emb_for_text = self.motionclip.encoder({
                    'x': motion_seq_for_text,
                    'y': torch.zeros(motion_seq_for_text.shape[0], dtype=int, device=self.mld_device),
                    'mask': lengths_to_mask(lengths_text, device=self.mld_device)
                })["mu"]
                self.motionclip.train()
            
            # 将用于对齐的两个 embedding 保存下来
            text_emb_for_align = text_features.squeeze(1) # (batch_size / 2, 512)
            motion_emb_for_align = gt_motion_emb_for_text # (batch_size / 2, 512)
            
        # print("\n--- Step 3: Final Check before _diffusion_process ---")

        # 4. [核心] 打包所有条件
        multi_cond_emb = [cond_emb, motion_style_cond, trans_cond]
        # 我们把你之前的检查站放在这里
        # if torch.isnan(z).any(): raise RuntimeError("FATAL: NaN in 'z'")
        # if torch.isnan(cond_emb).any(): raise RuntimeError("FATAL: NaN in 'cond_emb'")
        # if torch.isnan(motion_style_cond).any(): raise RuntimeError("FATAL: NaN in 'motion_style_cond'")
        # if torch.isnan(trans_cond).any(): raise RuntimeError("FATAL: NaN in 'trans_cond'")
        # if torch.isnan(text_style_cond).any(): raise RuntimeError("FATAL: NaN in 'text_style_cond'")
        # print("Step 3: All inputs to Denoiser are clean.")
        # print("="*50 + "\n")
        
        # 注意：我们将 text_style_cond 作为一个独立的参数传入，这与你之前的设计保持一致
        n_set = self._diffusion_process(z, multi_cond_emb, lengths, style_text_feature=text_style_cond) # 回顾：style_text_feature的shape是torch.Size([batch_size, 1, 512])
        # [核心修改] 将用于 align_loss 的张量加入返回字典
        if text_emb_for_align is not None:
            n_set['text_style_emb'] = text_emb_for_align
            n_set['motion_style_emb_for_text'] = motion_emb_for_align
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
                rs_set = self.train_diffusion_forward(batch)

            # ... (后续的 vae 和 loss 计算逻辑保持不变)
            elif self.stage == "vae":
                 rs_set = self.train_vae_forward(batch)
                 rs_set["lat_t"] = rs_set["lat_m"]
            else:
                raise ValueError(f"Unsupported stage for loss calculation: {self.stage}!")

            # --- [最终日志修复] ---
            # 计算损失，并用 loss_val 变量接收
            total_loss, loss_dict = self.losses[split].update(rs_set)
            if loss_dict is None:
                if self.trainer.sanity_checking and total_loss is None:
                    return None
                raise ValueError("Loss dictionary is None from torchmetrics.")

            # 记录日志，这个逻辑也保持你现有的不变
            for key, value in loss_dict.items():
                self.log(f"{split}/{key}_loss", value, prog_bar=(key == 'total'), on_step=(split == 'train'), on_epoch=True, batch_size=self.cfg.TRAIN.BATCH_SIZE)

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

        # return forward output rather than loss during test
        if split in ["test"]:
            return rs_set["joints_rst"], batch["length"]
        # return loss
        # 返回总损失给优化器
        # if isinstance(loss_val, dict):
        #     return loss_val['total']
        
        # 如果 loss_val 是张量，直接返回它
        return total_loss