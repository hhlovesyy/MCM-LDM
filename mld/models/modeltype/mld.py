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

from mld.models.architectures.scpa_encoder import SCPAEncoder, SCPAEncoderSimple



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




        # self.text_encoder = instantiate_from_config(cfg.model.text_encoder)

        parameters = read_yaml_to_dict("configs/motionclip_config/motionclip_params_263.yaml")
        parameters["device"] = 'cuda:{}'.format(cfg["DEVICE"][0])        
        # self.motionclip = get_model_and_data(parameters, split='vald')
        # print("load motion clip-xyz-263")
        # print("Restore weights..")
        # checkpointpath = "checkpoints/motionclip_checkpoint/motionclip.pth.tar"
        # state_dict = torch.load(checkpointpath, map_location=parameters["device"])
        # load_model_wo_clip(self.motionclip, state_dict)

        self.mean = torch.tensor(self.datamodule.mean).to(parameters["device"])
        self.std = torch.tensor(self.datamodule.std).to(parameters["device"])

        # #don't train motionclip
        # self.motionclip.training = False
        # for p in self.motionclip.parameters():
        #     p.requires_grad = False

        self.motionclip = None

        # [PhysiMoS 修改] 2. 实例化 Physics Encoder
        # 假设 cfg 中有相关配置，如果没有，我们使用硬编码默认值（探针阶段为了稳）
        # 实际上你应该在 configs/model.yaml 里加这部分，或者像下面这样动态注入：
        print("[PhysiMoS] Initializing SCPAEncoder...")
        # self.physics_encoder = SCPAEncoder(
        #     scene_cat_dim=1,      # 对应你的 heavy, light 等 5 类
        #     phys_params_dim=3,    # 对应 mass, strength 等 4 个参数
        #     d_model=self.latent_dim[-1], # 256
        #     n_head=4,
        #     n_layers=1
        # )
        self.physics_encoder = SCPAEncoderSimple(
            scene_cat_dim=1, 
            phys_params_dim=3,
        )

        self.vae = instantiate_from_config(cfg.model.motion_vae)
        # Don't train the motion encoder and decoder
        if self.stage == "diffusion":
            self.vae.training = False
            for p in self.vae.parameters():
                p.requires_grad = False

        self.denoiser = instantiate_from_config(cfg.model.denoiser)

        self.scheduler = instantiate_from_config(cfg.model.scheduler)
        self.noise_scheduler = instantiate_from_config(
            cfg.model.noise_scheduler)

        # [PhysiMoS 修改] 4. 精细化冻结参数 (Surgical Freezing)
        # 目标：冻结 VAE, Denoiser 主干。只训练 PhysicsAdapter 和 SCPAEncoder。
        
        # A. 冻结 Denoiser 所有参数
        for name, param in self.denoiser.named_parameters():
            param.requires_grad = False
        
        # B. 解冻 Physics Adapter (我们在 mld_denoiser.py 里新加的模块)
        trainable_params = []
        print("[PhysiMoS] Unfreezing Physics Adapters and Encoder...")
        for name, param in self.denoiser.named_parameters():
            if "phys_adapter" in name: # 只要名字里带这个，就训练
                param.requires_grad = True
                trainable_params.append(param)
                # print(f"  - Unfrozen: {name}")
        
        # C. 解冻 Physics Encoder (全量训练)
        for param in self.physics_encoder.parameters():
            param.requires_grad = True
            trainable_params.append(param)

        # self._get_t2m_evaluator(cfg)

        # D. Optimizer 只传入可训练参数
        if cfg.TRAIN.OPTIM.TYPE.lower() == "adamw":
            self.optimizer = AdamW(lr=cfg.TRAIN.OPTIM.LR, params=trainable_params)
            print(f"[PhysiMoS] Optimizer initialized with {len(trainable_params)} tensor groups.")
        else:
            raise NotImplementedError("Do not support other optimizer for now.")

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
    def forward(self, batch, return_attn=False):
        # [PhysiMoS 修改] 重写 Inference 逻辑
        
        # 1. 准备数据
        lengths = batch["length"]
        # Content Motion (Input Condition)
        # 记得：batch['motion_before'] 是我们在 dataset 里造的 input
        content_motion = batch['motion_before'].to(self.device) # torch.Size([1, 38, 263])
        
        # Trajectory Condition (Input)
        # 使用 content motion 的轨迹作为输入
        trans_motion = content_motion.clone()
        trans_cond = trans_motion[...,:3] # [B, S, 3]

        # Physics Condition (Input)
        phys_params = batch['phys_params'].to(self.device) # torch.Size([1, 4])
        scene_cat = batch['scene_cat'].to(self.device)

        scale = batch.get("tag_scale", 1.0) # 默认 scale
        
        if self.cfg.TEST.COUNT_TIME:
            self.starttime = time.time()
            
        if self.stage in ['diffusion', 'vae_diffusion']:
            
            # 2. 编码 Content
            # 注意：这里需要 VAE encode。原来的代码逻辑是 encode -> repeat(uncond)
            # 我们的 VAE 被冻结了，直接用。
            with torch.no_grad():
                # 注意：VAE encode 期望输入是 Normalized 过的吗？是的。Dataset 已经 normalized 了。
                z_content, dist_m = self.vae.encode(content_motion.float(), lengths) # torch.Size([7, 1, 256])
            
            # [重要] 原版这里为了 classifier-free guidance 做了 concat([z, z])
            # 我们也照做，虽然物理部分可能不需要 unconditional，但为了维度对齐
            motion_emb_content = z_content.permute(1,0,2) # torch.Size([1, 7, 256])

            # 3. 编码 Physics (替代 Style)
            # 我们用 SCPAEncoder 提取特征
            physics_emb, attn_weights = self.physics_encoder(scene_cat, phys_params, need_weights=True) 
            
            # Classifier-Free Guidance 准备
            # Uncond Condition: 物理参数全零，或者用一个特殊的 learnable token？
            # 简单起见，我们这里造一个全零的物理嵌入作为 uncond
            uncond_physics_emb = torch.zeros_like(physics_emb)
            
            # 如果做 CFG，需要把 condition 翻倍 (cond, uncond)
            # 这里我们在 _diffusion_reverse 里处理翻倍逻辑吗？
            # 原版 _diffusion_reverse 里：latents 翻倍，encoder_hidden_states 不翻倍？
            # 不，原版代码有点乱。通常做法是把 encoder_hidden_states 也在外边翻倍。
            # 让我们看 _diffusion_reverse: 
            #   noise_pred = self.denoiser(..., encoder_hidden_states, ...)
            #   noise_pred.chunk(2)
            # 这意味着 denoiser 一次性处理了 (cond_batch + uncond_batch)。
            # 所以我们需要在这里把 condition 翻倍。
            
            # Check mld_denoiser's forward: 它接收的 hidden_states 是 List。
            # 它内部不做翻倍。所以我们要传进去双倍的 batch。
            
            motion_emb_content = torch.cat([motion_emb_content, motion_emb_content], dim=0) # [2B, S, D]
            physics_emb = torch.cat([uncond_physics_emb, physics_emb], dim=0) # [2B, 1, D]
            trans_cond = torch.cat([trans_cond, trans_cond], dim=0) # [2B, S, 3]

            # 4. 组装条件
            multi_cond_emb = [motion_emb_content, physics_emb, trans_cond]

            # 5. 逆向扩散采样
            z = self._diffusion_reverse(multi_cond_emb, lengths, scale)

        elif self.stage in ['vae']:
            motions = batch['motion_after'] # Ground Truth
            z, dist_m = self.vae.encode(motions, lengths)

        with torch.no_grad():
            feats_rst = self.vae.decode(z, lengths)

        joints = self.feats2joints(feats_rst.detach().cpu())
        joints = remove_padding(joints, lengths)
        if return_attn:
            return joints, attn_weights
        return joints
    
    
    def _diffusion_reverse(self, encoder_hidden_states, lengths=None, scale=None):
        # init latents
        # 注意：encoder_hidden_states[0] 已经是翻倍后的 batch (2B)，如果开了 CFG
        bsz = encoder_hidden_states[0].shape[0] # Batch dimension is 1 for content [S, B, D]
        if self.do_classifier_free_guidance:
            bsz = bsz // 2

        latents = torch.randn(
            (bsz, self.latent_dim[0], self.latent_dim[-1]),
            device=encoder_hidden_states[0].device,
            dtype=torch.float,
        )  # torch.Size([1, 7, 256])

        latents = latents * self.scheduler.init_noise_sigma
        self.scheduler.set_timesteps(self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(encoder_hidden_states[0].device)
        
        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta
        
        # reverse
        for i, t in enumerate(timesteps):
            latent_model_input = (torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents)
            lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance else lengths)
            
            # [重要] Denoiser Forward
            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths_reverse,
            )[0]  # torch.Size([2, 7, 256])
            
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + scale * (noise_pred_text - noise_pred_uncond) # torch.Size([1, 7, 256])
            
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        latents = latents.permute(1, 0, 2) # torch.Size([7, 1, 256])
        return latents





    def _diffusion_process(self, latents, encoder_hidden_states, lengths=None):
        """
        heavily from https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
        """
        # our latent   [batch_size, n_token=1 or 5 or 10, latent_dim=256]
        # sd  latent   [batch_size, [n_token0=64,n_token1=64], latent_dim=4]
        # [n_token, batch_size, latent_dim] -> [batch_size, n_token, latent_dim]
        latents = latents.permute(1, 0, 2)  # torch.Size([bs, 7, 256])

        # Sample noise that we'll add to the latents
        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(latents)
        bsz = latents.shape[0] # bs
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=latents.device,
        )  # torch.Size([bs])
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents.clone(), noise,
                                                       timesteps)  # torch.Size([bs, 7, 256])
        # Predict the noise residual
        noise_pred = self.denoiser(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            lengths=lengths,
            return_dict=False,
        )[0]  # torch.Size([bs, 7, 256])
        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
        if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
            noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
            noise, noise_prior = torch.chunk(noise, 2, dim=0)
        else:
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
    # [PhysiMoS 修改] 核心训练逻辑
    def train_diffusion_forward(self, batch):
        # 1. 获取数据
        # feats_ref (Target): motion_after
        feats_ref = batch["motion_after"]  # torch.Size([bs, motion_seq_len, 263])
        
        # feats_content (Input): motion_before
        # dataset 已经帮我们处理好了 motion_before = motion_after (或被 mask)
        feats_content = batch["motion_before"] # torch.Size([bs, motion_seq_len, 263])
        
        # Physics Input
        phys_params = batch["phys_params"] # torch.Size([bs, 4])
        scene_cat = batch["scene_cat"] # torch.Size([bs, 5])

        lengths = batch["length"] # torch.Size([bs])

        # ============================================================
        # [PhysiMoS 核心修复] 破坏捷径 (Shortcut Breaking)
        # ============================================================
        # 我们以 50% 的概率，把输入的内容动作 (feats_content) 全部抹零。
        # 此时，模型为了恢复 feats_ref，就只能依赖 physics_emb 了！
        # 这会强迫 Attention 机制去寻找正确的物理参数。
        
        # 生成一个与 Batch Size 相同的随机掩码
        # True 表示保留内容，False 表示抹除内容
        keep_prob = 0.5  # 50% 的概率保留内容，50% 的概率抹除
        mask_content = torch.rand(feats_ref.shape[0], device=feats_ref.device) < keep_prob
        
        # 扩展掩码维度以便广播: [B] -> [B, 1, 1]
        mask_content = mask_content.unsqueeze(1).unsqueeze(2)
        
        # 应用掩码：被 mask 的样本，内容变成全 0
        feats_content = feats_content * mask_content
        # ============================================================
        
        # 2. 编码 Content (通过 VAE)
        with torch.no_grad():
            # Target Latent (z) - 我们要预测的目标
            z, dist = self.vae.encode(feats_ref, lengths)  # z:torch.Size([7, bs, 256])
            
            # Condition Latent (z_content) - 给模型的提示
            # 这里我们直接编码 motion_before。
            z_content, dist_c = self.vae.encode(feats_content, lengths)
            cond_emb = z_content.permute(1,0,2) # torch.Size([bs, 7, 256])

        # 3. 编码 Physics (New Style)
        # 调用我们的 SCPAEncoder
        # 注意：我们在 init 里把这个模块放进了 optimizer，所以这里有梯度
        physics_emb = self.physics_encoder(scene_cat, phys_params) # torch.Size([bs, 1, 256])
        
        # 随机 Drop (CFG Training)
        # 10% 的概率把 physics_emb 置零，强迫模型学会 unconditionally (或者只依赖 content) 生成
        if self.guidance_uncodp > 0:
            mask_uncond = torch.rand(physics_emb.shape[0], device=physics_emb.device) < self.guidance_uncodp
            physics_emb[mask_uncond] = 0 # Zero out physics condition

        # 4. Trajectory Condition
        # 使用 content motion 的前3维 (root position/velocity)
        # dataset 里的 motion 已经是 normalized 的 feature，前3维通常是 root velocity/height
        # [注意] 原版是 batch["motion"][...,:3]，我们也取 content 的前3维
        trans_cond = feats_content[...,:3] # torch.Size([bs, motion_seq_len, 3])

        # 5. 打包所有条件
        # 顺序必须和 mld_denoiser.py 里的解包顺序一致：
        # [0]: Content——torch.Size([bs, 7, 256]), [1]: Physics —— torch.Size([bs, 1, 256]), [2]: Trajectory——torch.Size([bs, motion_seq_len, 3])
        multi_cond_emb = [cond_emb, physics_emb, trans_cond]

        # 6. 进入扩散过程计算 Loss
        n_set = self._diffusion_process(z, multi_cond_emb, lengths)
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

        # # return forward output rather than loss during test
        # if split in ["test"]:
        #     return rs_set["joints_rst"], batch["length"]
            
        # ==============================================
        # [PhysiMoS 修复] 显式记录日志到 TensorBoard
        # ==============================================
        # 记录总 Loss
        self.log(f"losses/{split}/total", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
