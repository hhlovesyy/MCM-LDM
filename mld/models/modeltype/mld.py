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


def differentiable_global_pos(features, mean, std):
    """
    输入 features: [Batch, Length, 4] (RotVel, VelX, VelZ, Height) - 归一化后的数据
    输入 mean, std: [263] - 数据集的均值和方差
    输出: [Batch, Length, 2] (Global X, Global Z)
    """
    # 1. 准备反归一化参数 (切片取前4维)
    # [263] -> [1, 1, 4]
    device = features.device
    mean_root = mean[:4].view(1, 1, 4).to(device)
    std_root = std[:4].view(1, 1, 4).to(device)
    
    # 2. 反归一化 (关键！必须还原到真实物理尺度才能算坐标)
    features_denorm = features * std_root + mean_root
    
    # 3. 提取分量
    # HumanML3D: Index 0=RotVelY, 1=VelX, 2=VelZ
    rot_vel = features_denorm[..., 0] # [B, T]
    local_vel_x = features_denorm[..., 1]
    local_vel_z = features_denorm[..., 2]
    
    # 4. 计算绝对朝向 (Global Rotation)
    # 累加角速度。假设初始朝向为0。
    global_rot = torch.cumsum(rot_vel, dim=1)
    
    # 5. 投影到全局坐标系
    # 公式：旋转矩阵变换
    cos_rot = torch.cos(global_rot)
    sin_rot = torch.sin(global_rot)
    
    # HumanML3D 常用坐标系转换
    global_vel_x = local_vel_x * cos_rot + local_vel_z * sin_rot
    global_vel_z = local_vel_z * cos_rot - local_vel_x * sin_rot
    
    # 6. 累加得到位置
    global_pos_x = torch.cumsum(global_vel_x, dim=1)
    global_pos_z = torch.cumsum(global_vel_z, dim=1)
    
    return torch.stack([global_pos_x, global_pos_z], dim=-1)


from .base import BaseModel

def generate_safe_trajectory(batch_size, length, mean, std, shape_type="rectangle", device="cpu"):
    """
    基于数据集统计量生成绝对安全的轨迹。
    确保归一化后的数值在 [-3, 3] 之间。
    """
    # 确保 mean/std 是 tensor
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, device=device)
        std = torch.tensor(std, device=device)
    
    # 取前4维
    mean = mean[:4]
    std = std[:4]
    
    # 初始化: 全部设为均值 (最安全的状态，归一化后为0)
    traj = mean.view(1, 1, 4).repeat(batch_size, length, 1)
    
    # 定义"快"的标准：均值 + 2倍标准差 (归一化后=2.0)
    # 这是一个非常健康的数值，既有明显动作，又不会崩
    FAST_SPEED = mean[2] + 2.0 * std[2] 
    
    # 定义"转弯"的标准：均值 + 2倍标准差 (归一化后=2.0)
    # HumanML3D 的 Std[0] 是 0.0128 rad/frame (约 0.7度/帧)
    # 2.0 倍就是 1.4度/帧。转90度大概需要 60 帧。
    TURN_SPEED = mean[0] + 2.0 * std[0]

    if shape_type == "straight":
        # === 直走 ===
        traj[..., 2] = FAST_SPEED # Z轴前进
        
    elif shape_type == "rectangle":
        # === 矩形 ===
        # 我们根据 TURN_SPEED 反推需要转多久
        # 目标转角: 90度 (pi/2)
        # 所需帧数 = (pi/2) / (2.0 * std[0])
        # 0.0128 * 2 = 0.0256. 1.57 / 0.0256 ≈ 60 帧
        
        turn_duration = 20 
        walk_duration = 20
        cycle_len = (turn_duration + walk_duration) * 4
        
        for t in range(length):
            phase = t % (walk_duration + turn_duration)
            
            if phase < walk_duration:
                # 直走阶段
                traj[:, t, 2] = FAST_SPEED * 2
            else:
                # 转弯阶段
                traj[:, t, 0] = TURN_SPEED * 2 # 向左转
                traj[:, t, 2] = mean[2] + 0.5 * std[2] # 转弯时稍微减速 (0.5 sigma)

    elif shape_type == "circle":
        # === 画圆 ===
        traj[..., 2] = mean[2] + 2.0 * std[2] # 前进
        traj[..., 0] = mean[0] - 2.0 * std[0] # 持续向左转 (1 sigma)
        
    return traj

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


        self._get_t2m_evaluator(cfg)

        if cfg.TRAIN.OPTIM.TYPE.lower() == "adamw":
            self.optimizer = AdamW(lr=cfg.TRAIN.OPTIM.LR,
                                   params=self.parameters())
        else:
            raise NotImplementedError(
                "Do not support other optimizer for now.")

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
    def forward(self, batch):

        lengths = batch["length"]
        # style
        motion = batch["style_motion"].clone()
        motion[...,:4] = 0


        # content
        content_motion = batch['content_motion']
        content_motion = (content_motion - self.mean.to(content_motion.device))/self.std.to(content_motion.device)

        # # trajectory
        # # trans_motion = content_motion.clone()
        style_motion = batch['style_motion']
        style_motion_norm = (style_motion - self.mean.to(style_motion.device))/self.std.to(style_motion.device)
        trans_motion = style_motion_norm.clone()
        trans_cond = trans_motion[...,:4]
        
        # ========================================================
        # 【测试】 注入伪造轨迹 (直接修改 trans_cond)
        # ========================================================
        # USE_FAKE_TRAJ = True 
        # FAKE_SHAPE = "circle" 
        # if USE_FAKE_TRAJ:
        #     print(f"!!! USING FAKE TRAJECTORY: {FAKE_SHAPE} !!!")
        #     bsz = content_motion.shape[0]
        #     max_len = content_motion.shape[1]
            
        #     # A. 生成 Raw Trajectory (米, 弧度)
        #     fake_traj_raw = generate_safe_trajectory(
        #         bsz, max_len, self.mean, self.std, shape_type=FAKE_SHAPE, device=content_motion.device
        #     )
            
        #     # B. 归一化 (关键！必须把物理数值映射到 Latent 能够理解的分布)
        #     # 我们只归一化前 4 维
        #     mean_root = self.mean.to(content_motion.device)[:4]
        #     std_root = self.std.to(content_motion.device)[:4]
            
        #     fake_traj_norm = (fake_traj_raw - mean_root) / std_root
            
        #     # C. 赋值给 trans_cond
        #     # 注意：如果 batch size > 1，这里会把所有样本的轨迹都改成一样的
        #     trans_cond = fake_traj_norm

        # 
        content_motion[...,:4] = 0


        scale = batch["tag_scale"]
        lengths1 = [content_motion.shape[1]]* content_motion.shape[0]
        
        if self.cfg.TEST.COUNT_TIME:
            self.starttime = time.time()
            
        if self.stage in ['diffusion', 'vae_diffusion']:\
            #add style text in test
            
            
            # content motion
            with torch.no_grad():
                z, dist_m = self.vae.encode(content_motion.float(), lengths1)
            uncond_tokens = torch.cat([z, z], dim = 1).permute(1,0,2)
            motion_emb_content = uncond_tokens

            # style motion
            lengths11 = [motion.shape[1]]* motion.shape[0]

# for motion input (bs,60,22,3)->(bs,22,3,60)
            # motion_seq = feats_ref*std + mean
            motion_seq = motion.unsqueeze(-1).permute(0,2,3,1)


            motion_emb = self.motionclip.encoder({'x': motion_seq.float(),
                            'y': torch.zeros(motion_seq.shape[0], dtype=int, device=motion_seq.device),
                            'mask': lengths_to_mask(lengths11, device=motion_seq.device)})["mu"]
            motion_emb = motion_emb.unsqueeze(1)

            # cfree
            uncond_motion_emb = torch.zeros(motion_emb.shape).to(motion_seq.device)
            motion_emb = torch.cat([uncond_motion_emb, motion_emb], dim=0)

            # gendurations = torch.ones((12, 1), dtype=int) * 100
            # generation = self.motionclip.generate(motion_emb.permute(1,0,2), gendurations,
            #                     is_amass=True,
            #                     is_clip_features=True)
            # fff = generation['output_xyz']
            # fff = fff.permute(0,3,1,2)
            # fff = fff.cpu().numpy()
            # np.save("eee.npy",fff)

            # trajectory
            # trans_cond = trans_motion[...,:4]
            uncond_trans = torch.cat([trans_cond, trans_cond], dim = 0)

            # three conditions
            multi_cond_emb = [motion_emb_content, motion_emb, uncond_trans]

            target_traj_feat = batch["style_motion"][..., :4].clone() # [B, T, 4]


            z = self._diffusion_reverse(multi_cond_emb, lengths, scale) # , target_traj_feat
            # z = self._diffusion_reverse(multi_cond_emb, lengths, self.guidance_scale, trans_cond)

        elif self.stage in ['vae']:
            motions = batch['motion']
            z, dist_m = self.vae.encode(motions, lengths)

        with torch.no_grad():
            feats_rst = self.vae.decode(z, lengths)
            # feats_rst[...,:3] = trans_motion[...,:3] # if copy trajectory

        joints = self.feats2joints(feats_rst.detach().cpu())

        return remove_padding(joints, lengths), style_motion.clone()[...,:4]
        # return remove_padding(joints, lengths), fake_traj_raw
    


    def _diffusion_reverse(self, encoder_hidden_states, lengths=None, scale=None):
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

    # def _diffusion_reverse(self, encoder_hidden_states, lengths=None, scale=None, target_traj_feat=None):
    #     # target_traj_feat: [B, T, 4] 归一化后的目标轨迹特征
        
    #     # 1. 初始化 Latents (保持不变)
    #     bsz = encoder_hidden_states[0].shape[0]
    #     if self.do_classifier_free_guidance:
    #         bsz = bsz // 2
        
    #     latents = torch.randn(
    #         (bsz, self.latent_dim[0], self.latent_dim[-1]),
    #         device=encoder_hidden_states[0].device,
    #         dtype=torch.float,
    #     )
    #     latents = latents * self.scheduler.init_noise_sigma
        
    #     self.scheduler.set_timesteps(self.cfg.model.scheduler.num_inference_timesteps)
    #     timesteps = self.scheduler.timesteps.to(encoder_hidden_states[0].device)
        
    #     extra_step_kwargs = {}
    #     if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
    #         extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

    #     # ========================================================
    #     # 【准备 Guidance 目标】
    #     # ========================================================
    #     gt_global_pos = None
    #     # 定义引导强度：这是一场拉锯战，Scale 要给大一点
    #     GUIDANCE_SCALE = 300.0  
    #     # 只在前 60% 的步数做引导 (太靠后了 Latent 已经定型，强改会崩)
    #     GUIDANCE_STEPS = len(timesteps) * 0.6 
        
    #     if target_traj_feat is not None:
    #         # 预先计算好目标的全局路径 (Ground Truth Path)
    #         # 这里的 target_traj_feat 应该是 Style Motion 的轨迹
    #         with torch.no_grad():
    #             gt_global_pos = differentiable_global_pos(
    #                 target_traj_feat, self.mean, self.std
    #             )

    #     # ========================================================
    #     # 【开始去噪循环】
    #     # ========================================================
    #     for i, t in enumerate(timesteps):
    #         # 判断是否进行引导
    #         do_guidance = (gt_global_pos is not None) and (i < GUIDANCE_STEPS)
            
    #         # 临时变量，用于存储梯度修正量
    #         guidance_grad = torch.zeros_like(latents)
            
    #         # --- Guidance 计算分支 ---
    #         if do_guidance:
    #             with torch.enable_grad():
    #                 # 1. Detach 并开启梯度
    #                 # 我们只对当前的 latents 求导
    #                 latents_in = latents.detach().requires_grad_(True)
                    
    #                 # 2. 预测 x0 (Denoised Sample)
    #                 # 为了节省显存，这里我们只用 Cond 分支预测，或者简单的单次预测
    #                 # 这里的 encoder_hidden_states 是 [Uncond, Cond] 拼接的
    #                 # 我们取后半部分 (Cond) 来做预测，因为我们希望生成的动作符合 Cond
    #                 cond_hidden_states = [e.chunk(2, dim=0)[1] for e in encoder_hidden_states]
                    
    #                 noise_pred_guide = self.denoiser(
    #                     sample=latents_in,
    #                     timestep=t,
    #                     encoder_hidden_states=cond_hidden_states,
    #                     lengths=lengths, 
    #                 )[0]
                    
    #                 # 3. 使用调度器公式反推 x0
    #                 # x0 = (xt - sqrt(1-alpha_bar) * eps) / sqrt(alpha_bar)
    #                 alpha_prod_t = self.scheduler.alphas_cumprod[t]
    #                 beta_prod_t = 1 - alpha_prod_t
    #                 pred_x0 = (latents_in - beta_prod_t ** 0.5 * noise_pred_guide) / (alpha_prod_t ** 0.5)
                    
    #                 # 4. VAE Decode (这一步最耗显存)
    #                 # 注意维度转换: [B, 7, 256] -> [7, B, 256] (视你的 VAE 定义而定)
    #                 z_in = pred_x0.permute(1, 0, 2) 
    #                 pred_motion = self.vae.decode(z_in, lengths) # 输出 [B, 263, T] 或 [B, T, 263]
                    
    #                 # 确保维度是 [B, T, 263]
    #                 if pred_motion.shape[1] == 263:
    #                     pred_motion = pred_motion.permute(0, 2, 1)
                        
    #                 # 5. 提取预测轨迹并计算全局位置
    #                 pred_root_feat = pred_motion[..., :4] # 取前4维
    #                 pred_global_pos = differentiable_global_pos(
    #                     pred_root_feat, self.mean, self.std
    #                 )
    #                 # ========================================================
    #                 # 【修复】 轨迹长度对齐 (Interpolation)
    #                 # ========================================================
    #                 # pred_global_pos: [B, T_pred, 2]
    #                 # gt_global_pos:   [B, T_gt, 2]
                    
    #                 if pred_global_pos.shape[1] != gt_global_pos.shape[1]:
    #                     # 1. 转换维度适配 interpolate: [B, T, C] -> [B, C, T]
    #                     gt_pos_permuted = gt_global_pos.permute(0, 2, 1)
                        
    #                     # 2. 线性插值拉伸 GT 轨迹，使其长度等于 Pred 轨迹
    #                     gt_pos_resized = torch.nn.functional.interpolate(
    #                         gt_pos_permuted,
    #                         size=pred_global_pos.shape[1], # 目标长度 (199)
    #                         mode='linear',
    #                         align_corners=True
    #                     )
                        
    #                     # 3. 转回原来的维度: [B, C, T] -> [B, T, C]
    #                     gt_global_pos_aligned = gt_pos_resized.permute(0, 2, 1)
    #                 else:
    #                     gt_global_pos_aligned = gt_global_pos

    #                 # ========================================================

    #                  # ========================================================
    #                 # 【修复】 相对位置 Loss (防止梯度爆炸)
    #                 # ========================================================
                    
    #                 # 1. 归零起点 (Zero-centering)
    #                 # 让两条线的起点重合，只比较形状和走向
    #                 pred_pos_centered = pred_global_pos - pred_global_pos[:, 0:1, :]
    #                 gt_pos_centered = gt_global_pos_aligned - gt_global_pos_aligned[:, 0:1, :]
                    
    #                 # 2. 计算位置 Loss
    #                 loss_pos = torch.nn.functional.mse_loss(pred_pos_centered, gt_pos_centered)
                    
    #                 # 3. 计算速度 Loss (辅助稳定)
    #                 # 提取线速度 (RotVel 不算)
    #                 # pred_root_feat: [B, T, 4] -> Index 1,2 是 VelX, VelZ
    #                 # 记得反归一化回去算 Loss，或者直接在归一化空间算
    #                 # 这里为了简单，直接算归一化后的特征差异
    #                 # gt_root_feat 需要从 target_traj_feat (interpolate后的) 提取
                    
    #                 # 简单的 Latent 空间速度约束:
    #                 # 我们希望生成的 root feat 接近 target
    #                 # (注意：需要把 target interpolate 到和 pred 一样长)
                    
    #                 target_traj_permuted = target_traj_feat.permute(0, 2, 1)
    #                 target_traj_resized = torch.nn.functional.interpolate(
    #                     target_traj_permuted, size=pred_root_feat.shape[1], mode='linear'
    #                 ).permute(0, 2, 1)
                    
    #                 # loss_vel = torch.nn.functional.mse_loss(pred_root_feat, target_traj_resized)
                    
    #                 # # 4. 混合 Loss
    #                 # # 主要靠 velocity (稳定)，辅以 position (修正累积误差)
    #                 # loss = loss_vel * 10.0 + loss_pos * 1.0
    #                 # print(f"Step {i}, Loss Pos: {loss_pos.item():.6f}, Loss Vel: {loss_vel.item():.6f}, Total Loss: {loss.item():.6f}")

    #                 loss = torch.nn.functional.smooth_l1_loss(pred_pos_centered, gt_pos_centered)
                
    #                 print(f"Step {i}, Guidance Loss: {loss.item():.6f}")
                    
    #                 # ========================================================

    #                 # 6. 计算 Loss (使用对齐后的 GT)
    #                 # loss = torch.nn.functional.mse_loss(pred_global_pos, gt_global_pos_aligned)
                    
    #                 # 7. 反向传播
    #                 # 我们想要 Loss 变小 -> 梯度下降
    #                 grad = torch.autograd.grad(loss, latents_in)[0]

    #                 # grad = torch.clamp(grad, -0.1, 0.1)
    #                 grad = torch.clamp(grad, -0.02, 0.02)
                    
    #                 # 记录修正量
    #                 guidance_grad = grad * GUIDANCE_SCALE
                    
    #                 # 清理计算图，防止爆显存
    #                 del latents_in, pred_x0, pred_motion, pred_global_pos, loss, grad

    #         # --- 正常的去噪步骤 ---
    #         latent_model_input = (torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents)
    #         lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance else lengths)
            
    #         noise_pred = self.denoiser(
    #             sample=latent_model_input,
    #             timestep=t,
    #             encoder_hidden_states=encoder_hidden_states,
    #             lengths=lengths_reverse,
    #         )[0]
            
    #         if self.do_classifier_free_guidance:
    #             noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #             noise_pred = noise_pred_uncond + scale * (noise_pred_text - noise_pred_uncond)
            
    #         # 【应用 Guidance】
    #         # 公式: epsilon_hat = epsilon - sqrt(1-alpha) * grad
    #         if do_guidance:
    #             beta_prod_t = 1 - self.scheduler.alphas_cumprod[t]
    #             # 减去梯度方向，让生成的图像往 Loss 小的方向走
    #             noise_pred = noise_pred + (beta_prod_t ** 0.5) * guidance_grad

    #         latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

    #     latents = latents.permute(1, 0, 2)
    #     return latents





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
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=latents.device,
        )
        timesteps = timesteps.long()  # torch.Size([32])
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents.clone(), noise,
                                                       timesteps)  # torch.Size([32, 7, 256])
        # Predict the noise residual
        noise_pred = self.denoiser(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            lengths=lengths,
            return_dict=False,
        )[0] # torch.Size([32, 7, 256])
        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
        if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
            noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
            noise, noise_prior = torch.chunk(noise, 2, dim=0)
        else:
            noise_pred_prior = 0
            noise_prior = 0


        n_set = {
            "noise": noise, # torch.Size([32, 7, 256])
            "noise_prior": noise_prior,
            "noise_pred": noise_pred, # torch.Size([32, 7, 256])
            "noise_pred_prior": noise_pred_prior,
            "noisy_latents": noisy_latents,
            "timesteps": timesteps,
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
        feats_ref = batch["motion"] # torch.Size([32, 40, 263])
        feats_content = batch["motion"].clone() # torch.Size([32, 40, 263])
        feats_content[...,:4] = 0.0
        lengths = batch["length"]
        bsz = feats_ref.shape[0]
        
        # content condition
        with torch.no_grad():
            z, dist = self.vae.encode(feats_ref, lengths) # z:torch.Size([7, 32, 256]), dist: torch.Size([7, 32, 256])
            z_content, dist = self.vae.encode(feats_content, lengths)
            cond_emb = z_content.permute(1,0,2)  # torch.Size([32, 7, 256])          
        # style condition
        motion_seq = feats_ref*self.std + self.mean  # 【QUESTION】看起来风格有做一步反归一化处理，而content并没有被反归一化，是这样么？
        motion_seq[...,:4]=0.0
        motion_seq = motion_seq.unsqueeze(-1).permute(0,2,3,1) # torch.Size([32, 263, 1, 40])
        motion_emb = self.motionclip.encoder({'x': motion_seq,
                        'y': torch.zeros(motion_seq.shape[0], dtype=int, device='cuda:{}'.format(self.cfg["DEVICE"][0])),
                        'mask': lengths_to_mask(lengths, device='cuda:{}'.format(self.cfg["DEVICE"][0]))})["mu"] # 一个style被提取成了512维的tensor，torch.Size([32, 512])
        motion_emb = motion_emb.unsqueeze(1) # torch.Size([32, 1, 512])
        
        # mask_uncond = torch.rand(motion_emb.shape[0]) < self.guidance_uncodp # [F,F,...,F,T,F,T,...]
        # motion_emb[mask_uncond, ...] = 0

        # trans condition
        trans_cond = batch["motion"][...,:4]  # torch.Size([32, 40, 3])

        # A. Content Dropout (30%)
        # 解决你的顾虑：切断 Content 依赖
        mask_content = torch.rand(bsz, device=self.device) < 0.15
        cond_emb[mask_content] = 0
        
        # B. Style Dropout (30%)
        # 解决你的顾虑：切断 Style 依赖，防止 Style 篡位
        # 注意：这是独立的随机数，所以会有 0.3*0.3=9% 的概率两者同时消失
        mask_style = torch.rand(bsz, device=self.device) < 0.3
        motion_emb[mask_style] = 0
        
        # C. Trajectory Dropout (30%)
        # 作用：让模型在没有轨迹时也能生成自然动作(无条件生成)
        # 保护机制：如果 Content 和 Style 都没了，千万不能丢 Traj (否则就是全黑)
        # 逻辑：Drop Traj IF (Content OK OR Style OK)
        can_drop_traj = (~mask_content) | (~mask_style) # 只要有一个还活着，就可以丢轨迹
        
        rand_traj = torch.rand(bsz, device=self.device) < 0.5
        # 最终 mask: 随机到了要丢 且 允许丢
        final_mask_traj = rand_traj & can_drop_traj
        
        # 这里我们要用 mask 乘以 tensor (广播)
        trans_cond[final_mask_traj.squeeze()] = 0

        # three condition
        multi_cond_emb = [cond_emb, motion_emb, trans_cond] # 复习一下： cond_emb：内容（torch.Size([32, 7, 256])），motion_emb：风格（torch.Size([32, 1, 512])），trans_cond：轨迹（torch.Size([32, 40, 3])）


        # diffusion process return with noise and noise_pred
        n_set = self._diffusion_process(z, multi_cond_emb, lengths) # 返回的n_set是一个字段，包含计算loss的时候pytorch_lightning所关心的内容
        
        # ============================================================
        # 【ICME 解耦核心】 显式轨迹损失 (Explicit Trajectory Loss)
        # ============================================================
        if self.training:
            # 5.1 反解 x0
            z_t = n_set['noisy_latents']
            t = n_set['timesteps']
            noise_pred = n_set['noise_pred']
            
            alphas = self.noise_scheduler.alphas_cumprod.to(self.device)
            # 处理维度广播 [B] -> [B, 1, 1]
            sqrt_alpha = alphas[t] ** 0.5
            sqrt_one_minus_alpha = (1 - alphas[t]) ** 0.5
            while len(sqrt_alpha.shape) < len(z_t.shape):
                sqrt_alpha = sqrt_alpha.unsqueeze(-1)
                sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
                
            pred_z0 = (z_t - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
            
            # 5.2 解码 (Decode)
            # [B, 7, 256] -> [7, B, 256]
            pred_motion = self.vae.decode(pred_z0.permute(1,0,2), lengths) # torch.Size([32, 196, 263])
            pred_motion = pred_motion.permute(0, 2, 1) # [B, T, 263] -> [B, 263, T] torch.Size([32, 263, 196])
            
            # 2. 提取根节点特征 (Pred & GT)
            # 取前 4 维：[Rot, VelX, VelZ, Height] 
            pred_root_feat = pred_motion[:, :4, :]  # torch.Size([32, 4, 196])
            gt_root_feat = batch["motion"][..., :4].permute(0, 2, 1) # torch.Size([32, 4, 196])
            
            # ========================================================
            # 【ICME 杀手锏】 累积位置损失 (Accumulated Loss)
            # ========================================================
            # 我们不仅比对速度，还要比对积分后的位置！
            # 只有这样，模型才不敢有一丝一毫的偏离。
            
            # 2.1 速度/高度损失 (基础)
            loss_vel = torch.nn.functional.mse_loss(pred_root_feat, gt_root_feat)
            # 2.2 恢复位置 (积分)
            # 简化计算：我们假设初始位置都是 0，直接累加速度
            # 注意：真实的轨迹恢复需要处理旋转，这里为了 Loss 可导，我们做简化近似
            # 我们只累加 X 和 Z 的线速度
            
            # pred_vels: [B, 2, T] (VelX, VelZ)
            pred_vels = pred_root_feat[:, 1:3, :]
            gt_vels = gt_root_feat[:, 1:3, :]
            
            # 累加 (Cumsum) -> 得到相对位移轨迹
            pred_pos = torch.cumsum(pred_vels, dim=-1)
            gt_pos = torch.cumsum(gt_vels, dim=-1)
            
            # 2.3 位置损失 (Global Position Loss)
            # 这个 Loss 会随着时间 t 变大，惩罚力度极强
            loss_pos = torch.nn.functional.mse_loss(pred_pos, gt_pos)
            
            # 3. 最终轨迹损失
            # 速度损失权重 10，位置损失权重 5 (因为它数值本来就大)
            # 加上高度损失 (Index 3) 确保能跳起来
            
            # 这里的 loss_traj 包含了极强的约束
            n_set['loss_traj'] = (loss_vel * 5.0) + (loss_pos * 0.05)
            if self.global_step % 100 == 0:
                print(f"DEBUG: Loss Vel: {loss_vel.item():.4f} | Loss Pos: {loss_pos.item():.4f}")
        
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

            loss_diff = self.losses[split].update(rs_set)
            if loss_diff is None:
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

        # return forward output rather than loss during test
        if split in ["test"]:
            return rs_set["joints_rst"], batch["length"]
        
        # 2. 轨迹 Loss
        loss_traj = rs_set.get('loss_traj', 0.0)
        
        # 3. 总 Loss
        total_loss = loss_diff + loss_traj
        
        # 4. Logging
        if split == "train":
            self.log('train/loss_total', total_loss, on_step=True, logger=True)
            self.log('train/loss_diff', loss_diff, on_step=True, logger=True)
            if isinstance(loss_traj, torch.Tensor):
                self.log('train/loss_traj', loss_traj, on_step=True, logger=True)
        
        if self.global_step % 100 == 0:
            print(f"DEBUG: Loss Diff: {loss_diff.item():.4f}")
        return total_loss
