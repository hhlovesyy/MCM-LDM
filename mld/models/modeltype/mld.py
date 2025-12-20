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
from mld.data.humanml.scripts.motion_process import (process_file,
                                                     recover_from_ric,
                                                     extract_features)
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
import torch.nn.functional as F
import matplotlib.pyplot as plt
# import evaluate.utils.rotation_conversions as geometry
from mld.models.modeltype.trajectory_utils import *




from .base import BaseModel

def calculate_trajectory_correct(data):
    """
    修正后的积分逻辑，完美对齐 HumanML3D 的 recover_from_ric。
    data: [Batch, Seq, 4] (RotVel, VelX, VelZ, Height)
    """
    # 1. 提取特征
    rot_vel = data[..., 0]
    local_vel_x = data[..., 1]
    local_vel_z = data[..., 2]
    
    # 2. 积分得到累积角度 (Feature Space)
    # 模仿 HumanML3D 的错位逻辑：第 t 帧的速度是用 t-1 帧的角度旋转的
    # 所以角度累加要比速度慢一拍，或者速度序列要做 shift
    # 这里我们采用与 GT 脚本完全一致的逻辑：
    r_rot_ang = torch.zeros_like(rot_vel)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)
    
    # 【核心修正 1】: 真实的物理角度是特征角度的 2 倍！
    # 因为 Quaternion q = [cos(a), 0, sin(a), 0] 代表旋转 2a
    real_angle = r_rot_ang * 2.0
    
    # 3. 计算旋转后的世界速度
    # HumanML3D 使用 qinv 进行旋转，对应的是反向旋转
    # 对于 (0, 1, 0) 轴，正角度通常是逆时针(向左)。
    # qinv 意味着我们要用负角度公式，或者交换 sin/cos 的符号
    
    c = torch.cos(real_angle)
    s = torch.sin(real_angle)
    
    # 再次模仿 GT 的错位：第 0 帧位置不变，第 1 帧位置由 data[0] 决定
    vel_x_shifted = torch.zeros_like(local_vel_x)
    vel_z_shifted = torch.zeros_like(local_vel_z)
    vel_x_shifted[..., 1:] = local_vel_x[..., :-1]
    vel_z_shifted[..., 1:] = local_vel_z[..., :-1]
    
    # 【核心修正 2】: 适配 qinv 的旋转方向 (GT向左弯，对应 -X)
    # 公式推导：
    # X_global = x_local * cos - z_local * sin
    # Z_global = x_local * sin + z_local * cos
    global_vel_x = vel_x_shifted * c - vel_z_shifted * s
    global_vel_z = vel_x_shifted * s + vel_z_shifted * c
    
    # 4. 积分得到位置
    pred_pos = torch.zeros_like(data[..., :3])
    pred_pos[..., 0] = torch.cumsum(global_vel_x, dim=-1)
    pred_pos[..., 2] = torch.cumsum(global_vel_z, dim=-1)
    pred_pos[..., 1] = data[..., 3] # 高度
    
    return pred_pos

def debug_plot_trajectory(target_traj, pred_traj, target_pos=None, save_path="debug_traj.png", obstacle_info=None):
    """
    绘制轨迹对比图 (XZ 平面俯视图)
    target_traj: [Length, 3] numpy array (Global Target)
    pred_traj:   [Length, 3] numpy array (Generated Root)
    
    """
    plt.figure(figsize=(10, 10))
    
    # 1. 画目标轨迹 (红色虚线)
    if target_traj is not None:
        plt.plot(target_traj[:, 0]-target_traj[0, 0], target_traj[:, 2]-target_traj[0,2], 'r--', linewidth=2, label='Target (Guidance)')
        plt.scatter(target_traj[0, 0], target_traj[0, 2], c='red', marker='x', s=100, label='Target Start')

    # 2. 画生成轨迹 (蓝色实线)
    plt.plot(pred_traj[:, 0], pred_traj[:, 2], 'b-', linewidth=2, label='Generated (Output)')
    plt.scatter(pred_traj[0, 0], pred_traj[0, 2], c='blue', marker='o', s=100, label='Gen Start')

    target_pos_np = target_pos.detach().cpu().numpy()
    # 点一个target pos的点
    plt.scatter(target_pos_np[0], target_pos_np[2], c='red',marker='x', s=100, label="target pos")

    # 画障碍物
    if obstacle_info is not None:
        circle = plt.Circle(
            (obstacle_info['center'][0], obstacle_info['center'][1]), 
            obstacle_info['radius'], 
            color='black', alpha=0.3, label='Obstacle'
        )
        plt.gca().add_patch(circle)
    
    # 3. 标注终点
    plt.scatter(pred_traj[-1, 0], pred_traj[-1, 2], c='blue', marker='^', s=100)
    
    plt.title("Trajectory Control Check (Top-down View)")
    plt.xlabel("X Position (meters)")
    plt.ylabel("Z Position (meters)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # 保证圆看起来是圆的
    
    # 保存
    plt.savefig(save_path)
    plt.close()
    print(f"[Debug] Trajectory plot saved to {save_path}")

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


        self.traj_processor = TrajectoryProcessor()




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

    def compute_obstacle_guidance(self, latents, t, obstacle_info, encoder_hidden_states, lengths):
        """
        计算避障的斥力梯度
        obstacle_info: {'center': [x, z], 'radius': r}
        """
        with torch.enable_grad():
            latents = latents.detach().requires_grad_(True)
            
            # 1. 预测 & 反推 z0 (标准流程)
            noise_pred = self.denoiser(
                sample=latents,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths,
            )[0]
            
            alpha_prod_t = self.scheduler.alphas_cumprod[t[0].item()]
            beta_prod_t = 1 - alpha_prod_t
            pred_z0 = (latents - beta_prod_t ** 0.5 * noise_pred) / (alpha_prod_t ** 0.5)
            
            # 2. Decode & 反归一化 (必须在物理空间做避障！)
            pred_z0_input = pred_z0.permute(1, 0, 2)
            fake_lengths = [lengths[0]] * latents.shape[0]
            pred_motion_norm = self.vae.decode(pred_z0_input, fake_lengths)
            
            # 反归一化
            if self.mean.device != latents.device:
                self.mean = self.mean.to(latents.device)
                self.std = self.std.to(latents.device)
            pred_motion = pred_motion_norm * self.std + self.mean
            
            # 3. 积分得到轨迹 (XZ平面)
            pred_rot_vel = pred_motion[..., 0]
            pred_vel_x = pred_motion[..., 1]
            pred_vel_z = pred_motion[..., 2]
            
            pred_rot = torch.cumsum(pred_rot_vel, dim=1)
            global_vel_x = pred_vel_x * torch.cos(pred_rot) - pred_vel_z * torch.sin(pred_rot)
            global_vel_z = pred_vel_x * torch.sin(pred_rot) + pred_vel_z * torch.cos(pred_rot)
            
            pred_pos_x = torch.cumsum(global_vel_x, dim=1)
            pred_pos_z = torch.cumsum(global_vel_z, dim=1)
            
            # 归零起点 (消除初始偏差)
            pred_pos_x = pred_pos_x - pred_pos_x[:, 0:1]
            pred_pos_z = pred_pos_z - pred_pos_z[:, 0:1]
            
            # 4. 计算避障 Loss (Repulsive Loss)
            obs_center = torch.tensor(obstacle_info['center'], device=latents.device)
            obs_radius = obstacle_info['radius']
            safety_margin = 0.4  # 安全余量 0.4米
            effective_radius = obs_radius + safety_margin
            
            # 计算每一帧到圆心的距离
            # pred_pos_x: [B, L]
            dist_to_obs = torch.sqrt((pred_pos_x - obs_center[0])**2 + (pred_pos_z - obs_center[1])**2)
            
            # 核心 Loss: 只惩罚进入半径的点
            # ReLU: 小于0变0，大于0保留
            # 我们希望 dist > radius -> radius - dist < 0 -> loss = 0
            # 我们希望 dist < radius -> radius - dist > 0 -> loss > 0
            # collision_loss = torch.nn.functional.relu(obs_radius - dist_to_obs).mean()
            penalty = torch.nn.functional.relu(effective_radius - dist_to_obs)
            collision_loss = (penalty ** 2).sum()
            
            # 5. 求导
            if collision_loss > 1e-6:
                grad = torch.autograd.grad(collision_loss, latents)[0]
            else:
                grad = torch.zeros_like(latents)
                
            return grad
    
    def compute_keypoint_guidance(self, latents, t, target_joint_idx, target_frame_idx, target_pos, encoder_hidden_states, lengths):
        """
        [新] 关键点引导函数
        target_joint_idx: 整数, e.g., 11 (右手腕)
        target_frame_idx: 整数, e.g., 99 (第100帧)
        target_pos: [3] Tensor, e.g., [0.5, 1.2, 0.8]
        """
        # print("now in compute_keypoint")
        with torch.enable_grad():
            latents = latents.detach().requires_grad_(True)
            
            # 1. 预测噪声 & 反推 z0
            noise_pred = self.denoiser(
                sample=latents,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths,
            )[0]
            
            alpha_prod_t = self.scheduler.alphas_cumprod[t[0].item()]
            beta_prod_t = 1 - alpha_prod_t
            pred_z0 = (latents - beta_prod_t ** 0.5 * noise_pred) / (alpha_prod_t ** 0.5)
            
            pred_z0_input = pred_z0.permute(1, 0, 2)
            
            # 2. VAE Decode (得到归一化特征)
            # 注意：这里的长度需要是完整的
            full_lengths = [lengths[0]] * latents.shape[0] if lengths is not None else [pred_z0_input.shape[0]] * latents.shape[0]
            pred_motion_norm = self.vae.decode(pred_z0_input, full_lengths)
            
            # 3. 反归一化 (得到物理特征)
            pred_motion = pred_motion_norm * self.std.to(latents.device) + self.mean.to(latents.device)
            
            # 4. 【核心】正向运动学 (FK)
            # recover_from_ric 输入 [B, L, 263]，输出 [B, L, 22, 3]
            # 这是将 263 维特征还原成全局 XYZ 坐标的关键函数
            # 你需要确保你的项目里有这个函数，或者类似功能的函数
            # 我们之前在 debug 脚本里用过它！
            all_joint_positions = recover_from_ric(pred_motion, 22) # 假设是 22 个关节

            # 5. 提取目标关节在目标帧的位置
            # [Batch, Frames, Joints, 3]
            pred_joint_pos = all_joint_positions[:, target_frame_idx, target_joint_idx, :] # torch.Size([1, 3])
            target_pos = target_pos.unsqueeze(0)
            pred_joint_pos[:, 1] = 0
            target_pos[:, 1] = 0
            
            # 6. 计算 Loss (简单的 MSE)
            loss = F.mse_loss(pred_joint_pos, target_pos.to(latents.device)) # 
            
            # 7. 求导
            grad = torch.autograd.grad(loss, latents)[0]
            
            return grad

    def estimate_speed_from_motion(self, motion_features):
        """
        从 HumanML3D 的 263 维特征中估算平均速度。
        motion_features: [Batch, Frames, 263] Tensor
        """
        # 1. 提取局部速度特征
        # Index 1: Root Linear Vel X (local)
        # Index 2: Root Linear Vel Z (local)
        root_vel_x = motion_features[..., 1] # [B, L]
        root_vel_z = motion_features[..., 2] # [B, L]

        height = motion_features[..., 3]  # Root Height (Y轴位置)，暂时不用
        
        # 2. 计算每帧的速度大小 (模长)
        # speed = sqrt(vx^2 + vz^2)
        # 这里的速度是局部坐标系下的，但对于“向前走”的动作，
        # 它约等于全局速度的大小。
        per_frame_speed = torch.sqrt(root_vel_x**2 + root_vel_z**2) # [B, L]
        
        # 3. 计算整个序列的平均速度
        # 我们只计算 Batch 中每个样本的平均值
        # .mean(dim=1) 会在时间维度上求平均
        avg_speed_per_sample = per_frame_speed.mean(dim=1) # [B]
        
        # 4. 安全性处理: 防止速度为0或负数
        # 如果一个动作是原地不动，给一个很小的默认速度
        avg_speed_per_sample = torch.clamp(avg_speed_per_sample, min=0.01)
    
        # 返回一个 [Batch] 大小的 Tensor，每个元素是对应样本的平均速度
        return avg_speed_per_sample, height
    
    def augment_content_rotation(self, features, angle_range):
        """
        对 Content Motion 进行随机的 Y 轴旋转增强。
        features: [Batch, Frames, 263] 原来数据集动作的263维特征向量
        """
        device = features.device
        bs, frames, dims = features.shape
        
        # 限制旋转范围在 -90 到 +90 度之间 (更稳妥)
        angle_range = (angle_range / 180.0) * torch.pi 
        thetas = (torch.rand(bs, device=device) * 2 - 1) * angle_range 
        
        c = torch.cos(thetas)
        s = torch.sin(thetas)
        
        # 旋转 Root Linear Velocity (Indices 1, 2)
        root_vx = features[..., 1]
        root_vz = features[..., 2]
        new_root_vx = root_vx * c.view(bs, 1) - root_vz * s.view(bs, 1)
        new_root_vz = root_vx * s.view(bs, 1) + root_vz * c.view(bs, 1)
        features[..., 1] = new_root_vx
        features[..., 2] = new_root_vz
        
        # 旋转 Local Joint Positions (Indices 4 ~ 67)
        joint_pos = features[..., 4:67].reshape(bs, frames, 21, 3)
        pos_x = joint_pos[..., 0]
        pos_z = joint_pos[..., 2]
        new_pos_x = pos_x * c.view(bs, 1, 1) - pos_z * s.view(bs, 1, 1)
        new_pos_z = pos_x * s.view(bs, 1, 1) + pos_z * c.view(bs, 1, 1)
        joint_pos[..., 0] = new_pos_x
        joint_pos[..., 2] = new_pos_z
        features[..., 4:67] = joint_pos.reshape(bs, frames, -1)
        
        # 旋转 Local Joint Velocities (Indices 193 ~ 259)
        joint_vel = features[..., 193:259].reshape(bs, frames, 22, 3)
        vel_x = joint_vel[..., 0]
        vel_z = joint_vel[..., 2]
        new_vel_x = vel_x * c.view(bs, 1, 1) - vel_z * s.view(bs, 1, 1)
        new_vel_z = vel_x * s.view(bs, 1, 1) + vel_z * c.view(bs, 1, 1)
        joint_vel[..., 0] = new_vel_x
        joint_vel[..., 2] = new_vel_z
        features[..., 193:259] = joint_vel.reshape(bs, frames, -1)
        
        return features
    
    def generate_custom_trajectory(self, batch_size, length, estimated_speed, estimated_height,  shape_type='circle', device='cuda'):
        """
        生成两样东西：
        1. trans_cond: [B, L, 4] -> (RotVel, VelX, VelZ, PosY)，这是喂给 DiT 的条件
        2. target_global_pos: [B, L, 3] -> (GlobalX, GlobalY, GlobalZ)，这是计算 Loss 的目标
        """
        estimated_speed = estimated_speed
        speed = estimated_speed.view(-1, 1) # [B, 1]
        radius = 2.5
        # 初始化
        trans_cond = torch.zeros((batch_size, length, 4), device=device)
        target_global_pos = torch.zeros((batch_size, length, 3), device=device)
        
        
        if shape_type == 'circle':
            # 这里的 RotVel 是 Y轴角速度
            # omega = v / r
            angular_velocity = speed / radius # [B, 1]
            
            # 广播赋值: trans_cond[..., 0] 是 [B, L]，angular_velocity 是 [B, 1]
            trans_cond[..., 0] = angular_velocity
            trans_cond[..., 2] = speed.squeeze(1).unsqueeze(1).expand(-1, length) # 确保维度正确
            trans_cond[..., 3] = estimated_height  # 0.95
              
        elif shape_type == 'line':
            # ... (直线逻辑同理修改) ...
            trans_cond[..., 0] = 0.0
            trans_cond[..., 2] = speed.squeeze(1).unsqueeze(1).expand(-1, length)
            trans_cond[..., 3] = estimated_height
        elif shape_type == 'line_left':
            # 1. 计算角速度
            angular_velocity = speed / radius # 结果通常为 [B, 1]
            
            # 2. 定义转弯的转折点（例如前 30 帧）
            turn_len = min(30, length) 
            
            # --- 处理前 turn_len 帧：圆周运动 (左转) ---
            # 索引 0: 角速度 (Angular Velocity)
            trans_cond[:, :turn_len, 0] = angular_velocity.expand(-1, turn_len)
            # 索引 2: 线速度 (Forward Speed)
            trans_cond[:, :turn_len, 2] = speed.expand(-1, turn_len)
            
            # --- 处理剩余帧：直线运动 ---
            if length > turn_len:
                # 索引 0: 角速度归零
                trans_cond[:, turn_len:, 0] = 0.0
                # 索引 2: 保持线速度
                trans_cond[:, turn_len:, 2] = speed.expand(-1, length - turn_len)
            
            # 3. 设置高度 (通常是索引 3)
            trans_cond[..., 3] = estimated_height

        
        target_global_pos = calculate_trajectory_correct(trans_cond)
        
        # 归一化处理
        mean = self.mean.to(device)
        std = self.std.to(device)
        
        mean_cond = mean[..., :4]
        std_cond = std[..., :4]
        
        trans_cond_norm = (trans_cond - mean_cond) / std_cond

        return trans_cond_norm, target_global_pos
    
    def compute_spatial_guidance(self, latents, t, target_global_pos, encoder_hidden_states, lengths):
        with torch.enable_grad():
            latents = latents.detach().requires_grad_(True)
            
            # 1. 预测 & 反推 (保持不变)
            noise_pred = self.denoiser(
                sample=latents,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths,
            )[0]
            
            alpha_prod_t = self.scheduler.alphas_cumprod[t[0].item()]
            beta_prod_t = 1 - alpha_prod_t
            pred_z0 = (latents - beta_prod_t ** 0.5 * noise_pred) / (alpha_prod_t ** 0.5)
            
            # 2. Decode & 反归一化 (保持不变)
            pred_z0_input = pred_z0.permute(1, 0, 2) 
            fake_lengths = [target_global_pos.shape[1]] * latents.shape[0]
            pred_motion_norm = self.vae.decode(pred_z0_input, fake_lengths)
            
            if self.mean.device != latents.device:
                self.mean = self.mean.to(latents.device)
                self.std = self.std.to(latents.device)
            pred_motion = pred_motion_norm * self.std + self.mean
            
            # 3. 积分得到物理轨迹 (使用修正后的 calculate_trajectory_correct)
            # calculate_pos: [Batch, Seq, 3]
            calculate_pos = calculate_trajectory_correct(pred_motion)
            
            # ================== 【修改点 1: 坐标系对齐】 ==================
            # 我们不关心绝对坐标，只关心相对形状。
            # 让生成轨迹和目标轨迹的第0帧都归零。
            # 这样消除了“起点不一致”带来的巨大 Loss。
            pred_traj_centered = calculate_pos - calculate_pos[:, 0:1, :]
            target_traj_centered = target_global_pos - target_global_pos[:, 0:1, :]
            
            # ================== 【修改点 2: 稀疏关键点约束】 ==================
            # 不要每一帧都算 Loss，那是强迫症。
            # 每隔 interval 帧算一次，给模型“呼吸”的空间，缓解滑步。
            interval = 20  # 每 1 秒检查一次 (假设 20fps)
            
            # 生成索引: [0, 20, 40, ..., last_frame]
            seq_len = calculate_pos.shape[1]
            key_indices = torch.arange(0, seq_len, interval, device=latents.device)
            
            # 确保最后一帧也被包含进去（终点很重要）
            if key_indices[-1] != seq_len - 1:
                key_indices = torch.cat([key_indices, torch.tensor([seq_len-1], device=latents.device)])
            
            # 只取关键点计算 MSE
            loss = F.mse_loss(
                pred_traj_centered[:, key_indices, :], 
                target_traj_centered[:, key_indices, :]
            )
            
            # 4. 求导
            grad = torch.autograd.grad(loss, latents)[0]
            
            return grad

# test
    def forward(self, batch, scene_data=None): # 这个是本来的推理的代码

        lengths = batch["length"]
        # style
        motion = batch["style_motion"].clone()
        motion[...,:3] = 0


        # content
        content_motion = batch['content_motion']

        # avg_speed, pelvis_height = self.estimate_speed_from_motion(content_motion) # 此时速度还没有归一化
        # print(f"Estimated avg speed: {avg_speed.mean().item():.4f} m/frame")
        
        content_motion = (content_motion - self.mean.to(content_motion.device))/self.std.to(content_motion.device)
        if self.cfg.TRAJECTORY.ROOT_MASKING_DIM4:
            content_motion[...,:4] = 0
        else:
            trans_motion = content_motion.clone()
            content_motion[...,:3] = 0
            


        scale = batch["tag_scale"]
        lengths1 = [content_motion.shape[1]]* content_motion.shape[0]

        # # -----------------------------------------------------------
        # # 【修改点 1】: 强制替换轨迹为自定义形状 (验证解耦性)
        # # -----------------------------------------------------------
        # custom_shape = 'circle'      # 'circle' or 'line'
        
        # # 直接完全遵循轨迹，效果很差，但也可以试试
        # if self.cfg.TRAJECTORY.GUIDANCE.MODE in ['strict_track', 'mixed', 'waypoints'] and self.cfg.TRAJECTORY.ENABLED:
        #     # 生成自定义轨迹特征 [B, L, 4] 和 全局目标 [B, L, 3]
        #     # 注意长度要和 content 保持一致 (lengths[0])
        #     custom_trans_cond, target_global_pos = self.generate_custom_trajectory(
        #         batch_size=content_motion.shape[0], 
        #         length=content_motion.shape[1], 
        #         estimated_speed=avg_speed,
        #         estimated_height=pelvis_height,
        #         shape_type=custom_shape,
        #         device=content_motion.device,
        #     )
        #     # 替换 trans_motion 的前4维
        #     # 构造一个符合 Trajectory Encoder 输入的 tensor
        #     # 假设 Trajectory Encoder 需要 [B, L, 4]
        #     trans_cond_input = custom_trans_cond
        # ==========================================
        # 🔥 核心接入点：轨迹生成 🔥
        # ==========================================
        target_global_pos = None # 用于 Guidance
        trans_cond_norm = None   # 用于 Denoiser Input
        
        if scene_data is not None and self.cfg.TRAJECTORY.ENABLED:
            raw_content = batch['content_motion'].clone()  # 使用未归一化的数据
            # A. 计算 Content 距离 Profile
            # raw_content[0] 取 Batch 中第一个样本作为参考
            content_npy = raw_content[0].detach().cpu().numpy()
            dist_profile = self.traj_processor.calculate_cumulative_distance(content_npy) # (39,)
            
            # B. 生成避障路径 (A*)
            waypoints = scene_data['trajectory']['points']
            obstacles = scene_data['environment']['obstacles']
            dense_curve = self.traj_processor.generate_collision_free_path(waypoints, obstacles) #shape:(200, 2)
            
            # C. 重采样 (对齐 Content 长度)
            # dist_profile[1:] 对应每一帧结束时的距离
            target_dists = dist_profile[1:] 
            # 确保长度对齐
            target_dists = target_dists[:lengths[0]]
            
            resampled_pts, _ = self.traj_processor.resample_by_arc_length(dense_curve, target_dists)
            
            # D. 计算 4维特征 (RotVel, VelX, VelZ, Height)
            trans_cond_phys, _ = self.traj_processor.compute_trajectory_features(resampled_pts)
            
            # E. 构造 Target Global Pos (用于 Guidance)
            # 补上 Y 轴 (从 Content 或默认值)
            # resampled_pts is [L, 2] (x, z)
            # target_global_pos needs [B, L, 3]
            frames = len(resampled_pts)
            bs = content_motion.shape[0]
            
            # 构造 [L, 3]
            traj_3d = np.zeros((frames, 3))
            traj_3d[:, 0] = resampled_pts[:, 0]
            traj_3d[:, 2] = resampled_pts[:, 1]
            traj_3d[:, 1] = 0.0 # 或者 raw_content[0, :frames, 3].cpu().numpy() (高度)
            
            # 转 Tensor 并广播到 Batch
            target_global_pos = torch.from_numpy(traj_3d).float().to(self.device)
            target_global_pos = target_global_pos.unsqueeze(0).repeat(bs, 1, 1) # torch.Size([1, 38, 3])
            
            # F. 构造 Trans Cond (用于 Denoiser) 并归一化
            # trans_cond_phys is [L, 4]
            trans_tensor = torch.from_numpy(trans_cond_phys).float().to(self.device)
            trans_tensor = trans_tensor.unsqueeze(0).repeat(bs, 1, 1)
            
            # 归一化 (使用 Dataset 的 mean/std)
            # mean/std 是 [1, 263] -> 取前4维
            mean_cond = self.mean.to(self.device)[..., :4]
            std_cond = self.std.to(self.device)[..., :4]
            trans_cond_input = (trans_tensor - mean_cond) / std_cond # torch.Size([1, 38, 4])
        else:
            # 原有逻辑: 从 content clone 也就是所谓的 "trans_motion"
            trans_motion = batch['content_motion'].clone()  # torch.Size([1, 38, 263])
            if self.cfg.TRAJECTORY.ROOT_MASKING_DIM4:
                trans_cond_input = trans_motion[..., :4]
            else:
                trans_cond_input = trans_motion[..., :3]  # torch.Size([1, 38, 3])
            target_global_pos = None # 没有目标，不做 Guidance

        # -----------------------------------------------------------
        
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
            # trans_cond = trans_motion[...,:3]
            # uncond_trans = torch.cat([trans_cond, trans_cond], dim = 0)
            uncond_trans = torch.cat([trans_cond_input, trans_cond_input], dim=0) # [2*B, L, 4]

            # three conditions
            multi_cond_emb = [motion_emb_content, motion_emb, uncond_trans]


            z = self._diffusion_reverse(multi_cond_emb, lengths, scale, target_global_pos)

        elif self.stage in ['vae']:
            motions = batch['motion']
            z, dist_m = self.vae.encode(motions, lengths)

        with torch.no_grad():
            feats_rst = self.vae.decode(z, lengths)
            # feats_rst[...,:3] = trans_motion[...,:3] # if copy trajectory

        joints = self.feats2joints(feats_rst.detach().cpu()) # torch.Size([1, 38, 22, 3])

        # ================= [新增] 埋点可视化逻辑 =================
        # 只画 Batch 中的第 0 个样本
        if True: # 可以改成 if self.cfg.TEST.DEBUG_PLOT:
            try:
                # 1. 提取生成的根节点轨迹
                # 假设 joints 维度是 [Batch, 22, 3, Length] torch.Size([1, 38, 22, 3])
                # Root joint 通常是 index 0
                if joints.shape[1] == 22 or joints.shape[1] == 21: # [B, J, 3, L]
                    pred_root_traj = joints[0, 0, :, :].permute(1, 0) # [3, L] -> [L, 3]
                else:
                    # 如果维度不一样，打印出来看看
                    print(f"Joints shape check: {joints.shape}")
                    # 尝试自适应: 假设第0维是Batch，包含3的那一维是坐标
                    pred_root_traj = joints[0, ..., 0, :].squeeze() # [38, 3] 世界空间的

                pred_root_traj_np = pred_root_traj.detach().cpu().numpy()
                
                # 2. 提取目标轨迹
                # 之前生成的 target_global_pos 是 [Batch, Length, 3]
                target_traj_np = None
                if 'target_global_pos' in locals() and target_global_pos is not None:
                    target_traj_np = target_global_pos[0].detach().cpu().numpy()
                
                # 3. 这里的长度可能不一致 (VAE 下采样 vs 原始长度)
                # 简单的截断或补齐，为了画图对齐
                min_len = min(len(pred_root_traj_np), len(target_traj_np)) if target_traj_np is not None else len(pred_root_traj_np)
                
                # 4. 调用画图
                # name 是当前时间戳
                name = int(time.time())
                debug_plot_trajectory(
                    target_traj_np[:min_len] if target_traj_np is not None else None, 
                    pred_root_traj_np[:min_len],
                    self.target_pos,
                    # save_path=f"vis_debug/traj_step_n.png"
                    # 每个样本给个不同的文件名
                    save_path=f"vis_debug/traj_debug_{name}.png",
                    obstacle_info=self.obstacle_info
                )
            except Exception as e:
                print(f"[Warning] Failed to plot trajectory: {e}")
        # =======================================================

        return remove_padding(joints, lengths)
    


    def _diffusion_reverse(self, encoder_hidden_states, lengths=None, scale=None, target_global_pos=None):
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
        
        # 【新增】定义我们的引导目标
        use_guidance = self.cfg.TRAJECTORY.GUIDANCE.ENABLED
        use_keypoint_guidance = True
        target_joint_idx = 0  # pelvis
        target_frame_idx = lengths[0] - 1  # 第 100 帧
        self.target_pos = torch.tensor([-3.5, 0.0, 0.0]) # 目标位置
        # 2. 障碍物 (斥力) - 挡在中间
        self.obstacle_info = {
            'center': [0.0, 2.0], # 在 X=0, Z=2 处
            'radius': 0.8         # 半径 0.8米
        }
        self.obstacle_info = None

        # reverse
        for i, t in enumerate(timesteps):

            # ================= [新增: 动态 K 值策略] =================
            # 早期 (t > 500): 只优 1 次 (避免被噪声带偏)
            # 晚期 (t <= 500): 优化 5 次 (强力贴合)
            # 冲刺期 (t <= 100): 优化 10 次 (确保摸到球)
            if t > 500:
                num_opt_steps = 1
            elif t > 100:
                num_opt_steps = 5
            else:
                num_opt_steps = 10

            # -----------------------------------------------------------
            # 【修改点 3】: Spatial Guidance 梯度回传
            # -----------------------------------------------------------
            # 只有当提供了 target 且 在某些步骤（比如前50%）才做，为了省时间
            # 或者是全程做 (精度最高)
            # 1. 【时间调度】: 刚开始(t>600)全是噪声，算出来的几何梯度是不可信的，别乱导！
            # 只有当 t < 600 (动作轮廓大概出来后) 再开始引导
            for k in range(num_opt_steps):
                total_grad = torch.zeros_like(latents).to(latents.device)
                # if target_global_pos is not None:
                if use_guidance and self.cfg.TRAJECTORY.GUIDANCE.WAYPOINTS_MODE:
                    way_guidance_scale = self.cfg.TRAJECTORY.GUIDANCE.WAYPOINTS_GUIDE_STRENGTH
                    if t < 1001: 
                        # 计算梯度
                        grad = self.compute_spatial_guidance(
                            latents, 
                            t.unsqueeze(0).repeat(bsz), # expand t
                            target_global_pos, 
                            [h[bsz:] for h in encoder_hidden_states],  # [uncond, cond] -> [cond]
                            lengths
                        )

                        grad = grad * way_guidance_scale

                        # 2. 梯度裁剪 (保持你现在的逻辑，非常稳)
                        grad_norm = grad.norm()
                        max_grad_norm = 5.0 # 或者根据 t 动态调整
                        if grad_norm > max_grad_norm:
                            scale_factor = max_grad_norm / (grad_norm + 1e-8)
                            grad = grad * scale_factor
                        
                        # 3. 更新 Latents
                        # 这里的 step_size 可以小一点，因为我们跑很多次
                        step_size = 1.0 
                        latents = latents - step_size * grad
                        latents = latents.detach().requires_grad_(True) # 记得 detach 并重新开启梯度追踪

                        print(f"Step {t.item()} Inner {k}: Loss:", grad.norm())

                    # grad = self.compute_keypoint_guidance(
                    #     latents, 
                    #     t.unsqueeze(0).repeat(bsz), 
                    #     target_joint_idx, 
                    #     target_frame_idx, 
                    #     self.target_pos,
                    #     [h[bsz:] for h in encoder_hidden_states], 
                    #     lengths
                    # )
                    
                    
                    # 【Debug】打印梯度大小，如果这个值很大(>1)，说明力度太大了
                    # print(f"Step {t.item()} Grad Norm: {grad.norm().item()}")
                    
                    # 【修复】更安全的更新方式：梯度归一化
                    # 无论梯度多大，我们限制它每次只改动一点点方向
                    # grad_norm = grad.norm()
                    # if grad_norm > 1e-8: # 防止除以0
                    #     grad = grad / grad_norm 
                    
                    # 3. 【力度控制】: Latent Space 的步长必须非常小！
                    # 推荐值: 1.0 ~ 3.0。绝对不要超过 5.0。
                    # 随着 t 变小，动作越来越确定，可以稍微加大一点力度
                    # if t > 100:
                    #     current_scale = 10.0
                    # else:
                    #     current_scale = 10.0 # 最后冲刺阶段稍微用力一点

                    # 打印 Log 观察 (必须看到 Scale 后的值是可控的)
                    # print(f"Step {t.item()} | Raw Norm: {grad_norm:.2f} | Update Norm: {current_scale}")

                    # 然后再乘一个步长，这个步长对应 Latent Space 的距离
                    # 建议从小一点开始试，比如 20.0，甚至 5.0
                    # latents = latents - current_scale * grad  # 这里一定是-
                    # latents = latents.detach()
                    # total_grad += way_guidance_scale * grad
            
            # 2. 障碍物斥力 (Training-free 核心)
            if use_guidance and t < 1000 and (self.obstacle_info is not None) and False: # 全程避障
                grad_repel = self.compute_obstacle_guidance(
                    latents, 
                    t.view(-1).repeat(bsz), 
                    self.obstacle_info, 
                    # 这里的 hidden_states 传入 cond 部分
                    [h[bsz:] for h in encoder_hidden_states] if self.do_classifier_free_guidance else encoder_hidden_states,
                    lengths
                )
                
                # 避障通常需要更大的力度，因为它是一堵墙
                repel_strength = 100.0 # 这里的力度要大！
                
                # 累加梯度
                total_grad += repel_strength * grad_repel
                
                # 打印一下看看有没有撞到
                if i % 10 == 0 and grad_repel.norm() > 0:
                    print(f"Step {t.item()} | Collision Detected! Repelling...")

            # 3. 更新 Latents
            # 梯度归一化 (可选，建议做)
            if use_guidance and False:
                # print("origin grad norm:", total_grad.norm(), "timestep: ", t)
                # 设定一个最大步长 (Max Step Size)
                max_grad_norm = 5.0  # 这个值根据经验调，2.0~10.0 比较合理

                grad_norm = total_grad.norm() # 向量的模长

                # 【关键逻辑】: 只有当梯度超过阈值时，才进行缩放；否则保持原样
                if grad_norm > max_grad_norm:
                    # 缩放比例 = 目标 / 当前
                    scale_factor = max_grad_norm / (grad_norm + 1e-8)
                    total_grad = total_grad * scale_factor
                    # 此时模长被压回了 5.0
                else:
                    # 如果梯度很小 (比如 0.1)，就保持 0.1
                    # 这样模型在接近目标时会自然减速收敛，不会震荡
                    pass

                latents = latents - total_grad
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
            "noise_pred": noise_pred, # torch.Size([32, 7, 256]) 训练的时候预测噪声就好了，没必要去噪
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
        feats_ref = batch["motion"] # torch.Size([32, 40, 263])，原始动作，轨迹也是原始动作的轨迹
        feats_content = batch["motion"].clone() # torch.Size([32, 40, 263])， 这是content
        # 1. 旋转增强 (根据 Config 开关)
        if self.cfg.TRAJECTORY.AUGMENTATION.ENABLED:
            angle = self.cfg.TRAJECTORY.AUGMENTATION.ROTATION_RANGE
            feats_content = self.augment_content_rotation(feats_content, angle)  # torch.Size([32, 40, 263])

        if self.cfg.TRAJECTORY.ROOT_MASKING_DIM4:
            feats_content[...,0:4] = 0.0 # 修改1：把Y的位置也去掉
        else:
            feats_content[...,:3] = 0.0
        lengths = batch["length"]

        if self.cfg.TRAJECTORY.USE_CONTENT_DROPOUT:
            # print("Content Dropout Enabled")
            # 添加对content的随机dropout处理，这样模型就可以去学习轨迹了
            content_drop_prob = self.cfg.TRAJECTORY.CONTENT_DROPOUT_PROB
            bsz = feats_content.shape[0]
            mask_content_drop = torch.rand(bsz, device=feats_content.device) < content_drop_prob  # 几乎都是False，偶尔有True
        
        # content condition
        with torch.no_grad():
            z, dist = self.vae.encode(feats_ref, lengths) # z（没去除轨迹，给他加噪声）:torch.Size([7, 32, 256]), dist: torch.Size([7, 32, 256])
            z_content, dist = self.vae.encode(feats_content, lengths) # z_content（纯动作内容，没轨迹，如果是我们的（开了旋转增强，随机屏蔽50%））是条件，concat到z上，不加噪声，作为强条件
            cond_emb = z_content.permute(1,0,2)  # torch.Size([32, 7, 256])    
            if self.cfg.TRAJECTORY.USE_CONTENT_DROPOUT:
                cond_emb[mask_content_drop] = 0.0       
        # style condition：一行没改，本来MotionCLIP（style encoder）也是冻结的，batch里的东西都是归一化的，MotionCLIP吃的动作是反归一化的空间的
        motion_seq = feats_ref*self.std + self.mean 
        motion_seq[...,:3]=0.0
        motion_seq = motion_seq.unsqueeze(-1).permute(0,2,3,1) # torch.Size([32, 263, 1, 40])
        motion_emb = self.motionclip.encoder({'x': motion_seq,
                        'y': torch.zeros(motion_seq.shape[0], dtype=int, device='cuda:{}'.format(self.cfg["DEVICE"][0])),
                        'mask': lengths_to_mask(lengths, device='cuda:{}'.format(self.cfg["DEVICE"][0]))})["mu"] # 一个style被提取成了512维的tensor，torch.Size([32, 512])
        motion_emb = motion_emb.unsqueeze(1) # torch.Size([32, 1, 512])
        mask_uncond = torch.rand(motion_emb.shape[0]) < self.guidance_uncodp # [F,F,...,F,T,F,T,...]
        motion_emb[mask_uncond, ...] = 0 # style为什么要随机置空？为了做CFG
        


        # trans condition
        if self.cfg.TRAJECTORY.ROOT_MASKING_DIM4:
            trans_cond = batch["motion"][...,:4]  # torch.Size([32, 40, 4])
        else:
            trans_cond = batch["motion"][...,:3]  # torch.Size([32, 40, 3])， 训练的时候轨迹是归一化的！！自然推理的时候轨迹也要归一化（与训练的策略保持一致）
        # three condition
        multi_cond_emb = [cond_emb, motion_emb, trans_cond] # 复习一下： cond_emb：内容（torch.Size([32, 7, 256])），motion_emb：风格（torch.Size([32, 1, 512])），trans_cond：轨迹（torch.Size([32, 40, 4])）


        # diffusion process return with noise and noise_pred
        n_set = self._diffusion_process(z, multi_cond_emb, lengths) # 返回的n_set是一个字段，包含计算loss的时候pytorch_lightning所关心的内容
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

        # return forward output rather than loss during test
        if split in ["test"]:
            return rs_set["joints_rst"], batch["length"]
        # tensorboard log一下loss
        self.log_dict(
            {f"{split}_loss": loss},
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch["motion"].shape[0], # 这个参数是用来做平均的
        )
        return loss
