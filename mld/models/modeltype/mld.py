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

def calculate_trajectory_correct(data):
    """
    修正后的积分逻辑
    data: [Batch, Seq, 4] (RotVel, VelX, VelZ, Height)
    """
    # 1. 提取特征
    rot_vel = data[..., 0]
    local_vel_x = data[..., 1]
    local_vel_z = data[..., 2]
    
    # 2. 积分角度
    # r_rot_ang[i] 代表第 i 帧相对于第 0 帧的旋转角
    r_rot_ang = torch.zeros_like(rot_vel)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)
    
    real_angle = r_rot_ang * 2.0 # Quaternion mapping
    
    c = torch.cos(real_angle)
    s = torch.sin(real_angle)
    
    # 3. 准备世界坐标速度
    # 依然保持 shift，因为特征是对上一帧的 delta
    vel_x_shifted = torch.zeros_like(local_vel_x)
    vel_z_shifted = torch.zeros_like(local_vel_z)
    vel_x_shifted[..., 1:] = local_vel_x[..., :-1]
    vel_z_shifted[..., 1:] = local_vel_z[..., :-1]
    
    # 旋转投影
    # HumanML3D/T2M GPT 使用的是 (vel_x * c - vel_z * s, vel_x * s + vel_z * c)
    # 对应逆时针旋转
    global_vel_x = vel_x_shifted * c - vel_z_shifted * s
    global_vel_z = vel_x_shifted * s + vel_z_shifted * c
    
    # 4. 积分位置
    pred_pos = torch.zeros_like(data[..., :3])
    pred_pos[..., 0] = torch.cumsum(global_vel_x, dim=-1)
    pred_pos[..., 2] = torch.cumsum(global_vel_z, dim=-1)
    
    # 【强制对齐】：确保第 0 帧一定是 (0,0,0)，消除任何累积误差的初始偏移
    # 这样 guidance 计算 diff 时，起点永远是对齐的
    pred_pos = pred_pos - pred_pos[:, 0:1, :]
    
    pred_pos[..., 1] = data[..., 3] # 高度直接赋值
    
    return pred_pos

# ================= [新增：多形状轨迹生成器] =================
def generate_target_trajectory(batch_size, length, device, conf):
    """
    根据配置生成不同形状的目标轨迹 (X, 0, Z)。
    """
    # 1. 读取参数
    shape = conf.SHAPE.lower() # 确保不区分大小写
    speed = conf.SPEED
    turn_start = conf.TURN_START # 第几帧开始转弯
    amp = conf.CURVE_AMPLITUDE   # 弯曲幅度/偏转力度

    # 2. 基础 Z 轴推进 (所有人都要向前走)
    # [0, 1, ..., L-1]
    t_steps = torch.arange(length, device=device).float()
    target_z = t_steps * speed
    target_x = torch.zeros(length, device=device)

    # 3. 根据形状修改 X 轴
    if shape == 'straight':
        pass # 保持 X=0

    # === [修改点开始] ===
    
    # 新增：纯向左 (从一开始就往左偏，配合 Z 轴就是走斜线)
    elif shape == 'left':
        # 简单的线性关系：X 随着时间变大
        # 系数 0.5 可以控制偏的角度，amp 可以进一步放大
        target_x = t_steps * speed * 0.5 * amp 

    # 新增：纯向右
    elif shape == 'right':
        target_x = -1.0 * t_steps * speed * 0.5 * amp

    # 修改：原先的"直走后左转"改名为 straight_then_left
    elif shape == 'straight_then_left':
        mask = t_steps > turn_start
        steps_from_turn = t_steps[mask] - turn_start
        target_x[mask] = steps_from_turn * 0.02 * amp 

    # 修改：原先的"直走后右转"改名为 straight_then_right
    elif shape == 'straight_then_right':
        mask = t_steps > turn_start
        steps_from_turn = t_steps[mask] - turn_start
        target_x[mask] = -1.0 * steps_from_turn * 0.02 * amp

    elif shape == 's_curve':
        # S形曲线: x = A * sin(freq * t)
        # 频率控制：让它在整个序列长度内大概扭 1.5 个周期
        freq = (2 * torch.pi) / (length * 0.6) 
        target_x = torch.sin(t_steps * freq) * (amp * 0.5)
        
    # 新增：纯向后走 (倒退)
    elif shape == 'backward':
        # 覆盖掉默认的向前 Z，改为负数
        target_z = -1.0 * t_steps * speed

    # 新增：向左后方走 (斜着倒退)
    elif shape == 'backward_left':
        # Z 轴后退
        target_z = -1.0 * t_steps * speed
        # X 轴向左 (与 'left' 逻辑一致)
        target_x = t_steps * speed * 0.5 * amp

    # 4. 组合 [X, Y, Z] -> [Batch, Length, 3]
    # Y 轴设为 0 (只引导水平面位置)
    traj = torch.stack([target_x, torch.zeros_like(t_steps), target_z], dim=-1)
    
    return traj.unsqueeze(0).expand(batch_size, -1, -1)
# ==========================================================

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
        self.physics_encoder = instantiate_from_config(cfg.model.physics_encoder)

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


    def _compute_waypoint_loss(self, latents, t, target_global_pos, encoder_hidden_states, lengths, interval=20):
            
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
        
        # 生成索引: [0, 20, 40, ..., last_frame]
        # ================= [新增/检查 Mask 逻辑] =================
        # 我们只计算 valid 长度内的 loss
        # 创建一个 [B, L] 的 mask
        seq_len = calculate_pos.shape[1] # 199，batch里最长的动作的长度
        bs = calculate_pos.shape[0] 
        
        # 生成 Mask: True 代表有效帧，False 代表 Padding
        # range_tensor: [0, 1, 2, ..., L-1]
        range_tensor = torch.arange(seq_len, device=latents.device).unsqueeze(0) # [1, L]
        # lengths_tensor: [B, 1]
        lengths_tensor = torch.tensor(lengths, device=latents.device).unsqueeze(1)
        mask = range_tensor < lengths_tensor # [B, L]
        mask = mask.unsqueeze(-1) # [B, L, 1] 广播到坐标维度
        key_indices = torch.arange(0, seq_len, interval, device=latents.device)
        
        # 1. 提取关键帧 (Key Indices)
        pred_sampled = pred_traj_centered[:, key_indices, :]
        target_sampled = target_traj_centered[:, key_indices, :]
        mask_sampled = mask[:, key_indices, :] # [B, K, 1]
        
        # 2. 手动计算 MSE
        # 只有 mask 为 1 的地方有值，其他地方 diff 为 0
        diff = (pred_sampled - target_sampled) * mask_sampled 
        
        # 平方误差总和
        sum_squared_error = (diff ** 2).sum()
        
        # 有效像素总和 (防止除以0，加个极小值)
        valid_element_count = mask_sampled.sum() * 3 + 1e-8 # *3 是因为坐标有 (x,y,z) 3个维度
        
        # 计算真正的平均 Loss
        loss = sum_squared_error / valid_element_count
        
        # # 4. 求导
        # grad = torch.autograd.grad(loss, latents)[0]
        
        # return grad
        return loss


    #     return loss
    def _compute_ceiling_loss(self, latents, t, ceiling_height, safety_margin, encoder_hidden_states, lengths):
        # 1. 预测 & 反推 x0 (保持不变)
        noise_pred = self.denoiser(sample=latents, timestep=t, encoder_hidden_states=encoder_hidden_states, lengths=lengths)[0]
        alpha_prod_t = self.scheduler.alphas_cumprod[t[0].item()]
        beta_prod_t = 1 - alpha_prod_t
        pred_z0 = (latents - beta_prod_t ** 0.5 * noise_pred) / (alpha_prod_t ** 0.5)
        
        # 2. Decode & 反归一化
        pred_z0_input = pred_z0.permute(1, 0, 2)
        pred_motion_norm = self.vae.decode(pred_z0_input, lengths)
        
        d_mean = self.mean.to(latents.device)
        d_std = self.std.to(latents.device)
        pred_motion = pred_motion_norm * d_std + d_mean

        # 3. 组装全身高度
        root_y = pred_motion[..., 3:4] # [B, L, 1]
        bs, seq_len = pred_motion.shape[:2]
        
        local_joints = pred_motion[..., 4:67].view(bs, seq_len, 21, 3)
        local_joints_y = local_joints[..., 1] # [B, L, 21]
        
        # 绝对高度 = Root + Local
        joints_y_abs = root_y + local_joints_y
        
        # 拼合 Root 和其他关节 -> [B, L, 22]
        all_joints_y = torch.cat([root_y, joints_y_abs], dim=2)

        # ================= [核心修正：全局地面校准] =================
        # 你的逻辑是对的：要找所有帧、所有关节里的最低点作为地面参考
        
        # 1. 先找每一帧的最低点 [B, L]
        frame_min_y, _ = all_joints_y.min(dim=2)
        
        # 2. 再找整个序列的最低点 [B]
        # 这就是这个动作的"地面水平面"
        sequence_min_y, _ = frame_min_y.min(dim=1)
        
        # 3. 广播维度以匹配 [B, L, 22]
        floor_level = sequence_min_y.view(bs, 1, 1)
        
        # 4. 校准：所有高度减去地面高度
        # 这样如果 Frame 10 跳起来了，它的高度 = (AbsHeight - Floor) 就会包含跳跃高度
        grounded_joints_y = all_joints_y - floor_level
        
        # ================= [计算 Loss] =================
        # limit_tensor = torch.tensor(ceiling_height, device=latents.device)
        
        
        # ================= [修改点：应用安全距离] =================
        # 实际限制 = 物理天花板高度 - 安全缓冲
        # 比如天花板 1.5m，缓冲 0.1m -> 关节不能超过 1.4m
        effective_limit = ceiling_height - safety_margin
        
        limit_tensor = torch.tensor(effective_limit, device=latents.device)
        
        
        # 计算超出部分
        excess = torch.nn.functional.relu(grounded_joints_y - limit_tensor)
        
        # 策略：
        # 最高点惩罚 (Loss Max): 只要有任何一帧、任何一个关节撞了，就重罚
        # 整体惩罚 (Loss Mean): 压低整体趋势
        loss_max = (excess.max(dim=2)[0].max(dim=1)[0] ** 2).mean() # [B] -> scalar
        loss_mean = (excess ** 2).mean()
        
        # 组合 (给最高点更高权重，逼它把头缩回去)
        loss = loss_mean + 4.0 * loss_max
        
        return loss


    # def _compute_gap_loss(self, latents, t, gap_width, safety_margin, encoder_hidden_states, lengths):
    #     """
    #     计算狭窄缝隙 Loss (基于局部坐标系)。
        
    #     原理：
    #     不关心人在世界哪里，只关心肢体是否张得太开。
    #     通过限制关节相对于 Root 的横向距离 (Local X)，强迫模型生成“缩手缩脚”的姿态。
    #     """
    #     # 1. 预测 & 反推 x0 (标准流程)
    #     noise_pred = self.denoiser(sample=latents, timestep=t, encoder_hidden_states=encoder_hidden_states, lengths=lengths)[0]
    #     alpha_prod_t = self.scheduler.alphas_cumprod[t[0].item()]
    #     beta_prod_t = 1 - alpha_prod_t
    #     pred_z0 = (latents - beta_prod_t ** 0.5 * noise_pred) / (alpha_prod_t ** 0.5)
        
    #     # 2. Decode & 反归一化
    #     pred_z0_input = pred_z0.permute(1, 0, 2)
    #     pred_motion_norm = self.vae.decode(pred_z0_input, lengths)
        
    #     d_mean = self.mean.to(latents.device)
    #     d_std = self.std.to(latents.device)
    #     # 得到真实的物理数值 [Batch, Length, 263]
    #     pred_motion = pred_motion_norm * d_std + d_mean

    #     # ================= [核心修改：使用局部特征] =================
        
    #     # 3. 提取局部关节位置
    #     # HumanML3D 特征定义: 
    #     # Index 4~66 是 21 个关节相对于 Root 的局部位置 (Local Position)
    #     # 这些位置已经经过旋转对齐，X 轴永远代表"身体右侧"，Z 轴永远代表"身体前方"
    #     bs, seq_len = pred_motion.shape[:2]
    #     local_joints = pred_motion[..., 4:67].view(bs, seq_len, 21, 3)
        
    #     # 4. 提取局部 X 轴坐标 (Local Lateral Offset)
    #     # 这代表了关节离脊柱中线的左右距离
    #     local_joints_x = local_joints[..., 0] # [Batch, Length, 21]

    #     # 5. 定义限制范围
    #     # gap_width 是总宽度 (如 0.5m)，半宽就是 0.25m
    #     # 意味着手脚伸出去不能超过中线 0.25m
    #     half_width = gap_width / 2.0
        
    #     # 减去安全距离 (Safety Margin)
    #     # 比如墙宽 0.5m (半宽0.25)，安全距离 0.05
    #     # 那么关节必须限制在 0.20m 以内，留出 0.05 给皮肤/衣服厚度
    #     effective_limit = half_width - safety_margin
        
    #     limit_tensor = torch.tensor(effective_limit, device=latents.device)
        
    #     # 6. 计算惩罚 (ReLU)
    #     # abs(local_x) 代表偏离中线的程度，无论左右
    #     # 只要 |x| > limit，就产生 Loss
    #     excess = torch.nn.functional.relu(local_joints_x.abs() - limit_tensor)
        
    #     # ================= [策略：重点打击] =================
    #     # 我们不能只算平均值，因为"平均宽度"可能很小，但手可能甩得很大。
    #     # 只要有一个关节撞墙，整个动作就是失败的。
        
    #     # A. 最大违规惩罚 (Max Penalty): 
    #     # 找出每一帧里最“宽”的那个关节 (通常是手腕或手肘)，重罚！
    #     # max(dim=2)[0] 得到每一帧的最大违规量 [Batch, Length]
    #     loss_max = (excess.max(dim=2)[0] ** 2).mean() 
        
    #     # B. 平均违规惩罚 (Mean Penalty):
    #     # 压制整体趋势，让大家尽量往中间靠
    #     loss_mean = (excess ** 2).mean()
        
    #     # 组合 Loss: 10倍权重给最大违规，强迫收回最突出的部位
    #     loss = loss_mean + 10.0 * loss_max
        
    #     return loss
    
    def _compute_gap_loss(self, latents, t, gap_width, safety_margin, encoder_hidden_states, lengths):
        """
        [修正版] 世界坐标系缝隙 Loss (World-Space Gap Loss)。
        
        原理：
        1. 恢复 Root 的世界位置和朝向角度。
        2. 将 Local Joints 旋转并平移到世界坐标系。
        3. 限制 World X 的范围 (模拟固定的直走廊)。
        这样当角色侧身 (旋转90度) 时，原本宽的肩膀 (Local X) 会变成 World Z，
        而较窄的胸背厚度变成 World X，从而通过缝隙。
        """
        # 1. 预测 & 反推 x0 (标准流程)
        noise_pred = self.denoiser(sample=latents, timestep=t, encoder_hidden_states=encoder_hidden_states, lengths=lengths)[0]
        alpha_prod_t = self.scheduler.alphas_cumprod[t[0].item()]
        beta_prod_t = 1 - alpha_prod_t
        pred_z0 = (latents - beta_prod_t ** 0.5 * noise_pred) / (alpha_prod_t ** 0.5)
        
        pred_z0_input = pred_z0.permute(1, 0, 2)
        pred_motion_norm = self.vae.decode(pred_z0_input, lengths)
        
        d_mean = self.mean.to(latents.device)
        d_std = self.std.to(latents.device)
        pred_motion = pred_motion_norm * d_std + d_mean # [Batch, Length, 263]

        # ================= [步骤 A: 计算 Root 的世界状态] =================
        # 提取特征
        rot_vel = pred_motion[..., 0]      # Y轴角速度
        local_vel_x = pred_motion[..., 1]  # 局部线速度 X
        local_vel_z = pred_motion[..., 2]  # 局部线速度 Z
        
        # 1. 积分得到绝对朝向角度 (Heading Angle)
        # cumsum dim=1
        rot_ang = torch.cumsum(rot_vel, dim=1) 
        # 如果 Dataset 预处理有缩放，这里可能需要 * scale，通常 HumanML3D 不需要
        
        # 计算旋转矩阵所需的 sin/cos
        c = torch.cos(rot_ang) # [Batch, Length]
        s = torch.sin(rot_ang)
        
        # 2. 计算 Root 的世界坐标 (为了确定人走到哪了)
        # 投影速度到世界系
        global_vel_x = local_vel_x * c - local_vel_z * s
        # 积分得到 Root World X
        root_world_x = torch.cumsum(global_vel_x, dim=1)
        # 归零起点 (假设走廊中心线从起点开始)
        root_world_x = root_world_x - root_world_x[:, 0:1]
        
        # ================= [步骤 B: 将关节转到世界坐标] =================
        bs, seq_len = pred_motion.shape[:2]
        # Index 4~66: 局部关节位置 (相对于 Root，且对齐 Root 朝向)
        local_joints = pred_motion[..., 4:67].view(bs, seq_len, 21, 3)
        
        # 提取局部坐标
        # local_j_x: 左右 (Right)
        # local_j_z: 前后 (Forward)
        local_j_x = local_joints[..., 0] # [Batch, Length, 21]
        local_j_z = local_joints[..., 2] # [Batch, Length, 21]
        
        # 扩展 c, s 维度以便广播: [B, L] -> [B, L, 1]
        c_exp = c.unsqueeze(-1)
        s_exp = s.unsqueeze(-1)
        
        # 3. 旋转变换 (2D Rotation)
        # 公式: World_X_Offset = Local_X * cos - Local_Z * sin
        # (注意：HumanML3D 是逆时针旋转定义)
        joint_offset_world_x = local_j_x * c_exp - local_j_z * s_exp
        
        # 4. 加上 Root 的世界坐标
        # Root_World_X 广播到 [B, L, 1]
        root_world_x_exp = root_world_x.unsqueeze(-1)
        
        # 得到全身 21 个关节的 World X
        joints_world_x = root_world_x_exp + joint_offset_world_x
        
        # 把 Root 自己也拼进去 (Root 的 Offset 是 0)
        all_joints_world_x = torch.cat([root_world_x_exp, joints_world_x], dim=2) # [B, L, 22]

        # ================= [步骤 C: 计算墙壁碰撞 Loss] =================
        # 定义世界坐标系下的墙： X = ± (Width/2 - Margin)
        half_width = gap_width / 2.0
        effective_limit = half_width - safety_margin
        
        limit_tensor = torch.tensor(effective_limit, device=latents.device)
        
        # 计算绝对值超出部分 (不管是偏左还是偏右撞墙)
        excess = torch.nn.functional.relu(all_joints_world_x.abs() - limit_tensor)
        
        # 策略：重罚最宽的部位
        loss_max = (excess.max(dim=2)[0].max(dim=1)[0] ** 2).mean() 
        loss_mean = (excess ** 2).mean()
        
        loss = loss_mean + 10.0 * loss_max
        
        return loss

    def _compute_side_step_loss(self, latents, t, encoder_hidden_states, lengths):
        """
        [新增] 侧身引导 Loss。
        强迫角色旋转 90 度 (Side-Stepping)。
        """
        # 1. 预测 & 反推 (标准流程)
        noise_pred = self.denoiser(sample=latents, timestep=t, encoder_hidden_states=encoder_hidden_states, lengths=lengths)[0]
        alpha_prod_t = self.scheduler.alphas_cumprod[t[0].item()]
        beta_prod_t = 1 - alpha_prod_t
        pred_z0 = (latents - beta_prod_t ** 0.5 * noise_pred) / (alpha_prod_t ** 0.5)
        
        # 2. Decode
        pred_z0_input = pred_z0.permute(1, 0, 2)
        pred_motion_norm = self.vae.decode(pred_z0_input, lengths)
        
        # 3. 提取旋转速度 (Index 0)
        # 注意：这里不需要反归一化，因为我们只需要趋势，或者假设 std 接近 1。
        # 为了严谨，最好反归一化，但直接用归一化数据的正负号通常也够用。
        # 这里我们做完整的反归一化以防万一。
        d_mean = self.mean.to(latents.device)
        d_std = self.std.to(latents.device)
        pred_motion = pred_motion_norm * d_std + d_mean
        
        rot_vel = pred_motion[..., 0] # [Batch, Length] Y轴角速度
        
        # 4. 积分得到绝对朝向角度 (Heading Angle)
        # 假设初始朝向是 0 (面朝 Z 轴/前方)
        rot_ang = torch.cumsum(rot_vel, dim=1)
       
        # 5. [修改] 强迫朝向特定的角度 (比如 +90度 = PI/2)
        # 这样模型就不用纠结是左转还是右转了
        # target_angle = torch.tensor(1.57, device=latents.device) # 1.57 ≈ 90度
        # [修改] 不要在代码里硬编码 1.57，直接写角度让它自己算
        target_degree = 45.0  # 如果你想试 30度，改成 30.0 即可
        target_radian = float(target_degree * np.pi / 180.0)
        
        target_angle = torch.tensor(target_radian, device=latents.device)
        # print(f"[Side-Step Loss] Target Angle (radian): {target_radian:.4f}")
        
        # 计算当前角度与目标角度的距离 (MSE)
        # 注意：这里可能需要处理周期性 (比如 360度 = 0度)，但简单场景下直接 MSE 够用
        loss = ((rot_ang - target_angle) ** 2).mean()
        
        # 5. 计算 Loss: 逼近 +/- 90 度
        # 正常直走: 角度 ≈ 0, cos(0) = 1 -> Loss 大
        # 侧身行走: 角度 ≈ 90, cos(90) = 0 -> Loss 小
        
        # cos_ang = torch.cos(rot_ang)
        
        # 目标是让 cos_ang 接近 0
        # loss = (cos_ang ** 2).mean()
        
        return loss

    # 基于你提供的原始代码修改，添加了 Config Mock 和 t 的处理
    def _apply_spatial_guidance(self, latents, t, ctx):
        """
        计算并应用基于梯度的空间引导 (Waypoints & Obstacles)。
        """
        # ================= [临时 Config Mock] =================
        # class TempConf:
        #     ENABLED = True
        #     GUIDANCE_START = 1000
        #     GUIDACE_END = 0
        #     WAYPOINTS_MODE = True
        #     WAYPOINTS_GUIDE_STRENGTH = 2000.0 # 强度加大，确保能看到直线效果
        #     OBSTACLE_MODE = False
        #     WAYPOINTS_INTERVAL = 20
        # conf = TempConf()
        # ======================================================
        
        # ================= [读取 YAML 配置] =================
        # 直接指向 configs/config_physimos_probe.yaml 里新加的 TRAJECTORY.GUIDANCE
        conf = self.cfg.TRAJECTORY.GUIDANCE
        # ======================================================
        
        if not conf.ENABLED:    
            return latents

        # 2. 时间窗口检查
        # t 可能是 Tensor，取值
        t_val = t.item() if isinstance(t, torch.Tensor) else t
        if not (conf.GUIDACE_END < t_val < conf.GUIDANCE_START):
            return latents

        # 3. 确定优化步数
        if t_val > 500: num_opt_steps = 1
        elif t_val > 100: num_opt_steps = 5
        else: num_opt_steps = 10

        # 4. 梯度下降循环
        current_latents = latents.detach().requires_grad_(True)
        
        with torch.enable_grad():
            for _ in range(num_opt_steps):
                total_loss = 0.0
                
                # A. 路点引导 (Waypoints)
                if conf.WAYPOINTS_MODE and ctx.get('target_pos') is not None:
                    # 处理 timestep 的维度问题：如果是标量，扩展为 [Batch]
                    # 如果已经是 [Batch]，则直接使用
                    if t.dim() == 0:
                        t_input = t.unsqueeze(0).repeat(ctx['bsz'])
                    else:
                        t_input = t

                    loss_traj = self._compute_waypoint_loss(
                        current_latents, 
                        t_input, 
                        ctx['target_pos'], 
                        ctx['cond_embeddings'], 
                        ctx['lengths'],
                        interval = conf.WAYPOINTS_INTERVAL # 使用配置里的间隔
                    )
                    total_loss += loss_traj * conf.WAYPOINTS_GUIDE_STRENGTH

                # ================= [插入在这里] =================
                # B. 天花板引导 (Low Ceiling)
                # 读取配置: 如果没配置，默认 False
                if conf.get('CEILING_MODE', False):
                    # 处理 t 维度 (同上)
                    if t.dim() == 0: t_input = t.unsqueeze(0).repeat(ctx['bsz'])
                    else: t_input = t
                    
                    # 获取参数
                    c_height = conf.get('CEILING_HEIGHT', 1.4)
                    c_strength = conf.get('CEILING_STRENGTH', 1000.0)
                    
                    # 读取安全距离，默认为 0.1m
                    c_margin = conf.get('SAFETY_MARGIN', 0.1)

                    loss_ceil = self._compute_ceiling_loss(
                        current_latents, 
                        t_input, 
                        c_height,
                        c_margin, # <--- 传入这个新参数
                        ctx['cond_embeddings'], 
                        ctx['lengths']
                    )
                    total_loss += loss_ceil * c_strength
                # ===============================================
                
                # ================= [C. 缝隙引导 (Narrow Gap)] =================
                if conf.get('GAP_MODE', False):
                    # 处理 t 维度 (同上)
                    if t.dim() == 0: t_input = t.unsqueeze(0).repeat(ctx['bsz'])
                    else: t_input = t
                    
                    
                    g_width = conf.get('GAP_WIDTH', 0.5)
                    g_strength = conf.get('GAP_STRENGTH', 2000.0)
                    g_margin = conf.get('SAFETY_MARGIN_GAP', 0.05) # 默认 5cm 缓冲

                    loss_gap = self._compute_gap_loss(
                        current_latents, 
                        t_input, 
                        g_width,
                        g_margin,
                        ctx['cond_embeddings'], 
                        ctx['lengths']
                    )
                    total_loss += loss_gap * g_strength
                    # =========================================================
                # ================= [D. 侧身引导 (Side-Stepping)] =================
                if conf.get('SIDE_STEP_MODE', False):
                    # 2. [核心新增] 自动侧身触发器
                    # 读取阈值，如果没配默认 0.6m
                    # 只有当缝隙小于 0.6m 时，才强制侧身
                    # 处理 t 维度 (同上)
                    if t.dim() == 0: t_input = t.unsqueeze(0).repeat(ctx['bsz'])
                    else: t_input = t
                    
                    
                    side_thresh = conf.get('SIDE_STEP_THRESHOLD', 0.6)
                    g_width = conf.get('GAP_WIDTH', 0.5)
                    if g_width < side_thresh:
                        side_strength = conf.get('SIDE_STEP_STRENGTH', 1500.0)
                        
                        loss_side = self._compute_side_step_loss(
                            current_latents, t_input,
                            ctx['cond_embeddings'], ctx['lengths']
                        )
                        total_loss += loss_side * side_strength
                        
                        # Debug 打印 (只打印一次防止刷屏)
                        if _ == 0 and t_val % 100 == 0:
                            print(f" [SideStep] Gap={g_width} < {side_thresh}, Loss={loss_side.item():.4f}")
                # =======================================================
                
                
                # 如果没有 Loss，直接退出
                if isinstance(total_loss, float) and total_loss == 0.0:
                    break

                # C. 反向传播
                grad = torch.autograd.grad(total_loss, current_latents)[0]

                # D. 梯度裁剪
                grad_norm = grad.norm()
                max_norm = 0.5 # 防止平移的关键，如果不动可以稍微调大
                if grad_norm > max_norm:
                    grad = grad * (max_norm / (grad_norm + 1e-8))

                # E. 更新 Latents
                step_size = 1.0 
                current_latents = current_latents - step_size * grad
                
        return current_latents.detach()


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
        scene_cat = batch['scene_cat'].to(self.device)  # [1.0]

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
            physics_emb, attn_weights = self.physics_encoder(phys_params, scene_cat, need_weights=True) 
            
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
    
    
    # def _diffusion_reverse(self, encoder_hidden_states, lengths=None, scale=None):
    #     # init latents
    #     # 注意：encoder_hidden_states[0] 已经是翻倍后的 batch (2B)，如果开了 CFG
    #     bsz = encoder_hidden_states[0].shape[0] # Batch dimension is 1 for content [S, B, D]
    #     if self.do_classifier_free_guidance:
    #         bsz = bsz // 2

    #     latents = torch.randn(
    #         (bsz, self.latent_dim[0], self.latent_dim[-1]),
    #         device=encoder_hidden_states[0].device,
    #         dtype=torch.float,
    #     )  # torch.Size([1, 7, 256])

    #     latents = latents * self.scheduler.init_noise_sigma
    #     self.scheduler.set_timesteps(self.cfg.model.scheduler.num_inference_timesteps)
    #     timesteps = self.scheduler.timesteps.to(encoder_hidden_states[0].device)
        
    #     extra_step_kwargs = {}
    #     if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
    #         extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta
        
    #     # reverse
    #     for i, t in enumerate(timesteps):
    #         latent_model_input = (torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents)
    #         lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance else lengths)
            
    #         # [重要] Denoiser Forward
    #         noise_pred = self.denoiser(
    #             sample=latent_model_input,
    #             timestep=t,
    #             encoder_hidden_states=encoder_hidden_states,
    #             lengths=lengths_reverse,
    #         )[0]  # torch.Size([2, 7, 256])
            
    #         if self.do_classifier_free_guidance:
    #             noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #             noise_pred = noise_pred_uncond + scale * (noise_pred_text - noise_pred_uncond) # torch.Size([1, 7, 256])
            
    #         latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

    #     latents = latents.permute(1, 0, 2) # torch.Size([7, 1, 256])
    #     return latents
    
    # 包含直线生成的 Reverse 函数
    def _diffusion_reverse(self, encoder_hidden_states, lengths=None, scale=None):
        
        # ================= [物理开关逻辑] =================
        # 默认开启 (True)，如果在 yaml 里写了 False 则关闭
        # 注意：这里我们直接修改 encoder_hidden_states 这个列表
        if not self.cfg.get('ENABLE_PHYSICS', True):
            # encoder_hidden_states 结构通常是: [Content, Physics, Trajectory]
            # Index 1 是 Physics Embedding
            if len(encoder_hidden_states) > 1:
                # 创建全 0 的 Tensor，形状和设备与原 Physics Emb 一致
                zero_phys = torch.zeros_like(encoder_hidden_states[1])
                encoder_hidden_states[1] = zero_phys
                
                # 打印一次提示，防止你忘了自己关了物理
                print(">>> [WARNING] Physics effect is MANUALLY DISABLED in inference!")
        # =================================================
        
        bsz = encoder_hidden_states[0].shape[0]
        if self.do_classifier_free_guidance:
            bsz = bsz // 2
        device = encoder_hidden_states[0].device

        # # ================= [构造 Waypoint (Config版)] =================
        # # 1. 读取配置
        # conf = self.cfg.TRAJECTORY.GUIDANCE
        # max_len = max(lengths) if lengths else 196
        # target_global_pos = torch.zeros((bsz, max_len, 3), device=device)
        
        # # 2. 基础直线逻辑 (读取 conf.SPEED)
        # t_steps = torch.arange(max_len, device=device).float()
        
        # # 判断形状 (目前先只写 straight，下个回答我们在 helper 函数里扩展)
        # if conf.SHAPE == 'straight':
        #     target_z = t_steps * conf.SPEED
        #     target_x = torch.zeros_like(t_steps)
        # else:
        #     # 暂时 fallback 到直线，防止报错
        #     target_z = t_steps * conf.SPEED
        #     target_x = torch.zeros_like(t_steps)

        # # 3. 组合并扩展维度
        # # X轴在 index 0, Z轴在 index 2
        # target_global_pos[..., 0] = target_x.unsqueeze(0).repeat(bsz, 1)
        # target_global_pos[..., 2] = target_z.unsqueeze(0).repeat(bsz, 1)
        # # =====================================================
        
        
        # ================= [调用新函数生成轨迹] =================
        conf = self.cfg.TRAJECTORY.GUIDANCE
        max_len = max(lengths) if lengths else 196
        
        # 直接把配置 conf 传进去，自动判断形状
        target_global_pos = generate_target_trajectory(bsz, max_len, device, conf)
        # ======================================================

        # # ================= [构造直线 Waypoint] =================
        # max_len = max(lengths) if lengths else 196
        # # 生成一个向 Z 轴 (前方) 走的直线
        # # [Batch, Length, 3]
        # target_global_pos = torch.zeros((bsz, max_len, 3), device=device)
        # speed = 0.05 
        # z_steps = torch.arange(max_len, device=device).float().unsqueeze(0) * speed
        # target_global_pos[..., 2] = z_steps.repeat(bsz, 1)
        # # =====================================================

        latents = torch.randn(
            (bsz, self.latent_dim[0], self.latent_dim[-1]),
            device=device,
            dtype=torch.float,
        )
        latents = latents * self.scheduler.init_noise_sigma
        self.scheduler.set_timesteps(self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(device)
        
        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta
        
        # 准备 Guidance Context
        # 剥离 Cond 部分传给 Guidance
        cond_embs = []
        for emb in encoder_hidden_states:
            if self.do_classifier_free_guidance:
                cond_embs.append(emb[bsz:]) 
            else:
                cond_embs.append(emb)

        guidance_ctx = {
            'target_pos': target_global_pos,
            'cond_embeddings': cond_embs,
            'lengths': lengths,
            'bsz': bsz,
            'obstacles': []
        }

        # 循环
        for i, t in enumerate(timesteps):
            # 1. 调用引导 (使用你提供的逻辑)
            # 注意：传入原始标量 t，函数内部会处理
            latents = self._apply_spatial_guidance(latents, t, guidance_ctx)

            # 2. 原生去噪
            latent_model_input = (torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents)
            lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance else lengths)
            
            # timestep 扩展
            t_batch = torch.tensor([t] * latent_model_input.shape[0], device=device)

            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t_batch,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths_reverse,
            )[0]
            
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + scale * (noise_pred_text - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        latents = latents.permute(1, 0, 2)
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
        
        # === [DEBUG 埋点: 开始] ===
        # 打印一下，看看是不是真的进来了，顺便看看物理参数长什么样
        if self.global_step % 10 == 0: # 防止刷屏，每10步打印一次
            print(f"\n[DEBUG] >>> Training Step {self.global_step} Entered! <<<")
            print(f"[DEBUG] Batch Phys Params Shape: {batch['phys_params'].shape}")
            # 这里的 shape 应该是 [BatchSize, 4] (如果你还没改 Dataset)
            # 或者 [BatchSize, Length, 4] (如果你已经改了 Dataset)
        # === [DEBUG 埋点: 结束] ===
        
        
        # 1. 获取数据
        # feats_ref (Target): motion_after
        feats_ref = batch["motion_after"]  # torch.Size([bs, motion_seq_len, 263])
        
        # feats_content (Input): motion_before
        # dataset 已经帮我们处理好了 motion_before = motion_after (或被 mask)
        feats_content = batch["motion_before"] # torch.Size([bs, motion_seq_len, 263])
        
        # Physics Input
        phys_params = batch["phys_params"] # torch.Size([bs, 4])
        scene_cat = batch["scene_cat"] # torch.Size([bs, 1])

        lengths = batch["length"] # torch.Size([bs])

        # ============================================================
        # [PhysiMoS 核心修复] 破坏捷径 (Shortcut Breaking)
        # ============================================================
        # 我们以 50% 的概率，把输入的内容动作 (feats_content) 全部抹零。
        # 此时，模型为了恢复 feats_ref，就只能依赖 physics_emb 了！
        # 这会强迫 Attention 机制去寻找正确的物理参数。
        
        # 生成一个与 Batch Size 相同的随机掩码
        # True 表示保留内容，False 表示抹除内容
        keep_prob = 0.8  # 50% 的概率保留内容，50% 的概率抹除
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
        physics_emb = self.physics_encoder(phys_params, scene_cat) # torch.Size([bs, 1, 256])
        
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
