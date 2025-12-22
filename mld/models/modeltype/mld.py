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

from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModel
import torch.nn as nn
from mld.data.humanml.scripts.motion_process import (process_file,
                                                     recover_from_ric,
                                                     extract_features)
import torch.nn.functional as F
import matplotlib.pyplot as plt
# import evaluate.utils.rotation_conversions as geometry
from mld.models.modeltype.trajectory_utils import *



from .base import BaseModel

DEFAULT_SCALAR_VAL = 3.0 

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=14):
        super().__init__()
        self.net = nn.Sequential(
            # 第一层直接降维到 64 或 128
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2), # 换个激活函数，防止死区
            
            # 加大 Dropout 到 0.5
            nn.Dropout(0.5),
            
            # 输出层
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)
    
def plot_gradient_history(history, save_path="debug_grad_analysis.png", max_norm=5.0):
    """
    绘制梯度变化曲线
    history: dict, 包含 'traj', 'obs', 'final', 'timesteps' 四个 list
    """
    plt.figure(figsize=(12, 6))
    
    steps = np.arange(len(history['timesteps']))
    # 为了直观，X轴显示 diffusion timestep (从1000到0)
    timesteps = history['timesteps']
    
    # 1. 绘制各分量
    plt.plot(steps, history['traj'], label='Trajectory Attraction', color='blue', alpha=0.6)
    plt.plot(steps, history['obs'], label='Obstacle Repulsion', color='red', alpha=0.6)
    
    # 2. 绘制最终合成并裁剪后的梯度
    plt.plot(steps, history['final'], label='Final Grad (Clipped)', color='black', linewidth=2, linestyle='--')
    
    # 3. 绘制裁剪阈值线
    plt.axhline(y=max_norm, color='gray', linestyle=':', label=f'Clip Threshold ({max_norm})')
    
    # 设置 X 轴标签 (每隔10步显示一次t)
    # 既然 step 是顺序的，我们可以只标注几个关键点
    tick_indices = np.linspace(0, len(steps)-1, 10, dtype=int)
    plt.xticks(tick_indices, [str(timesteps[i]) for i in tick_indices])
    plt.xlabel("Diffusion Timestep (t)")
    
    plt.ylabel("Gradient Norm")
    plt.title("Guidance Gradient Dynamics over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Debug] Gradient analysis saved to {save_path}")

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

def debug_plot_trajectory(target_traj, pred_traj, scene_data=None, save_path="debug_traj.png", interval=20):
    """
    绘制轨迹对比图 (XZ 平面俯视图)，包含障碍物、关键帧指引点和偏差连线。
    
    Args:
        target_traj: [Length, 3] numpy array (Global Target, 物理坐标)
        pred_traj:   [Length, 3] numpy array (Generated Root, 物理坐标)
        scene_data:  dict, 包含环境信息 (obstacles)
        interval:    int, 绘制关键点的间隔帧数
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # ================= 0. 数据预处理与对齐 =================
    # 为了公平对比形状，我们将生成轨迹的起点平移到目标轨迹的起点
    # 这样可以消除 VAE 解码时的初始绝对位置偏差
    if target_traj is not None and pred_traj is not None:
        start_offset = target_traj[0] - pred_traj[0]
        # 只平移 X 和 Z，保持 Y 不变 (其实画图只看 XZ)
        pred_traj_aligned = pred_traj + start_offset
    else:
        pred_traj_aligned = pred_traj

    # ================= 1. 画障碍物 (Environment) =================
    if scene_data and 'environment' in scene_data:
        obstacles = scene_data['environment']['obstacles']
        for obs in obstacles:
            # 解析中心点 [x, z]
            cx, cz = obs['center']
            
            if obs['type'] == 'cylinder':
                # 画圆 (半透明灰色填充 + 红色边框)
                radius = obs['radius']
                circle = patches.Circle((cx, cz), radius, 
                                      facecolor='gray', edgecolor='red', 
                                      alpha=0.3, linewidth=2, label='Obstacle')
                ax.add_patch(circle)
                
            elif obs['type'] == 'box':
                # 画矩形 (Matplotlib 的 Rectangle 接受左下角坐标)
                # 兼容 size (w, h, d) 或 extent (w, d)
                if 'extent' in obs:
                    w, d = obs['extent']
                elif 'size' in obs:
                    w, d = obs['size'][0], obs['size'][2]
                else:
                    continue # 无法解析
                
                # 计算左下角
                rect_x = cx - w / 2
                rect_z = cz - d / 2
                rect = patches.Rectangle((rect_x, rect_z), w, d, 
                                       facecolor='gray', edgecolor='blue', 
                                       alpha=0.3, linewidth=2, label='Obstacle')
                ax.add_patch(rect)

    # ================= 2. 画完整轨迹 (Trajectories) =================
    # 目标轨迹 (红色虚线)
    if target_traj is not None:
        ax.plot(target_traj[:, 0], target_traj[:, 2], 'r--', linewidth=2, alpha=0.6, label='Target Path')
        # 起点
        ax.scatter(target_traj[0, 0], target_traj[0, 2], c='red', marker='x', s=150, label='Target Start', zorder=5)

    # 生成轨迹 (蓝色实线)
    if pred_traj_aligned is not None:
        ax.plot(pred_traj_aligned[:, 0], pred_traj_aligned[:, 2], 'b-', linewidth=3, alpha=0.8, label='Generated Path')
        # 起点
        ax.scatter(pred_traj_aligned[0, 0], pred_traj_aligned[0, 2], c='blue', marker='o', s=100, label='Gen Start', zorder=5)

    # ================= 3. 画关键引导点 (Guidance Keypoints) =================
    # 我们不仅画点，还画出“偏差连线”，这样你能直观看到 Guidance 拉扯的方向
    if target_traj is not None and pred_traj_aligned is not None:
        length = len(target_traj)
        # 生成关键帧索引：0, 20, 40... 以及最后一帧
        indices = list(range(0, length, interval))
        if indices[-1] != length - 1:
            indices.append(length - 1)
            
        # 提取关键点坐标
        t_pts = target_traj[indices]
        p_pts = pred_traj_aligned[indices]
        
        # A. 画目标点 (红色空心圆)
        ax.scatter(t_pts[:, 0], t_pts[:, 2], s=100, facecolors='none', edgecolors='red', linewidth=2, label='Guide Points', zorder=10)
        
        # B. 画实际到达点 (蓝色实心点)
        ax.scatter(p_pts[:, 0], p_pts[:, 2], s=60, c='blue', marker='o', zorder=10)
        
        # C. 画误差连线 (灰色细线)
        # 这条线越长，说明该处的 Guidance 效果越差，或者 Content 阻力越大
        for tx, tz, px, pz in zip(t_pts[:, 0], t_pts[:, 2], p_pts[:, 0], p_pts[:, 2]):
            ax.plot([tx, px], [tz, pz], color='gray', linestyle=':', linewidth=1, alpha=0.7)

    # ================= 4. 图表设置 =================
    ax.set_title(f"Trajectory Debug (Interval={interval})", fontsize=14)
    ax.set_xlabel("X Position (meters)")
    ax.set_ylabel("Z Position (meters)")
    
    # 强制等比例，否则圆会变成椭圆
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 处理图例去重 (防止多个障碍物导致图例重复)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')

    # 保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"[Debug] Visualization saved to {save_path}")


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

        self.traj_processor = TrajectoryProcessor()


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


        print("Loading CLIP for Scene Guidance...")
        import os
        local_clip_path = "/root/autodl-tmp/MyRepository/MCM-LDM/local_clip_model" 
        # 或者如果是相对路径: local_clip_path = "./local_clip_model"

        print(f"Loading CLIP from local path: {local_clip_path}")
        # 加载预训练模型
        self.scene_tokenizer = CLIPTokenizer.from_pretrained(local_clip_path)
        self.scene_text_encoder = CLIPTextModel.from_pretrained(local_clip_path)

        # 冻结参数（这一点非常重要，否则显存会爆，且难以训练）
        self.scene_text_encoder.eval()
        self.scene_text_encoder.requires_grad_(False)
        for p in self.scene_text_encoder.parameters():
            p.requires_grad = False
        
        # 作用：把 CLIP 的 512 维特征，映射到适合做 Motion FiLM 的 512 维空间
        # 这就是你说的 "CLIP 之后加几层 MLP"
        self.scene_projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512), # 输出给 FiLM MLP 用
            nn.LayerNorm(512)    # 再次 Norm 保证稳定
        )
        # 初始化 Projector
        # 让它初始接近 Identity，或者稍微有波动
        nn.init.xavier_uniform_(self.scene_projector[0].weight)
        nn.init.zeros_(self.scene_projector[-2].weight) # 最后一层 Linear 权重置0
        nn.init.zeros_(self.scene_projector[-2].bias)   # 这样初始输出接近 0 (经过Norm后会变)

        # 【ICME 新增】1. Vision Encoder
        # 必须和 Text Encoder 版本一致 (e.g. clip-vit-base-patch32)
        print("Loading CLIP Vision Model...")
        self.scene_vision_encoder = CLIPVisionModel.from_pretrained(local_clip_path)
        self.scene_vision_encoder.eval()
        for p in self.scene_vision_encoder.parameters(): 
            p.requires_grad = False
        
        # 【ICME 新增】2. Image Projector
        # 结构建议和 Text Projector 保持一致 (假设 Text Projector 是 3 层 MLP)
        # CLIP Vision (ViT-Base) 输出通常是 768维，需要映射到 512维
        self.scene_image_projector = nn.Sequential(
            nn.Linear(768, 512), 
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512)
        )
        
        # 初始化
        nn.init.xavier_uniform_(self.scene_image_projector[0].weight)
        nn.init.zeros_(self.scene_image_projector[-2].weight)
        nn.init.zeros_(self.scene_image_projector[-2].bias)

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

        self.num_scenes = 14
        # self.scene_embedding_table = torch.nn.Embedding(self.num_scenes, 512)
        # 2. FiLM MLP (输入 512，输出 1024)
        self.film_mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 1024)
        )
        # 初始化为 Identity
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)

        self.scene_norm = nn.LayerNorm(512)
        # Output Norm (最关键！对齐分布！)
        self.style_norm = nn.LayerNorm(512)
        

        # 【ICME 改动】根据配置加载分类器
        if self.cfg.SCENE_MODIFF_ABLATION.USE_SCENE_CLS:
            print("Loading Scene Classifier for Semantic Guidance...")
            self.scene_classifier = SimpleClassifier(num_classes=14)
            ckpt = torch.load("checkpoints/1204/scene_classifier.pth")
            self.scene_classifier.load_state_dict(ckpt)
            

            self.scene_classifier.eval()
            for p in self.scene_classifier.parameters():
                p.requires_grad = False

        # self._get_t2m_evaluator(cfg)

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

        # 【新增】定义一个专门用于 MLP Fusion 的网络
        # 输入是拼接后的 512+512=1024 维，输出是你最终需要的 512 维
        self.fusion_mlp = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, 512) # 输出 512 维，与原始 style_emb 维度相同
        )

    def compute_obstacle_guidance(self, latents, t, obstacles, encoder_hidden_states, lengths):
        """
        SDF 避障引导
        obstacles: list of dict (从 json 解析来的障碍物列表)
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
            
            # 2. Decode & 反归一化
            pred_z0_input = pred_z0.permute(1, 0, 2)
            # 这里可以用一个假的 lengths，或者取 max length，只要保证解码长度对就行
            # 假设 obstacles 不随时间变化，我们用全长
            fake_lengths = [latents.shape[2]] * latents.shape[0] # latent seq len? 
            # 注意: VAE decode 出来的长度通常与输入 latent 长度相关，但也可能被 lengths 截断
            # 这里为了安全，建议用 self.generate_custom_trajectory 里的长度逻辑
            # 简单起见，我们假设 VAE 输出长度就是我们想要的轨迹长度
            
            pred_motion_norm = self.vae.decode(pred_z0_input, lengths) # 使用传入的 lengths
            
            # 反归一化
            if self.mean.device != latents.device:
                self.mean = self.mean.to(latents.device)
                self.std = self.std.to(latents.device)
            pred_motion = pred_motion_norm * self.std + self.mean
            
            # 3. 积分得到物理轨迹 (使用修正后的逻辑)
            pred_rot_vel = pred_motion[..., 0]
            pred_vel_x = pred_motion[..., 1]
            pred_vel_z = pred_motion[..., 2]
            
            pred_rot = torch.cumsum(pred_rot_vel, dim=1)
            c = torch.cos(pred_rot)
            s = torch.sin(pred_rot)
            global_vel_x = pred_vel_x * c - pred_vel_z * s
            global_vel_z = pred_vel_x * s + pred_vel_z * c
            
            pred_pos_x = torch.cumsum(global_vel_x, dim=1)
            pred_pos_z = torch.cumsum(global_vel_z, dim=1)
            
            # 归零起点 (很重要，只关心相对形状避障)
            pred_pos_x = pred_pos_x - pred_pos_x[:, 0:1]
            pred_pos_z = pred_pos_z - pred_pos_z[:, 0:1]
            
            # 组合成 [B, L, 2] 的点集
            current_points = torch.stack([pred_pos_x, pred_pos_z], dim=-1)
            
            # 4. 计算 SDF Loss
            total_obstacle_loss = torch.tensor(0.0, device=latents.device)
            
            # 定义安全边距 (Margin): 我们希望人离障碍物至少有 0.3 米的距离
            safety_margin = 0.3
            
            for obs in obstacles:
                center = torch.tensor(obs['center'], device=latents.device)
                
                if obs['type'] == 'cylinder':
                    radius = obs['radius']
                    # 计算 SDF
                    sdf = diff_sdf_circle(current_points, center, radius)
                    
                elif obs['type'] == 'box':
                    # box 需要 size [width, depth]
                    # json 里的 extent 可能是 [w, d]
                    size = torch.tensor(obs.get('extent', [1.0, 1.0]), device=latents.device)
                    sdf = diff_sdf_box(current_points, center, size)
                else:
                    continue
                
                # 核心避障公式: Loss = ReLU(Margin - SDF)
                # 如果 SDF > Margin (很远)，Loss = 0
                # 如果 SDF < Margin (太近或撞上了)，Loss > 0
                # 撞得越深，Loss 越大
                penetration = torch.nn.functional.relu(safety_margin - sdf)
                
                # 平方惩罚，让梯度更平滑且对深层碰撞反应剧烈
                total_obstacle_loss += (penetration ** 2).sum()

            # 5. 求导
            if total_obstacle_loss > 1e-6:
                grad = torch.autograd.grad(total_obstacle_loss, latents)[0]
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
        features: [Batch, Frames, 263]
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
    
    def compute_spatial_guidance(self, latents, t, target_global_pos, encoder_hidden_states, lengths, interval=20):
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
        

    # def configure_optimizers(self):
    #     # 从配置文件中读取两组学习率，并提供默认值以防万一
    #     lr_denoiser = self.cfg.TRAIN.OPTIM.get("LR_DENOISER", 1e-5)
    #     lr_adapter = self.cfg.TRAIN.OPTIM.get("LR_ADAPTER", 5e-4)

    #     print(f"Optimizer Config --> LR for Denoiser: {lr_denoiser}, LR for Adapter: {lr_adapter}")

    #     # 1. Denoiser 参数组
    #     denoiser_params = list(self.denoiser.parameters())
        
    #     # 2. Adapter (新模块) 参数组
    #     adapter_params = []
    #     # (这部分代码保持不变)
    #     if hasattr(self, "scene_projector"):
    #         adapter_params.extend(list(self.scene_projector.parameters()))
    #     if hasattr(self, "scene_image_projector"):
    #         adapter_params.extend(list(self.scene_image_projector.parameters()))
    #     if hasattr(self, "film_mlp"):
    #         adapter_params.extend(list(self.film_mlp.parameters()))
    #     if hasattr(self, "scene_norm"):
    #         adapter_params.extend(list(self.scene_norm.parameters()))
    #     if hasattr(self, "style_norm"):
    #         adapter_params.extend(list(self.style_norm.parameters()))

    #     # 3. 构造参数组列表
    #     param_groups = [
    #         {"params": denoiser_params, "lr": lr_denoiser},
    #         {"params": adapter_params, "lr": lr_adapter},
    #     ]

    #     # 4. 实例化优化器
    #     optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)
        
    #     print(f"Optimizer initialized. Denoiser Group Size: {len(denoiser_params)}, Adapter Group Size: {len(adapter_params)}")
        
    #     return {"optimizer": optimizer}
        
    # def configure_optimizers(self):
    #     # 只训练 Denoiser，学习率设得很小
    #     optimizer = torch.optim.AdamW(self.denoiser.parameters(), lr=1e-6)
    #     return {"optimizer": optimizer}

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
    def forward_allModel3Branch(self, batch):
        lengths = batch["length"]
        
        # 1. 准备基础数据
        # Style
        motion = batch["style_motion"].clone()
        motion[...,:3] = 0 # torch.Size([1, 199, 263])
        
        # Content
        content_motion = batch['content_motion'] # torch.Size([1, 38, 263])
        # 保持归一化逻辑不变，因为测试的时候是直接加载的npy，没有数据集的归一化处理，所以在推理的forward里面需要手动进行一下归一化
        content_motion = (content_motion - self.mean.to(content_motion.device)) / self.std.to(content_motion.device)
        content_motion[...,:3] = 0
        
        # Trajectory
        trans_motion = content_motion.clone() 
        trans_cond = trans_motion[...,:3] # torch.Size([1, 38, 3])
        
        # 统一长度 (用于 VAE 编码)
        lengths1 = [content_motion.shape[1]] * content_motion.shape[0]
        
        if self.cfg.TEST.COUNT_TIME:
            self.starttime = time.time()
            
        if self.stage in ['diffusion', 'vae_diffusion']:
            
            # --- A. Content Encoding ---
            with torch.no_grad():
                z, dist_m = self.vae.encode(content_motion.float(), lengths1)
            cond_emb = z # torch.Size([7, 1, 256])

            # --- B. Style Encoding (MotionCLIP) ---
            lengths11 = [motion.shape[1]] * motion.shape[0]
            motion_seq = motion.unsqueeze(-1).permute(0, 2, 3, 1)  # torch.Size([1, 263, 1, 199])
            
            motion_emb = self.motionclip.encoder({
                'x': motion_seq.float(),
                'y': torch.zeros(motion_seq.shape[0], dtype=int, device=motion_seq.device),
                'mask': lengths_to_mask(lengths11, device=motion_seq.device)
            })["mu"]  # torch.Size([1, 512])
            motion_emb = motion_emb.unsqueeze(1) # [B, 1, 512]

            # --- C. Scene Encoding (CLIP) ---
            has_image = batch["has_image"]
            use_image_for_inference = has_image.item()
            
            if not use_image_for_inference:
                # Text Branch
                print("Using text for CLIP Multi Modal...")
                scene_texts = batch.get("scene_text", [""] * len(lengths))
                with torch.no_grad():
                    text_inputs = self.scene_tokenizer(scene_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                    clip_feat = self.scene_text_encoder(**text_inputs).pooler_output 
                scene_feat = self.scene_projector(clip_feat) 
            else:
                # Image Branch
                print("Using image for CLIP Multi Modal...")
                scene_images = batch.get("scene_image").to(motion_seq.device)
                with torch.no_grad():
                    vision_out = self.scene_vision_encoder(pixel_values=scene_images)
                    image_feat_raw = vision_out.pooler_output 
                scene_feat = self.scene_image_projector(image_feat_raw)
            
            # --- D. FiLM Modulation ---
            scene_feat_norm = self.scene_norm(scene_feat) # 记得 Norm, scene_feat的维度是torch.Size([1, 512])
            if self.cfg.SCENE_MODIFF_ABLATION.FUSION_MODE == "film":
                film_params = self.film_mlp(scene_feat_norm)
                gamma_raw, beta_raw = film_params.chunk(2, dim=-1)
                
                gamma = (1.0 + torch.tanh(gamma_raw)).unsqueeze(1)
                beta = beta_raw.unsqueeze(1)
                
                # 这里的 motion_emb 是原始的 Style
                filmed_emb = gamma * motion_emb + beta
            elif self.cfg.SCENE_MODIFF_ABLATION.FUSION_MODE == "mlp":
                # MLP 融合
                # 拼接 Style 和 Scene 特征
                combined_feat = torch.cat([motion_emb.squeeze(1), scene_feat_norm], dim=-1)
                filmed_emb = self.fusion_mlp(combined_feat)
                filmed_emb = filmed_emb.unsqueeze(1) # 恢复维度 [B, 1, 512]

            else:
                raise NotImplementedError(
                    f"Unknown fusion mode: {self.cfg.SCENE_MODIFF_ABLATION.FUSION_MODE}"
                )
            # 归一化 (给 Denoiser 用的)
            style_raw_normed = self.style_norm(motion_emb) # 纯 Style
            style_mix_normed = self.style_norm(filmed_emb) # Style + Scene

            # ========================================================
            # 【ICME 核心】 构造 3-Branch Batch for Dual Guidance
            # ========================================================
            # 顺序: [Uncond, Style_Only, Mix]
            
            # 1. Style Condition
            uncond_style = torch.zeros_like(style_raw_normed)
            # 拼接: [0, Style, Mix]
            motion_emb_cfg = torch.cat([uncond_style, style_raw_normed, style_mix_normed], dim=0) # torch.Size([3, 1, 512])

            # 2. Scene Condition (作为第4个输入 f_scene_indep)
            # Branch 1 (Style Only) 不需要 Scene 显式输入，所以给 0 或者给 scene_feat 都可以
            # 为了严谨: [0, 0, Scene]
            scene_feat_expanded = scene_feat_norm.unsqueeze(1)
            scene_zero = torch.zeros_like(scene_feat_expanded)
            scene_emb_cfg = torch.cat([scene_zero, scene_zero, scene_feat_expanded], dim=0)  # torch.Size([3, 1, 512])

            # 3. Content Condition
            # 复制 3 份，因为 Content 始终保持不变
            motion_emb_content = torch.cat([cond_emb, cond_emb, cond_emb], dim=1) # 注意 dim=1 因为 cond_emb 是 [Seq, Batch, Dim] torch.Size([7, 3, 256])

            # 4. Trajectory Condition (策略选择)
            trans_zero = torch.zeros_like(trans_cond)
            
            # 【策略 A】全部置零 (激进，完全由 Style/Scene 决定轨迹) -> 你现在的做法
            # uncond_trans = torch.cat([trans_zero, trans_zero, trans_zero], dim=0) # torch.Size([3, 38, 3])
            
            # 【策略 B】Uncond=0, Style=Real, Mix=Real (推荐，如果动作不动的话用这个)
            # 这样 Style 分支能保住走路的趋势
            # uncond_trans = torch.cat([trans_zero, trans_cond, trans_cond], dim=0)

            # 【策略C】 参考MCM-LDM的原来做法，直接强制有轨迹
            uncond_trans = torch.cat([trans_cond, trans_cond, trans_cond], dim=0)

            # 组装
            motion_emb_content = motion_emb_content.permute(1, 0, 2) # torch.Size([3, 7, 256])
            multi_cond_emb = [motion_emb_content, motion_emb_cfg, uncond_trans, scene_emb_cfg]

            # ========================================================
            # 设置 Scale
            # ========================================================
            # tag_scale 是外部传入的，通常是 float
            
            # 【调试建议】手动指定，方便观察
            # 含义: (保留原始风格的力度, 注入场景变化的力度)
            final_scale = (self.cfg.TEST.CFG_STYLE, self.cfg.TEST.CFG_SCENE) 

            # 调用修改后的 _diffusion_reverse
            z = self._diffusion_reverse(multi_cond_emb, lengths, final_scale) 

        elif self.stage in ['vae']:
            # VAE 测试逻辑不变
            motions = batch['motion']
            z, dist_m = self.vae.encode(motions, lengths)

        with torch.no_grad():
            feats_rst = self.vae.decode(z, lengths) 
            
        joints = self.feats2joints(feats_rst.detach().cpu())
        return remove_padding(joints, lengths)
    

    def forward_wo_sceneLoss(self, batch, scene_data):
        # print("Forward w/o scene classification loss...")
        lengths = batch["length"]
        # style
        motion = batch["style_motion"].clone()
        motion[...,:3] = 0


        # content
        content_motion = batch['content_motion']
        content_motion = (content_motion - self.mean.to(content_motion.device))/self.std.to(content_motion.device)

        if self.cfg.TRAJECTORY.ROOT_MASKING_DIM4:
            content_motion[...,:4] = 0
        else:
            trans_motion = content_motion.clone()
            content_motion[...,:3] = 0

        target_global_pos = None # 用于 Guidance
        trans_cond_norm = None   # 用于 Denoiser Input
        
        if scene_data is not None and self.cfg.TRAJECTORY.ENABLED:
            raw_content = batch['content_motion'].clone()  # 使用未归一化的数据
            # A. 计算 Content 距离 Profile
            # raw_content[0] 取 Batch 中第一个样本作为参考
            content_npy = raw_content[0].detach().cpu().numpy()
            dist_profile = self.traj_processor.calculate_cumulative_distance(content_npy) # (39,)
            
            # B. 生成避障路径 (A*)
            import copy
            waypoints = copy.deepcopy(scene_data['trajectory']['points'])
            obstacles = copy.deepcopy(scene_data['environment']['obstacles'])
            
            startPosX, startPosY = waypoints[0][0], waypoints[0][1]
            for p in waypoints:
                p[0] -= startPosX
                p[1] -= startPosY
            for obs in obstacles:
                if 'center' in obs:
                    obs['center'][0] -= startPosX
                    obs['center'][1] -= startPosY
                elif 'pos' in obs: # 防御性编程，有的格式可能是 pos
                    obs['pos'][0] -= startPosX
                    obs['pos'][1] -= startPosY
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

            # # trajectory
            # trans_cond = trans_motion[...,:3]
            # trans_uncond = torch.zeros(trans_cond.shape).to(motion_seq.device)
            # uncond_trans = torch.cat([trans_cond, trans_cond], dim = 0)
            # uncond_trans = torch.cat([trans_uncond, trans_uncond], dim=0)
            # uncond_trans = torch.cat([trans_uncond, trans_cond], dim=0)
            uncond_trans = torch.cat([trans_cond_input, trans_cond_input], dim=0) # [2*B, L, 4]

            has_image = batch["has_image"] # tensor([True], device='cuda:0')
            use_image_for_inference = has_image.item()
            print("use_image_for_inference :", use_image_for_inference)
            scene_feat = None
            if not use_image_for_inference:
                scene_texts = batch.get("scene_text", ["A person moving in a normal environment"] * len(lengths))
                with torch.no_grad():
                    text_inputs = self.scene_tokenizer(scene_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                    clip_feat = self.scene_text_encoder(**text_inputs).pooler_output # [B, 512]
                scene_feat = self.scene_projector(clip_feat) # [Batch, 512]
                # scene_feat_norm = self.scene_norm(scene_feat)
            else:
                scene_images = batch.get("scene_image").to(motion_seq.device) # torch.Size([1, 3, 224, 224])
                with torch.no_grad():
                    vision_out = self.scene_vision_encoder(pixel_values=scene_images)
                    image_feat_raw = vision_out.pooler_output # [B, 768]
                image_feat = self.scene_image_projector(image_feat_raw) # [B, 512]
                scene_feat = image_feat
                

            # scene_feat_norm = self.scene_norm(scene_feat)
            scene_scalar = batch.get("scene_scalar", DEFAULT_SCALAR_VAL) 
            
            # 如果传入的是 Tensor (单个数)，转为 float，防止乘法广播出问题
            if isinstance(scene_scalar, torch.Tensor):
                scene_scalar = scene_scalar.item()
                
            print(f"DEBUG: Using scene_scalar = {scene_scalar}") # 调试用，跑通后可注释
            
            film_params = self.film_mlp(scene_feat * scene_scalar)
            gamma_raw, beta_raw = film_params.chunk(2, dim=-1)
            gamma = (1.0 + torch.tanh(gamma_raw)).unsqueeze(1) # [B, 1, 512]
            beta = beta_raw.unsqueeze(1)                       # [B, 1, 512]
            # motion_emb = torch.zeros_like(motion_emb) # 先试试不要style
            
            filmed_emb = gamma * motion_emb + beta
            adapted_style_normed = self.style_norm(filmed_emb)

            # D. 构造 CFG 输入
            # Uncond 分支：给全 0 (代表"无风格")
            # Cond 分支：给 Adapted Style
            uncond_style = torch.zeros_like(adapted_style_normed)
            
            # 拼接顺序：[Uncond, Cond], 这个是场景指导后的style
            motion_emb_cfg = torch.cat([uncond_style, adapted_style_normed], dim=0)
 

            scene_feat_reshaped = scene_feat.unsqueeze(1)
            uncond_scene = torch.zeros_like(scene_feat_reshaped)
            scene_emb_cfg = torch.cat([uncond_scene, scene_feat_reshaped], dim=0)  # 其实目前的去噪网络用不到这一项，以防后面可能需要就放进来了

            # three conditions
            multi_cond_emb = [motion_emb_content, motion_emb_cfg, uncond_trans]


            z = self._diffusion_reverse(multi_cond_emb, lengths, scale, target_global_pos, scene_data)

        elif self.stage in ['vae']:
            motions = batch['motion']
            z, dist_m = self.vae.encode(motions, lengths)

        with torch.no_grad():
            feats_rst = self.vae.decode(z, lengths)
            # feats_rst[...,:3] = trans_motion[...,:3] # if copy trajectory

        joints = self.feats2joints(feats_rst.detach().cpu())

        if startPosX != 0 or startPosY != 0:
            # 只平移根节点和所有子节点的位置
            # 0 是 x 轴, 2 是 z 轴 (根据你的 trajectory 生成逻辑)
            joints[..., 0] += startPosX
            joints[..., 2] += startPosY
        
        # 同时，如果 target_global_pos 需要用于可视化 debug，也加回去
        if target_global_pos is not None:
             target_global_pos[..., 0] += startPosX
             target_global_pos[..., 2] += startPosY
        # =======================================================

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
                    # self.target_pos,
                    scene_data = scene_data,
                    # save_path=f"vis_debug/traj_step_n.png"
                    # 每个样本给个不同的文件名
                    save_path=f"vis_debug/traj_debug_{name}.png",
                    interval = self.cfg.TRAJECTORY.GUIDANCE.WAYPOINTS_INTERVAL
                )
            except Exception as e:
                print(f"[Warning] Failed to plot trajectory: {e}")
        # =======================================================


        return remove_padding(joints, lengths)
    
    def forward_step1_warmup_withOurSceneDataset(self, batch):
        print("forward_step1_warmup_withOurSceneDataset")
        lengths = batch["length"]
        # style
        motion = batch["style_motion"].clone()
        motion[...,:3] = 0


        # content
        content_motion = batch['content_motion']
        content_motion = (content_motion - self.mean.to(content_motion.device))/self.std.to(content_motion.device)

        # trajectory
        trans_motion = content_motion.clone()
        # 
        content_motion[...,:3] = 0


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
            trans_cond = trans_motion[...,:3]
            uncond_trans = torch.cat([trans_cond, trans_cond], dim = 0)

            # three conditions
            multi_cond_emb = [motion_emb_content, motion_emb, uncond_trans]


            z = self._diffusion_reverse(multi_cond_emb, lengths, scale)

        elif self.stage in ['vae']:
            motions = batch['motion']
            z, dist_m = self.vae.encode(motions, lengths)

        with torch.no_grad():
            feats_rst = self.vae.decode(z, lengths)
            # feats_rst[...,:3] = trans_motion[...,:3] # if copy trajectory

        joints = self.feats2joints(feats_rst.detach().cpu())

        return remove_padding(joints, lengths)

    # 学习CLIP和FiLM调控的forward函数
    def forward(self, batch, scene_data=None):
        if self.cfg.SCENE_MODIFF_ABLATION.JUST_FINETUNE_BASELINE == True:
            return self.forward_step1_warmup_withOurSceneDataset(batch)
        if self.cfg.SCENE_MODIFF_ABLATION.USE_SCENE_CLS == False and self.cfg.SCENE_MODIFF_ABLATION.FUSION_MODE == "film":
            return self.forward_wo_sceneLoss(batch, scene_data)
        elif self.cfg.SCENE_MODIFF_ABLATION.USE_SCENE_CLS == True and self.cfg.SCENE_MODIFF_ABLATION.FUSION_MODE == "film":
            # return self.forward_allModel3Branch(batch)
            return self.forward_wo_sceneLoss(batch, scene_data)
        else:
            return self.forward_wo_sceneLoss(batch, scene_data)
        

    def _diffusion_reverse(self, encoder_hidden_states, lengths=None, scale=None, target_global_pos=None, scene_data=None):
        # 1. 自动判断 CFG 模式
        total_bsz = encoder_hidden_states[0].shape[0]  # 3
        base_bsz = len(lengths)  # 1
        # 准备障碍物数据
        obstacles = []
        if scene_data is not None:
            obstacles = scene_data['environment']['obstacles']
        
        # 【新增】梯度记录器
        grad_history = {
            'traj': [],
            'obs': [],
            'final': [],
            'timesteps': []
        }
        
        # 计算倍率: 
        # 2 -> 标准 CFG [Uncond, Cond]
        # 3 -> 双重引导 [Uncond, Style, Mix]
        cfg_factor = total_bsz // base_bsz 
        
        # 初始化 Latents (只需要 Base Batch Size)
        latents = torch.randn(
            (base_bsz, self.latent_dim[0], self.latent_dim[-1]),
            device=encoder_hidden_states[0].device,
            dtype=torch.float,
        )

        # scale the initial noise
        latents = latents * self.scheduler.init_noise_sigma
        
        # set timesteps
        self.scheduler.set_timesteps(
            self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(encoder_hidden_states[0].device)
        
        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta
        
        # 解析 Scale 参数
        # 如果是 3 倍模式，且 scale 只是一个浮点数，我们默认两个系数都用这个数
        # 如果 scale 是列表/元组 (e.g., [7.5, 7.5])，则分别赋值
        scale_style = scale
        scale_scene = scale
        if cfg_factor == 3 and isinstance(scale, (list, tuple)):
            scale_style = scale[0]
            scale_scene = scale[1]

        use_guidance = self.cfg.TRAJECTORY.GUIDANCE.ENABLED
        # Reverse Loop
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
            start_t = self.cfg.TRAJECTORY.GUIDANCE.GUIDANCE_START # 比如1000
            end_t = self.cfg.TRAJECTORY.GUIDANCE.GUIDACE_END
            interval = self.cfg.TRAJECTORY.GUIDANCE.WAYPOINTS_INTERVAL
            way_guidance_scale = self.cfg.TRAJECTORY.GUIDANCE.WAYPOINTS_GUIDE_STRENGTH
            obstacle_guidance_scale = self.cfg.TRAJECTORY.GUIDANCE.OBSTACLE_GUIDE_STRENGTH
            if end_t < t < start_t: 
                for k in range(num_opt_steps):
                    grad_traj = torch.zeros_like(latents).to(latents.device)
                    grad_obs = torch.zeros_like(latents).to(latents.device)
                    if use_guidance and self.cfg.TRAJECTORY.GUIDANCE.WAYPOINTS_MODE:
                        # 计算梯度
                        grad_traj = self.compute_spatial_guidance(
                            latents, 
                            t.unsqueeze(0).repeat(base_bsz), # expand t
                            target_global_pos, 
                            [h[base_bsz:] for h in encoder_hidden_states],  # [uncond, cond] -> [cond]
                            lengths,
                            interval=interval
                        )
                    if use_guidance and self.cfg.TRAJECTORY.GUIDANCE.OBSTACLE_MODE:
                        grad_obs = self.compute_obstacle_guidance(
                            latents, 
                            t.unsqueeze(0).repeat(base_bsz), 
                            obstacles, 
                            [h[base_bsz:] for h in encoder_hidden_states],
                            lengths,
                        )

                    # 计算加权后的模长 (用于记录)
                    norm_traj = (grad_traj * way_guidance_scale).norm().item()
                    norm_obs = (grad_obs * obstacle_guidance_scale).norm().item()
                    grad = grad_traj * way_guidance_scale + grad_obs * obstacle_guidance_scale

                    # 2. 梯度裁剪 (保持你现在的逻辑，非常稳)
                    grad_norm = grad.norm()
                    max_grad_norm = 5.0 # 或者根据 t 动态调整
                    if grad_norm > max_grad_norm:
                        scale_factor = max_grad_norm / (grad_norm + 1e-8)
                        grad = grad * scale_factor
                    norm_final = grad.norm().item()
                    
                    # 3. 更新 Latents
                    # 这里的 step_size 可以小一点，因为我们跑很多次
                    step_size = 1.0 
                    latents = latents - step_size * grad
                    latents = latents.detach().requires_grad_(True) # 记得 detach 并重新开启梯度追踪

                    # 【新增】只记录 Inner Loop 的最后一次迭代，或者记录每一次？
                    # 建议：为了曲线清晰，只记录 k == 0 (第一次) 或者 k == num_opt_steps-1 (最后一次)
                    # 这里我们记录最后一次，代表这一步最终施加的力
                    if k == num_opt_steps - 1:
                        grad_history['traj'].append(norm_traj)
                        grad_history['obs'].append(norm_obs)
                        grad_history['final'].append(norm_final)
                        grad_history['timesteps'].append(t.item())

                        # print(f"Step {t.item()} Inner {k}: Loss:", grad.norm())
            # 2. 扩展 Latents 以匹配 Condition 的倍率 (2倍或3倍), latents是[1,7,256]
                        
            latent_model_input = torch.cat([latents] * cfg_factor, dim=0) # torch.Size([3, 7, 256])
            
            # 扩展 Lengths
            lengths_reverse = lengths * cfg_factor  # [38, 38, 38]
            
            # 3. 预测噪声
            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths_reverse,
            )[0] # torch.Size([3, 7, 256])
            
            # 4. 执行 Guidance (核心修改部分)
            if cfg_factor == 3:
                # ============================================
                # 【ICME 核心】双重引导 (Dual-Guidance)
                # 分割顺序: [Uncond, Style_Only, Mix]
                # ============================================
                noise_uncond, noise_style, noise_mix = noise_pred.chunk(3, dim=0)
                
                # 公式:
                # 第一部分: 把动作拉向 Style (老人/举手)
                # 第二部分: 把动作从 Style 拉向 Scene (弯腰/大风)
                # 两个 Scale 互不干扰，可以同时很大！
                noise_pred = noise_uncond + \
                             scale_style * (noise_style - noise_uncond) + \
                             scale_scene * (noise_mix - noise_style)
                             
            elif cfg_factor == 2:
                # 标准 CFG
                noise_uncond, noise_text = noise_pred.chunk(2, dim=0)
                noise_pred = noise_uncond + scale * (noise_text - noise_uncond)
            
            # (如果是 1 倍则不处理，直接用 noise_pred)

            # 5. Step 更新
            latents = self.scheduler.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample

        latents = latents.permute(1, 0, 2) # torch.Size([7, 1, 256])
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
        if self.cfg.SCENE_MODIFF_ABLATION.JUST_FINETUNE_BASELINE == True:
            if self.global_step == 0:
                print("Just finetune baseline diffusion model, without scene modulation.")
            return self.train_diffusion_forward_finetune_baseline(batch)
        
        feats_ref = batch["motion"] # torch.Size([32, 40, 263])
        feats_content = batch["motion"].clone() # torch.Size([32, 40, 263])

        # 1. 旋转增强 (根据 Config 开关)
        if self.cfg.TRAJECTORY.AUGMENTATION.ENABLED:
            angle = self.cfg.TRAJECTORY.AUGMENTATION.ROTATION_RANGE
            feats_content = self.augment_content_rotation(feats_content, angle)

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

        bsz = feats_ref.shape[0]
        
        # content condition
        with torch.no_grad():
            z, dist = self.vae.encode(feats_ref, lengths) # z:torch.Size([7, 32, 256]), dist: torch.Size([7, 32, 256])
            z_content, dist = self.vae.encode(feats_content, lengths)
            cond_emb = z_content.permute(1,0,2)  # torch.Size([32, 7, 256])     
            if self.cfg.TRAJECTORY.USE_CONTENT_DROPOUT:
                cond_emb[mask_content_drop] = 0.0   

        # style condition
        motion_seq = feats_ref*self.std + self.mean 
        motion_seq[...,:3]=0.0
        motion_seq = motion_seq.unsqueeze(-1).permute(0,2,3,1) # torch.Size([32, 263, 1, 40])
        motion_emb_raw = self.motionclip.encoder({'x': motion_seq,
                        'y': torch.zeros(motion_seq.shape[0], dtype=int, device='cuda:{}'.format(self.cfg["DEVICE"][0])),
                        'mask': lengths_to_mask(lengths, device='cuda:{}'.format(self.cfg["DEVICE"][0]))})["mu"] # 一个style被提取成了512维的tensor，torch.Size([32, 512])
        motion_emb_raw = motion_emb_raw.unsqueeze(1) # torch.Size([32, 1, 512])
        
        # 多模态的条件获取
        scene_texts = batch.get("scene_text", [""] * len(batch["length"]))
        scene_images = batch.get("scene_image", None) # [B, 3, 224, 224]
        has_image = batch['has_image']      # [B] (Bool, 哪些样本有图)

        multi_modal_type = self.cfg.TRAIN_STRTEGY.MULTI_MODAL_FUSION
        use_image = False
        if multi_modal_type == "image":
            use_image = (scene_images is not None) and (has_image.all()) 
        elif multi_modal_type == "text":
            use_image = False
        elif multi_modal_type == "select":
            use_image = (scene_images is not None) and (has_image.all()) and (torch.rand(1).item() < 0.5)

        if use_image:
            with torch.no_grad():
                vision_out = self.scene_vision_encoder(pixel_values=scene_images)
            scene_feat_raw = self.scene_image_projector(vision_out.pooler_output)
        else:
            with torch.no_grad():
                text_inputs = self.scene_tokenizer(scene_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                text_out = self.scene_text_encoder(**text_inputs)
            scene_feat_raw = self.scene_projector(text_out.pooler_output) # torch.Size([32, 512])
            
        scene_feat = scene_feat_raw.unsqueeze(1) # [B, 1, 512]

        # # 【新增：对style motion进行随机的corrupt，为了强迫模型看scene的信息】
        # if self.cfg.LOSS.USE_SCENE_CLS:
        #     corruption_mask = torch.rand(bsz, 1, 1, device=self.device) < self.cfg.LOSS.CORRUPTION_PROB 
        #     # 生成随机缩放系数 (0.1 ~ 0.6)，让 Style 变得很弱
        #     scaling_factor = 0.1 + 0.5 * torch.rand(bsz, 1, 1, device=self.device)
        #     motion_emb = torch.where(corruption_mask, motion_emb_raw * scaling_factor, motion_emb_raw)  # torch.where(condition, x, y),如果condition符合，取x，否则取y
        motion_emb = motion_emb_raw

        # 默认全部保留
        mask_style = torch.zeros(bsz, dtype=torch.bool, device=self.device)
        mask_scene = torch.zeros(bsz, dtype=torch.bool, device=self.device)
        probs = torch.rand(bsz, device=self.device)
        # 设定阈值
        # [0, 0.5): Keep Both
        # [0.5, 0.7): Drop Style (Keep Scene Only) -> 强迫模型看 Beta
        mask_style[probs >= 0.5] = True 
        mask_style[probs >= 0.7] = False # 0.7以上 Style 恢复回来

        # [0.7, 0.9): Drop Scene (Keep Style Only) -> 退化为 MCM-LDM
        mask_scene[probs >= 0.7] = True
        mask_scene[probs >= 0.9] = False # 0.9以上 Scene 恢复回来

        # [0.9, 1.0): Drop Both (Uncond)
        mask_style[probs >= 0.9] = True
        mask_scene[probs >= 0.9] = True

        # 应用 Mask
        motion_emb[mask_style] = 0 
        scene_feat[mask_scene] = 0

        # trans condition
        if self.cfg.TRAJECTORY.ROOT_MASKING_DIM4:
            trans_cond = batch["motion"][...,:4]  # torch.Size([32, 40, 4])
        else:
            trans_cond = batch["motion"][...,:3]  # torch.Size([32, 40, 3])

        if self.cfg.SCENE_MODIFF_ABLATION.FUSION_MODE == "film":
            # 3. 生成 FiLM 参数
            scene_feat_norm = self.scene_norm(scene_feat)  # 这句归一化应该是必不可少的，不然尺度都对不上
            film_params = self.film_mlp(scene_feat_norm)  # torch.Size([32, 1, 1024])
            # 【修正开始：强制让被 Mask 的样本不产生 FiLM 调制】
            # 这一步非常重要！确保当没有 Scene 时，FiLM 是 Identity (gamma=1, beta=0)
            # 这里的 mask_scene 维度是 [B]，需要扩展为 [B, 1, 1] 才能乘 film_params
            mask_scene_expanded = mask_scene.unsqueeze(1).unsqueeze(2).float() # [B, 1, 1]
            
            # 如果 mask 为 1 (True)，则让 film_params 变 0
            film_params = film_params * (1 - mask_scene_expanded)
            # 【修正结束】
            gamma_raw, beta_raw = film_params.chunk(2, dim=-1)  # 两个都是torch.Size([32, 1, 512])
            
            # 维度对齐 [Batch, 1, 512]，在刚开始train的时候，gamma是1，beta是0，相当于一个zero映射，不破坏网络本来的学习
            gamma = (1.0 + torch.tanh(gamma_raw)) # [Batch, 1, 512]
            beta = beta_raw                 # [Batch, 1, 512]

            # 执行融合 (先 Mask Style，后 FiLM)
            # 1. 如果 Style 被 Mask，Scene 没被 Mask -> Output = Beta (纯场景)
            # 2. 如果 Style 没 Mask，Scene 被 Mask -> Output = Style (纯风格)
            # 3. 如果都存在 -> Output = Modulated Style (融合)
            # 4. 如果都 Mask -> Output = 0 (无条件)
            adapted_style_emb = gamma * motion_emb + beta  # torch.Size([32, 1, 512])  注意：这里用的 motion_emb 可能是被 Corrupt 或 Mask 过的
        elif self.cfg.SCENE_MODIFF_ABLATION.FUSION_MODE == "mlp":
            # 【新】MLP Fusion 逻辑
            # a. 准备特征 (去掉那个多余的 unsqueeze(1))
            style_feat = motion_emb_raw.squeeze(1)
            scene_feat_norm = self.scene_norm(scene_feat.squeeze(1))
            
            # b. 拼接
            combined_feat = torch.cat([style_feat, scene_feat_norm], dim=1) # shape: [B, 1024]
            
            # c. 通过 MLP 进行融合
            fused_emb = self.fusion_mlp(combined_feat) # shape: [B, 512]
            
            # d. 变回 [B, 1, 512] 的标准格式
            adapted_style_emb = fused_emb.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported fusion mode: {self.cfg.SCENE_MODIFF_ABLATION.FUSION_MODE}. Should be 'film' or 'add'.")
        
        adapted_style_emb = self.style_norm(adapted_style_emb)
 
        # three condition
        # multi_cond_emb = [cond_emb, motion_emb, trans_cond] # 复习一下： cond_emb：内容（torch.Size([32, 7, 256])），motion_emb：风格（torch.Size([32, 1, 512])），trans_cond：轨迹（torch.Size([32, 40, 3])）
        # scene_emb: 我们新增的场景的自然语言描述：torch.Size([32, 1, 512])
        multi_cond_emb = [cond_emb, adapted_style_emb, trans_cond]  

        # diffusion process return with noise and noise_pred
        n_set = self._diffusion_process(z, multi_cond_emb, lengths) # 返回的n_set是一个字段，包含计算loss的时候pytorch_lightning所关心的内容
        
        # ==========================================
        # 5. 计算新 Loss (Scene Classifier Guidance)
        # ==========================================
        if self.cfg.SCENE_MODIFF_ABLATION.USE_SCENE_CLS:
            z_t = n_set['noisy_latents']  # torch.Size([32, 7, 256])
            t = n_set['timesteps']
            noise_pred = n_set['noise_pred']
            
            # ========================================================
            # 【修复】手动计算 pred_z0 (x_start)，支持 Batch 内不同 Timesteps,详见DDPM的公式（15）
            # ========================================================
            # 1. 获取 Alphas Cumprod (bar_alpha)
            # 确保 alphas 在正确的设备上
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(z_t.device)
            
            # 2. 根据 t 取出对应的 alpha 值
            # alphas_cumprod[t] 形状是 [Batch]
            sqrt_alpha_prod = alphas_cumprod[t] ** 0.5
            sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[t]) ** 0.5
            
            # 3. 调整维度以支持广播 (Broadcasting)
            # z_t 的形状是 [Batch, 7, 256]
            # 我们需要把 alpha 变成 [Batch, 1, 1]
            while len(sqrt_alpha_prod.shape) < len(z_t.shape):
                sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
            
            # 4. 反解公式: z0 = (zt - sqrt(1-alpha)*eps) / sqrt(alpha)
            pred_original_sample = (z_t - sqrt_one_minus_alpha_prod * noise_pred) / sqrt_alpha_prod # torch.Size([32, 7, 256])
            # ========================================================


            pred_motion = self.vae.decode(pred_original_sample.permute(1,0,2), lengths) # torch.Size([32, 436, 263])
            pred_motion_denorm = pred_motion * self.std + self.mean
            pred_motion_denorm[..., :3] = 0.0 # 去掉位置信息，只看姿态
            pred_motion_denorm = pred_motion_denorm.permute(0,2,1) # [B, 263, T]
            # [B, 263, T] -> [B, 263, 1, T]
            pred_input = pred_motion_denorm.unsqueeze(2) # torch.Size([32, 263, 1, 436])
            motion_feat_pred = self.motionclip.encoder({
                'x': pred_input,
                'y': torch.zeros(bsz, dtype=int, device=self.device),
                'mask': lengths_to_mask(lengths, device=self.device)
            })["mu"] # [B, 512]

            logits = self.scene_classifier(motion_feat_pred)

            scene_ids = batch['scene_id']
            loss_scene = torch.nn.functional.cross_entropy(logits, scene_ids)

            n_set['loss_scene'] = loss_scene * self.cfg.SCENE_MODIFF_ABLATION.LAMBDA_SCENE
            n_set['style_mask'] = mask_style
            n_set['scene_mask'] = mask_scene
        
        return {**n_set}


    def train_diffusion_forward_finetune_baseline(self, batch):
        feats_ref = batch["motion"]
        feats_content = batch["motion"].clone()
        feats_content[...,:3] = 0.0
        lengths = batch["length"]
        
        # content condition
        with torch.no_grad():
            z, dist = self.vae.encode(feats_ref, lengths)
            z_content, dist = self.vae.encode(feats_content, lengths)
            cond_emb = z_content.permute(1,0,2)            
        # style condition
        motion_seq = feats_ref*self.std + self.mean
        motion_seq[...,:3]=0.0
        motion_seq = motion_seq.unsqueeze(-1).permute(0,2,3,1)
        motion_emb = self.motionclip.encoder({'x': motion_seq,
                        'y': torch.zeros(motion_seq.shape[0], dtype=int, device='cuda:{}'.format(self.cfg["DEVICE"][0])),
                        'mask': lengths_to_mask(lengths, device='cuda:{}'.format(self.cfg["DEVICE"][0]))})["mu"]
        motion_emb = motion_emb.unsqueeze(1)
        mask_uncond = torch.rand(motion_emb.shape[0]) < self.guidance_uncodp
        motion_emb[mask_uncond, ...] = 0
        


        # trans condition
        trans_cond = batch["motion"][...,:3]

        # three condition
        multi_cond_emb = [cond_emb, motion_emb, trans_cond]


        # diffusion process return with noise and noise_pred
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

            loss_diff = self.losses[split].update(rs_set)
            if loss_diff is None:
                raise ValueError(
                    "Loss is None, this happend with torchmetrics > 0.7")
            loss_scene = rs_set.get("loss_scene", torch.tensor(0.0).to(loss_diff.device)) 
            # 获取 scene_mask，它告诉我们哪些样本的 Scene 条件被 drop 掉了 (True)
            if split == 'train' and rs_set.get('scene_mask', None) is not None:
                scene_mask = rs_set['scene_mask'] # shape: [B]
            else:
                scene_mask = torch.zeros(rs_set['noise_pred'].shape[0], dtype=torch.bool, device=self.device)
            # is_scene_present 的地方是 True，代表 Scene 条件存在
            is_scene_present = ~scene_mask 
            
            # 如果当前 batch 中，没有任何一个样本是需要学习 Scene 的，
            # 那么 loss_scene 应该是 0，我们也不需要做任何特殊处理。
            if not is_scene_present.any():
                total_loss = loss_diff + loss_scene # loss_scene is 0 here
            else:
                # 【核心逻辑】
                # 对于那些 Scene 条件存在的样本，我们降低 loss_diff 的权重。
                # 这是一种“软性”的对比：我们不直接计算特征差异，而是调整学习目标。
                
                # 这是一个近似，但非常有效：
                # 我们计算一下这个 batch 里，有多少比例的样本是带 Scene 的
                scene_ratio = is_scene_present.float().mean()
                
                # 根据这个比例，我们动态调整 loss_diff 的权重
                # Scene 样本越多，loss_diff 的权重就越低
                diffusion_penalty_factor = self.cfg.SCENE_MODIFF_ABLATION.get("DIFFUSION_PENALTY", 0.5)
                
                # (1 - scene_ratio) * 1.0  => 对没有 scene 的样本，权重是 1.0
                # scene_ratio * penalty_factor => 对有 scene 的样本，权重是 penalty_factor
                # 这是一个平滑的、在 batch 级别上的权重调整
                dynamic_diff_weight = (1 - scene_ratio) * 1.0 + scene_ratio * diffusion_penalty_factor
                
                total_loss = loss_diff * dynamic_diff_weight + loss_scene

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
  
        # 5. 【关键】分别记录 Loss 到 TensorBoard
        # 这样你就能看到两条曲线：一条下降(重建)，一条下降(分类准确)
        if split == "train":
            # 记录总 Loss (带进度条)
            self.log('train/loss_total', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            # 记录 基础重建 Loss
            self.log('train/loss_diff', loss_diff, on_step=True, on_epoch=True, logger=True)
            
            # 记录 场景分类 Loss (如果是 Tensor 才记录)
            if isinstance(loss_scene, torch.Tensor):
                self.log('train/loss_scene', loss_scene, on_step=True, on_epoch=True, logger=True)
        
        elif split == "val":
            # 验证集同样的逻辑
            self.log('val/loss_total', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val/loss_diff', loss_diff, on_step=False, on_epoch=True, logger=True)
            if isinstance(loss_scene, torch.Tensor):
                self.log('val/loss_scene', loss_scene, on_step=False, on_epoch=True, logger=True)

        return total_loss
