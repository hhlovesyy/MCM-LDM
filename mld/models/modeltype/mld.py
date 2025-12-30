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

DEFAULT_SCALAR_VAL = 2.0 

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
    
    def _get_current_stage_params(self):
        """
        根据当前 global_step 或 current_epoch 获取配置参数。
        这样就不需要在 forward 里写一大堆 if epoch < x else ...
        """
        # 如果是微调 Baseline 模式，直接返回特定标记
        if self.cfg.SCENEMODIFF_GLOBAL_CONFIG.JUST_FINETUNE_BASELINE:
             return {"is_baseline_finetune": True}

        curr_epoch = self.current_epoch
        stages = self.cfg.CURRICULUM.STAGES
        stage_cfg = stages[-1] # 默认取最后一个
        for stage in stages:
            if stage['EPOCH_START'] <= curr_epoch < stage['EPOCH_END']:
                stage_cfg = stage
                break
        return stage_cfg
    
    def _encode_scene_condition(self, batch):
        """处理 Text/Image/Select 的复杂逻辑，返回统一的 [B, 1, 512]"""
        scene_texts = batch.get("scene_text", [""] * len(batch["length"]))
        scene_images = batch.get("scene_image", None)
        has_image = batch.get('has_image', None)
        
        mode = self.cfg.SCENEMODIFF_GLOBAL_CONFIG_TRAIN.MULTI_MODAL_FUSION
        
        # 逻辑判定：到底用哪个模态
        use_image = False
        if mode == "image":
            use_image = (scene_images is not None) and (has_image.all())
        elif mode == "mixed":
            # 这里的 0.5 也可以做到 yaml 里，暂时硬编码
            use_image = (scene_images is not None) and (has_image.all()) and (torch.rand(1).item() < 0.5)
            
        # 具体的编码逻辑
        if use_image:
            with torch.no_grad():
                vision_out = self.scene_vision_encoder(pixel_values=scene_images)
            scene_feat_raw = self.scene_image_projector(vision_out.pooler_output)
        else:
            with torch.no_grad():
                text_inputs = self.scene_tokenizer(scene_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                text_out = self.scene_text_encoder(**text_inputs)
            scene_feat_raw = self.scene_projector(text_out.pooler_output)

        return scene_feat_raw.unsqueeze(1) # [B, 1, 512]
    

    def _generate_condition_masks(self, bsz, strategy_cfg):
        """根据当前阶段的策略生成 mask"""
        mask_style = torch.zeros(bsz, dtype=torch.bool, device=self.device)
        mask_scene = torch.zeros(bsz, dtype=torch.bool, device=self.device)
        
        strategy_name = strategy_cfg.get('STRATEGY', 'probabilistic_mix')
        
        if strategy_name == 'keep_style_only':
            mask_scene[:] = True # 把 Scene 全遮住
            probs = torch.rand(bsz, device=self.device)
            mask_style[probs > (1.0-self.cfg.SCENEMODIFF_GLOBAL_CONFIG.STEP1_DROP_STYLE_PROB)] = True # 10% 概率 Drop Style
            
        elif strategy_name == 'probabilistic_mix':
            # 读取配置里的阈值，代替硬编码的 0.5, 0.7, 0.9
            probs = torch.rand(bsz, device=self.device)
            thresholds = strategy_cfg.get('PROBS', [0.5, 0.7, 0.9]) # p1, p2, p3
            
            # [0, p1): Keep Both (全 False)
            # [p1, p2): Drop Style
            mask_style[(probs >= thresholds[0]) & (probs < thresholds[1])] = True
            
            # [p2, p3): Drop Scene
            mask_scene[(probs >= thresholds[1]) & (probs < thresholds[2])] = True
            
            # [p3, 1.0): Drop Both (Uncond)
            mask_style[probs >= thresholds[2]] = True
            mask_scene[probs >= thresholds[2]] = True
            
        return mask_style, mask_scene
    
    def _apply_film_fusion(self, style_emb, scene_feat, mask_scene):
        """
        处理 FiLM 融合逻辑，包含对 mask_scene 的特殊处理
        """
        scene_feat_norm = self.scene_norm(scene_feat)
        film_params = self.film_mlp(scene_feat_norm)
        
        # 核心逻辑：确保 Mask 掉 Scene 时，FiLM 参数失效 (退化为 Identity)
        mask_scene_expanded = mask_scene.unsqueeze(1).unsqueeze(2).float()
        film_params = film_params * (1 - mask_scene_expanded)
        
        gamma_raw, beta_raw = film_params.chunk(2, dim=-1)
        gamma = (1.0 + torch.tanh(gamma_raw))
        beta = beta_raw
        
        return gamma * style_emb + beta

    def _apply_mlp_fusion(self, style_emb, scene_feat, mask_scene):
        style_feat = style_emb.squeeze(1)
        scene_feat_norm = self.scene_norm(scene_feat.squeeze(1))
        combined_feat = torch.cat([style_feat, scene_feat_norm], dim=1) # shape: [B, 1024]
        fused_emb = self.fusion_mlp(combined_feat) # shape: [B, 512]
        adapted_style_emb = fused_emb.unsqueeze(1)
        return adapted_style_emb

    def _create_sparse_content(self, feats_content, lengths, sparsity_cfg):
        """根据配置生成稀疏的 content 特征"""
        if not sparsity_cfg.get('ENABLED', False):
            return feats_content

        mode = sparsity_cfg.get('MODE', 'first_frame')
        bsz, seq_len, _ = feats_content.shape
        content_mask = torch.zeros_like(feats_content, dtype=torch.bool)
        
        if mode == 'first_frame':
            content_mask[:, 0, :] = True
            
        elif mode == 'key_frames':
            num_frames = sparsity_cfg.get('NUM_KEY_FRAMES', 5)
            for i in range(bsz):
                indices = torch.linspace(0, lengths[i] - 1, num_frames, dtype=torch.long)
                content_mask[i, indices, :] = True
                
        # feats_content 中只有 mask 为 True 的地方保留原值，其余为 0
        return feats_content * content_mask

    def _compute_scene_guidance_loss(self, n_set, lengths, target_scene_ids):
        """
        从 noisy latents 恢复 x0 并计算分类 Loss
        """
        z_t = n_set['noisy_latents']
        t = n_set['timesteps']
        noise_pred = n_set['noise_pred']
        
        # 1. 恢复 z0 (使用 reparameterization trick 的逆过程)
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(z_t.device)
        sqrt_alpha_prod = alphas_cumprod[t] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[t]) ** 0.5
        
        # 维度对齐 broadcasting
        while len(sqrt_alpha_prod.shape) < len(z_t.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
            
        pred_original_sample = (z_t - sqrt_one_minus_alpha_prod * noise_pred) / sqrt_alpha_prod
        
        # 2. VAE Decode -> MotionCLIP Encode
        # 注意：这里可能需要 detach gradient 吗？通常引导 loss 是需要梯度传回 diffusion 的，
        # 但不需要传回 VAE 或 MotionCLIP。根据你的需求决定是否加 detach。
        # 原代码没加，这里保持原样。
        pred_motion = self.vae.decode(pred_original_sample.permute(1,0,2), lengths)
        pred_motion_denorm = pred_motion * self.std + self.mean
        pred_motion_denorm[..., :3] = 0.0 # 去根位置
        
        pred_input = pred_motion_denorm.permute(0,2,1).unsqueeze(2) # [B, 263, 1, T]
        
        motion_feat_pred = self.motionclip.encoder({
            'x': pred_input,
            'y': torch.zeros(len(t), dtype=int, device=self.device),
            'mask': lengths_to_mask(lengths, device=self.device)
        })["mu"]

        # 3. Classifier Loss
        logits = self.scene_classifier(motion_feat_pred)
        loss = torch.nn.functional.cross_entropy(logits, target_scene_ids)
        return loss
    

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
    
    def _prepare_trajectory_input(self, batch, scene_data):
        """
        处理轨迹输入：从 SceneData 生成稠密轨迹，或从 Batch 获取，并进行物理特征提取。
        返回: 
            trans_cond_input: [B, T, 4] (归一化后的物理特征，用于 Condition)
            target_global_pos: [B, T, 3] (世界坐标 XYZ，用于 Guidance Loss)
        """
        device = batch['content_motion'].device
        bs = batch['content_motion'].shape[0]
        lengths = batch['length']
        dims_to_mask = 4 if self.cfg.SCENEMODIFF_GLOBAL_CONFIG.ROOT_MASKING_DIM4 else 3

        # 1. 如果没有 Scene Data 或不启用轨迹，直接回退到 Content 自身的轨迹
        if scene_data is None or not self.cfg.TRAJECTORY.ENABLED:
            trans_motion = batch['content_motion'].clone() # torch.Size([6, 179, 263])
            if self.cfg.TRAJECTORY.ROOT_MASKING_DIM4:
                trans_cond_input = trans_motion[..., :4]
            else:
                trans_cond_input = trans_motion[..., :3]
            # 这种情况下没有 Target Guidance
            return trans_cond_input, None

        raw_content = batch['content_motion'].clone() 
        bs = raw_content.shape[0]
        content_npy = raw_content[0].detach().cpu().numpy()
        dist_profile = self.traj_processor.calculate_cumulative_distance(content_npy)

        import copy
        from torch.nn.utils.rnn import pad_sequence
        
        # 1. 获取轨迹配置和障碍物
        traj_config = copy.deepcopy(scene_data['trajectory'])
        obstacles = copy.deepcopy(scene_data['environment']['obstacles'])
        # 2. 坐标归一化 (将起点的 (x,z) 归零)
        # 无论 A* 还是手绘，都把第一个点作为 (0,0) 参考系
        waypoints = traj_config['points'] # 这里不管是 A*的点 还是 手绘的点，都是 list
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
        # 3. 生成稠密曲线 (Dense Curve)
        dense_curve = self.traj_processor.get_path_from_config(traj_config, obstacles)
        
        # 2. 逐样本计算轨迹 (Loop over Batch)
        list_target_pos = []
        list_trans_cond = []
        
        for i in range(bs): # 以2个content 4个style为例
            # 获取当前样本的真实长度
            curr_len = lengths[i]  # 是第i个任务的content的动作长度：199
            # 取出有效数据
            curr_content = raw_content[i, :curr_len].detach().cpu().numpy() # shape:(199, 263)
            
            # 计算该样本的距离分布
            dist_profile = self.traj_processor.calculate_cumulative_distance(curr_content)  #shape:(200,)
            target_dists = dist_profile[1:][:curr_len] # 对齐长度 shape:(199,) target_dists = dist_profile[1 : 1 + curr_len]这么写可能更好，每一帧应该累加走的距离，第一帧就有值
            
            # 重采样
            resampled_pts, _ = self.traj_processor.resample_by_arc_length(dense_curve, target_dists) # shape:(199, 2)
            
            # 计算物理特征
            trans_cond_phys, _ = self.traj_processor.compute_trajectory_features(resampled_pts)
            trans_cond_phys = trans_cond_phys[..., :dims_to_mask]

            # 构造 Target Pos
            safe_len = min(len(resampled_pts), curr_len)
            traj_3d = np.zeros((curr_len, 3))
            traj_3d[:safe_len, 0] = resampled_pts[:safe_len, 0]
            traj_3d[:safe_len, 2] = resampled_pts[:safe_len, 1]
            
            list_target_pos.append(torch.from_numpy(traj_3d).float().to(self.device))
            list_trans_cond.append(torch.from_numpy(trans_cond_phys[:safe_len]).float().to(self.device))
            
        # ================= [修复 Padding 逻辑] =================
        # 1. 获取当前 Batch 的最大长度
        # 注意：raw_content 可能是 batch 里最长的，也可能因为切片变短了，以 list 里最长的为准
        max_len = max([t.shape[0] for t in list_target_pos]) # 199
        
        # 2. 对 Target Global Pos 做 "Edge Padding" (补最后一帧)
        padded_target_pos_list = []
        for t in list_target_pos:
            curr_len = t.shape[0]
            diff = max_len - curr_len
            if diff > 0:
                # 取最后一帧 [1, 3]
                last_frame = t[-1:] 
                # 复制 diff 次
                padding = last_frame.repeat(diff, 1)
                # 拼接到后面 -> 效果：走完了就停在原地
                t_padded = torch.cat([t, padding], dim=0)
            else:
                t_padded = t
            padded_target_pos_list.append(t_padded)
        
        target_global_pos = torch.stack(padded_target_pos_list) # torch.Size([8, 199, 3])
        
        # 3. 对 Trans Cond 做 "Zero Padding" (补 0)
        # 物理特征(速度、旋转角速度)补 0 是对的，代表"停止运动"
        trans_tensor = pad_sequence(list_trans_cond, batch_first=True, padding_value=0.0) # torch.Size([8, 199, 4])
        mean_cond = self.mean.to(device)[..., :dims_to_mask]
        std_cond = self.std.to(device)[..., :dims_to_mask]
        trans_cond_input = (trans_tensor - mean_cond) / std_cond 
        return trans_cond_input, target_global_pos  # trans_cond_input: torch.Size([6, 179, 4]), target_global_pos:torch.Size([6, 179, 3])

    def _encode_conditions(self, batch, trans_cond_input, lengths):
        """
        编码所有条件：Content(VAE), Style(MotionCLIP+Scene), Trajectory
        返回: multi_cond_emb (列表)
        """
        bsz = batch['content_motion'].shape[0]
        device = batch['content_motion'].device
        content_motion = batch['content_motion'].clone()
        content_motion = (content_motion - self.mean.to(device)) / self.std.to(device)
        dims_to_mask = 4 if self.cfg.SCENEMODIFF_GLOBAL_CONFIG.ROOT_MASKING_DIM4 else 3
        content_motion[..., :dims_to_mask] = 0.0
        with torch.no_grad():
            z, _ = self.vae.encode(content_motion.float(), lengths)
            motion_emb_content = torch.cat([z, z], dim=1).permute(1, 0, 2) # torch.Size([12, 7, 256])
        
        style_motion = batch["style_motion"].clone()
        # style_motion[..., :dims_to_mask] = 0.0 
        # 本来的训练和推理都是只mask了前三维，这里也先这样吧：
        style_motion[..., :3] = 0.0
        if "style_length" in batch:
            lengths11 = batch["style_length"] # [199, 199, 199, 199, 199, 199]
        else:
            lengths11 = [style_motion.shape[1]] * style_motion.shape[0]
        motion_seq = style_motion.unsqueeze(-1).permute(0,2,3,1) # torch.Size([6, 263, 1, 199]),训练的代码debug之后的结果是# torch.Size([32, 263, 1, 40])，所以是能对上的


        motion_emb = self.motionclip.encoder({'x': motion_seq.float(),
                        'y': torch.zeros(motion_seq.shape[0], dtype=int, device=motion_seq.device),
                        'mask': lengths_to_mask(lengths11, device=motion_seq.device)})["mu"]  # torch.Size([6, 512])
        motion_emb = motion_emb.unsqueeze(1) # torch.Size([6, 1, 512])

        use_scene = self.cfg.SCENEMODIFF_GLOBAL_CONFIG_TRAIN.get('USE_SCENE_CLS', False)
        # 这里加一个消融开关：如果在做消融实验，强制把 use_scene 置 False
        if batch.get('ablation_no_scene', False):
            use_scene = False
            
        if use_scene:
            scene_feat = self._encode_scene_condition(batch) # 复用训练代码
        else:
            scene_feat = torch.zeros(bsz, 1, 512, device=device) # 必须给个 0 占位

        # Fusion (FiLM / MLP)
        motion_emb_cond = motion_emb # torch.Size([6, 1, 512])
        # 伪造 mask (推理时全 False)
        dummy_mask = torch.zeros(bsz, dtype=torch.bool, device=device)
        
        if self.cfg.SCENE_MODIFF_ABLATION.FUSION_MODE == "film":
            adapted_style = self._apply_film_fusion(motion_emb_cond, scene_feat, dummy_mask)
        else:
            adapted_style = self._apply_mlp_fusion(motion_emb_cond, scene_feat)
            
        adapted_style = self.style_norm(adapted_style)

        uncond_style = torch.zeros_like(adapted_style)
        motion_emb_cfg = torch.cat([uncond_style, adapted_style], dim=0)
        uncond_trans = torch.cat([trans_cond_input, trans_cond_input], dim=0)

        return [motion_emb_content, motion_emb_cfg, uncond_trans]


    def forward(self, batch, scene_data=None):
        ''' 推理重构之后的新入口 '''
        # 1.基础信息
        lengths = batch["length"] # [161, 161, 179, 179, 38, 38]
        scale = batch["tag_scale"] # 2.5
        trans_cond_input, target_global_pos = self._prepare_trajectory_input(batch, scene_data)
        multi_cond_emb = self._encode_conditions(batch, trans_cond_input, lengths)
        
        z = self._diffusion_reverse(
            multi_cond_emb, 
            lengths, 
            scale, 
            target_global_pos=target_global_pos, 
            scene_data=scene_data
        )

        # 5. 解码
        with torch.no_grad():
            feats_rst = self.vae.decode(z, lengths) # torch.Size([6, 179, 263])
        
        joints = self.feats2joints(feats_rst.detach().cpu())  # torch.Size([6, 179, 22, 3])
        import copy
        from torch.nn.utils.rnn import pad_sequence
        
        # 1. 获取轨迹配置和障碍物
        traj_config = copy.deepcopy(scene_data['trajectory'])
        obstacles = copy.deepcopy(scene_data['environment']['obstacles'])
        # 2. 坐标归一化 (将起点的 (x,z) 归零)
        # 无论 A* 还是手绘，都把第一个点作为 (0,0) 参考系
        waypoints = traj_config['points'] # 这里不管是 A*的点 还是 手绘的点，都是 list
        startPosX, startPosY = waypoints[0][0], waypoints[0][1]

        if startPosX != 0 or startPosY != 0:
            # 只平移根节点和所有子节点的位置
            # 0 是 x 轴, 2 是 z 轴 (根据你的 trajectory 生成逻辑)
            joints[..., 0] += startPosX
            joints[..., 2] += startPosY
        
        # 同时，如果 target_global_pos 需要用于可视化 debug，也加回去
        if target_global_pos is not None:
             target_global_pos[..., 0] += startPosX
             target_global_pos[..., 2] += startPosY
        
        if True and False: 
            try:
                # 0. 准备工作
                batch_size = joints.shape[0]
                timestamp = int(time.time())
                
                # 确保输出目录存在
                os.makedirs("vis_debug", exist_ok=True)

                # 遍历 Batch 中的每一个样本
                for i in range(batch_size):
                    # 1. 提取生成的根节点轨迹
                    # joints 通常是 [Batch, Joints, 3, Length] -> [B, 22, 3, L]
                    if joints.shape[1] == 22 or joints.shape[1] == 21: 
                        # 取第 i 个样本，第 0 个关节(Root)，所有维度
                        pred_root_traj = joints[i, 0, :, :].permute(1, 0) # [3, L] -> [L, 3]
                    else:
                        # 备用情况: 假设维度是 [B, L, J, 3] 或者其他变体
                        # 这里的逻辑根据之前的 else 修改，确保取到第 i 个
                        # print(f"Joints shape check: {joints.shape}")
                        pred_root_traj = joints[i, ..., 0, :].squeeze() 

                    pred_root_traj_np = pred_root_traj.detach().cpu().numpy()
                    
                    # 2. 提取目标轨迹 (如果存在)
                    target_traj_np = None
                    if 'target_global_pos' in locals() and target_global_pos is not None:
                        # target_global_pos 是 [Batch, Length, 3]
                        target_traj_np = target_global_pos[i].detach().cpu().numpy()
                    
                    # 3. 对齐长度 (取两者较小值，防止画图越界)
                    min_len = len(pred_root_traj_np)
                    if target_traj_np is not None:
                        min_len = min(len(pred_root_traj_np), len(target_traj_np))
                    
                    # 4. 调用画图
                    # 文件名格式: traj_debug_{时间戳}_sample_{序号}.png
                    save_path = f"vis_debug/traj_debug_{timestamp}_sample_{i}.png"
                    
                    debug_plot_trajectory(
                        target_traj_np[:min_len] if target_traj_np is not None else None, 
                        pred_root_traj_np[:min_len],
                        # 假设 scene_data 对整个 Batch 是通用的，直接传入
                        # 如果 scene_data 也是 batch list，则需要改成 scene_data[i]
                        scene_data = scene_data,
                        save_path = save_path,
                        interval = self.cfg.TRAJECTORY.GUIDANCE.WAYPOINTS_INTERVAL
                    )
                
                # 打印一次提示即可
                print(f"[Debug] Batch Visualization saved {batch_size} images to vis_debug/")
                    
            except Exception as e:
                print(f"[Warning] Failed to plot trajectory: {e}")
                # 打印详细报错方便调试
                import traceback
                traceback.print_exc()
        # =======================================================
        # =======================================================


        return remove_padding(joints, lengths), target_global_pos
    
    def _parse_cfg_scales(self, scale, cfg_factor):
        """解析 CFG Scale 参数"""
        scale_style = scale
        scale_scene = scale
        
        if isinstance(scale, (list, tuple)):
            if len(scale) >= 2:
                scale_style = scale[0]
                scale_scene = scale[1]
            else:
                scale_style = scale[0]
                scale_scene = scale[0]
                
        # 如果是 3 倍模式，且只给了一个 float，默认两者相等
        # 这个逻辑在调用处已经隐含处理了
        return scale_style, scale_scene

    def _apply_spatial_guidance(self, latents, t, ctx):
        """
        计算并应用基于梯度的空间引导 (Waypoints & Obstacles)。
        """
        # 1. 全局开关检查
        conf = self.cfg.TRAJECTORY.GUIDANCE
        if not conf.ENABLED:
            return latents

        # 2. 时间窗口检查 (Guard Clause)
        # 只有在特定的去噪阶段才进行引导
        if not (conf.GUIDACE_END < t < conf.GUIDANCE_START):
            return latents

        # 3. 确定优化步数 (Dynamic K Strategy)
        # 将硬编码的逻辑保留在这里，或者提取到配置中
        if t > 500: num_opt_steps = 1
        elif t > 100: num_opt_steps = 5
        else: num_opt_steps = 10

        # 4. 梯度下降循环
        # 注意：这里我们是在冻结模型的情况下，通过梯度修改 latents
        current_latents = latents.detach().requires_grad_(True)
        
        with torch.enable_grad():
            for _ in range(num_opt_steps):
                total_loss = 0.0
                
                # A. 路点引导 (Waypoints)
                if conf.WAYPOINTS_MODE and ctx['target_pos'] is not None:
                    # 这里的 compute_spatial_loss 是你原来的 compute_spatial_guidance 里的 loss 计算部分
                    # 需要你把它拆出来，只返回 loss，不要在里面求导
                    loss_traj = self._compute_waypoint_loss(
                        current_latents, t.unsqueeze(0).repeat(ctx['bsz']), ctx['target_pos'], ctx['cond_embeddings'], ctx['lengths'],
                        interval = self.cfg.TRAJECTORY.GUIDANCE.WAYPOINTS_INTERVAL
                    )
                    total_loss += loss_traj * conf.WAYPOINTS_GUIDE_STRENGTH

                # B. 避障引导 (Obstacles)
                if conf.OBSTACLE_MODE and len(ctx['obstacles']) > 0:
                    loss_obs = self._compute_obstacle_loss(
                        current_latents, t, ctx['obstacles'], ctx['cond_embeddings'], ctx['lengths']
                    )
                    total_loss += loss_obs * conf.OBSTACLE_GUIDE_STRENGTH
                
                # 如果没有 Loss，直接退出
                if isinstance(total_loss, float) and total_loss == 0.0:
                    break

                # C. 反向传播
                grad = torch.autograd.grad(total_loss, current_latents)[0]

                # D. 梯度裁剪 (Gradient Clipping)
                grad_norm = grad.norm()
                max_norm = 5.0 # 可以写进配置
                if grad_norm > max_norm:
                    grad = grad * (max_norm / (grad_norm + 1e-8))

                # E. 更新 Latents
                # alpha 缩放: 随着 t 变小(接近真实图像)，梯度的权重应该变小
                # 或者直接用 step_size = 1.0
                # scale_factor = (1 - self.scheduler.alphas_cumprod[t]) ** 0.5
                step_size = 1.0 
                current_latents = current_latents - step_size * grad
                current_latents = current_latents.detach().requires_grad_(True)
                
        # 5. 返回更新后的 Latents (不再需要梯度)
        return current_latents.detach()
    
    def _compute_cfg_noise(self, latents, t, encoder_hidden_states, lengths, cfg_factor, scale_style, scale_scene):
        """
        执行模型前向传播，并根据 cfg_factor (1, 2, 3) 计算最终噪声。
        """
        # 1. 扩展输入
        # 如果 cfg_factor > 1，需要复制 latents
        if cfg_factor > 1:
            latent_input = torch.cat([latents] * cfg_factor, dim=0)
            lengths_input = lengths * cfg_factor
        else:
            latent_input = latents
            lengths_input = lengths

        # 2. 模型前向 (Model Forward)
        # 注意：Denoiser 不需要知道我们在做 CFG，它只管处理 Batch
        noise_pred = self.denoiser(
            sample=latent_input,
            timestep=t,
            encoder_hidden_states=encoder_hidden_states,
            lengths=lengths_input,
        )[0] # torch.Size([12, 7, 256])

        # 3. 应用 CFG 公式
        if cfg_factor == 1:
            return noise_pred
            
        elif cfg_factor == 2:
            # 标准 CFG: [Uncond, Cond]
            noise_uncond, noise_text = noise_pred.chunk(2, dim=0)
            return noise_uncond + scale_style * (noise_text - noise_uncond)
            
        elif cfg_factor == 3:
            # 双重引导 (ICME 核心): [Uncond, Style, Mix]
            noise_uncond, noise_style, noise_mix = noise_pred.chunk(3, dim=0)
            
            # 组合逻辑：
            # Base = Uncond
            # + Style 方向 (从 Uncond 指向 Style)
            # + Scene 方向 (从 Style 指向 Mix)
            return noise_uncond + \
                   scale_style * (noise_style - noise_uncond) + \
                   scale_scene * (noise_mix - noise_style)
        
        else:
            raise ValueError(f"Unsupported CFG factor: {cfg_factor}")

    def _diffusion_reverse(self, encoder_hidden_states, lengths=None, scale=None, target_global_pos=None, scene_data=None):
        bsz = len(lengths)
        device = encoder_hidden_states[0].device
        # 确定 CFG 模式 (1倍, 2倍, 3倍)
        total_bsz = encoder_hidden_states[0].shape[0]
        cfg_factor = total_bsz // bsz
        # 解析 Scale (如果是双重引导，scale 可能是个 tuple)
        scale_style, scale_scene = self._parse_cfg_scales(scale, cfg_factor)
        latents = torch.randn((bsz, self.latent_dim[0], self.latent_dim[-1]), device=device, dtype=torch.float)
        latents = latents * self.scheduler.init_noise_sigma

        self.scheduler.set_timesteps(self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(device)
        
        # 准备引导所需的静态数据 (避免在循环里重复提取)
        guidance_context = {
            'target_pos': target_global_pos,
            'obstacles': scene_data['environment']['obstacles'] if scene_data else [],
            'cond_embeddings': [h[bsz:] for h in encoder_hidden_states], # 剥离出 cond 部分用于引导
            'lengths': lengths,
            'bsz': bsz
        }

        for i, t in enumerate(timesteps):
            # Step 1: 空间引导 (Spatial Guidance) - 修改 Latent 位置
            # 这里的 if 逻辑被封装在函数内部，主循环不需要关心
            latents = self._apply_spatial_guidance(latents, t, guidance_context)

            # Step 2: 噪声预测 (Noise Prediction with CFG) - 预测噪声
            noise_pred = self._compute_cfg_noise(
                latents, t, encoder_hidden_states, lengths, 
                cfg_factor, scale_style, scale_scene
            )  # torch.Size([6, 7, 256])

            # Step 3: 调度器步进 (Scheduler Step) - 走向下一步
            # 处理 eta (DDIM)
            extra_kwargs = {}
            if "eta" in inspect.signature(self.scheduler.step).parameters:
                extra_kwargs["eta"] = self.cfg.model.scheduler.eta
                
            latents = self.scheduler.step(noise_pred, t, latents, **extra_kwargs).prev_sample

        # 维度调整 [B, C, T] -> [B, T, C]
        latents = latents.permute(1, 0, 2) # torch.Size([7, 6, 256])
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
    # TODO[0]:Scene Classifier可以考虑做在latent space当中，因为梯度链太长了，模型可能学不到东西，同时可以减小显存的占用
    def train_diffusion_forward(self, batch):
        stage_cfg = self._get_current_stage_params()
        if self.cfg.SCENEMODIFF_GLOBAL_CONFIG.JUST_FINETUNE_BASELINE:
            if self.global_step == 0:
                print("Just finetune baseline diffusion model...")
            return self.train_diffusion_forward_finetune_baseline(batch)
        
        feats_ref = batch["motion"]
        lengths = batch["length"]
        bsz = feats_ref.shape[0]

        # 关于训练的时候的旋转增强以及随机mask掉的策略
        feats_content = feats_ref.clone()
        traj_cfg = stage_cfg['TRAJECTORY']
        current_mean = self.mean.to(feats_content.device)
        current_std = self.std.to(feats_content.device)
        if traj_cfg.get('ROTATION_AUG', False):
            angle = traj_cfg.get('ROTATION_RANGE', 0) # 如果yaml里没写 range，默认0或者去全局cfg取
            feats_content_phys = feats_content * current_std + current_mean  # 在反归一化的空间做完旋转增强后，再归一化回来，因为x和z方向的mean和std不一样，会让模型生成长短不一的腿
            # 这里调用你原来的 augment 函数，假设它叫 augment_content_rotation
            feats_content_rotated_phys = self.augment_content_rotation(feats_content_phys, angle)
            feats_content = (feats_content_rotated_phys - current_mean) / current_std
        
        dims_to_mask = 4 if self.cfg.SCENEMODIFF_GLOBAL_CONFIG.ROOT_MASKING_DIM4 else 3
        feats_content[..., :dims_to_mask] = 0.0
        
        sparsity_cfg = stage_cfg['TRAJECTORY'].get('CONTENT_SPARSITY', {})
        feats_content_sparse = self._create_sparse_content(feats_content, lengths, sparsity_cfg)
        
        # 关于content drop
        mask_content_drop = None
        drop_prob = traj_cfg.get('CONTENT_DROPOUT_PROB', 0.0)
        if drop_prob > 0:
            mask_content_drop = torch.rand(bsz, device=self.device) < drop_prob

        # 编码 content（VAE）， style（Frozen MotionCLIP）
        with torch.no_grad():
            # A. Content Encoding
            z, _ = self.vae.encode(feats_ref, lengths)
            z_content, _ = self.vae.encode(feats_content_sparse, lengths)
            cond_emb = z_content.permute(1, 0, 2)
            
            # 应用 Content Dropout
            if mask_content_drop is not None:
                cond_emb[mask_content_drop] = 0.0

            # B. Style Encoding (MotionCLIP)
            motion_seq = feats_ref * current_std + current_mean
            motion_seq[..., :3] = 0.0 # 去除根节点位移
            motion_seq = motion_seq.unsqueeze(-1).permute(0, 2, 3, 1)
            
            motion_emb_raw = self.motionclip.encoder({
                'x': motion_seq,
                'y': torch.zeros(bsz, dtype=int, device=feats_content.device),
                'mask': lengths_to_mask(lengths, device=feats_content.device)
            })["mu"].unsqueeze(1) # [B, 1, 512]

        # C. Scene Encoding (Text or Image)，这一步需要梯度
        # 优化：如果是纯轨迹阶段，根本不需要跑 Scene Encoder
        if stage_cfg['MASKING']['STRATEGY'] == 'keep_style_only':
            scene_feat = torch.zeros(bsz, 1, 512, device=feats_content.device)
        else:
            scene_feat = self._encode_scene_condition(batch)
        mask_style, mask_scene = self._generate_condition_masks(bsz, stage_cfg['MASKING'])

        motion_emb = motion_emb_raw.clone()
        motion_emb[mask_style] = 0
        
        curr_scene_feat = scene_feat.clone()
        curr_scene_feat[mask_scene] = 0

        if stage_cfg['MASKING']['STRATEGY'] == 'keep_style_only':
            adapted_style_emb = motion_emb
        else:
            # 执行融合 (FiLM / MLP)
            if self.cfg.SCENE_MODIFF_ABLATION.FUSION_MODE == "film":
                adapted_style_emb = self._apply_film_fusion(motion_emb, curr_scene_feat, mask_scene)
            elif self.cfg.SCENE_MODIFF_ABLATION.FUSION_MODE == "mlp":
                adapted_style_emb = self._apply_mlp_fusion(motion_emb, curr_scene_feat)
            else:
                raise ValueError("Unknown fusion mode")
            
        adapted_style_emb = self.style_norm(adapted_style_emb)
        trans_cond = batch["motion"][..., :dims_to_mask]
        multi_cond_emb = [cond_emb, adapted_style_emb, trans_cond]
        n_set = self._diffusion_process(z, multi_cond_emb, lengths)

        lambda_scene = stage_cfg['LOSS'].get('LAMBDA_SCENE', 0.0)
        if lambda_scene > 0 and self.cfg.SCENEMODIFF_GLOBAL_CONFIG_TRAIN.USE_SCENE_CLS:
            loss_scene = self._compute_scene_guidance_loss(n_set, lengths, batch['scene_id'])
            n_set['loss_scene'] = loss_scene * lambda_scene
            # 记录 mask 用于 debug
            n_set['style_mask'] = mask_style
            n_set['scene_mask'] = mask_scene
        
        return n_set

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
            total_loss = loss_diff + loss_scene

            # 之前下面这个操作可能是迷惑而没有意义的，维持最佳的a+λb可能就是最好的了，不要把diff项再去掉了

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
