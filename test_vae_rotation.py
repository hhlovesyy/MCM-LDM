import torch
import numpy as np
import os
import shutil
from mld.config import parse_args
from mld.models.get_model import get_model
from mld.data.get_data import get_datasets
from mld.utils.logger import create_logger

# 引入你的可视化工具 (假设路径是 visualization.vis_utils)
# 如果报错找不到，请修改为你项目实际的 visual_pos 位置
try:
    from visual import visual_pos 
except ImportError:
    # 简单的 fallback，防止报错阻断逻辑验证
    print("Warning: 'visual_pos' not found. Will skip MP4 generation.")
    def visual_pos(npy, mp4): pass

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import splprep, splev
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec  # 引入 GridSpec 用于复杂排版

# 1. 告诉 matplotlib 你的字体文件在哪里
font_path = '/root/autodl-tmp/MyRepository/MCM-LDM/paper_figures/simhei.ttf'  
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)

# 2. 全局设置 matplotlib 使用这个中文字体
plt.rcParams['font.sans-serif'] = prop.get_name() 
# 3. 解决坐标轴负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

def plot_robustness_curve(angles, mses):
    """
    绘制并保存 VAE 鲁棒性分析折线图
    """
    plt.figure(figsize=(10, 6), dpi=150)
    
    # 绘制主曲线
    plt.plot(angles, mses, color='#1f77b4', linewidth=2.5, marker='o', markersize=4, label='VAE 重建损失 (Avg)')
    
    # 设定区域
    # 0-30: Safe (Green)
    # 30-60: Trade-off (Orange) - 45度在这里
    # >60: Collapse (Red)
    plt.axvspan(0, 30, color='green', alpha=0.1, label='安全区域')
    plt.axvspan(30, 60, color='orange', alpha=0.1, label='权衡区域')
    plt.axvspan(60, 180, color='red', alpha=0.1, label='模式坍缩区域')
    
    # 标记 45度
    if 45 in angles:
        idx = angles.index(45)
        val = mses[idx]
        plt.axvline(x=45, color='red', linestyle='--', linewidth=2, label='最终选择 (45°)')
        plt.scatter([45], [val], color='red', s=100, zorder=5)
    
    plt.title('旋转增强对VAE重建损失的影响趋势', fontsize=14)
    plt.xlabel('旋转角度', fontsize=12)
    plt.ylabel('平均MSE损失', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    
    # 保存
    save_path = "vae_robustness_analysis.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\n✅ Plot saved to {save_path}")

def apply_rotation(features_phys, angle_deg, device):
    """
    对物理空间(反归一化后)的特征进行 Y 轴旋转。
    features_phys: [Batch, Frames, 263]
    """
    # 复制一份，不修改原数据
    feats = features_phys.clone()
    bs, frames, dims = feats.shape
    
    # 角度转弧度
    angle_rad = (angle_deg / 180.0) * np.pi
    # 创建旋转矩阵元素
    # 注意：这里我们对整个 batch 用同一个角度，方便观察
    # 如果想随机，可以用 torch.rand
    c = torch.cos(torch.tensor(angle_rad, device=device))
    s = torch.sin(torch.tensor(angle_rad, device=device))
    
    # 1. 旋转根节点线速度 (Root Linear Velocity, Index 1~2)
    # x' = x*c - z*s
    # z' = x*s + z*c
    root_vx = feats[..., 1].clone()
    root_vz = feats[..., 2].clone()
    feats[..., 1] = root_vx * c - root_vz * s
    feats[..., 2] = root_vx * s + root_vz * c
    
    # 2. 旋转关节位置 (Local Joint Positions, Index 4~67)
    # Shape: [21, 3] -> 63
    joint_pos = feats[..., 4:67].reshape(bs, frames, 21, 3)
    pos_x = joint_pos[..., 0].clone()
    pos_z = joint_pos[..., 2].clone()
    
    joint_pos[..., 0] = pos_x * c - pos_z * s
    joint_pos[..., 2] = pos_x * s + pos_z * c
    feats[..., 4:67] = joint_pos.reshape(bs, frames, -1)
    
    # 3. 旋转关节速度 (Local Joint Velocities, Index 193~259)
    # Shape: [22, 3] -> 66
    joint_vel = feats[..., 193:259].reshape(bs, frames, 22, 3)
    vel_x = joint_vel[..., 0].clone()
    vel_z = joint_vel[..., 2].clone()
    
    joint_vel[..., 0] = vel_x * c - vel_z * s
    joint_vel[..., 2] = vel_x * s + vel_z * c
    feats[..., 193:259] = joint_vel.reshape(bs, frames, -1)
    
    return feats

def test_angle(model, dataset, motion_raw, length, angle, output_dir, mean, std, render=False, index=0):
    """
    测试单个角度的重建效果
    """
    device = motion_raw.device
    print(f"\n>>> Testing Rotation Angle: {angle}°")
    
    # 1. 反归一化
    motion_phys = motion_raw * std + mean
    
    # 2. 旋转增强 (在物理空间)
    if angle != 0:
        motion_phys_rot = apply_rotation(motion_phys, angle, device)
    else:
        motion_phys_rot = motion_phys
        
    # 3. 再归一化 (变成 VAE 的输入)
    motion_input = (motion_phys_rot - mean) / std
    
    # 4. VAE Forward
    with torch.no_grad():
        # Encode
        z, _ = model.vae.encode(motion_input, [length])
        # Decode
        recon_output = model.vae.decode(z, [length]) # [1, T, 263]
    
    # 5. 计算 MSE (在 Normalized 空间计算，反映网络拟合能力)
    mse = torch.mean((motion_input - recon_output) ** 2).item()
    status = "✅ PASS" if mse < 0.1 else "❌ FAIL"
    print(f"    MSE Loss: {mse:.6f} {status}")
    
    # 6. 生成可视化文件 (在 Physical 空间)
    # GT (旋转后的输入)
    joints_gt = dataset.feats2joints(motion_input.cpu())
    gt_name = f"rot_{angle}_gt"
    if index == 10 and angle in [0, 30, 45, 90]:
        np.save(os.path.join(output_dir, f"{gt_name}.npy"), joints_gt[0].cpu().numpy())
    
    # Recon (VAE 的输出，需反归一化)
    # recon_phys = recon_output * std + mean
    recon_phys = recon_output
    joints_recon = dataset.feats2joints(recon_phys.cpu())
    recon_name = f"rot_{angle}_recon"
    if index == 10 and angle in [0, 30, 45, 90]:
        np.save(os.path.join(output_dir, f"{recon_name}.npy"), joints_recon[0].cpu().numpy())
    
    if render and index == 10:  # 只渲染index=10的那个样本
        # 7. 渲染视频
        print(f"    Rendering videos...")
        try:
            visual_pos(os.path.join(output_dir, f"{gt_name}.npy"), 
                    os.path.join(output_dir, f"{gt_name}.mp4"))
            visual_pos(os.path.join(output_dir, f"{recon_name}.npy"), 
                    os.path.join(output_dir, f"{recon_name}.mp4"))
        except Exception as e:
            print(f"    Warning: Visualization skipped due to error: {e}")

    return mse

def main():
    # 0. 准备工作
    output_dir = "test_vae_results"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    cfg = parse_args()
    logger = create_logger(cfg, phase="test")
    datasets = get_datasets(cfg, logger=logger)
    # 注意：这里需要拿到 Dataset 对象来调用 feats2joints
    # mld 代码中 datasets[0] 通常是 DataModule, train_dataset 是其中的属性
    if hasattr(datasets[0], 'train_dataset'):
        dataset_obj = datasets[0].train_dataset
        # 如果 dataset_obj 只是 dataset，没有 feats2joints，可能挂在 DataModule 上
        # 这里做一个兼容处理
        feat_converter = datasets[0] 
    else:
        # 如果 datasets[0] 本身就是 dataset
        dataset_obj = datasets[0]
        feat_converter = datasets[0]

    # 1. 加载模型 & 权重
    model = get_model(cfg, datasets[0])
    
    # === 你的 VAE 加载逻辑 ===
    vae_path = "/root/autodl-tmp/MyRepository/MCM-LDM/checkpoints/vae_checkpoint/vae7.ckpt"
    if not os.path.exists(vae_path):
        vae_path = '/root/autodl-tmp/MyRepository/MCM-LDM/checkpoints/vae_checkpoint/vae7.ckpt'
    
    print(f"Loading VAE from: {vae_path}")
    state_dict = torch.load(vae_path, map_location="cpu")["state_dict"]
    from collections import OrderedDict
    vae_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.split(".")[0] == "vae":
            name = k.replace("vae.", "")
            vae_dict[name] = v
    model.vae.load_state_dict(vae_dict, strict=True)
    print("VAE Weights Loaded!")
    # ========================

    model.eval().cuda()
    
    # 2. 准备数据
    # 找一个运动幅度大的样本 (比如 walking / running)
    # 简单遍历一下，找 root 速度大的
    target_idx = 0
    max_vel = 0
    
    # 这里的 mean/std 用于反归一化
    mean = torch.tensor(feat_converter.hparams.mean).cuda() if hasattr(feat_converter, 'hparams') else torch.tensor(dataset_obj.mean).cuda()
    std = torch.tensor(feat_converter.hparams.std).cuda() if hasattr(feat_converter, 'hparams') else torch.tensor(dataset_obj.std).cuda()

    # 3. 设置测试参数
    num_samples_to_test = 50 # 跑50个样本求平均，这样才严谨
    angles = list(range(0, 181, 5)) # 0, 5, 10 ... 180
    avg_mse_list = []
    
    print(f"Starting Robustness Analysis on {num_samples_to_test} samples...")
    print(f"Scanning Angles: {angles}")

    # 4. 循环测试
    for angle in angles:
        total_mse = 0.0
        
        # 对每个角度，跑 num_samples_to_test 个样本
        for i in range(num_samples_to_test):
            # 获取数据
            data = dataset_obj[i]
            if isinstance(data, dict):
                m = data["motion"]
                l = data["length"]
            else:
                m = torch.from_numpy(data[4])
                l = data[5]
            
            # 增加 Batch 维度并放到 GPU
            motion_raw = m.unsqueeze(0).cuda()
            
            # 调用你的 test_angle (render=False 加快速度)
            # 注意：这会反复覆盖 npy 文件，这是正常的，我们只需要最后的 MSE
            mse = test_angle(model, feat_converter, motion_raw, l, angle, output_dir, mean, std, render=True, index=i)
            total_mse += mse
            
        # 计算该角度的平均 MSE
        avg_mse = total_mse / num_samples_to_test
        avg_mse_list.append(avg_mse)
        print(f"Angle {angle}° : Avg MSE = {avg_mse:.6f}")

    # 5. 输出结果并绘图
    print("\n====== Final Data for Paper ======")
    print(f"Angles: {angles}")
    print(f"MSEs: {avg_mse_list}")
    
    # 调用绘图函数
    try:
        plot_robustness_curve(angles, avg_mse_list)
    except Exception as e:
        print(f"Plotting failed (maybe no matplotlib): {e}")

if __name__ == "__main__":
    main()