import torch
import numpy as np
import matplotlib.pyplot as plt

from os.path import join as pjoin

from common.skeleton import Skeleton
import numpy as np
import os
from common.quaternion import *
from paramUtil import *

import torch
from tqdm import tqdm
import os


def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions

# ==========================================
# Part 2: 我们的简化积分逻辑 (用于 Guidance/Generation)
# ==========================================
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

# ==========================================
# Part 3: 验证与绘图
# ==========================================
def verify_and_plot():
    # 1. 造假数据 (模拟一个左转画圆的动作)
    # seq_len = 100
    # batch_size = 1
    # fake_data = torch.zeros((batch_size, seq_len, 263))
    
    # # 设置速度: 类似我们之前的 circle 生成逻辑
    # # 假设每帧转 0.05 弧度，向前走 0.05 米
    # fake_data[..., 0] = 0.05 # Rot Vel
    # fake_data[..., 1] = 0.0  # Local X
    # fake_data[..., 2] = 0.05 # Local Z
    # fake_data[..., 3] = 0.0  # Height
    # 1. 加载数据
    npy_path = "/root/autodl-tmp/MyRepository/MCM-LDM/demo/content_motion/walk_and_turn.npy"
    fake_data = np.load(npy_path)  # 形状为 (161, 263)
    # 2. 转换为 Tensor (推荐：共享内存，速度最快)
    fake_tensor = torch.from_numpy(fake_data)
    fake_tensor = fake_tensor.unsqueeze(0)
    # 3. 如果需要转换数据类型（例如转为 Float32，这是深度学习最常用的格式）
    fake_tensor = fake_tensor.float()

    # 4. 如果需要移动到 GPU (AutoDL 环境常用)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fake_tensor = fake_tensor.to(device)
    
    # 2. 运行两种逻辑
    gt_pos = recover_from_ric(fake_tensor, 22)[:,:,0,:4] # torch.Size([1, 100, 3])
    my_pos = calculate_trajectory_correct(fake_tensor[..., :4])  # torch.Size([1, 100, 3])
    
    gt_np = gt_pos[0].detach().cpu().numpy()
    my_np = my_pos[0].detach().cpu().numpy()
    
    # 3. 计算误差
    mse = np.mean((gt_np - my_np)**2)
    print(f"MSE Difference: {mse:.8f}")
    
    if mse < 1e-6:
        print("✅ SUCCESS: 简化逻辑与官方逻辑完美匹配！")
    else:
        print("❌ WARNING: 逻辑存在偏差，请检查旋转公式。")

    # 4. 画图对比
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Comparison (XZ Plane)")
    plt.plot(gt_np[:, 0], gt_np[:, 2], 'r--', linewidth=3, label='GT (recover_from_ric)')
    plt.plot(my_np[:, 0], my_np[:, 2], 'b-', linewidth=1, label='Ours (Simple Calc)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.title("Coordinate Difference (Error)")
    plt.plot(gt_np[:, 0] - my_np[:, 0], label='Diff X')
    plt.plot(gt_np[:, 2] - my_np[:, 2], label='Diff Z')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("verification_result.png")
    print("对比图已保存至 verification_result.png")

if __name__ == "__main__":
    verify_and_plot()