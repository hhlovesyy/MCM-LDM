import torch
import numpy as np
import matplotlib.pyplot as plt

def verify_logic():
    # 模拟参数: 100帧
    seq_len = 100
    
    # 构造假数据: [RotVelY, VelX, VelZ, Height]
    # 假设数据已经归一化，我们这里模拟反归一化后的真实物理数值
    # 1. RotVelY: 每帧向左转 0.05 弧度 (约3度) -> 这是一个持续的转弯
    rot_vel = torch.ones(1, seq_len) * 0.05 
    
    # 2. VelX: 0 (不横移)
    vel_x = torch.zeros(1, seq_len)
    
    # 3. VelZ: 0.1 (每帧向前走 0.1 米)
    vel_z = torch.ones(1, seq_len) * 0.1
    
    # 4. Height: 0 (不重要)
    height = torch.zeros(1, seq_len)
    
    # 组合成特征 [B, T, 4]
    features_denorm = torch.stack([rot_vel, vel_x, vel_z, height], dim=-1)

    # ==========================================
    # 方案 A: 旧逻辑 (简单累加，你之前代码里的)
    # ==========================================
    # 只累加 X 和 Z 的局部速度
    fake_pred_pos_x = torch.cumsum(vel_x, dim=1)
    fake_pred_pos_z = torch.cumsum(vel_z, dim=1)
    # 结果: X永远是0，Z一直在增加 -> 直线

    # ==========================================
    # 方案 B: 新逻辑 (考虑旋转投影)
    # ==========================================
    # 1. 累加角速度 -> 绝对朝向
    global_rot = torch.cumsum(rot_vel, dim=1) # [1, 100]
    
    # 2. 投影到全局
    # 假设初始朝向是 Z 轴正向
    cos_rot = torch.cos(global_rot)
    sin_rot = torch.sin(global_rot)
    
    # 旋转矩阵投影
    # Global_X = Local_X * cos + Local_Z * sin
    # Global_Z = -Local_X * sin + Local_Z * cos (HumanML3D 常用)
    global_vel_x = vel_x * cos_rot + vel_z * sin_rot
    global_vel_z = -vel_x * sin_rot + vel_z * cos_rot
    
    real_pos_x = torch.cumsum(global_vel_x, dim=1)
    real_pos_z = torch.cumsum(global_vel_z, dim=1)

    # ==========================================
    # 绘图对比
    # ==========================================
    plt.figure(figsize=(10, 5))
    
    # 画旧逻辑
    plt.plot(fake_pred_pos_x[0].numpy(), fake_pred_pos_z[0].numpy(), 
             label='Old Logic (Simple Cumsum)', color='red', linestyle='--')
    
    # 画新逻辑
    plt.plot(real_pos_x[0].numpy(), real_pos_z[0].numpy(), 
             label='New Logic (Rotation Aware)', color='green', linewidth=2)
    
    plt.title("Trajectory Reconstruction Logic Comparison\nSimulated Motion: Walking Forward while Turning Left")
    plt.xlabel("Global X (Meters)")
    plt.ylabel("Global Z (Meters)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # 保证比例一致
    
    print("保存对比图到 trajectory_comparison.png")
    plt.savefig("trajectory_comparison.png")

if __name__ == "__main__":
    verify_logic()