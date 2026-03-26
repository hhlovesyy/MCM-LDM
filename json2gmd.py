import numpy as np
import json
import argparse
import os
import sys

# 把你的 TrajectoryProcessor 导进来
from mld.models.modeltype.trajectory_utils import * 

def create_gmd_template(json_path, output_npy_path, orig_template_path, target_speed=1.3, fps=20):
    print(f"[GMD Translator] 正在读取场景配置: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        scene_data = json.load(f)

    # 1. 提取并归零轨迹
    traj_config = scene_data['trajectory']
    obstacles = scene_data.get('environment', {}).get('obstacles', [])
    waypoints = traj_config['points']
    
    startPosX, startPosY = waypoints[0][0], waypoints[0][1]
    for p in waypoints:
        p[0] -= startPosX
        p[1] -= startPosY
    for obs in obstacles:
        if 'center' in obs:
            obs['center'][0] -= startPosX
            obs['center'][1] -= startPosY
            
    # 2. 生成稠密曲线
    processor = TrajectoryProcessor()
    dense_curve = processor.get_path_from_config(traj_config, obstacles)
    
    # ================= [修复核心] =================
    # 3. 窃取原版骨架和长度
    orig_template = np.load(orig_template_path) # Shape: (1, 22, 3, 120)
    GMD_MAX_FRAMES = orig_template.shape[3]     # 提取它的钦定长度 (120)
    base_pose = orig_template[:, :, :, 0]       # 窃取第一帧的完美骨架 Shape: (1, 22, 3)
    
    # 记录原骨架根节点的初始 X, Z
    orig_root_x = base_pose[0, 0, 0]
    orig_root_z = base_pose[0, 0, 2]
    # ==============================================

    # 4. 计算我们真实需要的帧数 T
    diffs = dense_curve[1:] - dense_curve[:-1]
    total_distance = np.sum(np.linalg.norm(diffs, axis=1))
    T = max(int((total_distance / target_speed) * fps), 40)
    # 确保不超过 GMD 的最大容忍度
    T = min(T, GMD_MAX_FRAMES) 
    
    indices = np.linspace(0, len(dense_curve) - 1, T).astype(int)
    resampled_curve = dense_curve[indices]
    
    # 5. 构建全新的模板矩阵 (用 GMD_MAX_FRAMES 铺满)
    gmd_template = np.zeros((1, 22, 3, GMD_MAX_FRAMES), dtype=np.float32)
    
    for t in range(GMD_MAX_FRAMES):
        # 先把整具骨架 copy 进去
        gmd_template[0, :, :, t] = base_pose[0, :, :]
        
        # 计算偏移量 (如果走完了，就停在最后一个点)
        if t < T:
            shift_x = resampled_curve[t, 0] - orig_root_x
            shift_z = resampled_curve[t, 1] - orig_root_z
        else:
            shift_x = resampled_curve[-1, 0] - orig_root_x
            shift_z = resampled_curve[-1, 1] - orig_root_z
            
        # 把偏移量施加到所有 22 个关节上！(保证腿长/骨骼结构不被破坏)
        gmd_template[0, :, 0, t] += shift_x
        gmd_template[0, :, 2, t] += shift_z
        
    np.save(output_npy_path, gmd_template)
    print(f"[GMD Translator] ✅ 成功借尸还魂！生成轨迹: {output_npy_path}")
    print(f"[GMD Translator] 真实运动帧数: {T} | 最终 Padding 长度: {GMD_MAX_FRAMES}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True, help="输入的场景 json 路径")
    parser.add_argument("--out", type=str, required=True, help="输出的 template_joints.npy 路径")
    parser.add_argument("--orig", type=str, required=True, help="GMD 原版的 template_joints.npy 路径")
    args = parser.parse_args()
    
    create_gmd_template(args.json, args.out, args.orig)