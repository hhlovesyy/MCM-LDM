import sys
import os
import torch
import numpy as np
import json
import glob
import argparse

# ================= 1. 环境与路径配置 (严格照做) =================
project_root = "/root/autodl-tmp/MyRepository/MCM-LDM"
if project_root not in sys.path:
    sys.path.append(project_root)

# 依赖检查
try:
    from visual import visual_pos
    # 修正后的路径
    from mld.data.humanml.common.quaternion import qrot, qinv 
    print(">>> 成功导入 visual_pos 和 common.quaternion")
except ImportError as e:
    print(f"【错误】导入失败: {e}")
    print("请确保 visual.py 在当前目录，且 mld 路径正确。")
    sys.exit(1)

# 数据路径配置
DATASET_DIR = os.path.join(project_root, "datasets")
JSON_DIR = os.path.join(DATASET_DIR, "PhysicsDataset/json_files")
MOTION_DIR = os.path.join(DATASET_DIR, "PhysicsDataset/new_joint_vecs")
MEAN_PATH = os.path.join(DATASET_DIR, "humanml3d/Mean.npy")
STD_PATH = os.path.join(DATASET_DIR, "humanml3d/Std.npy")
OUTPUT_DIR = "zero_wind_trusted_videos"

# ================= 2. 核心还原函数 (完全复制你提供的代码) =================

def recover_root_rot_pos(data):
    """
    完全复制 HumanML3D Cell 4 的 recover_root_rot_pos 函数
    """
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    # Get Y-axis rotation from rotation velocity
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    # 这里的切片 [..., :-1, 1:3] 是正确的，因为 data 是 (T, D)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3] 
    
    # Add Y-axis rotation to root position
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    # 关键：这里做了累加，实现了位移
    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos

def recover_from_ric(data, joints_num=22):
    """
    完全复制 HumanML3D Cell 4 的 recover_from_ric 函数
    """
    # 确保是 Tensor
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    r_rot_quat, r_pos = recover_root_rot_pos(data)
    
    # 提取 RIC
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    # Add Y-axis rotation to local joints
    # 这里的逻辑是：将 RIC (局部坐标) 转回 世界坐标
    # 原 notebook 使用的是 qinv(r_rot_quat)
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    # Add root XZ to joints
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    # Concate root and joints
    # 将计算好的 Root 位置和 Body 位置拼起来
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions

# ================= 3. 处理流程 =================

def process_and_visualize(npy_path, output_mp4_path, mean, std):
    """
    加载 -> 反归一化 -> 还原 -> 可视化
    """
    try:
        # 1. 加载数据
        raw_data = np.load(npy_path)
        feature = torch.from_numpy(raw_data).float()
        
        # 2. 反归一化 (De-normalize)
        # 这一步绝对不能少，否则速度值极小，cumsum 后依然是原地
        feature = feature * std + mean
        
        # 3. 调用可信的还原函数
        # recover_from_ric 期望输入 (Seq, 263) 或 (Batch, Seq, 263)
        # 这里的代码逻辑支持 (Seq, 263)
        joints = recover_from_ric(feature, joints_num=22) # (Seq, 22, 3)
        
        # 4. 保存为临时文件供 visual_pos 使用
        temp_npy = output_mp4_path.replace(".mp4", "_temp.npy")
        np.save(temp_npy, joints.cpu().numpy())
        
        # 5. 渲染
        visual_pos(temp_npy, output_mp4_path)
        print(f"    -> 生成视频: {output_mp4_path}")
        
        # 清理
        if os.path.exists(temp_npy):
            os.remove(temp_npy)
            
    except Exception as e:
        print(f"    -> [失败] {os.path.basename(npy_path)}: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", type=str, help="指定单个263维npy文件的路径进行可视化")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载 Mean/Std
    print(">>> 加载 Mean/Std (用于反归一化)...")
    try:
        mean = torch.from_numpy(np.load(MEAN_PATH)).float()
        std = torch.from_numpy(np.load(STD_PATH)).float()
    except Exception as e:
        print(f"无法加载 Mean/Std: {e}")
        return

    # === 模式 A: 单文件验证 ===
    if args.single:
        if not os.path.exists(args.single):
            print(f"文件不存在: {args.single}")
            return
        print(f">>> 单文件模式: {args.single}")
        filename = os.path.basename(args.single)
        output_path = os.path.join(OUTPUT_DIR, filename.replace(".npy", ".mp4"))
        process_and_visualize(args.single, output_path, mean, std)
        return

    # === 模式 B: 批量扫描 0 风数据 ===
    print(f">>> 批量模式: 扫描 {JSON_DIR} 下的 0 风数据...")
    json_files = glob.glob(os.path.join(JSON_DIR, "W_*.json"))
    
    count = 0
    zero_threshold = 10.0 # 你的设定
    
    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
            
            # 读取风力
            params = data.get("parameters", {})
            wf = params.get("wind_force", {})
            mag = np.sqrt(wf.get('x', 0)**2 + wf.get('y', 0)**2)
            
            if mag < zero_threshold:
                basename = os.path.splitext(os.path.basename(jf))[0]
                npy_path = os.path.join(MOTION_DIR, basename + ".npy")
                
                if not os.path.exists(npy_path):
                    continue
                
                count += 1
                print(f"[{count}] 发现0风数据: {basename} (Mag: {mag:.2f})")
                output_path = os.path.join(OUTPUT_DIR, f"{basename}.mp4")
                process_and_visualize(npy_path, output_path, mean, std)
                
        except Exception as e:
            continue

    print(f"\n全部完成。处理了 {count} 个文件。请查看 {OUTPUT_DIR}。")

if __name__ == "__main__":
    main()