# import os
# import json
# import numpy as np
# import glob
# import sys
# import torch

# # ================= 配置区域 =================
# ROOT_DIR = "/root/autodl-tmp/MyRepository/MCM-LDM"
# DATASET_DIR = os.path.join(ROOT_DIR, "datasets")

# # 1. 物理参数路径
# JSON_DIR = os.path.join(DATASET_DIR, "PhysicsDataset/json_files")
# # 2. 动作特征路径 (263维)
# MOTION_DIR = os.path.join(DATASET_DIR, "PhysicsDataset/new_joint_vecs")
# # 3. Mean/Std 路径 (用于反归一化)
# # 通常 HumanML3D 的均值方差通用，或者用你自己算出来的
# MEAN_PATH = os.path.join(DATASET_DIR, "humanml3d/Mean.npy")
# STD_PATH = os.path.join(DATASET_DIR, "humanml3d/Std.npy")

# # 输出目录
# OUTPUT_DIR = "zero_wind_videos_v2"
# # 判定为0的阈值
# ZERO_THRESHOLD = 10.0 
# # ===========================================

# # 或者直接
# # 导入visual
# # import visual
# # 获取 MCM-LDM 文件夹的绝对路径
# project_root = "/root/autodl-tmp/MyRepository/MCM-LDM"
# if project_root not in sys.path:
#     sys.path.append(project_root)

# # 现在可以直接引用 visual.py 了
# import visual
# from visual import visual_pos

# # 尝试导入 MLD 的核心转换函数
# try:
#     from visual import visual_pos
#     # 需要这个函数把 263维特征 变成 XYZ 坐标
#     from mld.data.humanml.scripts.motion_process import recover_from_ric
#     print(">>> 成功导入 visual_pos 和 recover_from_ric")
# except ImportError as e:
#     print(f"【错误】导入失败: {e}")
#     print("请确保在项目根目录下运行 (export PYTHONPATH=$PYTHONPATH:.)")
#     sys.exit(1)

# def visualize_zeros_v2():
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
    
#     # 1. 加载 Mean 和 Std
#     print(f">>> 正在加载 Mean/Std 从: {MEAN_PATH}")
#     try:
#         mean = torch.from_numpy(np.load(MEAN_PATH)).float()
#         std = torch.from_numpy(np.load(STD_PATH)).float()
#         print(">>> Mean/Std 加载成功。")
#     except Exception as e:
#         print(f"【致命错误】无法加载 Mean/Std: {e}")
#         return

#     # 2. 扫描 JSON 寻找 0 风数据
#     print(f">>> 正在扫描 JSON 文件...")
#     json_files = glob.glob(os.path.join(JSON_DIR, "W_*.json"))
    
#     count = 0
    
#     for jf in json_files:
#         try:
#             with open(jf, 'r') as f:
#                 data = json.load(f)
            
#             params = data.get("parameters", {})
#             wf = params.get("wind_force", {})
#             mag = np.sqrt(wf.get('x', 0)**2 + wf.get('y', 0)**2)

#             # 筛选条件
#             if mag < ZERO_THRESHOLD:
#                 filename = os.path.splitext(os.path.basename(jf))[0]
#                 npy_path = os.path.join(MOTION_DIR, filename + ".npy")
                
#                 if not os.path.exists(npy_path):
#                     continue
                
#                 count += 1
#                 print(f"\n[{count}] 处理文件: {filename} (Mag: {mag:.2f})")

#                 # === 核心转换步骤 ===
                
#                 # A. 加载 263 维特征
#                 feature = torch.from_numpy(np.load(npy_path)).float() # [Seq, 263]
                
#                 # B. 反归一化 (De-normalize)
#                 # features * std + mean
#                 feature = feature * std + mean
                
#                 # C. 还原成关节坐标 (Recover to Joints)
#                 # recover_from_ric 期望输入是 [Batch, Seq, Dim] 或 [1, 263, 1, Seq] ?
#                 # 查看 MLD 源码，recover_from_ric 通常输入是 [Batch, Seq, 263]
#                 # 输出通常是 [Batch, Seq, 22, 3]
                
#                 # 增加 Batch 维度: [1, Seq, 263]
#                 feature = feature.unsqueeze(0) 
                
#                 # 转换: 263 -> 22 joints XYZ
#                 # 参数 22 是关节数量
#                 joints = recover_from_ric(feature, 22) # 返回 [1, Seq, 22, 3]
                
#                 # 转换回 numpy 并去掉 batch 维: [Seq, 22, 3]
#                 joints_np = joints.squeeze(0).cpu().numpy()
                
#                 # D. 保存为临时文件供 visual_pos 使用
#                 # visual_pos 能够识别 shape 为 (Seq, 22, 3) 的 npy
#                 temp_npy_path = os.path.join(OUTPUT_DIR, f"temp_{filename}.npy")
#                 mp4_path = os.path.join(OUTPUT_DIR, f"{filename}.mp4")
                
#                 np.save(temp_npy_path, joints_np)
                
#                 # E. 渲染
#                 try:
#                     visual_pos(temp_npy_path, mp4_path)
#                     print(f"    -> 视频生成成功: {mp4_path}")
#                 except Exception as e:
#                     print(f"    -> 渲染失败: {e}")
#                 finally:
#                     # 清理临时文件
#                     if os.path.exists(temp_npy_path):
#                         os.remove(temp_npy_path)

#         except Exception as e:
#             print(f"    -> 处理出错: {e}")
#             import traceback
#             traceback.print_exc()

#     print(f"\n完成。共处理 {count} 个文件。请去 {OUTPUT_DIR} 查看视频。")

# if __name__ == "__main__":
#     # 检查是否有显卡，recover_from_ric 可能会用到 cuda
#     if torch.cuda.is_available():
#         print("Using CUDA for calculation (if needed)")
#     visualize_zeros_v2()


import os
import json
import numpy as np
import glob
import sys
import torch
import argparse

project_root = "/root/autodl-tmp/MyRepository/MCM-LDM"
if project_root not in sys.path:
    sys.path.append(project_root)

# ================= 配置区域 =================
ROOT_DIR = "/root/autodl-tmp/MyRepository/MCM-LDM"
DATASET_DIR = os.path.join(ROOT_DIR, "datasets")

# 1. 物理参数路径
JSON_DIR = os.path.join(DATASET_DIR, "PhysicsDataset/json_files")
# 2. 动作特征路径 (263维)
MOTION_DIR = os.path.join(DATASET_DIR, "PhysicsDataset/new_joint_vecs")
# 3. Mean/Std 路径
MEAN_PATH = os.path.join(DATASET_DIR, "humanml3d/Mean.npy")
STD_PATH = os.path.join(DATASET_DIR, "humanml3d/Std.npy")

# 输出目录
OUTPUT_DIR = "zero_wind_videos_v3"
# 判定为0的阈值
ZERO_THRESHOLD = 10.0 
# ===========================================

# 依赖检查：尝试导入 visualize_pos 和 common 库
try:
    from visual import visual_pos
    from mld.data.humanml.common.quaternion import qrot, qinv # /root/autodl-tmp/MyRepository/MCM-LDM/mld/data/humanml/common/quaternion.py
    print(">>> 成功导入 visual_pos 和 common.quaternion")
except ImportError as e:
    print(f"【错误】导入失败: {e}")
    print("请确保在项目根目录下运行，且 common/ 文件夹存在。")
    sys.exit(1)

# ==========================================
# 你的同学写的核心还原函数 (PyTorch版)
# ==========================================
def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    # Get Y-axis rotation from rotation velocity
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3] 
    
    # Add Y-axis rotation to root position
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    # 关键：这里做了累加，实现了位移
    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos

def recover_from_ric_custom(data, joints_num=22):
    # 确保是 Tensor
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    r_rot_quat, r_pos = recover_root_rot_pos(data)
    
    # 提取 RIC
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    # Add Y-axis rotation to local joints
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    # Add root XZ to joints
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    # Concate root and joints
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions

# ==========================================
# 核心处理逻辑
# ==========================================
def process_single_file(npy_path, output_path, mean, std):
    print(f"正在处理: {os.path.basename(npy_path)}")
    
    # 1. 加载 263 维特征
    feature = torch.from_numpy(np.load(npy_path)).float() 
    
    # 2. 反归一化 (De-normalize)
    # 这一步非常重要，如果不做，速度值会很小，累加起来还是原地
    feature = feature * std + mean
    
    # 3. 还原成关节坐标 (Recover)
    # 输入不需要Batch维，函数里处理了 [Seq, Dim]
    try:
        joints = recover_from_ric_custom(feature, 22) # [Seq, 22, 3]
    except Exception as e:
        # 如果维度对不上，可能需要 unsqueeze
        feature = feature.unsqueeze(0)
        joints = recover_from_ric_custom(feature, 22)
        joints = joints.squeeze(0)

    # 4. 保存临时文件
    temp_npy_path = output_path.replace(".mp4", "_temp.npy")
    np.save(temp_npy_path, joints.cpu().numpy())
    
    # 5. 渲染
    try:
        visual_pos(temp_npy_path, output_path)
        print(f"    -> 视频生成成功: {output_path}")
    except Exception as e:
        print(f"    -> 渲染失败: {e}")
    finally:
        if os.path.exists(temp_npy_path):
            os.remove(temp_npy_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", type=str, help="指定单个263维npy文件的路径进行可视化")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载 Mean 和 Std
    print(f">>> 正在加载 Mean/Std...")
    try:
        mean = torch.from_numpy(np.load(MEAN_PATH)).float()
        std = torch.from_numpy(np.load(STD_PATH)).float()
    except Exception as e:
        print(f"【致命错误】无法加载 Mean/Std: {e}")
        return

    # === 功能 1: 单文件模式 ===
    if args.single:
        if not os.path.exists(args.single):
            print(f"错误：文件 {args.single} 不存在。")
            return
        
        filename = os.path.basename(args.single)
        output_path = os.path.join(OUTPUT_DIR, filename.replace(".npy", ".mp4"))
        process_single_file(args.single, output_path, mean, std)
        return

    # === 功能 2: 批量扫描模式 (0 风数据) ===
    print(f">>> 正在扫描 JSON 文件寻找 0 风数据...")
    json_files = glob.glob(os.path.join(JSON_DIR, "W_*.json"))
    
    count = 0
    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
            
            params = data.get("parameters", {})
            wf = params.get("wind_force", {})
            mag = np.sqrt(wf.get('x', 0)**2 + wf.get('y', 0)**2)

            if mag < ZERO_THRESHOLD:
                filename = os.path.splitext(os.path.basename(jf))[0]
                npy_path = os.path.join(MOTION_DIR, filename + ".npy")
                
                if not os.path.exists(npy_path): continue
                
                count += 1
                output_path = os.path.join(OUTPUT_DIR, f"{filename}.mp4")
                process_single_file(npy_path, output_path, mean, std)

        except Exception as e:
            print(f"Skipping {jf}: {e}")

    print(f"\n批量处理完成。共处理 {count} 个文件。")

if __name__ == "__main__":
    main()