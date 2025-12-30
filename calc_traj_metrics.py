# import os
# import torch
# import numpy as np
# import pickle
# import argparse
# from tqdm import tqdm
# from glob import glob
# import matplotlib.pyplot as plt

# import os
# import torch
# import numpy as np
# import pickle
# import argparse
# from tqdm import tqdm
# from glob import glob

# class EvaluationMetrics:
#     def __init__(self):
#         pass 

#     def calculate_tsi(self, pred_joints, gt_joints):
#         """
#         计算 TSI (Trajectory Similarity Index)
#         Input: [T, 22, 3]
#         """
#         if isinstance(pred_joints, np.ndarray): pred_joints = torch.from_numpy(pred_joints)
#         if isinstance(gt_joints, np.ndarray): gt_joints = torch.from_numpy(gt_joints)
        
#         # 1. 提取 Root Joint (Index 0) 的 XZ 平面坐标
#         pred_traj = pred_joints[:, 0, [0, 2]] 
#         gt_traj = gt_joints[:, 0, [0, 2]]

#         # 2. 长度对齐
#         min_len = min(pred_traj.shape[0], gt_traj.shape[0])
#         pred_traj = pred_traj[:min_len]
#         gt_traj = gt_traj[:min_len]

#         # 3. 归零校准 (非常重要！对齐起点)
#         # 比较的是轨迹的"形状"和"走势"，而非绝对世界坐标
#         pred_traj = pred_traj - pred_traj[0:1, :]
#         gt_traj = gt_traj - gt_traj[0:1, :]

#         # 4. 计算欧氏距离
#         distance = torch.norm(pred_traj - gt_traj, dim=1)
#         return distance.mean().item()

#     def calculate_fsf(self, pred_joints):
#         """
#         计算 FSF (Foot Sliding Factor)
#         Input: [T, 22, 3]
#         """
#         if isinstance(pred_joints, np.ndarray): pred_joints = torch.from_numpy(pred_joints)
        
#         # HumanML3D 关节: 10(L_Foot), 11(R_Foot)
#         l_foot = pred_joints[:, 10, :] # [T, 3]
#         r_foot = pred_joints[:, 11, :] # [T, 3]
        
#         # 【关键修改】提高阈值到 5cm
#         # 很多生成模型会有浮空现象，提高阈值能捕获到那些"悬浮滑步"
#         contact_thresh = 0.095
        
#         # 判断接触
#         l_contact = (l_foot[:, 1] < contact_thresh).float()
#         r_contact = (r_foot[:, 1] < contact_thresh).float()
        
#         # 计算水平速度 (只看 XZ 平面)
#         l_vel = torch.norm(l_foot[1:, [0, 2]] - l_foot[:-1, [0, 2]], dim=1)
#         r_vel = torch.norm(r_foot[1:, [0, 2]] - r_foot[:-1, [0, 2]], dim=1)
        
#         # 对齐长度
#         l_contact = l_contact[1:]
#         r_contact = r_contact[1:]
        
#         # 只累加接触时的滑步
#         l_slide = l_vel * l_contact
#         r_slide = r_vel * r_contact
        
#         # 计算逻辑 A: 平均每一帧的滑步距离 (cm/frame)
#         # 也就是：总滑步距离 / 总帧数
#         fsf = (l_slide.sum() + r_slide.sum()) / (l_slide.shape[0] + r_slide.shape[0])
        
#         return fsf.item() * 100

# def load_mean_std(path):
#     mean = np.load(os.path.join(path, "mean.npy"))
#     std = np.load(os.path.join(path, "std.npy"))
#     return mean, std

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--pkl_path", type=str, required=True, help="Generated results (.pkl)")
#     parser.add_argument("--gt_joints_dir", type=str, required=True, help="Converted GT joints folder")
#     args = parser.parse_args()

#     evaluator = EvaluationMetrics()

#     print(f"Loading PKL: {args.pkl_path}...")
#     with open(args.pkl_path, 'rb') as f:
#         data = pickle.load(f)

#     # 兼容不同的 PKL 结构
#     if "joints" in data:
#         joints_list = data["joints"]
#         id_list = data["id"]
#     else:
#         print("Error: PKL structure unknown.")
#         return

#     tsi_scores = []
#     fsf_scores = []
    
#     print(f"Evaluating {len(joints_list)} samples...")

#     for i, (pred_joints, id_str) in enumerate(tqdm(zip(joints_list, id_list), total=len(joints_list))):
#         # print("pred_joints shape:", np.array(pred_joints).shape)
#         # === 核心：通过 ID 找回对应的 Content GT ===
#         try:
#             # ID 格式: "content{NAME}_style..."
#             # 解析出 {NAME}
#             part1 = id_str.split("_style")[0]
#             content_name = part1.replace("content", "")
#         except:
#             print(f"Skipping malformed ID: {id_str}")
#             continue
            
#         gt_path = os.path.join(args.gt_joints_dir, content_name + ".npy")
        
#         if not os.path.exists(gt_path):
#             # print(f"Warning: GT not found for {content_name}")
#             continue
            
#         gt_joints = np.load(gt_path)
#         # 看一下gt_joints的统计学信息和pred_joints的统计学信息
#         # print("gt_joints shape:", np.array(gt_joints).shape)
#         # # mean和std
#         # print("pred_joints mean/std:", np.mean(pred_joints), np.std(pred_joints))
#         # print("gt_joints mean/std:", np.mean(gt_joints), np.std(gt_joints))

#         try:
#             # 计算 TSI (需要 GT)
#             tsi = evaluator.calculate_tsi(pred_joints, gt_joints)
#             tsi_scores.append(tsi)
            
#             # 计算 FSF (不需要 GT)
#             fsf = evaluator.calculate_fsf(pred_joints)
#             fsf_scores.append(fsf)
            
#         except Exception as e:
#             print(f"Error evaluating {id_str}: {e}")

#     print("\n" + "="*40)
#     print(f"RESULTS for {os.path.basename(args.pkl_path)}")
#     print(f"Evaluated samples: {len(tsi_scores)}")
#     print("="*40)
#     print(f"TSI: {np.mean(tsi_scores):.4f}")
#     print(f"FSF: {np.mean(fsf_scores):.4f}")
#     print("="*40)

# if __name__ == "__main__":
#     main()

import os
import torch
import numpy as np
import pickle
import argparse
from tqdm import tqdm

class EvaluationMetrics:
    def __init__(self):
        pass 

    def calculate_tsi(self, pred_joints, target_traj):
        """
        计算 Trajectory Error (TSI)
        对比 生成轨迹 vs 目标轨迹条件 (Target Condition)
        """
        if isinstance(pred_joints, np.ndarray): pred_joints = torch.from_numpy(pred_joints)
        if isinstance(target_traj, np.ndarray): target_traj = torch.from_numpy(target_traj)
        
        # 1. 提取 Root Joint (Index 0) 的 XZ 平面坐标
        # pred_joints: [T, 22, 3] -> pred_root: [T, 3]
        pred_root = pred_joints[:, 0, [0, 2]] 
        
        # target_traj 通常是 [T, 3]
        if target_traj.shape[-1] == 3: 
            target_root = target_traj[:, [0, 2]] # 取 x, z
        elif target_traj.shape[-1] == 2:
            target_root = target_traj
        else:
            # 兼容可能的 [1, T, 3]
            target_root = target_traj.reshape(-1, target_traj.shape[-1])[:, [0, 2]]

        # 2. 长度对齐
        min_len = min(pred_root.shape[0], target_root.shape[0])
        pred_root = pred_root[:min_len]
        target_root = target_root[:min_len]

        # 3. 归零校准 (比较轨迹走势，而非绝对坐标)
        pred_root = pred_root - pred_root[0:1, :]
        target_root = target_root - target_root[0:1, :]

        # 4. 计算欧氏距离 (L2 Norm) 并求平均
        distance = torch.norm(pred_root - target_root, dim=1)
        
        return distance.mean().item()

    def calculate_fsf(self, pred_joints):
        """
        计算 FSF (Foot Sliding Fraction)
        Input: [T, 22, 3]
        """
        if isinstance(pred_joints, np.ndarray): pred_joints = torch.from_numpy(pred_joints)
        
        # HumanML3D 关节: 10(L_Foot), 11(R_Foot)
        l_foot = pred_joints[:, 10, :] 
        r_foot = pred_joints[:, 11, :] 
        
        # 【阈值设定】 2.5cm
        # contact_thresh = 0.025 
        contact_thresh = 0.05
        
        # 判断接触
        l_contact = (l_foot[:, 1] < contact_thresh).float()
        r_contact = (r_foot[:, 1] < contact_thresh).float()
        
        # 计算水平速度 (XZ平面位移)
        l_vel = torch.norm(l_foot[1:, [0, 2]] - l_foot[:-1, [0, 2]], dim=1)
        r_vel = torch.norm(r_foot[1:, [0, 2]] - r_foot[:-1, [0, 2]], dim=1)
        
        # 对齐长度
        l_contact = l_contact[1:]
        r_contact = r_contact[1:]
        
        # 只累加接触时的滑步距离
        l_slide = l_vel * l_contact
        r_slide = r_vel * r_contact
        
        # 计算逻辑: 平均每帧滑步距离 (cm/frame)
        total_frames = l_slide.shape[0] + r_slide.shape[0] + 1e-6
        fsf = (l_slide.sum() + r_slide.sum()) / total_frames
        
        # 结果乘100变厘米
        return fsf.item() * 100 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=str, required=True, help="Path to the .pkl results file")
    # 增加一个参数用于指定输出的 txt 文件名，如果不传则自动生成
    parser.add_argument("--output_txt", type=str, default=None, help="Path to append results txt")
    args = parser.parse_args()

    evaluator = EvaluationMetrics()

    print(f"Loading PKL: {args.pkl_path}...")
    with open(args.pkl_path, 'rb') as f:
        data = pickle.load(f)

    joints_list = []
    target_traj_list = []

    # === [核心修复] 智能判断数据结构 ===
    if isinstance(data, dict):
        # 情况A: 字典套列表 (Dictionary of Lists) <- 你现在是这个
        print("Detected structure: Dictionary of Lists")
        if "joints" in data:
            joints_list = data["joints"]
        else:
            raise ValueError("Key 'joints' not found in pickle dictionary.")
            
        if "target_traj" in data:
            target_traj_list = data["target_traj"]
        else:
            raise ValueError("Key 'target_traj' not found. Please re-run inference with the fixed script.")
            
    elif isinstance(data, list):
        # 情况B: 列表套字典 (List of Dictionaries)
        print("Detected structure: List of Dictionaries")
        for item in data:
            joints_list.append(item["joints"])
            if "target_traj" in item:
                target_traj_list.append(item["target_traj"])
            else:
                 # 如果列表模式下没有 traj，可能得跳过或者报错
                 pass
    else:
        raise ValueError(f"Unknown pickle structure: {type(data)}")

    # 检查长度一致性
    if len(joints_list) != len(target_traj_list):
        print(f"Warning: Length mismatch! Joints: {len(joints_list)}, Targets: {len(target_traj_list)}")
        # 取交集长度
        min_len = min(len(joints_list), len(target_traj_list))
        joints_list = joints_list[:min_len]
        target_traj_list = target_traj_list[:min_len]

    tsi_scores = []
    fsf_scores = []
    
    print(f"Evaluating {len(joints_list)} samples...")

    # 使用 zip 同时遍历生成动作和目标轨迹
    for pred_joints, target_traj in tqdm(zip(joints_list, target_traj_list), total=len(joints_list)):
        try:
            # 计算 TSI
            tsi = evaluator.calculate_tsi(pred_joints, target_traj)
            tsi_scores.append(tsi)
            
            # 计算 FSF
            fsf = evaluator.calculate_fsf(pred_joints)
            fsf_scores.append(fsf)
            
        except Exception as e:
            print(f"Error evaluating sample: {e}")

    # 统计结果
    mean_tsi = np.mean(tsi_scores)
    mean_fsf = np.mean(fsf_scores)

    # 构造输出字符串
    result_str = (
        f"\n{'='*40}\n"
        f"RESULTS for {os.path.basename(args.pkl_path)}\n"
        f"Evaluated samples: {len(tsi_scores)}\n"
        f"{'='*40}\n"
        f"TSI (Mean Traj Error): {mean_tsi:.4f}\n"
        f"FSF (Foot Sliding):    {mean_fsf:.4f}\n"
        f"{'='*40}\n"
    )
    
    # 打印到控制台
    print(result_str)

    # 写入文件
    # 1. 确定文件名
    if args.output_txt:
        txt_path = args.output_txt
    else:
        # 默认跟 pkl 同名，把后缀改成 .txt
        txt_path = os.path.splitext(args.pkl_path)[0] + "_metrics.txt"

    # 2. 追加写入
    try:
        with open(txt_path, "a") as f:
            f.write(result_str)
        print(f"Results appended to: {txt_path}")
    except Exception as e:
        print(f"Failed to write to file: {e}")

if __name__ == "__main__":
    main()