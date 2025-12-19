import os
import torch
import numpy as np
import pickle
import argparse
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt

import os
import torch
import numpy as np
import pickle
import argparse
from tqdm import tqdm
from glob import glob

class EvaluationMetrics:
    def __init__(self):
        pass 

    def calculate_tsi(self, pred_joints, gt_joints):
        """
        计算 TSI (Trajectory Similarity Index)
        Input: [T, 22, 3]
        """
        if isinstance(pred_joints, np.ndarray): pred_joints = torch.from_numpy(pred_joints)
        if isinstance(gt_joints, np.ndarray): gt_joints = torch.from_numpy(gt_joints)
        
        # 1. 提取 Root Joint (Index 0) 的 XZ 平面坐标
        pred_traj = pred_joints[:, 0, [0, 2]] 
        gt_traj = gt_joints[:, 0, [0, 2]]

        # 2. 长度对齐
        min_len = min(pred_traj.shape[0], gt_traj.shape[0])
        pred_traj = pred_traj[:min_len]
        gt_traj = gt_traj[:min_len]

        # 3. 归零校准 (非常重要！对齐起点)
        # 比较的是轨迹的"形状"和"走势"，而非绝对世界坐标
        pred_traj = pred_traj - pred_traj[0:1, :]
        gt_traj = gt_traj - gt_traj[0:1, :]

        # 4. 计算欧氏距离
        distance = torch.norm(pred_traj - gt_traj, dim=1)
        return distance.mean().item()

    def calculate_fsf(self, pred_joints):
        """
        计算 FSF (Foot Sliding Factor)
        Input: [T, 22, 3]
        """
        if isinstance(pred_joints, np.ndarray): pred_joints = torch.from_numpy(pred_joints)
        
        # HumanML3D 关节: 10(L_Foot), 11(R_Foot)
        l_foot = pred_joints[:, 10, :] # [T, 3]
        r_foot = pred_joints[:, 11, :] # [T, 3]
        
        # 【关键修改】提高阈值到 5cm
        # 很多生成模型会有浮空现象，提高阈值能捕获到那些"悬浮滑步"
        contact_thresh = 0.095
        
        # 判断接触
        l_contact = (l_foot[:, 1] < contact_thresh).float()
        r_contact = (r_foot[:, 1] < contact_thresh).float()
        
        # 计算水平速度 (只看 XZ 平面)
        l_vel = torch.norm(l_foot[1:, [0, 2]] - l_foot[:-1, [0, 2]], dim=1)
        r_vel = torch.norm(r_foot[1:, [0, 2]] - r_foot[:-1, [0, 2]], dim=1)
        
        # 对齐长度
        l_contact = l_contact[1:]
        r_contact = r_contact[1:]
        
        # 只累加接触时的滑步
        l_slide = l_vel * l_contact
        r_slide = r_vel * r_contact
        
        # 计算逻辑 A: 平均每一帧的滑步距离 (cm/frame)
        # 也就是：总滑步距离 / 总帧数
        fsf = (l_slide.sum() + r_slide.sum()) / (l_slide.shape[0] + r_slide.shape[0])
        
        return fsf.item() * 100

def load_mean_std(path):
    mean = np.load(os.path.join(path, "mean.npy"))
    std = np.load(os.path.join(path, "std.npy"))
    return mean, std

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=str, required=True, help="Generated results (.pkl)")
    parser.add_argument("--gt_joints_dir", type=str, required=True, help="Converted GT joints folder")
    args = parser.parse_args()

    evaluator = EvaluationMetrics()

    print(f"Loading PKL: {args.pkl_path}...")
    with open(args.pkl_path, 'rb') as f:
        data = pickle.load(f)

    # 兼容不同的 PKL 结构
    if "joints" in data:
        joints_list = data["joints"]
        id_list = data["id"]
    else:
        print("Error: PKL structure unknown.")
        return

    tsi_scores = []
    fsf_scores = []
    
    print(f"Evaluating {len(joints_list)} samples...")

    for i, (pred_joints, id_str) in enumerate(tqdm(zip(joints_list, id_list), total=len(joints_list))):
        # print("pred_joints shape:", np.array(pred_joints).shape)
        # === 核心：通过 ID 找回对应的 Content GT ===
        try:
            # ID 格式: "content{NAME}_style..."
            # 解析出 {NAME}
            part1 = id_str.split("_style")[0]
            content_name = part1.replace("content", "")
        except:
            print(f"Skipping malformed ID: {id_str}")
            continue
            
        gt_path = os.path.join(args.gt_joints_dir, content_name + ".npy")
        
        if not os.path.exists(gt_path):
            # print(f"Warning: GT not found for {content_name}")
            continue
            
        gt_joints = np.load(gt_path)
        # 看一下gt_joints的统计学信息和pred_joints的统计学信息
        # print("gt_joints shape:", np.array(gt_joints).shape)
        # # mean和std
        # print("pred_joints mean/std:", np.mean(pred_joints), np.std(pred_joints))
        # print("gt_joints mean/std:", np.mean(gt_joints), np.std(gt_joints))

        try:
            # 计算 TSI (需要 GT)
            tsi = evaluator.calculate_tsi(pred_joints, gt_joints)
            tsi_scores.append(tsi)
            
            # 计算 FSF (不需要 GT)
            fsf = evaluator.calculate_fsf(pred_joints)
            fsf_scores.append(fsf)
            
        except Exception as e:
            print(f"Error evaluating {id_str}: {e}")

    print("\n" + "="*40)
    print(f"RESULTS for {os.path.basename(args.pkl_path)}")
    print(f"Evaluated samples: {len(tsi_scores)}")
    print("="*40)
    print(f"TSI: {np.mean(tsi_scores):.4f}")
    print(f"FSF: {np.mean(fsf_scores):.4f}")
    print("="*40)

if __name__ == "__main__":
    main()