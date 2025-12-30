import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading {args.pkl_path}...")
    with open(args.pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # 兼容数据结构
    if isinstance(data, dict):
        joints_list = data["joints"]
        target_list = data["target_traj"]
    else:
        joints_list = [x["joints"] for x in data]
        target_list = [x["target_traj"] for x in data]

    print(f"Loaded {len(joints_list)} samples.")

    # 取前 4 个样本进行可视化
    num_vis = 4
    fig, axes = plt.subplots(1, num_vis, figsize=(20, 5))

    for i in range(num_vis):
        # 1. 获取数据
        pred = joints_list[i] # [T, 22, 3]
        target = target_list[i] # 可能是 [T, 3] 或 [T, 1, 3]

        # 2. 提取 Pred Root XZ
        # pred_root: [T, 2]
        pred_root = pred[:, 0, [0, 2]] 
        
        # 3. 处理 Target (各种可能的 Shape)
        if isinstance(target, torch.Tensor): target = target.cpu().numpy()
        if isinstance(pred_root, torch.Tensor): pred_root = pred_root.cpu().numpy()

        target_root = target
        if target.shape[-1] == 3:
            target_root = target[:, [0, 2]]
        
        # 挤压多余维度
        target_root = target_root.squeeze() # [T, 2]
        
        # 4. 长度对齐
        min_len = min(pred_root.shape[0], target_root.shape[0])
        pred_root = pred_root[:min_len]
        target_root = target_root[:min_len]

        # 5. 打印统计信息 (这是破案的关键)
        print(f"\n--- Sample {i} ---")
        print(f"Pred Shape: {pred_root.shape}, Range: X[{pred_root[:,0].min():.2f}, {pred_root[:,0].max():.2f}]")
        print(f"Targ Shape: {target_root.shape}, Range: X[{target_root[:,0].min():.2f}, {target_root[:,0].max():.2f}]")
        
        # 6. 归零校准
        pred_plot = pred_root - pred_root[0]
        target_plot = target_root - target_root[0]
        
        # 7. 绘图
        ax = axes[i]
        ax.plot(pred_plot[:, 0], pred_plot[:, 1], 'r-', label='Pred (Gen)', linewidth=2)
        ax.plot(target_plot[:, 0], target_plot[:, 1], 'b--', label='Target (Cond)', linewidth=2)
        ax.set_title(f"Sample {i}")
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig("debug_traj.png")
    print("\n[Done] Visualization saved to debug_traj.png. Please check it!")

if __name__ == "__main__":
    main()