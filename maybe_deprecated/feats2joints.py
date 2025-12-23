from mld.data.humanml.scripts.motion_process import (process_file,
                                                     recover_from_ric,
                                                     extract_features)

import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from visual import visual_pos 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=False, default="/root/autodl-tmp/MyRepository/MCM-LDM/demo/style_motion", help="Folder containing .npy (263 dims)")
    parser.add_argument("--output_dir", type=str, required=False, default="/root/autodl-tmp/MyRepository/MCM-LDM/demo/style_motion_vis", help="Folder to save .npy (22x3 joints)")
    parser.add_argument("--mean_path", type=str, required=False,default="/root/autodl-tmp/MyRepository/MCM-LDM/datasets/humanml3d/Mean.npy", help="Path to Mean.npy")
    parser.add_argument("--std_path", type=str, required=False, default="/root/autodl-tmp/MyRepository/MCM-LDM/datasets/humanml3d/Std.npy", help="Path to Std.npy")
    parser.add_argument("--vis_count", type=int, default=1, help="Visualize first N samples to check")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "vis_check")
    os.makedirs(vis_dir, exist_ok=True)

    # 1. 加载 Mean/Std
    print("Loading Mean/Std...")
    mean = torch.from_numpy(np.load(args.mean_path)).float()
    std = torch.from_numpy(np.load(args.std_path)).float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = mean.to(device)
    std = std.to(device)

    files = [f for f in os.listdir(args.input_dir) if f.endswith(".npy")]
    print(f"Found {len(files)} files. Starting conversion...")

    for i, filename in enumerate(tqdm(files)):
        # 2. 加载特征
        feat_path = os.path.join(args.input_dir, filename)
        features = np.load(feat_path) # [263, T] or [T, 263]
        
        features_tensor = torch.from_numpy(features).float().to(device)
        
        # 维度调整 -> [1, T, 263] (recover_from_ric 通常需要 Batch First)
        if features_tensor.shape[0] == 263:
            features_tensor = features_tensor.permute(1, 0) # [T, 263]
        
        features_tensor = features_tensor.unsqueeze(0) # [1, T, 263]

        # 3. 反归一化 (核心步骤)
        # features_tensor = features_tensor * std + mean
        
        # 4. 转关节 (recover_from_ric)
        # return shape: [1, T, 22, 3]
        xyz = recover_from_ric(features_tensor, 22) 
        
        # 5. 保存
        xyz_np = xyz.squeeze(0).cpu().numpy() # [T, 22, 3]
        save_path = os.path.join(args.output_dir, filename)
        np.save(save_path, xyz_np)

        # 6. 可视化检查 (前几个)
        if i < args.vis_count:
            print(f"Visualizing {filename}...")
            mp4_path = os.path.join(vis_dir, filename.replace('.npy', '_vis.mp4'))
            visual_pos(save_path, mp4_path)
            

    print("Conversion Done!")

if __name__ == "__main__":
    main()