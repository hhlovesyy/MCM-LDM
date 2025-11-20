# 查看数据集的统计信息，包括 motion npy(new_joint_vecs) 文件和对应的物理参数 json 文件。
import os
import numpy as np
import json
import argparse
import torch

def inspect_data(motion_dir, json_dir=None):
    """
    随机抽取几个文件，打印其统计信息和对应的物理参数。
    """
    files = [f for f in os.listdir(motion_dir) if f.endswith('.npy')]
    if not files:
        print("No npy files found.")
        return

    # 随机抽 3 个
    samples = np.random.choice(files, min(3, len(files)), replace=False)
    
    print(f"=== Inspecting {len(samples)} Random Samples from {motion_dir} ===")
    
    for fname in samples:
        print(f"\n--- File: {fname} ---")
        
        # 1. 检查 Motion NPY
        npy_path = os.path.join(motion_dir, fname)
        motion = np.load(npy_path)
        print(f"[Motion NPY]")
        print(f"  Shape: {motion.shape} (Frames, Features)")
        print(f"  Range: Min={motion.min():.4f}, Max={motion.max():.4f}")
        print(f"  Mean:  {motion.mean():.4f}")
        # 检查是否有 NaN
        if np.isnan(motion).any():
            print("  WARNING: Contains NaN values!")
            
        # 2. 检查对应的 JSON (如果存在)
        if json_dir:
            # 假设 json 和 npy 同名
            json_fname = fname.replace('.npy', '.json')
            json_path = os.path.join(json_dir, json_fname)
            
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    meta = json.load(f)
                print(f"[Physics JSON]")
                # 尝试打印 parameters 字段
                if "parameters" in meta:
                    print(f"  Parameters: {meta['parameters']}")
                    # 简单的归一化预估
                    if "wind_force" in meta["parameters"]:
                        wf = meta["parameters"]["wind_force"]
                        # 假设最大风力是 100,000
                        norm_x = wf['x'] / 100000.0
                        print(f"  (Preview) Norm Wind X (approx): {norm_x:.4f}")
                else:
                    print("  No 'parameters' field found.")
            else:
                print(f"  JSON file not found at: {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 通常 npy 和 json 在同一个文件夹，或者分开
    parser.add_argument("--motion_dir", type=str, required=True, help="Path to .npy files")
    parser.add_argument("--json_dir", type=str, default=None, help="Path to .json files (optional, default same as motion_dir)")
    
    args = parser.parse_args()
    
    # 如果没传 json_dir，默认和 motion_dir 一样
    j_dir = args.json_dir if args.json_dir else args.motion_dir
    
    inspect_data(args.motion_dir, j_dir)