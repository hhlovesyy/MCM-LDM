# scripts/analyze_motion_length.py

import numpy as np
import os
from os.path import join as pjoin
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def analyze_dataset(dataset_root, motion_dir_name, split_files):
    """对给定的数据集进行动作长度分析"""
    print(f"\n--- Analyzing Dataset at: {dataset_root} ---")
    
    all_lengths = []
    
    for split_file in split_files:
        split_path = pjoin(dataset_root, split_file)
        if not os.path.exists(split_path):
            print(f"Warning: Split file not found, skipping: {split_path}")
            continue
            
        print(f"Reading split file: {split_file}")
        with open(split_path, 'r') as f:
            ids = [line.strip() for line in f.readlines()]
        
        for name in tqdm(ids, desc=f"Processing {split_file}"):
            motion_path = pjoin(dataset_root, motion_dir_name, name + ".npy")
            try:
                motion = np.load(motion_path)
                all_lengths.append(len(motion))
            except FileNotFoundError:
                # 在 100Style 中，文件名可能带有前缀，这里简化处理
                # 真实的 dataset 类会处理这个问题
                pass

    if not all_lengths:
        print("No motion data found. Please check paths.")
        return None

    lengths = np.array(all_lengths)
    
    print("\n--- Statistics ---")
    print(f"Total motions analyzed: {len(lengths)}")
    print(f"Min length: {lengths.min()}")
    print(f"Max length: {lengths.max()}")
    print(f"Mean length: {lengths.mean():.2f}")
    print(f"Median length: {np.median(lengths)}")
    print(f"95th percentile: {np.percentile(lengths, 95):.2f}")
    print(f"99th percentile: {np.percentile(lengths, 99):.2f}")
    
    return lengths

def plot_distributions(humanml_lengths, style100_lengths):
    """绘制两个数据集的长度分布直方图"""
    plt.figure(figsize=(14, 6))
    
    sns.histplot(humanml_lengths, color="skyblue", kde=True, label='HumanML3D', binwidth=5)
    sns.histplot(style100_lengths, color="red", kde=True, label='100Style', binwidth=5, alpha=0.6)
    
    plt.title('Motion Length Distribution Comparison')
    plt.xlabel('Number of Frames')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    # 设置一个合理的 x 轴范围，以便观察
    combined_max = max(humanml_lengths.max(), style100_lengths.max())
    plt.xlim(0, min(combined_max + 10, 500)) # 限制 x 轴最大值，避免极端值影响
    
    output_path = "motion_length_distribution.png"
    plt.savefig(output_path)
    print(f"\nDistribution plot saved to {output_path}")

if __name__ == '__main__':
    # --- 请在这里配置你的路径 ---
    # /root/autodl-tmp/MyRepository/MCM-LDM/datasets
    HUMANML3D_ROOT = "./datasets/humanml3d"
    STYLE100_ROOT = "./datasets/100StyleDataset"
    
    # 运行分析
    humanml_lengths = analyze_dataset(
        dataset_root=HUMANML3D_ROOT,
        motion_dir_name="new_joint_vecs",
        split_files=["train.txt", "val.txt", "test.txt"]
    )
    
    style100_lengths = analyze_dataset(
        dataset_root=STYLE100_ROOT,
        motion_dir_name="new_joint_vecs",
        split_files=["train.txt", "val.txt", "test.txt"]
    )
    
    # 绘制分布图
    if humanml_lengths is not None and style100_lengths is not None:
        plot_distributions(humanml_lengths, style100_lengths)