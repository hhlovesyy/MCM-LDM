# 计算数据集的new_joints_vecs文件夹里的mean和std，写入到指定的数据集里面
import os
import numpy as np
import argparse
from tqdm import tqdm

def calculate_mean_std(data_dir, output_dir):
    """
    计算指定目录下所有 .npy 文件的 Mean 和 Std。
    假设 npy shape 为 (Seq_Len, Feature_Dim)
    """
    print(f"Scanning files in {data_dir}...")
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    if not file_list:
        print("No .npy files found!")
        return

    print(f"Found {len(file_list)} files. Loading data...")
    
    all_features = []
    
    # 逐个加载文件
    for fname in tqdm(file_list):
        path = os.path.join(data_dir, fname)
        try:
            data = np.load(path)
            # data shape: (Seq, 263)
            # 我们不仅要考虑不同文件，还要考虑文件内的每一帧
            # 所以我们把所有帧都收集起来
            all_features.append(data)
        except Exception as e:
            print(f"Error loading {fname}: {e}")

    # 拼接所有数据 -> (Total_Frames, 263)
    # 注意：如果数据量极大，这样可能会爆内存。
    # 现在的 180 个文件 * 240 帧完全没问题 (约 43k 帧)。
    all_data = np.concatenate(all_features, axis=0)
    
    print(f"Total frames: {all_data.shape[0]}")
    print(f"Feature dim: {all_data.shape[1]}")
    
    print("Calculating Mean and Std...")
    # 沿第0维（帧）计算均值和标准差
    mean = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0)
    
    # 处理 Std 为 0 的情况 (防止除以零)
    # 如果某个特征在所有帧里都不变，std 为 0，我们设为 1 或一个小数值
    std[std < 1e-5] = 1e-5

    print("Saving results...")
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "Mean.npy"), mean)
    np.save(os.path.join(output_dir, "Std.npy"), std)
    
    print(f"Done! Saved to {output_dir}")
    print(f"Mean shape: {mean.shape}, Std shape: {std.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 修改这里为你存放 UE5 导出的 npy 文件的路径
    parser.add_argument("--data_dir", type=str, required=True, help="Path to folder containing .npy files")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save Mean.npy and Std.npy")
    
    args = parser.parse_args()
    calculate_mean_std(args.data_dir, args.output_dir)