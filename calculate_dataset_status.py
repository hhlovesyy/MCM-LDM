import numpy as np
import os
from tqdm import tqdm

def calculate_mean_std(data_dir):
    """
    遍历指定目录下的所有 .npy 文件，计算整个数据集的均值和标准差。
    
    Args:
        data_dir (str): 包含 .npy 动作文件的目录路径。

    Returns:
        tuple: (mean, std)
               mean (np.ndarray): 数据集的均值向量。
               std (np.ndarray): 数据集的标准差向量。
    """
    print(f"Starting calculation for directory: {data_dir}")
    
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    all_motions = []
    
    # 遍历目录下的所有 .npy 文件
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    if not file_list:
        raise ValueError(f"No .npy files found in {data_dir}")

    print(f"Found {len(file_list)} .npy files. Loading data...")
    for filename in tqdm(file_list, desc="Loading motions"):
        file_path = os.path.join(data_dir, filename)
        try:
            # 加载每个动作序列
            motion_data = np.load(file_path)
            # motion_data 的 shape 应该是 (num_frames, num_features)
            if motion_data.ndim != 2:
                print(f"Warning: Skipping file {filename} with unexpected shape {motion_data.shape}")
                continue
            all_motions.append(motion_data)
        except Exception as e:
            print(f"Error loading file {filename}: {e}")
            
    if not all_motions:
        raise ValueError("Could not load any valid motion data.")

    # 将所有动作序列在时间维度上拼接成一个巨大的数组
    # 这是最关键的一步，确保我们是在所有帧上计算统计数据
    print("Concatenating all motion frames...")
    # all_motions 是一个 list of arrays, [(T1, D), (T2, D), ...]
    # a huge array of shape (T1+T2+..., D)
    all_frames = np.concatenate(all_motions, axis=0)
    
    print(f"Total frames processed: {all_frames.shape[0]}")
    print(f"Feature dimension: {all_frames.shape[1]}")
    
    # 沿着特征维度（axis=0）计算均值和标准差
    print("Calculating mean and std...")
    mean = all_frames.mean(axis=0)
    std = all_frames.std(axis=0)
    
    # 【重要】处理标准差为0的情况，防止后续出现除零错误
    # 如果某个特征在整个数据集中都是一个常数，其 std 会是 0
    # 我们将其设为 1.0，这样在归一化时 (x - mean) / 1.0 不会改变它
    std[std == 0] = 1.0
    
    print("Calculation finished.")
    return mean, std

def main():
    # --- 配置区 ---
    # 请将此路径替换为你的数据集的准确路径
    dataset_path = "/root/autodl-tmp/MyRepository/MCM-LDM/datasets/humanml3d/new_joint_vecs"
    
    # 输出文件的保存路径（建议保存在同一目录下）
    output_dir = "/root/autodl-tmp/MyRepository/MCM-LDM/dataset_inspect/scene_dataset"
    
    # --- 执行区 ---
    try:
        mean_vec, std_vec = calculate_mean_std(dataset_path)
        
        # 打印一些信息以供验证
        print("\n--- Results ---")
        print(f"Mean vector shape: {mean_vec.shape}")
        print(f"Std vector shape: {std_vec.shape}")
        print(f"Sample of Mean (first 5 values): {mean_vec[:5]}")
        print(f"Sample of Std (first 5 values): {std_vec[:5]}")
        
        # 保存结果
        mean_path = os.path.join(output_dir, "mean.npy")
        std_path = os.path.join(output_dir, "std.npy")
        
        np.save(mean_path, mean_vec)
        np.save(std_path, std_vec)
        
        print(f"\nSuccessfully saved:")
        print(f"  Mean vector to: {mean_path}")
        print(f"  Std vector to: {std_path}")

    except (FileNotFoundError, ValueError) as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()