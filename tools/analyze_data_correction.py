import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import argparse

# ================= 配置区域 =================

# HumanML3D (SMPL) 22关节的标准顺序映射
JOINT_MAP = {
    'pelvis': 0, 'left_hip': 1, 'right_hip': 2, 'spine1': 3, 
    'left_knee': 4, 'right_knee': 5, 'spine2': 6, 
    'left_ankle': 7, 'right_ankle': 8, 'spine3': 9, 
    'left_foot': 10, 'right_foot': 11, 'neck': 12, 
    'left_collar': 13, 'right_collar': 14, 'head': 15, 
    'left_shoulder': 16, 'right_shoulder': 17, 'left_elbow': 18, 
    'right_elbow': 19, 'left_wrist': 20, 'right_wrist': 21
}

# [可修改] 你想分析哪些关节的高度？
# 比如分析手腕高度：['left_wrist', 'right_wrist']
# 比如分析重心高度：['pelvis']
# 比如分析头的高度：['head']
# TARGET_JOINTS_NAMES = ['left_wrist', 'right_wrist'] 
TARGET_JOINTS_NAMES = ['pelvis']

# [可修改] 高度轴是哪个？
# HumanML3D 标准是 Y-up (index 1)。如果你的数据是 Z-up，改成 2。
HEIGHT_AXIS = 1 

# ===========================================

def parse_filename(filename):
    """
    解析文件名: W_0p0_Back_0k_0029.npy
    返回: direction (str), magnitude (int)
    """
    parts = filename.replace('.npy', '').split('_')
    # 假设格式固定，倒数第2个是大小(0k)，倒数第3个是方向(Back)
    # 你可以根据实际情况调整这里的逻辑
    try:
        # 寻找带 'k' 的部分作为 magnitude
        mag_str = next(p for p in parts if p.endswith('k') and p[:-1].isdigit())
        mag_idx = parts.index(mag_str)
        
        magnitude = int(mag_str.replace('k', ''))
        
        # 假设方向在 magnitude 前面
        direction = parts[mag_idx - 1]
        
        return direction, magnitude
    except StopIteration:
        return None, None
    except Exception as e:
        # print(f"Warning: Could not parse {filename}: {e}")
        return None, None

def analyze_data(data_dir, output_img):
    print(f"Scanning directory: {data_dir}")
    files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    # 获取目标关节的索引
    target_indices = [JOINT_MAP[name] for name in TARGET_JOINTS_NAMES]
    print(f"Analyzing Average Height for joints: {TARGET_JOINTS_NAMES} (Indices: {target_indices})")
    print(f"Assuming Height Axis is index: {HEIGHT_AXIS} (0=X, 1=Y, 2=Z)")

    records = []

    for fname in tqdm(files):
        # 1. 解析文件名
        direction, magnitude = parse_filename(fname)
        if direction is None or magnitude is None:
            continue
            
        # 2. 加载数据
        path = os.path.join(data_dir, fname)
        try:
            # Shape: (Frames, 22, 3)
            motion = np.load(path)
            
            # 3. 计算高度指标
            # 取出目标关节在所有帧的坐标
            # shape: (Frames, Num_Targets, 3)
            target_joints_data = motion[:, target_indices, :]
            
            # 取出高度轴的数据
            # shape: (Frames, Num_Targets)
            heights = target_joints_data[:, :, HEIGHT_AXIS]
            
            # 计算这个文件里，这些关节的平均高度
            # 先对关节取平均，再对时间取平均
            avg_height = np.mean(heights)
            
            records.append({
                "Wind Magnitude (k)": magnitude,
                "Average Height": avg_height,
                "Direction": direction
            })
            
        except Exception as e:
            print(f"Error reading {fname}: {e}")

    # 转为 Pandas DataFrame 方便绘图
    df = pd.DataFrame(records)
    
    if len(df) == 0:
        print("No valid data found!")
        return

    print(f"\nCollected {len(df)} samples.")
    print("-" * 30)
    print(df.groupby("Direction")["Wind Magnitude (k)"].count())
    print("-" * 30)

    # 4. 计算相关系数 (所有数据)
    corr = df["Wind Magnitude (k)"].corr(df["Average Height"])
    print(f"\n>>> Global Pearson Correlation: {corr:.4f}")
    if corr > 0.3:
        print(">>> 结论: 正相关 (风越大，手越高)")
    elif corr < -0.3:
        print(">>> 结论: 负相关 (风越大，手越低)")
    else:
        print(">>> 结论: 无明显线性相关")

    # 5. 绘图
    sns.set_theme(style="whitegrid")
    
    # 创建画布
    plt.figure(figsize=(10, 6))
    
    # 绘制散点图 + 回归线
    # lmplot 会自动画出散点和拟合直线，hue参数按方向分类
    g = sns.lmplot(
        data=df, 
        x="Wind Magnitude (k)", 
        y="Average Height", 
        hue="Direction", 
        height=6, 
        aspect=1.5,
        scatter_kws={'alpha':0.6},
        line_kws={'alpha':0.8}
    )
    
    title_str = f"Correlation: Wind Mag vs {TARGET_JOINTS_NAMES} Height\n(Global Corr: {corr:.2f})"
    plt.title(title_str)
    plt.subplots_adjust(top=0.9) # 调整标题位置
    
    save_path = output_img
    plt.savefig(save_path)
    print(f"\nGraph saved to: {save_path}")
    print("Please check the graph to see if the trend lines are going UP or DOWN.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, 
                        default="/root/autodl-tmp/HumanML3D/HumanML3D/dataset_res/new_joints/",
                        help="Path to the new_joints folder containing .npy files")
    parser.add_argument("--output", type=str, default="wind_height_analysis_pelvis.png",
                        help="Output image filename")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory {args.data_dir} does not exist.")
    else:
        analyze_data(args.data_dir, args.output)