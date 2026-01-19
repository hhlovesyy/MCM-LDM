import sys
import os
import torch
import numpy as np
from mld.data.physimos_dataset import PhysicsDataset

# ================= 配置 =================
PROJECT_ROOT = "/root/autodl-tmp/MyRepository/MCM-LDM"
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets")
JSON_DIR = os.path.join(DATASET_DIR, "PhysicsDataset/json_files")
MOTION_DIR = os.path.join(DATASET_DIR, "PhysicsDataset/new_joint_vecs")
MEAN_PATH = os.path.join(DATASET_DIR, "humanml3d/Mean.npy")
STD_PATH = os.path.join(DATASET_DIR, "humanml3d/Std.npy")

def check_gap_logic():
    print(">>> 正在检查 Gap 反向归一化逻辑...")
    
    # 1. 加载 Mean/Std
    mean = np.load(MEAN_PATH)
    std = np.load(STD_PATH)
    
    # 2. 实例化 Dataset
    ds = PhysicsDataset(
        mean=mean, std=std, split_file=None, motion_dir=MOTION_DIR, json_dir=JSON_DIR,
        phys_noise_scale=0.0 # 关掉噪声，看纯值
    )
    
    targets = {"Narrow": None, "Wide": None, "Offset": None}
    
    # 3. 寻找典型样本
    for i in range(len(ds)):
        item = ds.data_list[i]
        fname = os.path.basename(item["json_path"])
        
        if "Gap" not in fname: continue
        
        # 简单解析文件名找样本 (假设文件名类似 Gap_40cm_...)
        # 或者你需要去读取 json 内容判断，这里为了快直接读 Dataset 输出
        
        # 读取这个样本处理后的参数
        # 注意：__getitem__ 比较慢因为要读npy，我们这里只读几十个看看
        data = ds[i]
        params = data["phys_params"].numpy() # [6]
        
        gap_width_val = params[4]
        gap_offset_val = params[5]
        
        # 判断类型
        if gap_width_val > 0.5: # 这是一个强信号 -> 应该是窄缝隙
            if targets["Narrow"] is None:
                targets["Narrow"] = (fname, params)
        elif gap_width_val == 0.0: # 这是一个0信号 -> 应该是宽缝隙
            if targets["Wide"] is None:
                targets["Wide"] = (fname, params)
                
        if abs(gap_offset_val) > 0.1 and targets["Offset"] is None:
             targets["Offset"] = (fname, params)
             
        if all(v is not None for v in targets.values()):
            break
    
    # 4. 打印结果
    print("-" * 80)
    print(f"{'类型':<10} | {'文件名':<40} | {'Gap Width (Index 4)'} | {'Offset (Index 5)'}")
    print("-" * 80)
    
    if targets["Narrow"]:
        name, p = targets["Narrow"]
        print(f"Narrow    | {name[:40]} | {p[4]:.4f} (High!)    | {p[5]:.4f}")
    else:
        print("Narrow    | 未找到 (检查是否所有缝隙都很宽?)")

    if targets["Wide"]:
        name, p = targets["Wide"]
        print(f"Wide      | {name[:40]} | {p[4]:.4f} (Should be 0) | {p[5]:.4f}")
    else:
        print("Wide      | 未找到 (检查是否 MAX_HACK 设置太大了?)")
        
    print("-" * 80)
    print("预期结果：")
    print("1. Narrow (e.g. 40cm): Width 值应该很大 (比如 > 0.6)。")
    print("2. Wide (e.g. 130cm): Width 值应该严格为 0.0000。")

if __name__ == "__main__":
    check_gap_logic()