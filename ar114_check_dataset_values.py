import sys
import os
import torch
import numpy as np
import random
from mld.data.physimos_dataset import PhysicsDataset

# ================= 配置区域 =================
# 请确保这些路径和你训练配置一致
PROJECT_ROOT = "/root/autodl-tmp/MyRepository/MCM-LDM"
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets")
JSON_DIR = os.path.join(DATASET_DIR, "PhysicsDataset/json_files")
MOTION_DIR = os.path.join(DATASET_DIR, "PhysicsDataset/new_joint_vecs")
MEAN_PATH = os.path.join(DATASET_DIR, "humanml3d/Mean.npy")
STD_PATH = os.path.join(DATASET_DIR, "humanml3d/Std.npy")

def verify_dataset():
    print(">>> 正在初始化 Dataset (模拟训练读取)...")
    
    # 加载 Mean/Std
    mean = np.load(MEAN_PATH)
    std = np.load(STD_PATH)
    
    # 实例化 (参数必须和你修改后的一致)
    ds = PhysicsDataset(
        mean=mean,
        std=std,
        split_file=None,
        motion_dir=MOTION_DIR,
        json_dir=JSON_DIR,
        max_wind_force=30000.0,
        max_ceiling_height=220.0,
        phys_noise_scale=0.0  # 验证时关掉噪声，看核心数值
    )
    
    print(f"Dataset 总样本数: {len(ds)}")
    
    # 分类桶
    categories = {
        "Wind_Zero": [], # 文件名含 W_0p0
        "Wind_High": [], # 文件名含 W_ 但不含 0p0
        "Ceiling": [],
        "Gap": []
    }
    
    # 遍历所有数据进行分类 (只存索引)
    for i in range(len(ds)):
        item = ds.data_list[i]
        fname = os.path.basename(item["json_path"])
        
        if "W_0p0" in fname:
            categories["Wind_Zero"].append(i)
        elif "W_" in fname or "Wind" in fname:
            categories["Wind_High"].append(i)
        elif "Ceiling" in fname:
            categories["Ceiling"].append(i)
        elif "Gap" in fname:
            categories["Gap"].append(i)
            
    # --- 开始抽检 ---
    print("\n" + "="*80)
    print(f"{'类别':<12} | {'文件名':<35} | {'物理参数 Tensor [Wx, Wy, Mag, Ceil, GW, GO]'}")
    print("="*80)

    def check_samples(category_name, indices, count=5):
        if not indices:
            print(f"{category_name:<12} | (无样本)")
            return

        # 随机抽样
        samples = random.sample(indices, min(len(indices), count))
        
        for idx in samples:
            # === 这里会触发你的 __getitem__ ===
            data = ds[idx] 
            phys = data["phys_params"].numpy()
            fname = os.path.basename(ds.data_list[idx]["json_path"])
            
            # 格式化打印
            vals = [f"{x:.2f}" for x in phys]
            print(f"{category_name:<12} | {fname[:35]:<35} | {vals}")
            
            # === 自动断言检查 (Fail-Fast) ===
            if category_name == "Wind_Zero":
                # 检查点：Mag (Index 2) 必须 >= 0.5
                if phys[2] < 0.49: 
                    print(f"  >>> [错误] 0风 Hack 失败！Mag={phys[2]}")
                # 检查点：Ceiling/Gap (Index 3,4,5) 必须为 0
                if sum(abs(phys[3:])) > 0:
                    print(f"  >>> [错误] 互斥失败！Ceiling/Gap 有值: {phys[3:]}")

            elif category_name == "Gap":
                # 检查点：Wind (Index 0,1,2) 必须为 0
                if sum(abs(phys[:3])) > 0:
                    print(f"  >>> [错误] Gap 场景混入了风力: {phys[:3]}")
                # 检查点：Gap Width (Index 4) 如果是宽缝隙应该为0，窄缝隙应该大
                if "130cm" in fname and phys[4] > 0.1:
                     print(f"  >>> [错误] 宽缝隙 (130cm) 未归零: {phys[4]}")

            elif category_name == "Ceiling":
                # 检查点：220cm 应该为 0
                if "220cm" in fname and phys[3] > 0.1:
                    print(f"  >>> [错误] 高天花板 (220cm) 未归零: {phys[3]}")

    # 执行抽检
    check_samples("Wind_Zero", categories["Wind_Zero"], 5)
    print("-" * 80)
    check_samples("Wind_High", categories["Wind_High"], 3)
    print("-" * 80)
    check_samples("Ceiling", categories["Ceiling"], 5)
    print("-" * 80)
    check_samples("Gap", categories["Gap"], 5)
    print("="*80)

if __name__ == "__main__":
    verify_dataset()