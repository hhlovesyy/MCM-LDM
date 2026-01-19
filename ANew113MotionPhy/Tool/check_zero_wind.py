import os
import json
import numpy as np
import matplotlib.pyplot as plt
import glob

# ================= 配置区域 =================
DATASET_JSON_DIR = "/root/autodl-tmp/MyRepository/MCM-LDM/datasets/PhysicsDataset/json_files"
# 阈值：风力模长小于多少算作“0风”？(UE5有时候会有浮点误差，设为 1.0 很安全)
ZERO_THRESHOLD = 10.0 
# ===========================================

def check_zero_wind():
    print(f"正在扫描目录: {DATASET_JSON_DIR} ...")
    
    # 找到所有 W 开头的文件
    json_files = glob.glob(os.path.join(DATASET_JSON_DIR, "W_*.json"))
    # 也可以加上 Wind_ 开头的以防万一
    json_files += glob.glob(os.path.join(DATASET_JSON_DIR, "Wind_*.json"))
    
    if not json_files:
        print("未找到任何风力(W_*)数据文件！")
        return

    zero_wind_vectors = []
    zero_wind_files = []
    
    print(f"找到 {len(json_files)} 个风力样本，正在筛选 '0风' 数据...")
    print("-" * 60)
    print(f"{'文件名 (部分)':<40} | {'模长':<10} | {'JSON数据 (X, Y)'}")
    print("-" * 60)

    count = 0
    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
            
            params = data.get("parameters", {})
            wf = params.get("wind_force", {})
            
            x = wf.get('x', 0.0)
            y = wf.get('y', 0.0)
            mag = np.sqrt(x**2 + y**2)
            
            # 筛选逻辑：模长极小
            if mag < ZERO_THRESHOLD:
                count += 1
                filename = os.path.basename(jf)
                
                # 记录数据用于画图
                zero_wind_vectors.append([x, y])
                zero_wind_files.append(filename)
                
                # 打印前20个看看样子
                if count <= 20:
                    print(f"{filename:<40} | {mag:<10.4f} | ({x:.2f}, {y:.2f})")
                    
        except Exception as e:
            continue

    print("-" * 60)
    print(f"扫描结束。在 {len(json_files)} 个风力文件中，有 {count} 个是 '0风' (Mag < {ZERO_THRESHOLD})。")

    # ================= 可视化部分 =================
    if count == 0:
        print("没有找到 0 风数据，无需绘制。")
        return

    vectors = np.array(zero_wind_vectors)
    
    # 检查是否全是纯0
    is_pure_zero = np.all(vectors == 0)
    
    plt.figure(figsize=(8, 8))
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    
    if is_pure_zero:
        print("\n【关键发现】: 所有的 '0风' 数据，JSON 里的 X 和 Y 确实都是纯 0.0！")
        print("这意味着：文件名里可能写着 'Left/Right'，但物理参数里丢失了方向信息。")
        plt.scatter([0], [0], color='red', s=200, label='All Zero Points')
        plt.title(f"Zero Wind Vectors (All {count} points are strictly 0,0)")
    else:
        print("\n【关键发现】: '0风' 数据中存在微小数值！")
        plt.scatter(vectors[:, 0], vectors[:, 1], alpha=0.5, color='blue')
        plt.title(f"Zero Wind Vectors Distribution (Count: {count})")
        # 画箭头
        for v in vectors:
            if np.linalg.norm(v) > 0:
                plt.arrow(0, 0, v[0], v[1], head_width=0.1, head_length=0.1, fc='blue', ec='blue')

    plt.xlabel("Wind X")
    plt.ylabel("Wind Y")
    plt.grid(True)
    plt.savefig("check_zero_wind_dir.png")
    print("\n可视化结果已保存至: check_zero_wind_dir.png")

if __name__ == "__main__":
    check_zero_wind()