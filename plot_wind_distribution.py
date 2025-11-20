import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_distribution(json_dir):
    files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    wind_vectors = []
    
    for fname in tqdm(files):
        with open(os.path.join(json_dir, fname), 'r') as f:
            data = json.load(f)
            wf = data["parameters"]["wind_force"]
            # 记录原始 UE5 的 X, Y (忽略 Z)
            wind_vectors.append([wf['x'], wf['y']])
            
    wind_vectors = np.array(wind_vectors)
    
    # 绘图
    plt.figure(figsize=(6, 6))
    plt.scatter(wind_vectors[:, 0], wind_vectors[:, 1], alpha=0.5)
    plt.title(f"Wind Force Distribution (N={len(files)})")
    plt.xlabel("UE5 Wind X")
    plt.ylabel("UE5 Wind Y")
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.savefig("wind_distribution.png")
    print("Saved distribution plot to wind_distribution.png")

if __name__ == "__main__":
    # 修改为你的 json 文件夹路径
    json_dir = "/root/autodl-tmp/MyRepository/MCM-LDM/datasets/PhysicsDataset/json_files" 
    plot_distribution(json_dir)