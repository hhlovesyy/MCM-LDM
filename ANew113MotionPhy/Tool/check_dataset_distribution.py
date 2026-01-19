import os
import json
import numpy as np
import matplotlib.pyplot as plt
import glob
from collections import Counter

# ================= 配置区域 =================
DATASET_JSON_DIR = "/root/autodl-tmp/MyRepository/MCM-LDM/datasets/PhysicsDataset/json_files"
# ===========================================

def get_category_from_filename(filename):
    """根据文件名判断样本类别"""
    name = os.path.basename(filename)
    if name.startswith("Ceiling"):
        return "LowCeiling"
    elif name.startswith("Gap"):
        return "NarrowGap"
    elif name.startswith("W_") or name.startswith("Wind"):
        return "Windy"
    else:
        return "Unknown"

def check_distribution():
    print(f"正在读取数据目录: {DATASET_JSON_DIR} ...")
    
    json_files = glob.glob(os.path.join(DATASET_JSON_DIR, "*.json"))
    total_files = len(json_files)
    
    if total_files == 0:
        print("错误：未找到任何 JSON 文件，请检查路径是否正确！")
        return

    print(f"找到 {total_files} 个样本。正在解析...")

    # 数据容器
    stats = {
        "LowCeiling": {"heights": []},
        "NarrowGap": {"widths": [], "offsets": []},
        "Windy": {"mags": [], "angles": []},
        "Unknown": []
    }
    
    # 计数器
    category_counter = Counter()

    for jf in json_files:
        try:
            category = get_category_from_filename(jf)
            category_counter[category] += 1
            
            with open(jf, 'r') as f:
                data = json.load(f)
                
            # 核心修正：参数在 'parameters' 键下
            params = data.get("parameters", {})
            
            # === 1. 处理天花板数据 ===
            if category == "LowCeiling":
                h = params.get("ceiling_height")
                if h is not None:
                    stats["LowCeiling"]["heights"].append(h)

            # === 2. 处理缝隙数据 ===
            elif category == "NarrowGap":
                w = params.get("gap_width")
                o = params.get("gap_offset")
                if w is not None: stats["NarrowGap"]["widths"].append(w)
                if o is not None: stats["NarrowGap"]["offsets"].append(o)

            # === 3. 处理风力数据 ===
            elif category == "Windy":
                wf = params.get("wind_force", {})
                x = wf.get("x", 0.0)
                y = wf.get("y", 0.0)
                # 计算模长
                mag = np.sqrt(x**2 + y**2)
                stats["Windy"]["mags"].append(mag)
                
                # 计算角度 (只统计有风的情况)
                if mag > 10.0: # 这里的数值很大(35350)，所以阈值设大点
                    angle = np.degrees(np.arctan2(y, x))
                    if angle < 0: angle += 360
                    stats["Windy"]["angles"].append(angle)

        except Exception as e:
            print(f"解析错误 {jf}: {e}")

    # ================= 打印统计结果 =================
    print("\n" + "="*30)
    print("【分类统计报告】")
    print(f"总文件数: {total_files}")
    for cat, count in category_counter.items():
        print(f"  - {cat}: {count} 个")
    print("="*30 + "\n")

    # ================= 开始绘图 =================
    plt.figure(figsize=(18, 12))
    plt.suptitle(f"Dataset Distribution (Total: {total_files})", fontsize=20)

    # 1. 风力大小
    plt.subplot(2, 3, 1)
    mags = stats["Windy"]["mags"]
    if mags:
        plt.hist(mags, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title(f"Wind Force Magnitude\n(Count: {len(mags)})")
        plt.xlabel("Force Value")
    else:
        plt.text(0.5, 0.5, "No Wind Data", ha='center', fontsize=12)

    # 2. 风向角度
    plt.subplot(2, 3, 2)
    angles = stats["Windy"]["angles"]
    if angles:
        plt.hist(angles, bins=36, range=(0, 360), color='salmon', edgecolor='black', alpha=0.7)
        plt.title(f"Wind Direction\n(Count: {len(angles)})")
        plt.xlabel("Angle (Degree)")
        plt.xticks([0, 90, 180, 270], ['E', 'N', 'W', 'S'])
    else:
        plt.text(0.5, 0.5, "No Wind Angle Data", ha='center', fontsize=12)

    # 3. 天花板高度
    plt.subplot(2, 3, 3)
    heights = stats["LowCeiling"]["heights"]
    if heights:
        plt.hist(heights, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
        plt.title(f"Ceiling Height\n(Count: {len(heights)})")
        plt.xlabel("Height (cm)")
    else:
        plt.text(0.5, 0.5, "No Ceiling Data", ha='center', fontsize=12)

    # 4. 缝隙宽度
    plt.subplot(2, 3, 4)
    widths = stats["NarrowGap"]["widths"]
    if widths:
        plt.hist(widths, bins=20, color='gold', edgecolor='black', alpha=0.7)
        plt.title(f"Gap Width\n(Count: {len(widths)})")
        plt.xlabel("Width (cm)")
    else:
        plt.text(0.5, 0.5, "No Gap Width Data", ha='center', fontsize=12)

    # 5. 缝隙偏移
    plt.subplot(2, 3, 5)
    offsets = stats["NarrowGap"]["offsets"]
    if offsets:
        plt.hist(offsets, bins=20, color='violet', edgecolor='black', alpha=0.7)
        plt.title(f"Gap Offset\n(Count: {len(offsets)})")
        plt.xlabel("Offset (cm)")
    else:
        plt.text(0.5, 0.5, "No Gap Offset Data", ha='center', fontsize=12)

    # 6. 总体类别饼图
    plt.subplot(2, 3, 6)
    labels = list(category_counter.keys())
    sizes = list(category_counter.values())
    if sizes:
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
        plt.title("Category Composition")
    
    plt.tight_layout()
    save_path = "dataset_distribution_v2.png"
    plt.savefig(save_path)
    print(f"统计图表已保存至: {save_path}")

if __name__ == "__main__":
    check_distribution()