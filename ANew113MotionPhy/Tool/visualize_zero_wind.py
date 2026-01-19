import os
import json
import numpy as np
import glob
import sys

# ================= 配置区域 =================
# 根目录
ROOT_DIR = "/root/autodl-tmp/MyRepository/MCM-LDM"
# 物理参数目录
JSON_DIR = os.path.join(ROOT_DIR, "datasets/PhysicsDataset/json_files")
# 动作数据目录 (注意：visual_pos 需要读取 .npy)
MOTION_DIR = os.path.join(ROOT_DIR, "datasets/PhysicsDataset/new_joint_vecs")
# 输出视频的文件夹
OUTPUT_DIR = "zero_wind_videos"
# 判定为0的阈值
ZERO_THRESHOLD = 10.0 
# ===========================================

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 上溯到项目根目录 /root/autodl-tmp/MyRepository/
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
# 将项目根目录添加到Python路径
sys.path.insert(0, project_root)


# 或者直接
# 导入visual
# import visual
# 获取 MCM-LDM 文件夹的绝对路径
project_root = "/root/autodl-tmp/MyRepository/MCM-LDM"
if project_root not in sys.path:
    sys.path.append(project_root)

# 现在可以直接引用 visual.py 了
import visual
from visual import visual_pos

def visualize_zeros():
    # 1. 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"结果将保存至: {OUTPUT_DIR}/")

    # 2. 扫描文件
    print(f"正在扫描: {JSON_DIR} ...")
    json_files = glob.glob(os.path.join(JSON_DIR, "W_*.json"))
    
    count = 0
    processed_count = 0
    
    print("-" * 60)
    print(f"{'文件名':<40} | {'JSON风力':<15} | {'状态'}")
    print("-" * 60)

    for jf in json_files:
        try:
            # 读取 JSON
            with open(jf, 'r') as f:
                data = json.load(f)
            
            params = data.get("parameters", {})
            wf = params.get("wind_force", {})
            mag = np.sqrt(wf.get('x', 0)**2 + wf.get('y', 0)**2)

            # 筛选条件：风力极小
            if mag < ZERO_THRESHOLD:
                count += 1
                
                # 获取对应的文件名 (去后缀)
                basename = os.path.splitext(os.path.basename(jf))[0]
                
                # 寻找对应的 NPY 文件
                npy_path = os.path.join(MOTION_DIR, basename + ".npy")
                
                if not os.path.exists(npy_path):
                    print(f"{basename:<40} | {mag:<15.2f} | [跳过] NPY缺失")
                    continue

                # 设置输出路径
                mp4_path = os.path.join(OUTPUT_DIR, basename + ".mp4")
                
                # === 核心：调用 visual_pos ===
                # 注意：visual_pos 通常会打印很多日志，可能会刷屏
                try:
                    visual_pos(str(npy_path), str(mp4_path))
                    print(f"{basename:<40} | {mag:<15.2f} | [成功] 生成视频")
                    processed_count += 1
                except Exception as e:
                    print(f"{basename:<40} | {mag:<15.2f} | [失败] 渲染错误: {e}")

        except Exception as e:
            print(f"处理文件出错: {jf}, 错误: {e}")

    print("-" * 60)
    print(f"处理完成。共发现 {count} 个0风样本，成功渲染 {processed_count} 个视频。")
    print(f"请进入文件夹 '{OUTPUT_DIR}' 查看结果。")

if __name__ == "__main__":
    visualize_zeros()