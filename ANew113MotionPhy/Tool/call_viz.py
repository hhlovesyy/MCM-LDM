import json
import os
import torch
import numpy as np
import glob
import sys
import argparse

# ================= 配置 =================
PROJECT_ROOT = "/root/autodl-tmp/MyRepository/MCM-LDM"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 数据路径
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets")
JSON_DIR = os.path.join(DATASET_DIR, "PhysicsDataset/json_files")
MOTION_DIR = os.path.join(DATASET_DIR, "PhysicsDataset/new_joint_vecs")
MEAN_PATH = os.path.join(DATASET_DIR, "humanml3d/Mean.npy")
STD_PATH = os.path.join(DATASET_DIR, "humanml3d/Std.npy")
OUTPUT_DIR = "viz_result_called"

# ================= 核心：直接调用你同学的代码 =================
try:
    # 试图从 visual 导入渲染器
    from visual import visual_pos
    # 试图从你保存的文件 import 那个函数
    from my_utils import recover_from_ric 
    print(">>> 成功调用 my_utils.recover_from_ric 和 visual_pos")
except ImportError as e:
    print(f"【严重错误】导入失败: {e}")
    print("请确认：\n1. 你把同学的代码保存为了 'my_utils.py'\n2. 它在项目根目录下")
    sys.exit(1)

def process_file(npy_path, save_mp4, mean, std):
    # 1. 加载 263维 特征
    data = np.load(npy_path)
    data = torch.from_numpy(data).float()
    
    # 2. 反归一化 (这一步必须做，否则数值太小，recover出来就是原地不动)
    # 除非你同学的代码里已经内置了反归一化（通常 recover_from_ric 不包含）
    # data = data * std + mean
    
    # 3. 【直接调用】同学写的还原函数
    # 他的函数签名是: recover_from_ric(data, joints_num)
    # 他代码里写了 joints_num=22
    joints = recover_from_ric(data, 22)  # 返回 [Seq, 22, 3] 或者包含 Batch 维
    
    # 兼容性处理：转 numpy
    if isinstance(joints, torch.Tensor):
        joints = joints.detach().cpu().numpy()
        
    # 如果维度多了 (1, Seq, 22, 3)，去掉第一个
    if len(joints.shape) == 4:
        joints = joints[0]
        
    # 4. 保存临时文件给 visual_pos 用
    temp_name = save_mp4.replace(".mp4", "_temp.npy")
    np.save(temp_name, joints)
    
    # 5. 渲染
    try:
        visual_pos(temp_name, save_mp4)
        print(f"OK: {os.path.basename(save_mp4)}")
    except Exception as e:
        print(f"Render Error: {e}")
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", type=str, help="测试单个NPY文件")
    args = parser.parse_args()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载均值方差
    mean = torch.from_numpy(np.load(MEAN_PATH)).float()
    std = torch.from_numpy(np.load(STD_PATH)).float()

    # 模式1: 指定单个文件验证 (验证脚本对不对)
    if args.single:
        process_file(args.single, os.path.join(OUTPUT_DIR, "single_test.mp4"), mean, std)
        return

    # 模式2: 扫描所有 0 风数据
    print("正在扫描 0 风数据...")
    json_files = glob.glob(os.path.join(JSON_DIR, "W_*.json"))
    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                d = json.load(f)
            # 简单粗暴判断 0 风
            wf = d.get("parameters", {}).get("wind_force", {})
            mag = wf.get('x',0)**2 + wf.get('y',0)**2
            
            if mag < 100: # 阈值稍微大点防止浮点误差
                name = os.path.splitext(os.path.basename(jf))[0]
                npy = os.path.join(MOTION_DIR, name + ".npy")
                if os.path.exists(npy):
                    process_file(npy, os.path.join(OUTPUT_DIR, f"{name}.mp4"), mean, std)
        except:
            pass

if __name__ == "__main__":
    main()