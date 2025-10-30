# ===> START: batch_visualize_styles_v3.py (最终安全版) <===
import logging
import os
from pathlib import Path
import datetime
import numpy as np
import torch
import random
from collections import defaultdict
import re

# [核心] 我们将复用您项目已有的工具
from mld.config import parse_args
from mld.data.get_data import get_datasets

try:
    from visual import visual_pos
except ImportError:
    print("\n[错误] 无法导入 'visual_pos' 函数。请检查您的项目结构。\n")
    exit()

def parse_style_dict(file_path):
    """解析 Style_name_dict.txt 文件"""
    style_to_ids = defaultdict(list)
    style_name_pattern = r'([A-Za-z]+)_'
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            
            seq_id, bvh_name = parts[0], parts[1]
            match = re.match(style_name_pattern, bvh_name)
            if match:
                style_name = match.group(1)
                # 确保不添加重复的 ID
                if seq_id not in style_to_ids[style_name]:
                    style_to_ids[style_name].append(seq_id)
    return style_to_ids

def visualize_single_motion(seq_id, style_name, cfg, datamodule, output_dir):
    """可视化单个 .npy 文件的核心逻辑"""
    npy_file_path = os.path.join(cfg.DATASET.STYLE100.ROOT, "new_joint_vecs", f"{seq_id}.npy")
    
    if not os.path.exists(npy_file_path):
        logging.warning(f"  文件未找到，跳过: {npy_file_path}")
        return

    logging.info(f"  正在处理: {os.path.basename(npy_file_path)} (风格: {style_name})")
    
    raw_features_np = np.load(npy_file_path)
    mean = datamodule.norms['mean']
    std = datamodule.norms['std']
    normalized_features_np = (raw_features_np - mean) / std
    features_tensor = torch.from_numpy(normalized_features_np).float()
    
    with torch.no_grad():
        joints_tensor = datamodule.feats2joints(features_tensor.unsqueeze(0))
    
    joints_np = joints_tensor[0].cpu().numpy()

    temp_npy_path = os.path.join(output_dir, f"{seq_id}_joints_temp.npy")
    np.save(temp_npy_path, joints_np)
    
    mp4_filename = f"{style_name}_{seq_id}.mp4"
    mp4_path = os.path.join(output_dir, mp4_filename)
    
    try:
        visual_pos(temp_npy_path, mp4_path, view_mode="camera_follow")
        logging.info(f"    => 视频已保存到: {mp4_path}")
    except Exception as e:
        logging.error(f"    可视化过程中发生错误: {e}", exc_info=True)
    finally:
        if os.path.exists(temp_npy_path):
            os.remove(temp_npy_path)

def main():
    # --- [核心修改] 我们不再使用 argparse，而是直接调用 parse_args ---
    # 这与您所有的脚本 (train.py, demo.py) 的行为完全一致
    cfg = parse_args(phase="demo") 
    
    # --- [手动配置区] ---
    # 您可以在这里修改每个风格抽样的数量
    SAMPLES_PER_STYLE = 3
    # --- [手动配置区结束] ---

    time_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join(cfg.FOLDER, "style_dataset_inspection", time_str)
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"所有可视化结果将保存在: {output_dir}")

    logging.info("正在加载 DataModule...")
    datamodule = get_datasets(cfg, logger=None, phase="val")[0]
    datamodule.setup()
    logging.info("DataModule 加载完毕。")

    style_dict_path = "/root/autodl-tmp/MyRepository/MCM-LDM/datasets/100StyleDataset/Style_name_dict.txt"
    logging.info(f"正在解析风格字典文件: {style_dict_path}")
    style_to_ids = parse_style_dict(style_dict_path)
    if not style_to_ids:
        logging.error("风格字典解析失败或为空。")
        return
    logging.info(f"成功解析出 {len(style_to_ids)} 种风格。")
    
    for style_name, seq_ids in sorted(style_to_ids.items()):
        logging.info(f"\n--- 开始检查风格: '{style_name}' (共 {len(seq_ids)} 个样本) ---")
        
        if len(seq_ids) > SAMPLES_PER_STYLE:
            ids_to_visualize = random.sample(seq_ids, SAMPLES_PER_STYLE)
        else:
            ids_to_visualize = seq_ids
            
        logging.info(f"  将可视化以下序列 ID: {ids_to_visualize}")
        
        for seq_id in ids_to_visualize:
            visualize_single_motion(seq_id, style_name, cfg, datamodule, output_dir)
            
    logging.info("\n--- 所有风格检查完毕 ---")

if __name__ == "__main__":
    main()
# ===> END: batch_visualize_styles_v3.py <===