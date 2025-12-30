import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 导入你提供的渲染接口
from visual import visual_pos 

def batch_convert_npy_to_mp4():
    # 1. 路径设置 (直接写死)
    input_dir = "/root/autodl-tmp/MyRepository/MCM-LDM/demo/content_test_joints"
    output_dir = os.path.join(input_dir, "render_videos") # 子文件夹路径
    
    # 2. 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 3. 遍历目录下的所有 npy 文件
    files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    
    if not files:
        print(f"No .npy files found in {input_dir}")
        return

    print(f"Found {len(files)} files to process...")

    for file_name in tqdm(files):
        # 拼凑完整的输入路径
        npy_path = os.path.join(input_dir, file_name)
        
        # 拼凑输出路径 (将 .npy 替换为 .mp4)
        mp4_name = file_name.replace('.npy', '.mp4')
        mp4_path = os.path.join(output_dir, mp4_name)
        
        try:
            # 4. 调用接口进行渲染
            # 注意：如果 visual_pos 内部需要读取保存后的文件，npy_path 已经是现成的路径
            visual_pos(npy_path, mp4_path)
        except Exception as e:
            print(f"Error rendering {file_name}: {e}")

    print("\nBatch processing completed!")
    print(f"Videos are saved in: {output_dir}")

if __name__ == "__main__":
    batch_convert_npy_to_mp4()