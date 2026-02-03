import os
import sys
import argparse
import glob
from pathlib import Path
from tqdm import tqdm

# 确保能导入项目中的模块
sys.path.append(os.getcwd())

# 尝试导入 visual_pos
try:
    from visual import visual_pos
except ImportError:
    print("Error: 找不到 visual.py。请确保将此脚本放在项目根目录下运行。")
    sys.exit(1)

def parse_args():   
    parser = argparse.ArgumentParser(description="批量可视化 HumanML3D 数据集")
    
    # 默认路径指向 HumanML3D 的 new_joint_vecs (通常存放 npy 的地方)
    # 如果你的 npy 在 texts 同级目录下，请修改这里
    default_input = "./datasets/humanml3d/new_joints"
    default_text = "./datasets/humanml3d/texts"
    
    parser.add_argument("--input_dir", type=str, default=default_input, help="存放 .npy 文件的目录")
    parser.add_argument("--text_dir", type=str, default=default_text, help="存放 .txt 描述文件的目录 (用于预览文本)")
    parser.add_argument("--output_dir", type=str, default="./vis_dataset_debug", help="视频保存目录")
    
    parser.add_argument("--start", type=int, default=0, help="从第几个文件开始 (Index)")
    parser.add_argument("--count", type=int, default=10, help="一共渲染多少个视频")
    parser.add_argument("--keyword", type=str, default=None, help="[可选] 仅渲染描述中包含此关键词的动作 (例如 'forward')")
    
    return parser.parse_args()

def get_text_content(npy_stem, text_dir):
    """尝试读取对应的文本描述"""
    # 文本文件通常是名字一样的 .txt
    # 这里的命名规则需要根据你的实际数据集调整，通常是 000000.txt
    # 有时候可能是 M000000.txt，这里假设是直接对应的
    txt_path = Path(text_dir) / f"{npy_stem}.txt"
    
    if txt_path.exists():
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                # HumanML3D 的文本格式通常是 "caption#..."，我们只取第一部分
                content = f.read().strip()
                # 简单清洗一下，HumanML3D 有时候会有多个 caption 换行
                lines = content.split('\n')
                # 取第一行，去掉 # 及其后面的内容
                first_line = lines[0].split('#')[0]
                return first_line
        except Exception:
            return "[Read Error]"
    return "[No Text Found]"

def main():
    args = parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 获取所有 npy 文件并排序
    print(f"扫描文件: {input_path}/*.npy")
    npy_files = sorted(list(input_path.glob("*.npy")))
    
    total_files = len(npy_files)
    print(f"总共找到 {total_files} 个文件。")
    
    if total_files == 0:
        print("未找到文件，请检查 --input_dir 路径是否正确。")
        return

    # 2. 确定处理范围
    start_idx = args.start
    end_idx = min(start_idx + args.count, total_files)
    
    # 如果指定了关键词搜索，逻辑稍微变一下：我们遍历所有文件直到找到 count 个符合条件的
    target_files = []
    
    print("-" * 50)
    print(f"开始筛选/处理... 目标: {args.count} 个视频")
    if args.keyword:
        print(f"过滤关键词: '{args.keyword}'")
    print("-" * 50)

    processed_count = 0
    current_idx = start_idx
    
    # 使用 tqdm 进度条，或者手动循环
    while processed_count < args.count and current_idx < total_files:
        npy_file = npy_files[current_idx]
        stem = npy_file.stem # 文件名不带后缀 (e.g., 000000)
        
        # 获取文本描述
        caption = get_text_content(stem, args.text_dir)
        
        # 关键词过滤
        if args.keyword:
            if args.keyword.lower() not in caption.lower():
                current_idx += 1
                continue # 跳过不包含关键词的
        
        # 准备输出路径
        save_path = output_path / f"{stem}.mp4"
        
        print(f"[{current_idx}] {stem}.npy | Caption: {caption}")
        
        try:
            # 调用 visual.py 中的 visual_pos 函数
            # visual_pos(motion_path, save_path, caption)
            visual_pos(str(npy_file), str(save_path), caption)
            print(f"   -> Saved to: {save_path}")
            processed_count += 1
        except Exception as e:
            print(f"   -> [Render Error] {e}")
            # 如果是 HumanML3D 的原始数据，可能是 feature 格式，
            # 而 visual_pos 期望的是 joint position 格式。
            # 如果报错，可能需要这一步转换。但根据你的 demo_physics.py，
            # 这里先假设 visual_pos 能处理或者数据已经是 pos。
        
        current_idx += 1

    print("-" * 50)
    print(f"完成。共生成 {processed_count} 个视频。")
    print(f"下次继续请使用: --start {current_idx}")

if __name__ == "__main__":
    main()