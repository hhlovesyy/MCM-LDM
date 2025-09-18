import numpy as np
import pickle
import sys
import os

def convert_npy_to_pkl(npy_file_path, pkl_output_path, text_description):
    """
    将一个 .npy 文件转换为特定的 .pkl 文件格式。

    参数:
    npy_file_path (str): 输入的 .npy 文件路径。
    pkl_output_path (str): 输出的 .pkl 文件路径。
    text_description (str): 描述动作的文本。
    """
    try:
        # 1. 加载 .npy 文件
        print(f"正在加载 .npy 文件: {npy_file_path}...")
        joints_data = np.load(npy_file_path)

        # 确保数据形状是三维的 (frames, joints, coordinates)
        if joints_data.ndim != 3:
            print("错误: .npy 文件中的数组必须是三维的 (frames, joints, coords)。")
            return

        # 2. 从 .npy 数组的形状中计算 length
        length = joints_data.shape[0]

        # 3. 构造字典数据
        pkl_data = {
            'joints': joints_data,
            'text': text_description,
            'length': length,
            'hint': None
        }

        # 4. 将字典保存为 .pkl 文件
        with open(pkl_output_path, 'wb') as f:
            pickle.dump(pkl_data, f)
            
        print("-" * 40)
        print(f"转换成功！")
        print(f"文件已保存至: {pkl_output_path}")
        print("以下是保存的数据结构：")
        print(f"  'joints' 形状: {joints_data.shape}")
        print(f"  'text' 内容: '{text_description}'")
        print(f"  'length' 长度: {length}")
        print(f"  'hint' 内容: None")
        print("-" * 40)

    except FileNotFoundError:
        print(f"错误: 文件未找到。请检查路径: {npy_file_path}")
    except Exception as e:
        print(f"转换过程中发生错误: {e}")

if __name__ == "__main__":
    # 示例用法
    # 使用命令行参数
    if len(sys.argv) < 4:
        print("用法: python convert_script.py <input_npy_file> <output_pkl_file> \"<text_description>\"")
        print("\n示例:")
        print("  python convert_script.py data/walk.npy output/walk_data.pkl \"the man walks straight ahead.\"")
        sys.exit(1)

    input_npy_path = sys.argv[1]
    output_pkl_path = sys.argv[2]
    text_desc = sys.argv[3]
    
    convert_npy_to_pkl(input_npy_path, output_pkl_path, text_desc)