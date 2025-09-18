import numpy as np
import sys

def view_npy_file(file_path, num_rows=5):
    """
    .npy 文件是 NumPy 数组的二进制文件格式，它本身并没有“字段”或“列名”的概念，只有一个单一的数组。
    加载一个.npy文件并打印其属性及部分内容。

    参数:
    file_path (str): .npy文件的路径。
    num_rows (int): 想要打印的行数。
    """
    try:
        # 加载 .npy 文件
        data = np.load(file_path)

        # 打印数组的基本信息
        print(f"成功加载文件：{file_path}")
        print("-" * 30)
        print(f"数组的形状 (Shape): {data.shape}")
        print(f"数组的数据类型 (Dtype): {data.dtype}")
        
        # 打印部分数据
        print("\n--- 数组的前几行数据 ---")
        if data.ndim == 1:
            # 如果是一维数组
            print(data[:num_rows])
        elif data.ndim > 1:
            # 如果是多维数组，打印前几行
            print(data[:num_rows, ...])
        else:
            print("该数组是标量，无法按行打印。")

    except FileNotFoundError:
        print(f"错误：文件未找到，请检查路径。路径为: {file_path}")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")

if __name__ == "__main__":
    # 示例用法
    # 请将 'your_file.npy' 替换为你的 .npy 文件路径
    if len(sys.argv) > 1:
        npy_file_path = sys.argv[1]
    else:
        # 如果没有提供命令行参数，则使用这个默认路径
        npy_file_path = "/root/autodl-tmp/MyRepository/MCM-LDM/results/mld/SceMoDiff_All_LowCeiling/style_transfer2025-06-11-12-20/jump_hands_high_LowCeiling_2-5.npy" 
        
    view_npy_file(npy_file_path)