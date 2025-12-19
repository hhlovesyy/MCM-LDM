import numpy as np
import sys
import os

def check_npy_shape(file_path):
    """
    加载 .npy 文件并打印其 NumPy 数组的维度。
    """
    
    # 检查文件路径是否提供
    if not file_path:
        print("错误: 请提供 .npy 文件路径作为命令行参数。")
        return

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件未找到: {file_path}")
        return

    # 检查文件扩展名
    if not file_path.lower().endswith('.npy'):
        print(f"警告: 文件 {file_path} 似乎不是 .npy 文件。仍然尝试加载...")

    try:
        # 使用 numpy.load() 加载文件
        data = np.load(file_path)
        
        print("--- NPY 文件维度信息 ---")
        print(f"文件路径: {file_path}")
        print(f"数据类型 (dtype): {data.dtype}")
        print(f"数组维度 (shape): {data.shape}")
        print(f"元素总数 (size): {data.size}")
        print("--------------------------")
        
        # 打印前几个元素，以进行快速验证
        if data.size > 0:
             # 如果维度大于1，打印前几行/切片
            if data.ndim > 1:
                print(f"前几行切片:")
                # 打印第一个维度的前2个元素
                print(data[tuple(slice(0, 2) if i == 0 else slice(None) for i in range(data.ndim))])
            else:
                # 如果是1D数组，打印前10个元素
                print(f"前10个元素: {data[:10]}")

    except Exception as e:
        print(f"加载或处理文件时发生错误: {e}")

if __name__ == "__main__":
    # 从命令行参数获取文件路径 (sys.argv[0] 是脚本名, sys.argv[1] 是第一个参数)
    file_path = "/root/autodl-tmp/MyRepository/MCM-LDM/demo/content_test_feats/000486-4.npy"
    check_npy_shape(file_path)