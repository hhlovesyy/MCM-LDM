import numpy as np
import os

# 定义你要操作的文件路径
FILE_PATH = "/root/autodl-tmp/MyRepository/MCM-LDM/datasets/humanml3d_scene/Mean.npy"
FILE_PATH_2 = "/root/autodl-tmp/MyRepository/MCM-LDM/datasets/humanml3d_scene/Std.npy"
# 定义要读取的前 N 维数量
N_DIMENSIONS = 5

def read_first_n_dimensions(file_path, n):
    """
    加载 .npy 文件，打印其维度信息，并显示数组的第一个维度上的前 N 个切片。
    """
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 错误: 文件未找到: {file_path}")
        return

    try:
        # 1. 使用 numpy.load() 加载文件
        mean_vec = np.load(file_path)
        
        print("--- NPY 文件信息 ---")
        print(f"文件路径: {file_path}")
        print(f"数组维度 (Shape): {mean_vec.shape}")
        print(f"数据类型 (Dtype): {mean_vec.dtype}")
        print("--------------------")

        # 2. 检查数组是否可切片，并读取前 N 维
        if mean_vec.size == 0:
            print("警告: 数组为空，无法读取切片。")
            return
        
        # 确保 N 不超过数组的第一个维度大小
        n_to_read = min(n, mean_vec.shape[0])

        # 使用切片 mean_vec[:N] 读取第一个维度上的前 N 个元素
        # 如果数组是多维的 (例如 (100, 263))，这会取出前 N 行/切片
        print(f"✅ 读取前 {n_to_read} 个切片 (mean_vec[:{n_to_read}]):")
        
        # 获取前 n_to_read 个切片
        prefix_data = mean_vec[:n_to_read]
        
        # 打印切片结果
        print(prefix_data)

    except Exception as e:
        print(f"加载或处理文件时发生错误: {e}")

if __name__ == "__main__":
    read_first_n_dimensions(FILE_PATH, N_DIMENSIONS)
    read_first_n_dimensions(FILE_PATH_2, N_DIMENSIONS)