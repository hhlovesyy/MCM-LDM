import pickle
import sys
import torch
import numpy as np

def print_object_details(obj, prefix="", level=0, max_level=3):
    """
    递归地打印对象中的所有内容，包括其类型、形状和数据类型。
    """
    indent = "  " * level
    
    # 限制递归深度，防止无限循环或打印过多内容
    if level > max_level:
        print(f"{indent}... (达到最大打印深度)")
        return

    # 打印 PyTorch Tensor 的信息
    if isinstance(obj, torch.Tensor):
        print(f"{indent}类型: {type(obj).__name__}, 形状: {list(obj.shape)}, 数据类型: {obj.dtype}")
        
    # 打印 NumPy 数组的信息
    elif isinstance(obj, np.ndarray):
        print(f"{indent}类型: {type(obj).__name__}, 形状: {list(obj.shape)}, 数据类型: {obj.dtype}")

    # 打印字典的内容
    elif isinstance(obj, dict):
        print(f"{indent}类型: {type(obj).__name__}, 包含 {len(obj)} 个键")
        for key, value in obj.items():
            print(f"{indent}  键 '{key}':")
            # 递归调用
            print_object_details(value, prefix=f"{prefix}.{key}", level=level + 1, max_level=max_level)
            
    # 打印列表或元组的内容
    elif isinstance(obj, (list, tuple)):
        print(f"{indent}类型: {type(obj).__name__}, 包含 {len(obj)} 个元素")
        for i, item in enumerate(obj):
            print(f"{indent}  元素 {i}:")
            # 递归调用
            print_object_details(item, prefix=f"{prefix}[{i}]", level=level + 1, max_level=max_level)
            
    # 打印其他基本类型
    else:
        # 对于较小的基本类型，直接打印其值
        if sys.getsizeof(obj) < 100:
            print(f"{indent}类型: {type(obj).__name__}, 值: {obj}")
        else:
            print(f"{indent}类型: {type(obj).__name__}, 大小: {sys.getsizeof(obj)} 字节")

def analyze_pkl_file(file_path):
    """
    加载并解析一个 .pkl 文件，并打印其内容。
    """
    try:
        with open(file_path, 'rb') as f:
            # 使用 pickle.load() 加载数据
            data = pickle.load(f)
            
            print(f"--- 正在解析文件: {file_path} ---")
            
            # 调用函数来打印数据结构
            print_object_details(data)
            
            print("\n--- 解析完成 ---")

    except FileNotFoundError:
        print(f"错误：文件未找到，请检查路径。路径为: {file_path}")
    except pickle.UnpicklingError as e:
        print(f"错误：无法反序列化文件，可能文件已损坏或不是有效的 pickle 文件。详细信息：{e}")
    except Exception as e:
        print(f"解析文件时发生意外错误: {e}")

if __name__ == "__main__":
    # 示例用法
    # 请将 'your_file.pkl' 替换为你的 .pkl 文件路径
    if len(sys.argv) > 1:
        pkl_file_path = sys.argv[1]
    else:
        # 如果没有提供命令行参数，则使用这个默认路径
        pkl_file_path = "test20250918.pkl" 
        
    analyze_pkl_file(pkl_file_path)