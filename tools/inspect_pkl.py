import pickle
import numpy as np
import torch
import os

def inspect_pkl(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    print(f"{'='*20} Inspecting: {os.path.basename(file_path)} {'='*20}")
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return

    if isinstance(data, dict):
        keys = list(data.keys())
        print(f"Root Type: dict | Keys found: {keys}\n")
        
        for key, value in data.items():
            print(f"--- Key: ['{key}'] ---")
            analyze_value(value)
            print("")
    else:
        print(f"Root Type: {type(data)}")
        analyze_value(data)

def analyze_value(value):
    # 处理 Numpy Array
    if isinstance(value, np.ndarray):
        print(f"  Type: numpy.ndarray")
        print(f"  Shape: {value.shape}")
        print(f"  Dtype: {value.dtype}")
        if np.issubdtype(value.dtype, np.number):
            print(f"  Stats: Min={value.min():.4f}, Max={value.max():.4f}, Mean={value.mean():.4f}")
            # 如果维度较小，打印具体值
            if value.size < 10:
                print(f"  Value: {value}")
    
    # 处理 Torch Tensor
    elif isinstance(value, torch.Tensor):
        print(f"  Type: torch.Tensor")
        print(f"  Shape: {value.shape}")
        print(f"  Device: {value.device}")
        if value.numel() > 0 and not value.is_complex():
            print(f"  Stats: Min={value.min():.4f}, Max={value.max():.4f}, Mean={value.mean().item():.4f}")

    # 处理 List/Tuple
    elif isinstance(value, (list, tuple)):
        print(f"  Type: {type(value).__name__}")
        print(f"  Length: {len(value)}")
        if len(value) > 0:
            print(f"  First Element Type: {type(value[0])}")
            # 如果是纯数字列表
            try:
                arr = np.array(value)
                if np.issubdtype(arr.dtype, np.number):
                     print(f"  Stats (as array): Min={arr.min():.4f}, Max={arr.max():.4f}")
            except:
                pass

    # 处理标量
    elif isinstance(value, (int, float, str, bool)):
        print(f"  Type: {type(value).__name__}")
        print(f"  Value: {value}")
    
    else:
        print(f"  Type: {type(value)}")

if __name__ == "__main__":
    # 修改这里为你的pkl文件路径
    target_file = "/root/autodl-tmp/MyRepository/MotionLCM/MotionLCM/walk_circle_R0D1_anim.pkl" 
    # 如果你想测试，可以先运行一下看看你现在的 _mesh.pkl 里有什么
    inspect_pkl(target_file)