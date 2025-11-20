import numpy as np

# 替换成你的npy文件路径
motion_path = "/root/autodl-tmp/MyRepository/MCM-LDM/results/demo_outputs/PhysiMoS_Probe_Finetune_v1/heavy/jump.npy"

try:
    data = np.load(motion_path)
    
    print(f"File: {motion_path}")
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Contains NaN: {np.isnan(data).any()}")
    print(f"Contains Infinity: {np.isinf(data).any()}")
    print(f"Max value: {np.max(data)}")
    print(f"Min value: {np.min(data)}")
    print(f"Mean value: {np.mean(data)}")
    
except Exception as e:
    print(f"Error loading or analyzing file: {e}")