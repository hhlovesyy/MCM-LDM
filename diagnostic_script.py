# diagnostic_script_v3_final.py
from omegaconf import OmegaConf
from mld.data.mixed_datamodule import MixedDataModule
import torch
import numpy as np # <--- 导入 numpy

def check_dataset_shapes():
    # 1. 加载您的配置文件
    cfg_path = "/root/autodl-tmp/MyRepository/MCM-LDM/configs/config_mld_mixed.yaml" # <--- 请再次替换为您的配置文件路径
    cfg = OmegaConf.load(cfg_path)

    # 2. 实例化 DataModule
    print("Instantiating DataModule...")
    datamodule = MixedDataModule(cfg)
    
    # 3. 手动调用 setup 来创建数据集
    print("Running datamodule.setup()...")
    datamodule.setup()
    
    # 4. 从训练集和验证集中各取一个样本
    print("\n--- Checking Raw Sample Shapes (pre-collate) ---")
    
    # 从组合的训练集中取第一个样本 (来自 HumanML3D 训练集)
    train_sample = datamodule.train_dataset.datasets[0][0]
    train_motion_data = train_sample[0]
    
    # [修正] 正确处理 numpy.ndarray 或 torch.Tensor
    if not isinstance(train_motion_data, (np.ndarray, torch.Tensor)):
        print(f"Error: The first element of the training sample is not a NumPy array or Tensor, but a {type(train_motion_data)}.")
        return
    train_motion_shape = train_motion_data.shape
    print(f"Shape of one TRAIN sample's 'motion' array: {train_motion_shape}")

    # 从验证集中取第一个样本
    val_sample = datamodule.val_dataset[0]
    val_motion_data = val_sample[0]
    
    # [修正] 同样，正确处理 numpy.ndarray 或 torch.Tensor
    if not isinstance(val_motion_data, (np.ndarray, torch.Tensor)):
        print(f"Error: The first element of the validation sample is not a NumPy array or Tensor, but a {type(val_motion_data)}.")
        return
    val_motion_shape = val_motion_data.shape
    print(f"Shape of one VALIDATION sample's 'motion' array: {val_motion_shape}")

    # 5. 结论
    print("\n--- Conclusion ---")
    if train_motion_shape == val_motion_shape:
        print("✅ Shapes are consistent at the Dataset level.")
        print("This is unexpected. If the error persists, the issue might be in how `mld_collate` handles them differently, which is highly unlikely but possible.")
    else:
        print("🔥🔥🔥 BINGO! The shapes are INCONSISTENT at the raw Dataset level.")
        print(f"Train shape ({train_motion_shape}) implies (Length, Features).")
        print(f"Validation shape ({val_motion_shape}) implies (Features, Length).")
        print("\nThis is the definitive root cause of your RuntimeError.")
        print("SOLUTION: You must fix the data preprocessing pipeline for the validation set to ensure it saves the .npy files with the same (Length, Features) dimension order as the training set.")

if __name__ == "__main__":
    check_dataset_shapes()