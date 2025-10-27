# debug_normalization.py (v2.0 - Final Correct Version)

import torch
import yaml
from mld.config import parse_args # [正确修复] 使用你项目中的 parse_args 函数
from mld.data.get_data import get_datasets
import os

# --- 步骤 1: 加载你的配置文件和数据模块 (与 train.py 完全一致) ---
print("Initializing Config and DataModule...")
# parse_args() 会自动找到并解析你的 'configs/config_mld_mixed.yaml' (如果通过命令行参数指定)
# 为了脚本的独立性，我们在这里手动指定
cfg = parse_args(phase="train") 

datamodule = get_datasets(cfg)[0]
datamodule.setup()
train_loader = datamodule.train_dataloader()
print("Config and DataModule setup complete.")

# --- 步骤 2: 从数据加载器中取出一个 batch ---
print("Fetching one batch of data...")
batch = next(iter(train_loader))
print("Batch fetched successfully.")

# --- 步骤 3: 手动加载 MotionClip 模型 (与 mld.py 中完全一致) ---
from mld.models.motionclip_263.utils.get_model_and_data import get_model_and_data
from mld.models.motionclip_263.utils.misc import load_model_wo_clip
from mld.utils.temos_utils import lengths_to_mask

def read_yaml_to_dict(yaml_path: str):
    with open(yaml_path) as file:
        return yaml.load(file.read(), Loader=yaml.FullLoader)

device = f'cuda:{cfg.DEVICE[0]}'
print(f"Loading MotionClip to device: {device}")
parameters = read_yaml_to_dict("configs/motionclip_config/motionclip_params_263.yaml")
parameters["device"] = device
motionclip = get_model_and_data(parameters, split='vald')
checkpointpath = "checkpoints/motionclip_checkpoint/motionclip.pth.tar"
state_dict = torch.load(checkpointpath, map_location=device)
load_model_wo_clip(motionclip, state_dict)
motionclip.eval() # 确保是评估模式
print("MotionClip loaded successfully.")

# --- 步骤 4: [核心诊断] 模拟 mld.py 中的反归一化和编码过程 ---
print("Starting diagnostic test...")

# a. 从 batch 中解包数据
feats_ref = batch["motion"].to(device)
lengths = batch["length"]
is_text_guided = batch['is_text_guided']

# b. [关键] 使用固定的、来自 HumanML3D 的 mean/std
#    我们在这里复现导致问题的“错误归一化”
humanml3d_mean = torch.tensor(datamodule.norms['mean']).to(device)
humanml3d_std = torch.tensor(datamodule.norms['std']).to(device)
print("Using HumanML3D's mean and std for anti-normalization.")

# c. [关键] 只对 100Style 的数据进行反归一化，因为它们是问题源头
style100_indices = is_text_guided
if style100_indices.any():
    print(f"Found {style100_indices.sum()} samples from 100Style dataset in this batch.")
    
    motion_seq_100style = feats_ref[style100_indices] * humanml3d_std + humanml3d_mean
    motion_seq_100style[..., :3] = 0.0
    motion_seq_100style = motion_seq_100style.unsqueeze(-1).permute(0, 2, 3, 1)
    lengths_100style = [lengths[i] for i, flag in enumerate(style100_indices) if flag]

    # d. 将这个“被错误处理过”的数据送入编码器
    print("Encoding the incorrectly anti-normalized motion sequence...")
    with torch.no_grad():
        motion_emb = motionclip.encoder({
            'x': motion_seq_100style.float(), # 确保输入是 float 类型
            'y': torch.zeros(motion_seq_100style.shape[0], dtype=int, device=device),
            'mask': lengths_to_mask(lengths_100style, device=device)
        })["mu"]

    # e. [最终审判] 检查输出中是否包含 NaN
    print("\n--- FINAL VERDICT ---")
    contains_nan = torch.isnan(motion_emb).any()
    print(f"Does the output motion embedding contain NaN? -> {contains_nan}")
    if contains_nan:
        print("CONFIRMED: The frozen MotionClip encoder produced NaN values when fed with incorrectly normalized data.")
        print("This proves the root cause is the normalization mismatch.")
    else:
        print("UNEXPECTED: No NaN detected. The issue might be even more subtle. Let's check the embedding values:")
        print("Min value:", torch.min(motion_emb))
        print("Max value:", torch.max(motion_emb))
        print("Mean value:", torch.mean(motion_emb))

else:
    print("This batch did not contain any data from 100Style. Please run the script again.")