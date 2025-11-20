import logging
import os
import json
from pathlib import Path
import numpy as np
import torch
from collections import OrderedDict

# MLD 核心模块
from mld.config import parse_args
from mld.data.get_data import get_datasets
from mld.models.get_model import get_model
from visual import visual_pos

import matplotlib.pyplot as plt
import seaborn as sns

# [新增] 硬编码场景列表，必须与训练时的 Dataset 一致
SCENE_CATEGORIES = [
    "Front", "Back", "Left", "Right", 
    "FrontLeft", "FrontRight", "BackLeft", "BackRight"
]

# --- [配置区] 请确保这里指向正确的文件 ---
path_config = {
    # 官方预训练权重 (地基)
    "pretrained_vae": "checkpoints/vae_checkpoint/vae7.ckpt",
    "pretrained_denoiser": "checkpoints/denoiser_checkpoint/denoiser.ckpt",
    
    # 你的微调权重 (灵魂 - 163MB那个)
    "finetuned_checkpoint": "/root/autodl-tmp/MyRepository/MCM-LDM/experiments/mld/PhysiMoS_Probe_Finetune_v1/checkpoints/epoch=299.ckpt" 
}

def plot_attention(attn_weights, scene_name, save_path):
    """
    绘制 Attention 热力图
    attn_weights: Tensor [1, 1, 4] (Batch, Query, Key)
    """
    # 1. 转换数据
    # 取出第一个样本，squeeze掉 batch 和 query 维
    # 最终形状应该是 (4,) 代表 4 个物理参数的权重
    data = attn_weights[0].squeeze().cpu().numpy() 
    
    # 2. 定义标签 (对应 scenes.json 里的 physical_parameters_desc)
    labels = ["WindX", "WindY", "WindStrength"]
    
    # 3. 绘图
    plt.figure(figsize=(8, 3))
    # 把它变成 (1, 4) 矩阵方便画热力图
    sns.heatmap(data.reshape(1, -1), annot=True, cmap="Reds", 
                xticklabels=labels, yticklabels=[scene_name],
                vmin=0, vmax=1, cbar=True, square=True)
    
    plt.title(f"SCPA Attention: What matters for '{scene_name}'?")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  -> Attention heatmap saved: {save_path}")

# def load_physics_from_json(json_path, scenes_json_path):
#     """ 从JSON文件加载物理参数和场景信息 """
#     with open(json_path, 'r') as f:
#         physics_data = json.load(f)
#     with open(scenes_json_path, 'r') as f:
#         scenes_config = json.load(f)
#     scene_name = physics_data["scene_category"]
#     phys_params = physics_data["physical_parameters"]
#     scene_list = scenes_config["scene_categories"]
#     if scene_name not in scene_list:
#         raise ValueError(f"Scene '{scene_name}' in '{json_path}' not found in scenes configuration file!")
#     scene_idx = scene_list.index(scene_name)
#     scene_cat = torch.zeros(len(scene_list))
#     scene_cat[scene_idx] = 1.0
#     return torch.tensor(phys_params).float(), scene_cat.float()
    
# [替换] 原来的 load_physics_from_json 替换为这个：
def parse_inference_json(json_path):
    """ 
    直接从测试 JSON 解析推理条件 
    Input JSON Example: { "scene_category": "Right", "physical_parameters": [1.0, 0.0, 0.0, 1.0] }
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # 1. 解析场景 One-Hot,这个是多个场景的情况，把前后左右风认为是不同的场景
    # scene_name = data["scene_category"]
    # if scene_name not in SCENE_CATEGORIES:
    #     raise ValueError(f"Scene '{scene_name}' not found in SCENE_CATEGORIES list!")
    
    # scene_idx = SCENE_CATEGORIES.index(scene_name)
    # scene_cat = torch.zeros(len(SCENE_CATEGORIES))
    # scene_cat[scene_idx] = 1.0
        
    # [修改] 不再检查 json 里的具体方向，或者强制忽略它
    # 无论 json 里写的是 "Right" 还是 "Left"，我们都生成 "Windy" 的 One-Hot
    scene_cat = torch.zeros(1)
    scene_cat[0] = 1.0
    
    # [核心修复] 增加坐标系转换逻辑
    raw_params = data["physical_parameters"] # 假设是 [x, y, z, mag] (UE5 Z-up)
    
    raw_x = raw_params[0]
    raw_y = raw_params[1] # UE5 Right/Left axis
    # raw_z = raw_params[2] # UE5 Up/Down axis
    mag   = raw_params[2]

    # # 对应 Dataset 里的逻辑: wind_vec = np.array([raw_x, raw_z, -raw_y])
    # # HumanML3D (Y-up)
    # new_x = raw_x
    # new_y = raw_z        # 原来的 Z 变成了 Y (垂直)
    # new_z = -1.0 * raw_y # 原来的 Y 变成了 -Z (深度/侧向)

    # 重新组装
    phys_params = torch.tensor([raw_x, raw_y, mag]).float()
    
    
    return phys_params, scene_cat


def manual_check_keys(model, state_dict, module_name="Module"):
    """ 手动检查关键权重是否存在，不依赖 load_state_dict 返回值 """
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    
    # 检查交集
    loaded_keys = model_keys.intersection(ckpt_keys)
    if len(loaded_keys) == 0:
        print(f"[WARNING] {module_name}: No keys matched! Checkpoint might be wrong.")
        return False
    
    print(f"[{module_name}] Matched {len(loaded_keys)} keys.")
    return True

def load_pretrained_weights(model, vae_path, denoiser_path):
    """
    [双重加载策略 - 第一步] 恢复预训练的主干网络 (VAE + Denoiser Backbone)
    """
    print(f"\n--- [Base Loader] Restoring Pretrained Backbone ---")
    
    # 1. 加载 VAE
    print(f"  -> Loading VAE from {vae_path}")
    if not os.path.exists(vae_path):
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_path}")
        
    # strict load vae model
    state_dict = torch.load(vae_path,
                            map_location="cpu")["state_dict"]
    # extract encoder/decoder
    from collections import OrderedDict
    vae_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.split(".")[0] == "vae":
            name = k.replace("vae.", "")
            vae_dict[name] = v
    model.vae.load_state_dict(vae_dict, strict=True)

    # 2. 加载 Denoiser 主干
    print(f"  -> Loading Denoiser Backbone from {denoiser_path}")
    if not os.path.exists(denoiser_path):
        raise FileNotFoundError(f"Denoiser checkpoint not found: {denoiser_path}")

    denoiser_ckpt = torch.load(denoiser_path, map_location="cpu")
    denoiser_state = denoiser_ckpt["state_dict"] if "state_dict" in denoiser_ckpt else denoiser_ckpt
    
    new_denoiser_dict = OrderedDict()
    # 自动判断前缀
    has_prefix = any(k.startswith("denoiser.") for k in denoiser_state.keys())
    
    for k, v in denoiser_state.items():
        if "sequence_pos_encoding.pe" in k: continue # 忽略位置编码
        target_key = k if has_prefix else f"denoiser.{k}"
        new_denoiser_dict[target_key] = v
    
    # [关键验证] 在加载前，检查主干权重是否存在于字典中
    critical_key = "denoiser.blocks.0.attn.qkv.weight"
    if critical_key not in new_denoiser_dict:
         # 尝试不带前缀找一下，防止判断失误
         if "blocks.0.attn.qkv.weight" not in denoiser_state:
            raise RuntimeError("CRITICAL: The provided Denoiser checkpoint does not contain backbone weights!")
    
    # 执行加载 (不接收返回值)
    model.load_state_dict(new_denoiser_dict, strict=False)
    print("  -> Backbone loaded successfully (Strict=False).")


def main():
    # 1. 配置与环境
    cfg = parse_args(phase="demo")
    cfg.TRAIN.BATCH_SIZE = 1 # 推理时 Batch Size 设为 1
    
    if cfg.ACCELERATOR == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in cfg.DEVICE)
        device = torch.device(f"cuda:{cfg.DEVICE[0]}")
    else:
        device = torch.device("cpu")

    # 2. 初始化
    print("Initializing datamodule...")
    # 获取数据集是为了拿到 mean/std 等统计信息
    dataset = get_datasets(cfg, phase="test")[0]
    
    print("Initializing model structure (Random Weights)...")
    model = get_model(cfg, dataset)

    # 3. [关键步骤] 双重权重加载
    
    # (A) 先恢复大脑 (Base Weights)
    load_pretrained_weights(model, path_config["pretrained_vae"], path_config["pretrained_denoiser"])
    
    # (B) 再注入物理灵魂 (Fine-tuned Weights)
    finetuned_path = path_config["finetuned_checkpoint"]
    print(f"\n--- [Adapter Loader] Loading Fine-tuned Weights from {finetuned_path} ---")
    if not os.path.exists(finetuned_path):
        raise FileNotFoundError(f"Checkpoint not found: {finetuned_path}")
        
    ft_state = torch.load(finetuned_path, map_location="cpu")["state_dict"]
    
    # [手动验证] 检查微调权重里是否包含我们的新模块
    ft_keys = set(ft_state.keys())
    
    # 检查 SCPAEncoder
    has_physics_encoder = any("physics_encoder" in k for k in ft_keys)
    # 检查 Adapter
    has_adapter = any("phys_adapter" in k for k in ft_keys)
    
    if has_physics_encoder and has_adapter:
        print(">>> Diagnostic: Fine-tuned checkpoint looks HEALTHY. Contains Physics modules.")
    else:
        print("!!! WARNING: Fine-tuned checkpoint seems to be MISSING physics modules. Output might be garbage.")
        print(f"    Has Encoder: {has_physics_encoder}, Has Adapter: {has_adapter}")

    # 加载微调权重 (覆盖 Base 权重中重叠的部分，并填入新模块)
    model.load_state_dict(ft_state, strict=False)
    print(">>> Fine-tuned weights loaded.")

    model.to(device)
    model.eval()

    # 4. 准备 I/O
    content_folder = cfg.DEMO.content_motion_dir
    physics_json_file = cfg.DEMO.PHYSICS_JSON 
    
    # 确保输出目录存在
    if physics_json_file:
         config_name = Path(physics_json_file).stem
    else:
         config_name = "default"
         
    output_dir = Path(os.path.join(cfg.TEST.FOLDER, "demo_outputs", cfg.NAME, config_name))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving outputs to: {output_dir}")

    # scenes_json_path = cfg.DATASET.PHYSIMOS100STYLE.SCENE_MAPPING_PATH

    # 5. 加载条件
    print(f"Loading physics from: {physics_json_file}")
    # 新代码: 直接调用新的解析函数
    phys_params, scene_cat = parse_inference_json(physics_json_file)
    
    # 扩充维度到 [1, D]
    phys_params = phys_params.unsqueeze(0).to(device) 
    scene_cat = scene_cat.unsqueeze(0).to(device)
    
    # 6. 循环推理
    print(f"Processing motions from: {content_folder}")
    if not os.path.exists(content_folder):
        raise FileNotFoundError(f"Folder not found: {content_folder}")

    count = 0
    mean = torch.from_numpy(dataset.mean).float().to(device)
    std = torch.from_numpy(dataset.std).float().to(device)
    for content_filename in os.listdir(content_folder):
        if not content_filename.endswith('.npy'): continue
            
        content_path = os.path.join(content_folder, content_filename)
        print(f"Processing: {content_filename}")

        motion_before_data = np.load(content_path)
        motion_before_tensor = torch.from_numpy(motion_before_data).float().to(device)
        
        # [核心修复] ！！！手动归一化！！！
        # (Raw - Mean) / Std
        motion_before_norm = (motion_before_tensor - mean) / std
        
        # 增加 Batch 维度 [Seq, D] -> [1, Seq, D]
        motion_before_norm = motion_before_norm.unsqueeze(0)
        
        lengths = [motion_before_norm.shape[1]]

        batch = {
            "motion_before": motion_before_norm, # <--- 传入归一化后的数据
            "length": lengths,
            "phys_params": phys_params, 
            "scene_cat": scene_cat,
            "motion": motion_before_norm 
        }
        
        with torch.no_grad():
            # model(batch) 调用 MLD.forward()
            # 返回的是 joints (由 VAE decode 后的结果)
            joints, attn_weights = model(batch, return_attn=True)
            
        motion_output = joints[0].detach().cpu().numpy()
        content_name = Path(content_filename).stem
        npypath = output_dir / f"{content_name}.npy"
        mp4path = npypath.with_suffix('.mp4')
        attpath = npypath.with_suffix('.png') # 图片路径
        
        np.save(npypath, motion_output)
        if isinstance(attn_weights, list):
            final_attn = attn_weights[-1]
        else:
            final_attn = attn_weights
        # 获取当前测试的 Scene Name (用于图表标题)
        # 我们从 physics_json_file 读取的
        with open(physics_json_file, 'r') as f:
            current_scene_name = json.load(f)["scene_category"]

        plot_attention(final_attn, current_scene_name, str(attpath))

        count += 1
        
        try:
            # 尝试渲染视频
            visual_pos(str(npypath), str(mp4path))
            print(f"  -> Video saved: {mp4path}")
        except Exception as e:
            print(f"  -> Video render failed (Skipping): {e}")

    print(f"\nDone. Processed {count} files.")

if __name__ == "__main__":
    main()