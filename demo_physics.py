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

import shutil   # <--- 【新增】用于复制文件的工具库

from visual_side_view import render_side_view # <--- 新增这行
from visual_top_view import render_top_view
from visual_front_view import render_front_view

from omegaconf import OmegaConf,ListConfig # 确保引入了这个

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
    # "finetuned_checkpoint": "/root/autodl-tmp/MyRepository/MCM-LDM/experiments/mld/PhysiMoS_Finetune_v2_1125_3scenes/checkpoints/epoch=499.ckpt" ,
    "finetuned_checkpoint": "/root/autodl-tmp/MyRepository/MCM-LDM/experiments/mld/debug--PhysiMoS_Finetune_v2_1125_3scenes/checkpoints/epoch=1149.ckpt" ,
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
    labels = ["WindX", "WindY", "WindStrength", "CeilingHeight", "Gap Width", "GapOffset"]
    
    # 3. 绘图
    plt.figure(figsize=(8, 6))
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
    
# # [替换] 原来的 load_physics_from_json 替换为这个：
# def parse_inference_json(json_path):
#     """ 
#     直接从测试 JSON 解析推理条件 
#     Input JSON Example: { "scene_category": "Right", "physical_parameters": [1.0, 0.0, 0.0, 1.0] }
#     """
#     with open(json_path, 'r') as f:
#         data = json.load(f)
        
#     # 1. 解析场景 One-Hot,这个是多个场景的情况，把前后左右风认为是不同的场景
#     # scene_name = data["scene_category"]
#     # if scene_name not in SCENE_CATEGORIES:
#     #     raise ValueError(f"Scene '{scene_name}' not found in SCENE_CATEGORIES list!")
    
#     # scene_idx = SCENE_CATEGORIES.index(scene_name)
#     # scene_cat = torch.zeros(len(SCENE_CATEGORIES))
#     # scene_cat[scene_idx] = 1.0
        
#     # [修改] 不再检查 json 里的具体方向，或者强制忽略它
#     # 无论 json 里写的是 "Right" 还是 "Left"，我们都生成 "Windy" 的 One-Hot
#     scene_cat = torch.zeros(1)
#     scene_cat[0] = 1.0
    
#     # [核心修复] 增加坐标系转换逻辑
#     raw_params = data["physical_parameters"] # 假设是 [x, y, z, mag] (UE5 Z-up)
    
#     raw_x = raw_params[0]
#     raw_y = raw_params[1] # UE5 Right/Left axis
#     # raw_z = raw_params[2] # UE5 Up/Down axis
#     mag   = raw_params[2]

#     # # 对应 Dataset 里的逻辑: wind_vec = np.array([raw_x, raw_z, -raw_y])
#     # # HumanML3D (Y-up)
#     # new_x = raw_x
#     # new_y = raw_z        # 原来的 Z 变成了 Y (垂直)
#     # new_z = -1.0 * raw_y # 原来的 Y 变成了 -Z (深度/侧向)

#     # 重新组装
#     phys_params = torch.tensor([raw_x, raw_y, mag]).float()
    
    
#     return phys_params, scene_cat

def parse_inference_json(json_path, max_wind=330000.0, max_height=220.0):
    """
    [纯物理驱动版]
    直接从 JSON 解析 physical_parameters 列表，并归一化。
    """
    """
    [推理专用] 解析用户输入
    Wind: 输入 0-100 (0=无风, 100=最大风) -> 映射为 0.0-1.0
    Ceiling: 输入 80-220 (80=低, 220=高) -> 映射为 1.0-0.0
    Gap: 输入 40-130 (40=窄, 130=宽) -> 映射为 1.0-0.0
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    params = data.get("physical_parameters", {})
    
    # 初始化 4 维向量
    phys_vec = torch.zeros(6)
    
    # === 1. 风力处理 (用户输入 0-100) ===
    if "wind_force" in params:
        wf = params["wind_force"]
        # 假设用户在 json 里填写的 x,y 是 0-100 的数值
        user_x = wf.get('x', 0.0)
        user_y = wf.get('y', 0.0)
        
        user_mag = np.sqrt(user_x**2 + user_y**2)
        
        if user_mag < 1e-4:
            # 输入 0 -> 模型接收 0 -> 对应 Gap 场景学到的"正常走"
            phys_vec[0:3] = 0.0
        else:
            # 计算方向
            dir_x = user_x / user_mag
            dir_y = user_y / user_mag
            
            # 映射: 0-100 -> 0.0-1.0
            # 输入 50 -> 0.5 (触发挡风)
            # 输入 100 -> 1.0 (最大挡风)
            norm_mag = min(user_mag, 100.0) / 100.0
            
            phys_vec[0] = dir_x * norm_mag
            phys_vec[1] = dir_y * norm_mag
            phys_vec[2] = norm_mag
            
    # === 2. 天花板处理 (反向归一化) ===
    # 只要 json 里有这个字段就处理，不再用 elif
    if "ceiling_height" in params:
        ch = params["ceiling_height"]
        # 训练时的逻辑: 220->0, 80->0.64
        # 这里保持一致
        MAX_CEIL = 220.0
        val = max(0.0, 1.0 - (ch / MAX_CEIL))
        if val < 0.01: val = 0.0
        phys_vec[3] = val

    # === 3. 缝隙处理 (反向归一化) ===
    if "gap_width" in params:
        gw = params["gap_width"]
        # 训练时的逻辑: 130->0 (宽), 40->0.7 (窄)
        MAX_GAP_W = 130.0
        val = max(0.0, 1.0 - (gw / MAX_GAP_W))
        if val < 0.01: val = 0.0
        phys_vec[4] = val
        
        # Offset 保持除以 50 (根据之前的逻辑)
        if "gap_offset" in params:
            phys_vec[5] = params["gap_offset"] / 50.0
    # 填充风力
    # if "wind_force" in params:
    #     wf = params["wind_force"]
    #     wind_vec_xy = np.array([wf.get('x', 0), wf.get('y', 0)])
    #     mag = np.linalg.norm(wind_vec_xy)# 计算风力模长
    #     phys_vec[0] = wf.get('x', 0) / max_wind
    #     phys_vec[1] = wf.get('y', 0) / max_wind
    #     phys_vec[2] = mag / max_wind
        
    # # 填充天花板
    # if "ceiling_height" in params:
    #     ch = params["ceiling_height"]
    #     phys_vec[3] = max(0, 1.0 - (ch / max_height))
    
    # if "gap_width" in params:
    #     gw = params["gap_width"]
    #     phys_vec[4] = gw / 120.0
    # if "gap_offset" in params:
    #     go = params["gap_offset"]
    #     phys_vec[5] = go / 20.0

    # 生成一个虚拟的 scene_cat
    scene_cat = torch.ones(1)
    
    return phys_vec.float(), scene_cat.float()

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


import datetime # <--- 【新增】引入时间模块
def main():
    # 1. 配置与环境
    cfg = parse_args(phase="demo")
    
    # === [暴力修复] 强行读取你的 YAML 补全 DEMO 部分 ===
    # 既然 parse_args 没读进去，我们就自己动手
    print(">>> [DEBUG] Manually reloading config to fix missing DEMO keys...")
    
    # 手动加载你的配置文件
    my_cfg = OmegaConf.load("configs/config_physimos_probe.yaml")
    
    # 只要 DEMO 里的内容，拼接到 cfg 里
    if "DEMO" in my_cfg:
        cfg.DEMO = my_cfg.DEMO
        print(f">>> [DEBUG] Force loaded DEMO config: {cfg.DEMO}")
    else:
        print(">>> [ERROR] Still cannot find DEMO in the yaml file!")
    # =======================================================

     # 2. [新增] 补全 TRAJECTORY (为了读取最新的 TIMELINE)
    if "TRAJECTORY" in my_cfg:
        cfg.TRAJECTORY = my_cfg.TRAJECTORY
        print(f">>> [DEBUG] Force loaded TRAJECTORY config.")
    
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
    # physics_json_file = cfg.DEMO.PHYSICS_JSON 
        # 直接写死路径，绝对不会错
    physics_json_file = "/root/autodl-tmp/MyRepository/MCM-LDM/heavy.json" 
    
    if not os.path.exists(physics_json_file):
        print("完了，文件真的不存在！路径写错了！")
    
    # 确保输出目录存在
    if physics_json_file:
         config_name = Path(physics_json_file).stem
    else:
         config_name = "default"
         
    # output_dir = Path(os.path.join(cfg.TEST.FOLDER, "demo_outputs", cfg.NAME, config_name))
    # === [修改开始] ===
    # 1. 获取当前时间，格式为 年-月-日-时-分 (例如: 2023-10-27-14-30)
    current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    
    # 2. 拼接文件夹名称： 原有配置名 + 时间戳
    # 这样每次运行都不会覆盖旧文件，而是生成新文件夹
    folder_name = f"{config_name}_{current_time}"
    
    # 3. 组合完整路径
    output_dir = Path(os.path.join(cfg.TEST.FOLDER, "demo_outputs", cfg.NAME, folder_name))
    # === [修改结束] ===



    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving outputs to: {output_dir}")

    # scenes_json_path = cfg.DATASET.PHYSIMOS100STYLE.SCENE_MAPPING_PATH

    # # === [新增代码开始] ===
    # # 将 physics_json_file 复制到 output_dir 中
    # if os.path.exists(physics_json_file):
    #     try:
    #         # shutil.copy(源文件路径, 目标文件夹路径)
    #         shutil.copy(physics_json_file, output_dir)
    #         print(f">>> [Backup] Configuration file copied to: {output_dir}")
    #     except Exception as e:
    #         print(f">>> [Warning] Failed to backup config file: {e}")
    # else:
    #     print(f">>> [Warning] Config file not found, cannot backup: {physics_json_file}")
    # # === [新增代码结束] ===
    
    # === [修改后的备份代码] ===
    if os.path.exists(physics_json_file):
        try:
            # 1. 在这里自定义你想要的新名字
            new_filename = "config_backup.json" 
            
            # 2. 拼接完整的【目标路径 + 目标文件名】
            # output_dir 是 Path 对象，可以直接用 / 拼接
            target_path = output_dir / new_filename
            
            # 3. 复制文件 (源路径 -> 带新名字的目标路径)
            shutil.copy(physics_json_file, target_path)
            
            print(f">>> [Backup] Configuration file copied as: {new_filename}")
        except Exception as e:
            print(f">>> [Warning] Failed to backup config file: {e}")
    else:
        print(f">>> [Warning] Config file not found, cannot backup: {physics_json_file}")
    # ========================

    


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
            result = model(batch, return_attn=True)
            if isinstance(result, tuple):
                joints, attn_weights = result
            else:
                joints = result
                attn_weights = None
            
        motion_output = joints[0].detach().cpu().numpy()
        content_name = Path(content_filename).stem
        npypath = output_dir / f"{content_name}.npy"
        mp4path = npypath.with_suffix('.mp4')
        attpath = npypath.with_suffix('.png') # 图片路径
        
        np.save(npypath, motion_output)
        
        # =================[新增：配套生成场景 JSON 文件] =================
        import json
        
        scene_data = {}
        traj_cfg = cfg.get("TRAJECTORY", {}).get("GUIDANCE", {})
        
        # 1. 获取天花板空间参数
        if traj_cfg.get("CEILING_MODE", False):
            c_spatial = traj_cfg.get("CEILING_SPATIAL", None)
            if c_spatial is not None:
                # 转换 ListConfig 为标准 Python 列表
                if isinstance(c_spatial, ListConfig):
                    c_spatial = OmegaConf.to_container(c_spatial, resolve=True)
                
                # YAML 里格式是 [[2.0, 5.0, 0.7]]，为了对齐你的 Blender (单列表)，提取第一项
                if isinstance(c_spatial, list) and len(c_spatial) > 0:
                    scene_data["ceiling_spatial"] = c_spatial[0]
        
        # 2. 获取狭窄缝隙空间参数 (如果你后续需要渲染缝隙的话)
        if traj_cfg.get("GAP_MODE", False):
            g_spatial = traj_cfg.get("GAP_SPATIAL", None)
            if g_spatial is not None:
                if isinstance(g_spatial, ListConfig):
                    g_spatial = OmegaConf.to_container(g_spatial, resolve=True)
                if isinstance(g_spatial, list) and len(g_spatial) > 0:
                    scene_data["gap_spatial"] = g_spatial[0]
        
         # ==================== [此处为新增的第 3 步：侧身空间参数] ====================
        # 3. 获取侧身引导空间参数
        if traj_cfg.get("SIDE_STEP_MODE", False):
            s_spatial = traj_cfg.get("SIDE_STEP_SPATIAL", None)
            if s_spatial is not None:
                if isinstance(s_spatial, ListConfig):
                    s_spatial = OmegaConf.to_container(s_spatial, resolve=True)
                if isinstance(s_spatial, list) and len(s_spatial) > 0:
                    # 写入字典，Blender那边可以读这个 "side_step_spatial" 字段
                    scene_data["side_step_spatial"] = s_spatial[0]
                    
        # 3. 只有当确实存在空间约束时，才保存 JSON
        if scene_data:
            # 命名为：原动作名_scene.json (完美匹配你 Blender 脚本的最高优先级)
            json_path = output_dir / f"{content_name}_scene.json"
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(scene_data, f, indent=4)
                print(f"  -> Scene JSON saved: {json_path}")
            except Exception as e:
                print(f"  -> Failed to save Scene JSON: {e}")
        # =================================================================
        
        
        if isinstance(attn_weights, list):
            final_attn = attn_weights[-1]
        else:
            final_attn = attn_weights
        # 获取当前测试的 Scene Name (用于图表标题)
        # 我们从 physics_json_file 读取的
        # with open(physics_json_file, 'r') as f:
        #     current_scene_name = json.load(f)["scene_category"]
        current_scene_name = "Varsapura"

        # 如果 attn_weights 是 None，就跳过画图
        draw_attn_map = cfg.get("DRAW_ATTN_MAP", False)
        if draw_attn_map:
            if attn_weights is not None:
                plot_attention(final_attn, current_scene_name, str(attpath))
            else:
                print("Note: MLP encoder does not produce attention maps.")

        count += 1
        
        try:
            # 尝试渲染视频
            visual_pos(str(npypath), str(mp4path))
            print(f"  -> Video saved: {mp4path}")
        except Exception as e:
            print(f"  -> Video render failed (Skipping): {e}")

        # ================= [环境可视化渲染] =================
        vis_env = cfg.get("VISUALIZATION", {}).get("RENDER_ENVIRONMENT", False)
        if vis_env:
            traj_cfg = cfg.get("TRAJECTORY", {}).get("GUIDANCE", {})
            spatial_cfg = traj_cfg.get("CEILING_SPATIAL", None)
            
            
            if spatial_cfg is not None and isinstance(spatial_cfg, ListConfig):
                spatial_cfg = OmegaConf.to_container(spatial_cfg, resolve=True)
            
            if spatial_cfg is not None and len(spatial_cfg) > 0:
                try:
                    env_mp4path = output_dir / f"{content_name}_env.mp4"
                    print(f"  -> Rendering Environment video (Side View) for SPATIAL constraint...")
                    
                    # [核心修改] 传入 view_angles=(15, -45) 形成绝佳的侧前方立体视角
                    # (elev=15表示略微抬高相机，azim=-45表示侧前方45度)
                    # 如果你想要纯正侧面，可以改成 (10, 0)
                    # visual_pos(str(npypath), str(env_mp4path), ceiling_spatial=spatial_cfg, view_angles=(15, -45))
                    # 【修改点】：传 (120, -45) 获得侧前方的完美视角！
                    # visual_pos(str(npypath), str(env_mp4path), ceiling_spatial=spatial_cfg, view_angles=(180, -90))
                    # 【核心魔法】：保持漂亮的 105 度正面微俯视，同时把小人和天花板向左转90度看侧面！
                    visual_pos(str(npypath), str(env_mp4path), ceiling_spatial=spatial_cfg, view_angles=(105, -90), is_side_view=True)

                    print(f"  -> Environment video saved: {env_mp4path}")
                except Exception as e:
                    print(f"  -> Environment render failed: {e}")
            
        # === [新增] 侧视图 + 天花板可视化 ===
        # 1. 读取开关
        vis_cfg = cfg.get("VISUALIZATION", {})
        do_side_view = vis_cfg.get("SIDE_VIEW_EXPORT", False)
        
        if do_side_view:
            # 2. 读取天花板高度
            # 注意：要从 TRAJECTORY 配置里读，而不是从 json 文件读，因为引导用的是配置值
            traj_cfg = cfg.get("TRAJECTORY", {}).get("GUIDANCE", {})
            
            # # 只有当开启了天花板引导，且有高度值时，才画线
            # ceil_h = None
            # if traj_cfg.get("CEILING_MODE", False):
            #     ceil_h = traj_cfg.get("CEILING_HEIGHT", None)
            
            # # 3. 定义保存路径 (例如 jump_side.mp4)
            # side_mp4_path = output_dir / f"{content_name}_side.mp4"
            
            # # 4. 调用独立脚本渲染
            # # motion_output 是 numpy array
            # render_side_view(
            #     motion_data=motion_output, 
            #     save_path=str(side_mp4_path), 
            #     ceiling_height=ceil_h
            # )
            # === [修改后] ===
            ceil_info = None
            if traj_cfg.get("CEILING_MODE", False):
                # 优先读取 Timeline
                timeline = traj_cfg.get("CEILING_TIMELINE", None)

                # [核心修复] 将 OmegaConf 对象转为 Python 原生 List
                # 这样 render_side_view 里的 isinstance(x, list) 才会是 True
                if timeline is not None:
                    # to_container 会把 ListConfig 变成 [ [0,80,0.8], ... ]
                    timeline = OmegaConf.to_container(timeline, resolve=True)

                # 如果 timeline 存在且不为空，就用它
                if timeline and len(timeline) > 0:
                    ceil_info = timeline
                else:
                    # 否则读取固定高度
                    ceil_info = traj_cfg.get("CEILING_HEIGHT", None)
            
            # 3. 定义保存路径
            side_mp4_path = output_dir / f"{content_name}_side.mp4"
            
            # 4. 调用
            render_side_view(
                motion_data=motion_output, 
                save_path=str(side_mp4_path), 
                ceiling_info=ceil_info  # <--- 注意参数名变了 (或者你保持 ceiling_height 也可以，python不强类型)
            )
        # ========================================        
        
        do_top_view = vis_cfg.get("TOP_VIEW_EXPORT", False)
        # === [新增] 俯视图可视化 ===
        # if do_top_view: # 复用开关
        #     traj_cfg = cfg.get("TRAJECTORY", {}).get("GUIDANCE", {})
        #     gap_w = None
        #     if traj_cfg.get("GAP_MODE", False):
        #         gap_w = traj_cfg.get("GAP_WIDTH", None)
            
        #     top_mp4_path = output_dir / f"{content_name}_top.mp4"
            
        #     render_top_view(
        #         motion_data=motion_output, # 注意检查是否需要转置
        #         save_path=str(top_mp4_path),
        #         gap_width=gap_w
        #     )
        if do_top_view: 
            traj_cfg = cfg.get("TRAJECTORY", {}).get("GUIDANCE", {})
            
            # 读取 Gap Info (可以是 Timeline 或 float)
            gap_info = None
            if traj_cfg.get("GAP_MODE", False):
                timeline = traj_cfg.get("GAP_TIMELINE", None)
                if timeline is not None:
                        # 确保转换类型 (假设你已经在 main 里删除了多余的 import OmegaConf)
                        # 并且在文件头 import 了
                    
                    timeline = OmegaConf.to_container(timeline, resolve=True)

                if timeline and len(timeline) > 0:
                    gap_info = timeline
                else:
                    gap_info = traj_cfg.get("GAP_WIDTH", None)
            
            top_mp4_path = output_dir / f"{content_name}_top.mp4"
            
            # 调用
            render_top_view(
                motion_data=motion_output, 
                save_path=str(top_mp4_path),
                gap_info=gap_info # 参数名改为 gap_info 以支持列表
            )
            
        # === [新增] 正视图 (Front View) 可视化 ===
        do_front_view = vis_cfg.get("FRONT_VIEW_EXPORT", False)
        if do_front_view: 
            # 复用 gap_w 变量
            front_mp4_path = output_dir / f"{content_name}_front.mp4"
            
            render_front_view(
                motion_data=motion_output,
                save_path=str(front_mp4_path),
                gap_width=gap_w
            )

    print(f"\nDone. Processed {count} files.")

if __name__ == "__main__":
    main()