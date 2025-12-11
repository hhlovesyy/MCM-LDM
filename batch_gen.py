import argparse
import json
import os
import sys
import datetime
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# 引入必要的项目模块
from mld.config import parse_args
from mld.data.get_data import get_datasets
from mld.models.get_model import get_model
from mld.utils.logger import create_logger
from mld.models.modeltype import mld as mld_module  # <--- 核心：导入定义 scalar 的模块
from visual import visual_pos 

# 场景描述字典 (与你提供的一致)
SCENE_DESCRIPTIONS = {
    "Dumuqiao": "Walking on a narrow bridge.",
    "DiAiTongDao": "Crouching while walking.",
    "ShuiKengDiMian": "Walking on muddy ground.",
    "BoLiFangJian": "Walking in a glass room.",
    "T_Stage": "Fashion model walking.",
    "CroudedPlace": "Walking through a crowd.",
    "DiAiTianhuaban": "Walking under a low ceiling.",
    "Bar": "Drunk walking.",
    "WalkInSnowOrSand": "Walking in deep snow.",
    "Dark": "Walking in the dark.",
    "LeanLeft": "Leaning left.",
    "WetFloor": "Slippery floor.",
    "BaoFengYu": "Walking in strong wind.",
    "IcyRoad": "Walking on ice."
}

def main():
    # 1. 解析参数
    # 我们手动构建 parser，因为 mld 的 parse_args 会读取 sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_json', type=str, required=True, help="Path to the task configuration JSON")
    parser.add_argument('--cfg', type=str, required=True, help="Base config file")
    parser.add_argument('--cfg_assets', type=str, required=True, help="Assets config file")
    args = parser.parse_args()

    # 读取任务配置
    with open(args.task_json, 'r') as f:
        task_config = json.load(f)

    # 2. 伪造 sys.argv 以骗过 mld.config.parse_args
    # 这样我们就可以复用原本的配置加载逻辑
    sys.argv = [sys.argv[0], "--cfg", args.cfg, "--cfg_assets", args.cfg_assets, "--nodebug"]
    
    cfg = parse_args(phase="demo")
    
    # 强制覆盖 Checkpoint 路径
    cfg.TEST.CHECKPOINTS = task_config['checkpoint']
    
    # 设置输出目录
    # output_dir = Path(task_config['output_dir'])
    # Path(task_config['output_dir']) / npyout
    output_parent = Path(task_config['output_dir'])
    output_parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path(task_config['output_dir']) / "npyout"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置 Logger
    cfg.FOLDER = str(output_dir)
    cfg.Name = "BatchGen"
    logger = create_logger(cfg, phase="demo")
    
    # 3. 加载模型 (只加载一次！)
    logger.info(f"Loading Checkpoint: {cfg.TEST.CHECKPOINTS}")
    
    if cfg.ACCELERATOR == "gpu":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # 获取数据集配置 (为了拿 nfeats 等参数)
    dataset = get_datasets(cfg, logger=logger, phase="test")[0]
    
    # 初始化模型
    model = get_model(cfg, dataset)
    state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully. Starting batch generation...")

    # 4. 开始排列组合循环
    # 顺序：Scalar -> Content -> Style -> Scene
    
    scalars = task_config.get('scalars', [3.0])
    contents = task_config.get('contents', [])
    styles = task_config.get('styles', [])
    scenes = task_config.get('scenes', [])
    render_mp4 = task_config.get('render_mp4', False)
    
    # 预加载 Content 和 Style 数据，避免重复 IO
    content_cache = {}
    style_cache = {}
    
    logger.info("Caching Motion Data...")
    for c_name in contents:
        c_path = os.path.join(task_config['content_dir'], c_name)
        if os.path.exists(c_path):
            data = np.load(c_path)
            content_cache[c_name] = torch.tensor(data[np.newaxis, ...]).to(device)
            
    for s_name in styles:
        s_path = os.path.join(task_config['style_dir'], s_name)
        if os.path.exists(s_path):
            data = np.load(s_path)
            style_cache[s_name] = torch.tensor(data[np.newaxis, ...]).to(device)

    # 进度条
    total_steps = len(scalars) * len(contents) * len(styles) * len(scenes)
    pbar = tqdm(total=total_steps, desc="Generating")

    for scalar_val in scalars:
        # =================================================
        # 【核心魔法】内存热修补 (Monkey Patching)
        # 直接修改 mld 模块内存中的变量，无需重启，无需改文件
        # =================================================
        try:
            mld_module.scene_scalar = float(scalar_val)
            # 为了保险，也尝试设置 model 实例的属性 (如果代码里是 self.scene_scalar)
            if hasattr(model, 'scene_scalar'):
                model.scene_scalar = float(scalar_val)
            # 甚至可以注入到 model 内部的 diffusion model 里
            if hasattr(model, 'model') and hasattr(model.model, 'scene_scalar'):
                model.model.scene_scalar = float(scalar_val)
                
            # print(f"Set Scene Scalar to: {scalar_val}")
        except Exception as e:
            logger.warning(f"Failed to inject scalar: {e}")

        for c_name in contents:
            if c_name not in content_cache: continue
            content_tensor = content_cache[c_name]
            length = content_tensor.shape[1]
            lengths = [int(length)]
            
            # 文件名简写 (C_Walk)
            c_short = c_name.split('.')[0]

            for s_name in styles:
                if s_name not in style_cache: continue
                style_tensor = style_cache[s_name]
                s_short = s_name.split('.')[0]

                for scene_key in scenes:
                    # 获取 Prompt
                    # 处理两种情况：Key不存在时用Key本身，或者空字符串
                    prompt = SCENE_DESCRIPTIONS.get(scene_key, "")
                    if not prompt and scene_key == "Custom":
                        prompt = "Walking carefully." # 兜底

                    # 构造 Batch
                    # 注意：为了让你的 scalar hack 生效，batch 里我也传一份，双重保险
                    batch = {
                        "length": lengths,
                        "content_motion": content_tensor,
                        "style_motion": style_tensor,
                        "scene_text": [prompt] * len(lengths),
                        "tag_scale": cfg.DEMO.scale, # 这里的 scale 是 style 的 scale (默认2.5)
                        "scene_scalar": float(scalar_val), # 放入 batch 备用
                        "has_image": torch.tensor([False]).to(device), # 默认不使用图像
                        # 这是一个占位符，防止报错
                        "scene_id": torch.tensor([0]).to(device) 
                    }
                    
                    # 推理
                    with torch.no_grad():
                        # 这里调用模型
                        # 如果你的 forward 里用了全局变量 scene_scalar，上面修补的就会生效
                        joints = model(batch)
                        
                        # 拿到结果 (Batch size 1)
                        motion = joints[0].detach().cpu().numpy()
                        
                        # =========================================
                        # 生成带有元数据的文件名
                        # 格式: C_{content}_S_{style}_Sc_{scene}_v{scalar}.npy
                        # =========================================
                        out_filename = f"C_{c_short}_S_{s_short}_Sc_{scene_key}_v{scalar_val}.npy"
                        save_path = output_dir / out_filename
                        
                        np.save(save_path, motion)
                        if render_mp4:
                            mp4path = save_path.with_suffix('.mp4')
                            visual_pos(save_path, mp4path)

                        
                        # (可选) 生成对应的 txt 描述文件，方便后续查看
                        # txt_path = output_dir / f"C_{c_short}_S_{s_short}_Sc_{scene_key}_v{scalar_val}.txt"
                        # with open(txt_path, 'w') as f:
                        #    f.write(f"Prompt: {prompt}\nScalar: {scalar_val}")

                    pbar.update(1)

    pbar.close()
    logger.info(f"Batch generation completed! Saved {total_steps} files to {output_dir}")

if __name__ == "__main__":
    main()