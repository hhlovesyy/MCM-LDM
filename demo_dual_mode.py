# demo.py
import logging
import os
from pathlib import Path
import datetime
import numpy as np
import torch
from mld.config import parse_args
from mld.data.get_data import get_datasets
from mld.models.get_model import get_model
from mld.utils.logger import create_logger
from visual import visual_pos  # 假设 visual_pos 是您的可视化函数

def main():
    # 1. --- 配置加载与设置 ---
    cfg = parse_args(phase="demo")
    
    # [核心] 根据 DEMO 配置设置输出文件夹
    time_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if cfg.DEMO.STYLE_TEXT:
        # 如果是文本模式，在输出文件夹中体现
        mode_name = f"text_guidance_{time_str}"
    else:
        # 否则是动作模式
        mode_name = f"motion_guidance_{time_str}"
        
    output_dir = Path(os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME), mode_name))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = create_logger(cfg, phase="demo")

    # 2. --- 设备与模型加载 ---
    if cfg.ACCELERATOR == "gpu":
        device = torch.device(f"cuda:{cfg.DEVICE[0]}")
    else:
        device = torch.device("cpu")
    
    # 加载数据集仅为获取元数据 (nfeats)
    datamodule = get_datasets(cfg, logger=logger, phase="test")[0]
    datamodule.setup() # 确保 datamodule.norms 等已加载

    # 创建模型
    model = get_model(cfg, datamodule)
    
    # 加载训练好的 checkpoint
    logger.info(f"Loading checkpoints from {cfg.DEMO.CHECKPOINT}")
    state_dict = torch.load(cfg.DEMO.CHECKPOINT, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    
    model.to(device)
    model.eval()

    # 3. --- 模式判断与执行 ---
    scale = cfg.DEMO.SCALE

    # 模式一: 文本引导 (Text-Guided Style Transfer)
    if cfg.DEMO.STYLE_TEXT:
        logger.info("--- Running in Text-Guided Mode ---")
        logger.info(f"Style Text: '{cfg.DEMO.STYLE_TEXT}'")
        
        content_path = Path(cfg.DEMO.CONTENT_MOTION_DIR)
        for content_file in content_path.glob("*.npy"):
            logger.info(f"Processing content motion: {content_file.name}")
            
            # 准备内容动作
            content_motion = torch.from_numpy(np.load(content_file)).unsqueeze(0).float().to(device)
            lengths = [content_motion.shape[1]]
            
            # 准备 batch
            batch = {
                "length": lengths,
                "content_motion": content_motion,
                "style_text": [cfg.DEMO.STYLE_TEXT], # [核心] 将文本放入列表
                "tag_scale": scale
            }

            # 模型推理
            with torch.no_grad():
                joints = model(batch)

            # 保存结果
            motion_np = joints[0].detach().cpu().numpy()
            
            # 创建一个对文件名友好的文本描述
            text_desc = cfg.DEMO.STYLE_TEXT.lower().replace(" ", "_")[:50] # 取前50个字符
            
            out_filename = f"{content_file.stem}_styled_by_{text_desc}.npy"
            npypath = str(output_dir / out_filename)
            mp4path = npypath.replace('.npy', '.mp4')
            
            np.save(npypath, motion_np)
            logger.info(f"Saved generated motion to: {npypath}")
            
            # 可视化
            visual_pos(npypath, mp4path)
            logger.info(f"Saved visualization to: {mp4path}")

    # 模式二: 动作引导 (Motion-Guided Style Transfer)
    else:
        logger.info("--- Running in Motion-Guided Mode ---")
        style_path = Path(cfg.DEMO.STYLE_MOTION_DIR)
        content_path = Path(cfg.DEMO.CONTENT_MOTION_DIR)

        for content_file in content_path.glob("*.npy"):
            # 准备内容动作
            content_motion = torch.from_numpy(np.load(content_file)).unsqueeze(0).float().to(device)
            lengths = [content_motion.shape[1]]

            for style_file in style_path.glob("*.npy"):
                logger.info(f"Applying style '{style_file.name}' to content '{content_file.name}'")
                
                # 准备风格动作
                style_motion = torch.from_numpy(np.load(style_file)).unsqueeze(0).float().to(device)
                
                # 准备 batch
                batch = {
                    "length": lengths,
                    "content_motion": content_motion,
                    "style_motion": style_motion, # [核心] 传入 style_motion
                    "tag_scale": scale
                }
                
                # 模型推理
                with torch.no_grad():
                    joints = model(batch)
                    
                # 保存结果
                motion_np = joints[0].detach().cpu().numpy()
                out_filename = f"{content_file.stem}_styled_by_{style_file.stem}.npy"
                npypath = str(output_dir / out_filename)
                mp4path = npypath.replace('.npy', '.mp4')
                
                np.save(npypath, motion_np)
                logger.info(f"Saved generated motion to: {npypath}")
                
                # 可视化
                visual_pos(npypath, mp4path, view_mode='camera_follow')  # view_mode='camera_follow', fixed_camera
                logger.info(f"Saved visualization to: {mp4path}")

if __name__ == "__main__":
    main()