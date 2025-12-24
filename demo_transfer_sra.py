import logging
import os
import time
from builtins import ValueError
from multiprocessing.sharedctypes import Value
from pathlib import Path
import datetime
import pickle
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader
# from torchsummary import summary
from tqdm import tqdm

from mld.config import parse_args
# from mld.datasets.get_dataset import get_datasets
from mld.data.get_data import get_datasets
from mld.data.sampling import subsample, upsample
from mld.models.get_model import get_model
from mld.utils.logger import create_logger

from visual import visual_pos  
import itertools
from torch.nn.utils.rnn import pad_sequence
import json


SCENE_CONFIG = {
    # 1. 独木桥 (Dumuqiao)
    "Dumuqiao": "Walking on a narrow bridge.",
    # 2. 低矮通道 (DiAiTongDao)
    "DiAiTongDao": "Crouching while walking.",
    # 3. 水坑地面 (ShuiKengDiMian)
    "ShuiKengDiMian": "Walking on muddy ground.",
    # 4. 玻璃房间 (BoLiFangJian)
    "BoLiFangJian": "Walking in a glass room.",
    # 5. T台走秀 (T_Stage)
    "T_Stage":"Fashion model walking.",
    # 6. 拥挤场合 (CroudedPlace)
    "CroudedPlace": "Walking through a crowd.",
    # 7. 低矮天花板 (DiAiTianhuaban)
    "DiAiTianhuaban": "Walking under a low ceiling.",
    # 8. 酒吧/醉酒 (Bar)
    "Bar": "Drunk walking.",
    # 9. 雪地/沙地 (WalkInSnowOrSand)
    "WalkInSnowOrSand": "Walking in deep snow.",
    # 10. 摸黑 (Dark)
    "Dark": "Walking in the dark.",
    # 11. 左倾 (LeanLeft)
    "LeanLeft": "Leaning left.",
    # 12. 潮湿地面 (WetFloor)
    "WetFloor": "Slippery floor.",
    # 13. 暴风雨 (BaoFengYu)
    "BaoFengYu": "Walking in strong wind.",
    # 14. 冰面 (IcyRoad)
    "IcyRoad": "Walking on ice."
}
SCENE_NAMES = list(SCENE_CONFIG.keys())
SCENE_TEXTS = list(SCENE_CONFIG.values())



def main():
    start_time = time.perf_counter()
    # 1. 配置与初始化 (保持不变)
    cfg = parse_args(phase="demo")
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.Name = "demo--" + cfg.NAME
    logger = create_logger(cfg, phase="demo")

    style_path = cfg.DEMO.style_motion_dir
    content_path = cfg.DEMO.content_motion_dir
    scale = cfg.DEMO.scale

    eval_name = "style"
    eval_id = 0
    logger.info("cfg.DEMO.SAVE_PATH_FOR_EVAL: {}".format(cfg.DEMO.SAVE_PATH_FOR_EVAL))
    
    if cfg.DEMO.SAVE_PATH_FOR_EVAL is not None:
        save_path = cfg.DEMO.SAVE_PATH_FOR_EVAL
    else:
        save_path = Path(os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME)))
        save_path.mkdir(parents=True, exist_ok=True)
        # 保持原文件名的生成逻辑
        filename = eval_name+'-'+str(eval_id)+'_expname_'+str(cfg.NAME)+"_scale_"+str(cfg.DEMO.scale).replace('.','-') + '.pkl'
        save_path = os.path.join(save_path, filename)

    logger.info(f"save path: {save_path}")

    # CUDA 设置
    if cfg.ACCELERATOR == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in cfg.DEVICE)
        device = torch.device("cuda:0")

    # 模型加载
    dataset = get_datasets(cfg, logger=logger, phase="test")[0]
    model = get_model(cfg, dataset)
    logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))
    state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    logger.info("model {} loaded".format(cfg.model.model_type))
    model.sample_mean = cfg.TEST.MEAN
    model.fact = cfg.TEST.FACT
    model.to(device)
    model.eval()

    json_path = "/root/autodl-tmp/MyRepository/MCM-LDM/task_config.json"

    # 使用 with 语句打开文件（这样会自动关闭文件，更安全）
    with open(json_path, 'r', encoding='utf-8') as f:
        scene_data = json.load(f)

    # 2. 预加载数据 (Pre-loading)
    print("🚀 Pre-loading data...")
    
    # 加载 Contents
    all_contents = []
    # 使用 sorted 确保顺序固定
    for f in sorted(os.listdir(content_path)):
        if not f.endswith('.npy'): continue # 安全检查
        path = os.path.join(content_path, f)
        data = np.load(path)
        if len(data.shape) == 3: data = data[0]
        
        fname = f.split('.')[0]
        all_contents.append({
            "name": fname,
            "motion": torch.tensor(data).float(),
            "length": data.shape[0],
            "label": fname.split("-")[-1] # 提取 label_content
        })

    # 加载 Styles
    all_styles = []
    for f in sorted(os.listdir(style_path)):
        if not f.endswith('.npy'): continue
        path = os.path.join(style_path, f)
        data = np.load(path)
        if len(data.shape) == 3: data = data[0]
        
        fname = f.split('.')[0]
        all_styles.append({
            "name": fname,
            "motion": torch.tensor(data).float(),
            "length": data.shape[0],
            "label": fname.split("-")[-1] # 提取 label_style
        })

    # 生成任务列表 (Content x Style)
    all_tasks = list(itertools.product(all_contents, all_styles))
    total_tasks = len(all_tasks)
    print(f"🔥 Total Evaluation Tasks: {total_tasks}")

    # 3. 初始化结果容器
    save_all = {
        "joints": [],
        "id": [],
        "label_content": [],
        "label_style": []
    }

    # 4. Batch 推理循环
    BATCH_SIZE = 128 # 建议从 256 开始尝试，显存不够就降到 128 或 64
    print(f"🚀 Inference Batch Size: {BATCH_SIZE}")

    # 预构建 Dummy Tensors
    dummy_image = torch.zeros(1, 3, 224, 224, device=device)
    dummy_has_image = torch.tensor([False], device=device)
    
    # 你的原代码里写的是 fixed scene text = [""]
    # 并且不涉及 scene data (None)
    
    for i in tqdm(range(0, total_tasks, BATCH_SIZE), desc="Eval Batches"):
        # 获取当前 Batch 的任务
        batch_tasks = all_tasks[i : i + BATCH_SIZE]
        current_bs = len(batch_tasks)

        b_c_motions = []
        b_s_motions = []
        b_c_lens = []
        b_s_lens = []
        
        for item in batch_tasks:
            c_data, s_data = item
            b_c_motions.append(c_data["motion"])
            b_s_motions.append(s_data["motion"])
            b_c_lens.append(c_data["length"])
            b_s_lens.append(s_data["length"])

        # Padding
        c_padded = pad_sequence(b_c_motions, batch_first=True, padding_value=0.0).to(device)
        s_padded = pad_sequence(b_s_motions, batch_first=True, padding_value=0.0).to(device)

        # 构造 Batch Input
        batch = {
            "length": b_c_lens,
            "content_motion": c_padded,
            "style_length": b_s_lens,     # 支持 Mask
            "style_motion": s_padded,
            "tag_scale": scale,
            "scene_text": [""] * current_bs, # 原代码逻辑：空字符串
            "has_image": dummy_has_image,
            "scene_image": dummy_image.repeat(current_bs, 1, 1, 1)
        }

        # 推理
        with torch.no_grad():
            joints,_ = model(batch, scene_data)

        # 处理结果
        if isinstance(joints, torch.Tensor):
            joints = joints.detach().cpu().numpy()
            
        for idx, item in enumerate(batch_tasks):
            c_data, s_data = item
            
            motion_res = joints[idx]
            # ================= [修复点 Start] =================
            # 1. 强制转 Numpy (防止它是 Tensor)
            if isinstance(motion_res, torch.Tensor):
                motion_res = motion_res.detach().cpu().numpy() # (59, 22, 3)
                # print(motion_res.shape)
            # ================= [修复点 End] =================
            real_len = b_c_lens[idx]
            
            # ================= [关键修改区 Start] =================
            
            # 1. 切除 Padding (只保留真实帧数)
            # 自动判断维度: 通常是 [Length, J, 3]
            if motion_res.shape[0] >= real_len:
                motion_res = motion_res[:real_len]
            elif motion_res.shape[-1] >= real_len: # 防御性编程: 假如是 [J, 3, Length]
                motion_res = motion_res[..., :real_len]

            # 2. 压缩体积 (float32 -> float16)
            # 这步至关重要，能让文件减半且不影响评估
            # motion_res = motion_res.astype(np.float16)
            
            # ================= [关键修改区 End] =================

            save_all["joints"].append(motion_res)
            
            idid = "content" + c_data["name"] + "_" + "style" + s_data["name"] + "_scale_" + str(scale).replace('.', '-')
            save_all["id"].append(idid)
            save_all["label_content"].append(c_data["label"])
            save_all["label_style"].append(s_data["label"])

    # 5. 保存
    print(f"💾 Saving results to {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump(save_all, f)
        
    end_time = time.perf_counter()
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    
    print(f"✅ SRA Evaluation Done!")
    print(f"🚀 Time elapsed: {end_time - start_time:.4f} s")
    print(f"📦 File size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    main()