import logging
import os
import time
import pickle
import numpy as np
import torch
import random
import itertools
from pathlib import Path
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from mld.config import parse_args
from mld.data.get_data import get_datasets
from mld.models.get_model import get_model
from mld.utils.logger import create_logger
import json

# 场景定义保持不变
SCENE_LIST = sorted([
    "BaoFengYu", "Bar", "BoLiFangJian", "CroudedPlace", "Dark", 
    "DiAiTianhuaban", "DiAiTongDao", "Dumuqiao", "IcyRoad", "LeanLeft", 
    "ShuiKengDiMian", "T_Stage", "WalkInSnowOrSand", "WetFloor"
])

SCENE_DESCRIPTIONS = {
    "Dumuqiao": "Walking on a narrow bridge.",
    "DiAiTongDao": "Crouching while walking.",
    "ShuiKengDiMian": "Walking on muddy ground.",
    "BoLiFangJian": "Walking in a glass room.",
    "T_Stage":"Fashion model walking.",
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
    start_time = time.perf_counter()
    # 1. 配置与初始化
    cfg = parse_args(phase="demo")
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.Name = "demo--" + cfg.NAME
    logger = create_logger(cfg, phase="demo")

    style_path = cfg.DEMO.style_motion_dir
    content_path = cfg.DEMO.content_motion_dir
    scale = cfg.DEMO.scale
    json_path = "/root/autodl-tmp/MyRepository/MCM-LDM/task_config.json"

    # 使用 with 语句打开文件（这样会自动关闭文件，更安全）
    with open(json_path, 'r', encoding='utf-8') as f:
        scene_data = json.load(f)

    # 路径处理
    if cfg.DEMO.SAVE_PATH_FOR_EVAL is not None:
        save_path = cfg.DEMO.SAVE_PATH_FOR_EVAL
    else:
        save_path = Path(os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME)))
        save_path.mkdir(parents=True, exist_ok=True)
        eval_name = "style"
        eval_id = 0
        filename = f"{eval_name}-{eval_id}_expname_{cfg.NAME}_scale_{str(scale).replace('.','-')}.pkl"
        save_path = os.path.join(save_path, filename)

    logger.info(f"💾 Save path: {save_path}")

    # CUDA
    if cfg.ACCELERATOR == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in cfg.DEVICE)
        device = torch.device("cuda:0")

    # Load Model
    dataset = get_datasets(cfg, logger=logger, phase="test")[0]
    model = get_model(cfg, dataset)
    state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # 2. 数据加载
    print("🚀 Pre-loading data...")
    
    all_contents = []
    # 使用 sorted 确保顺序一致，过滤非 npy 文件
    content_files = sorted([f for f in os.listdir(content_path) if f.endswith('.npy')])
    for f in content_files:
        path = os.path.join(content_path, f)
        data = np.load(path)
        if len(data.shape) == 3: data = data[0]
        
        all_contents.append({
            "name": f.split('.')[0],
            "motion": torch.tensor(data).float(), 
            "length": data.shape[0],
            "label": f.split('.')[0].split("-")[-1]
        })

    all_styles = []
    style_files = sorted([f for f in os.listdir(style_path) if f.endswith('.npy')])
    for f in style_files:
        path = os.path.join(style_path, f)
        data = np.load(path)
        if len(data.shape) == 3: data = data[0]
        
        all_styles.append({
            "name": f.split('.')[0],
            "motion": torch.tensor(data).float(),
            "length": data.shape[0],
            "label": f.split('.')[0].split("-")[-1]
        })

    # 3. 生成任务
    all_tasks = list(itertools.product(all_contents, all_styles))
    total_tasks = len(all_tasks)
    
    print(f"📊 Content files: {len(all_contents)}")
    print(f"📊 Style files: {len(all_styles)}")
    print(f"🔥 Total Pairs to Generate: {total_tasks}")
    
    # 【预警】如果任务数超过 10,000，可能真的会生成很大的文件
    if total_tasks > 10000:
        print("⚠️ Warning: Task count is large. The output file might be huge.")
        print("   If you only want a subset, please modify the code to sample styles.")

    # 4. Batch 推理
    save_all = {"joints": [], "id": [], "label_content": [], "label_style": [], "scene_id": [], "label_scene": []}
    
    BATCH_SIZE = 128
    print(f"🚀 Inference Batch Size: {BATCH_SIZE}")

    dummy_image = torch.zeros(1, 3, 224, 224, device=device)
    dummy_has_image = torch.tensor([False], device=device)

    for i in tqdm(range(0, total_tasks, BATCH_SIZE), desc="Eval Batches"):
        batch_tasks = all_tasks[i : i + BATCH_SIZE]
        current_bs = len(batch_tasks)

        b_c_motions, b_s_motions = [], []
        b_c_lens, b_s_lens = [], []
        # 【新增】用于暂存这一个 Batch 的场景信息
        b_scene_texts = [] 
        b_scene_ids = []
        b_scene_names = []

        random_scene_name = random.choice(SCENE_LIST)
        scene_text = SCENE_DESCRIPTIONS[random_scene_name]
        scene_id = SCENE_LIST.index(random_scene_name) # 获取场景对应的数字ID
        
        for (c, s) in batch_tasks:
            b_c_motions.append(c["motion"])
            b_s_motions.append(s["motion"])
            b_c_lens.append(c["length"])
            b_s_lens.append(s["length"])
            # 【新增】为每个生成任务随机分配一个场景 (这样SCA评估才有意义)
            # 如果你只是想跑通，不关心分数，也可以固定一个
            random_scene_name = random.choice(SCENE_LIST)
            scene_text = SCENE_DESCRIPTIONS[random_scene_name]
            scene_id = SCENE_LIST.index(random_scene_name) # 获取场景对应的数字ID
            b_scene_ids.append(scene_id)
            b_scene_names.append(random_scene_name)
            b_scene_texts.append(scene_text)
            

        c_padded = pad_sequence(b_c_motions, batch_first=True, padding_value=0.0).to(device)
        s_padded = pad_sequence(b_s_motions, batch_first=True, padding_value=0.0).to(device)

        batch = {
            "length": b_c_lens,
            "content_motion": c_padded,
            "style_length": b_s_lens,
            "style_motion": s_padded,
            "tag_scale": scale,
            "scene_text": b_scene_texts,
            "scene_id": b_scene_ids,
            "has_image": dummy_has_image,
            "scene_image": dummy_image.repeat(current_bs, 1, 1, 1)
        }

        with torch.no_grad():
            joints = model(batch, scene_data) # 修复后的 forward 返回 [B, L, J, 3] 或 [B, J, 3, L]

        if isinstance(joints, torch.Tensor):
            joints = joints.detach().cpu().numpy()
            
        for idx, item in enumerate(batch_tasks):
            c_data, s_data = item
            motion_res = joints[idx]
            # ================= [修复点 Start] =================
            # 1. 强制转 Numpy (防止它是 Tensor)
            if isinstance(motion_res, torch.Tensor):
                motion_res = motion_res.detach().cpu().numpy()
            # ================= [修复点 End] =================
            real_len = b_c_lens[idx]
            
            # ================= [关键修复] 去除 Padding =================
            # 自动判断哪个维度是 Length
            # 假设 shape 是 [Length, 22, 3] 或者 [22, 3, Length]
            shape = motion_res.shape
            
            # 情况 1: (Length, J, 3) - 通常是这种情况
            if shape[0] >= real_len: 
                motion_res = motion_res[:real_len]
                
            # 情况 2: (J, 3, Length) - 如果模型输出还没 permute
            elif shape[-1] >= real_len:
                motion_res = motion_res[..., :real_len]
                # 转置回 (Length, J, 3) 以保持统一 (可选，看你之前的 pkl 格式)
                # motion_res = motion_res.transpose(2, 0, 1) 
            
            # ================= [空间优化] float32 -> float16 =================
            # 这步能让文件大小直接减半，且不影响评估指标
            # motion_res = motion_res.astype(np.float16)
            
            save_all["joints"].append(motion_res)
            
            # 构建 ID
            idid = f"content{c_data['name']}_style{s_data['name']}_scale_{str(scale).replace('.', '-')}"
            save_all["id"].append(idid)
            save_all["label_content"].append(c_data["label"])
            save_all["label_style"].append(s_data["label"])
            save_all["label_scene"].append(b_scene_names[idx])
            save_all["scene_id"].append(b_scene_ids[idx]) # 【关键修复】这里存进去，SCA脚本才能读到！

    # 6. 保存 PKL
    print(f"💾 Saving results to {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump(save_all, f)
    
    print("✅ Evaluation Done!")
    end_time = time.perf_counter()

    # 计算差值
    elapsed_time = end_time - start_time
    print(f"🚀 CRA评估任务生成执行耗时: {elapsed_time:.4f} 秒")

if __name__ == "__main__":
    main()