# evaluate_sca.py

import torch
import torch.nn as nn
import numpy as np
import pickle
from tqdm import tqdm
from argparse import ArgumentParser
from torch import Tensor

# ... (复制 SimpleClassifier, lengths_to_mask, read_yaml_to_dict 的定义) ...
from mld.models.motionclip_263.utils.get_model_and_data import get_model_and_data
from mld.data.get_data import get_datasets # <--- 需要这个来获取 datamodule
from mld.config import parse_args
import yaml
from typing import Dict, List
from torch.utils.data import DataLoader

# ...
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=14):
        super().__init__()
        self.net = nn.Sequential(
            # 第一层直接降维到 64 或 128
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2), # 换个激活函数，防止死区
            
            # 加大 Dropout 到 0.5
            nn.Dropout(0.5),
            
            # 输出层
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def read_yaml_to_dict(yaml_path: str, ):
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value
    
def lengths_to_mask(lengths: List[int],
                    device: torch.device,
                    max_len: int = None) -> Tensor:
    lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def evaluate_sca(pkl_path, classifier_ckpt_path, motionclip_ckpt_path, datamodule, device):
    """
    加载生成的动作，使用预训练的分类器进行评估，并计算SCA。
    """
    # 1. 加载预训练的 MotionCLIP (用于特征提取)
    print("Loading frozen MotionCLIP...")
    parameters = read_yaml_to_dict("configs/motionclip_config/motionclip_params_263.yaml")
    parameters["device"] = device
    frozen_motionclip = get_model_and_data(parameters, split='vald')
    state_dict = torch.load(motionclip_ckpt_path, map_location=device)
    frozen_motionclip.load_state_dict(state_dict, strict=False)
    frozen_motionclip.eval()
    for p in frozen_motionclip.parameters():
        p.requires_grad = False
    
    # 2. 加载你训练好的场景分类器
    print(f"Loading Scene Classifier from {classifier_ckpt_path}...")
    classifier = SimpleClassifier(num_classes=14).to(device)
    classifier.load_state_dict(torch.load(classifier_ckpt_path, map_location=device))
    classifier.eval()

    # 3. 加载生成的动作数据 (不变)
    print(f"Loading generated motions from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    generated_joints = data['joints']
    gt_scene_ids = data['scene_id']

    # 4. 开始评估
    print(f"Evaluating {len(generated_joints)} samples...")
    correct_predictions = 0
    correct_top3 = 0
    total_samples = len(generated_joints)

    device_mean = torch.tensor(datamodule.hparams.mean, device=device).float()
    device_std = torch.tensor(datamodule.hparams.std, device=device).float()
    
    with torch.no_grad():
        # 我们采用逐个处理，逻辑最清晰
        for i in tqdm(range(total_samples), desc="Evaluating SCA"):
            joints_np = generated_joints[i] # [T, 22, 3]
            gt_label = gt_scene_ids[i]
            
            # --- 【核心修正】确保传入 joints2feats 的是 NumPy 数组 ---
            # a. [T, 22, 3] -> [1, T, 22, 3]
            # joints2feats 可能期望一个 batch，所以我们增加一个 batch 维度
            joints_np_batch = np.expand_dims(joints_np, axis=0)

            # b. 直接将 NumPy 数组传入函数
            # 函数内部会处理它，我们不需要提前转成 Tensor
            feats_normalized_np = datamodule.joints2feats(joints_np_batch) # [1, T, 263], 应该返回 NumPy

            # c. 得到函数返回的 NumPy 结果后，再将其转换为 Tensor 以进行后续 GPU 计算
            feats_normalized = torch.from_numpy(feats_normalized_np).to(device).float()

            # --- 后续逻辑保持不变 ---
            # d. 反归一化得到真实尺度的 feats
            # feats_unnormalized = feats_normalized * device_std.to(device) + device_mean.to(device)
            feats_unnormalized = feats_normalized
            
            # c. 去除轨迹，准备送入 MotionCLIP
            feats_unnormalized[..., :3] = 0.0
            
            motion_for_clip = feats_unnormalized.unsqueeze(-1).permute(0, 2, 3, 1) # [1, 263, 1, T]

            # d. 提取特征
            lengths = [joints_np.shape[0]]
            motion_emb = frozen_motionclip.encoder({
                'x': motion_for_clip,
                'y': torch.zeros(1, dtype=int, device=device),
                'mask': lengths_to_mask(lengths, device=device)
            })["mu"]

            # e. 分类
            logits = classifier(motion_emb)
            _, topk_preds = torch.topk(logits, k=3, dim=1) # 获取前3个预测
            # in_top3 = gt_label in topk_preds.squeeze().cpu().numpy()
            if gt_label in topk_preds.squeeze().tolist():
                correct_top3 += 1
            pred_label = torch.argmax(logits, dim=1).item()

            if pred_label == gt_label:
                correct_predictions += 1

    sca_score = (correct_predictions / total_samples) * 100
    sca_score_top3 = (correct_top3 / total_samples) * 100
    print("\n" + "="*50)
    print(f"Scene Consistency Accuracy (SCA): {sca_score:.2f}%")
    print(f"({correct_predictions} / {total_samples} correct)")
    print(f"Top-3 Scene Consistency Accuracy: {sca_score_top3:.2f}%")
    print(f"({correct_top3} / {total_samples} correct in Top-3)")
    print("="*50)

    return sca_score

def evaluate_sca_debug(pkl_path, classifier_ckpt_path, motionclip_ckpt_path, datamodule, device, debug_index=5):
    """
    [DEBUG VERSION] 加载单个生成动作，详细打印每一步的数据转换过程。
    """
    
    # --- 1. 初始化 (与原版相同) ---
    print("Loading frozen MotionCLIP...")
    parameters = read_yaml_to_dict("configs/motionclip_config/motionclip_params_263.yaml")
    parameters["device"] = device
    frozen_motionclip = get_model_and_data(parameters, split='vald')
    state_dict = torch.load(motionclip_ckpt_path, map_location=device)
    frozen_motionclip.load_state_dict(state_dict, strict=False)
    frozen_motionclip.eval()
    for p in frozen_motionclip.parameters():
        p.requires_grad = False
    
    print(f"Loading Scene Classifier from {classifier_ckpt_path}...")
    classifier = SimpleClassifier(num_classes=14).to(device)
    classifier.load_state_dict(torch.load(classifier_ckpt_path, map_location=device))
    classifier.eval()

    print(f"Loading generated motions from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    generated_joints = data['joints']
    gt_scene_ids = data['scene_id']

    # 提前准备好 Tensor 格式的均值和标准差
    device_mean = torch.tensor(datamodule.hparams.mean, device=device).float()
    device_std = torch.tensor(datamodule.hparams.std, device=device).float()

    # --- 2. 【核心调试部分】 ---
    print("\n" + "#"*50)
    print("###      STARTING SINGLE SAMPLE DEBUG      ###")
    print("#"*50 + "\n")

    with torch.no_grad():
        # 选择一个样本进行调试
        if debug_index >= len(generated_joints):
            print(f"Error: debug_index ({debug_index}) is out of bounds. Using index 0 instead.")
            debug_index = 0
            
        joints_np = generated_joints[debug_index] # [T, 22, 3]
        gt_label = gt_scene_ids[debug_index]
        
        # --- 步骤 0: 原始 Joints 数据 ---
        print(f"--- [Step 0] Input Joints Data ---")
        print(f"  - Sample Index: {debug_index}")
        print(f"  - Ground Truth Scene ID: {gt_label}")
        print(f"  - Data Type: {type(joints_np)}")
        print(f"  - Shape: {joints_np.shape}")
        print(f"  - Mean value: {np.mean(joints_np):.4f}, Std: {np.std(joints_np):.4f}")
        print(f"  - Min: {np.min(joints_np):.4f}, Max: {np.max(joints_np):.4f}")

        # --- 步骤 1: `joints2feats` 转换 ---
        joints_np_batch = np.expand_dims(joints_np, axis=0) # [1, T, 22, 3]
        feats_normalized_np = datamodule.joints2feats(joints_np_batch) # [1, T, 263]
        
        print(f"\n--- [Step 1] Output of `joints2feats` (Normalized Feats) ---")
        print(f"  - Data Type: {type(feats_normalized_np)}")
        print(f"  - Shape: {feats_normalized_np.shape}")
        print(f"  - Mean value: {np.mean(feats_normalized_np):.4f}, Std: {np.std(feats_normalized_np):.4f}")
        print(f"  - Min: {np.min(feats_normalized_np):.4f}, Max: {np.max(feats_normalized_np):.4f}")
        
        # --- 步骤 2: 反归一化 ---
        feats_normalized_tensor = torch.from_numpy(feats_normalized_np).to(device).float()
        feats_unnormalized = feats_normalized_tensor * device_std + device_mean # [1, T, 263]
        
        print(f"\n--- [Step 2] Un-normalized Feats (for MotionCLIP) ---")
        print(f"  - Data Type: {type(feats_unnormalized)}")
        print(f"  - Shape: {feats_unnormalized.shape}")
        print(f"  - Mean value: {feats_unnormalized.mean().item():.4f}, Std: {feats_unnormalized.std().item():.4f}")
        print(f"  - Min: {feats_unnormalized.min().item():.4f}, Max: {feats_unnormalized.max().item():.4f}")

        # --- 步骤 3: 准备 MotionCLIP 输入 ---
        feats_for_clip = feats_unnormalized.clone()
        feats_for_clip[..., :3] = 0.0 # 去除轨迹
        motion_for_clip = feats_for_clip.unsqueeze(-1).permute(0, 2, 3, 1) # [1, 263, 1, T]

        # --- 步骤 4: MotionCLIP 特征提取 ---
        lengths = [joints_np.shape[0]]
        motion_emb = frozen_motionclip.encoder({
            'x': motion_for_clip,
            'y': torch.zeros(1, dtype=int, device=device),
            'mask': lengths_to_mask(lengths, device=device)
        })["mu"] # [1, 512]

        print(f"\n--- [Step 4] MotionCLIP Embedding ---")
        print(f"  - Data Type: {type(motion_emb)}")
        print(f"  - Shape: {motion_emb.shape}")
        print(f"  - Mean value: {motion_emb.mean().item():.4f}, Std: {motion_emb.std().item():.4f}")
        print(f"  - Min: {motion_emb.min().item():.4f}, Max: {motion_emb.max().item():.4f}")
        
        # --- 步骤 5: 分类器输出 ---
        logits = classifier(motion_emb) # [1, 14]
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        pred_label = torch.argmax(logits, dim=1).item()
        
        print(f"\n--- [Step 5] Final Classification Result ---")
        print(f"  - PREDICTED LABEL: {pred_label}")
        print(f"  - GROUND TRUTH:    {gt_label}")
        print(f"  - CORRECT?          {'YES' if pred_label == gt_label else 'NO'}")
        print(f"  - Confidence: {probabilities.max().item():.2%}")
        # 打印出每个类别的概率，方便观察
        probs_list = [f"{p:.1%}" for p in probabilities.squeeze().cpu().tolist()]
        print(f"  - Probabilities per class: {probs_list}")
        
    print("\n" + "#"*50)
    print("###        DEBUG MODE FINISHED         ###")
    print("#"*50)

    # 在调试模式下，我们不返回任何值，只看打印结果
    return None


if __name__ == '__main__':
    cfg = parse_args()  # parse config file

    # create logger
    # logger = create_logger(cfg, phase="train")
    cfg.TRAIN.BATCH_SIZE = 64
    datamodule = get_datasets(cfg, logger=None, phase="train")[0]
    datamodule.setup(stage="fit")
    real_dataset = datamodule.train_dataset
    collate_fn = datamodule.dataloader_options.get("collate_fn", None)
    # 重新创建一个dataloader，可以自定义训练的配置，不用管那个cfg里面的yaml，因为我们这个算是一个独立的模块
    dataloader = DataLoader(
        real_dataset, 
        batch_size=64, # 这里可以随意改，不用管 cfg.TRAIN.BATCH_SIZE
        shuffle=True, 
        num_workers=4, 
        collate_fn=collate_fn,
        drop_last=True
    )
    print(f"Data Loaded. Dataset size: {len(real_dataset)}")
    # 检查一下collate_fn的函数名
    print(f"Collate fn: {collate_fn.__name__ if collate_fn else 'None'}")
    DEVICE = 'cuda:{}'.format(cfg["DEVICE"][0])

    input_path = "/root/autodl-tmp/MyRepository/MCM-LDM/results/mld/SceneMo_1220_2320_Full_Eval/crafmd-0_expname_SceneMo_1220_2320_Full_Eval_scale_2-5.pkl"

    evaluate_sca(input_path, 
                 "checkpoints/1204/scene_classifier.pth", 
                 "checkpoints/motionclip_checkpoint/motionclip.pth.tar", datamodule, DEVICE)
    # 运行debug的版本
    # evaluate_sca_debug("/root/autodl-tmp/MyRepository/MCM-LDM/results/mld/SceMoDiff_Evaluation_Full/crafmd-0_expname_SceMoDiff_Evaluation_Full_scale_2-5.pkl", "checkpoints/1204/scene_classifier.pth", "checkpoints/motionclip_checkpoint/motionclip.pth.tar", datamodule, DEVICE, debug_index=5)