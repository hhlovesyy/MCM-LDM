import os
from pprint import pformat

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.strategies.ddp import DDPStrategy

from mld.callback import ProgressLogger
from mld.config import parse_args
from mld.data.get_data import get_datasets
from mld.models.get_model import get_model
from mld.utils.logger import create_logger
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from mld.models.motionclip_263.utils.get_model_and_data import get_model_and_data
import torch.optim as optim
from tqdm import tqdm
from typing import Dict, List
from torch import Tensor

# class SimpleClassifier(nn.Module):
#     def __init__(self, input_dim=512, num_classes=14):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, num_classes)
#         )
#     def forward(self, x):
#         return self.net(x)

# 1. 修改模型定义：大幅缩减参数量
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


def main():
    cfg = parse_args()  # parse config file

    # create logger
    logger = create_logger(cfg, phase="train")
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


    logger.info("datasets module {} initialized".format("".join(
        cfg.TRAIN.DATASETS)))
    parameters = read_yaml_to_dict("configs/motionclip_config/motionclip_params_263.yaml")
    parameters["device"] = 'cuda:{}'.format(cfg["DEVICE"][0])        
    frozen_motionclip = get_model_and_data(parameters, split='vald')
    print("load motion clip-xyz-263")
    print("Restore weights..")
    checkpointpath = "checkpoints/motionclip_checkpoint/motionclip.pth.tar"
    state_dict = torch.load(checkpointpath, map_location=parameters["device"])
    frozen_motionclip.load_state_dict(state_dict, strict=False)

    #don't train motionclip
    frozen_motionclip.training = False
    for p in frozen_motionclip.parameters():
        p.requires_grad = False
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 3. 初始化分类器
    classifier = SimpleClassifier().to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    # 2. 修改 Loss 定义：加入标签平滑
    # label_smoothing=0.1 意味着告诉模型：你别太自信，保留 10% 的怀疑
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    dataset_mean = torch.tensor(datamodule.hparams.mean).to(device)
    dataset_std = torch.tensor(datamodule.hparams.std).to(device)

    # 开始训练！
    print(">>> Start Training Classifier...")
    for epoch in range(10): # 20个epoch足够
        total_loss = 0
        correct = 0
        total = 0
        for batch in tqdm(dataloader):
            # 获取数据
            motion = batch["motion"].to(device) # torch.Size([64, 436, 263])
            # 假设 dataset 返回了 scene_id
            labels = batch["scene_id"].to(device) # torch.Size([64])
            lengths = batch["length"] 
            with torch.no_grad():
                # 1. 随机噪声 (Random Noise)
                # 给动作加一点抖动，模拟生成模型初期不完美的输出
                noise = torch.randn_like(motion) * 0.05 
                motion = motion + noise
                
                # 2. 随机缩放 (Random Scaling)
                # 改变动作的幅度 (0.8x ~ 1.2x)
                scale = 0.8 + 0.4 * torch.rand(motion.shape[0], 1, 1, device=device)
                motion = motion * scale

                motion_seq = motion * dataset_std + dataset_mean  # torch.Size([64, 436, 263])
                motion_seq[...,:3]=0.0
                motion_seq = motion_seq.unsqueeze(-1).permute(0,2,3,1) # torch.Size([64, 263, 1, 436])
                motion_emb = frozen_motionclip.encoder({'x': motion_seq,
                        'y': torch.zeros(motion_seq.shape[0], dtype=int, device=device),
                        'mask': lengths_to_mask(lengths, device=device)})["mu"] # 一个style被提取成了512维的tensor，torch.Size([64, 512])
                # motion_emb = motion_emb.unsqueeze(1) # torch.Size([32, 1, 512])
            
            # --- 数据增强 2: 特征层面的高斯噪声 (【新加】) ---
            # 这一步非常重要！模拟 Diffusion 生成的不完美特征
            # 随着 Epoch 增加，噪声可以减小，或者一直保持
            feature_noise = torch.randn_like(motion_emb) * 0.1 # 0.1 的强度很大了
            motion_emb_noisy = motion_emb + feature_noise
            # 训练分类器
            logits = classifier(motion_emb_noisy) # torch.Size([64, 14])
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            
        print(f"Epoch {epoch}: Acc {correct/total:.4f}")
        
    torch.save(classifier.state_dict(), "checkpoints/1204/scene_classifier.pth")
    print("Saved classifier!")

if __name__ == "__main__":
    main()