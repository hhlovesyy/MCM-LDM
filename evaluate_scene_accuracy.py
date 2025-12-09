import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from mld.config import parse_args
from mld.data.get_data import get_datasets # 假设你的数据集加载逻辑在这里
from mld.models.get_model import get_model # 你的生成模型
# from mld.models.architectures.mld_denoiser import SimpleClassifier # 你的分类器定义
import torch.nn as nn

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

def main(args):
    # 1. 加载配置和设备
    cfg = parse_args(phase="test") # 沿用 test 的配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 2. 加载数据集
    # 注意：这里的 get_datasets 需要能返回带场景标签的测试集
    test_dataset = get_datasets(cfg, logger=None, phase="test")[0] 
    # 使用你自己的 collate_fn，因为它处理了 scene 字段
    from mld.data.utils import scene_collate 
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=scene_collate)

    # 3. 加载预训练的生成模型 (Ours)
    gen_model = get_model(cfg, test_dataset)
    print(f"Loading generator model from: {args.gen_ckpt_path}")
    state_dict_gen = torch.load(args.gen_ckpt_path, map_location="cpu")["state_dict"]
    gen_model.load_state_dict(state_dict_gen, strict=True)
    gen_model.to(device)
    gen_model.eval()

    # 4. 加载预训练的场景分类器
    # 假设你的分类器有 14 个类别
    scene_classifier = SimpleClassifier(input_dim=512, num_classes=14) # 假设类别数在配置里
    print(f"Loading scene classifier from: {args.classifier_ckpt_path}")
    # 这里要看你的分类器是怎么保存的，可能需要适配
    state_dict_cls = torch.load(args.classifier_ckpt_path, map_location="cpu")
    # 如果是 pytorch_lightning 保存的，state_dict 可能在 "state_dict" key 下
    if "state_dict" in state_dict_cls:
        # 并且 key 可能带有 "scene_classifier." 前缀，需要去掉
        new_state_dict = {}
        for k, v in state_dict_cls["state_dict"].items():
            if k.startswith("scene_classifier."):
                new_state_dict[k.replace("scene_classifier.", "")] = v
        scene_classifier.load_state_dict(new_state_dict, strict=True)
    else:
        scene_classifier.load_state_dict(state_dict_cls, strict=True)
    
    scene_classifier.to(device)
    scene_classifier.eval()

    # 5. 开始评测
    total_samples = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Scene Accuracy"):
            # 把 batch 里的所有东西都放到 device 上
            # 注意：你的 collate_fn 返回的是一个字典
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # 为了公平，我们需要一个 style motion
            # 假设测试集的 motion 就是 style motion
            batch["style_motion"] = batch["motion"]
            batch["content_motion"] = batch["motion"].clone()
            batch["content_motion"][...,:3] = 0.0

            # Step A: 生成动作
            # 假设模型返回 (joints, extra_info)
            generated_joints, _ = gen_model(batch) # generated_joints: [B, T, D]

            # Step B: 提取生成动作的特征 (用 MotionCLIP)
            # 这一步需要复用你训练代码里的特征提取逻辑
            motion_feat_pred = gen_model.motionclip.encoder({
                'x': generated_joints.permute(0, 2, 1).unsqueeze(2), # [B, D, 1, T]
                'y': torch.zeros(generated_joints.shape[0], dtype=int, device=device),
                'mask': torch.ones(generated_joints.shape[0], generated_joints.shape[1], dtype=bool, device=device) # 简化的 mask
            })["mu"] # [B, 512]
            
            # Step C: 进行分类
            logits = scene_classifier(motion_feat_pred)
            predicted_labels = torch.argmax(logits, dim=1)
            
            # Step D: 比较结果
            true_labels = batch["scene_id"]
            correct_predictions += (predicted_labels == true_labels).sum().item()
            total_samples += len(true_labels)

    accuracy = (correct_predictions / total_samples) * 100
    print(f"Scene Consistency Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_samples})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_ckpt_path", type=str, required=True, help="Path to your generator model checkpoint.")
    parser.add_argument("--classifier_ckpt_path", type=str, required=True, help="Path to your scene classifier checkpoint.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    args = parser.parse_args()
    main(args)