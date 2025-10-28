# ===> START: 新版 visualize_embeddings.py (带诊断) <===
import torch
import numpy as np
import clip
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse
from mld.config import parse_args
from mld.data.get_data import get_datasets
from mld.models.get_model import get_model
from mld.utils.logger import create_logger
from torch.utils.data import DataLoader
from mld.data.mixed_datamodule import mixed_collate_fn # 确保导入
import matplotlib.cm as cm
import logging # 使用 logging 模块
from mld.data.mixed_datamodule import mixed_collate_fn, MixedBatchSampler # <--- 修改这一行

def visualize_embeddings(cfg, model, dataloader, device, logger):
    """
    提取 embedding 并进行 t-SNE 可视化。
    """
    # ... (这部分函数不变，主要是 main 函数需要修改以进行诊断) ...
    # (为了简洁，我省略了 visualize_embeddings 函数本身的代码，因为它没有变)
    model.to(device)
    model.eval()
    
    all_text_labels = []
    all_text_embeddings = []
    all_motion_embeddings = []
    
    logger.info("开始从数据集中提取 embedding...")
    with torch.no_grad():
        num_batches_to_process = getattr(cfg.DEMO, 'NUM_BATCHES', 5)
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches_to_process:
                logger.info(f"已处理 {num_batches_to_process} 个 batch，停止提取。")
                break
            
            # --- [诊断代码 1] 打印每个 batch 的关键信息 ---
            logger.info(f"--- Batch {i+1}/{num_batches_to_process} ---")
            logger.info(f"  Batch keys: {list(batch.keys())}")
            logger.info(f"  'motion' tensor shape: {batch['motion'].shape}")
            if 'is_text_guided' in batch:
                is_text_guided_mask = batch['is_text_guided']
                logger.info(f"  'is_text_guided' tensor: {is_text_guided_mask}")
                logger.info(f"  'is_text_guided' sum (text-guided count): {torch.sum(is_text_guided_mask).item()}")
            else:
                logger.error("  'is_text_guided' key NOT FOUND in the batch!")
                continue # 如果没有这个 key，直接跳到下一个 batch
            # --- [诊断代码结束] ---

            if not is_text_guided_mask.any():
                logger.warning(f"  Batch {i+1} 中没有 text-guided 样本，跳过。")
                continue

            texts = [t for t, flag in zip(batch['text'], is_text_guided_mask) if flag]
            motion = batch['motion'][is_text_guided_mask].to(device)
            lengths = [l for l, flag in zip(batch['length'], is_text_guided_mask) if flag]
            
            if not texts:
                continue
            
            text_tokens = clip.tokenize(texts).to(device)
            with torch.cuda.amp.autocast(enabled=False):
                raw_clip_features = model.clip_model.encode_text(text_tokens).float()
            text_emb = model.text_adapter(raw_clip_features)
            
            motion_unnormalized = motion * model.std.to(device) + model.mean.to(device)
            motion_unnormalized[..., :3] = 0.0
            motion_seq = motion_unnormalized.unsqueeze(-1).permute(0, 2, 3, 1)
            
            from mld.utils.temos_utils import lengths_to_mask
            mask = lengths_to_mask(lengths, device)

            motion_emb = model.motionclip_teacher.encoder({
                'x': motion_seq,
                'y': torch.zeros(motion_seq.shape[0], dtype=int, device=device),
                'mask': mask
            })["mu"]
            
            all_text_labels.extend(texts)
            all_text_embeddings.append(text_emb.cpu())
            all_motion_embeddings.append(motion_emb.cpu())
            
            logger.info(f"  成功处理了 {len(texts)} 个 text-guided 样本。")

    if not all_text_labels:
        logger.error("在指定数量的 batch 中没有找到任何 text-guided 的样本，请检查数据集配置。")
        return

    # ... (后续的 t-SNE 和绘图代码保持不变) ...
    all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
    all_motion_embeddings = torch.cat(all_motion_embeddings, dim=0)
    logger.info(f"Embedding 提取完毕 (共 {len(all_text_labels)} 个样本)，开始进行 t-SNE 降维...")
    perplexity_value = min(30, len(all_text_labels) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value, max_iter=1000)
    combined_embeddings = torch.cat([all_text_embeddings, all_motion_embeddings], dim=0)
    embeddings_2d = tsne.fit_transform(combined_embeddings.numpy())
    text_embeddings_2d = embeddings_2d[:len(all_text_labels)]
    motion_embeddings_2d = embeddings_2d[len(all_text_labels):]
    logger.info("t-SNE 降维完成，开始绘图...")
    plt.figure(figsize=(20, 20))
    unique_labels = sorted(list(set(all_text_labels)))
    colors = cm.get_cmap('tab20', len(unique_labels))
    label_to_color = {label: colors(i) for i, label in enumerate(unique_labels)}
    for i, label in enumerate(all_text_labels):
        color = label_to_color[label]
        plt.scatter(text_embeddings_2d[i, 0], text_embeddings_2d[i, 1], color=color, marker='o', s=100, alpha=0.8, edgecolors='black')
        plt.scatter(motion_embeddings_2d[i, 0], motion_embeddings_2d[i, 1], color=color, marker='x', s=100, alpha=0.8)
        plt.plot([text_embeddings_2d[i, 0], motion_embeddings_2d[i, 0]], [text_embeddings_2d[i, 1], motion_embeddings_2d[i, 1]], color=color, linestyle='--', linewidth=0.6, alpha=0.7)
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=label_to_color[label], markersize=12) for label in unique_labels]
    legend_elements.extend([Line2D([0], [0], marker='o', mfc='none', mec='gray', label='Text Embedding', markersize=12), Line2D([0], [0], marker='x', color='gray', label='Motion Embedding', markersize=12, linestyle='None')])
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', title='Styles')
    plt.title('t-SNE Visualization of Text and Motion Embeddings', fontsize=18)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    output_dir = cfg.FOLDER
    save_path = f"{output_dir}/embedding_visualization2.png"
    plt.savefig(save_path)
    logger.info(f"可视化图像已保存到: {save_path}")

def main():
    cfg = parse_args(phase="demo")
    output_dir = create_logger(cfg, phase='demo')
    logger = logging.getLogger()

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.ACCELERATOR == "gpu" else "cpu")
    logger.info(f"Using device: {device}")
    
    # --- [诊断代码 2] 检查数据集和采样器 ---
    logger.info("Loading validation dataset for DIAGNOSIS...")
    datamodule = get_datasets(cfg, logger=logger, phase="val")[0]
    datamodule.setup()
    
    logger.info("--- Dataset Diagnosis ---")
    val_dataset = datamodule.val_dataset
    logger.info(f"  Validation dataset type: {type(val_dataset)}")
    if isinstance(val_dataset, torch.utils.data.ConcatDataset):
        logger.info(f"  Dataset is a ConcatDataset with {len(val_dataset.datasets)} subsets.")
        dataset_sizes = [len(d) for d in val_dataset.datasets]
        logger.info(f"  Subset sizes: {dataset_sizes}")
        # 假设第一个是 humanml3d, 第二个是 style100
        if len(dataset_sizes) > 1 and dataset_sizes[1] == 0:
            logger.error("  CRITICAL: The second dataset (expected Style100) is EMPTY!")
    logger.info("--- End of Dataset Diagnosis ---")

    # ===> START: 替换代码 <===
    logger.info("Creating MIXED validation dataloader with MixedBatchSampler for visualization...") 

    # a. 获取验证集的子集大小
    val_dataset_sizes = [len(d) for d in datamodule.val_dataset.datasets]
    if any(size == 0 for size in val_dataset_sizes):
        logger.error("One of the validation datasets is empty. Cannot proceed.")
        return # 提前退出

    # b. 创建 MixedBatchSampler
    val_sampler = MixedBatchSampler(
        dataset_sizes=val_dataset_sizes, 
        batch_size=cfg.TEST.BATCH_SIZE, # 使用配置中的批次大小
        batch_ratio=cfg.DATASET.MIXED.BATCH_RATIO # 使用与训练时相同的混合比例
    )

    # c. 创建 DataLoader，注意：当使用 batch_sampler 时，batch_size, shuffle, drop_last 必须为 None
    val_loader = DataLoader(
        datamodule.val_dataset,
        batch_sampler=val_sampler, # <--- [核心修复] 使用我们创建的采样器
        num_workers=cfg.TRAIN.NUM_WORKERS,
        collate_fn=mixed_collate_fn 
    )
    # ===> END: 替换代码 <===
    
    model = get_model(cfg, datamodule)
    
    logger.info(f"Loading checkpoints from {cfg.DEMO.CHECKPOINT}")
    state_dict = torch.load(cfg.DEMO.CHECKPOINT, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=True)

    visualize_embeddings(cfg, model, val_loader, device, logger)

if __name__ == "__main__":
    main()
# ===> END: 新版 visualize_embeddings.py (带诊断) <===