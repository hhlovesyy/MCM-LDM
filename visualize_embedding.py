# ===> START: 最终修复版 visualize_embeddings.py (基于您的版本，最小化修改) <===
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
from mld.data.mixed_datamodule import mixed_collate_fn
import matplotlib.cm as cm
import logging
from mld.data.mixed_datamodule import MixedBatchSampler

# 注意：我将您提供的脚本中的 visualize_embeddings 函数补全了
# 您之前发的版本中，这个函数的主体被省略了，我现在把它加回来

def visualize_embeddings(cfg, model, dataloader, device, logger):
    """
    提取 embedding 并进行 t-SNE 可视化 (最终安全版)。
    """
    # --- [手动编辑区] ---
    # 如果想聚焦，请在这里填入风格列表。如果想看全部，请设置为 None。
    focus_styles = ["proud", "old", "chicken", "walking"]
    # --- [手动编辑区结束] ---

    # 1. --- 数据提取 (与您能运行的版本完全一致) ---
    model.to(device)
    model.eval()
    
    all_text_labels = []
    all_text_embeddings = []
    all_motion_embeddings = []
    
    logger.info("开始从数据集中提取 embedding...")
    with torch.no_grad():
        num_batches_to_process = getattr(cfg.DEMO, 'NUM_BATCHES', 15)
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches_to_process:
                logger.info(f"已处理 {num_batches_to_process} 个 batch，停止提取。")
                break
            
            if 'is_text_guided' not in batch or not batch['is_text_guided'].any():
                continue

            is_text_guided_mask = batch['is_text_guided']
            texts = [t for t, flag in zip(batch['text'], is_text_guided_mask) if flag]
            motion = batch['motion'][is_text_guided_mask].to(device)
            lengths = [l for l, flag in zip(batch['length'], is_text_guided_mask) if flag]
            
            if not texts:
                continue
            
            text_tokens = clip.tokenize(texts).to(device)
            with torch.cuda.amp.autocast(enabled=False):
                raw_clip_features = model.clip_model.encode_text(text_tokens).float()
            raw_text_emb = model.text_adapter(raw_clip_features)
            text_emb = model.text_emb_norm(raw_text_emb)
            
            motion_unnormalized = motion * model.std.to(device) + model.mean.to(device)
            motion_unnormalized[..., :3] = 0.0
            motion_seq = motion_unnormalized.unsqueeze(-1).permute(0, 2, 3, 1)
            
            from mld.utils.temos_utils import lengths_to_mask
            mask = lengths_to_mask(lengths, device)

            raw_motion_emb = model.motionclip_teacher.encoder({
                'x': motion_seq,
                'y': torch.zeros(motion_seq.shape[0], dtype=int, device=device),
                'mask': mask
            })["mu"]
            motion_emb = model.motion_emb_norm(raw_motion_emb)
            
            all_text_labels.extend(texts)
            all_text_embeddings.append(text_emb.cpu())
            all_motion_embeddings.append(motion_emb.cpu())
            
            logger.info(f"  成功处理了 {len(texts)} 个 text-guided 样本。")

    if not all_text_labels:
        logger.error("在指定数量的 batch 中没有找到任何 text-guided 的样本。")
        return

    # 2. --- t-SNE 降维 (与您能运行的版本完全一致) ---
    all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
    all_motion_embeddings = torch.cat(all_motion_embeddings, dim=0)
    logger.info(f"Embedding 提取完毕 (共 {len(all_text_labels)} 个样本)，开始进行 t-SNE 降维...")
    perplexity_value = min(30, len(all_text_labels) - 1)
    if perplexity_value <= 0:
        logger.error(f"样本数量 ({len(all_text_labels)}) 过少，无法进行 t-SNE。")
        return
        
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value, max_iter=1000)
    combined_embeddings = torch.cat([all_text_embeddings, all_motion_embeddings], dim=0)
    embeddings_2d = tsne.fit_transform(combined_embeddings.numpy())
    text_embeddings_2d = embeddings_2d[:len(all_text_labels)]
    motion_embeddings_2d = embeddings_2d[len(all_text_labels):]
    logger.info("t-SNE 降维完成，开始绘图...")

    # 3. --- 绘图 (带有筛选逻辑) ---
    plt.figure(figsize=(20, 20))
    
    unique_labels = sorted(list(set(all_text_labels)))
    colors = cm.get_cmap('tab20', len(unique_labels))
    label_to_color = {label: colors(i) for i, label in enumerate(unique_labels)}
    
    # [核心修改] 只在这里进行筛选
    indices_to_plot = []
    if focus_styles:
        logger.info(f"聚焦模式绘图，只显示以下风格: {focus_styles}")
        for i, label in enumerate(all_text_labels):
            if label in focus_styles:
                indices_to_plot.append(i)
    else:
        indices_to_plot = list(range(len(all_text_labels)))
        
    if not indices_to_plot:
        logger.error(f"在提取的所有样本中，未找到指定的聚焦风格: {focus_styles}")
        return

    for i in indices_to_plot:
        label = all_text_labels[i]
        color = label_to_color[label]
        plt.scatter(text_embeddings_2d[i, 0], text_embeddings_2d[i, 1], color=color, marker='o', s=100, alpha=0.8, edgecolors='black')
        plt.scatter(motion_embeddings_2d[i, 0], motion_embeddings_2d[i, 1], color=color, marker='x', s=100, alpha=0.8)
        plt.plot([text_embeddings_2d[i, 0], motion_embeddings_2d[i, 0]], 
                 [text_embeddings_2d[i, 1], motion_embeddings_2d[i, 1]], 
                 color=color, linestyle='--', linewidth=0.6, alpha=0.7)
                 
    from matplotlib.lines import Line2D
    legend_labels_to_show = focus_styles if focus_styles else unique_labels
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=label_to_color.get(label, 'gray'), markersize=12) for label in legend_labels_to_show]
    legend_elements.extend([Line2D([0], [0], marker='o', mfc='none', mec='gray', label='Text Embedding', markersize=12), Line2D([0], [0], marker='x', color='gray', label='Motion Embedding', markersize=12, linestyle='None')])
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', title='Styles')
    plt.title('t-SNE Visualization of Text and Motion Embeddings', fontsize=18)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    output_dir = cfg.FOLDER
    filename = "embedding_visualization_focused.png" if focus_styles else "embedding_visualization_all.png"
    save_path = f"{output_dir}/{filename}"
    plt.savefig(save_path)
    logger.info(f"可视化图像已保存到: {save_path}")

def main():
    # 您的 main 函数保持完全不变，我只是把它从您提供的代码中复制过来
    cfg = parse_args(phase="demo")
    output_dir = create_logger(cfg, phase='demo')
    logger = logging.getLogger()

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.ACCELERATOR == "gpu" else "cpu")
    logger.info(f"Using device: {device}")
    
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
        if len(dataset_sizes) > 1 and dataset_sizes[1] == 0:
            logger.error("  CRITICAL: The second dataset (expected Style100) is EMPTY!")
    logger.info("--- End of Dataset Diagnosis ---")

    logger.info("Creating MIXED validation dataloader with MixedBatchSampler for visualization...") 
    val_dataset_sizes = [len(d) for d in datamodule.val_dataset.datasets]
    if any(size == 0 for size in val_dataset_sizes):
        logger.error("One of the validation datasets is empty. Cannot proceed.")
        return

    val_sampler = MixedBatchSampler(
        dataset_sizes=val_dataset_sizes, 
        batch_size=cfg.TEST.BATCH_SIZE,
        batch_ratio=cfg.DATASET.MIXED.BATCH_RATIO
    )

    val_loader = DataLoader(
        datamodule.val_dataset,
        batch_sampler=val_sampler,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        collate_fn=mixed_collate_fn 
    )
    
    model = get_model(cfg, datamodule)
    
    logger.info(f"Loading checkpoints from {cfg.DEMO.CHECKPOINT}")
    state_dict = torch.load(cfg.DEMO.CHECKPOINT, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=True)

    visualize_embeddings(cfg, model, val_loader, device, logger)

if __name__ == "__main__":
    main()
# ===> END: 最终修复版 visualize_embeddings.py (基于您的版本，最小化修改) <===