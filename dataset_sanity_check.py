# sanity_check.py
import hydra
from omegaconf import DictConfig
import logging
import torch
import numpy as np
import traceback  # 导入 traceback 模块
import os

# --- [配置日志] ---
# 配置日志，确保所有信息都被捕获并输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler("sanity_check_1025.log"), # 保存到文件
        logging.StreamHandler()                    # 同时在控制台显示
    ]
)
logger = logging.getLogger(__name__)

# --- [辅助函数 - 已修复] ---
def print_tensor_stats(name: str, tensor: torch.Tensor):
    """
    一个更智能的辅助函数，用于打印张量的详细统计信息。
    它现在可以区分处理浮点数张量和布尔型张量。
    """
    if not isinstance(tensor, torch.Tensor):
        logger.info(f"'{name}' is not a tensor, it's a {type(tensor)}.")
        return

    # 基础信息对所有类型都适用
    logger.info(
        f"  - Stats for '{name}':\n"
        f"      Shape: {tensor.shape}\n"
        f"      Dtype: {tensor.dtype}\n"
        f"      Device: {tensor.device}"
    )

    # --- [核心修复] 根据数据类型选择性地打印统计信息 ---
    if tensor.dtype == torch.bool:
        # 如果是布尔型，打印 True/False 的数量
        num_true = tensor.sum().item()
        num_false = tensor.numel() - num_true
        logger.info(
            f"      Num True: {num_true}\n"
            f"      Num False: {num_false}"
        )
    elif tensor.is_floating_point() or tensor.is_complex():
        # 如果是浮点数或复数，打印完整的数学统计
        logger.info(
            f"      Is NaN?: {torch.isnan(tensor).any()}\n"
            f"      Is Inf?: {torch.isinf(tensor).any()}\n"
            f"      Min val: {tensor.min():.4f}\n"
            f"      Max val: {tensor.max():.4f}\n"
            f"      Mean val: {tensor.mean():.4f}\n"
            f"      Std val: {tensor.std():.4f}"
        )
    else:
        # 对于其他类型（如整数），只打印基础信息
        logger.info("      (Tensor is not a float/bool, skipping detailed stats.)")

def main():
    try:
        logger.info("--- [STARTING SIMPLE DATA PIPELINE TEST] ---")

        # --- [2. 硬编码配置] ---
        # !!! 请根据你的实际情况修改下面的路径 !!!
        root_dir = "/root/autodl-tmp/MyRepository/MCM-LDM/"
        humanml3d_root = os.path.join(root_dir, "datasets/humanml3d")
        style100_root = os.path.join(root_dir, "datasets/100StyleDataset")
        word_vectorizer_path = os.path.join(root_dir, "deps/t2m/glove/")
        
        batch_size = 128
        batch_ratio = [64, 64]
        num_workers = 0  # [重要] 调试时先设为 0，避免多进程问题

        # --- [3. 动态导入我们的模块] ---
        from mld.data.humanml.data.dataset import Text2MotionDatasetV2
        from mld.data.style100_dataset import Style100Dataset
        from mld.data.mixed_utils import MixedBatchSampler, mixed_collate_fn, mld_collate_dict, collate_tensors # 确保所有函数都被导入
        from mld.data.humanml.utils.word_vectorizer import WordVectorizer

        # --- [4. 实例化所有组件] ---
        logger.info("Step 1: Instantiating components...")
        
        # Word Vectorizer
        w_vectorizer = WordVectorizer(word_vectorizer_path, "our_vab")

        # 归一化参数 (只使用 HumanML3D 的)
        mean = np.load(os.path.join(humanml3d_root, "Mean.npy"))
        std = np.load(os.path.join(humanml3d_root, "Std.npy"))

        # HumanML3D Dataset
        humanml3d_train = Text2MotionDatasetV2(
            mean=mean, std=std, split_file=os.path.join(humanml3d_root, 'train.txt'),
            motion_dir=os.path.join(humanml3d_root, 'new_joint_vecs'), text_dir=os.path.join(humanml3d_root, 'texts'),
            w_vectorizer=w_vectorizer, max_motion_length=196, min_motion_length=40,
            max_text_len=20, unit_length=4, min_filter_length=40, max_filter_length=200
        )
        logger.info(f"HumanML3D train dataset loaded. Size: {len(humanml3d_train)}")

        # 100Style Dataset
        style100_train = Style100Dataset(
            mean=mean, std=std, split_file=os.path.join(style100_root, 'train.txt'),
            motion_dir=os.path.join(style100_root, 'new_joint_vecs'), text_dir=os.path.join(style100_root, 'texts'),
            style_dict_path=os.path.join(style100_root, 'Style_name_dict.txt'),
            w_vectorizer=w_vectorizer, max_motion_length=196, min_motion_length=40,
            max_text_len=20, unit_length=4, min_filter_length=40, max_filter_length=720
        )
        logger.info(f"100Style train dataset loaded. Size: {len(style100_train)}")

        # 合并数据集
        train_dataset = torch.utils.data.ConcatDataset([humanml3d_train, style100_train])
        logger.info(f"Concatenated dataset created. Total size: {len(train_dataset)}")

        # 采样器
        sampler = MixedBatchSampler(
            dataset_sizes=[len(humanml3d_train), len(style100_train)],
            batch_size=batch_size,
            batch_ratio=batch_ratio
        )
        logger.info("MixedBatchSampler created.")

        # 数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_sampler=sampler,
            num_workers=num_workers, collate_fn=mixed_collate_fn
        )
        logger.info("DataLoader created.")

        # --- [5. 取出一个批次并检查] ---
        logger.info("Step 2: Fetching one batch...")
        batch = next(iter(train_loader))
        logger.info("Successfully fetched one batch.")

        # 5. [核心] 对批次进行深度检查
        logger.info("--- [DEEP BATCH INSPECTION] ---")
        if not batch:
            logger.error("Batch is empty! This should not happen.")
            return

        # 检查 batch 是否是字典
        if not isinstance(batch, dict):
            logger.error(f"FATAL: Batch is not a dictionary, but a {type(batch)}. Check your collate_fn!")
            return
            
        logger.info(f"Batch type: {type(batch)}")
        logger.info(f"Batch keys: {list(batch.keys())}")

        # 检查关键 key 是否存在
        expected_keys = ["motion", "text", "length", "is_text_guided", "word_embs"]
        for key in expected_keys:
            if key not in batch:
                logger.error(f"FATAL: Expected key '{key}' not found in the batch!")
                return
        
        # 详细检查 'is_text_guided'
        is_text_guided_tensor = batch['is_text_guided']
        logger.info("\n--- Checking 'is_text_guided' flag ---")
        print_tensor_stats("is_text_guided", is_text_guided_tensor)
        num_text_guided = torch.sum(is_text_guided_tensor).item()
        num_motion_guided = len(is_text_guided_tensor) - num_text_guided
        logger.info(f"Number of Text-Guided samples (from 100Style): {num_text_guided}")
        logger.info(f"Number of Motion-Guided samples (from HumanML3D): {num_motion_guided}")
        
        # 检查 motion 张量
        logger.info("\n--- Checking 'motion' tensor ---")
        motion_tensor = batch['motion']
        print_tensor_stats("motion", motion_tensor)
        
        # 检查归一化是否正确（抽样检查）
        # 我们期望经过归一化后，数据的均值接近0，标准差接近1（但不完全是，因为每个batch都不同）
        logger.info("Checking normalization stats (should be around mean=0, std=1):")
        # 分别检查两种来源的数据分布
        motion_guided_data = motion_tensor[~is_text_guided_tensor]
        text_guided_data = motion_tensor[is_text_guided_tensor]
        if len(motion_guided_data) > 0:
            logger.info(f"  - Motion-Guided Subset Mean/Std: {motion_guided_data.mean():.4f} / {motion_guided_data.std():.4f}")
        if len(text_guided_data) > 0:
            logger.info(f"  - Text-Guided Subset Mean/Std: {text_guided_data.mean():.4f} / {text_guided_data.std():.4f}")
        
        # 检查其他数据
        logger.info("\n--- Checking other tensors ---")
        print_tensor_stats("word_embs", batch['word_embs'])
        
        logger.info("\n--- Checking list-based data ---")
        logger.info(f"'length' is a list of lengths. Length of list: {len(batch['length'])}. First 5 lengths: {batch['length'][:5]}")
        logger.info(f"'text' is a list of captions. Length of list: {len(batch['text'])}. First caption (HML): '{batch['text'][0]}'")
        if num_text_guided > 0:
             logger.info(f"Last caption (100Style): '{batch['text'][-1]}'") # 假设数据是拼接的
        
        logger.info("\n--- [SANITY CHECK PASSED] ---")
        logger.info("The data pipeline appears to be working correctly. The output batch has the correct structure, keys, and data types.")

    except Exception as e:
        logger.error("--- [SANITY CHECK FAILED] ---", exc_info=True)
        # exc_info=True 会将完整的错误堆栈信息打印到日志中
        traceback.print_exc()

if __name__ == "__main__":
    main()