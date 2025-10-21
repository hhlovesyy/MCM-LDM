# mld/data/mixed_utils.py

import torch
import torch.utils.data
import numpy as np
import logging
from .utils import mld_collate # 从现有 utils 导入原始的 collate 函数

# 初始化日志记录器
logger = logging.getLogger(__name__)

class MixedBatchSampler(torch.utils.data.Sampler):
    """
    一个自定义的 Sampler，用于从多个数据集中按【固定的样本数量】采样，以创建混合批次。
    """
    def __init__(self, dataset_sizes, batch_size, batch_ratio):
        """
        初始化 Sampler。
        :param dataset_sizes: 一个包含每个数据集大小的列表, e.g., [23384, 6429]
        :param batch_size: 总的批次大小, e.g., 128
        :param batch_ratio: 一个包含每个数据集中【每个批次应包含的样本数】的列表, e.g., [108, 20]
        """
        self.dataset_sizes = dataset_sizes
        self.batch_size = batch_size
        
        # --- [核心修复] 直接使用 batch_ratio 作为绝对数量 ---
        self.num_samples_per_dataset = np.array(batch_ratio, dtype=int)
        
        # --- [增加] 健壮性检查 ---
        if self.num_samples_per_dataset.sum() != self.batch_size:
            raise ValueError(
                f"The sum of BATCH_RATIO ({self.num_samples_per_dataset.sum()}) "
                f"must be equal to the BATCH_SIZE ({self.batch_size})."
            )
        if (self.num_samples_per_dataset <= 0).any():
             raise ValueError("All values in BATCH_RATIO must be positive.")
        
        self.total_samples = sum(dataset_sizes)

        # 索引范围的计算保持不变
        self.ranges = []
        start = 0
        for size in dataset_sizes:
            self.ranges.append(range(start, start + size))
            start += size
            
        logger.info(f"MixedBatchSampler initialized with dataset sizes {dataset_sizes}, "
                    f"batch size {batch_size}, and a fixed batch composition of {self.num_samples_per_dataset}.")

    def __iter__(self):
        # 1. 为每个数据集生成打乱的索引列表
        shuffled_indices_pools = [np.random.permutation(r) for r in self.ranges]
        # 2. 为每个索引池创建一个迭代器
        pool_iterators = [iter(pool) for pool in shuffled_indices_pools]

        # 3. [优化] 计算总批次数时，应基于【瓶颈】数据集
        # 找到哪个数据集会最先耗尽
        num_samples_per_epoch = np.array([len(r) for r in self.ranges])
        batches_per_epoch_per_dataset = num_samples_per_epoch / self.num_samples_per_dataset
        # 以最少的那个为准，确保每个 epoch 内不会过度重复采样
        num_batches = int(np.floor(batches_per_epoch_per_dataset.min()))

        for _ in range(num_batches):
            batch_indices = []
            # self.num_samples_per_dataset 已经是在 __init__ 中计算好的 [108, 20]
            for i, num_samples in enumerate(self.num_samples_per_dataset):
                for _ in range(num_samples):
                    try:
                        batch_indices.append(next(pool_iterators[i]))
                    except StopIteration:
                        # 索引耗尽的逻辑保持不变，这是正确的
                        shuffled_indices_pools[i] = np.random.permutation(self.ranges[i])
                        pool_iterators[i] = iter(shuffled_indices_pools[i])
                        batch_indices.append(next(pool_iterators[i]))
            
            # 打乱最终的 batch 索引，避免数据总是以 [HML, HML, ..., Style, Style] 的顺序出现
            np.random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        # [优化] __len__ 应该与 __iter__ 的行为一致
        num_samples_per_epoch = np.array([len(r) for r in self.ranges])
        batches_per_epoch_per_dataset = num_samples_per_epoch / self.num_samples_per_dataset
        return int(np.floor(batches_per_epoch_per_dataset.min()))
    
def mixed_collate_fn(batch):
    """
    一个 collate 函数，它能处理来自 ConcatDataset 的、已经由我们各自的 Dataset
    处理好的、格式完全统一的样本。
    由于我们已经付出了巨大努力统一了 __getitem__ 的输出格式，
    这个函数现在变得非常简单：它只需要直接调用原始的 mld_collate 即可。
    我们保留这个函数是为了逻辑上的清晰和未来的可扩展性。
    """
    # 我们的 __getitem__ 返回的是一个元组。
    # 我们需要确保传入 mld_collate 的是这个元组的列表。
    # 在这个阶段，我们不需要再做任何特殊的处理了。
    
    # 原始的 mld_collate 期望一个元组列表，这正是 DataLoader 传给我们的。
    return mld_collate(batch)