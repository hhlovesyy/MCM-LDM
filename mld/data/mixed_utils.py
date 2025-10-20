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
    一个自定义的 Sampler，用于从多个数据集中按固定比例采样，以创建混合批次。
    这对于处理数据不平衡问题至关重要。
    """
    '''
    目的: 它的唯一目的就是对抗数据不平衡。通过严格按照 ratio 配置，它确保了在每个 batch 中，来自 100Style 的“新知识题”都有稳定且足够的出场机会，不会被 HumanML3D 的“复习题”所淹没。
    '''
    def __init__(self, dataset_sizes, batch_size, ratio):
        """
        初始化 Sampler。
        :param dataset_sizes: 一个包含每个数据集大小的列表, e.g., [23384, 6429]
        :param batch_size: 总的批次大小, e.g., 128
        :param ratio: 一个包含每个数据集采样比例的列表, e.g., [4, 1]
        """
        self.dataset_sizes = dataset_sizes
        self.batch_size = batch_size
        
        # 将比例转换为归一化的概率分布
        self.ratio = np.array(ratio)
        self.ratio = self.ratio / self.ratio.sum()
        
        self.total_samples = sum(dataset_sizes)

        # 计算每个数据集的索引范围
        # e.g., dataset1: range(0, 23384), dataset2: range(23384, 23384 + 6429)
        self.ranges = []
        start = 0
        for size in dataset_sizes:
            self.ranges.append(range(start, start + size))
            start += size
            
        logger.info(f"MixedBatchSampler initialized with dataset sizes {dataset_sizes}, "
                    f"batch size {batch_size}, and normalized ratio {self.ratio}.")

    def __iter__(self):
        # $\mathbf{\_\_iter\_\_}$ 函數的目標是生成一個迭代器 (Iterator)，該迭代器按順序吐出訓練數據集中的索引 (Indices)，但這些索引是以批次 (Batch) 的形式組織的。
        # 1. 计算每个 batch 中来自每个数据集的样本数
        num_samples_per_dataset = (self.ratio * self.batch_size).round().astype(int)

        # 2. 微调以确保总数严格等于 batch_size
        diff = self.batch_size - num_samples_per_dataset.sum()
        # 将差异加到样本数最多的那个数据集上，以保持比例最稳定
        num_samples_per_dataset[np.argmax(num_samples_per_dataset)] += diff

        if (num_samples_per_dataset <= 0).any():
             raise ValueError("Batch size is too small for the given ratio, "
                              "resulting in 0 samples for some datasets.")
        logger.debug(f"Samples per batch from each dataset: {num_samples_per_dataset}")

        '''
        在开始一个 Epoch 之前，它会为每个数据集（通过 self.ranges 确定索引范围）独立地生成一个完全打乱的索引列表。然后，将这些列表转换为迭代器，这样每次取索引时都非常高效。
        '''
        # 3. 为每个数据集生成打乱的索引列表
        shuffled_indices_pools = [np.random.permutation(r) for r in self.ranges]
        # 4. 为每个索引池创建一个迭代器
        pool_iterators = [iter(pool) for pool in shuffled_indices_pools]

        # 5. 计算总共可以生成多少个完整的 batch
        num_batches = self.total_samples // self.batch_size

        for _ in range(num_batches):
            batch_indices = []
            for i, num_samples in enumerate(num_samples_per_dataset):
                for _ in range(num_samples):
                    try:
                        # 从对应的迭代器中取出一个索引
                        batch_indices.append(next(pool_iterators[i]))
                    except StopIteration:
                        # 如果某个数据集的索引耗尽，重新打乱该数据集的索引并创建新的迭代器
                        logger.debug(f"Replenishing pool for dataset {i}")
                        shuffled_indices_pools[i] = np.random.permutation(self.ranges[i])
                        pool_iterators[i] = iter(shuffled_indices_pools[i])
                        batch_indices.append(next(pool_iterators[i]))
            
            # yield 一个完整的、混合好的 batch 索引列表
            yield batch_indices

    def __len__(self):
        return self.total_samples // self.batch_size
    
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