import torch
import torch.utils.data
import numpy as np
import logging
from .utils import mld_collate 
from .utils import collate_tensors

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
        self.num_samples_per_dataset = np.array(batch_ratio, dtype=int)
        
        if self.num_samples_per_dataset.sum() != self.batch_size:
            raise ValueError(
                f"The sum of BATCH_RATIO ({self.num_samples_per_dataset.sum()}) "
                f"must be equal to the BATCH_SIZE ({self.batch_size})."
            )
        if (self.num_samples_per_dataset <= 0).any():
             raise ValueError("All values in BATCH_RATIO must be positive.")
        
        self.total_samples = sum(dataset_sizes)

        self.ranges = []
        start = 0
        for size in dataset_sizes:
            self.ranges.append(range(start, start + size))
            start += size
            
        logger.info(f"MixedBatchSampler initialized with dataset sizes {dataset_sizes}, "
                    f"batch size {batch_size}, and a fixed batch composition of {self.num_samples_per_dataset}.")

    def __iter__(self):
        shuffled_indices_pools = [np.random.permutation(r) for r in self.ranges]
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
                    except StopIteration: # 耗尽了，重新打乱并创建新的迭代器
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
    一个智能的 collate 函数，专门用于处理混合了 'humanml3d' 和 'style100' 数据源的批次。
    它会：
    1. 将不同来源的样本分离开。
    2. 对每个子集分别使用原始的 mld_collate 进行处理。
    3. 将结果合并，并创建一个新的 'is_text_guided' 张量来标记每个样本的类型。
    """
    # 1. 根据 'source' 标志将样本分流
    humanml3d_batch = [b for b in batch if b['source'] == 'humanml3d']
    style100_batch = [b for b in batch if b['source'] == 'style100']

    final_batch = {}
    
    # 2. 分别处理，如果某个子集不为空
    if humanml3d_batch:
        collated_hml = mld_collate_dict(humanml3d_batch) # 使用新的字典版 collate
    if style100_batch:
        collated_s100 = mld_collate_dict(style100_batch) # 使用新的字典版 collate

    # 3. 合并结果
    if humanml3d_batch and style100_batch:
        # 如果两个都有，合并它们
        for key in collated_hml.keys():
            if isinstance(collated_hml[key], torch.Tensor):
                final_batch[key] = torch.cat([collated_hml[key], collated_s100[key]], dim=0)
            elif isinstance(collated_hml[key], list):
                final_batch[key] = collated_hml[key] + collated_s100[key]
        
        # 4. [核心] 创建区分标志
        is_text_guided = torch.cat([
            torch.zeros(len(humanml3d_batch), dtype=torch.bool), # HumanML3D 样本为 False
            torch.ones(len(style100_batch), dtype=torch.bool)    # 100Style 样本为 True
        ], dim=0)

    elif humanml3d_batch:
        # 如果只有 HumanML3D
        final_batch = collated_hml
        is_text_guided = torch.zeros(len(humanml3d_batch), dtype=torch.bool)
    
    elif style100_batch:
        final_batch = collated_s100
        is_text_guided = torch.ones(len(style100_batch), dtype=torch.bool)
    else:
        # 如果 batch 为空
        return {} # 返回空字典

    final_batch['is_text_guided'] = is_text_guided
    
    logger.debug(f"Mixed collate created a batch. Text-guided samples: {is_text_guided.sum()}/{len(is_text_guided)}")

    return final_batch


def mld_collate_dict(batch):
    """
    mld_collate 的一个适配器版本，用于处理字典列表而不是元组列表。
    """
    # 按照句子长度排序 (如果需要的话，原始 mld_collate 做了这件事)
    batch.sort(key=lambda x: x['sent_len'], reverse=True)
    
    # 使用与 mld_collate 相同的逻辑，但从字典中按 key 取值
    return {
        "motion": collate_tensors([torch.tensor(b['motion']).float() for b in batch]),
        "text": [b['caption'] for b in batch],
        "length": [b['m_length'] for b in batch],
        "word_embs": collate_tensors([torch.tensor(b['word_embeddings']).float() for b in batch]),
        "pos_ohot": collate_tensors([torch.tensor(b['pos_one_hots']).float() for b in batch]),
        "text_len": collate_tensors([torch.tensor(b['sent_len']) for b in batch]),
        "tokens": [b['tokens'] for b in batch],
        "style_id": [b['style_id'] for b in batch],
        "source": [b['source'] for b in batch],  # 保留来源信息
    }