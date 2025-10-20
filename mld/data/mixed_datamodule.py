# mld/data/mixed_datamodule.py

import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from os.path import join as pjoin
import logging
import os

# 导入我们所有的数据层组件
from .humanml.data.dataset import Text2MotionDatasetV2
from .style100_dataset import Style100Dataset
from .mixed_utils import MixedBatchSampler, mixed_collate_fn # [关键] 导入我们新建的工具
from .utils import mld_collate # 导入原始 collate，以备验证/测试时使用

# 初始化日志记录器
logger = logging.getLogger(__name__)
# 确保日志能被看到
logging.basicConfig(level=logging.INFO) 

class MixedDataModule(pl.LightningDataModule):
    """
    一个 PyTorch Lightning DataModule，专门用于混合 HumanML3D 和 100Style 数据集。
    它使用 MixedBatchSampler 来确保按比例采样，以应对数据不平衡问题。
    """
    def __init__(self, cfg, **kwargs):
        super().__init__()
        # 保存整个配置树，方便在各个方法中按需取用
        self.cfg = cfg
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.val_batch_size = cfg.EVAL.BATCH_SIZE 
        self.num_workers = cfg.TRAIN.NUM_WORKERS

        self.w_vectorizer = None # WordVectorizer 将在 setup 阶段被懒加载
        
        # 初始化元数据，将在 setup 中被正确赋值
        self.nfeats = 263  # 这个应该是263，不是0吧
        self.njoints = 22 # 两个数据集的关节点数量是相同的

        logger.info("MixedDataModule initialized. Datasets will be set up soon.")
        # [单元测试思路] 可以在这里测试 __init__ 是否成功保存了 batch_size 等参数。
    
    def setup(self, stage=None):
        """
        PyTorch Lightning 在 .fit() 或 .test() 之前会自动调用此方法。
        这是实例化数据集(Dataset)的最佳位置。
        """
        # 懒加载 WordVectorizer，避免在非必要时加载
        if self.w_vectorizer is None:
             logger.info(f"Loading WordVectorizer from: {self.cfg.DATASET.WORD_VERTILIZER_PATH}")
             from .humanml.utils.word_vectorizer import WordVectorizer
             if not os.path.exists(self.cfg.DATASET.WORD_VERTILIZER_PATH):
                 raise FileNotFoundError(f"Word vectorizer path not found: {self.cfg.DATASET.WORD_VERTILIZER_PATH}")
             self.w_vectorizer = WordVectorizer(self.cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")

        # 在 stage is None 或者 'fit' 的时候，我们需要同时准备 train 和 val
        if stage is None or stage == 'fit':
            # --- 实例化 HumanML3D 训练集 ---
            logger.info("--- Setting up HumanML3D training dataset ---")
            humanml3d_cfg = self.cfg.DATASET.HUMANML3D
            humanml3d_mean = np.load(pjoin(humanml3d_cfg.ROOT, "Mean.npy"))
            humanml3d_std = np.load(pjoin(humanml3d_cfg.ROOT, "Std.npy"))

            self.humanml3d_train = Text2MotionDatasetV2(
                mean=humanml3d_mean, std=humanml3d_std,
                split_file=pjoin(humanml3d_cfg.ROOT, 'train.txt'),
                motion_dir=pjoin(humanml3d_cfg.ROOT, 'new_joint_vecs'),
                text_dir=pjoin(humanml3d_cfg.ROOT, 'texts'),
                w_vectorizer=self.w_vectorizer,
                tiny=self.cfg.DEBUG, debug=self.cfg.DEBUG,
                max_motion_length=humanml3d_cfg.SAMPLER.MAX_LEN,
                min_motion_length=humanml3d_cfg.SAMPLER.MIN_LEN,
                max_text_len=humanml3d_cfg.SAMPLER.MAX_TEXT_LEN,
                unit_length=humanml3d_cfg.UNIT_LEN,
                min_filter_length=humanml3d_cfg.MIN_FILTER_LEN,
                max_filter_length=humanml3d_cfg.MAX_FILTER_LEN,
            )

            # --- 实例化 100Style 训练集 ---
            logger.info("--- Setting up 100Style training dataset ---")
            style100_cfg = self.cfg.DATASET.STYLE100
            style100_mean = np.load(pjoin(style100_cfg.ROOT, "Mean.npy"))
            style100_std = np.load(pjoin(style100_cfg.ROOT, "Std.npy"))

            self.style100_train = Style100Dataset(
                mean=style100_mean, std=style100_std,
                split_file=pjoin(style100_cfg.ROOT, 'train.txt'),
                motion_dir=pjoin(style100_cfg.ROOT, 'new_joint_vecs'),
                text_dir=pjoin(style100_cfg.ROOT, 'texts'),
                style_dict_path=pjoin(style100_cfg.ROOT, 'Style_name_dict.txt'),
                w_vectorizer=self.w_vectorizer,
                tiny=self.cfg.DEBUG, debug=self.cfg.DEBUG,
                max_motion_length=style100_cfg.SAMPLER.MAX_LEN,
                min_motion_length=style100_cfg.SAMPLER.MIN_LEN,
                max_text_len=style100_cfg.SAMPLER.MAX_TEXT_LEN,
                unit_length=style100_cfg.UNIT_LEN,
                min_filter_length=style100_cfg.MIN_FILTER_LEN,
                max_filter_length=style100_cfg.MAX_FILTER_LEN,
            )

            self.nfeats = self.humanml3d_train.nfeats
            self.train_dataset = ConcatDataset([self.humanml3d_train, self.style100_train])
            
            logger.info("--- Setting up VALIDATION dataset (HumanML3D only) ---")
            # [新增] 创建 HumanML3D 的验证集实例, 防止模型忘本
            self.val_dataset = Text2MotionDatasetV2(
                mean=humanml3d_mean, std=humanml3d_std,
                split_file=pjoin(humanml3d_cfg.ROOT, 'val.txt'), # [关键] 使用 val.txt
                motion_dir=pjoin(humanml3d_cfg.ROOT, 'new_joint_vecs'),
                text_dir=pjoin(humanml3d_cfg.ROOT, 'texts'),
                w_vectorizer=self.w_vectorizer,
                tiny=self.cfg.DEBUG, debug=self.cfg.DEBUG,
                # 使用与训练集相同的参数
                max_motion_length=humanml3d_cfg.SAMPLER.MAX_LEN,
                min_motion_length=humanml3d_cfg.SAMPLER.MIN_LEN,
                max_text_len=humanml3d_cfg.SAMPLER.MAX_TEXT_LEN,
                unit_length=humanml3d_cfg.UNIT_LEN,
                min_filter_length=humanml3d_cfg.MIN_FILTER_LEN,
                max_filter_length=humanml3d_cfg.MAX_FILTER_LEN,
            )
            logger.info(f"HumanML3D validation size: {len(self.val_dataset)}")


    def train_dataloader(self):
        logger.info("Creating train dataloader with MixedBatchSampler.")
        dataset_sizes = [len(self.humanml3d_train), len(self.style100_train)]
        ratio = self.cfg.DATASET.MIXED.BATCH_RATIO
        sampler = MixedBatchSampler(dataset_sizes=dataset_sizes, batch_size=self.batch_size, ratio=ratio)
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=mixed_collate_fn,
            pin_memory=True
        )

    def val_dataloader(self):
        """
        [新增] 创建并返回验证数据加载器。
        """
        logger.info("Creating validation dataloader (from HumanML3D val set).")
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size, # 使用独立的验证 batch size
            shuffle=False, # 验证集不需要打乱
            num_workers=self.num_workers,
            collate_fn=mld_collate, # 直接使用原始的 collate 函数
            pin_memory=True,
            drop_last=True  # DataLoader 的行为: 当 DataLoader 的 batch_size（这里是 2）大于数据集的总样本数（这里是 1）时，并且 drop_last=False（这是 DataLoader 的默认行为），它仍然会产出一个 batch，但这个 batch 只会包含数据集中所有可用的样本。当然这也是val_dataset太小了的边界情况，正常应该不会
        )

    def test_dataloader(self):
        # 同样，我们可以为 test 实现类似 val 的逻辑
        logger.info("Test dataloader is not implemented yet.")
        return []