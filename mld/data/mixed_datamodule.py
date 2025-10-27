import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from os.path import join as pjoin
import logging
import os
import torch
from omegaconf import OmegaConf

from .humanml.data.dataset import Text2MotionDatasetV2
from .style100_dataset import Style100Dataset
from .mixed_utils import MixedBatchSampler, mixed_collate_fn, mld_collate_dict
from .utils import mld_collate
from .humanml.scripts.motion_process import recover_from_ric, extract_features

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) 

class MixedDataModule(pl.LightningDataModule):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.val_batch_size = cfg.EVAL.BATCH_SIZE 
        self.num_workers = cfg.TRAIN.NUM_WORKERS

        # print(f"\n[DIAGNOSTIC TRACE] 1. Initializing MixedDataModule instance with id: {id(self)}")
        self.w_vectorizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.nfeats = None # 将在 setup 中被正确赋值
        self.njoints = 22

        self.is_mm = False
        
        # 3. 创建一个普通的字典来存储归一化等参数
        self.norms = {}

        logger.info("MixedDataModule initialized.")

    def prepare_data(self): # 保留的用于数据下载等操作的hook
        pass

    def setup(self, stage=None):
        # 这个方法会在每个 GPU 进程中被调用，用于创建数据集
        print(f"\n[DIAGNOSTIC TRACE] 2. Entering 'setup' method for instance {id(self)} with stage: '{stage}'")
        
        # 防止重复执行
        if self.train_dataset is not None and self.val_dataset is not None:
            print(f"[DIAGNOSTIC TRACE] 2a. Datasets already exist. Skipping setup.")
            return

        # 懒加载 WordVectorizer
        logger.info(f"Loading WordVectorizer from: {self.cfg.DATASET.WORD_VERTILIZER_PATH}")
        from .humanml.utils.word_vectorizer import WordVectorizer
        if not os.path.exists(self.cfg.DATASET.WORD_VERTILIZER_PATH):
            raise FileNotFoundError(f"Word vectorizer path not found: {self.cfg.DATASET.WORD_VERTILIZER_PATH}")
        self.w_vectorizer = WordVectorizer(self.cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")

        # 6. 加载通用的归一化参数，目前都使用HumanML3D的mean和std做归一化
        humanml3d_cfg = self.cfg.DATASET.HUMANML3D
        self.norms['mean'] = np.load(pjoin(humanml3d_cfg.ROOT, "Mean.npy"))
        self.norms['std'] = np.load(pjoin(humanml3d_cfg.ROOT, "Std.npy"))
        
        t2m_meta_path = pjoin(self.cfg.model.t2m_path, "t2m", "Comp_v6_KLD01", "meta")  # 【QUESTION】这个评估一般是什么？现阶段是不是还用不到，不过你可以展开简单讲一讲
        self.norms['mean_eval'] = np.load(pjoin(t2m_meta_path, "mean.npy"))
        self.norms['std_eval'] = np.load(pjoin(t2m_meta_path, "std.npy"))
        
        # 7. 实例化所有需要的数据集
        
        # HumanML3D 训练集
        logger.info("--- Setting up HumanML3D training dataset ---")
        humanml3d_train = Text2MotionDatasetV2(
            mean=self.norms['mean'], std=self.norms['std'],
            split_file=pjoin(humanml3d_cfg.ROOT, 'train.txt'),
            motion_dir=pjoin(humanml3d_cfg.ROOT, 'new_joint_vecs'),
            text_dir=pjoin(humanml3d_cfg.ROOT, 'texts'),
            w_vectorizer=self.w_vectorizer,
            tiny=self.cfg.DEBUG, debug=self.cfg.DEBUG,
            max_motion_length=humanml3d_cfg.SAMPLER.MAX_LEN, min_motion_length=humanml3d_cfg.SAMPLER.MIN_LEN,
            max_text_len=humanml3d_cfg.SAMPLER.MAX_TEXT_LEN, unit_length=humanml3d_cfg.UNIT_LEN,
            min_filter_length=humanml3d_cfg.MIN_FILTER_LEN, max_filter_length=humanml3d_cfg.MAX_FILTER_LEN
        )

        logger.info("--- Setting up 100Style training dataset ---")
        style100_cfg = self.cfg.DATASET.STYLE100
        style100_train = Style100Dataset(
            mean=self.norms['mean'], std=self.norms['std'],
            split_file=pjoin(style100_cfg.ROOT, 'train.txt'),
            motion_dir=pjoin(style100_cfg.ROOT, 'new_joint_vecs'),
            text_dir=pjoin(style100_cfg.ROOT, 'texts'),
            style_dict_path=pjoin(style100_cfg.ROOT, 'Style_name_dict.txt'),
            w_vectorizer=self.w_vectorizer,
            tiny=self.cfg.DEBUG, debug=self.cfg.DEBUG,
            max_motion_length=style100_cfg.SAMPLER.MAX_LEN, min_motion_length=style100_cfg.SAMPLER.MIN_LEN,
            max_text_len=style100_cfg.SAMPLER.MAX_TEXT_LEN, unit_length=style100_cfg.UNIT_LEN,
            min_filter_length=style100_cfg.MIN_FILTER_LEN, max_filter_length=style100_cfg.MAX_FILTER_LEN
        )

        # HumanML3D 验证集，纯粹的只有HumanML3D的，不包含100Style，先留在这里，后面可能用得上
        
        # logger.info("--- Setting up HumanML3D validation dataset ---")
        # self.val_dataset = Text2MotionDatasetV2(
        #     mean=self.norms['mean'], std=self.norms['std'],
        #     split_file=pjoin(humanml3d_cfg.ROOT, 'val.txt'),
        #     motion_dir=pjoin(humanml3d_cfg.ROOT, 'new_joint_vecs'),
        #     text_dir=pjoin(humanml3d_cfg.ROOT, 'texts'),
        #     w_vectorizer=self.w_vectorizer,
        #     tiny=self.cfg.DEBUG, debug=self.cfg.DEBUG,
        #     max_motion_length=humanml3d_cfg.SAMPLER.MAX_LEN, min_motion_length=humanml3d_cfg.SAMPLER.MIN_LEN,
        #     max_text_len=humanml3d_cfg.SAMPLER.MAX_TEXT_LEN, unit_length=humanml3d_cfg.UNIT_LEN,
        #     min_filter_length=humanml3d_cfg.MIN_FILTER_LEN, max_filter_length=humanml3d_cfg.MAX_FILTER_LEN
        # )

        logger.info("--- Setting up HumanML3D validation dataset ---")
        humanml3d_val = Text2MotionDatasetV2( 
            mean=self.norms['mean'], 
            std=self.norms['std'], 
            split_file=pjoin(humanml3d_cfg.ROOT, 'val.txt'), 
            motion_dir=pjoin(humanml3d_cfg.ROOT, 'new_joint_vecs'), 
            text_dir=pjoin(humanml3d_cfg.ROOT, 'texts'), 
            w_vectorizer=self.w_vectorizer, 
            tiny=self.cfg.DEBUG, debug=self.cfg.DEBUG, 
            max_motion_length=humanml3d_cfg.SAMPLER.MAX_LEN, min_motion_length=humanml3d_cfg.SAMPLER.MIN_LEN, 
            max_text_len=humanml3d_cfg.SAMPLER.MAX_TEXT_LEN, unit_length=humanml3d_cfg.UNIT_LEN, 
            min_filter_length=humanml3d_cfg.MIN_FILTER_LEN, max_filter_length=humanml3d_cfg.MAX_FILTER_LEN)

        logger.info("--- Setting up 100Style validation dataset ---")
        style100_val_split_file = pjoin(style100_cfg.ROOT, 'val.txt')
        if not os.path.exists(style100_val_split_file):
             # 如果沒有驗證集，我們可以重用測試集或一小部分訓練集作為替代，打印警告，查了一下应该是有验证集的
             logger.warning(f"Validation split '{style100_val_split_file}' not found for 100Style.")
             logger.warning("Falling back to use the 'test.txt' split for validation.")
             style100_val_split_file = pjoin(style100_cfg.ROOT, 'test.txt')

        style100_val = Style100Dataset( 
            mean=self.norms['mean'], 
            std=self.norms['std'], 
            split_file=style100_val_split_file, 
            motion_dir=pjoin(style100_cfg.ROOT, 'new_joint_vecs'), 
            text_dir=pjoin(style100_cfg.ROOT, 'texts'), 
            style_dict_path=pjoin(style100_cfg.ROOT, 'Style_name_dict.txt'), 
            w_vectorizer=self.w_vectorizer, 
            tiny=self.cfg.DEBUG, debug=self.cfg.DEBUG, 
            max_motion_length=style100_cfg.SAMPLER.MAX_LEN, min_motion_length=style100_cfg.SAMPLER.MIN_LEN, 
            max_text_len=style100_cfg.SAMPLER.MAX_TEXT_LEN, unit_length=style100_cfg.UNIT_LEN, 
            min_filter_length=style100_cfg.MIN_FILTER_LEN, max_filter_length=style100_cfg.MAX_FILTER_LEN)
        
        self.val_dataset = ConcatDataset([humanml3d_val, style100_val])
        print(f"[DIAGNOSTIC TRACE] 2c. 'val_dataset' instantiated. Length: {len(self.val_dataset)}")

        # 8. 组合训练集并更新元数据
        self.train_dataset = ConcatDataset([humanml3d_train, style100_train])
        self.nfeats = humanml3d_train.nfeats

        logger.info("--- MixedDataModule setup complete ---")
        logger.info(f"Feature dimension (nfeats) set to: {self.nfeats}")
        logger.info(f"Total concatenated train size: {len(self.train_dataset)}")
        logger.info(f"HumanML3D validation size: {len(self.val_dataset)}")
        print(f"[DIAGNOSTIC TRACE] 2d. 'setup' method finished for instance {id(self)}.")

    def train_dataloader(self):
        logger.info("Creating train dataloader with MixedBatchSampler...")
        
        dataset_sizes = [len(self.train_dataset.datasets[0]), len(self.train_dataset.datasets[1])]
        batch_ratio = self.cfg.DATASET.MIXED.BATCH_RATIO
        
        sampler = MixedBatchSampler(
            dataset_sizes=dataset_sizes, 
            batch_size=self.batch_size, 
            batch_ratio=batch_ratio
        )
        
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=mixed_collate_fn,
            pin_memory=True
        )

    def val_dataloader(self):
        # 【QUESTION】目前我们的验证阶段的数据集也改成了混合数据集，请你确认这是正确的，符合我们当前阶段目标的

        # logger.info("Creating validation dataloaloader from HumanML3D val set...")
        # # setup() 会在 trainer.validate() 开始时被自动调用
        
        # batch_size = min(self.val_batch_size, len(self.val_dataset))
        # if batch_size == 0:
        #     logger.warning("Validation dataset is empty, returning empty dataloader.")
        #     return DataLoader([])
            
        # return DataLoader(
        #     self.val_dataset,
        #     batch_size=batch_size,
        #     shuffle=False,
        #     num_workers=self.num_workers,
        #     # --- [核心修复 3] 验证集也应该使用同样的 collate 逻辑 ---
        #     collate_fn=mld_collate_dict,
        #     pin_memory=True
        # )
        logger.info("Creating MIXED validation dataloader with MixedBatchSampler...") 
        val_dataset_sizes = [len(d) for d in self.val_dataset.datasets]
        if any(size == 0 for size in val_dataset_sizes):
            logger.warning("One of the validation datasets is empty. Returning an empty dataloader.")
            return DataLoader([])

        val_sampler = MixedBatchSampler(
            dataset_sizes=val_dataset_sizes, 
            batch_size=self.val_batch_size, # 使用驗證批次大小
            batch_ratio=self.cfg.DATASET.MIXED.BATCH_RATIO # 使用與訓練時相同的混合比例
        )
        
        return DataLoader(
            self.val_dataset,
            batch_sampler=val_sampler,
            num_workers=self.num_workers,
            collate_fn=mixed_collate_fn, 
            pin_memory=True
        )

    def test_dataloader(self):
        logger.info("Test dataloader not implemented yet.")
        return []

    # --- 辅助函数区 (移植自 HumanML3DDataModule) ---
    def feats2joints(self, features):
        mean = torch.tensor(self.norms['mean'], device=features.device, dtype=features.dtype)
        std = torch.tensor(self.norms['std'], device=features.device, dtype=features.dtype)
        features = features * std + mean
        return recover_from_ric(features, self.njoints)

    def joints2feats(self, features):
        feature_list = []
        for i in range(features.shape[0]):
            feature = extract_features(features[i, ...].cpu().numpy())
            feature_list.append(feature)
        features_np = np.array(feature_list)
        features_np = (features_np - self.norms['mean']) / self.norms['std']
        return torch.from_numpy(features_np).to(features.device, dtype=features.dtype)

    def renorm4t2m(self, features):
        ori_mean = torch.tensor(self.norms['mean'], device=features.device, dtype=features.dtype)
        ori_std = torch.tensor(self.norms['std'], device=features.device, dtype=features.dtype)
        eval_mean = torch.tensor(self.norms['mean_eval'], device=features.device, dtype=features.dtype)
        eval_std = torch.tensor(self.norms['std_eval'], device=features.device, dtype=features.dtype)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features