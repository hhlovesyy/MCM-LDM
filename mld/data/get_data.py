from os.path import join as pjoin

import numpy as np
from .humanml.utils.word_vectorizer import WordVectorizer
from .HumanML3D import HumanML3DDataModule
from .utils import *


def get_mean_std(phase, cfg, dataset_name):
    # if phase == 'gt':
    #     # used by T2M models (including evaluators)
    #     mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
    #     std = np.load(pjoin(opt.meta_dir, 'std.npy'))
    # elif phase in ['train', 'val', 'text_only']:
    #     # used by our models
    #     mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    #     std = np.load(pjoin(opt.data_root, 'Std.npy'))

    # todo: use different mean and val for phases
    name = "t2m" if dataset_name == "humanml3d" else dataset_name
    assert name in ["t2m", "kit"]
    # if phase in ["train", "val", "test"]:
    if phase in ["val"]:
        if name == 't2m':
            data_root = pjoin(cfg.model.t2m_path, name, "Comp_v6_KLD01",
                              "meta")
        elif name == 'kit':
            data_root = pjoin(cfg.model.t2m_path, name, "Comp_v6_KLD005",
                              "meta")
        else:
            raise ValueError("Only support t2m and kit")
        mean = np.load(pjoin(data_root, "mean.npy"))
        std = np.load(pjoin(data_root, "std.npy"))
    else:
        data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
        mean = np.load(pjoin(data_root, "Mean.npy"))
        std = np.load(pjoin(data_root, "Std.npy"))

    return mean, std


def get_WordVectorizer(cfg, phase, dataset_name):
    if phase not in ["text_only"]:
        if dataset_name.lower() in ["humanml3d", "kit"]:
            return WordVectorizer(cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")
        else:
            raise ValueError("Only support WordVectorizer for HumanML3D")
    else:
        return None


def get_collate_fn(name, phase="train"):
    if name.lower() in ["humanml3d", "kit"]:
        return mld_collate
    elif name.lower() in ["humanact12", 'uestc']:
        return a2m_collate
    # else:
    #     return all_collate
    # if phase == "test":
    #     return eval_collate
    # else:


# map config name to module&path
dataset_module_map = {
    "humanml3d": HumanML3DDataModule,
}
motion_subdir = {"humanml3d": "new_joint_vecs", "kit": "new_joint_vecs"}


def get_datasets(cfg, logger=None, phase="train"):
    """
    工厂函数，根据配置实例化并返回一个或多个 DataModule。
    为混合训练任务增加了专门的处理逻辑。
    """
    
    log_func = logger.info if logger else print
    
    # --- [核心修复] ---
    # 只要 config 中指定了 TYPE: 'mixed'，我们就应该使用 MixedDataModule，
    # 无论是在训练 (train) 还是在演示/测试 (test) 阶段。
    if cfg.DATASET.get('TYPE', 'single') == 'mixed':
        from .mixed_datamodule import MixedDataModule
        
        log_func(f"Configuration specifies 'mixed' type. Initializing MixedDataModule for phase: '{phase}'.")
        
        # 实例化我们新的 DataModule
        datamodule = MixedDataModule(cfg)
        
        # [关键] 手动调用 setup 来提前获取 nfeats 和 norms
        # Demo 阶段也需要这个步骤
        log_func("Running manual setup on MixedDataModule to fetch metadata...")
        datamodule.setup(stage=phase) # 使用当前的 phase
        
        # 更新全局配置
        if datamodule.nfeats is None:
            raise ValueError("MixedDataModule.nfeats was not set after setup().")
        cfg.DATASET.NFEATS = datamodule.nfeats
        cfg.DATASET.NJOINTS = datamodule.njoints
        
        log_func(f"Configuration updated: NFEATS={cfg.DATASET.NFEATS}")
        
        return [datamodule]
    
    # --- 如果不使用 MixedDataModule，则执行原始逻辑 ---
    log_func(f"Initializing with standard single-dataset datamodules for phase: '{phase}'.")
    
    dataset_names = eval(f"cfg.{phase.upper()}.DATASETS")
    datasets = []
    # ( ... 原始的 for 循环逻辑完全保持不变 ... )
    for dataset_name in dataset_names:
        if dataset_name.lower() in ["humanml3d", "kit"]:
            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            mean, std = get_mean_std(phase, cfg, dataset_name)
            mean_eval, std_eval = get_mean_std("val", cfg, dataset_name)
            wordVectorizer = get_WordVectorizer(cfg, phase, dataset_name)
            collate_fn = get_collate_fn(dataset_name, phase)
            
            dataset = dataset_module_map[dataset_name.lower()](
                cfg=cfg,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                debug=cfg.DEBUG,
                collate_fn=collate_fn,
                mean=mean,
                std=std,
                mean_eval=mean_eval,
                std_eval=std_eval,
                w_vectorizer=wordVectorizer,
                text_dir=pjoin(data_root, "texts"),
                motion_dir=pjoin(data_root, motion_subdir[dataset_name]),
                # ... 其他参数 ...
                max_motion_length=cfg.DATASET.SAMPLER.MAX_LEN,
                min_motion_length=cfg.DATASET.SAMPLER.MIN_LEN,
                max_text_len=cfg.DATASET.SAMPLER.MAX_TEXT_LEN,
                unit_length=eval(f"cfg.DATASET.{dataset_name.upper()}.UNIT_LEN"),
            )
            datasets.append(dataset)
        else:
            raise NotImplementedError
            
    # 更新全局配置
    cfg.DATASET.NFEATS = datasets[0].nfeats
    cfg.DATASET.NJOINTS = datasets[0].njoints
    return datasets
