from os.path import join as pjoin

import numpy as np
from .humanml.utils.word_vectorizer import WordVectorizer
from .HumanML3D import HumanML3DDataModule
from .physimos_dataset import PhysiMoS100StyleDataset
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
    elif name.lower() in ["physimos100style"]:
        return physimos_collate
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
    # get dataset names form cfg
    dataset_names = eval(f"cfg.{phase.upper()}.DATASETS")
    datasets = []
    for dataset_name in dataset_names:
        if dataset_name.lower() in ["humanml3d", "kit"]:
            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            # get mean and std corresponding to dataset
            mean, std = get_mean_std(phase, cfg, dataset_name)
            mean_eval, std_eval = get_mean_std("val", cfg, dataset_name)
            # get WordVectorizer
            wordVectorizer = get_WordVectorizer(cfg, phase, dataset_name)
            # get collect_fn
            collate_fn = get_collate_fn(dataset_name, phase)
            # get dataset module
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
                style_text_dir=pjoin(data_root, "texts"),
                motion_dir=pjoin(data_root, motion_subdir[dataset_name]),
                max_motion_length=cfg.DATASET.SAMPLER.MAX_LEN,
                min_motion_length=cfg.DATASET.SAMPLER.MIN_LEN,
                max_text_len=cfg.DATASET.SAMPLER.MAX_TEXT_LEN,
                unit_length=eval(
                    f"cfg.DATASET.{dataset_name.upper()}.UNIT_LEN"),
            )
            datasets.append(dataset)
        elif dataset_name.lower() in ["humanact12", 'uestc']:
            # get collect_fn
            collate_fn = get_collate_fn(dataset_name, phase)
            # get dataset module
            dataset = dataset_module_map[dataset_name.lower()](
                datapath=eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT"),
                cfg=cfg,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                debug=cfg.DEBUG,
                collate_fn=collate_fn,
                num_frames=cfg.DATASET.HUMANACT12.NUM_FRAMES,
                sampling=cfg.DATASET.SAMPLER.SAMPLING,
                sampling_step=cfg.DATASET.SAMPLER.SAMPLING_STEP,
                pose_rep=cfg.DATASET.HUMANACT12.POSE_REP,
                max_len=cfg.DATASET.SAMPLER.MAX_LEN,
                min_len=cfg.DATASET.SAMPLER.MIN_LEN,
                num_seq_max=cfg.DATASET.SAMPLER.MAX_SQE
                if not cfg.DEBUG else 100,
                glob=cfg.DATASET.HUMANACT12.GLOB,
                translation=cfg.DATASET.HUMANACT12.TRANSLATION)
            cfg.DATASET.NCLASSES = dataset.nclasses
            datasets.append(dataset)
        elif dataset_name.lower() == "physimos100style":
            cfg_ds = cfg.DATASET.PHYSIMOS100STYLE # 从配置中读取我们数据集的专属设置
            data_root = cfg_ds.ROOT
            
            # get mean and std, 我们可以复用 humanml3d 的
            mean, std = get_mean_std(phase, cfg, "humanml3d")
            
            # get collect_fn
            collate_fn = get_collate_fn(dataset_name, phase)

            # Determine the correct split file to use
            if phase == 'train':
                split_file = pjoin(cfg.DATASET.SPLIT_DIR, 'train.txt')
            else: # for val/test
                split_file = pjoin(cfg.DATASET.SPLIT_DIR, 'test.txt')

            # Directly instantiate our custom Dataset
            our_dataset = PhysiMoS100StyleDataset(
                mean=mean,
                std=std,
                split_file=split_file,
                motion_dir=pjoin(data_root, "new_joint_vecs"),
                max_motion_length=cfg.DATASET.SAMPLER.MAX_LEN,
                min_motion_length=cfg.DATASET.SAMPLER.MIN_LEN,
                unit_length=cfg_ds.UNIT_LEN,
                style_dict_path=pjoin(data_root, "Style_name_dict.txt"),
                scene_mapping_path=cfg_ds.SCENE_MAPPING_PATH,
                # Use .get() for safety in case the key doesn't exist in the yaml
                style_subset=cfg_ds.get("STYLE_SUBSET", None) 
            )

            # A simple wrapper class to mimic the Pytorch Lightning DataModule interface
            # that train.py expects.
            class SimpleDataModule:
                def __init__(self, train_dataset, val_dataset, mean_val, std_val):
                    self.train_dataset = train_dataset
                    self.val_dataset = val_dataset
                    # Make sure essential attributes are available
                    self.nfeats = getattr(train_dataset, 'nfeats', 0)
                    self.njoints = getattr(train_dataset, 'njoints', self.nfeats // 22) 
                    self.mean = mean_val
                    self.std = std_val

                def train_dataloader(self):
                    return torch.utils.data.DataLoader(
                        self.train_dataset,
                        batch_size=cfg.TRAIN.BATCH_SIZE,
                        shuffle=True,
                        num_workers=cfg.TRAIN.NUM_WORKERS,
                        collate_fn=collate_fn,
                        pin_memory=True
                    )
                
                # We also need a validation dataloader
                def val_dataloader(self):
                    return torch.utils.data.DataLoader(
                        self.val_dataset,
                        batch_size=cfg.EVAL.BATCH_SIZE,
                        shuffle=False,
                        num_workers=cfg.TRAIN.NUM_WORKERS,
                        collate_fn=collate_fn,
                        pin_memory=True
                    )

            # We need a validation set too. For our probe, we can just reuse the same
            # dataset instance. Pytorch Lightning will handle it.
            # In a real scenario, you'd create another instance with a 'val.txt' split.
            train_phase_dataset = our_dataset
            val_phase_dataset = PhysiMoS100StyleDataset(
                mean=mean, std=std, split_file=pjoin(cfg.DATASET.SPLIT_DIR, 'val.txt'), # using test set for validation
                motion_dir=pjoin(data_root, "new_joint_vecs"),
                max_motion_length=cfg.DATASET.SAMPLER.MAX_LEN,
                min_motion_length=cfg.DATASET.SAMPLER.MIN_LEN,
                unit_length=cfg_ds.UNIT_LEN,
                style_dict_path=pjoin(data_root, "Style_name_dict.txt"),
                scene_mapping_path=cfg_ds.SCENE_MAPPING_PATH,
                style_subset=cfg_ds.get("STYLE_SUBSET", None) 
            )
            datasets.append(SimpleDataModule(train_phase_dataset, val_phase_dataset, mean, std))
        else:
            raise NotImplementedError
    cfg.DATASET.NFEATS = datasets[0].nfeats
    cfg.DATASET.NJOINTS = datasets[0].njoints
    return datasets
