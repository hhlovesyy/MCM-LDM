import numpy as np
import torch

from mld.data.humanml.scripts.motion_process import (process_file,
                                                     recover_from_ric,
                                                     extract_features)

from .base import BASEDataModule
from .humanml.data.dataset import Text2MotionDatasetV2, TextOnlyDataset, Style100Dataset

from typing import Optional, Union, Any, Callable, List, Tuple, Dict
from os.path import join as pjoin

class Style100DataModule(BASEDataModule):
    """
    LightningDataModule for the 100Style dataset.
    This module handles the creation of datasets and dataloaders for training,
    validation, and testing phases for motion generation conditioned on text and scene.
    """

    def __init__(self,
                 # --- Non-default arguments FIRST ---
                 cfg: Any, 
                 batch_size: int,
                 num_workers: int,
                 
                 mean: Union[str, np.ndarray], # 移除了 "= None"
                 std: Union[str, np.ndarray],  # 移除了 "= None"
                 w_vectorizer: Any,           # 移除了 "= None" (或者用具体类型 WordVectorizer)

                 data_root: str,  # '/root/autodl-tmp/MyRepository/MCM-LDM/datasets/100StyleDataset'
                 motion_dir_name: str, # 'new_joint_vecs'
                 text_dir_name: str, # 'texts'
                 style_label_filepath: str, # 'Style_name_dict.txt'
                 num_style_classes : int, # 100
                 split_train_filename: str, # train.txt
                 split_val_filename: str, # val.txt
                                  
                 max_motion_length: int,
                 min_motion_length: int,
                 max_text_len: int,
                 unit_length: int,

                 # --- Default arguments LAST ---
                 collate_fn: Optional[Callable[[List[Tuple]], Dict[str, Any]]] = None,
                 split_test_filename: Optional[str] = None,
                 mean_eval: Optional[Union[str, np.ndarray]] = None,
                 std_eval: Optional[Union[str, np.ndarray]] = None,
                 debug: bool = False,
                 tiny: bool = False,
                 
                 **kwargs: Any):
        
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=collate_fn)
        
        self.save_hyperparameters(logger=False) 
        
        self.name = "100Style"
        self.cfg = cfg

        self.Dataset = Style100Dataset
        
        # Set number of joints (typically fixed for HumanML3D/SMPL)
        self.njoints = 22  # Standard for SMPL model used in HumanML3D
        
        sample_overrides_for_nfeats = {
            # Use the validation split filename for creating the sample set
            "split_filename_override": self.hparams.split_val_filename, # 'val.txt'
            "tiny": True,         # Load a minimal subset
            "debug": False,       # Ensure debug is off for this utility load
            "progress_bar": False # No progress bar for this utility load
        }
        self._sample_set = self.get_sample_set(overrides=sample_overrides_for_nfeats)
        self.nfeats = self._sample_set.nfeats  # 每一帧的特征数： 263 features
        
        # Initialize for mm_mode (multi-modal evaluation sampling)
        self.is_mm = False # Inherited from BASEDataModule, can be controlled by mm_mode method
        self.name_list_backup_for_mm = None # To store original name_list of test_dataset

    def _get_split_file_path(self, split_name_in_cfg: str, default_filename_hparam: str) -> str:
        """
        Helper to determine the full path for a split file.
        It prioritizes cfg.DATASET.HUMANML3D_SCENE.SPLIT_ROOT if defined,
        otherwise uses self.hparams.data_root.
        """
        # Use the filename passed via hparams (e.g., self.hparams.split_train_filename)
        filename = getattr(self.hparams, default_filename_hparam)
        
        # Try to get a specific SPLIT_ROOT for humanml3d_scene from cfg
        try:
            # Path like: cfg.DATASET.HUMANML3D_SCENE.SPLIT_ROOT
            split_root = eval(f"self.cfg.DATASET.{self.name.upper()}.SPLIT_ROOT")
        except:
            # Fallback to self.hparams.data_root if SPLIT_ROOT is not defined for this dataset
            split_root = self.hparams.data_root
            
        return pjoin(split_root, filename)

    def get_sample_set(self, overrides: dict = None) -> Style100Dataset:
        """
        Creates a 'sample' instance of the HumanML3DSceneDataset,
        typically used to derive metadata like nfeats or for quick tests.
        Overrides allow modifying some parameters for this sample instance.
        """
        # Start with a copy of all hyperparameters saved during __init__
        # self.hparams is an HParams object (usually OmegaConf or similar, supports attr access)
        # We want sample_params to also support attribute access if possible,
        # or consistently use dictionary access.
        # If self.hparams.copy() returns a dict, we need to use dict access.
        
        # Let's check the type of self.hparams to be sure
        # print(f"Type of self.hparams in get_sample_set: {type(self.hparams)}")
        
        # Create a new dictionary for sample_params based on hparams
        # This ensures we are working with a mutable copy for overrides.
        # And we will use dictionary access which is safe.
        sample_params_dict = {}
        for key in self.hparams: # Iterate over keys in hparams
            sample_params_dict[key] = self.hparams[key]

        # Apply any overrides passed to this method
        if overrides:
            sample_params_dict.update(overrides)

        # Determine the split file for the sample set.
        split_filename_for_sample = sample_params_dict.get("split_filename_override", self.hparams.split_val_filename)
        
        try:
            split_root_for_sample = eval(f"self.cfg.DATASET.{self.name.upper()}.SPLIT_ROOT")
        except:
            split_root_for_sample = sample_params_dict['data_root'] # Use data_root from sample_params_dict
        
        actual_split_file_path = pjoin(split_root_for_sample, split_filename_for_sample)

        # Construct arguments for the 100StyleDataset constructor using dictionary access
        dataset_args_for_sample = {
            "mean": sample_params_dict['mean'], 
            "std": sample_params_dict['std'],   
            "w_vectorizer": sample_params_dict['w_vectorizer'], 
            "split_file": actual_split_file_path,
            "max_motion_length": sample_params_dict['max_motion_length'], 
            "min_motion_length": sample_params_dict['min_motion_length'], 
            "max_text_len": sample_params_dict['max_text_len'],           
            "unit_length": sample_params_dict['unit_length'],             
            "motion_dir": pjoin(sample_params_dict['data_root'], sample_params_dict['motion_dir_name']), # <--- 使用字典访问
            "text_dir": pjoin(sample_params_dict['data_root'], sample_params_dict['text_dir_name']),     # <--- 使用字典访问
            "style_label_filepath": pjoin(sample_params_dict['data_root'], sample_params_dict['style_label_filepath']), # <--- 使用字典访问
            "style_classes": sample_params_dict['num_style_classes'],   # <--- 使用字典访问
            "tiny": sample_params_dict.get("tiny", True), # Use .get() for optional keys from overrides
            "debug": sample_params_dict.get("debug", False),
            "progress_bar": sample_params_dict.get("progress_bar", False)
        }
        return self.Dataset(**dataset_args_for_sample)

    def __getattr__(self, item: str):
        """
        Custom attribute getter to lazily initialize train/val/test datasets.
        This overrides the BASEDataModule.__getattr__ to correctly construct
        kwargs for Style100Dataset.、
        在 Python 中，__getattr__ 是一个“魔术方法”。当代码尝试访问一个不存在的对象属性时（例如 self.train_dataset），Python 就会自动调用这个方法。
        """
        if item.endswith("_dataset") and not item.startswith("_"):
            subset_name = item[:-len("_dataset")]  # "train", "val", or "test"
            cached_item_name = "_" + item # e.g., "_train_dataset"

            if cached_item_name not in self.__dict__:
                # Prepare arguments for HumanML3DSceneDataset constructor
                # Most arguments come directly from self.hparams (saved in __init__)
                
                # Determine the correct split filename for this subset
                if subset_name == "train":
                    split_filename = self.hparams.split_train_filename
                elif subset_name == "val":
                    split_filename = self.hparams.split_val_filename
                elif subset_name == "test":
                    split_filename = self.hparams.split_test_filename
                    if not split_filename: # If test split is not defined
                        # print(f"Warning: Test split file not configured for {self.name}. Test dataset will be None.")
                        self.__dict__[cached_item_name] = None
                        return None
                else:
                    raise ValueError(f"Unknown dataset subset: {subset_name}")

                # Construct full path to the split file.
                # Prefer SPLIT_ROOT from cfg if available for this dataset type.
                try:
                    # Path like: cfg.DATASET.HUMANML3D_SCENE.SPLIT_ROOT
                    split_root = eval(f"self.cfg.DATASET.{self.name.upper()}.SPLIT_ROOT")
                except:
                    # Fallback to the main data_root of the scene dataset if SPLIT_ROOT is not specific
                    split_root = self.hparams.data_root
                
                actual_split_file_path = pjoin(split_root, split_filename)

                dataset_constructor_args = {
                    # Core components
                    "mean": self.hparams.mean,
                    "std": self.hparams.std,
                    "w_vectorizer": self.hparams.w_vectorizer,
                    # Paths and identifiers
                    "split_file": actual_split_file_path,
                    "motion_dir": pjoin(self.hparams.data_root, self.hparams.motion_dir_name),
                    "text_dir": pjoin(self.hparams.data_root, self.hparams.text_dir_name),
                    "style_label_filepath": pjoin(self.hparams.data_root, self.hparams.style_label_filepath),
                    "num_style_classes": self.hparams.num_style_classes,
                    # Processing parameters
                    "max_motion_length": self.hparams.max_motion_length,
                    "min_motion_length": self.hparams.min_motion_length,
                    "max_text_len": self.hparams.max_text_len,
                    "unit_length": self.hparams.unit_length,
                    # Control flags
                    "debug": self.hparams.debug,
                    "tiny": self.hparams.tiny,
                    # "progress_bar": True # Enable progress bar for actual dataset loading
                }
                
                self.__dict__[cached_item_name] = self.Dataset(**dataset_constructor_args)
            return getattr(self, cached_item_name)
        
        # Fallback to default __getattr__ for other attributes
        return super().__getattr__(item)

    # `setup` method is inherited from BASEDataModule and should work correctly
    # if `__getattr__` is properly overridden to supply the datasets.

    # `train_dataloader`, `val_dataloader`, `test_dataloader`, `predict_dataloader`
    # are inherited from BASEDataModule and should use the dataloader_options
    # initialized in super().__init__ along with the lazily loaded datasets.

    # === Methods potentially copied/adapted from HumanML3DDataModule ===
    # These methods assume that `self.hparams` contains `mean`, `std`, 
    # `mean_eval` (optional), `std_eval` (optional), and `self.njoints` is correctly set.

    def feats2joints(self, features: torch.Tensor) -> torch.Tensor:
        """Converts normalized features back to joint coordinates."""
        # Ensure mean/std are tensors on the same device as features
        mean = torch.tensor(self.hparams.mean, dtype=features.dtype, device=features.device)
        std = torch.tensor(self.hparams.std, dtype=features.dtype, device=features.device)
        features = features * std + mean  # Denormalize
        
        # recover_from_ric needs to be imported or accessible in the scope
        # from mld.utils.motion_process import recover_from_ric # Example import
        # Make sure recover_from_ric is properly imported in your project
        try:
            from mld.data.humanml.scripts.motion_process import recover_from_ric
            return recover_from_ric(features, self.njoints)
        except ImportError:
            raise ImportError("Function 'recover_from_ric' not found. Please ensure it's correctly imported.")


    def joints2feats(self, features: np.ndarray) -> np.ndarray:
        """
        Converts joint coordinates (e.g., shape [batch_size, frames, njoints, 3])
        to the feature representation used by the model.
        This method relies on an `extract_features` function.
        The input `features` is expected to be a NumPy array of batched joint positions.
        """
        # extract_features needs to be imported or accessible
        # from mld.utils.motion_process import extract_features # Example import
        try:
            from mld.data.humanml.scripts.motion_process import extract_features
        except ImportError:
            raise ImportError("Function 'extract_features' not found. Please ensure it's correctly imported.")

        feature_list = []
        for i in range(features.shape[0]):  # Iterate over batch
            # Assuming features[i] is (frames, njoints, 3)
            # extract_features should process one motion sequence at a time.
            # It might expect a Tensor, so convert if necessary.
            # The exact API of extract_features (input type, other args) needs to be matched.
            one_motion_joints_np = features[i]
            one_motion_joints_tensor = torch.from_numpy(one_motion_joints_np).float()
            
            # The original HumanML3DDataModule's extract_features call might look different.
            # This is a placeholder, adjust based on the actual 'extract_features' signature.
            # It might take seq_len or other parameters.
            # For example, if it requires seq_len:
            # feature_sample_tensor = extract_features(one_motion_joints_tensor, seq_len=one_motion_joints_tensor.shape[0])
            feature_sample_tensor = extract_features(one_motion_joints_tensor) # Simplified call
            feature_sample_np = feature_sample_tensor.numpy() # Convert back to NumPy if needed

            feature_list.append(feature_sample_np)
        
        processed_features_batch = np.array(feature_list)
        
        # Normalization (if performed after feature extraction)
        # This was commented out in the original HumanML3DDataModule.
        # If your HumanML3DSceneDataset already returns normalized features, this step might not be needed here.
        # If features are extracted raw and then normalized:
        # mean_np = np.array(self.hparams.mean)
        # std_np = np.array(self.hparams.std)
        # processed_features_batch = (processed_features_batch - mean_np) / std_np
        
        return processed_features_batch

    def renorm4t2m(self, features: torch.Tensor) -> torch.Tensor:
        """
        Re-normalizes features from this dataset's normalization to
        the T2M (Text2Motion) dataset's normalization, typically for evaluation.
        """
        # Check if evaluation-specific mean/std are provided
        if self.hparams.mean_eval is None or self.hparams.std_eval is None:
            # print("Warning: mean_eval or std_eval not provided in hparams for renorm4t2m. Returning original features.")
            return features

        # Ensure all mean/std are tensors on the same device and dtype as features
        ori_mean = torch.tensor(self.hparams.mean, dtype=features.dtype, device=features.device)
        ori_std = torch.tensor(self.hparams.std, dtype=features.dtype, device=features.device)
        eval_mean = torch.tensor(self.hparams.mean_eval, dtype=features.dtype, device=features.device)
        eval_std = torch.tensor(self.hparams.std_eval, dtype=features.dtype, device=features.device)
        
        # Denormalize from current dataset's mean/std
        features_denorm = features * ori_std + ori_mean
        # Normalize to target (T2M) dataset's mean/std
        features_renorm = (features_denorm - eval_mean) / eval_std
        return features_renorm

    def mm_mode(self, mm_on: bool = True):
        """
        Activates or deactivates "multi-modal" mode for the test_dataset.
        In mm_mode, a random subset of test samples is used, often for qualitative evaluation.
        """
        # Ensure test_dataset is loaded (it will be if accessed, due to __getattr__)
        if self.test_dataset is None: # Accesses the property, which loads it
            # print("Warning: test_dataset is not available (e.g., no test split configured). Cannot activate mm_mode.")
            return

        if mm_on:
            if not self.is_mm:  # Only backup the full list once
                # test_dataset.name_list should exist if HumanML3DSceneDataset has it
                self.name_list_backup_for_mm = self.test_dataset.name_list[:] # Create a copy
            
            self.is_mm = True # Set flag defined in BASEDataModule
            
            # Get number of samples for MM evaluation from config
            # Example: self.cfg.TEST.MM_NUM_SAMPLES
            num_mm_samples = self.cfg.TEST.get("MM_NUM_SAMPLES", 30) # Default to 30 if not in cfg
            
            if not self.name_list_backup_for_mm: # Should not happen if test_dataset loaded
                 # print("Warning: Original name_list for test_dataset is not available for mm_mode.")
                 return

            if len(self.name_list_backup_for_mm) == 0:
                # print("Warning: test_dataset.name_list is empty. Cannot select MM samples.")
                self.test_dataset.name_list = []
                return

            if len(self.name_list_backup_for_mm) < num_mm_samples:
                # print(f"Warning: Requested MM_NUM_SAMPLES ({num_mm_samples}) is more than available "
                #       f"test samples ({len(self.name_list_backup_for_mm)}). Using all available.")
                self.mm_list_current = self.name_list_backup_for_mm[:]
            else:
                self.mm_list_current = np.random.choice(self.name_list_backup_for_mm,
                                                        num_mm_samples,
                                                        replace=False).tolist()
            self.test_dataset.name_list = self.mm_list_current
            # print(f"MM mode ON. Using {len(self.test_dataset.name_list)} samples for testing.")
        else:  # Turn mm_mode off
            if self.is_mm and self.name_list_backup_for_mm is not None:
                # Restore the original full list of names to the test_dataset
                self.test_dataset.name_list = self.name_list_backup_for_mm
                # print(f"MM mode OFF. Restored test_dataset to {len(self.test_dataset.name_list)} samples.")
            self.is_mm = False

class HumanML3DDataModule(BASEDataModule):

    def __init__(self,
                 cfg,
                 batch_size,
                 num_workers,
                 collate_fn=None,
                 phase="train",
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=collate_fn)
        self.save_hyperparameters(logger=False)
        self.name = "humanml3d"
        self.njoints = 22
        if phase == "text_only":
            self.Dataset = TextOnlyDataset
        else:
            self.Dataset = Text2MotionDatasetV2
        self.cfg = cfg
        sample_overrides = {
            "split": "val",
            "tiny": True,
            "progress_bar": False
        }
        self._sample_set = self.get_sample_set(overrides=sample_overrides)
        # Get additional info of the dataset
        self.nfeats = self._sample_set.nfeats
        # self.transforms = self._sample_set.transforms

    def feats2joints(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return recover_from_ric(features, self.njoints)

    def joints2feats(self, features):
        # chuli (bs, frame, 22, 3)

        # batch里面逐个动作处理
        feature_list = []
        for i in range(features.shape[0]):
            feature = extract_features(features[i,...])

            # 复制最后一帧
            last_frame = feature[-1].copy()
            feature = np.concatenate((feature, np.expand_dims(last_frame, axis=0)), axis=0)

            feature_list.append(feature)
        features = np.array(feature_list)
        

        # mean = self.hparams.mean
        # std = self.hparams.std
        # features = (features - mean) / std
        return features

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def mm_mode(self, mm_on=True):
        # random select samples for mm
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.TEST.MM_NUM_SAMPLES,
                                            replace=False)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list
