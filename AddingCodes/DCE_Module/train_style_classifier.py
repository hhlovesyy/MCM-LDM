from collections import OrderedDict
import os
import sys
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, LearningRateMonitor
import numpy as np
import torch

# 将当前文件的上上上级目录（即项目根目录 MCM-LDM）添加到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# 在 macOS/Linux 上，可以简化为下面的写法，效果一样
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入你的 DataModule 和 collate_fn
from mld.data.HumanML3D import Style100DataModule 
from mld.data.utils import mld_collate_style      

# 导入我们修改后的 StyleClassifierTransformer 模型
from style_classifier_model import StyleClassifierTransformer 

# 导入 MLD 项目的 instantiate_from_config，用于加载VAE
from mld.config import instantiate_from_config

def parse_cli_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # 默认配置文件改为新的，专门用于在潜变量上训练Cstyle
    parser.add_argument("--config", type=str, default="/root/autodl-tmp/MyRepository/MCM-LDM/AddingCodes/DCE_Module/style_classifier_config.yaml", 
                        help="Path to the configuration file for training StyleClassifier on latent codes.")
    return parser.parse_args()

# --- 新的或修改的 DataModule ---
# 我们需要一个DataModule，它在内部处理原始动作到VAE潜变量的转换
# 或者，StyleClassifierTransformer的step方法自己处理这个转换
# 为了保持StyleClassifierTransformer的step方法简洁，我们让DataModule提供VAE实例，
# 并在StyleClassifierTransformer的step方法中调用它。

class StyleClassifierLatentDataModule(Style100DataModule):
    def __init__(self, 
                 cfg_for_humanml_dm: DictConfig, # 主MLD项目的配置，供父类使用
                 vae_model_instance: torch.nn.Module, 
                 mean_for_vae_input: np.ndarray, # 均值，用于归一化VAE的输入
                 std_for_vae_input: np.ndarray,  # 标准差，用于归一化VAE的输入
                 **kwargs # 其他 HumanML3DSceneDataModule 的参数 (batch_size, data_root etc.)
                ):
        # 将必要的参数传递给父类 HumanML3DSceneDataModule
        # HumanML3DSceneDataModule 的 __init__ 期望的第一个参数是 cfg (主项目的cfg)
        super().__init__(cfg=cfg_for_humanml_dm, **kwargs)
        
        # 应该是冻结过一遍了，以防万一再写一遍问题也不大
        self.vae_model = vae_model_instance # 存储冻结的VAE实例
        self.vae_model.eval() # 确保是评估模式
        for param in self.vae_model.parameters():
            param.requires_grad = False
        
        # 存储用于VAE输入的均值和标准差 (作为Tensor)
        self.mean_for_vae = torch.tensor(mean_for_vae_input, dtype=torch.float32)
        self.std_for_vae = torch.tensor(std_for_vae_input, dtype=torch.float32)
        print("StyleClassifierLatentDataModule initialized with VAE.")

    def setup(self, stage: str = None):
        super().setup(stage) # 调用父类的setup
        # 确保VAE模型和均值/标准差在正确的设备上
        # PyTorch Lightning会自动将LightningModule（我们的Cstyle模型）移动到设备
        # DataModule中的普通nn.Module需要我们注意
        # 最好的方式是在使用它们的地方（即Cstyle的step方法中）确保它们和数据在同一设备
        # 或者，如果trainer可用，可以在这里尝试移动
        if self.trainer and hasattr(self.trainer, 'lightning_module') and self.trainer.lightning_module:
            target_device = self.trainer.lightning_module.device
            self.vae_model = self.vae_model.to(target_device)
            self.mean_for_vae = self.mean_for_vae.to(target_device)
            self.std_for_vae = self.std_for_vae.to(target_device)
            print(f"VAE model and norm tensors moved to device: {target_device} in DataModule setup.")
        else:
            print("Trainer not available in DataModule setup, VAE device will be handled in Cstyle step or on first use.")


    def normalize_motion_for_vae(self, motion_data: torch.Tensor) -> torch.Tensor:
        """归一化原始动作数据以适配VAE输入"""
        # 确保均值和标准差与motion_data在同一设备,
        # note: 我们必须确认的是，这里的归一化应该是使用HumanML3D的均值和方差进行归一化的，因为VAE冻结且有先验知识
        mean = self.mean_for_vae.to(motion_data.device)
        std = self.std_for_vae.to(motion_data.device)
        return (motion_data - mean) / (std + 1e-8)


def main(cfg: DictConfig):
    pl.seed_everything(cfg.experiment.seed_value)

    # --- Logger Setup (与你之前的脚本类似) ---
    exp_name = cfg.experiment.name
    base_exp_folder = cfg.checkpoint.dirpath.split('/checkpoints')[0] if '/checkpoints' in cfg.checkpoint.dirpath else cfg.checkpoint.dirpath
    log_save_dir = os.path.join(base_exp_folder, exp_name)
    os.makedirs(log_save_dir, exist_ok=True)
    loggers_list = []
    if cfg.logger.tensorboard:
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_save_dir, name="tensorboard_logs", version="")
        loggers_list.append(tb_logger)
    if cfg.logger.wandb.enable:
        wandb_logger = pl_loggers.WandbLogger(project=cfg.logger.wandb.project, entity=cfg.logger.wandb.entity, name=exp_name, save_dir=log_save_dir, offline=cfg.logger.wandb.offline)
        loggers_list.append(wandb_logger)

    # --- 1. 加载并冻结预训练的 VAE (修改后) ---
    print("--- VAE Initialization for Cstyle Training ---")
    print("Instantiating VAE from config...")
    vae_model_instance = None # 初始化以备错误处理时使用
    try:
        # cfg.vae_dependency.config 包含了VAE的 'target' 和 'params'
        # 确保 cfg.vae_dependency.config 是正确的 DictConfig 对象
        vae_config_for_instantiation = cfg.vae_dependency.config
        if not isinstance(vae_config_for_instantiation, DictConfig):
            # 如果是从YAML直接读取的普通字典，可能需要转换
            # 但通常 OmegaConf.load() 会返回 DictConfig
            print(f"Warning: cfg.vae_dependency.config is type {type(vae_config_for_instantiation)}, attempting to use directly.")

        print("Using VAE configuration:")
        print(OmegaConf.to_yaml(vae_config_for_instantiation)) # 打印配置

        vae_model_instance = instantiate_from_config(vae_config_for_instantiation)
        print("VAE instantiated successfully.")

        # 打印实例化后的 VAE 的键名，用于对比
        current_vae_model_keys = list(vae_model_instance.state_dict().keys())
        print(f"Total keys in instantiated VAE model: {len(current_vae_model_keys)}")
        print("Sample keys from instantiated VAE model (before loading weights):")
        for i, key in enumerate(current_vae_model_keys[:10]):
            print(f"  {i+1}. {key}")

    except ImportError as e_imp:
        print(f"CRITICAL: Could not import or instantiate VAE. Target: {cfg.vae_dependency.config.get('target', 'N/A')}. Error: {e_imp}")
        sys.exit(1)
    except Exception as e_inst:
        print(f"CRITICAL: An unexpected error occurred during VAE instantiation: {e_inst}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 加载预训练的 VAE 权重
    vae_checkpoint_file_path = cfg.vae_dependency.checkpoint_path
    if vae_checkpoint_file_path and os.path.exists(vae_checkpoint_file_path):
        print(f"\nAttempting to load VAE weights from: {vae_checkpoint_file_path}")
        try:
            checkpoint_data = torch.load(vae_checkpoint_file_path, map_location='cpu')
            
            if 'state_dict' not in checkpoint_data:
                print(f"CRITICAL: Checkpoint file at {vae_checkpoint_file_path} does not have a 'state_dict' key. Cannot load VAE weights.")
                # 根据你的需求，这里可以选择 sys.exit(1) 或者让VAE使用随机权重
                # sys.exit(1) # 如果预训练VAE是必须的
                print("VAE will use random weights as checkpoint format is unexpected.")
            else:
                full_model_state_dict = checkpoint_data['state_dict']
                print(f"  Successfully loaded 'state_dict' from checkpoint. It has {len(full_model_state_dict)} keys.")
                print("  Sample keys from checkpoint's full_model_state_dict (before filtering):")
                for i, key in enumerate(list(full_model_state_dict.keys())[:10]):
                     print(f"    {i+1}. {key}")

                vae_weights_to_load = OrderedDict()
                prefix_to_strip = "vae." # 根据之前的确认，我们使用 "vae." 前缀
                
                print(f"  Attempting to strip prefix: '{prefix_to_strip}' to extract VAE weights.")
                
                found_keys_with_prefix = False
                for k, v in full_model_state_dict.items():
                    if k.startswith(prefix_to_strip):
                        found_keys_with_prefix = True
                        new_key = k.replace(prefix_to_strip, "", 1)
                        vae_weights_to_load[new_key] = v
                
                if not found_keys_with_prefix:
                    print(f"  WARNING: No keys found in checkpoint state_dict starting with the prefix '{prefix_to_strip}'.")
                    print("  VAE weights might not be loaded correctly. VAE may remain randomly initialized.")
                
                if vae_weights_to_load:
                    print(f"  Extracted {len(vae_weights_to_load)} VAE-specific keys after stripping prefix '{prefix_to_strip}'.")
                    print("  Sample keys extracted for VAE (after stripping prefix):")
                    for i, key in enumerate(list(vae_weights_to_load.keys())[:10]):
                        print(f"    {i+1}. {key}")

                    print("\n  Comparing extracted keys with instantiated VAE model keys...")
                    print("  Attempting to load VAE weights with strict=True...")
                    try:
                        missing_keys, unexpected_keys = vae_model_instance.load_state_dict(vae_weights_to_load, strict=True)
                        print("  Pre-trained VAE weights loaded successfully with strict=True!")
                        if missing_keys: print(f"    (Strict mode) Missing keys (should be empty): {missing_keys}")
                        if unexpected_keys: print(f"    (Strict mode) Unexpected keys (should be empty): {unexpected_keys}")
                    except RuntimeError as e_strict_load:
                        print(f"  ERROR: load_state_dict with strict=True failed: {e_strict_load}")
                        print("  Attempting to load with strict=False to see details...")
                        missing_keys, unexpected_keys = vae_model_instance.load_state_dict(vae_weights_to_load, strict=False)
                        if missing_keys: print(f"    (Strict=False) Missing keys: {missing_keys}")
                        else: print("    (Strict=False) No missing keys.")
                        if unexpected_keys: print(f"    (Strict=False) Unexpected keys: {unexpected_keys}")
                        else: print("    (Strict=False) No unexpected keys.")
                        if not missing_keys:
                            print("    INFO: With strict=False, all expected VAE keys were found. Strict error was likely due to unexpected_keys.")
                        else:
                            print("    CRITICAL: Even with strict=False, VAE has missing keys. Weights not loaded correctly.")
                else: # vae_weights_to_load is empty
                    if found_keys_with_prefix: # Should not happen if logic is correct
                         print("  Logic error: Found keys with prefix but vae_weights_to_load is empty.")
                    else:
                         print(f"  CRITICAL: No VAE weights extracted after attempting to strip prefix '{prefix_to_strip}'. VAE will use random weights.")
        except FileNotFoundError:
            print(f"CRITICAL: VAE checkpoint file not found at {vae_checkpoint_file_path}. VAE will use random weights.")
            # sys.exit(1)
        except KeyError as e_key:
            if "'state_dict'" in str(e_key):
                print(f"CRITICAL: Checkpoint at {vae_checkpoint_file_path} does not have 'state_dict' key.")
            else:
                print(f"CRITICAL: KeyError during VAE weight loading: {e_key}")
            # sys.exit(1)
        except Exception as e_load:
            print(f"CRITICAL: An unexpected error occurred during VAE weight loading: {e_load}")
            import traceback
            traceback.print_exc()
            # sys.exit(1)
    else:
        print(f"Warning: VAE checkpoint path ({cfg.vae_dependency.checkpoint_path}) not found or not provided. VAE will use random weights.")
    
    # 冻结VAE (确保 vae_model_instance 已成功实例化)
    if vae_model_instance is not None:
        vae_model_instance.eval()
        for param in vae_model_instance.parameters():
            param.requires_grad = False
        print("Pre-trained VAE is frozen.")
    else:
        print("CRITICAL: VAE model instance is None, cannot freeze. Exiting.") # Should not happen if instantiation is successful
        sys.exit(1)
    print("--- VAE Initialization and Weight Loading Complete for Cstyle Training ---")

    # --- 2. DataModule ---
    print("Initializing DataModule for Cstyle_Latent training...")
    # 加载主MLD项目的配置，供HumanML3DSceneDataModule的父类初始化使用
    main_mld_cfg_for_dm = OmegaConf.load(cfg.dependencies.main_mld_config_path)
    base_mld_cfg_for_dm = OmegaConf.load(cfg.dependencies.base_mld_config_path)
    # 注意：这里的合并顺序和范围，确保 datamodule_init_cfg 是 HumanML3DSceneDataModule 期望的结构
    datamodule_init_cfg = OmegaConf.merge(main_mld_cfg_for_dm, base_mld_cfg_for_dm) 
                                       # (可能还需要合并一部分当前的cfg，如果HumanML3DSceneDataModule期望)

    # 加载用于VAE输入的均值和标准差，这两个是用的HumanML3D数据集，因为VAE是在HumanML3D数据集上做pretrain的，因此均值和方差取自HumanML3D数据集
    try:
        mean_for_vae_input_np = np.load(cfg.data.mean_path)
        std_for_vae_input_np = np.load(cfg.data.std_path)
    except Exception as e:
        print(f"CRITICAL: Could not load mean/std for VAE input: {e}")
        sys.exit(1)

    # 加载 WordVectorizer
    from mld.data.humanml.utils.word_vectorizer import WordVectorizer
    try:
        w_vectorizer = WordVectorizer(cfg.data.word_vectorizer_path, "our_vab") # 或你的词向量类型
    except Exception as e:
        print(f"CRITICAL: Could not load WordVectorizer: {e}")
        sys.exit(1)
        
    # 实例化我们修改后的DataModule
    datamodule = StyleClassifierLatentDataModule(
        cfg_for_humanml_dm=datamodule_init_cfg, # 传递主项目的配置结构
        vae_model_instance=vae_model_instance,   # 传递冻结的VAE
        mean_for_vae_input=mean_for_vae_input_np, # 传递VAE输入的均值
        std_for_vae_input=std_for_vae_input_np,   # 传递VAE输入的标准差
        # 以下是 HumanML3DSceneDataModule 的其他参数，从当前cfg读取
        batch_size=cfg.data.batch_size, # 64
        num_workers=cfg.data.num_workers, # 4
        data_root=cfg.data.data_root,  # '/root/autodl-tmp/MyRepository/MCM-LDM/datasets/100StyleDataset'
        motion_dir_name=cfg.data.motion_dir_name, # 'new_joint_vecs'，这是263维的feature作为输入
        text_dir_name=cfg.data.text_dir_name, # 'texts'
        style_label_filepath=cfg.data.style_label_filename, # 'Style_name_dict.txt'
        num_style_classes=cfg.model.params.num_styles, # 100
        split_train_filename=cfg.data.split_train_filename, # 'train.txt'
        split_val_filename=cfg.data.split_val_filename, # 'val.txt'
        split_test_filename=cfg.data.get("split_test_filename", None),
        max_motion_length=cfg.data.max_motion_length, # 200
        min_motion_length=cfg.data.min_motion_length,  # 10
        max_text_len=cfg.data.max_text_len, # 虽然Cstyle不直接用文本，但DataModule可能需要
        unit_length=cfg.data.unit_length,
        mean=mean_for_vae_input_np, # 父类HumanML3DSceneDataModule的hparams.mean
        std=std_for_vae_input_np,   # 父类HumanML3DSceneDataModule的hparams.std
        w_vectorizer=w_vectorizer,    # 父类HumanML3DSceneDataModule的hparams.w_vectorizer
        collate_fn=mld_collate_style
    )
    print("DataModule initialized.")

    # --- 3. Model (StyleClassifierTransformer for latent codes) ---
    print("Initializing StyleClassifierTransformer for latent codes...")
    # model_params 现在应该包含 input_feats=256, num_input_tokens=7 等
    model_params_dict = OmegaConf.to_container(cfg.model.params, resolve=True)
    cstyle_latent_model = StyleClassifierTransformer(**model_params_dict)
    print("StyleClassifierTransformer (latent) initialized.")

    if cfg.logger.wandb.enable and wandb_logger: # 确保wandb_logger已创建
        wandb_logger.watch(cstyle_latent_model, log="all", log_freq=100)


    # --- 4. Callbacks (与你之前的脚本类似) ---
    callbacks = [RichProgressBar()]
    if cfg.logger.wandb.enable:
        callbacks.append(LearningRateMonitor(logging_interval='step'))
    checkpoint_dir = os.path.join(log_save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=cfg.checkpoint.filename, monitor=cfg.checkpoint.monitor, mode=cfg.checkpoint.mode,
        save_top_k=cfg.checkpoint.save_top_k, save_last=cfg.checkpoint.save_last,
        every_n_epochs=cfg.checkpoint.every_n_epochs
    )
    callbacks.append(checkpoint_callback)

    # --- 5. Trainer (与你之前的脚本类似) ---
    trainer_params = {
        "accelerator": cfg.experiment.accelerator, "devices": list(cfg.experiment.devices),
        "max_epochs": cfg.trainer.max_epochs, "logger": loggers_list, "callbacks": callbacks,
        "log_every_n_steps": cfg.trainer.log_every_n_steps,
        "check_val_every_n_epoch": cfg.trainer.check_val_every_n_epoch,
    }
    if "precision" in cfg.trainer: trainer_params["precision"] = cfg.trainer.precision
    if "strategy" in cfg.trainer and cfg.trainer.strategy and len(cfg.experiment.devices) > 1 : trainer_params["strategy"] = cfg.trainer.strategy
    trainer = pl.Trainer(**trainer_params)

    # --- 6. Training ---
    print("Starting StyleClassifier_Latent training...")
    trainer.fit(cstyle_latent_model, datamodule=datamodule)
    print("StyleClassifier_Latent training finished.")

    # --- 7. (Optional) Test after training (与你之前的脚本类似) ---
    if datamodule.test_dataloader() is not None and cfg.trainer.get("run_test_after_train", True):
        print("Starting testing of trained StyleClassifier_Latent...")
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path and os.path.exists(best_model_path):
            print(f"Loading best Cstyle_Latent model for test: {best_model_path}")
            # test_model = StyleClassifierTransformer.load_from_checkpoint(best_model_path) # 应该能工作
            # trainer.test(test_model, datamodule=datamodule)
            trainer.test(ckpt_path='best', datamodule=datamodule) # 使用 'best' 更方便
        else:
            print("No best model path found, testing with last model state.")
            trainer.test(cstyle_latent_model, datamodule=datamodule)
        print("StyleClassifier_Latent testing finished.")

if __name__ == "__main__":
    args = parse_cli_args()
    config = OmegaConf.load(args.config) # 加载 /root/autodl-tmp/MCM-LDM/style_classifier_config.yaml
    main(config)