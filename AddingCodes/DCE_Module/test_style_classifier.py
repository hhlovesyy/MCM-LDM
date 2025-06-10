# test_cstyle_on_latent.py

from collections import OrderedDict
import os
import sys
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from argparse import ArgumentParser
import numpy as np

# 导入我们修改后的 StyleClassifierTransformer 模型 (能在潜变量上工作)
from style_classifier_model import StyleClassifierTransformer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, LearningRateMonitor

# 导入我们为 Cstyle_latent 训练定义的 DataModule 和 collate_fn
# 以及主项目的 instantiate_from_config
# (假设 train_cstyle_on_latent.py 和 style_classifier_model.py 在同一目录或PYTHONPATH可达)
from train_style_classifier import StyleClassifierLatentDataModule # 我们的DataModule
from mld.data.utils import mld_collate_scene                   # collate_fn
from mld.config import instantiate_from_config                 # 用于加载VAE

def parse_cli_args():
    parser = ArgumentParser()
    # 默认配置文件现在应该是用于潜变量Cstyle训练的那个
    parser.add_argument("--config", type=str, default="style_classifier_latent_config.yaml",
                        help="Path to the configuration file used for Cstyle_latent training.")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the trained Cstyle_latent model checkpoint (.ckpt).")
    parser.add_argument("--test_batch_size", type=int, default=None,
                        help="Override test batch size from config file.")
    parser.add_argument("--devices", type=str, default=None, # 与训练脚本保持一致
                        help="Override devices, e.g., '0' or '0,1'. Defaults to config experiment.devices.")
    return parser.parse_args()

def main(cfg: DictConfig, checkpoint_path: str, cli_args: object):
    # 应用命令行覆盖 (如果提供)
    if cli_args.test_batch_size is not None:
        cfg.data.batch_size = cli_args.test_batch_size # 假设配置文件中用 cfg.data.batch_size
    if cli_args.devices is not None:
        cfg.experiment.devices = [int(d.strip()) for d in cli_args.devices.split(',')]
    
    pl.seed_everything(cfg.experiment.seed_value)

    # --- 1. 加载并冻结预训练的 VAE (与训练Cstyle_latent时一致) ---
    print("Loading and freezing pre-trained VAE for testing Cstyle_Latent...")
    try:
        vae_model_instance = instantiate_from_config(cfg.vae_dependency.config)
        if cfg.vae_dependency.checkpoint_path and os.path.exists(cfg.vae_dependency.checkpoint_path):
            checkpoint_data = torch.load(cfg.vae_dependency.checkpoint_path, map_location='cpu')
            vae_state_dict_to_load = None
            if isinstance(checkpoint_data, dict) and 'state_dict' in checkpoint_data:
                vae_state_dict_to_load = checkpoint_data['state_dict']
                if any(key.startswith("vae.") for key in vae_state_dict_to_load.keys()):
                    temp_dict = {k.replace('vae.', '', 1): v for k, v in vae_state_dict_to_load.items() if k.startswith('vae.')}
                    if temp_dict: vae_state_dict_to_load = temp_dict
            elif isinstance(checkpoint_data, (dict, OrderedDict)): # PyTorch 3.7+ OrderedDict is a dict
                vae_state_dict_to_load = checkpoint_data
            
            if vae_state_dict_to_load:
                final_vae_state_dict = {}
                current_vae_keys = vae_model_instance.state_dict().keys()
                for k_ckpt, v_ckpt in vae_state_dict_to_load.items():
                    k_model = k_ckpt.replace("vae.", "", 1) if k_ckpt.startswith("vae.") else k_ckpt
                    if k_model in current_vae_keys: final_vae_state_dict[k_model] = v_ckpt
                
                if final_vae_state_dict:
                    missing_keys, unexpected_keys = vae_model_instance.load_state_dict(final_vae_state_dict, strict=False)
                    if missing_keys: print(f"VAE Missing keys during load: {missing_keys}")
                    if unexpected_keys: print(f"VAE Unexpected keys during load: {unexpected_keys}")
                    print("Pre-trained VAE weights loaded for testing.")
                else: print("Warning: No matching keys found for VAE in checkpoint.")
            else: print(f"Warning: Could not extract VAE state_dict from {cfg.vae_dependency.checkpoint_path}.")
        else: print(f"Warning: VAE checkpoint path not found. VAE might be randomly initialized (not ideal for testing).")
        
        vae_model_instance.eval()
        for param in vae_model_instance.parameters():
            param.requires_grad = False
        print("Pre-trained VAE is frozen for testing.")
    except Exception as e:
        print(f"CRITICAL: Error loading/instantiating VAE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- 2. DataModule ---
    print("Initializing DataModule for Cstyle_Latent testing...")
    # 加载主MLD项目的配置 (与训练脚本一致)
    main_mld_cfg_for_dm = OmegaConf.load(cfg.dependencies.main_mld_config_path)
    base_mld_cfg_for_dm = OmegaConf.load(cfg.dependencies.base_mld_config_path)
    datamodule_init_cfg = OmegaConf.merge(main_mld_cfg_for_dm, base_mld_cfg_for_dm)

    # 加载用于VAE输入的均值和标准差
    try:
        mean_for_vae_input_np = np.load(cfg.data.mean_path)
        std_for_vae_input_np = np.load(cfg.data.std_path)
    except Exception as e:
        print(f"CRITICAL: Could not load mean/std for VAE input: {e}")
        sys.exit(1)

    from mld.data.humanml.utils.word_vectorizer import WordVectorizer
    try:
        w_vectorizer = WordVectorizer(cfg.data.word_vectorizer_path, "our_vab")
    except Exception as e:
        print(f"CRITICAL: Could not load WordVectorizer: {e}")
        sys.exit(1)

    # 使用 StyleClassifierLatentDataModule
    datamodule = StyleClassifierLatentDataModule(
        cfg_for_humanml_dm=datamodule_init_cfg,
        vae_model_instance=vae_model_instance,
        mean_for_vae_input=mean_for_vae_input_np,
        std_for_vae_input=std_for_vae_input_np,
        # 其他参数来自当前测试的配置文件 (cfg，即 style_classifier_latent_config.yaml)
        batch_size=cfg.data.batch_size, # 使用更新后的batch_size (可能被命令行覆盖)
        num_workers=cfg.data.num_workers,
        data_root=cfg.data.data_root,
        motion_dir_name=cfg.data.motion_dir_name,
        text_dir_name=cfg.data.text_dir_name,
        scene_label_filename=cfg.data.scene_label_filename,
        num_scene_classes=cfg.model.params.num_styles, # Cstyle的类别数
        split_train_filename=cfg.data.split_train_filename, # test时通常不需要
        split_val_filename=cfg.data.split_val_filename,     # test时通常不需要
        split_test_filename=cfg.data.get("split_test_filename", None), # **必须提供测试集文件**
        max_motion_length=cfg.data.max_motion_length,
        min_motion_length=cfg.data.min_motion_length,
        max_text_len=cfg.data.max_text_len,
        unit_length=cfg.data.unit_length,
        mean=mean_for_vae_input_np, 
        std=std_for_vae_input_np,   
        w_vectorizer=w_vectorizer,    
        collate_fn=mld_collate_scene
    )
    print("DataModule initialized for testing.")
    
    if datamodule.test_dataloader() is None:
        print("ERROR: Test dataset is not available. Ensure 'data.split_test_filename' is configured in your YAML.")
        sys.exit(1)

    # --- 3. Model (StyleClassifierTransformer for latent codes) ---
    print(f"Loading Cstyle_Latent model from checkpoint: {checkpoint_path}")
    try:
        # StyleClassifierTransformer.load_from_checkpoint 会使用保存在ckpt中的hparams
        # 包括 input_feats=256, num_input_tokens=7 等
        cstyle_latent_model = StyleClassifierTransformer.load_from_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"Error loading Cstyle_Latent model from checkpoint: {e}")
        # 尝试打印模型定义时的hparams以帮助调试
        print("Expected hparams for StyleClassifierTransformer (from config):")
        print(OmegaConf.to_yaml(cfg.model.params))
        sys.exit(1)
        
    cstyle_latent_model.eval() # 确保模型在评估模式
    print("Cstyle_Latent model loaded.")

    # --- 4. Trainer (主要用于 .test() 功能) ---
    trainer_params = {
        "accelerator": cfg.experiment.accelerator,
        "devices": list(cfg.experiment.devices), # 使用更新后的devices
        "logger": False, # 测试时通常禁用logger
        "callbacks": [RichProgressBar()],
    }
    if "precision" in cfg.trainer: trainer_params["precision"] = cfg.trainer.precision
    if "strategy" in cfg.trainer and cfg.trainer.strategy and len(cfg.experiment.devices) > 1 :
        trainer_params["strategy"] = cfg.trainer.strategy
        
    trainer = pl.Trainer(**trainer_params)

    # --- 5. Testing ---
    print("Starting Cstyle_Latent testing...")
    # trainer.test() 将调用 StyleClassifierTransformer 中的 test_step,
    # 而 test_step 会调用 _common_step_latent,
    # _common_step_latent 会从 self.trainer.datamodule 获取 VAE 和归一化方法
    test_results = trainer.test(cstyle_latent_model, datamodule=datamodule, verbose=True)
    print("Cstyle_Latent testing finished.")
    
    if test_results:
        print("\n" + "="*30 + " Test Results " + "="*30)
        for i, result_dict in enumerate(test_results):
            print(f"Results for test_dataloader {i}:")
            for key, value in result_dict.items():
                print(f"  {key}: {value:.4f}") # 假设value是数值
        print("="*74 + "\n")
    else:
        print("No test results returned by trainer.test(). Check model's test_step implementation.")

if __name__ == "__main__":
    cli_args = parse_cli_args()
    # 加载用于Cstyle_latent训练的配置文件
    config = OmegaConf.load(cli_args.config) 
    
    main(config, cli_args.checkpoint_path, cli_args)