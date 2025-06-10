# train_dce.py

import os
import sys
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, LearningRateMonitor
import numpy as np # 用于加载mean/std

# 将当前文件的上上上级目录（即项目根目录 MCM-LDM）添加到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# 在 macOS/Linux 上，可以简化为下面的写法，效果一样
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入你的 DataModule 和 collate_fn
from mld.data.HumanML3D import Style100DataModule
from mld.data.utils import mld_collate_style         

# 导入我们定义的 DCETrainingModule
from dce_lightning_module import DCETrainingModule   

just_test = False
path_to_test = '/root/autodl-tmp/MCM-LDM/experiments/dce_checkpoints/DCE_Training_Experiment_v1/checkpoints/last.ckpt'

def parse_cli_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="AddingCodes/DCE_Module/dce_training_config.yaml", # 新的配置文件
                        help="Path to the DCE training configuration file.")
    return parser.parse_args()

def main(cfg: DictConfig):
    pl.seed_everything(cfg.experiment.seed_value)

    # --- Logger ---
    exp_name = cfg.experiment.name # 'DCE_Training_Experiment_v1'
    base_exp_folder = cfg.checkpoint.dirpath.split('/checkpoints')[0] if '/checkpoints' in cfg.checkpoint.dirpath else cfg.checkpoint.dirpath
    log_save_dir = os.path.join(base_exp_folder, exp_name)
    os.makedirs(log_save_dir, exist_ok=True)

    loggers_list = []
    if cfg.logger.tensorboard:
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_save_dir, name="tensorboard_logs", version="")
        loggers_list.append(tb_logger)
    if cfg.logger.wandb.enable:
        wandb_logger = pl_loggers.WandbLogger(
            project=cfg.logger.wandb.project,
            entity=cfg.logger.wandb.entity,
            name=exp_name,
            save_dir=log_save_dir,
            offline=cfg.logger.wandb.offline,
        )
        loggers_list.append(wandb_logger)

    # --- DataModule ---
    print("Initializing DataModule for DCE training...")
    # 模拟主项目配置，供DataModule内部使用
    main_project_cfg_for_dm = OmegaConf.load(cfg.dependencies.main_mld_config_path) # 从配置读取主MLD项目config路径
    base_cfg = OmegaConf.load(cfg.dependencies.base_mld_config_path) # 从配置读取base MLD项目config路径
    datamodule_specific_cfg = OmegaConf.merge(main_project_cfg_for_dm, base_cfg)

    # 加载 w_vectorizer, mean, std (路径来自 dce_training_config.yaml)
    from mld.data.humanml.utils.word_vectorizer import WordVectorizer
    try:
        w_vectorizer = WordVectorizer(cfg.data.word_vectorizer_path, "our_vab") # "our_vab" 或你的词向量类型
    except Exception as e:
        print(f"CRITICAL: Could not load WordVectorizer: {e}")
        sys.exit(1)
    try:
        mean_np = np.load(cfg.data.mean_path)
        std_np = np.load(cfg.data.std_path)
    except Exception as e:
        print(f"CRITICAL: Could not load mean/std: {e}")
        sys.exit(1)

    datamodule = Style100DataModule(
        cfg=datamodule_specific_cfg,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        data_root=cfg.data.data_root,
        motion_dir_name=cfg.data.motion_dir_name,
        text_dir_name=cfg.data.text_dir_name,
        style_label_filepath=cfg.data.style_label_filepath,
        num_style_classes=cfg.dce_module.cstyle_params.params.num_styles, # 从Cstyle配置中推断
        split_train_filename=cfg.data.split_train_filename,
        split_val_filename=cfg.data.split_val_filename,
        split_test_filename=cfg.data.get("split_test_filename", 'test.txt'), # 可选，默认为'test.txt'
        max_motion_length=cfg.data.max_motion_length, # 200
        min_motion_length=cfg.data.min_motion_length, # 10
        max_text_len=cfg.data.max_text_len,
        unit_length=cfg.data.unit_length,
        mean=mean_np, # 传递 NumPy 数组
        std=std_np,   # 传递 NumPy 数组
        w_vectorizer=w_vectorizer,
        collate_fn=mld_collate_style,
        debug=cfg.data.get("debug", False),
        tiny=cfg.data.get("tiny", False)
    )
    print("DataModule initialized.")

    # --- Model (DCETrainingModule) ---
    print("Initializing DCETrainingModule...")
    # DCETrainingModule的参数来自配置文件的 dce_module 部分
    # 将 DictConfig 转换为字典传递给 DCETrainingModule
    dce_training_params = {
        "cfg_dce": cfg.dce_module.dce_definition, # DCE自身的网络结构配置
        "cfg_cstyle": cfg.dce_module.cstyle_params, # Cstyle的原始配置 (用于加载结构)
        "cstyle_checkpoint_path": cfg.dependencies.cstyle_checkpoint_path,
        "cfg_vae": cfg.dce_module.vae_params, # VAE的原始配置 (用于加载结构)
        "vae_checkpoint_path": cfg.dependencies.vae_checkpoint_path, # VAE(MLD)的检查点
        "learning_rate": cfg.optimizer.learning_rate,
        "lambda_style": cfg.loss_weights.lambda_style,
        "lambda_content": cfg.loss_weights.lambda_content,
        "mean_path_for_vae_input": cfg.data.mean_path, # 从主配置文件读取
        "std_path_for_vae_input": cfg.data.std_path,   # 从主配置文件读取
        "lambda_grl": cfg.loss_weights.lambda_grl
    }

    
    model = DCETrainingModule(**dce_training_params)
    
    # 设置均值和标准差给模型 (在模型实例化后，DataModule准备好之前或之后)
    # DataModule 的 hparams.mean/std 已经是 NumPy 数组了
    # DCETrainingModule 内部会将它们转为Tensor并设为Parameter
    # model.set_mean_std(mean_np, std_np)
    print("DCETrainingModule initialized and mean/std set.")

    if cfg.logger.wandb.enable and len(loggers_list) > 0:
        for logger_instance in loggers_list:
            if isinstance(logger_instance, pl_loggers.WandbLogger):
                logger_instance.watch(model.dce, log="all", log_freq=100) # Watch DCE parameters
                break

    # --- Callbacks ---
    callbacks = [RichProgressBar()]
    if cfg.logger.wandb.enable:
        callbacks.append(LearningRateMonitor(logging_interval='step'))

    checkpoint_dir = os.path.join(log_save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=cfg.checkpoint.filename, # e.g., "dce-{epoch:02d}-{val_total_loss:.2f}"
        monitor=cfg.checkpoint.monitor,   # e.g., "val_total_loss"
        mode=cfg.checkpoint.mode,         # e.g., "min"
        save_top_k=cfg.checkpoint.save_top_k,
        save_last=cfg.checkpoint.save_last,
        every_n_epochs=cfg.checkpoint.every_n_epochs,
    )
    callbacks.append(checkpoint_callback)

    # --- Trainer ---
    trainer_params = {
        "accelerator": cfg.experiment.accelerator,
        "devices": list(cfg.experiment.devices),
        "max_epochs": cfg.trainer.max_epochs,
        "logger": loggers_list,
        "callbacks": callbacks,
        "log_every_n_steps": cfg.trainer.log_every_n_steps,
        "check_val_every_n_epoch": cfg.trainer.check_val_every_n_epoch,
    }
    if "precision" in cfg.trainer and cfg.trainer.precision in [16, 32, "bf16", "16-mixed", "bf16-mixed"]:
         trainer_params["precision"] = cfg.trainer.precision
    if "strategy" in cfg.trainer and cfg.trainer.strategy and len(cfg.experiment.devices) > 1:
         trainer_params["strategy"] = cfg.trainer.strategy

    trainer = pl.Trainer(**trainer_params)

    if just_test:
        print("Re-instantiating model for testing...")
        # 假设 dce_training_params 是之前用于实例化 model 的字典
        # 你需要确保这个字典的参数与检查点保存时的模型兼容
        model_for_test = DCETrainingModule(**dce_training_params) # 使用训练开始时的参数重新实例化

        print(f"Loading state_dict from {path_to_test} into re-instantiated model...")
        try:
            # 加载 state_dict (注意: Lightning 保存的 checkpoint 包含 'state_dict'键)
            import torch
            ckpt_content = torch.load(path_to_test, map_location=lambda storage, loc: storage)
            if 'state_dict' in ckpt_content:
                model_for_test.load_state_dict(ckpt_content['state_dict'])
                print("State_dict loaded into re-instantiated model.")
                results_manual_load = trainer.test(model_for_test, datamodule=datamodule)
                print("Testing results (manual model load):", results_manual_load)
            else:
                print(f"Error: Checkpoint {path_to_test} does not contain 'state_dict' key.")
        except Exception as e_manual_load:
            print(f"Error during manual model loading or testing: {e_manual_load}")
            print("Testing failed. Please check model instantiation and checkpoint compatibility.")
            
            print("No specific checkpoint path specified or found, testing with the model state at the end of training.")
            results_in_memory = trainer.test(model, datamodule=datamodule)
            print("Testing results (model in memory):", results_in_memory)
        
        print("DCE testing finished.")
        return
    
    
    # --- Training ---
    print("Starting DCE training...")
    trainer.fit(model, datamodule=datamodule)
    print("DCE training finished.")

if __name__ == "__main__":
    args = parse_cli_args()
    # 加载DCE训练专用的配置文件
    config = OmegaConf.load(args.config) 
    main(config)