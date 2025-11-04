import os
from pprint import pformat

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.strategies.ddp import DDPStrategy

from mld.callback import ProgressLogger
from mld.config import parse_args
from mld.data.get_data import get_datasets
from mld.models.get_model import get_model
from mld.utils.logger import create_logger

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def main():
    # parse options
    cfg = parse_args()  # parse config file

    # create logger
    logger = create_logger(cfg, phase="train")

    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(str(x) for x in cfg.DEVICE)

    # tensorboard logger and wandb logger
    loggers = []
    if cfg.LOGGER.WANDB.PROJECT:
        wandb_logger = pl_loggers.WandbLogger(
            project=cfg.LOGGER.WANDB.PROJECT,
            offline=cfg.LOGGER.WANDB.OFFLINE,
            id=cfg.LOGGER.WANDB.RESUME_ID,
            save_dir=cfg.FOLDER_EXP,
            version="",
            name=cfg.NAME,
            anonymous=False,
            log_model=False,
        )
        loggers.append(wandb_logger)
    if cfg.LOGGER.TENSORBOARD:
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=cfg.FOLDER_EXP,
                                                 # sub_dir="tensorboard",
                                                 sub_dir="",
                                                 version="",
                                                 name="")
        loggers.append(tb_logger)
    logger.info(OmegaConf.to_yaml(cfg))

    # create dataset
    datasets = get_datasets(cfg, logger=logger)
    logger.info("datasets module {} initialized".format("".join(
        cfg.TRAIN.DATASETS)))

    # create model
    model = get_model(cfg, datasets[0])
    logger.info("model {} loaded".format(cfg.model.model_type))

    # optimizer
    metric_monitor = {
        "Train_jf": "recons/text2jfeats/train",
        "Val_jf": "recons/text2jfeats/val",
        "Train_rf": "recons/text2rfeats/train",
        "Val_rf": "recons/text2rfeats/val",
        "APE root": "Metrics/APE_root",
        "APE mean pose": "Metrics/APE_mean_pose",
        "AVE root": "Metrics/AVE_root",
        "AVE mean pose": "Metrics/AVE_mean_pose",
        "R_TOP_1": "Metrics/R_precision_top_1",
        "R_TOP_2": "Metrics/R_precision_top_2",
        "R_TOP_3": "Metrics/R_precision_top_3",
        "gt_R_TOP_1": "Metrics/gt_R_precision_top_1",
        "gt_R_TOP_2": "Metrics/gt_R_precision_top_2",
        "gt_R_TOP_3": "Metrics/gt_R_precision_top_3",
        "FID": "Metrics/FID",
        "gt_FID": "Metrics/gt_FID",
        "Diversity": "Metrics/Diversity",
        "gt_Diversity": "Metrics/gt_Diversity",
        "MM dist": "Metrics/Matching_score",
        "Accuracy": "Metrics/accuracy",
        "gt_Accuracy": "Metrics/gt_accuracy",
    }


    callbacks = []

    # 1. 添加您喜欢的进度条 (这个很好，可以保留)
    callbacks.append(pl.callbacks.RichProgressBar())
    
    # 2. ProgressLogger (如果这是您自定义的日志记录器，可以保留)
    #    请确保 metric_monitor 与我们新的监控指标一致
    if metric_monitor: # 假设 metric_monitor 是一个外部变量
        callbacks.append(ProgressLogger(metric_monitor=metric_monitor))

    # 3. [核心] 配置我们需要的 ModelCheckpoint
    #    这个配置会自动为我们找到并保存最好的模型
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.FOLDER_EXP, "checkpoints"),
        filename="best-{epoch}-{val/total_loss:.2f}", # 文件名包含 epoch 和 loss，方便识别
        monitor="val/total_loss", # 监控我们最关心的指标
        mode="min",               # 越小越好
        save_top_k=2,             # 只保存最好的那 1 个
        save_last=True,           # 同时保存 last.ckpt 以便断点续训
        every_n_epochs=1          # 每个 epoch 都检查一次
    )
    callbacks.append(checkpoint_callback)

    # 4. [核心] 配置 EarlyStopping
    #    这个配置会在模型不再进步时停止训练，节省时间和资源
    early_stopping_callback = EarlyStopping(
        monitor="val/total_loss",
        mode="min",
        patience=30, # 如果连续30个epoch都没有进步，就停止
        min_delta=0.0005, # 认为进步至少要这么多才算数
        verbose=True
    )
    callbacks.append(early_stopping_callback)
    
    logger.info("Callbacks initialized with RichProgressBar, ModelCheckpoint (best model), and EarlyStopping.")

    if len(cfg.DEVICE) > 1:
        # ddp_strategy = DDPStrategy(find_unused_parameters=False)
        ddp_strategy = "ddp"
    else:
        ddp_strategy = None

    # trainer
    trainer = pl.Trainer(
        benchmark=False,
        max_epochs=cfg.TRAIN.END_EPOCH,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        #strategy=ddp_strategy,
        # move_metrics_to_cpu=True,
        default_root_dir=cfg.FOLDER_EXP,
        log_every_n_steps=30,
        deterministic=False,
        detect_anomaly=False,
        enable_progress_bar=True,
        logger=loggers,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
    )
    logger.info("Trainer initialized")

    vae_type = cfg.model.motion_vae.target.split(".")[-1].lower().replace(
        "vae", "")
    # strict load vae model
    if cfg.TRAIN.PRETRAINED_VAE:
        logger.info("Loading pretrain vae from {}".format(
            cfg.TRAIN.PRETRAINED_VAE))
        state_dict = torch.load(cfg.TRAIN.PRETRAINED_VAE,
                                map_location="cpu")["state_dict"]
        # extract encoder/decoder
        from collections import OrderedDict
        vae_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.split(".")[0] == "vae":
                name = k.replace("vae.", "")
                vae_dict[name] = v
        model.vae.load_state_dict(vae_dict, strict=True)

     # 2. 准备 checkpoint 路径变量，用于 trainer.fit
    ckpt_path = None
    
    # 3. 核心逻辑：区分“从断点恢复”和“加载预训练权重”
    if cfg.TRAIN.RESUME:
        logger.info(f"Resuming training from checkpoint: {cfg.TRAIN.RESUME}")
        if not os.path.isfile(cfg.TRAIN.RESUME):
             raise ValueError(f"Resume path '{cfg.TRAIN.RESUME}' is not a valid file.")
        ckpt_path = cfg.TRAIN.RESUME
        
    elif cfg.TRAIN.PRETRAINED:
        logger.info(f"Loading weights from PRETRAINED checkpoint: {cfg.TRAIN.PRETRAINED}")
        
        state_dict = torch.load(cfg.TRAIN.PRETRAINED, map_location="cpu")["state_dict"]
        
        # --- [THE FIX: 手术式加载并生成详细报告] ---
        from collections import OrderedDict

        # a. 定义我们要加载的子模块和它们在 checkpoint 中的前缀
        #    (模块名, 模型中的属性, checkpoint中的前缀)
        modules_to_load = [
            ("Denoiser", model.denoiser, "denoiser."),
            ("MotionCLIP", model.motionclip, "motionclip."),
            ("CLIP Model", model.clip_model, "clip_model."),
            ("Text Adapter", model.text_adapter, "text_adapter."),
            ("Text Norm", model.text_emb_norm, "text_emb_norm."),
            ("Motion Norm", model.motion_emb_norm, "motion_emb_norm."),
        ]
        
        all_loaded_keys = set()

        logger.info("--- Starting Surgical Weight Loading ---")
        for name, module, prefix in modules_to_load:
            # 为当前模块创建一个专属的 state_dict
            module_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    # 移除前缀，得到与子模块匹配的 key
                    new_key = k[len(prefix):]
                    module_state_dict[new_key] = v
            
            if not module_state_dict:
                logger.warning(f"  - No weights found for module '{name}' with prefix '{prefix}' in the checkpoint.")
                continue

            # 使用子模块自己的 load_state_dict，它会返回匹配信息
            missing_keys, unexpected_keys = module.load_state_dict(module_state_dict, strict=False)
            
            # 记录我们已经尝试加载过的 key
            all_loaded_keys.update(module_state_dict.keys())
            
            # 打印详细的日志报告
            logger.info(f"  - Loading for module: '{name}'")
            if missing_keys:
                logger.warning(f"    - Missing keys in module (normal if structure changed): {missing_keys}")
            if unexpected_keys:
                logger.warning(f"    - Unexpected keys in checkpoint (normal if from older version): {unexpected_keys}")
            if not missing_keys and not unexpected_keys:
                logger.info(f"    - All keys matched perfectly!")

        # 检查是否有任何 checkpoint 中的权重没有被分配到任何模块
        unaccounted_keys = [k for k in state_dict.keys() if not any(k.startswith(p[2]) for p in modules_to_load) and not k.startswith("vae.")]
        if unaccounted_keys:
             logger.warning(f"Found keys in checkpoint that were not assigned to any module: {unaccounted_keys}")

        logger.info("--- Surgical Weight Loading Complete ---")

    # 4. 统一调用 trainer.fit
    trainer.fit(model, datamodule=datasets[0], ckpt_path=ckpt_path)

    # checkpoint
    checkpoint_folder = trainer.checkpoint_callback.dirpath
    logger.info(f"The checkpoints are stored in {checkpoint_folder}")
    logger.info(
        f"The outputs of this experiment are stored in {cfg.FOLDER_EXP}")

    # end
    logger.info("Training ends!")


if __name__ == "__main__":
    main()
