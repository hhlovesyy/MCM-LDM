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


def main():
    # parse options
    cfg = parse_args()  # parse config file

    # create logger
    logger = create_logger(cfg, phase="train")

    # resume
    if cfg.TRAIN.RESUME:
        resume = cfg.TRAIN.RESUME
        # backcfg = cfg.TRAIN.copy()
        if os.path.exists(resume):
        #     file_list = sorted(os.listdir(resume), reverse=True)
        #     for item in file_list:
        #         if item.endswith(".yaml"):
        #             cfg = OmegaConf.load(os.path.join(resume, item))
        #             cfg.TRAIN = backcfg
        #             break
        #     checkpoints = sorted(os.listdir(os.path.join(
        #         resume, "checkpoints")),
        #                          key=lambda x: int(x[6:-5]),
        #                          reverse=True)
        #     for checkpoint in checkpoints:
        #         if "epoch=" in checkpoint:
        #             cfg.TRAIN.PRETRAINED = os.path.join(
        #                 resume, "checkpoints", checkpoint)
        #             break
        #     if os.path.exists(os.path.join(resume, "wandb")):
        #         wandb_list = sorted(os.listdir(os.path.join(resume, "wandb")),
        #                             reverse=True)
        #         for item in wandb_list:
        #             if "run-" in item:
        #                 cfg.LOGGER.WANDB.RESUME_ID = item.split("-")[-1]
            checkpoints = resume
            cfg.TRAIN.PRETRAINED = checkpoints

        else:
            raise ValueError("Resume path is not right.")
    # set seed
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
                                                 sub_dir="tensorboard",
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

    # callbacks
    callbacks = [
        pl.callbacks.RichProgressBar(),
        ProgressLogger(metric_monitor=metric_monitor),
        # ModelCheckpoint(dirpath=os.path.join(cfg.FOLDER_EXP,'checkpoints'),filename='latest-{epoch}',every_n_epochs=1,save_top_k=1,save_last=True,save_on_train_epoch_end=True),
        ModelCheckpoint(
            dirpath=os.path.join(cfg.FOLDER_EXP, "checkpoints"),
            filename="{epoch}",
            monitor="step",
            mode="max",
            every_n_epochs=cfg.LOGGER.SACE_CHECKPOINT_EPOCH,
            save_top_k=-1,
            save_last=False,
            save_on_train_epoch_end=True,
        ),
    ]
    logger.info("Callbacks initialized")

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
        log_every_n_steps=cfg.LOGGER.VAL_EVERY_STEPS,
        deterministic=False,
        detect_anomaly=False,
        enable_progress_bar=True,
        logger=loggers,
        callbacks=callbacks,
        # check_val_every_n_epoch=cfg.LOGGER.VAL_EVERY_STEPS,
        check_val_every_n_epoch=1
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

    if cfg.TRAIN.PRETRAINED:
        logger.info(f"[Loader] Loading weights from {cfg.TRAIN.PRETRAINED}")
        
        # 读取 Checkpoint
        ckpt = torch.load(cfg.TRAIN.PRETRAINED, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        
        # 获取当前模型的 Keys (作为真理标准)
        current_model_keys = set(model.state_dict().keys())
        
        # 遍历 Checkpoint 里的所有键
        for k, v in state_dict.items():
            # 1. 过滤掉不兼容的位置编码 (MCM-LDM 遗留问题)
            if "sequence_pos_encoding.pe" in k:
                continue
            
            # 2. 情况 A: 完美匹配 (你的 Projector, FiLM, 或者已经是 denoiser.xxx 的键)
            if k in current_model_keys:
                new_state_dict[k] = v
                
            # 3. 情况 B: 原始 MCM-LDM 权重 (缺 "denoiser." 前缀)
            # 比如 ckpt 里是 "blocks.0...", 但模型里是 "denoiser.blocks.0..."
            elif f"denoiser.{k}" in current_model_keys:
                new_state_dict[f"denoiser.{k}"] = v
                
            # 4. 其他情况: 确实不匹配的键，忽略
            else:
                pass

        # [手动诊断] 计算 Missing / Unexpected Keys
        ckpt_keys_processed = set(new_state_dict.keys())
        
        missing_keys = list(current_model_keys - ckpt_keys_processed)
        unexpected_keys = list(set(state_dict.keys()) - current_model_keys) # 粗略统计
        
        # 打印详细报告
        logger.info(f"====== Weight Loading Report ======")
        logger.info(f"Loaded Keys: {len(new_state_dict)}")
        logger.info(f"MISSING Keys (Init Randomly): {len(missing_keys)}")
        
        # 关键检查：看看我们的新模块是否在 Missing 列表里
        # 如果我们在 Resume，这些应该都不在 Missing 里
        check_modules = ["scene_projector", "film_mlp", "scene_vision_encoder"]
        for mod in check_modules:
            is_missing = any(mod in k for k in missing_keys)
            status = "MISSING (Random Init)" if is_missing else "LOADED (Success)"
            logger.info(f"  > Module '{mod}': {status}")
            
        logger.info(f"===================================")

        # 执行加载
        model.load_state_dict(new_state_dict, strict=False)
        
        # [安全验证] 验证主干网络是否加载成功
        # 检查第一层 Attention 的权重是否存在于 missing_keys 中
        backbone_key = "denoiser.blocks.0.attn.qkv.weight"
        if backbone_key in missing_keys:
            logger.error(f"!!! CRITICAL FAILURE !!! Backbone key '{backbone_key}' was NOT loaded.")
            logger.error("Your model is running with RANDOM weights. Please check 'num_layers' in yaml or key prefixes.")
            # 在这里抛出异常，阻止无效训练
            raise RuntimeError("Denoiser backbone weights failed to load.")
        else:
            logger.info(">>> SUCCESS: Denoiser backbone loaded. Ready for fine-tuning.")

    # =========================================================
    # [PhysiMoS] 强化版权重加载逻辑 - 结束
    # =========================================================

    # fitting
    if cfg.TRAIN.RESUME:
        trainer.fit(model,
                    datamodule=datasets[0],
                    ckpt_path=cfg.TRAIN.PRETRAINED)
    else:
        trainer.fit(model, datamodule=datasets[0])

    # checkpoint
    checkpoint_folder = trainer.checkpoint_callback.dirpath
    logger.info(f"The checkpoints are stored in {checkpoint_folder}")
    logger.info(
        f"The outputs of this experiment are stored in {cfg.FOLDER_EXP}")

    # end
    logger.info("Training ends!")


if __name__ == "__main__":
    main()