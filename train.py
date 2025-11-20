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
        backcfg = cfg.TRAIN.copy()
        if os.path.exists(resume):
            file_list = sorted(os.listdir(resume), reverse=True)
            for item in file_list:
                if item.endswith(".yaml"):
                    cfg = OmegaConf.load(os.path.join(resume, item))
                    cfg.TRAIN = backcfg
                    break
            checkpoints = sorted(os.listdir(os.path.join(
                resume, "checkpoints")),
                                 key=lambda x: int(x[6:-5]),
                                 reverse=True)
            for checkpoint in checkpoints:
                if "epoch=" in checkpoint:
                    cfg.TRAIN.PRETRAINED = os.path.join(
                        resume, "checkpoints", checkpoint)
                    break
            if os.path.exists(os.path.join(resume, "wandb")):
                wandb_list = sorted(os.listdir(os.path.join(resume, "wandb")),
                                    reverse=True)
                for item in wandb_list:
                    if "run-" in item:
                        cfg.LOGGER.WANDB.RESUME_ID = item.split("-")[-1]

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

    # 2. 加载 Denoiser 权重 (Stage 2 - 核心)
    if cfg.TRAIN.PRETRAINED:
        logger.info(f"[Loader] Loading Denoiser from {cfg.TRAIN.PRETRAINED}")
        
        # 读取 Checkpoint
        ckpt = torch.load(cfg.TRAIN.PRETRAINED, map_location="cpu")
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt # 有些ckpt直接就是dict
            
        # [智能前缀处理] 
        # 问题：denoiser.ckpt 里的键是 "blocks.0..." 还是 "denoiser.blocks.0..."?
        # 我们的 model 是 MLD 类，它的 denoiser 在 model.denoiser 下。
        # 所以我们需要让所有键都以 "denoiser." 开头。
        
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        
        # 侦测 Checkpoint 里的前缀格式
        keys_list = list(state_dict.keys())
        has_denoiser_prefix = any(k.startswith("denoiser.") for k in keys_list)
        
        logger.info(f"[Loader] Detected 'denoiser.' prefix in checkpoint? {has_denoiser_prefix}")

        for k, v in state_dict.items():
            # 过滤掉不兼容的旧位置编码 (Sequence Length 可能不同)
            if "sequence_pos_encoding.pe" in k:
                continue
                
            if has_denoiser_prefix:
                # 格式匹配：直接用
                if k.startswith("denoiser."):
                    new_state_dict[k] = v
            else:
                # 格式不匹配：假设 checkpoint 里全是 denoiser 的参数，手动加上前缀
                # 例如: "blocks.0.attn..." -> "denoiser.blocks.0.attn..."
                new_key = f"denoiser.{k}"
                new_state_dict[new_key] = v

        # [手动诊断] 计算 Missing / Unexpected Keys
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(new_state_dict.keys())
        
        missing_keys = list(model_keys - ckpt_keys)
        unexpected_keys = list(ckpt_keys - model_keys)
        
        # 打印详细报告
        logger.info(f"====== Weight Loading Report ======")
        logger.info(f"Keys in Model: {len(model_keys)} | Keys in Ckpt: {len(ckpt_keys)}")
        logger.info(f"MISSING Keys (Init Randomly): {len(missing_keys)}")
        
        # 关键检查：Physics Encoder 是否在 missing 列表里？(应该在)
        phys_missing = any("physics_encoder" in k for k in missing_keys)
        logger.info(f"  - Physics Encoder is missing (Expected)? {phys_missing}")
        
        logger.info(f"UNEXPECTED Keys (Ignored): {len(unexpected_keys)}")
        # 打印部分 Unexpected 以便排查
        for k in sorted(unexpected_keys)[:5]: logger.info(f"  - UNEXP: {k}")
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
