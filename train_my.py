import os
from pprint import pformat
import sys

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
import datetime


def main():
     # --- 开始重定向 stdout 和 stderr ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_log_{timestamp}.txt"
    
    # original_stdout = sys.stdout  # 保存原始 stdout
    # original_stderr = sys.stderr  # 保存原始 stderr
    
    # print(f"所有输出将被重定向到文件: {log_filename}") # 这条会打印到控制台

    # 'a' 表示追加模式，如果脚本多次运行会追加到同一个文件（如果文件名不变）
    # 'w' 表示写入模式，每次运行会覆盖旧文件
    # 建议使用 'w' 配合时间戳文件名，每次运行都有独立的日志
    log_file = open(log_filename, 'w', encoding='utf-8')
    
    # sys.stdout = log_file
    # sys.stderr = log_file
    # --- 重定向结束 ---
     # note: 2025.5.23 解决报错：CUDA not avaliable
    try:
        import torch
        torch.cuda.init()  # 显式初始化 CUDA 上下文
        print(f"[强制初始化] CUDA 状态: is_available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}")
        # parse options
        cfg = parse_args()  # parse config file

        # create logger
        logger = create_logger(cfg, phase="train")
        # note： 2025.5.23 添加，因为只有一个GPU
        cfg.DEVICE = [0]
        print("before if cfg.TRAIN.RESUME:", cfg.DEVICE) 
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
                        print(f"20250528: Resume from {checkpoint}")
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

        print("before cfg.ACCELERATOR:", cfg.DEVICE)
        # 2025.5.28 解决报错：CUDA not avaliable
        cfg.DEVICE = [0]
        # gpu setting
        if cfg.ACCELERATOR == "gpu":
            os.environ["PYTHONWARNINGS"] = "ignore"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            # os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(str(x) for x in cfg.DEVICE)

        # tensorboard logger and wandb logger
        loggers = []
        # 2025.05.29 为了省时间先写死了
        cfg.LOGGER.WANDB.PROJECT = False
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

        # --- START: Print ONLY trainable parameters from training script ---
        # print("\n" + "="*30 + " Trainable Parameters (requires_grad=True) " + "="*30)
        # trainable_params_count = 0
        # total_params_count = 0
        # trainable_parameter_names = []

        # for name, param in model.named_parameters(): # Accessing the model instance
        #     total_params_count += param.numel()
        #     if param.requires_grad:
        #         trainable_params_count += param.numel()
        #         trainable_parameter_names.append(name)
        #         print(f"TRAINABLE: {name:<60} | Size: {list(param.shape)}")

        # print(f"\n--- Summary ---")
        # print(f"Total model parameters: {total_params_count}")
        # print(f"Total TRAINABLE parameters: {trainable_params_count}")
        # if not trainable_parameter_names:
        #     print("No parameters are set to be trainable (requires_grad=True).")
        # else:
        #     print(f"Number of distinct trainable parameter tensors: {len(trainable_parameter_names)}")
        # print("="*80 + "\n")
        # # --- END: Print ONLY trainable parameters from training script ---

        # # ... (optimizer setup if not in LightningModule, callbacks, trainer instantiation) ...
        # trainer.fit(model, datamodule=datasets[0])

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
        # print cfg.DEVICE
        # logger.info("Creating trainer...in device 2025.5.28", cfg.DEVICE)
        trainer = pl.Trainer(
            benchmark=False,
            max_epochs=cfg.TRAIN.END_EPOCH,
            accelerator=cfg.ACCELERATOR,
            # 2025.5.28 解决报错：CUDA not avaliable
            devices=cfg.DEVICE,
            # devices='cuda:0',
            #strategy=ddp_strategy,
            # move_metrics_to_cpu=True,
            default_root_dir=cfg.FOLDER_EXP,
            log_every_n_steps=cfg.LOGGER.VAL_EVERY_STEPS,
            deterministic=False,
            detect_anomaly=False,
            enable_progress_bar=True,
            logger=loggers,
            callbacks=callbacks,
            check_val_every_n_epoch=cfg.LOGGER.VAL_EVERY_STEPS,
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

        print("cfg.TRAIN.PRETRAINED:", cfg.TRAIN.PRETRAINED) # checkpoints epoch=1499
        if cfg.TRAIN.PRETRAINED:
            logger.info("Loading pretrain mode from {}".format(
                cfg.TRAIN.PRETRAINED))
            logger.info("Attention! VAE will be recovered")
            state_dict = torch.load(cfg.TRAIN.PRETRAINED,
                                    map_location="cpu")["state_dict"]
            # remove mismatched and unused params
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            # print("debug dictionary")
            for k, v in state_dict.items():
                # print(k)
                if k not in ["denoiser.sequence_pos_encoding.pe"]:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)
            # 查看model有哪些dict
            # print("===============debug model================")
            # for k, v in model.state_dict().items():
                # print(k)

        # fitting
        print("cfg.TRAIN.RESUME:", cfg.TRAIN.RESUME) # cfg.TRAIN.RESUME: /root/autodl-tmp/MCM-LDM/experiments/mld/debug--test111
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
    except Exception as e:
        # 确保即使发生异常，错误信息也会被记录到日志文件
        import traceback
        print("发生未捕获的异常:", file=sys.__stderr__) # 使用原始stderr打印到控制台，以防日志文件写入问题
        traceback.print_exc(file=sys.__stderr__)
        traceback.print_exc(file=log_file) # 也记录到日志文件
        raise # 重新抛出异常

    finally:
        # --- 恢复 stdout 和 stderr ---
        # sys.stdout = original_stdout
        # sys.stderr = original_stderr
        log_file.close()
        print(f"日志已保存到: {log_filename}") # 这条会打印到控制台


if __name__ == "__main__":
    main()
