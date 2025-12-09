#!/bin/bash

# 确保激活了 Conda/虚拟环境（如果需要）
# source activate your_env_name

# 定义 Python 脚本路径
TRAIN_SCRIPT="/root/autodl-tmp/MyRepository/MCM-LDM/train.py"

# 定义参数
CONFIG_FILE="configs/scenemodiff_train_all.yaml"
ASSETS_FILE="configs/assets.yaml"
BATCH_SIZE="32"
DEBUG_MODE="--nodebug"

# 执行命令
python $TRAIN_SCRIPT \
    --cfg $CONFIG_FILE \
    --cfg_assets $ASSETS_FILE \
    --batch_size $BATCH_SIZE \
    $DEBUG_MODE