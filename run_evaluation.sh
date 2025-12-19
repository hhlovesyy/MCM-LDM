#!/bin/bash

# ==============================================================================
# 配置区域 (只需修改以下变量)
# ==============================================================================

# 基础配置路径
CONFIG_MLD="./configs/config_mld_humanml3d.yaml"
CONFIG_ASSETS="./configs/assets.yaml"
SCALE="2.5"

# 模型名称 (用于生成输出文件名的一部分)
EXP_NAME="SceMoDiff_Evaluation" # 假设您有一个统一的实验名称
EVAL_ID=0

# 期望的输出基础目录（两个生成脚本的输出应该都在这个目录下）
# 请根据您的实际配置调整这个路径，例如：/root/autodl-tmp/MCM-LDM/results/mld/
OUTPUT_BASE_DIR="/root/autodl-tmp/MyRepository/MCM-LDM/results/mld/${EXP_NAME}"

# 评估脚本的根目录
EVAL_ROOT_DIR="/root/autodl-tmp/MCM-LDM_Evaluation/MCM-LDM_Evaluation"


# ==============================================================================
# 自动生成文件名 (根据您提供的 Python 格式)
# save_path = os.path.join(save_path, eval_name+'-'+str(eval_id)+'_expname_'+str(cfg.NAME)+"_scale_"+str(cfg.DEMO.scale).replace('.','-') + '.pkl')
# ==============================================================================

# 替换 '.' 为 '-'
SCALE_FORMATTED=$(echo "$SCALE" | tr . -)

# FMD/CRA 输出文件路径
FMD_CRA_PKL_NAME="crafmd-${EVAL_ID}_expname_${EXP_NAME}_scale_${SCALE_FORMATTED}.pkl"
FMD_CRA_PKL_PATH="${OUTPUT_BASE_DIR}/${FMD_CRA_PKL_NAME}"

# FMD_CRA_PKL_PATH="/root/autodl-tmp/MyRepository/MCM-LDM/results/mld/testBaseline0904/content-0_expname_testBaseline0904_scale_2-5.pkl"
# SRA_PKL_PATH="/root/autodl-tmp/MyRepository/MCM-LDM/results/mld/testBaseline0904/style-0_expname_testBaseline0904_scale_2-5.pkl"

# SRA 输出文件路径
SRA_PKL_NAME="sra-${EVAL_ID}_expname_${EXP_NAME}_scale_${SCALE_FORMATTED}.pkl"
SRA_PKL_PATH="${OUTPUT_BASE_DIR}/${SRA_PKL_NAME}"

# 创建输出目录，确保存在
mkdir -p "${OUTPUT_BASE_DIR}"

echo "====================================================="
echo "🔄 开始执行数据生成脚本 (并行)"
echo "FMD/CRA 结果将保存在: ${FMD_CRA_PKL_PATH}"
echo "SRA 结果将保存在: ${SRA_PKL_PATH}"
echo "====================================================="

# --- 阶段 1: 数据生成 (串行执行，更容易排查问题) ---

# 1.1 FMD/CRA 数据生成
echo "Running FMD/CRA data generation..."
# 将标准输出 (1) 和标准错误 (2) 都重定向到一个日志文件
python demo_transfer_crafmd.py \
    --cfg "${CONFIG_MLD}" \
    --cfg_assets "${CONFIG_ASSETS}" \
    --style_motion_dir demo/content_test_feats \
    --content_motion_dir demo/content_test_feats \
    --output_path "${FMD_CRA_PKL_PATH}" \
    --scale "${SCALE}" > crafmd_gen.log 2>&1

# 检查上一个命令的退出状态。非零表示失败。
if [ $? -ne 0 ]; then
    echo "🚨 ERROR: demo_transfer_crafmd.py 运行失败。查看 crafmd_gen.log 获取详情。"
    exit 1
fi

# 1.2 SRA 数据生成
echo "Running SRA data generation..."
python demo_transfer_sra.py \
    --cfg "${CONFIG_MLD}" \
    --cfg_assets "${CONFIG_ASSETS}" \
    --style_motion_dir demo/style_test_feats \
    --content_motion_dir demo/style_test_feats \
    --output_path "${SRA_PKL_PATH}" \
    --scale "${SCALE}" > sra_gen.log 2>&1

if [ $? -ne 0 ]; then
    echo "🚨 ERROR: demo_transfer_sra.py 运行失败。查看 sra_gen.log 获取详情。"
    exit 1
fi

# # 等待所有后台任务完成
# wait
echo "✅ 数据生成脚本执行完毕。"
echo "FMD/CRA 结果将保存在: ${FMD_CRA_PKL_PATH}"
echo "SRA 结果将保存在: ${SRA_PKL_PATH}"

echo "====================================================="
echo "📈 开始执行评估脚本 (串行)"
echo "评估脚本目录: ${EVAL_ROOT_DIR}"
echo "====================================================="


# --- 阶段 2: 模型评估 (串行执行) ---
# --- 阶段 2: 模型评估 (串行执行) ---
# 切换到评估脚本的目录
cd "${EVAL_ROOT_DIR}" 

# 2.1 运行 FMD/CRA 评估
echo "运行 eval_FMD_CRA.py ..."
# 注意：这里文件名前面不需要 ${EVAL_ROOT_DIR}/ 了，因为已经 cd 进去了
python eval_FMD_CRA.py \
    --pkl_path "${FMD_CRA_PKL_PATH}"

# 2.2 运行 SRA 评估
echo "运行 eval_SRA.py ..."
python eval_SRA.py \
    --pkl_path "${SRA_PKL_PATH}"

cd - > /dev/null  # 返回原始目录并隐藏输出

echo "====================================================="
echo "🎉 所有评估任务完成！"
echo "====================================================="