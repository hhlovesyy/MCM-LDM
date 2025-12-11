#!/bin/bash

# ==============================================================================
# 0. 配置区域 (只需修改以下变量)
# ==============================================================================

# 基础配置路径
CONFIG_MLD="/root/autodl-tmp/MyRepository/MCM-LDM/configs/eval_temp_SceneMo_1211_1835_Full_FiLM_Loss.yaml"
CONFIG_ASSETS="./configs/assets.yaml"
SCALE="2.5"

# 模型名称 (用于生成输出文件名的一部分)
EXP_NAME="SceneMo_1211_1835_Full_FiLM_Loss_Eval" # 假设您有一个统一的实验名称
EVAL_ID=0

# 期望的输出基础目录（两个生成脚本的输出应该都在这个目录下）
OUTPUT_BASE_DIR="/root/autodl-tmp/MyRepository/MCM-LDM/results/mld/${EXP_NAME}"

# 评估脚本的根目录
EVAL_ROOT_DIR="/root/autodl-tmp/MCM-LDM_Evaluation/MCM-LDM_Evaluation"

# 最终结果输出文件
RESULTS_FILE="evaluation_metrics_${EXP_NAME}.txt"

# 默认不跳过生成阶段
SKIP_GENERATION=false

# ==============================================================================
# 1. 解析命令行参数
# ==============================================================================
for arg in "$@"; do
    case $arg in
        --skip-gen)
        SKIP_GENERATION=true
        shift # 移除参数
        ;;
        *)
        # 忽略其他参数
        ;;
    esac
done

# ==============================================================================
# 2. 自动生成文件名 (Bash变量)
# ==============================================================================

# 替换 '.' 为 '-'
SCALE_FORMATTED=$(echo "$SCALE" | tr . -)

# FMD/CRA 输出文件路径
FMD_CRA_PKL_NAME="crafmd-${EVAL_ID}_expname_${EXP_NAME}_scale_${SCALE_FORMATTED}.pkl"
FMD_CRA_PKL_PATH="${OUTPUT_BASE_DIR}/${FMD_CRA_PKL_NAME}"

# SRA 输出文件路径
SRA_PKL_NAME="sra-${EVAL_ID}_expname_${EXP_NAME}_scale_${SCALE_FORMATTED}.pkl"
SRA_PKL_PATH="${OUTPUT_BASE_DIR}/${SRA_PKL_NAME}"

# 创建输出目录，确保存在
mkdir -p "${OUTPUT_BASE_DIR}"

# 初始化结果文件
echo "--- Evaluation Metrics for ${EXP_NAME} ---" > "${RESULTS_FILE}"
echo "Run Date: $(date)" >> "${RESULTS_FILE}"
echo "=========================================" >> "${RESULTS_FILE}"

# 定义一个临时文件来存储评估脚本的所有输出
TEMP_LOG="evaluation_temp.log"
> "${TEMP_LOG}" # 清空临时日志


# ==============================================================================
# 3. 阶段 1: 数据生成 (受 --skip-gen 开关控制)
# ==============================================================================
if [ "$SKIP_GENERATION" = false ]; then
    echo "====================================================="
    echo "🔄 阶段 1: 开始执行数据生成脚本 (串行捕获日志)"
    echo "FMD/CRA 结果目标路径: ${FMD_CRA_PKL_PATH}"
    echo "SRA 结果目标路径: ${SRA_PKL_PATH}"
    echo "====================================================="
    
    # 3.1 FMD/CRA 数据生成
    echo "Running FMD/CRA data generation..."
    python demo_transfer_crafmd.py \
        --cfg "${CONFIG_MLD}" \
        --cfg_assets "${CONFIG_ASSETS}" \
        --style_motion_dir demo/content_test_feats \
        --content_motion_dir demo/content_test_feats \
        --output_path "${FMD_CRA_PKL_PATH}" \
        --scale "${SCALE}" > crafmd_gen.log 2>&1
    
    if [ $? -ne 0 ]; then
        echo "🚨 ERROR: demo_transfer_crafmd.py 运行失败。查看 crafmd_gen.log 获取详情。"
        exit 1
    fi

    # 3.2 SRA 数据生成
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
    echo "✅ 数据生成脚本执行完毕。"
else
    echo "====================================================="
    echo "⏩ 阶段 1: 跳过数据生成 (使用 --skip-gen 参数)"
    echo "====================================================="
fi


# ==============================================================================
# 4. 阶段 2: 模型评估与指标提取 (使用子shell捕获全部输出)
# ==============================================================================

echo "====================================================="
echo "📈 阶段 2: 开始执行评估脚本并捕获输出"
echo "评估脚本目录: ${EVAL_ROOT_DIR}"
echo "====================================================="

# 定义一个变量来存储评估脚本的所有输出
EVAL_OUTPUT=""

# 切换到评估脚本的目录，以正确解析相对路径
cd "${EVAL_ROOT_DIR}"

# 使用子shell () 捕获所有输出到 EVAL_OUTPUT 变量
EVAL_OUTPUT=$(
    # 4.1 运行 FMD/CRA 评估
    echo "运行 eval_FMD_CRA.py ..." >&2 # 提示信息输出到 stderr，不进入捕获
    # 捕获其所有输出 (stdout + stderr)
    python eval_FMD_CRA.py --pkl_path "${FMD_CRA_PKL_PATH}" 2>&1
    
    # 4.2 运行 SRA 评估
    echo "运行 eval_SRA.py ..." >&2 # 提示信息输出到 stderr，不进入捕获
    # 捕获其所有输出 (stdout + stderr)
    python eval_SRA.py --pkl_path "${SRA_PKL_PATH}" 2>&1
)

# 切换回原始目录
cd - > /dev/null

# 将捕获到的所有输出写入 TEMP_LOG 文件
echo "${EVAL_OUTPUT}" > "${TEMP_LOG}"

echo "✅ 评估脚本输出已捕获到 ${TEMP_LOG}。"
echo "以下是捕获的内容："
echo "---"
cat "${TEMP_LOG}"
echo "---"

# ==============================================================================
# 5. 阶段 3: 使用 Python 提取、重命名和存储指标
# ==============================================================================

echo "====================================================="
echo "📊 提取关键指标..."

# 运行 Python 脚本进行指标解析和重命名
# 参数 1: 临时日志文件路径
# 参数 2: 最终结果文件路径
# 参数 3: 实验名称 (用于日志头部)
python extract_metrics.py "${TEMP_LOG}" "${RESULTS_FILE}" "${EXP_NAME}"

# 打印完成信息
echo "====================================================="
echo "🎉 所有评估任务完成！"
echo "✨ 最终关键指标已存储到 ${RESULTS_FILE}，并打印在控制台。"

# 清理临时文件 (可选)
# 修复: 确保 rm -f 后面接正确的变量名
# rm -f crafmd_gen.log sra_gen.log "${TEMP_LOG}"

# (base) root@autodl-container-32de4894e4-6d817059:~/autodl-tmp/MyRepository/MCM-LDM# python /root/autodl-tmp/MCM-LDM_Evaluation/MCM-LDM_Evaluation/calc_traj_metrics.py --pkl_path=/root/autodl-tmp/MyRepository/MCM-LDM/results/mld/SceMoDiff_Evaluation/crafmd-0_expname_SceMoDiff_Evaluation_scale_2-5.pkl --gt_joints_dir=/root/autodl-tmp/MyRepository/MCM-LDM/demo/content_test_joints
echo "====================================================="
echo "轨迹和脚滑的相关指标，追加到 ${RESULTS_FILE}，并打印在控制台。"
python /root/autodl-tmp/MyRepository/MCM-LDM/calc_traj_metrics.py --pkl_path="${FMD_CRA_PKL_PATH}" --gt_joints_dir="/root/autodl-tmp/MyRepository/MCM-LDM/demo/content_test_joints" 2>&1 | tee -a "${RESULTS_FILE}"
echo "====================================================="