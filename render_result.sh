# #!/bin/bash
# # 使用方式：bash render_result.sh /root/autodl-tmp/MyRepository/MotionLCM/MotionLCM/npyInput_sceneMoDiff/FullModel/Windy 50 4
# # ==============================================================================
# #      SceMoDiff 自动化批处理脚本 (argparse 兼容版)
# # ==============================================================================

# # --- 1. 参数配置与检查 ---

# # ... (颜色代码和参数接收部分保持不变) ...
# GREEN='\033[0;32m'
# YELLOW='\033[1;33m'
# RED='\033[0;31m'
# NC='\033[0m'

# INPUT_NPY_FOLDER=$1
# SMPLIFY_ITERS=${2:-50}
# RENDER_NUM=${3:-4}

# # ... (参数检查部分保持不变) ...
# if [ -z "$INPUT_NPY_FOLDER" ]; then
#     echo -e "${RED}错误: 未提供输入的 .npy 文件夹路径！${NC}"
#     echo "用法: $0 /path/to/your/npy_folder [smplify_iters] [render_num]"
#     exit 1
# fi
# if [ ! -d "$INPUT_NPY_FOLDER" ]; then
#     echo -e "${RED}错误: 输入文件夹不存在: $INPUT_NPY_FOLDER${NC}"
#     exit 1
# fi

# # --- 2. 自动生成路径 ---
# OUTPUT_PKL_FOLDER="${INPUT_NPY_FOLDER}_pkl"

# # ... (打印初始信息部分保持不变) ...

# # --- 3. 逐一执行三个核心步骤 ---

# # --- 步骤 1: NPY -> PKL ---
# echo -e "\n${YELLOW}[步骤 1/3] 正在将 NPY 文件转换为 PKL...${NC}"

# # 【核心修正】使用 --key=value 的格式来调用 Python 脚本
# # 假设你的 npy2pkl.py 只需要 --input_folder，然后自动创建输出文件夹
# # 如果它也需要 --output_folder，就像这样写:
# # python npy2pkl.py --input_folder="$INPUT_NPY_FOLDER" --output_folder="$OUTPUT_PKL_FOLDER"
# python npy2pkl.py --input_folder="$INPUT_NPY_FOLDER"

# if [ $? -ne 0 ]; then
#     echo -e "${RED}错误: 步骤 1 (npy2pkl.py) 执行失败。脚本已中止。${NC}"
#     exit 1
# fi
# echo -e "${GREEN}步骤 1 完成。PKL 文件已生成于: ${OUTPUT_PKL_FOLDER}${NC}"
# echo -e "${GREEN}-------------------------------------------${NC}"


# # --- 步骤 2: 拟合 SMPL-X ---
# echo -e "\n${YELLOW}[步骤 2/3] 正在为 PKL 文件拟合 SMPL-X 网格...${NC}"

# # 【核心修正】同样使用 --dir=... 的格式
# python fit_origin.py --dir="$OUTPUT_PKL_FOLDER" --num_smplify_iters="$SMPLIFY_ITERS"

# if [ $? -ne 0 ]; then
#     echo -e "${RED}错误: 步骤 2 (fit_origin.py) 执行失败。脚本已中止。${NC}"
#     exit 1
# fi
# echo -e "${GREEN}步骤 2 完成。${NC}"
# echo -e "${GREEN}-------------------------------------------${NC}"


# # --- 步骤 3: Blender 渲染 ---
# echo -e "\n${YELLOW}[步骤 3/3] 正在启动 Blender 进行背景渲染...${NC}"

# # 【核心修正】Blender 的参数传递方式很特殊，-- --dir=... 是正确的
# blender --background --python render.py -- --dir="$OUTPUT_PKL_FOLDER" --num="$RENDER_NUM"

# if [ $? -ne 0 ]; then
#     echo -e "${RED}错误: 步骤 3 (Blender 渲染) 执行失败。脚本已中止。${NC}"
#     exit 1
# fi
# echo -e "${GREEN}步骤 3 完成。${NC}"
# echo -e "${GREEN}===========================================${NC}"
# echo -e "${YELLOW}🎉 所有步骤已成功完成！${NC}"
# echo -e "最终的渲染结果视频，请检查 Blender 脚本中指定的输出路径。"
# # echo -e "通常位于: ${OUTPUT_PKL_FOLDER}/render"
# echo -e "${GREEN}===========================================${NC}"

#!/bin/bash
# ==============================================================================
#      SceMoDiff 自动化渲染全流程脚本 (NPY -> PKL -> SMPL -> Blender)
# ==============================================================================

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# --- 1. 设置默认参数 ---
INPUT_NPY_FOLDER=""
SMPLIFY_ITERS=50
RENDER_MODE="sequence"  # video, sequence, frame
RENDER_NUM=4            # for sequence
FPS=20                  # for video
EXACT_FRAME=0.5         # for frame
RES="high"
GT="False"

# --- 2. 解析参数 (Parse Arguments) ---
while [[ $# -gt 0 ]]; do
  case $1 in
    --input_folder)
      INPUT_NPY_FOLDER="$2"
      shift 2
      ;;
    --iters)
      SMPLIFY_ITERS="$2"
      shift 2
      ;;
    --mode)
      RENDER_MODE="$2"
      shift 2
      ;;
    --num)
      RENDER_NUM="$2"
      shift 2
      ;;
    --fps)
      FPS="$2"
      shift 2
      ;;
    --exact_frame)
      EXACT_FRAME="$2"
      shift 2
      ;;
    --res)
      RES="$2"
      shift 2
      ;;
    --gt)
      GT="True"
      shift # boolean flag, no value needed usually, but logic depends on python arg parser
      ;;
    *)
      echo "Unknown argument: $1"
      shift
      ;;
  esac
done

# --- 3. 检查必填项 ---
if [ -z "$INPUT_NPY_FOLDER" ]; then
    echo -e "${RED}错误: 未提供输入的 .npy 文件夹路径 (--input_folder)${NC}"
    exit 1
fi
if [ ! -d "$INPUT_NPY_FOLDER" ]; then
    echo -e "${RED}错误: 输入文件夹不存在: $INPUT_NPY_FOLDER${NC}"
    exit 1
fi

OUTPUT_PKL_FOLDER="${INPUT_NPY_FOLDER}_pkl"

echo -e "${GREEN}=== SceMoDiff Render Pipeline ===${NC}"
echo "Input: $INPUT_NPY_FOLDER"
echo "Output PKL: $OUTPUT_PKL_FOLDER"
echo "Mode: $RENDER_MODE"

# --- 4. 执行流程 ---

# [步骤 1] NPY -> PKL
echo -e "\n${YELLOW}[1/3] NPY -> PKL...${NC}"
python npy2pkl.py --input_folder="$INPUT_NPY_FOLDER"
if [ $? -ne 0 ]; then echo -e "${RED}Failed at Step 1${NC}"; exit 1; fi

# [步骤 2] SMPL Fitting
echo -e "\n${YELLOW}[2/3] SMPL Fitting...${NC}"
python fit.py --dir="$OUTPUT_PKL_FOLDER" --num_smplify_iters="$SMPLIFY_ITERS"
if [ $? -ne 0 ]; then echo -e "${RED}Failed at Step 2${NC}"; exit 1; fi

# [步骤 3] Blender Render
echo -e "\n${YELLOW}[3/3] Blender Rendering...${NC}"

# 构造 Blender 参数字符串
BLENDER_ARGS="--dir=$OUTPUT_PKL_FOLDER --mode=$RENDER_MODE --res=$RES"

if [ "$GT" == "True" ]; then
    BLENDER_ARGS="$BLENDER_ARGS --gt=True"
fi

if [ "$RENDER_MODE" == "sequence" ]; then
    BLENDER_ARGS="$BLENDER_ARGS --num=$RENDER_NUM"
elif [ "$RENDER_MODE" == "video" ]; then
    BLENDER_ARGS="$BLENDER_ARGS --fps=$FPS"
elif [ "$RENDER_MODE" == "frame" ]; then
    BLENDER_ARGS="$BLENDER_ARGS --exact_frame=$EXACT_FRAME"
fi

echo "Blender Args: $BLENDER_ARGS"

# 执行 Blender
blender --background --python render.py -- $BLENDER_ARGS

if [ $? -ne 0 ]; then echo -e "${RED}Failed at Step 3${NC}"; exit 1; fi

echo -e "\n${GREEN}🎉 All Done! Output saved in: $OUTPUT_PKL_FOLDER${NC}"