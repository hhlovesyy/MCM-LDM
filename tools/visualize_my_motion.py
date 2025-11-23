# --- Python Script: Visualize a Single Motion .npy File ---
#
# 描述:
#   本脚本旨在直接可视化一个已经处理好的动作.npy文件。
#   它复用了`demo.py`中使用的`visual_pos`函数，
#   但剥离了所有模型加载和AI推理的复杂逻辑，专注于可视化本身。
#
# 如何使用:
# 1. 将此脚本与您的`visual.py`文件放在同一个目录下，
#    或者确保`visual.py`所在的路径在Python的搜索路径中。
#
# 2. 修改下面的【核心配置】部分，填入您想要可视化的.npy文件路径。
#
# 3. 运行脚本 (在终端中):
#    python visualize_my_motion.py
#
# --------------------------------------------------------------------

import os
from pathlib import Path
import logging

# 关键：从您项目中的visual模块导入可视化函数
# 请确保这个导入是有效的
try:
    from visual import visual_pos
except ImportError:
    print("错误: 无法导入'visual_pos'函数。")
    print("请确保本脚本与'visual.py'在同一目录，或者'visual.py'所在的路径已被添加到PYTHONPATH。")
    exit()

# ====================================================================
# --- (!!!) 核心配置 (!!!) ---
# ====================================================================

# 1. [输入] 请将这里修改为您想要可视化的.npy文件的【绝对路径】
#    这个文件应该是我们之前生成的，例如 'YHSMPLClapping100.npy'
NPY_FILE_TO_VISUALIZE = "/root/autodl-tmp/HumanML3D/HumanML3D/dataset_res/new_joints/W_0p0_Right_300k_0021.npy" # <-- 请修改我！把Blender产出的windows操作系统下的文件上传到Linux服务器上

# 2. [输出] 可视化视频将被保存在.npy文件相同的目录下
#    例如，输出文件将是 'YHSMPLClapping100.mp4'

# 3. [可选] 设置可视化参数
#    'fixed_camera': 固定机位，适合观察原地动作
#    'camera_follow': 摄像机跟随角色移动，适合观察行进动作
VIEW_MODE = 'fixed_camera' # 我们的拍手是原地动作，用固定机位效果最好

# ====================================================================

def main():
    """
    主函数，执行单个.npy文件的可视化。
    """
    # --- 1. 路径和文件名设置 ---
    npy_path = Path(NPY_FILE_TO_VISUALIZE)

    # 检查输入文件是否存在
    if not npy_path.exists():
        logging.error(f"错误: 输入的.npy文件不存在！路径: {npy_path}")
        return

    # 构建输出的.mp4文件路径
    # 例如：/path/to/your/file.npy -> /path/to/your/file.mp4
    mp4_path = npy_path.with_suffix(".mp4")
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("="*50)
    logging.info(f"🎬 开始可视化动作...")
    logging.info(f"   - 输入文件 (.npy): {npy_path}")
    logging.info(f"   - 输出文件 (.mp4): {mp4_path}")
    logging.info(f"   - 摄像机模式: {VIEW_MODE}")
    logging.info("="*50)

    # --- 2. 调用核心可视化函数 ---
    try:
        # 这是从demo.py中分析出的核心调用
        visual_pos(str(npy_path), str(mp4_path))
        
        logging.info("🎉 可视化成功！")
        logging.info(f"视频文件已保存至: {mp4_path}")

    except Exception as e:
        logging.error(f"❌ 在可视化过程中发生错误: {e}")
        logging.error("请检查 'visual_pos' 函数的实现以及您的环境配置 (例如，是否缺少OpenCV, Matplotlib等库)。")

if __name__ == '__main__':
    main()