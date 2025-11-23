import numpy as np
import pickle
import sys
import os

# --- 配置参数 ---
# 1. 输入文件夹 (包含所有要转换的 .npy 文件)
INPUT_DIR = "/root/autodl-tmp/MyRepository/MCM-LDM/results/demo_outputs/PhysiMoS_Finetune_v2_1122/fitInput" 
# 2. 输出文件夹 (将存放生成的 .pkl 文件)
OUTPUT_DIR = "visual_pkl" 
# 3. 文件名前缀 (用于对比，例如 'right_', 'left_', 'test_')
FILE_PREFIX = "" 
# 4. 默认的文本描述 (可以根据需要修改，但为了批量处理统一设定)
DEFAULT_TEXT_DESCRIPTION = "" 
# ------------------


def convert_npy_to_pkl(npy_file_path: str, pkl_output_path: str, text_description: str):
    """
    将一个 .npy 文件转换为特定的 .pkl 文件格式。

    参数:
    npy_file_path (str): 输入的 .npy 文件路径。
    pkl_output_path (str): 输出的 .pkl 文件路径。
    text_description (str): 描述动作的文本。
    """
    try:
        # 1. 加载 .npy 文件
        print(f"-> 正在加载: {os.path.basename(npy_file_path)}...")
        joints_data = np.load(npy_file_path)

        # 确保数据形状是三维的 (frames, joints, coordinates)
        if joints_data.ndim != 3:
            print(f"   [跳过] 错误: {os.path.basename(npy_file_path)} 数组形状不是三维。")
            return

        # 2. 从 .npy 数组的形状中计算 length
        length = joints_data.shape[0]

        # 3. 构造字典数据
        pkl_data = {
            'joints': joints_data,
            'text': text_description,
            'length': length,
            'hint': None
        }

        # 4. 将字典保存为 .pkl 文件
        with open(pkl_output_path, 'wb') as f:
            pickle.dump(pkl_data, f)
            
        print(f"   [成功] 保存为: {os.path.basename(pkl_output_path)} (长度: {length}, 形状: {joints_data.shape})")

    except FileNotFoundError:
        print(f"   [错误] 文件未找到。请检查路径: {npy_file_path}")
    except Exception as e:
        print(f"   [错误] 转换过程中发生错误: {e}")


if __name__ == "__main__":
    
    # 0. 准备工作
    current_dir = os.getcwd()
    input_full_path = os.path.join(current_dir, INPUT_DIR)
    output_full_path = os.path.join(current_dir, OUTPUT_DIR)
    
    # 创建输出文件夹
    if not os.path.exists(output_full_path):
        os.makedirs(output_full_path)
        print(f"创建输出文件夹: {output_full_path}")
    
    print("=" * 60)
    print("🤖 批量 NPY 转 PKL 脚本启动")
    print(f"   输入目录: {INPUT_DIR}")
    print(f"   输出目录: {OUTPUT_DIR}")
    print(f"   文件前缀: {FILE_PREFIX}")
    print(f"   默认描述: '{DEFAULT_TEXT_DESCRIPTION}'")
    print("=" * 60)

    # 1. 遍历输入文件夹中的文件
    processed_count = 0
    
    if not os.path.exists(input_full_path):
        print(f"\n[致命错误] 输入文件夹 '{INPUT_DIR}' 不存在。请创建该文件夹并将 .npy 文件放入其中。")
        sys.exit(1)

    for filename in os.listdir(input_full_path):
        if filename.endswith(".npy"):
            npy_file_path = os.path.join(input_full_path, filename)
            
            # 2. 生成新的 PKL 文件名 (加入前缀并更改后缀)
            base_name = os.path.splitext(filename)[0]
            new_pkl_filename = FILE_PREFIX + base_name + ".pkl"
            pkl_output_path = os.path.join(output_full_path, new_pkl_filename)
            
            # 3. 执行转换
            convert_npy_to_pkl(npy_file_path, pkl_output_path, DEFAULT_TEXT_DESCRIPTION)
            processed_count += 1
    
    print("=" * 60)
    print(f"✅ 批量转换完成。共处理 {processed_count} 个 .npy 文件。")
    print("=" * 60)