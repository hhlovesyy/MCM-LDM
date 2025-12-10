import numpy as np
import pickle
import os
import sys
from tqdm import tqdm

def convert_npy_to_pkl(npy_file_path, pkl_output_path, title="SceneMoDiff result"):
    """
    将一个 .npy 文件（关节数据）转换为包含更多信息的 .pkl 文件格式。

    参数:
    npy_file_path (str): 输入的 .npy 文件路径。
    pkl_output_path (str): 输出的 .pkl 文件路径。
    title (str): 描述动作的标题或文本。
    """
    try:
        # 1. 加载 .npy 文件
        joints_data = np.load(npy_file_path)

        # 确保数据形状是三维的 (frames, joints, coordinates)
        if joints_data.ndim != 3:
            print(f"\n警告: 文件 {os.path.basename(npy_file_path)} 的数组维度不是3维，已跳过。")
            return False

        # 2. 从形状中计算 length
        length = joints_data.shape[0]

        # 3. 【新增】提取轨迹信息
        # 假设根关节 (root joint) 是第一个关节 (index 0)
        # 轨迹通常是 X 和 Z 坐标在地面上的投影。我们提取所有3个坐标以备后用。
        # trajectory_data 的形状将是 (frames, 3)
        trajectory_data = joints_data[:, 0, :]

        # 4. 构造最终的字典数据
        pkl_data = {
            'joints': joints_data,      # 完整的关节数据 (T, J, 3)
            'text': title,              # 标题/文本描述
            'length': length,           # 动作长度
            'trajectory': trajectory_data, # 提取出的轨迹数据 (T, 3)
            'hint': None                # 保留字段
        }

        # 5. 将字典保存为 .pkl 文件
        with open(pkl_output_path, 'wb') as f:
            pickle.dump(pkl_data, f)
        
        return True # 返回成功状态

    except Exception as e:
        print(f"\n错误: 处理文件 {os.path.basename(npy_file_path)} 时发生错误: {e}")
        return False

def process_folder(input_folder, output_folder, title="SceneMoDiff result"):
    """
    遍历输入文件夹中的所有 .npy 文件，并将它们转换为 .pkl 文件保存在输出文件夹。
    """
    if not os.path.isdir(input_folder):
        print(f"错误: 输入文件夹不存在: {input_folder}")
        return

    if not os.path.exists(output_folder):
        print(f"目标文件夹不存在，正在创建: {output_folder}")
        os.makedirs(output_folder)

    # 找到所有 .npy 文件
    npy_files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]
    
    if not npy_files:
        print(f"警告: 在输入文件夹 {input_folder} 中没有找到任何 .npy 文件。")
        return

    print(f"找到 {len(npy_files)} 个 .npy 文件。开始转换...")

    success_count = 0
    # 使用 tqdm 创建一个进度条
    for filename in tqdm(npy_files, desc="转换进度"):
        input_path = os.path.join(input_folder, filename)
        
        # 构建输出文件名，将 .npy 替换为 .pkl
        output_filename = os.path.splitext(filename)[0] + '.pkl'
        output_path = os.path.join(output_folder, output_filename)
        
        if convert_npy_to_pkl(input_path, output_path, title):
            success_count += 1
            
    print("\n" + "="*50)
    print("转换完成！")
    print(f"总计文件: {len(npy_files)}")
    print(f"成功转换: {success_count}")
    print(f"失败/跳过: {len(npy_files) - success_count}")
    print(f"所有 .pkl 文件已保存至: {output_folder}")
    print("="*50)


if __name__ == "__main__":
    # 使用 argparse 来处理命令行参数，更健壮、更友好
    import argparse
    
    # parser = argparse.ArgumentParser(description="将一个文件夹中的所有 .npy 动作文件批量转换为 .pkl 文件。")
    # parser.add_argument("input_folder", type=str, help="包含源 .npy 文件的输入文件夹路径。")
    # parser.add_argument("output_folder", type=str, help="用于保存生成的 .pkl 文件的目标文件夹路径。")
    # parser.add_argument("--title", type=str, default="SceneMoDiff result", help="为所有动作指定的标题/文本描述（可选）。")
    
    # args = parser.parse_args()
    
    # # 调用主处理函数
    # process_folder(args.input_folder, args.output_folder, args.title)
    process_folder("/root/autodl-tmp/MyRepository/MotionLCM/MotionLCM/SceneMoDiff_input_npy/MLP_Loss/Windy",
                   "/root/autodl-tmp/MyRepository/MotionLCM/MotionLCM/SceneMoDiff_input_npy/MLP_Loss/Windy_pkl",
                   "FiLM Windy")