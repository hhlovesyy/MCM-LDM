import numpy as np
import pickle
import os
import sys
from tqdm import tqdm

def find_longest_continuous_good_segment(valid_frames_mask):
    """
    在一个布尔掩码中，找到最长的连续 True 片段。
    
    Args:
        valid_frames_mask (np.ndarray): 一个布尔数组，True 代表好帧。

    Returns:
        tuple: (start_index, end_index)
    """
    longest_len = 0
    best_start = 0
    best_end = 0
    current_len = 0
    current_start = 0

    for i, is_valid in enumerate(valid_frames_mask):
        if is_valid:
            if current_len == 0:
                current_start = i
            current_len += 1
        else:
            if current_len > longest_len:
                longest_len = current_len
                best_start = current_start
                best_end = i
            current_len = 0
    
    # 循环结束后，检查最后一个连续片段
    if current_len > longest_len:
        longest_len = current_len
        best_start = current_start
        best_end = len(valid_frames_mask)
        
    # 如果所有帧都是坏的，返回 (0, 0)
    if longest_len == 0 and len(valid_frames_mask) > 0:
        return 0, 0
        
    return best_start, best_end

def load_motion_compat(file_path):
    # 1. 加载数据
    data = np.load(file_path, allow_pickle=True)
    hint = None
    
    # 2. 字典提取逻辑
    if data.dtype == 'object':
        try:
            data_dict = data.item()
            if isinstance(data_dict, dict) and 'motion' in data_dict:
                motion = data_dict['motion']
            else:
                motion = np.array(data_dict)

            if isinstance(data_dict, dict) and 'hint' in data_dict:
                hint = data_dict['hint']

        except:
            motion = data
    else:
        motion = data

    # 3. 维度转换逻辑 (1, 22, 3, 196) -> (196, 22, 3)
    if hasattr(motion, 'detach'): # 如果是 torch tensor 先转 numpy
        motion = motion.detach().cpu().numpy()
    if hint is not None and hasattr(hint, 'detach'):
        hint = hint.detach().cpu().numpy()

    # 检查维度并转换, 这个就是针对OmniControl的，要把原点给他加上，我们的模型生成的结果是没有这个的
    if motion.ndim == 4 and motion.shape[0] == 1:
        # 步骤 1: 去掉 batch 维度 -> (22, 3, 196)
        motion = np.squeeze(motion, axis=0) 
        # 步骤 2: 将最后一维（帧数）提到最前面 -> (196, 22, 3)
        # 原维度索引: 22(0), 3(1), 196(2)
        # 目标索引顺序: 196(2), 22(0), 3(1)
        motion = motion.transpose(2, 0, 1)
        import copy
        import json
        json_path = "/root/autodl-tmp/MyRepository/MCM-LDM/task_config_baseline.json"

        # 使用 with 语句打开文件（这样会自动关闭文件，更安全）
        with open(json_path, 'r', encoding='utf-8') as f:
            scene_data = json.load(f)
        # 这部分直接复制你项目里的逻辑，确保坐标系对齐
        traj_config = copy.deepcopy(scene_data['trajectory'])
        obstacles = copy.deepcopy(scene_data['environment']['obstacles'])
        
        # 坐标归一化 (起点归零)
        waypoints = traj_config['points']
        startPosX, startPosY = waypoints[0][0], waypoints[0][1]
        motion[..., 0] += startPosX # X
        motion[..., 2] += startPosY

    if hint is not None and hint.ndim == 4 and hint.shape[0] == 1: # shape:(1, 196, 22, 3), 目前是针对OmniControl的
        hint = np.squeeze(hint, axis=0)
        
    return motion, hint

def convert_npy_to_pkl(npy_file_path, pkl_output_path, title="SceneMoDiff result", pos_threshold=50.0):
    """
    [Final Version] 将 (T, J, 3) 的 .npy 关节数据转换为 .pkl，
    并自动清洗掉坐标异常的帧，只保留最长的连续有效片段。
    """
    npy_trajectory_file_path = npy_file_path.replace('.npy', '_givenTraj.npy')
    try:
        joints_data, hint = load_motion_compat(npy_file_path)
        hint = None
        if os.path.exists(npy_trajectory_file_path) and hint is None:
            npy_trajectory_data = np.load(npy_trajectory_file_path, allow_pickle=True)
        else:
            npy_trajectory_data = hint
        
        if joints_data.ndim != 3 or joints_data.shape[2] != 3:
            print(f"\n警告: 文件 {os.path.basename(npy_file_path)} 格式不正确 (应为 T,J,3)，已跳过。")
            return False
        if npy_trajectory_data is not None:
            if npy_trajectory_data.ndim != 2 or npy_trajectory_data.shape[1] != 3:
                print(f"\n警告: 文件 {os.path.basename(npy_trajectory_file_path)} 格式不正确 (应为 T,3)，将被视为无轨迹处理。")
                print("shape: ", npy_trajectory_data.shape)
                if npy_trajectory_data.ndim == 3 and npy_trajectory_data.shape[1] == 22:
                    npy_trajectory_data = npy_trajectory_data[:, 0, :]
                    print("此时轨迹的维度是： ", npy_trajectory_data.shape)
                else:
                    npy_trajectory_data = None # 格式不对，直接当作没读取到

        # 1. 【核心】基于根关节的位置，识别所有“好帧”
        root_positions = joints_data[:, 0, :] # [T, 3]
        # 检查每一帧的根关节坐标的绝对值是否都小于阈值
        valid_frames_mask = np.all(np.abs(root_positions) < pos_threshold, axis=1)
        
        # 2. 如果存在任何坏帧，则执行清洗
        if not np.all(valid_frames_mask):
            start, end = find_longest_continuous_good_segment(valid_frames_mask)
            
            if (end - start) < 10: # 如果最长片段太短，放弃这个文件
                print(f"\n警告: 文件 {os.path.basename(npy_file_path)} 没有足够长的有效片段，已跳过。")
                return False

            print(f"\n信息: 在 {os.path.basename(npy_file_path)} 中发现异常帧。 "
                  f"正在提取最长有效片段 (帧 {start} 到 {end})。")
            
            joints_data = joints_data[start:end]
            if npy_trajectory_data is not None:
                npy_trajectory_data = npy_trajectory_data[start:end]

        # 3. 后续所有逻辑，都使用清洗后的 `joints_data`
        length = joints_data.shape[0]
        
        if length == 0:
             print(f"\n警告: 文件 {os.path.basename(npy_file_path)} 清洗后长度为0，已跳过。")
             return False

        # 提取 3D 根关节轨迹
        root_trajectory_3d = joints_data[:, 0, :]
        # 提取 2D 地面投影轨迹
        ground_trajectory = np.zeros_like(root_trajectory_3d)
        ground_trajectory[:, 0] = root_trajectory_3d[:, 0] # X
        ground_trajectory[:, 2] = root_trajectory_3d[:, 2] # Z

        if npy_trajectory_data is not None:
            given_hint = np.zeros_like(npy_trajectory_data)
            given_hint[:, 0] = npy_trajectory_data[:, 0]
            given_hint[:, 2] = npy_trajectory_data[:, 2]
            # assert ((given_hint.ndim == ground_trajectory.ndim) and (given_hint.shape[0] == ground_trajectory.shape[0])), "提示轨迹与模型生成轨迹的维度对不上，请检查脚本或者数据流！"
        else:
            print(f"\n警告: 没有轨迹的相关文件！轨迹为None")
            given_hint = None

        pkl_data = {
            'joints': joints_data,
            'text': title,
            'length': length,
            'root_trajectory_3d': root_trajectory_3d,
            'ground_trajectory': ground_trajectory,
            'hint': given_hint
        }

        with open(pkl_output_path, 'wb') as f:
            pickle.dump(pkl_data, f)
        
        return True

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
    npy_files = [f for f in os.listdir(input_folder) if (f.endswith('.npy') and not f.endswith('givenTraj.npy'))]
    
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
    
    parser = argparse.ArgumentParser(description="将一个文件夹中的所有 .npy 动作文件批量转换为 .pkl 文件。")
    parser.add_argument("--input_folder", type=str, help="包含源 .npy 文件的输入文件夹路径。")
    # parser.add_argument("--output_folder", type=str, help="用于保存生成的 .pkl 文件的目标文件夹路径。")
    # output_folder自动建立在input_folder的下面

    output_folder = parser.parse_args().input_folder + "_pkl"
    parser.add_argument("--title", type=str, default="SceneMoDiff result", help="为所有动作指定的标题/文本描述（可选）。")
    
    args = parser.parse_args()
    process_folder(args.input_folder, output_folder, args.title)
    
    # # 调用主处理函数
    # process_folder("/root/autodl-tmp/MyRepository/MCM-LDM/results/mld/scenemoDiff_1229_Baseline_no_scene/OmniControl",
    #                 "/root/autodl-tmp/MyRepository/MCM-LDM/results/mld/scenemoDiff_1229_Baseline_no_scene/OmniControl_pkl",
    #                 "FinetuneBaseline_Balance")