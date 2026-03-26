import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import shutil

class BaseMotionAdapter(ABC):
    """
    输入适配器基类：负责将不同来源的异构动作数据，清洗为标准的 [T, 22, 3] 矩阵。
    """
    def __init__(self, file_path: str, config: Dict[str, Any]):
        self.file_path = file_path
        self.config = config

    @abstractmethod
    def parse(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        解析文件。
        :return: (standard_motion, hint_trajectory)
                 standard_motion 必须是 shape 为 (T, 22, 3) 的 numpy 数组
                 hint_trajectory 可以是 (T, 3) 的 numpy 数组，如果没有则返回 None
        """
        pass

class BaseSceneDecorator(ABC):
    """
    环境上下文装饰器基类：负责场景坐标注入与障碍物信息打包。
    """
    def __init__(self, scene_config: Optional[Dict[str, Any]]):
        self.scene_config = scene_config

    @abstractmethod
    def apply_context(self, motion_data: np.ndarray, hint_data: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        应用场景上下文。
        :return: 打包好的标准 Payload 字典，直接用于后续的 pkl 序列化。
        """
        pass

class MCMAdapter(BaseMotionAdapter):
    """
    MCM-LDM 适配器：原生输出已经是 (T, 22, 3)，直接读取返回。
    """
    def parse(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        print(f"[Adapter] 正在使用 MCMAdapter 解析: {self.file_path}")
        
        # 1. 读取原生数据
        data = np.load(self.file_path, allow_pickle=True)
        
        # 2. 如果包含在 object/dict 里，剥离出来 (向下兼容你之前的报错保护)
        if data.dtype == 'object':
            data_dict = data.item()
            motion = data_dict['motion'] if isinstance(data_dict, dict) and 'motion' in data_dict else np.array(data_dict)
        else:
            motion = data
            
        # 3. Torch tensor 转换保护
        if hasattr(motion, 'detach'):
            motion = motion.detach().cpu().numpy()

        # 4. MCM-LDM 没有 hint 轨迹文件
        hint_trajectory = None

        # 确保维度是 (T, 22, 3)
        assert motion.ndim == 3 and motion.shape[1] == 22 and motion.shape[2] == 3, \
            f"MCMAdapter 预期维度 (T, 22, 3), 实际拿到 {motion.shape}"

        return motion, hint_trajectory

class MCMWithTrajAdapter(BaseMotionAdapter):
    """
    带轨迹扩展的 MCM 适配器：除了读取动作，还会自动探测同名的 _givenTraj.npy 文件。
    """
    def parse(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        print(f"[Adapter] 正在使用 MCMWithTrajAdapter 解析: {self.file_path}")
        
        # --- 1. 解析主动作 (复用你之前的清理逻辑) ---
        data = np.load(self.file_path, allow_pickle=True)
        if data.dtype == 'object':
            data_dict = data.item()
            motion = data_dict['motion'] if isinstance(data_dict, dict) and 'motion' in data_dict else np.array(data_dict)
        else:
            motion = data
            
        if hasattr(motion, 'detach'):
            motion = motion.detach().cpu().numpy()
            
        # --- 2. 探查并解析附属轨迹文件 ---
        hint_trajectory = None
        traj_file_path = self.file_path.replace('.npy', '_givenTraj.npy')
        
        if os.path.exists(traj_file_path):
            print(f"[Adapter] 🔍 发现提示轨迹文件: {os.path.basename(traj_file_path)}")
            # 注意加上 allow_pickle=True 防御报错
            traj_data = np.load(traj_file_path, allow_pickle=True) 
            
            # 【防御性编程】：抹平异构轨迹数据的维度差异
            if traj_data.ndim == 2 and traj_data.shape[1] == 3:
                # 已经是标准的 (T, 3) 纯轨迹矩阵
                hint_trajectory = traj_data 
            elif traj_data.ndim == 3 and traj_data.shape[1] == 22 and traj_data.shape[2] == 3:
                # 如果存成了一个完整的骨骼矩阵，只提取根节点 (Root) 的坐标作为轨迹
                hint_trajectory = traj_data[:, 0, :] 
            else:
                print(f"[Warning] 轨迹维度 {traj_data.shape} 异常，已安全忽略。")
        else:
            print(f"[Adapter] ⚠️ 未找到配套的 _givenTraj.npy，退化为无轨迹模式。")

        return motion, hint_trajectory

class NullSceneDecorator(BaseSceneDecorator):
    """
    空场景装饰器：用于基础 Baseline 渲染，不做任何平移，不注入任何障碍物。
    """
    def apply_context(self, motion_data: np.ndarray, hint_data: Optional[np.ndarray]) -> Dict[str, Any]:
        print("[Decorator] 触发 NullSceneDecorator: 保持原生坐标系，跳过场景注入。")
        
        # 提取 3D 真实轨迹 (Root)
        root_trajectory_3d = motion_data[:, 0, :]
        
        # 提取 2D 地面投影
        ground_trajectory = np.zeros_like(root_trajectory_3d)
        ground_trajectory[:, 0] = root_trajectory_3d[:, 0]
        ground_trajectory[:, 2] = root_trajectory_3d[:, 2]

        # 构造标准的 PKL Payload
        payload = {
            'joints': motion_data,
            'length': motion_data.shape[0],
            'root_trajectory_3d': root_trajectory_3d,
            'ground_trajectory': ground_trajectory,
            'hint': hint_data,
            # 留空障碍物和场景信息，供 Blender 备用
            'obstacles': [], 
            'scene_name': "default"
        }
        return payload


import os
import pickle
import argparse
from tqdm import tqdm

# 假设这里有一个简单的注册表字典，映射到我们之前写的基类实现
REGISTRY_BASELINES = {
    "MCM-LDM": MCMAdapter,
    "MCM-LDM-Traj": MCMWithTrajAdapter
    # "OmniControl": OmniControlAdapter,
    # "GMD": GMDAdapter
}

REGISTRY_SCENES = {
    "Default": NullSceneDecorator,
    # "Dumuqiao": StandardSceneDecorator
}

def filter_bad_frames(valid_frames_mask):
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

def compile_folder(input_folder, output_folder, model_type, scene_name, title):
    """
    动作编译总线：负责调度 Adapter 洗数据，调度 Decorator 挂载场景。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 1. 从注册表动态获取对应的策略类
    AdapterClass = REGISTRY_BASELINES.get(model_type, MCMAdapter)
    DecoratorClass = REGISTRY_SCENES.get(scene_name, NullSceneDecorator)

    # 2. 实例化场景装饰器 (整个文件夹共用同一个场景规则)
    # 如果是 Dumuqiao，这里可以传入读取的 JSON dict；Default 则传 None
    decorator = DecoratorClass(scene_config=None) 

    npy_files = [f for f in os.listdir(input_folder) if f.endswith('.npy') and not f.endswith('_givenTraj.npy')]
    waypoint_and_obstacle_file = [f for f in os.listdir(input_folder) if f.endswith('path_and_obstable.json')]
    waypoint_and_obstacle_file_path = os.path.join(input_folder, waypoint_and_obstacle_file[0]) if waypoint_and_obstacle_file else None
    if waypoint_and_obstacle_file_path is not None:
        shutil.copy2(waypoint_and_obstacle_file_path, os.path.join(output_folder, 'path_and_obstable.json'))

    success_count = 0
    for filename in tqdm(npy_files, desc=f"编译 {model_type} 数据"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace('.npy', '.pkl'))

        try:
            # 3. 实例化输入适配器并解析异构数据
            adapter = AdapterClass(input_path, config={})
            raw_motion, raw_hint = adapter.parse()

            # # 4. 健壮性过滤：异常帧清洗 (复用你原来写的防飞天遁地逻辑)
            # clean_motion, clean_hint = filter_bad_frames(raw_motion, raw_hint, threshold=50.0)
            # if clean_motion is None:
            #     continue
            clean_motion, clean_hint = raw_motion, raw_hint # TODO: 目前先不做过滤，后续根据需要再加

            # 5. 注入场景上下文，生成标准 Payload
            payload = decorator.apply_context(clean_motion, clean_hint)
            payload['text'] = title # 补充文本信息

            # 6. 统一打包落盘
            with open(output_path, 'wb') as f:
                pickle.dump(payload, f)
            
            success_count += 1

        except Exception as e:
            print(f"\n[Error] 编译文件 {filename} 失败: {e}")

    print(f"\n✅ 编译完成: 成功 {success_count}/{len(npy_files)}，输出至 {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motion Compiler: 异构动作数据统一编译管线")
    
    # 核心路径参数
    parser.add_argument("--input_folder", type=str, required=True, help="源 .npy 文件夹路径")
    
    # 架构路由参数 (新增)
    parser.add_argument("--model_type", type=str, default="MCM-LDM", 
                        choices=list(REGISTRY_BASELINES.keys()), 
                        help="数据来源模型 (决定使用的 Adapter)")
    parser.add_argument("--scene_name", type=str, default="Default", 
                        choices=list(REGISTRY_SCENES.keys()), 
                        help="场景名称 (决定使用的 Decorator)")
    
    # 附属信息
    parser.add_argument("--title", type=str, default="SceneMoDiff result", help="动作文本描述")

    args = parser.parse_args()
    output_folder = args.input_folder + "_pkl"

    print(f"🚀 启动动作编译管线 | 模型引擎: {args.model_type} | 场景装饰: {args.scene_name}")
    
    # 启动总线
    compile_folder(
        input_folder=args.input_folder,
        output_folder=output_folder,
        model_type=args.model_type,
        scene_name=args.scene_name,
        title=args.title
    )