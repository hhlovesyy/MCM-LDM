import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import shutil
import json
import copy
from mld.models.modeltype.trajectory_utils import * 

class BaseMotionAdapter(ABC):
    """
    输入适配器基类：负责将不同来源的异构动作数据，清洗为标准的 [T, 22, 3] 矩阵。
    """
    def __init__(self, file_path: str, config: Dict[str, Any], decorator: 'BaseSceneDecorator'):
        self.file_path = file_path
        self.config = config
        self.decorator = decorator # TODO:有空再重构吧，把场景相关的decorator塞入到Adapter里不是一个好的软件工程设计，违反了单一职责的原则，但是时间有限先做完需求，应该把逻辑写到decorator里面去

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

# class MCMWithTrajAdapter(BaseMotionAdapter):
#     """
#     带轨迹扩展的 MCM 适配器：除了读取动作，还会自动探测同名的 _givenTraj.npy 文件。
#     """
#     def parse(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
#         print(f"[Adapter] 正在使用 MCMWithTrajAdapter 解析: {self.file_path}")
        
#         # --- 1. 解析主动作 (复用你之前的清理逻辑) ---
#         data = np.load(self.file_path, allow_pickle=True)
#         if data.dtype == 'object':
#             data_dict = data.item()
#             motion = data_dict['motion'] if isinstance(data_dict, dict) and 'motion' in data_dict else np.array(data_dict)
#         else:
#             motion = data
            
#         if hasattr(motion, 'detach'):
#             motion = motion.detach().cpu().numpy()
            
#         # --- 2. 探查并解析附属轨迹文件 ---
#         hint_trajectory = None
#         traj_file_path = self.file_path.replace('.npy', '_givenTraj.npy')
        
#         if os.path.exists(traj_file_path):
#             print(f"[Adapter] 🔍 发现提示轨迹文件: {os.path.basename(traj_file_path)}")
#             # 注意加上 allow_pickle=True 防御报错
#             traj_data = np.load(traj_file_path, allow_pickle=True) 
            
#             # 【防御性编程】：抹平异构轨迹数据的维度差异
#             if traj_data.ndim == 2 and traj_data.shape[1] == 3:
#                 # 已经是标准的 (T, 3) 纯轨迹矩阵
#                 hint_trajectory = traj_data 
#             elif traj_data.ndim == 3 and traj_data.shape[1] == 22 and traj_data.shape[2] == 3:
#                 # 如果存成了一个完整的骨骼矩阵，只提取根节点 (Root) 的坐标作为轨迹
#                 hint_trajectory = traj_data[:, 0, :] 
#             else:
#                 print(f"[Warning] 轨迹维度 {traj_data.shape} 异常，已安全忽略。")
#         else:
#             print(f"[Adapter] ⚠️ 未找到配套的 _givenTraj.npy，退化为无轨迹模式。")

#         return motion, hint_trajectory
    
class MCMWithTrajAdapter(BaseMotionAdapter):
    """
    带轨迹扩展的 MCM 适配器：读取动作和 _givenTraj.npy。
    【核心修复】：将绝对坐标强行归零，对齐 Pipeline 的相对坐标标准。
    """
    def parse(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        print(f"[Adapter] 正在使用 MCMWithTrajAdapter 解析: {self.file_path}")
        
        # --- 1. 解析主动作 ---
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
            traj_data = np.load(traj_file_path, allow_pickle=True) 
            
            if traj_data.ndim == 2 and traj_data.shape[1] == 3:
                hint_trajectory = traj_data 
            elif traj_data.ndim == 3 and traj_data.shape[1] == 22 and traj_data.shape[2] == 3:
                hint_trajectory = traj_data[:, 0, :] 
            else:
                print(f"[Warning] 轨迹维度 {traj_data.shape} 异常，已安全忽略。")
        else:
            print(f"[Adapter] ⚠️ 未找到配套的 _givenTraj.npy，退化为无轨迹模式。")

        # ==========================================================
        # 🚀 核心修复：将绝对坐标强制转换为相对坐标 (归零化)
        # ==========================================================
        if motion is not None and len(motion) > 0:
            # 取第一帧根节点的 X 和 Z 作为绝对偏移量
            start_x = motion[0, 0, 0]
            start_z = motion[0, 0, 2]
            
            print(f"[Adapter] MCM 绝对起点探测: X={start_x:.4f}, Z={start_z:.4f}。正在执行归零化...")
            
            # 将生成的动作减去起点，退化为原点起步
            motion[..., 0] -= start_x
            motion[..., 2] -= start_z
            
            # 将指引轨迹也减去相同的起点，保持两者的相对位置关系绝对一致
            if hint_trajectory is not None and len(hint_trajectory) > 0:
                hint_trajectory[..., 0] -= start_x
                hint_trajectory[..., 2] -= start_z
                
            print("[Adapter] ✅ MCM 数据归零化完成，已对齐 Pipeline 标准。")
        # ==========================================================

        return motion, hint_trajectory

class OmniControlAdapter(BaseMotionAdapter):
    """
    OmniControl 适配器：
    专门处理包含 ['motion', 'lengths', 'hint'] 等键值的复杂字典。
    将 (1, 22, 3, 196) 降维转置并截断为标准的 (T, 22, 3)。
    """
    def parse(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        print(f"[Adapter] 正在使用 OmniControlAdapter 解析: {self.file_path}")
        
        # 1. 读取全家桶字典
        data_dict = np.load(self.file_path, allow_pickle=True).item()
        
        # 2. 获取真实的序列长度 (用于截断后面乱走的 padding 帧)
        # 假设批量大小为 1，直接取第 0 个元素的长度
        real_length = int(data_dict['lengths'][0])
        print(f"[Adapter] 探测到真实动作长度: {real_length} 帧")

        # 3. 清洗 Motion 数据
        # 原始 shape: (1, 22, 3, 196)
        raw_motion = data_dict['motion']
        # 步骤 A: 挤掉 Batch 维度 -> (22, 3, 196)
        motion = np.squeeze(raw_motion, axis=0) 
        # 步骤 B: 维度转置 (J, C, T) -> (T, J, C) 即 (196, 22, 3)
        motion = motion.transpose(2, 0, 1)
        # 步骤 C: 严格根据真实长度截断 -> (real_length, 22, 3)
        motion = motion[:real_length]
        
        # 4. 清洗 Hint 轨迹数据
        hint_trajectory = None
        if 'hint' in data_dict:
            raw_hint = data_dict['hint']
            # 原始 shape: (1, 196, 22, 3)
            # 挤掉 Batch 维度 -> (196, 22, 3)
            hint_full = np.squeeze(raw_hint, axis=0)
            
            # 提取根节点 (第 0 个关节)，并严格截断
            # 最终 shape: (real_length, 3)
            hint_trajectory = hint_full[:real_length, 0, :]
            print(f"[Adapter] 成功提取并截断 Hint 轨迹，Shape: {hint_trajectory.shape}")
        
        return motion, hint_trajectory

class GMDAdapter(BaseMotionAdapter):
    """
    GMD 适配器：
    解析 GMD 生成的 (B, 22, 3, T) 动作，同时读取原始 JSON 恢复 Hint 轨迹，
    保证在渲染时能完美展示对比效果。
    """
    def parse(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        print(f"[Adapter] 正在使用 GMDAdapter 解析: {self.file_path}")
        
        # --- 1. 读取 GMD 生成的动作 ---
        data_dict = np.load(self.file_path, allow_pickle=True).item()
        raw_motion = data_dict['motion']
        motion = raw_motion[0].transpose(2, 0, 1) # (T, 22, 3)
        real_length = int(data_dict['lengths'][0])
        motion = motion[:real_length]
        # motion[:, :, 0] = motion[:, :, 0] * -1.0
        
        # --- 2. 恢复 Hint 指引轨迹 (从 JSON 中提取) ---
        hint_trajectory = None
        
        # 你的 JSON 固定路径 (或者你可以通过 self.config 传进来)
        json_path = self.decorator.scene_config
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                scene_data = json.load(f)
                
            traj_config = copy.deepcopy(scene_data['trajectory'])
            obstacles = copy.deepcopy(scene_data.get('environment', {}).get('obstacles', []))
            waypoints = traj_config['points']
            
            # 【关键】：必须归零！因为 GMD 生成的 motion 是从 0 相对坐标起步的。
            # 我们必须给它一个起点为 0 的相对 hint。
            # 最后的绝对坐标平移，统统交给 StandardSceneDecorator 去做。
            startPosX, startPosY = waypoints[0][0], waypoints[0][1]
            for p in waypoints:
                p[0] -= startPosX
                p[1] -= startPosY
            for obs in obstacles:
                if 'center' in obs:
                    obs['center'][0] -= startPosX
                    obs['center'][1] -= startPosY
                    
            # 召唤处理器生成稠密曲线
            processor = TrajectoryProcessor()
            dense_curve = processor.get_path_from_config(traj_config, obstacles)
            
            # 按照生成动作的真实长度 (real_length) 进行重采样对齐
            indices = np.linspace(0, len(dense_curve) - 1, real_length).astype(int)
            resampled_curve = dense_curve[indices]
            
            # 组装成 (T, 3) 的标准 hint 张量，高度暂时给 0
            hint_trajectory = np.zeros((real_length, 3), dtype=np.float32)
            hint_trajectory[:, 0] = resampled_curve[:, 0] # X
            hint_trajectory[:, 2] = resampled_curve[:, 1] # Z
            
            print(f"[Adapter] 🎯 成功从 JSON 重构 Hint 轨迹，Shape: {hint_trajectory.shape}")
            
        except Exception as e:
            print(f"[Adapter Error] 读取或重构 JSON 轨迹失败: {e}")
            
        print(f"[Adapter] ✅ GMD 数据清洗完毕，Motion Shape: {motion.shape}")
        # ==========================================================
        # 🚀 终极杀手锏：自适应旋转对齐 (解决 HumanML3D 底层坐标系篡改)
        # ==========================================================
        if hint_trajectory is not None and real_length > 10:
            print("[Adapter] 正在校准 GMD 的坐标系旋转偏差...")
            
            # 1. 提取生成动作的前进向量 (取第 10 帧，避免第 1 帧的噪声)
            gen_root = motion[:, 0, :]  # 根节点轨迹 (T, 3)
            gen_vec = gen_root[10, [0, 2]] - gen_root[0, [0, 2]] # 取 X 和 Z
            
            # 2. 提取 Hint 轨迹的前进向量
            hint_vec = hint_trajectory[10, [0, 2]] - hint_trajectory[0, [0, 2]]
            
            # 3. 计算两者在 X-Z 平面上的夹角 theta
            # 使用 arctan2 计算绝对角度差异，极其稳定
            angle_gen = np.arctan2(gen_vec[1], gen_vec[0])
            angle_hint = np.arctan2(hint_vec[1], hint_vec[0])
            theta = angle_hint - angle_gen
            
            print(f"[Adapter] 探测到旋转偏差角: {np.degrees(theta):.2f} 度，正在执行逆向扭转...")
            
            # 4. 构造 2D 旋转矩阵 (绕 Y 轴旋转)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            R = np.array([
                [cos_t, -sin_t],
                [sin_t,  cos_t]
            ])
            
            # 5. 将旋转矩阵应用到所有关节的所有帧上！
            # motion 形状是 (T, 22, 3)
            # 提取 X 和 Z 坐标，展平为 (T*22, 2)
            xz_flat = motion[:, :, [0, 2]].reshape(-1, 2)
            
            # 矩阵乘法应用旋转
            xz_rotated = xz_flat @ R.T 
            
            # 重新赋值回 motion 矩阵
            motion[:, :, [0, 2]] = xz_rotated.reshape(real_length, 22, 2)
            
            print("[Adapter] ✅ GMD 坐标系逆向扭转完成！")
        # ==========================================================
        
        return motion, hint_trajectory

class MotionLCMAdapter(BaseMotionAdapter):
    """
    MotionLCM 适配器：
    解析干净的 .pkl 文件，直接提取 (T, 22, 3) 的动作和轨迹。
    """
    def parse(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        print(f"[Adapter] 正在使用 MotionLCMAdapter 解析: {self.file_path}")
        
        # 1. 读取 Pickle 文件
        with open(self.file_path, 'rb') as f:
            data_dict = pickle.load(f)
            
        # 2. 提取 Motion (已经是完美的 (T, 22, 3) 形状)
        motion = data_dict['joints']
        real_length = data_dict.get('length', motion.shape[0])
        
        # 保险起见，按 length 截断一下
        motion = motion[:real_length]
        
        # 3. 提取 Hint 轨迹
        hint_trajectory = None
        if 'hint' in data_dict:
            raw_hint = data_dict['hint']
            # 取出根节点 (第 0 个关节) 的轨迹，形状变为 (T, 3)
            hint_trajectory = raw_hint[:real_length, 0, :]
            print(f"[Adapter] 🎯 成功提取内置 Hint 轨迹，Shape: {hint_trajectory.shape}")
            
        print(f"[Adapter] ✅ MotionLCM 数据清洗完毕，最终 Shape: {motion.shape}")
        
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

class StandardSceneDecorator(BaseSceneDecorator):
    """
    标准场景装饰器：
    负责读取 scene.json，并将动作数据从相对原点 (0,0) 平移到世界坐标系中的起点。
    """
    def apply_context(self, motion_data: np.ndarray, hint_data: Optional[np.ndarray]) -> Dict[str, Any]:
        print(f"[Decorator] 触发 StandardSceneDecorator: 正在注入场景绝对坐标...")
        
        # 1. 拷贝数据，防止污染原始内存
        motion_world = motion_data.copy()
        hint_world = hint_data.copy() if hint_data is not None else None
        
        obstacles = []
        
        json_path = self.scene_config
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                scene_data = json.load(f)
            
            # 获取起点坐标 (StartPosX, StartPosY)
            waypoints = scene_data['trajectory']['points']
            startPosX, startPosY = waypoints[0][0], waypoints[0][1]
            print(f"[Decorator] 探测到场景起点: X={startPosX:.2f}, Z={startPosY:.2f}")
            
            # --- 核心平移逻辑 ---
            # 将火柴人的 X 和 Z 加上起点坐标偏移
            motion_world[..., 0] += startPosX
            motion_world[..., 2] += startPosY
            
            if hint_world is not None:
                hint_world[..., 0] += startPosX
                hint_world[..., 2] += startPosY
            
            # 读取障碍物 (留给 Blender 备用)
            obstacles = scene_data.get('environment', {}).get('obstacles', [])
                
        except Exception as e:
            print(f"[Warning] 场景配置文件读取或解析失败: {e}，将保持相对坐标。")

        # 3. 提取轨迹用于记录
        root_trajectory_3d = motion_world[:, 0, :]
        ground_trajectory = np.zeros_like(root_trajectory_3d)
        ground_trajectory[:, 0] = root_trajectory_3d[:, 0]
        ground_trajectory[:, 2] = root_trajectory_3d[:, 2]

        # 4. 组装最终 Payload
        payload = {
            'joints': motion_world,
            'length': motion_world.shape[0],
            'root_trajectory_3d': root_trajectory_3d,
            'ground_trajectory': ground_trajectory,
            'hint': hint_world,
            'obstacles': obstacles,
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
    "MCM-LDM-Traj": MCMWithTrajAdapter,
    "OmniControl": OmniControlAdapter,
    "GMD": GMDAdapter,
    "MotionLCM": MotionLCMAdapter
}

REGISTRY_SCENES = {
    "Default": NullSceneDecorator,
    "DefaultWithSceneCorrectOffset": StandardSceneDecorator  # 如果有场景相关的json的话，这里记得选这个，防止轨迹和障碍物因为offset的原因贴合不上。
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

    npy_files = [f for f in os.listdir(input_folder) if f.endswith('.npy') and not f.endswith('_givenTraj.npy')]
    waypoint_and_obstacle_file = [f for f in os.listdir(input_folder) if f.endswith('path_and_obstable.json')]
    waypoint_and_obstacle_file_path = os.path.join(input_folder, waypoint_and_obstacle_file[0]) if waypoint_and_obstacle_file else None
    if waypoint_and_obstacle_file_path is not None:
        shutil.copy2(waypoint_and_obstacle_file_path, os.path.join(output_folder, 'path_and_obstable.json'))

    # 2. 实例化场景装饰器 (整个文件夹共用同一个场景规则)
    # 如果有场景相关的JSON文件
    if waypoint_and_obstacle_file_path is None:
        decorator = DecoratorClass(scene_config=None)
    else:
        print(f"📂 发现场景配置文件: {os.path.basename(waypoint_and_obstacle_file_path)}，正在启用StandardSceneDecorator...")
        DecoratorClass = StandardSceneDecorator
        # 加入一个字段：
        decorator = DecoratorClass(scene_config=waypoint_and_obstacle_file_path)

    success_count = 0
    # 1. 兼容多种输入格式的后缀
    valid_extensions = ('.npy', '.pkl')
    # 获取文件列表，过滤掉隐藏文件或其他无关文件
    data_files = [f for f in os.listdir(input_folder) if f.endswith(valid_extensions) and not f.startswith('.')]
    
    for filename in tqdm(data_files, desc=f"编译 {model_type} 数据"):
        input_path = os.path.join(input_folder, filename)
        # 2. 安全的文件名后缀替换 (无视原本是 .npy 还是 .pkl，统一变成统一样式的 .pkl)
        base_name, ext = os.path.splitext(filename)
        output_filename = f"{base_name}.pkl"
        output_path = os.path.join(output_folder, output_filename)

        try:
            # 3. 实例化输入适配器并解析异构数据
            adapter = AdapterClass(input_path, config={}, decorator=decorator)
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