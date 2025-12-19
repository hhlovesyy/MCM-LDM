import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import make_interp_spline
import math
import torch
from path_planner import PathPlanner

# ==========================================
# 1. 模拟工具函数 (模拟 Content Motion 的数据)
# ==========================================
def mock_content_motion_features(frames=196, avg_speed=0.05):
    """
    模拟一个 Content Motion 的 263 维特征的前 4 维 (Root Data)。
    用于提供“速度分布”和“总路程”。
    """
    # 模拟忽快忽慢的速度 (正态分布)
    # [Frames, 4] -> RotVel, VelX, VelZ, Height
    feats = np.zeros((frames, 4))
    
    # VelZ (前进速度)
    speeds = np.abs(np.random.normal(loc=avg_speed, scale=0.02, size=frames))
    feats[:, 2] = speeds
    
    # Height (假设 0.95)
    feats[:, 3] = 0.95
    
    return feats

def calculate_cumulative_distance(motion_features):
    """
    从 motion features 计算每帧的位移和累积路程。
    """
    # VelX, VelZ
    vels = motion_features[:, 1:3]  # torch.Size([161, 2])
    # 每帧位移 = sqrt(vx^2 + vz^2)
    step_sizes = np.linalg.norm(vels, axis=1) # shape:(161,)
    
    # 累积路程: [0, d1, d1+d2, ...]
    # 长度 = Frames + 1
    cum_dist = np.concatenate(([0], np.cumsum(step_sizes))) # shape:(162,)
    return cum_dist

# ==========================================
# 2. 核心算法类
# ==========================================

# ==========================================
# 2. 核心处理类 (修改为使用 PathPlanner)
# ==========================================

class TrajectoryProcessor:
    def __init__(self):
        # 初始化规划器
        self.planner = PathPlanner(world_range=20.0, grid_res=0.1, margin=0.3)

    def generate_collision_free_path(self, control_points, obstacles):
        """
        使用 A* + Spline 生成避障路径
        """
        # 转换 obstacles 格式适配 PathPlanner
        planner_obs = []
        for obs in obstacles:
            item = {'type': obs['type'], 'center': obs['center']}
            if obs['type'] == 'cylinder':
                item['radius'] = obs['radius']
            elif obs['type'] == 'box':
                # 注意：你的 PathPlanner 需要 'extent' (w, d)
                # 而 json 里可能有 'size' 或 'extent'，这里做个适配
                if 'extent' in obs:
                    item['extent'] = obs['extent']
                elif 'size' in obs:
                    item['extent'] = [obs['size'][0], obs['size'][2]] # W, D (假设size是 W,H,D)
            planner_obs.append(item)

        # 调用规划器
        # raw_path 是经过 A* 和 Spline 平滑后的稠密点集
        smooth_path = self.planner.generate_path(control_points, planner_obs)
        
        return smooth_path

    def resample_by_arc_length(self, dense_curve, target_distances):
        # ... (保持不变) ...
        # 计算曲线的累积弧长
        diffs = np.linalg.norm(dense_curve[1:] - dense_curve[:-1], axis=1)
        curve_cum_dist = np.concatenate(([0], np.cumsum(diffs)))
        total_curve_len = curve_cum_dist[-1]
        
        resampled_points = []
        valid_mask = []
        
        for dist in target_distances:
            if dist > total_curve_len:
                resampled_points.append(dense_curve[-1])
                valid_mask.append(False)
            else:
                rx = np.interp(dist, curve_cum_dist, dense_curve[:, 0])
                rz = np.interp(dist, curve_cum_dist, dense_curve[:, 1])
                resampled_points.append([rx, rz])
                valid_mask.append(True)
                
        return np.array(resampled_points), np.array(valid_mask)

    def compute_trajectory_features(self, points):
        # ... (保持不变) ...
        frames = len(points)
        feats = np.zeros((frames, 4))
        tangents = np.diff(points, axis=0) 
        tangents = np.concatenate([tangents, tangents[-1:]], axis=0)
        headings = np.arctan2(tangents[:, 0], tangents[:, 1])
        headings = np.unwrap(headings)
        
        rot_vel = np.diff(headings)
        rot_vel = np.concatenate([rot_vel, [0]])
        feats[:, 0] = rot_vel
        
        step_sizes = np.linalg.norm(tangents, axis=1)
        feats[:, 1] = 0.0        
        feats[:, 2] = step_sizes 
        feats[:, 3] = 0.95
        return feats, headings

# ==========================================
# 3. 验证与可视化
# ==========================================

def visualize_verification(json_data, dense_curve, resampled_points, valid_mask, headings):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 1. 画障碍物
    obstacles = json_data['environment']['obstacles']
    for obs in obstacles:
        cx, cz = obs['center']
        if obs['type'] == 'cylinder':
            c = patches.Circle((cx, cz), obs['radius'], color='red', alpha=0.3, label='Obs Cylinder')
            ax.add_patch(c)
            # 画安全边界
            margin = 0.3 # 假设 margin
            c_safe = patches.Circle((cx, cz), obs['radius'] + margin, color='red', fill=False, linestyle='--', alpha=0.3)
            ax.add_patch(c_safe)
            
        elif obs['type'] == 'box':
            w, d = obs.get('extent', [1,1]) # 适配
            if 'size' in obs: w, d = obs['size'][0], obs['size'][2]
            
            rect = patches.Rectangle((cx - w/2, cz - d/2), w, d, color='blue', alpha=0.3, label='Obs Box')
            ax.add_patch(rect)

    # 2. 画原始控制点 (用户期望路径)
    ctrl_pts = np.array(json_data['trajectory']['points'])
    ax.plot(ctrl_pts[:, 0], ctrl_pts[:, 1], 'rx', markersize=10, label='User Waypoints')
    
    # 3. 画规划后的路径 (A* + Spline)
    ax.plot(dense_curve[:, 0], dense_curve[:, 1], 'k-', alpha=0.5, linewidth=2, label='Planned Path')
    
    # 4. 画重采样轨迹
    valid_pts = resampled_points[valid_mask]
    if len(valid_pts) > 0:
        ax.scatter(valid_pts[:, 0], valid_pts[:, 1], c='blue', s=15, label='Resampled (Motion Frame)')
        # 画稀疏箭头
        for i in range(0, len(valid_pts), 10):
            x, z = valid_pts[i]
            a = headings[i]
            ax.arrow(x, z, 0.3*np.sin(a), 0.3*np.cos(a), head_width=0.1, color='green')

    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True)
    ax.set_title(f"A* Path Planning & Resampling Verification")
    plt.savefig("scene_verification_astar.png")
    print("Verification plot saved to scene_verification_astar.png")

# ==========================================
# 4. 主流程
# ==========================================

def main():
    # 1. 解析 JSON (模拟读取)
    json_path = "/root/autodl-tmp/MyRepository/MCM-LDM/task_config.json"

    # 使用 with 语句打开文件（这样会自动关闭文件，更安全）
    with open(json_path, 'r', encoding='utf-8') as f:
        scene_data = json.load(f)
    
    # 2. 模拟 Content Motion (196帧，平均速度 0.05m/frame)
    # 实际项目中，这里你会读取 .npy 文件
    print("[1] Loading/Mocking Content Motion...")

    npy_path = "/root/autodl-tmp/MyRepository/MCM-LDM/demo/content_motion/walk_and_turn.npy"
    npy_data = np.load(npy_path)  # 形状为 (161, 263)
    # 2. 转换为 Tensor (推荐：共享内存，速度最快)
    npy_tensor = torch.from_numpy(npy_data)
    # npy_tensor = npy_tensor.unsqueeze(0)
    # 3. 如果需要转换数据类型（例如转为 Float32，这是深度学习最常用的格式）
    npy_tensor = npy_tensor[:, :4].float()
    frames = npy_tensor.shape[0]

    # 4. 如果需要移动到 GPU (AutoDL 环境常用)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # content_feats = npy_tensor.to(device) # torch.Size([161, 4])
    content_feats = npy_tensor
    
    # 计算累积距离 Profile
    dist_profile = calculate_cumulative_distance(content_feats)
    total_len = dist_profile[-1] 
    print(f"   Total Content Length: {total_len:.2f} meters") # 4.98m
    
    # 3. 处理
    processor = TrajectoryProcessor()
    
    # A. 生成避障路径 (核心改动)
    # 输入: Waypoints, Obstacles
    # 输出: 稠密的、避障的、平滑的路径
    print("[1] Planning Path...")
    control_points = scene_data['trajectory']['points']
    obstacles = scene_data['environment']['obstacles']
    dense_curve = processor.generate_collision_free_path(control_points, obstacles)
    
    # B. 重采样
    print("[2] Resampling...")
    target_dists = dist_profile[1:]
    resampled_points, valid_mask = processor.resample_by_arc_length(dense_curve, target_dists)
    
    # C. 特征
    trans_cond, headings = processor.compute_trajectory_features(resampled_points)
    
    # 4. 可视化
    visualize_verification(scene_data, dense_curve, resampled_points, valid_mask, headings)
    
if __name__ == "__main__":
    main()