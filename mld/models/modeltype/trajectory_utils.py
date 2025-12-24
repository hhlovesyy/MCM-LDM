import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import make_interp_spline
import math
import torch
from path_planner import PathPlanner

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
    
    def calculate_cumulative_distance(self, motion_features):
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
    
    def process_freehand_path(self, raw_points):
        """
        处理手绘路径：
        简单地将稀疏或不均匀的手绘点，插值成密集的点序列，模拟 dense_curve
        """
        path = np.array(raw_points)
        
        # 如果点太少，没法处理，直接返回
        if len(path) < 2:
            return path
            
        # 1. 计算手绘路径的总长度
        # 这一步是为了生成足够密度的点，保证后续重采样顺利
        diffs = np.linalg.norm(path[1:] - path[:-1], axis=1)
        total_len = np.sum(diffs)
        
        # 2. 生成密集点 (假设每 0.1米 一个点)
        num_points = int(total_len / 0.1) + 10  # 多给一点余量
        num_points = max(num_points, 100)       # 至少100个点
        
        # 3. 线性插值生成 dense_curve
        # (这里用简单的线性插值足够了，因为手绘本身就是密集的)
        # 如果想要更平滑，可以用 B-Spline，但这里先跑通为主
        t_old = np.linspace(0, 1, len(path))
        t_new = np.linspace(0, 1, num_points)
        
        x_new = np.interp(t_new, t_old, path[:, 0])
        z_new = np.interp(t_new, t_old, path[:, 1])
        
        dense_curve = np.stack([x_new, z_new], axis=1)
        return dense_curve

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


import torch

def diff_sdf_circle(point, center, radius):
    """
    圆形的 SDF 计算 (可导)
    point: [Batch, Frames, 2] (x, z)
    center: [2]
    radius: scalar
    """
    # 距离圆心的距离
    dist_to_center = torch.norm(point - center, dim=-1)
    # SDF = 距离 - 半径
    # 结果 < 0 表示在圆内，> 0 表示在圆外
    return dist_to_center - radius

def diff_sdf_box(point, center, size):
    """
    矩形(Box)的 SDF 计算 (可导)
    point: [Batch, Frames, 2] (x, z)
    center: [2] (cx, cz)
    size: [2] (width, depth) -> 全尺寸，不是半尺寸
    """
    # 1. 将点转换到 Box 的局部坐标系 (以 Box 中心为原点)
    # 并利用对称性，全部映射到第一象限 (abs)
    p = point - center
    d = torch.abs(p) - (size / 2.0)

    # 2. 计算外部距离 (outside distance)
    # max(d, 0)
    outside_dist = torch.norm(torch.clamp(d, min=0.0), dim=-1)

    # 3. 计算内部距离 (inside distance)
    # min(max(d.x, d.y), 0.0)
    # 只有当点在盒子内部时，这个值才有效(为负)，表示离最近边的距离
    inside_dist = torch.min(torch.max(d, dim=-1)[0], torch.tensor(0.0).to(point.device))

    return outside_dist + inside_dist