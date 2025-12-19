import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import matplotlib.patches as patches

# ==========================================
# 1. 模拟工具函数
# ==========================================

def mock_content_distance_profile(frames, avg_speed=0.05):
    """
    模拟 Content Motion 的累积路程。
    为了真实，我们加一点随机波动（模拟忽快忽慢）。
    """
    # 基础速度 + 随机波动
    speeds = np.abs(np.random.normal(loc=avg_speed, scale=0.02, size=frames))
    # 第一帧位移为 0
    dists = np.concatenate(([0], speeds))
    # 累积求和
    cum_dist = np.cumsum(dists)
    return cum_dist

def fit_spline_curve(points, num_dense=1000):
    """
    将稀疏的关键点拟合为平滑曲线 (B-Spline)
    """
    points = np.array(points)
    x = points[:, 0]
    z = points[:, 1]
    
    # 参数化 (这里简单用索引，更好的做法是用弦长参数化)
    t = np.linspace(0, 1, len(points))
    t_dense = np.linspace(0, 1, num_dense)
    
    # k=3 (三次样条), bc_type=natural (自然边界)
    spline_x = make_interp_spline(t, x, k=3)
    spline_z = make_interp_spline(t, z, k=3)
    
    dense_x = spline_x(t_dense)
    dense_z = spline_z(t_dense)
    
    dense_points = np.stack([dense_x, dense_z], axis=1)
    return dense_points

def resample_trajectory(dense_curve, target_distances):
    """
    核心算法：基于弧长的重采样 (截断策略)
    """
    # 1. 计算 dense curve 的累积弧长
    diffs = np.linalg.norm(dense_curve[1:] - dense_curve[:-1], axis=1)
    curve_cum_dist = np.concatenate(([0], np.cumsum(diffs)))
    max_curve_len = curve_cum_dist[-1]
    
    # 2. 对每个 target distance 进行插值
    resampled_points = []
    valid_mask = [] # 记录是否超出了曲线长度
    
    for dist in target_distances:
        if dist > max_curve_len:
            # 策略：超出部分停在终点 (或者你可以选择不做任何操作，让mask处理)
            resampled_points.append(dense_curve[-1])
            valid_mask.append(False) # 标记为无效/停止
        else:
            # 找到 dist 对应的索引
            # 使用 np.interp 在 (curve_cum_dist -> x/z) 之间做映射
            rx = np.interp(dist, curve_cum_dist, dense_curve[:, 0])
            rz = np.interp(dist, curve_cum_dist, dense_curve[:, 1])
            resampled_points.append([rx, rz])
            valid_mask.append(True)
            
    return np.array(resampled_points), np.array(valid_mask)

def calculate_orientation_and_feats(points):
    """
    从点序列反算朝向和 trans_cond
    """
    # 简单的差分计算朝向
    # dir[t] = point[t+1] - point[t]
    # 最后一帧沿用倒数第二帧的方向
    tangents = np.diff(points, axis=0)
    tangents = np.concatenate([tangents, tangents[-1:]], axis=0)
    
    # 计算角度 (atan2) -> 这是 World Heading
    angles = np.arctan2(tangents[:, 0], tangents[:, 1]) # 注意: x, z 的顺序对应 sin/cos
    
    # 平滑角度 (处理 -pi 到 pi 的突变)
    angles = np.unwrap(angles)
    
    return angles

# ==========================================
# 2. 可视化函数
# ==========================================
def visualize_pipeline(scene_data, dense_curve, resampled_points, valid_mask, angles):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 1. 画障碍物
    obstacles = scene_data['environment']['obstacles']
    for obs in obstacles:
        if obs['type'] == 'circle':
            circle = patches.Circle(obs['center'], obs['radius'], color='gray', alpha=0.5, label='Obstacle')
            ax.add_patch(circle)
        elif obs['type'] == 'box':
            # box 需要 min_bound/max_bound 或 center/size，这里假设 center/size
            cx, cz = obs['center']
            w, h = obs['size']
            rect = patches.Rectangle((cx - w/2, cz - h/2), w, h, color='gray', alpha=0.5)
            ax.add_patch(rect)

    # 2. 画用户输入的关键点
    user_points = np.array(scene_data['user_input']['trajectory_points'])
    ax.plot(user_points[:, 0], user_points[:, 1], 'rx', markersize=10, markeredgewidth=2, label='User Clicks')
    
    # 3. 画拟合的光滑曲线 (背景)
    ax.plot(dense_curve[:, 0], dense_curve[:, 1], 'k--', alpha=0.3, label='Fitted Spline')
    
    # 4. 画重采样后的点 (这就是 trans_cond 的来源)
    # 用颜色区分：有效路径(蓝色) vs 截断停止(红色)
    valid_points = resampled_points[valid_mask]
    stopped_points = resampled_points[~valid_mask]
    
    if len(valid_points) > 0:
        ax.scatter(valid_points[:, 0], valid_points[:, 1], c='blue', s=20, label='Resampled (Active)')
        # 画朝向箭头 (只画一部分，不然太密)
        for i in range(0, len(valid_points), 5): # 每5帧画一个
            x, z = valid_points[i]
            a = angles[i]
            # 长度代表速度? 这里固定长度方便看
            ax.arrow(x, z, 0.2*np.sin(a), 0.2*np.cos(a), head_width=0.05, head_length=0.1, fc='green', ec='green')

    if len(stopped_points) > 0:
        ax.scatter(stopped_points[:, 0], stopped_points[:, 1], c='red', s=50, marker='x', label='Truncated (Stop)')

    ax.set_aspect('equal')
    ax.legend()
    ax.set_title("Trajectory Processing Pipeline Test")
    ax.grid(True)
    plt.savefig("pipeline_test.png")
    print("可视化结果已保存至 pipeline_test.png")

# ==========================================
# 3. 主流程
# ==========================================
def main():
    # 1. 加载 Mock JSON
    # (实际使用时: with open('test_scene.json') as f: scene_data = json.load(f))
    scene_json_str = """
    {
      "environment": {
        "obstacles": [
          {"type": "circle", "center": [2.0, 2.0], "radius": 0.8},
          {"type": "box", "center": [4.0, 0.5], "size": [1.0, 3.0]}
        ]
      },
      "user_input": {
        "trajectory_points": [[0,0], [0.5, 2.0], [3.0, 3.5], [5.0, 1.0], [6.0, 4.0]]
      },
      "content_motion_mock": {
        "total_frames": 100,
        "avg_speed": 0.15
      }
    }
    """
    scene_data = json.loads(scene_json_str)
    
    # 2. 模拟 Content 的距离 Profile
    # 假设 Content Motion 是 100 帧
    frames = scene_data['content_motion_mock']['total_frames']
    avg_speed = scene_data['content_motion_mock']['avg_speed']
    cum_dist_profile = mock_content_distance_profile(frames, avg_speed)
    
    print(f"Content 总长度: {cum_dist_profile[-1]:.2f} 米")
    
    # 3. 拟合用户曲线
    user_points = scene_data['user_input']['trajectory_points']
    dense_curve = fit_spline_curve(user_points)
    
    # 计算一下用户画的线有多长
    curve_len = np.sum(np.linalg.norm(np.diff(dense_curve, axis=0), axis=1))
    print(f"用户曲线总长度: {curve_len:.2f} 米")
    
    # 4. 重采样 (Truncation 逻辑)
    # 我们只取前 frames 个点 (其实是 frames+1 个距离节点，取 diff 后变 frames)
    # 注意: cum_dist_profile 的长度是 frames + 1 (包含起点的0)
    # 我们需要 frames 个位置点 (对应每一帧的结束位置? 或者开始位置?)
    # 通常 trans_cond 是每一帧的速度，积分得到位置。
    # 简单起见，我们取 profile[1:] 作为每一帧结束时的目标距离
    target_dists = cum_dist_profile[1:] 
    
    resampled_points, valid_mask = resample_trajectory(dense_curve, target_dists)
    
    # 5. 计算特征 (朝向)
    angles = calculate_orientation_and_feats(resampled_points)
    
    # 6. 可视化
    visualize_pipeline(scene_data, dense_curve, resampled_points, valid_mask, angles)

if __name__ == "__main__":
    main()