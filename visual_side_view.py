# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import numpy as np

# def render_side_view(motion_data, save_path, ceiling_height=None, fps=20):
#     """
#     渲染侧视图 (YZ平面) 视频，专门用于观察高度和天花板约束。
    
#     Args:
#         motion_data (np.ndarray): 形状可以是 [Frames, 22, 3] 或 [22, 3, Frames]
#         save_path (str): 保存路径 (.mp4)
#         ceiling_height (float, optional): 天花板高度 (米). Defaults to None.
#         fps (int): 帧率. Defaults to 20.
#     """
    
#     # 1. 数据形状标准化 -> [Frames, 22, 3]
#     if motion_data.shape[0] == 22: # 如果是 [22, 3, Frames]
#         motion_data = motion_data.transpose(2, 0, 1)
    
#     frames = motion_data.shape[0]
    
#     # HumanML3D 标准骨架连接定义
#     kinematic_chain = [
#         [0, 1, 4, 7, 10], # 左腿
#         [0, 2, 5, 8, 11], # 右腿
#         [0, 3, 6, 9, 12, 15], # 脊柱 -> 头
#         [9, 13, 16, 18, 20], # 左臂
#         [9, 14, 17, 19, 21]  # 右臂
#     ]

#     # 2. 设置画布 (2D 平面)
#     fig, ax = plt.subplots(figsize=(10, 6))
    
#     # 预计算整个序列的范围，用于设置坐标轴
#     # 侧视图我们看 Z (前进方向) 和 Y (高度)
#     # 如果你是横着走的 (X轴)，这里可能要改成 motion_data[..., 0]
#     # 但通常 Forward 是 Z 轴
#     all_z = motion_data[..., 2]
#     all_y = motion_data[..., 1]
    
#     min_z, max_z = np.min(all_z), np.max(all_z)
    
#     def update(frame):
#         ax.clear()
        
#         # 获取当前帧姿态
#         # pose: [22, 3] -> (x, y, z)
#         pose = motion_data[frame]
        
#         # --- 绘制设置 ---
#         ax.set_title(f"Side View Analysis (Frame {frame}/{frames})")
#         ax.set_xlabel("Z Position (Forward)")
#         ax.set_ylabel("Y Position (Height)")
#         ax.grid(True, linestyle='--', alpha=0.5)
        
#         # 动态相机：跟随 Root 的 Z 轴，但 Y 轴固定
#         root_z = pose[0, 2]
#         window_size = 3.0 # 前后看 3 米
#         ax.set_xlim(root_z - window_size, root_z + window_size)
#         ax.set_ylim(0, 2.5) # 高度固定 0-2.5米
        
#         # --- 绘制天花板 ---
#         if ceiling_height is not None:
#             # 画一条粗红线
#             ax.axhline(y=ceiling_height, color='red', linewidth=3, linestyle='-', alpha=0.7)
#             # 标注文字
#             ax.text(root_z - window_size + 0.2, ceiling_height + 0.05, 
#                     f"Ceiling Limit: {ceiling_height:.2f}m", 
#                     color='red', fontsize=12, fontweight='bold')
            
#             # 可选：把天花板上面的区域涂成红色警告区
#             ax.fill_between([root_z - window_size, root_z + window_size], 
#                             ceiling_height, 3.0, color='red', alpha=0.1)

#         # --- 绘制骨架 ---
#         for chain in kinematic_chain:
#             # 取出 Z 和 Y 坐标
#             zs = pose[chain, 2]
#             ys = pose[chain, 1]
            
#             # 画线
#             ax.plot(zs, ys, color='#4D84AA', linewidth=2.5, alpha=0.9)
#             # 画关节
#             ax.scatter(zs, ys, color='#DD5A37', s=20, zorder=5)
            
#         # 特别标注头部 (Index 15)
#         head_y = pose[15, 1]
#         head_z = pose[15, 2]
#         # 如果头撞到了，标红点
#         head_color = 'red' if (ceiling_height and head_y > ceiling_height) else 'green'
#         ax.scatter(head_z, head_y, color=head_color, s=50, edgecolors='black', zorder=10, label='Head')

#     # 生成动画
#     ani = animation.FuncAnimation(fig, update, frames=frames, interval=50)
    
#     # 保存
#     try:
#         writer = animation.FFMpegWriter(fps=fps)
#         ani.save(save_path, writer=writer)
#         print(f"  -> Side-view debug video saved: {save_path}")
#     except Exception as e:
#         print(f"  -> Error saving side-view video: {e} (Ensure ffmpeg is installed)")
    
#     plt.close()

# if __name__ == "__main__":
#     # 简单的测试代码
#     dummy_data = np.random.rand(60, 22, 3) # 60帧随机数据
#     dummy_data[:, :, 1] *= 1.8 # 高度拉伸
#     render_side_view(dummy_data, "test_side.mp4", ceiling_height=1.4)


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def render_side_view(motion_data, save_path, ceiling_info=None, fps=20):
    """
    渲染侧视图 (YZ平面) 视频，支持固定高度或时序高度引导的可视化。
    
    Args:
        motion_data (np.ndarray): 形状可以是 [Frames, 22, 3] 或 [22, 3, Frames]
        save_path (str): 保存路径 (.mp4)
        ceiling_info (float | list | None): 
            - float: 全程固定高度 (例如 1.4)
            - list: 时序列表 (例如 [[0, 60, 0.9], [60, 100, 2.0]])
            - None: 不画线
        fps (int): 帧率. Defaults to 20.
    """
    
    # 1. 数据形状标准化 -> [Frames, 22, 3]
    if motion_data.shape[0] == 22: # 如果是 [22, 3, Frames]
        motion_data = motion_data.transpose(2, 0, 1)
    
    frames = motion_data.shape[0]
    
    # HumanML3D 标准骨架连接定义
    kinematic_chain = [
        [0, 1, 4, 7, 10], # 左腿
        [0, 2, 5, 8, 11], # 右腿
        [0, 3, 6, 9, 12, 15], # 脊柱 -> 头
        [9, 13, 16, 18, 20], # 左臂
        [9, 14, 17, 19, 21]  # 右臂
    ]

    # 2. 设置画布 (2D 平面)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 预计算整个序列的范围
    all_z = motion_data[..., 2]
    all_y = motion_data[..., 1]
    
    # === [辅助函数] 获取当前帧的天花板高度 ===
    def get_current_ceiling(frame_idx, c_info):
        if c_info is None:
            return None
        
        # 情况 A: 固定数值 (float/int)
        if isinstance(c_info, (float, int)):
            return float(c_info)
            
        # 情况 B: 列表或类似列表的对象 (包括 OmegaConf ListConfig)
        # 只要它有长度且能被遍历，且第一个元素也是列表/元组
        if hasattr(c_info, '__iter__') and len(c_info) > 0:
            # 简单的鸭子类型判断：如果它不是列表，但能迭代，我们就当它是列表
            for segment in c_info:
                # segment 可能是 [s, e, h]
                if len(segment) >= 3:
                    s, e, h = segment[:3]
                    if s <= frame_idx < e:
                        return float(h)
            return None
            
        return None

    def update(frame):
        ax.clear()
        
        # 获取当前帧姿态
        pose = motion_data[frame]
        
        # 获取当前帧限高
        curr_ceil = get_current_ceiling(frame, ceiling_info)
        
        # --- 绘制设置 ---
        title_str = f"Side View (Frame {frame}/{frames})"
        if curr_ceil is not None:
            title_str += f" | Ceiling: {curr_ceil:.2f}m"
        else:
            title_str += " | No Ceiling Limit"
            
        ax.set_title(title_str)
        ax.set_xlabel("Z Position (Forward)")
        ax.set_ylabel("Y Position (Height)")
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # 动态相机：跟随 Root 的 Z 轴
        root_z = pose[0, 2]
        window_size = 3.0 
        ax.set_xlim(root_z - window_size, root_z + window_size)
        ax.set_ylim(0, 3.0) # 稍微调高一点视野，方便看站起来的动作
        
        # --- 绘制天花板 (动态) ---
        if curr_ceil is not None:
            # 画红线
            ax.axhline(y=curr_ceil, color='red', linewidth=3, linestyle='-', alpha=0.7)
            # 填充红色区域
            ax.fill_between([root_z - window_size, root_z + window_size], 
                            curr_ceil, 3.5, color='red', alpha=0.1)
            
            # 在左上角或者红线上方写字
            ax.text(root_z - window_size + 0.2, curr_ceil + 0.1, 
                    f"LIMIT: {curr_ceil:.2f}m", 
                    color='red', fontsize=10, fontweight='bold')

        # --- 绘制骨架 ---
        for chain in kinematic_chain:
            zs = pose[chain, 2]
            ys = pose[chain, 1]
            ax.plot(zs, ys, color='#4D84AA', linewidth=2.5, alpha=0.9)
            ax.scatter(zs, ys, color='#DD5A37', s=20, zorder=5)
            
        # 头部 (Index 15)
        head_y = pose[15, 1]
        head_z = pose[15, 2]
        # 判定是否撞头
        is_hit = (curr_ceil is not None) and (head_y > curr_ceil)
        head_color = 'red' if is_hit else 'green'
        
        ax.scatter(head_z, head_y, color=head_color, s=60, edgecolors='black', zorder=10)

    # 生成动画
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50)
    
    try:
        writer = animation.FFMpegWriter(fps=fps)
        ani.save(save_path, writer=writer)
        print(f"  -> Side-view video saved: {save_path}")
    except Exception as e:
        print(f"  -> Error saving side-view video: {e}")
    
    plt.close()

if __name__ == "__main__":
    # 测试代码
    dummy_data = np.random.rand(100, 22, 3) 
    dummy_data[:, :, 1] *= 1.8 
    # 测试时序配置: 前50帧1.0米，后50帧2.0米
    timeline_test = [[0, 50, 1.0], [50, 100, 2.0]]
    render_side_view(dummy_data, "test_timeline.mp4", ceiling_info=timeline_test)