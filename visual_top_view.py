# # import matplotlib.pyplot as plt
# # import matplotlib.animation as animation
# # import numpy as np

# # def render_top_view(motion_data, save_path, gap_width=None, fps=20):
# #     """
# #     渲染俯视图 (XZ平面) 视频，适配“相对狭窄”引导。
# #     动态计算角色的朝向，并展示随身的宽度限制。
# #     """
# #     # [Frames, 22, 3]
# #     if motion_data.shape[0] == 22: 
# #         motion_data = motion_data.transpose(2, 0, 1)
# #     frames = motion_data.shape[0]
    
# #     kinematic_chain = [
# #         [0, 1, 4, 7, 10], # 左腿
# #         [0, 2, 5, 8, 11], # 右腿
# #         [0, 3, 6, 9, 12, 15], # 脊柱
# #         [9, 13, 16, 18, 20], # 左臂
# #         [9, 14, 17, 19, 21]  # 右臂
# #     ]

# #     # 画布设置
# #     fig, ax = plt.subplots(figsize=(8, 8))
    
# #     def update(frame):
# #         ax.clear()
# #         pose = motion_data[frame] # [22, 3]
        
# #         # --- 1. 计算当前身体坐标系 ---
# #         root = pose[0]
# #         l_hip = pose[1]
# #         r_hip = pose[2]
        
# #         # 使用左右胯的连线作为“局部 X 轴” (指向右侧)
# #         vec_right = r_hip - l_hip
# #         # 只看 XZ 平面
# #         vec_right[1] = 0 
# #         # 归一化
# #         norm = np.linalg.norm(vec_right)
# #         if norm > 1e-6:
# #             vec_right = vec_right / norm
# #         else:
# #             vec_right = np.array([1.0, 0.0, 0.0]) # 默认向右

# #         # --- 2. 绘制设置 ---
# #         ax.set_title(f"Local Gap Check (Frame {frame})")
# #         ax.set_xlabel("X (World)")
# #         ax.set_ylabel("Z (World)")
# #         ax.grid(True, linestyle='--', alpha=0.3)
        
# #         # 相机跟随 Root，保持 4x4 米的视野
# #         root_x, root_z = root[0], root[2]
# #         ax.set_xlim(root_x - 2, root_x + 2)
# #         ax.set_ylim(root_z - 2, root_z + 2)
        
# #         # --- 3. 绘制随身缝隙 (Gap Limits) ---
# #         if gap_width is not None:
# #             hw = gap_width / 2.0
            
# #             # 计算左边界点和右边界点 (相对于 Root)
# #             # Right Limit = Root + vec_right * hw
# #             # Left Limit  = Root - vec_right * hw
# #             p_right = root + vec_right * hw
# #             p_left  = root - vec_right * hw
            
# #             # 绘制一条横跨身体的线，代表当前的通道宽度
# #             ax.plot([p_left[0], p_right[0]], [p_left[2], p_right[2]], 
# #                     color='red', linewidth=1, linestyle='--', label='Max Width')
            
# #             # 在左右边界画两个短竖线 (模拟墙壁切面)
# #             # 计算垂直于 right 的 forward 向量
# #             vec_fwd = np.array([-vec_right[2], 0, vec_right[0]])
            
# #             for p in [p_left, p_right]:
# #                 w_start = p - vec_fwd * 0.5
# #                 w_end   = p + vec_fwd * 0.5
# #                 ax.plot([w_start[0], w_end[0]], [w_start[2], w_end[2]], 
# #                         color='gray', linewidth=3, alpha=0.5)

# #             ax.text(root_x, root_z + 1.8, f"Gap: {gap_width}m", ha='center', color='red')

# #         # --- 4. 绘制骨架 & 检测碰撞 ---
# #         for chain in kinematic_chain:
# #             xs = pose[chain, 0]
# #             zs = pose[chain, 2]
# #             ax.plot(xs, zs, color='#4D84AA', linewidth=2.0)
            
# #             # 绘制关节点 (带碰撞检测)
# #             for j_idx in chain:
# #                 j_pos = pose[j_idx]
                
# #                 # 碰撞检测逻辑：
# #                 # 计算关节相对于 Root 的向量
# #                 rel_pos = j_pos - root
# #                 # 投影到局部 X 轴上 (点积)
# #                 local_x = np.dot(rel_pos, vec_right)
                
# #                 # 判断是否超宽
# #                 is_colliding = False
# #                 if gap_width is not None:
# #                     if abs(local_x) > gap_width / 2.0:
# #                         is_colliding = True
                
# #                 col = 'red' if is_colliding else '#DD5A37'
# #                 size = 40 if is_colliding else 20
# #                 ax.scatter(j_pos[0], j_pos[2], color=col, s=size, zorder=5)

# #         # 标记手部 (L=20, R=21) 方便观察缩手效果
# #         l_hand = pose[20]
# #         r_hand = pose[21]
# #         ax.text(l_hand[0], l_hand[2], "L", fontsize=8, color='blue')
# #         ax.text(r_hand[0], r_hand[2], "R", fontsize=8, color='blue')

# #     ani = animation.FuncAnimation(fig, update, frames=frames, interval=50)
# #     try:
# #         writer = animation.FFMpegWriter(fps=fps)
# #         ani.save(save_path, writer=writer)
# #         print(f"  -> Top-view debug video saved: {save_path}")
# #     except Exception as e:
# #         print(f"Error: {e}")
# #     plt.close()

# # import matplotlib.pyplot as plt
# # import matplotlib.animation as animation
# # import numpy as np

# # def render_top_view(motion_data, save_path, gap_width=None, fps=20):
# #     """
# #     渲染俯视图 (XZ平面)。
# #     对应 World-Space Gap Loss：墙壁是固定的直线 X = ±Width/2。
# #     """
# #     if motion_data.shape[0] == 22: 
# #         motion_data = motion_data.transpose(2, 0, 1)
# #     frames = motion_data.shape[0]
    
# #     kinematic_chain = [
# #         [0, 1, 4, 7, 10], [0, 2, 5, 8, 11], [0, 3, 6, 9, 12, 15], 
# #         [9, 13, 16, 18, 20], [9, 14, 17, 19, 21]
# #     ]

# #     fig, ax = plt.subplots(figsize=(6, 10))
    
# #     def update(frame):
# #         ax.clear()
# #         pose = motion_data[frame] # [22, 3] World Coords
        
# #         # --- 1. 计算 Root 轨迹 (用于对齐显示) ---
# #         # 我们假设数据已经是世界坐标 (motion_output 出来就是世界坐标)
# #         root_z = pose[0, 2]
        
# #         ax.set_title(f"World Gap Check (Frame {frame})")
# #         ax.set_xlabel("World X")
# #         ax.set_ylabel("World Z")
# #         ax.grid(True, linestyle='--', alpha=0.3)
        
# #         # X轴固定显示 -1.5 ~ 1.5 米
# #         ax.set_xlim(-1.5, 1.5)
# #         # Z轴跟随角色
# #         ax.set_ylim(root_z - 2, root_z + 3)
        
# #         # --- 2. 绘制固定的世界墙壁 ---
# #         if gap_width is not None:
# #             hw = gap_width / 2.0
# #             # 左墙 (-X)
# #             ax.axvline(x=-hw, color='red', linewidth=3, alpha=0.6)
# #             # 画阴影区
# #             ax.fill_betweenx([root_z-5, root_z+5], -2, -hw, color='red', alpha=0.1)
            
# #             # 右墙 (+X)
# #             ax.axvline(x=hw, color='red', linewidth=3, alpha=0.6)
# #             ax.fill_betweenx([root_z-5, root_z+5], hw, 2, color='red', alpha=0.1)
            
# #             ax.text(0, root_z + 2.5, f"Gap: {gap_width}m", ha='center', color='red', fontweight='bold')

# #         # --- 3. 绘制骨架 ---
# #         for chain in kinematic_chain:
# #             xs = pose[chain, 0]
# #             zs = pose[chain, 2]
            
# #             # 检测碰撞并变色
# #             # 简单的 World X 判断
# #             is_hit = False
# #             if gap_width is not None:
# #                 # 只要链条上有任何一点出界，整条线变色 (或者只变点)
# #                 if np.any(np.abs(xs) > gap_width / 2.0):
# #                     is_hit = True
            
# #             color = '#DD5A37' if is_hit else '#4D84AA'
# #             ax.plot(xs, zs, color=color, linewidth=2.0)
# #             ax.scatter(xs, zs, color=color, s=20)

# #     ani = animation.FuncAnimation(fig, update, frames=frames, interval=50)
# #     try:
# #         writer = animation.FFMpegWriter(fps=fps)
# #         ani.save(save_path, writer=writer)
# #         print(f"  -> Top-view debug video saved: {save_path}")
# #     except Exception as e:
# #         print(f"Error: {e}")
# #     plt.close()

# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import numpy as np

# def render_top_view(motion_data, save_path, gap_width=None, fps=20):
#     """
#     渲染俯视图 (XZ平面) + 朝向箭头。
#     红色箭头 = 身体朝向 (Facing)
#     蓝色箭头 = 实际移动方向 (Velocity)
#     """
#     if motion_data.shape[0] == 22: 
#         motion_data = motion_data.transpose(2, 0, 1)
#     frames = motion_data.shape[0]
    
#     # ... (kinematic_chain 定义保持不变) ...
#     kinematic_chain = [
#         [0, 1, 4, 7, 10], [0, 2, 5, 8, 11], [0, 3, 6, 9, 12, 15], 
#         [9, 13, 16, 18, 20], [9, 14, 17, 19, 21]
#     ]

#     fig, ax = plt.subplots(figsize=(6, 8))
    
#     def update(frame):
#         ax.clear()
#         pose = motion_data[frame] # [22, 3] World Coords
        
#         # --- 计算向量 ---
#         root = pose[0]
#         root_x, root_z = root[0], root[2]
        
#         # 1. 身体朝向 (红色箭头)
#         # 通过左右胯 (L_Hip=1, R_Hip=2) 计算
#         l_hip = pose[1]
#         r_hip = pose[2]
#         vec_right = r_hip - l_hip
#         vec_right[1] = 0 # 忽略高度
#         # 朝向(Forward) 是 Right 向量逆时针旋转 90度
#         # vec_facing = np.array([-vec_right[2], 0, vec_right[0]])
#         # [修改] 如果发现箭头指向背后，直接加个负号取反即可
#         vec_facing = -1.0 * np.array([-vec_right[2], 0, vec_right[0]])
#         norm_f = np.linalg.norm(vec_facing)
#         if norm_f > 0: vec_facing /= norm_f
        
#         # 2. 实际移动速度 (蓝色箭头)
#         if frame < frames - 1:
#             next_root = motion_data[frame+1][0]
#             vec_vel = next_root - root
#             vec_vel[1] = 0
#             # 放大一点方便看
#             vec_vel = vec_vel * 5.0 
#         else:
#             vec_vel = np.array([0, 0, 0])

#         # --- 绘图设置 ---
#         ax.set_title(f"Top View Analysis (Frame {frame})")
#         ax.set_xlabel("World X")
#         ax.set_ylabel("World Z")
#         ax.grid(True, linestyle='--', alpha=0.3)
#         ax.set_xlim(-1.5, 1.5)
#         ax.set_ylim(root_z - 2, root_z + 3)
        
#         # --- 绘制墙壁 ---
#         if gap_width is not None:
#             hw = gap_width / 2.0
#             ax.axvline(x=-hw, color='black', linewidth=3, alpha=0.5)
#             ax.axvline(x=hw, color='black', linewidth=3, alpha=0.5)
#             ax.fill_betweenx([root_z-5, root_z+5], -2, -hw, color='gray', alpha=0.2)
#             ax.fill_betweenx([root_z-5, root_z+5], hw, 2, color='gray', alpha=0.2)

#         # --- 绘制骨架 ---
#         for chain in kinematic_chain:
#             xs = pose[chain, 0]
#             zs = pose[chain, 2]
#             # 碰撞检测
#             is_hit = False
#             if gap_width and np.any(np.abs(xs) > gap_width/2.0):
#                 is_hit = True
#             col = 'red' if is_hit else '#4D84AA'
#             ax.plot(xs, zs, color=col, linewidth=2)
#             ax.scatter(xs, zs, color=col, s=20)

#         # --- 绘制箭头 (Debug核心) ---
#         # 红色：脸朝哪
#         ax.arrow(root_x, root_z, vec_facing[0]*0.5, vec_facing[2]*0.5, 
#                  head_width=0.1, head_length=0.1, fc='red', ec='red', label='Facing')
#         # 蓝色：往哪走
#         ax.arrow(root_x, root_z, vec_vel[0], vec_vel[2], 
#                  head_width=0.1, head_length=0.1, fc='blue', ec='blue', label='Velocity')
        
#         ax.legend(loc='upper right')

#     ani = animation.FuncAnimation(fig, update, frames=frames, interval=50)
#     try:
#         writer = animation.FFMpegWriter(fps=fps)
#         ani.save(save_path, writer=writer)
#         print(f"  -> Debug Top-view saved: {save_path}")
#     except Exception as e:
#         print(f"Error: {e}")
#     plt.close()

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def render_top_view(motion_data, save_path, gap_info=None, fps=20):
    """
    渲染俯视图 (XZ平面) + 朝向箭头 + 动态墙壁。
    gap_info: 可以是 float, list (timeline), 或 None
    """
    if motion_data.shape[0] == 22: 
        motion_data = motion_data.transpose(2, 0, 1)
    frames = motion_data.shape[0]
    
    kinematic_chain = [
        [0, 1, 4, 7, 10], [0, 2, 5, 8, 11], [0, 3, 6, 9, 12, 15], 
        [9, 13, 16, 18, 20], [9, 14, 17, 19, 21]
    ]

    fig, ax = plt.subplots(figsize=(6, 8))
    
    # === [新增] 解析当前帧缝隙宽度 ===
    def get_current_gap(frame_idx, g_info):
        if g_info is None: return None
        # 1. 固定值
        if isinstance(g_info, (float, int)): return float(g_info)
        # 2. 列表 (Timeline)
        if hasattr(g_info, '__iter__') and len(g_info) > 0:
            for segment in g_info:
                # 兼容 [s, e, w] 格式
                if len(segment) >= 3:
                    s, e, w = segment[:3]
                    if s <= frame_idx < e:
                        return float(w)
            return None
        return None
    # =================================

    def update(frame):
        ax.clear()
        pose = motion_data[frame] # [22, 3] World Coords
        
        # 获取当前帧的 gap width
        curr_gap = get_current_gap(frame, gap_info)
        
        # --- 计算向量 ---
        root = pose[0]
        root_x, root_z = root[0], root[2]
        
        l_hip = pose[1]
        r_hip = pose[2]
        vec_right = r_hip - l_hip
        vec_right[1] = 0 
        vec_facing = -1.0 * np.array([-vec_right[2], 0, vec_right[0]])
        norm_f = np.linalg.norm(vec_facing)
        if norm_f > 0: vec_facing /= norm_f
        
        if frame < frames - 1:
            next_root = motion_data[frame+1][0]
            vec_vel = next_root - root
            vec_vel[1] = 0
            vec_vel = vec_vel * 5.0 
        else:
            vec_vel = np.array([0, 0, 0])

        # --- 绘图设置 ---
        title_str = f"Top View (Frame {frame})"
        if curr_gap is not None:
            title_str += f" | Gap: {curr_gap:.2f}m"
        else:
            title_str += " | Open"
            
        ax.set_title(title_str)
        ax.set_xlabel("World X")
        ax.set_ylabel("World Z")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlim(-2.0, 2.0) # 稍微宽一点视野
        ax.set_ylim(root_z - 2, root_z + 3)
        
        # --- [修改] 绘制墙壁 ---
        if curr_gap is not None:
            hw = curr_gap / 2.0
            # 画线
            ax.axvline(x=-hw, color='black', linewidth=3, alpha=0.5)
            ax.axvline(x=hw, color='black', linewidth=3, alpha=0.5)
            # 填充阴影
            ax.fill_betweenx([root_z-5, root_z+5], -3, -hw, color='gray', alpha=0.2)
            ax.fill_betweenx([root_z-5, root_z+5], hw, 3, color='gray', alpha=0.2)

        # --- 绘制骨架 ---
        for chain in kinematic_chain:
            xs = pose[chain, 0]
            zs = pose[chain, 2]
            
            # 简单的碰撞检测变色
            is_hit = False
            if curr_gap is not None and np.any(np.abs(xs) > curr_gap/2.0):
                is_hit = True
                
            col = 'red' if is_hit else '#4D84AA'
            ax.plot(xs, zs, color=col, linewidth=2)
            ax.scatter(xs, zs, color=col, s=20)

        # --- 绘制箭头 ---
        ax.arrow(root_x, root_z, vec_facing[0]*0.5, vec_facing[2]*0.5, 
                 head_width=0.1, head_length=0.1, fc='red', ec='red', label='Facing')
        ax.arrow(root_x, root_z, vec_vel[0], vec_vel[2], 
                 head_width=0.1, head_length=0.1, fc='blue', ec='blue', label='Velocity')
        
        ax.legend(loc='upper right')

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50)
    try:
        writer = animation.FFMpegWriter(fps=fps)
        ani.save(save_path, writer=writer)
        print(f"  -> Debug Top-view saved: {save_path}")
    except Exception as e:
        print(f"Error: {e}")
    plt.close()