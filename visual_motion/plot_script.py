import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap

# 辅助函数（与你的脚本保持一致）
def list_cut_average(ll, intervals):
    if intervals == 1: return ll
    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new



# def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset='humanml', figsize=(3, 3), fps=120,
#                    radius=3, vis_mode='default', gt_frames=[], view_mode='genshin_impact'):
#     matplotlib.use('Agg')

#     title = '\n'.join(wrap(title, 20))

#     def plot_xzPlane(ax, minx, maxx, miny, minz, maxz):
#         verts = [
#             [minx, miny, minz],
#             [minx, miny, maxz],
#             [maxx, miny, maxz],
#             [maxx, miny, minz]
#         ]
#         xz_plane = Poly3DCollection([verts])
#         xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
#         ax.add_collection3d(xz_plane)

#     # (seq_len, joints_num, 3)
#     data = joints.copy().reshape(len(joints), -1, 3)

#     # 1. 应用数据集缩放
#     if dataset == 'kit': data *= 0.003
#     elif dataset == 'humanml': data *= 1.3
#     elif dataset in ['humanact12', 'uestc']: data *= -1.5

#     fig = plt.figure(figsize=figsize)
#     ax = fig.add_subplot(111, projection='3d')
#     fig.suptitle(title, fontsize=10)

#     # 2. 计算 Min/Max 并分离姿态和轨迹
#     MINS = data.min(axis=0).min(axis=0)
#     MAXS = data.max(axis=0).max(axis=0)

#     height_offset = MINS[1]
#     data[:, :, 1] -= height_offset
#     trajec = data[:, 0, [0, 2]].copy() # 提取轨迹

#     # 将每一帧的角色都拉回原点 (X=0, Z=0)
#     data[..., 0] -= data[:, 0:1, 0]
#     data[..., 2] -= data[:, 0:1, 2]

#     frame_number = data.shape[0]
    
#     # 3. 定义 update 函数，这是动画的核心
#     def update(index):
#         # [核心修复] 清空上一帧的所有元素，为重绘做准备
#         ax.clear()

#         # --- 重新设置每一帧的坐标轴和视图 (与原始脚本一致) ---
#         ax.set_xlim3d([-radius / 2, radius / 2])
#         ax.set_ylim3d([0, radius])
#         ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
#         ax.view_init(elev=120, azim=-90)
#         ax.dist = 7.5
#         ax.grid(b=False)
#         plt.axis('off')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_zticklabels([])
#         # --- 设置结束 ---

#         # [核心修复] 在每一帧，根据当前轨迹，动态地重绘地面！
#         # 地面的坐标 = 固定的世界边界 - 当前帧的轨迹偏移
#         plot_xzPlane(ax, MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, 
#                      MINS[2] - trajec[index, 1], MAXS[2] - trajec[index, 1])

#         # --- 绘制骨架 (与原始脚本一致) ---
#         colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]
#         colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]
        
#         used_colors = colors_blue if index in gt_frames else colors_orange
#         if vis_mode == 'upper_body':
#             used_colors[0] = colors_blue[0]
#             used_colors[1] = colors_blue[1]
#         elif vis_mode == 'gt':
#             used_colors = colors_blue
            
#         for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
#             linewidth = 4.0 if i < 5 else 2.0
#             # 绘制的是已经被拉回原点的 data
#             ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], 
#                       linewidth=linewidth, color=color)

#     # 4. 创建并保存动画
#     ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, repeat=False)
    
#     writer = FFMpegFileWriter(fps=fps)
#     ani.save(save_path, writer=writer)
    
#     plt.close(fig)


import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap

def list_cut_average(ll, intervals):
    if intervals == 1: return ll
    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new =[]
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new

# # [修改点] 增加 ceiling_spatial 参数
# def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset='humanml', figsize=(3, 3), fps=120,
#                    radius=3, vis_mode='default', gt_frames=[], view_mode='genshin_impact',
#                    ceiling_spatial=None, view_angles=(120, -90)):
#     matplotlib.use('Agg')

#     title = '\n'.join(wrap(title, 20))

#     def plot_xzPlane(ax, minx, maxx, miny, minz, maxz):
#         verts = [
#             [minx, miny, minz],[minx, miny, maxz],
#             [maxx, miny, maxz],[maxx, miny, minz]
#         ]
#         xz_plane = Poly3DCollection([verts])
#         xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
#         ax.add_collection3d(xz_plane)

#     # (seq_len, joints_num, 3)
#     data = joints.copy().reshape(len(joints), -1, 3)

#     # 1. 应用数据集缩放 (记住这个缩放比例，天花板也要等比例缩放！)
#     scale_factor = 1.0
#     if dataset == 'kit': scale_factor = 0.003
#     elif dataset == 'humanml': scale_factor = 1.3
#     elif dataset in ['humanact12', 'uestc']: scale_factor = -1.5
#     data *= scale_factor

#     fig = plt.figure(figsize=figsize)
#     ax = fig.add_subplot(111, projection='3d')
#     fig.suptitle(title, fontsize=10)

#     # 2. 计算 Min/Max 并分离姿态和轨迹
#     MINS = data.min(axis=0).min(axis=0)
#     MAXS = data.max(axis=0).max(axis=0)

#     height_offset = MINS[1]
#     data[:, :, 1] -= height_offset
#     trajec = data[:, 0, [0, 2]].copy() # 提取轨迹 [X, Z]

#     # 将每一帧的角色都拉回原点 (X=0, Z=0)
#     data[..., 0] -= data[:, 0:1, 0]
#     data[..., 2] -= data[:, 0:1, 2]

#     frame_number = data.shape[0]
    
#     # def update(index):
#     #     ax.clear()

#     #     # --- 重新设置每一帧的坐标轴和视图 ---
#     #     ax.set_xlim3d([-radius / 2, radius / 2])
#     #     ax.set_ylim3d([0, radius])
#     #     ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        
#     #     #[核心修改 1] 使用外部传进来的相机视角
#     #     ax.view_init(elev=view_angles[0], azim=view_angles[1])
        
#     #     ax.dist = 7.5
#     #     ax.grid(b=False)
#     #     plt.axis('off')
#     #     ax.set_xticklabels([])
#     #     ax.set_yticklabels([])
#     #     ax.set_zticklabels([])

#     #     # --- 动态地重绘地面 ---
#     #     plot_xzPlane(ax, MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, 
#     #                  MINS[2] - trajec[index, 1], MAXS[2] - trajec[index, 1])

#     #     # =================[动态绘制物理空间天花板] =================
#     #     if ceiling_spatial is not None:
#     #         cur_tx = trajec[index, 0]
#     #         cur_tz = trajec[index, 1]
            
#     #         for segment in ceiling_spatial:
#     #             z_start = segment[0] * scale_factor
#     #             z_end = segment[1] * scale_factor
#     #             h = segment[2] * scale_factor
                
#     #             display_z_start = z_start - cur_tz
#     #             display_z_end = z_end - cur_tz
                
#     #             minx = MINS[0] - cur_tx - 2.0
#     #             maxx = MAXS[0] - cur_tx + 2.0
                
#     #             verts = [
#     #                 [minx, h, display_z_start],
#     #                 [minx, h, display_z_end],
#     #                 [maxx, h, display_z_end],[maxx, h, display_z_start]
#     #             ]
                
#     #             # [核心修改 2] 增加 edgecolor(深红色边框) 和 linewidth(边框粗细)
#     #             # 这样看起来就像是一个实体的玻璃顶棚，边缘非常清晰
#     #             ceil_plane = Poly3DCollection([verts], alpha=0.35, facecolor='#FF4444', 
#     #                                           edgecolor='darkred', linewidth=3.0)
#     #             ax.add_collection3d(ceil_plane)
#     #     # =====================================================================

#     #     # --- 绘制骨架 ---
#     #     colors_orange =["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]
#     #     colors_blue =["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]
        
#     #     used_colors = colors_blue if index in gt_frames else colors_orange
#     #     if vis_mode == 'upper_body':
#     #         used_colors[0] = colors_blue[0]
#     #         used_colors[1] = colors_blue[1]
#     #     elif vis_mode == 'gt':
#     #         used_colors = colors_blue
            
#     #     for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
#     #         linewidth = 4.0 if i < 5 else 2.0
#     #         ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], 
#     #                   linewidth=linewidth, color=color)
#     def update(index):
#         ax.clear()

#         # --- 重新设置坐标轴 (注意：Y和Z的范围互换了！) ---
#         ax.set_xlim3d([-radius / 2, radius / 2])
#         ax.set_ylim3d([-radius / 3., radius * 2 / 3.]) # 现在这里对应深度
#         ax.set_zlim3d([0, radius])                     # 现在这里对应高度
        
#         # 使用传进来的正常视角
#         ax.view_init(elev=view_angles[0], azim=view_angles[1])
#         ax.dist = 7.5
#         ax.grid(b=False)
#         plt.axis('off')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_zticklabels([])

#         # --- 绘制地面 ---
#         # 同样需要交换 Y 和 Z
#         minx, maxx = MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0]
#         minz, maxz = MINS[2] - trajec[index, 1], MAXS[2] - trajec[index, 1]
#         verts_ground = [[minx, minz, 0],
#             [minx, maxz, 0],
#             [maxx, maxz, 0],
#             [maxx, minz, 0]
#         ]
#         xz_plane = Poly3DCollection([verts_ground])
#         xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
#         ax.add_collection3d(xz_plane)

#         # ================= [绘制物理空间天花板] =================
#         if ceiling_spatial is not None:
#             cur_tx = trajec[index, 0]
#             cur_tz = trajec[index, 1]
            
#             for segment in ceiling_spatial:
#                 z_start = segment[0] * scale_factor
#                 z_end = segment[1] * scale_factor
#                 h = segment[2] * scale_factor
                
#                 display_z_start = z_start - cur_tz
#                 display_z_end = z_end - cur_tz
                
#                 minx_c = MINS[0] - cur_tx - 2.0
#                 maxx_c = MAXS[0] - cur_tx + 2.0
                
#                 # 坐标顺序：(X, Depth, Height)
#                 verts_ceil = [
#                     [minx_c, display_z_start, h],[minx_c, display_z_end, h],[maxx_c, display_z_end, h],
#                     [maxx_c, display_z_start, h]
#                 ]
                
#                 ceil_plane = Poly3DCollection([verts_ceil], alpha=0.35, facecolor='#FF4444', 
#                                                 edgecolor='darkred', linewidth=3.0)
#                 ax.add_collection3d(ceil_plane)
#         # =========================================================

#         # --- 绘制骨架 ---
#         colors_orange =["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]
#         colors_blue =["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]
        
#         used_colors = colors_blue if index in gt_frames else colors_orange
#         if vis_mode == 'upper_body':
#             used_colors[0] = colors_blue[0]
#             used_colors[1] = colors_blue[1]
#         elif vis_mode == 'gt':
#             used_colors = colors_blue
            
#         for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
#             linewidth = 4.0 if i < 5 else 2.0
#             # 【最关键的修正】：这里传入的顺序变成了 (X, Z深度, Y高度)
#             ax.plot3D(data[index, chain, 0], data[index, chain, 2], data[index, chain, 1], 
#                         linewidth=linewidth, color=color)

#             return ax.lines

#     # 4. 创建并保存动画
#     ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, repeat=False)
    
#     writer = FFMpegFileWriter(fps=fps)
#     ani.save(save_path, writer=writer)
    
#     plt.close(fig)


# 增加 ceiling_spatial 和 view_angles 参数
# def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset='humanml', figsize=(3, 3), fps=120,
#                    radius=3, vis_mode='default', gt_frames=[], view_mode='genshin_impact',
#                    ceiling_spatial=None, view_angles=(120, -90)): # 默认保持原版的 (120, -90)
#     matplotlib.use('Agg')

#     title = '\n'.join(wrap(title, 20))

#     def plot_xzPlane(ax, minx, maxx, miny, minz, maxz):
#         verts = [
#             [minx, miny, minz],[minx, miny, maxz],
#             [maxx, miny, maxz],[maxx, miny, minz]
#         ]
#         xz_plane = Poly3DCollection([verts])
#         xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
#         ax.add_collection3d(xz_plane)

#     # (seq_len, joints_num, 3)
#     data = joints.copy().reshape(len(joints), -1, 3)

#     # 1. 应用数据集缩放
#     scale_factor = 1.0
#     if dataset == 'kit': scale_factor = 0.003
#     elif dataset == 'humanml': scale_factor = 1.3
#     elif dataset in['humanact12', 'uestc']: scale_factor = -1.5
#     data *= scale_factor

#     fig = plt.figure(figsize=figsize)
#     ax = fig.add_subplot(111, projection='3d')
#     fig.suptitle(title, fontsize=10)

#     # 2. 计算 Min/Max 并分离姿态和轨迹
#     MINS = data.min(axis=0).min(axis=0)
#     MAXS = data.max(axis=0).max(axis=0)

#     height_offset = MINS[1]
#     data[:, :, 1] -= height_offset
#     trajec = data[:, 0, [0, 2]].copy() # 提取轨迹

#     # 将每一帧的角色都拉回原点 (X=0, Z=0)
#     data[..., 0] -= data[:, 0:1, 0]
#     data[..., 2] -= data[:, 0:1, 2]

#     frame_number = data.shape[0]
    
#     def update(index):
#         ax.clear()

#         # === 绝对保持原版的坐标系边界！===
#         ax.set_xlim3d([-radius / 2, radius / 2])
#         ax.set_ylim3d([0, radius])
#         ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        
#         # 应用视角 (原版是 120, -90)
#         ax.view_init(elev=view_angles[0], azim=view_angles[1])
#         ax.dist = 7.5
#         ax.grid(b=False)
#         plt.axis('off')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_zticklabels([])

#         # --- 动态重绘地面 (原版逻辑) ---
#         plot_xzPlane(ax, MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, 
#                      MINS[2] - trajec[index, 1], MAXS[2] - trajec[index, 1])

#         # ================= [安全绘制物理空间天花板] =================
#         if ceiling_spatial is not None:
#             cur_tx = trajec[index, 0]
#             cur_tz = trajec[index, 1]
            
#             for segment in ceiling_spatial:
#                 z_start = segment[0] * scale_factor
#                 z_end = segment[1] * scale_factor
#                 h = segment[2] * scale_factor
                
#                 display_z_start = z_start - cur_tz
#                 display_z_end = z_end - cur_tz
                
#                 minx_c = MINS[0] - cur_tx - 2.0
#                 maxx_c = MAXS[0] - cur_tx + 2.0
                
#                 # 严格使用原版的[X, Y, Z] 顺序，高度放在中间的 Y！
#                 verts_ceil = [[minx_c, h, display_z_start],
#                     [minx_c, h, display_z_end],[maxx_c, h, display_z_end],
#                     [maxx_c, h, display_z_start]
#                 ]
                
#                 ceil_plane = Poly3DCollection([verts_ceil], alpha=0.35, facecolor='#FF4444', 
#                                               edgecolor='darkred', linewidth=3.0)
#                 ax.add_collection3d(ceil_plane)
#         # =========================================================

#         # --- 绘制骨架 (完全恢复原版的顺序) ---
#         colors_orange =["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]
#         colors_blue =["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]
        
#         used_colors = colors_blue if index in gt_frames else colors_orange
#         if vis_mode == 'upper_body':
#             used_colors[0] = colors_blue[0]
#             used_colors[1] = colors_blue[1]
#         elif vis_mode == 'gt':
#             used_colors = colors_blue
            
#         for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
#             linewidth = 4.0 if i < 5 else 2.0
#             # 恢复了原版的 data[index, chain, 0], data[index, chain, 1], data[index, chain, 2]
#             ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], 
#                       linewidth=linewidth, color=color)

def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset='humanml', figsize=(3, 3), fps=120,
                   radius=3, vis_mode='default', gt_frames=[], view_mode='genshin_impact',
                   ceiling_spatial=None, view_angles=(120, -90), is_side_view=False):
    matplotlib.use('Agg')
    title = '\n'.join(wrap(title, 20))

    # [新增] 2D 旋转矩阵辅助函数
    def rot_y(x, z, angle_deg):
        rad = math.radians(angle_deg)
        c, s = math.cos(rad), math.sin(rad)
        return x * c - z * s, x * s + z * c

    # 如果是侧视图，世界旋转 90 度
    rot_angle = 90 if is_side_view else 0

    data = joints.copy().reshape(len(joints), -1, 3)
    scale_factor = 1.0
    if dataset == 'kit': scale_factor = 0.003
    elif dataset == 'humanml': scale_factor = 1.3
    elif dataset in ['humanact12', 'uestc']: scale_factor = -1.5
    data *= scale_factor

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(title, fontsize=10)

    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    data[:, :, 1] -= MINS[1]
    trajec = data[:, 0, [0, 2]].copy() 
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    frame_number = data.shape[0]
    
    def update(index):
        ax.clear()
        
        # 保持原汁原味的轴边界
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        
        # 使用外部视角 (默认 120, -90)
        ax.view_init(elev=view_angles[0], azim=view_angles[1])
        ax.dist = 7.5
        ax.grid(b=False)
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # --- 绘制地面 (带旋转支持) ---
        minx_f, maxx_f = MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0]
        minz_f, maxz_f = MINS[2] - trajec[index, 1], MAXS[2] - trajec[index, 1]
        
        fx = np.array([minx_f, minx_f, maxx_f, maxx_f])
        fz = np.array([minz_f, maxz_f, maxz_f, minz_f])
        
        # 如果需要看侧面，把地面旋转 90 度
        if rot_angle != 0: fx, fz = rot_y(fx, fz, rot_angle)
            
        verts_ground = [[fx[0], 0, fz[0]], [fx[1], 0, fz[1]], [fx[2], 0, fz[2]], [fx[3], 0, fz[3]]]
        xz_plane = Poly3DCollection([verts_ground], facecolor=(0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

        # ================= [绘制物理空间天花板] =================
        if ceiling_spatial is not None:
            cur_tx = trajec[index, 0]
            cur_tz = trajec[index, 1]
            
            for segment in ceiling_spatial:
                z_start, z_end, h = segment[0]*scale_factor, segment[1]*scale_factor, segment[2]*scale_factor
                
                display_z_start = z_start - cur_tz
                display_z_end = z_end - cur_tz
                minx_c = MINS[0] - cur_tx - 2.0
                maxx_c = MAXS[0] - cur_tx + 2.0
                
                cx = np.array([minx_c, minx_c, maxx_c, maxx_c])
                cz = np.array([display_z_start, display_z_end, display_z_end, display_z_start])
                
                # 如果看侧面，天花板一起转 90 度
                if rot_angle != 0: cx, cz = rot_y(cx, cz, rot_angle)
                    
                verts_ceil = [[cx[0], h, cz[0]], [cx[1], h, cz[1]], [cx[2], h, cz[2]], [cx[3], h, cz[3]]]
                ceil_plane = Poly3DCollection([verts_ceil], alpha=0.35, facecolor='#FF4444', edgecolor='darkred', linewidth=3.0)
                ax.add_collection3d(ceil_plane)
        # =========================================================

        # --- 绘制骨架 (带旋转支持) ---
        colors_orange =["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]
        colors_blue =["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]
        used_colors = colors_blue if index in gt_frames else colors_orange
            
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            linewidth = 4.0 if i < 5 else 2.0
            x_d = data[index, chain, 0]
            y_d = data[index, chain, 1]
            z_d = data[index, chain, 2]
            
            # 旋转骨架
            if rot_angle != 0: x_d, z_d = rot_y(x_d, z_d, rot_angle)
                
            ax.plot3D(x_d, y_d, z_d, linewidth=linewidth, color=color)

   
    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, repeat=False)
    writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, writer=writer)
    plt.close(fig)