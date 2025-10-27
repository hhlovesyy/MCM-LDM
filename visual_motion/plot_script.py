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
#                    # [FIX] 重新引入 radius 参数来控制大小归一化
#                    radius=3,
#                    view_mode='camera_follow', 
#                    vis_mode='default', gt_frames=[]):
#     matplotlib.use('Agg')

#     title = '\n'.join(wrap(title, 20))

#     def plot_xzPlane(ax, minx, maxx, miny, minz, maxz):
#         ## Plot a plane XZ
#         verts = [
#             [minx, miny, minz],
#             [minx, miny, maxz],
#             [maxx, miny, maxz],
#             [maxx, miny, minz]
#         ]
#         xz_plane = Poly3DCollection([verts])
#         xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
#         ax.add_collection3d(xz_plane)

#     fig = plt.figure(figsize=figsize)
#     # [FIX] 使用原始脚本的方式创建 3D 轴
#     # ax = p3.Axes3D(fig) # 原始脚本用法
#     ax = fig.add_subplot(111, projection='3d') # 你的脚本用法，保持一致
#     ax.view_init(elev=120, azim=-90)
#     ax.dist = 7.5 # 保持原始脚本的相机距离

#     fig.suptitle(title, fontsize=10)

#     # (seq_len, joints_num, 3)
#     data = joints.copy().reshape(len(joints), -1, 3)
    
#     # 1. 应用数据集缩放（决定人体绝对大小）
#     if dataset == 'kit': data *= 0.003
#     elif dataset == 'humanml': data *= 1.3
#     elif dataset in ['humanact12', 'uestc']: data *= -1.5

#     # 2. 计算 Min/Max 和应用平移（决定人体位置）
    
#     # 始终计算缩放后的数据的 Min/Max，用于地面绘制和相机跟随的偏移量计算
#     MINS_before_trans = data.min(axis=0).min(axis=0)
#     MAXS_before_trans = data.max(axis=0).max(axis=0)

#     if view_mode == 'camera_follow':
#         # 移除全局位移，让相机跟随角色 (与原始脚本逻辑一致)
#         height_offset = MINS_before_trans[1]
#         data[:, :, 1] -= height_offset # 将最低点移动到 Y=0 附近

#         # 计算根关节的 XZ 轨迹（用于可能绘制轨迹线，或计算 XZ 偏移量）
#         trajec = data[:, 0, [0, 2]].copy()
        
#         # 将当前帧的根关节 X, Z 坐标平移到 0
#         data[..., 0] -= data[:, 0:1, 0]
#         data[..., 2] -= data[:, 0:1, 2]

#     elif view_mode == 'fixed_camera':
#         # 不做任何处理，保留原始位移
#         # 此时 trajec 为 None
#         trajec = None
#     else:
#         raise ValueError(f"Unknown view_mode: {view_mode}")

#     # 3. [FIX] 使用固定 radius 来设置坐标轴范围，实现归一化大小
#     ax.set_xlim3d([-radius / 2, radius / 2])
#     ax.set_ylim3d([0, radius])
#     ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
    
#     ax.grid(b=False)
#     plt.axis('off')
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.set_zticklabels([])

#     # [FIX] 绘制地平面
#     if view_mode == 'camera_follow':
#         # 此时数据已经被平移，根关节在 X=0, Z=0，地平面范围固定在坐标轴范围
#         plot_xzPlane(ax, -radius/2, radius/2, 0, -radius/3, radius * 2 / 3)
#     elif view_mode == 'fixed_camera':
#         # 此时数据可能在很远的地方，但我们仍需一个地面作为参考。
#         # 这里用数据原始的 Min/Max 来决定地平面范围，Y=0
#         # 如果设置了固定轴，地平面也应该固定。这里我们使用固定轴的范围
#         plot_xzPlane(ax, -radius/2, radius/2, MINS_before_trans[1], -radius/3, radius * 2 / 3)


#     frame_number = data.shape[0]
    
#     lines = []
#     colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]
#     colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]
#     colors = colors_blue if vis_mode == 'gt' else colors_orange

#     for i, chain in enumerate(kinematic_tree):
#         linewidth = 4.0 if i < 5 else 2.0
#         # 使用空的 3D 线对象进行初始化
#         line, = ax.plot([], [], [], linewidth=linewidth, color=colors[i])
#         lines.append(line)

#     def update(index):
#         # [FIX] 移除 fixed_camera 下对 ax.dist 的调整，保持固定视图
#         # [FIX] 使用 ax.lines/ax.collections 清理，以支持轨迹线或地平面动态更新，
#         #       但在 FuncAnimation 中，最好只更新 line data，避免清空。
#         #       这里我们只更新人体线条。
        
#         used_colors = colors_blue if index in gt_frames else colors
        
#         for i, (line, chain) in enumerate(zip(lines, kinematic_tree)):
#             x_data, y_data, z_data = data[index, chain].T
            
#             # 动态调整颜色（如果需要）
#             linewidth = 4.0 if i < 5 else 2.0
#             line.set_color(used_colors[i])
#             line.set_linewidth(linewidth)
            
#             line.set_data(x_data, y_data)
#             line.set_3d_properties(z_data)
        
#         return lines

#     ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, blit=False)
    
#     # 原始脚本使用 FFMpegFileWriter，你的脚本使用 matplotlib.animation.FFMpegWriter
#     writer = matplotlib.animation.FFMpegWriter(fps=fps)
#     ani.save(save_path, writer=writer)
    
#     plt.close(fig)


def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset='humanml', figsize=(3, 3), fps=120,
                   radius=3, vis_mode='default', gt_frames=[], view_mode='genshin_impact'):
    matplotlib.use('Agg')

    title = '\n'.join(wrap(title, 20))

    def plot_xzPlane(ax, minx, maxx, miny, minz, maxz):
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    # 1. 应用数据集缩放
    if dataset == 'kit': data *= 0.003
    elif dataset == 'humanml': data *= 1.3
    elif dataset in ['humanact12', 'uestc']: data *= -1.5

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(title, fontsize=10)

    # 2. 计算 Min/Max 并分离姿态和轨迹
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]].copy() # 提取轨迹

    # 将每一帧的角色都拉回原点 (X=0, Z=0)
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    frame_number = data.shape[0]
    
    # 3. 定义 update 函数，这是动画的核心
    def update(index):
        # [核心修复] 清空上一帧的所有元素，为重绘做准备
        ax.clear()

        # --- 重新设置每一帧的坐标轴和视图 (与原始脚本一致) ---
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        ax.grid(b=False)
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        # --- 设置结束 ---

        # [核心修复] 在每一帧，根据当前轨迹，动态地重绘地面！
        # 地面的坐标 = 固定的世界边界 - 当前帧的轨迹偏移
        plot_xzPlane(ax, MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, 
                     MINS[2] - trajec[index, 1], MAXS[2] - trajec[index, 1])

        # --- 绘制骨架 (与原始脚本一致) ---
        colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]
        colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]
        
        used_colors = colors_blue if index in gt_frames else colors_orange
        if vis_mode == 'upper_body':
            used_colors[0] = colors_blue[0]
            used_colors[1] = colors_blue[1]
        elif vis_mode == 'gt':
            used_colors = colors_blue
            
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            linewidth = 4.0 if i < 5 else 2.0
            # 绘制的是已经被拉回原点的 data
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], 
                      linewidth=linewidth, color=color)

    # 4. 创建并保存动画
    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, repeat=False)
    
    writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, writer=writer)
    
    plt.close(fig)