import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os

# ---------------- 配置区域 ----------------
# 骨架连接定义 (SMPL 22关节简化版)
# 格式: [起点索引, 终点索引]
# 0: Pelvis (Root)
# Left: 1, 4, 7, 10 (Legs); 16, 18, 20 (Arms)
# Right: 2, 5, 8, 11 (Legs); 17, 19, 21 (Arms)
# Center: 3, 6, 9 (Spine), 12, 15 (Head/Neck), 13, 14 (Collar)

'''
左/右肢体着色：左边身体用蓝色，右边身体用红色，躯干用黑色。这样一眼就能看出手性是否反了。
动态跟随相机：相机会自动锁定在角色的骨盆（Root）上，这样无论人走到哪里，他都在画面中心，不会跑出屏幕。
坐标系指示器：在脚下画出 RGB 三色箭头（R=X轴, G=Y轴, B=Z轴），方便你核对 HumanML3D 的 Y-up 坐标系是否正确。
'''

# 左边身体链条 (蓝色)
LEFT_CHAIN = [
    [1, 4], [4, 7], [7, 10],      # 左腿
    [13, 16], [16, 18], [18, 20]  # 左臂 (13是左肩)
]
# 右边身体链条 (红色)
RIGHT_CHAIN = [
    [2, 5], [5, 8], [8, 11],      # 右腿
    [14, 17], [17, 19], [19, 21]  # 右臂 (14是右肩)
]
# 中间躯干链条 (黑色)
CENTER_CHAIN = [
    [0, 1], [0, 2], [0, 3],       # 根节点连接
    [3, 6], [6, 9], [9, 12], [9, 13], [9, 14], [12, 15] # 脊柱与头
]

def render_video(pose_data, save_path, fps=20, downsample=1):
    """
    渲染动作视频
    :param pose_data: (Frames, 22, 3) 动作数据
    :param save_path: 保存路径 (.mp4 或 .gif)
    :param fps: 帧率
    :param downsample: 为了加快渲染速度，可以跳帧 (比如设为2，每2帧画一次)
    """
    
    """
    专门适配 HumanML3D 坐标系的可视化: Y-up, Z-forward
    """
    frames_num = pose_data.shape[0]
    print(f"Start rendering HumanML3D style... Frames: {frames_num}")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # ---------------------------------------------
    # 关键修改：Matplotlib 默认 Z 是高。
    # HumanML3D 数据中 Y 是高。
    # 映射关系: 
    #   Plot X = Data X
    #   Plot Y = Data Z (深度)
    #   Plot Z = Data Y (高度)
    # ---------------------------------------------

    # 初始化绘图元素
    lines_left = [ax.plot([], [], [], c='blue', linewidth=2, label='Left')[0] for _ in LEFT_CHAIN]
    lines_right = [ax.plot([], [], [], c='red', linewidth=2, label='Right')[0] for _ in RIGHT_CHAIN]
    lines_center = [ax.plot([], [], [], c='black', linewidth=2)[0] for _ in CENTER_CHAIN]
    
    # 关节球
    scats = ax.scatter([], [], [], s=15, c='gray')

    # 坐标系箭头 (R, G, B) -> (X, Y, Z)
    # 起点在 Root 下方
    quivers = ax.quiver([0,0,0], [0,0,0], [0,0,0], [0.5,0,0], [0,0,0.5], [0,0.5,0], color=['r', 'b', 'g'])

    # 视角设置
    radius = 1.0

    def update(frame_idx):
        # (22, 3) -> (X, Y_up, Z_fwd)
        current_pose = pose_data[frame_idx]
        
        # 获取根节点位置用于跟随
        root = current_pose[0] # x, y, z
        
        # --- 绘图数据准备 ---
        # 我们把 Data Y 喂给 Plot Z，把 Data Z 喂给 Plot Y
        xs = current_pose[:, 0]
        ys = current_pose[:, 2] # Data Z -> Plot Y (Depth)
        zs = current_pose[:, 1] # Data Y -> Plot Z (Height)
        
        # 更新关节连接
        def update_lines(chain, lines):
            for i, link in enumerate(chain):
                # link[0] is start index, link[1] is end index
                x_line = [xs[link[0]], xs[link[1]]]
                y_line = [ys[link[0]], ys[link[1]]] # Depth
                z_line = [zs[link[0]], zs[link[1]]] # Height
                
                lines[i].set_data(x_line, y_line)
                lines[i].set_3d_properties(z_line)

        update_lines(LEFT_CHAIN, lines_left)
        update_lines(RIGHT_CHAIN, lines_right)
        update_lines(CENTER_CHAIN, lines_center)
        
        # 更新关节散点
        scats._offsets3d = (xs, ys, zs)

        # 更新坐标轴相机 (跟随 Root)
        # Root 在 Plot 中的坐标是 (root[0], root[2], root[1])
        cx, cy, cz = root[0], root[2], root[1]
        
        ax.set_xlim(cx - radius, cx + radius)
        ax.set_ylim(cy - radius, cy + radius)
        ax.set_zlim(0, cz + radius + 0.5) # 地面到头顶上方

        # 更新标题
        ax.set_title(f"HumanML3D View (Y-Up)\nFrame: {frame_idx}\nGreen Arrow = UP (Y), Blue Arrow = Forward (Z)")

    # 设置轴标签 (注意这里是指 Matplotlib 的轴，虽然我们已经交换了数据)
    ax.set_xlabel('X (Side)')
    ax.set_ylabel('Z (Forward/Depth)') 
    ax.set_zlabel('Y (Up/Height)')
    
    # 绘制地面网格 (在 Plot Z=0 处)
    # 我们简单画一个随动的网格没什么必要，Matplotlib 自带 grid 够用了
    
    # 调整初始视角：稍微俯视，看清前进方向
    ax.view_init(elev=20, azim=-45) 

    ani = FuncAnimation(fig, update, frames=frames_num, interval=1000/fps)
    
    print(f"Saving video to {save_path} ...")
    ani.save(save_path, writer='ffmpeg', fps=fps) # 如果没有ffmpeg请改为 'pillow'
    print("Done!")
    plt.close()
# ----------------- 使用部分 -----------------
if __name__ == "__main__":
    # 1. 这里的路径换成你上一步生成的那个 .npy 文件路径
    npy_path = "/root/autodl-tmp/MyRepository/MCM-LDM/datasets/humanml3d/new_joints/000011.npy" 
    
    # 自动搜索刚才生成的 .npy (为了方便你直接运行)
    if not os.path.exists(npy_path):
        import glob
        files = glob.glob("./output_joints/*.npy")
        if len(files) > 0:
            npy_path = files[0]
            print(f"Found file: {npy_path}")
        else:
            print("Error: No .npy file found. Please run the previous conversion script first.")
            exit()

    # 2. 加载数据
    data = np.load(npy_path)
    # 确保是 (T, 22, 3)
    if data.shape[1] != 22:
        print(f"Warning: Joint count is {data.shape[1]}, expected 22. Visualization might look weird.")

    # 3. 渲染视频
    output_video_path = npy_path.replace('.npy', '_vis.mp4')
    render_video(data, output_video_path, fps=20, downsample=1)