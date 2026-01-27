import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def render_front_view(motion_data, save_path, gap_width=None, fps=20):
    """
    渲染【第三人称追尾视角】(局部 XY 平面)。
    相机固定在角色背后，观察左右两侧的缝隙情况。
    """
    # [Frames, 22, 3]
    if motion_data.shape[0] == 22: 
        motion_data = motion_data.transpose(2, 0, 1)
    frames = motion_data.shape[0]
    
    kinematic_chain = [
        [0, 1, 4, 7, 10], # Left Leg
        [0, 2, 5, 8, 11], # Right Leg
        [0, 3, 6, 9, 12, 15], # Spine
        [9, 13, 16, 18, 20], # Left Arm
        [9, 14, 17, 19, 21]  # Right Arm
    ]

    # 画布：正方形，看截面
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def update(frame):
        ax.clear()
        pose = motion_data[frame] # [22, 3] World Coords
        
        # --- 1. 计算局部坐标系 (把世界坐标转为以 Root 为中心的局部坐标) ---
        root = pose[0]
        l_hip = pose[1]
        r_hip = pose[2]
        
        # X轴：左右 (Right Vector)
        vec_right = r_hip - l_hip
        vec_right[1] = 0
        norm = np.linalg.norm(vec_right)
        if norm > 1e-6: vec_right /= norm
        else: vec_right = np.array([1.0, 0.0, 0.0])
        
        # Y轴：垂直向上
        vec_up = np.array([0.0, 1.0, 0.0])
        
        # Z轴：前进方向 (Forward)
        vec_fwd = np.cross(vec_right, vec_up)
        
        # 构建旋转矩阵 [3, 3] (World -> Local)
        # 既然我们是从背后看，我们希望投影到 (Local X, Local Y) 平面
        
        # --- 2. 绘制设置 ---
        ax.set_title(f"Front/Back View (Frame {frame})")
        ax.set_xlabel("Local X (Left <-> Right)")
        ax.set_ylabel("Local Y (Height)")
        ax.set_xlim(-1.0, 1.0) # 左右各 1 米视野
        ax.set_ylim(0, 2.0)    # 高度 0-2 米
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # --- 3. 绘制墙壁 ---
        if gap_width is not None:
            hw = gap_width / 2.0
            # 左墙 (红色区域)
            ax.axvline(x=-hw, color='black', linewidth=3)
            ax.fill_betweenx([0, 2.5], -1.0, -hw, color='red', alpha=0.1, label='Wall')
            # 右墙
            ax.axvline(x=hw, color='black', linewidth=3)
            ax.fill_betweenx([0, 2.5], hw, 1.0, color='red', alpha=0.1)
            
            ax.text(0, 1.9, f"Gap: {gap_width}m", ha='center', color='red', fontweight='bold')

        # --- 4. 转换并绘制骨架 ---
        for chain in kinematic_chain:
            local_xs = []
            local_ys = []
            
            for j_idx in chain:
                j_pos_world = pose[j_idx]
                rel_pos = j_pos_world - root
                
                # 投影到局部轴
                lx = np.dot(rel_pos, vec_right)
                ly = j_pos_world[1] # 高度直接用世界高度即可 (假设地面平整)
                
                local_xs.append(lx)
                local_ys.append(ly)
                
                # 碰撞检测 (标红)
                is_hit = False
                if gap_width and abs(lx) > hw:
                    is_hit = True
                
                col = 'red' if is_hit else '#DD5A37'
                size = 40 if is_hit else 20
                ax.scatter(lx, ly, color=col, s=size, zorder=5)
            
            ax.plot(local_xs, local_ys, color='#4D84AA', linewidth=2.0)

        # 标记左右手
        # 获取手腕的局部坐标
        l_hand_world = pose[20]
        r_hand_world = pose[21]
        
        lx_l = np.dot(l_hand_world - root, vec_right)
        lx_r = np.dot(r_hand_world - root, vec_right)
        
        ax.text(lx_l, l_hand_world[1], "L", color='blue', fontsize=10, fontweight='bold')
        ax.text(lx_r, r_hand_world[1], "R", color='blue', fontsize=10, fontweight='bold')

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50)
    try:
        writer = animation.FFMpegWriter(fps=fps)
        ani.save(save_path, writer=writer)
        print(f"  -> Front-view debug video saved: {save_path}")
    except Exception as e:
        print(f"Error: {e}")
    plt.close()