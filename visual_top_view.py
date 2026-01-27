import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def render_top_view(motion_data, save_path, gap_width=None, fps=20):
    """
    渲染俯视图 (XZ平面) 视频，适配“相对狭窄”引导。
    动态计算角色的朝向，并展示随身的宽度限制。
    """
    # [Frames, 22, 3]
    if motion_data.shape[0] == 22: 
        motion_data = motion_data.transpose(2, 0, 1)
    frames = motion_data.shape[0]
    
    kinematic_chain = [
        [0, 1, 4, 7, 10], # 左腿
        [0, 2, 5, 8, 11], # 右腿
        [0, 3, 6, 9, 12, 15], # 脊柱
        [9, 13, 16, 18, 20], # 左臂
        [9, 14, 17, 19, 21]  # 右臂
    ]

    # 画布设置
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def update(frame):
        ax.clear()
        pose = motion_data[frame] # [22, 3]
        
        # --- 1. 计算当前身体坐标系 ---
        root = pose[0]
        l_hip = pose[1]
        r_hip = pose[2]
        
        # 使用左右胯的连线作为“局部 X 轴” (指向右侧)
        vec_right = r_hip - l_hip
        # 只看 XZ 平面
        vec_right[1] = 0 
        # 归一化
        norm = np.linalg.norm(vec_right)
        if norm > 1e-6:
            vec_right = vec_right / norm
        else:
            vec_right = np.array([1.0, 0.0, 0.0]) # 默认向右

        # --- 2. 绘制设置 ---
        ax.set_title(f"Local Gap Check (Frame {frame})")
        ax.set_xlabel("X (World)")
        ax.set_ylabel("Z (World)")
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 相机跟随 Root，保持 4x4 米的视野
        root_x, root_z = root[0], root[2]
        ax.set_xlim(root_x - 2, root_x + 2)
        ax.set_ylim(root_z - 2, root_z + 2)
        
        # --- 3. 绘制随身缝隙 (Gap Limits) ---
        if gap_width is not None:
            hw = gap_width / 2.0
            
            # 计算左边界点和右边界点 (相对于 Root)
            # Right Limit = Root + vec_right * hw
            # Left Limit  = Root - vec_right * hw
            p_right = root + vec_right * hw
            p_left  = root - vec_right * hw
            
            # 绘制一条横跨身体的线，代表当前的通道宽度
            ax.plot([p_left[0], p_right[0]], [p_left[2], p_right[2]], 
                    color='red', linewidth=1, linestyle='--', label='Max Width')
            
            # 在左右边界画两个短竖线 (模拟墙壁切面)
            # 计算垂直于 right 的 forward 向量
            vec_fwd = np.array([-vec_right[2], 0, vec_right[0]])
            
            for p in [p_left, p_right]:
                w_start = p - vec_fwd * 0.5
                w_end   = p + vec_fwd * 0.5
                ax.plot([w_start[0], w_end[0]], [w_start[2], w_end[2]], 
                        color='gray', linewidth=3, alpha=0.5)

            ax.text(root_x, root_z + 1.8, f"Gap: {gap_width}m", ha='center', color='red')

        # --- 4. 绘制骨架 & 检测碰撞 ---
        for chain in kinematic_chain:
            xs = pose[chain, 0]
            zs = pose[chain, 2]
            ax.plot(xs, zs, color='#4D84AA', linewidth=2.0)
            
            # 绘制关节点 (带碰撞检测)
            for j_idx in chain:
                j_pos = pose[j_idx]
                
                # 碰撞检测逻辑：
                # 计算关节相对于 Root 的向量
                rel_pos = j_pos - root
                # 投影到局部 X 轴上 (点积)
                local_x = np.dot(rel_pos, vec_right)
                
                # 判断是否超宽
                is_colliding = False
                if gap_width is not None:
                    if abs(local_x) > gap_width / 2.0:
                        is_colliding = True
                
                col = 'red' if is_colliding else '#DD5A37'
                size = 40 if is_colliding else 20
                ax.scatter(j_pos[0], j_pos[2], color=col, s=size, zorder=5)

        # 标记手部 (L=20, R=21) 方便观察缩手效果
        l_hand = pose[20]
        r_hand = pose[21]
        ax.text(l_hand[0], l_hand[2], "L", fontsize=8, color='blue')
        ax.text(r_hand[0], r_hand[2], "R", fontsize=8, color='blue')

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50)
    try:
        writer = animation.FFMpegWriter(fps=fps)
        ani.save(save_path, writer=writer)
        print(f"  -> Top-view debug video saved: {save_path}")
    except Exception as e:
        print(f"Error: {e}")
    plt.close()