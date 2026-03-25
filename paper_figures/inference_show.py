import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import splprep, splev
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec  # 引入 GridSpec 用于复杂排版

# 1. 告诉 matplotlib 你的字体文件在哪里
font_path = '/root/autodl-tmp/MyRepository/MCM-LDM/paper_figures/simhei.ttf'  
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)

# 2. 全局设置 matplotlib 使用这个中文字体
plt.rcParams['font.sans-serif'] = prop.get_name() 
# 3. 解决坐标轴负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

def create_method_diagrams():
    # 创建画布，调整 figsize 使其适合 2x2 的上下结构排版 (比如 16x12)
    fig = plt.figure(figsize=(16, 12), dpi=100)
    
    # 使用 GridSpec 定义 2 行 2 列的网格
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1])
    
    # ==========================================
    # 子图 (b): SDF 物理势能场可视化 (排在第一行左侧)
    # ==========================================
    ax2 = fig.add_subplot(gs[0, 0])
    
    # 定义网格范围
    grid_range = np.linspace(-2, 12, 100)
    X, Y = np.meshgrid(grid_range, grid_range)
    
    # 定义障碍物 1: 圆形 (Circle)
    c1 = np.array([3, 6])
    r1 = 1.8
    sdf_circle = np.sqrt((X - c1[0])**2 + (Y - c1[1])**2) - r1
    
    # 定义障碍物 2: 盒子 (Box)
    c2 = np.array([8, 4])
    s2 = np.array([1.5, 2.0]) 
    dx = np.abs(X - c2[0]) - s2[0]
    dy = np.abs(Y - c2[1]) - s2[1]
    outside = np.sqrt(np.maximum(dx, 0)**2 + np.maximum(dy, 0)**2)
    inside = np.minimum(np.maximum(dx, dy), 0)
    sdf_box = outside + inside
    
    # 场景总 SDF: 取两个障碍物的并集 (Min)
    sdf_total = np.minimum(sdf_circle, sdf_box)
    
    # 安全边界阈值
    delta = 0.5 
    
    # --- 绘制子图 (b) ---
    levels = np.linspace(-2, 5, 20)
    contour = ax2.contourf(X, Y, sdf_total, levels=levels, cmap='RdBu', alpha=0.7, extend='both')
    ax2.contour(X, Y, sdf_total, levels=[0], colors='black', linewidths=2.5)
    ax2.contour(X, Y, sdf_total, levels=[delta], colors='gold', linestyles='--', linewidths=2)
    
    # 标注
    ax2.text(c1[0], c1[1], "障碍物 1", ha='center', va='center', color='white', fontweight='bold')
    ax2.text(c2[0], c2[1], "障碍物 2", ha='center', va='center', color='white', fontweight='bold')
    ax2.text(1, 10, "安全区域\n(SDF > $\delta$)", color='navy', fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    fig.colorbar(contour, ax=ax2, label='Signed Distance 的值')
    ax2.set_title("(b) 基于SDF的物理作用场\n(红色: 排斥力区域, 蓝色: 安全区域)", fontsize=14, pad=15)
    ax2.set_aspect('equal')


    # ==========================================
    # 子图 (c): 推理阶段的梯度引导机制 (排在第一行右侧)
    # ==========================================
    ax3 = fig.add_subplot(gs[0, 1])
    
    # 模拟场景数据
    obs_center = np.array([5, 5])
    obs_radius = 2.0
    
    current_pos = np.array([6.2, 6.2]) 
    target_pos = np.array([8.0, 4.0])  
    
    # 1. 计算排斥力
    dir_to_obs = current_pos - obs_center
    dist_obs = np.linalg.norm(dir_to_obs)
    grad_sdf = dir_to_obs / dist_obs
    force_repulsion = grad_sdf * 1.8 
    
    # 2. 计算吸引力
    dir_to_target = target_pos - current_pos
    force_attraction = dir_to_target * 0.9 
    
    # 3. 合力
    force_total = force_repulsion + force_attraction
    next_pos = current_pos + force_total * 0.6
    
    # --- 绘制子图 (c) ---
    # 画障碍物
    c = patches.Circle(obs_center, obs_radius, color='#A93226', alpha=0.5, label='障碍物')
    ax3.add_patch(c)
    # 画安全边界
    c_safe = patches.Circle(obs_center, obs_radius+0.5, fill=False, edgecolor='#A93226', linestyle='--', label='安全边界 $\delta$')
    ax3.add_patch(c_safe)
    
    # 画位置点
    ax3.scatter(current_pos[0], current_pos[1], c='gray', s=180, label='噪声潜变量 $z_t$', zorder=6, edgecolors='black')
    ax3.scatter(target_pos[0], target_pos[1], c='#2E86C1', marker='X', s=180, label='目标路点 $P_{target}$', zorder=6, edgecolors='black')
    
    # 画力向量 (Arrows)
    # (1) 轨迹吸引力 (绿色)
    ax3.arrow(current_pos[0], current_pos[1], force_attraction[0], force_attraction[1], 
              head_width=0.3, head_length=0.4, fc='green', ec='green', width=0.05, length_includes_head=True)
    ax3.text(current_pos[0]+0.8, current_pos[1]-0.5, "$\mathcal{L}_{traj}$ (吸引力)", color='green', fontsize=11, fontweight='bold')

    # (2) 障碍物排斥力 (红色)
    ax3.arrow(current_pos[0], current_pos[1], force_repulsion[0], force_repulsion[1], 
              head_width=0.3, head_length=0.4, fc='red', ec='red', width=0.05, length_includes_head=True)
    ax3.text(current_pos[0]+0.3, current_pos[1]+1.2, "$\mathcal{L}_{obs}$ (排斥力)", color='red', fontsize=11, fontweight='bold')

    # (3) 合力/更新方向 (蓝色虚线箭头)
    ax3.annotate("", xy=next_pos, xytext=current_pos,
                 arrowprops=dict(arrowstyle="simple", color="#2E86C1", alpha=0.6, lw=1))
    ax3.text(next_pos[0]+0.2, next_pos[1], "更新方向\n$-\\nabla \mathcal{L}_{guide}$", color='#2E86C1', fontsize=12, fontweight='bold')

    ax3.set_xlim(2, 10)
    ax3.set_ylim(2, 9)
    ax3.set_title("(c) 梯度引导机制\n(对潜变量 $z_t$ 的受力分析)", fontsize=14, pad=15)
    ax3.legend(loc='lower left', framealpha=0.9)
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.set_aspect('equal')


    # ==========================================
    # 子图 (a): 轨迹预处理流水线 (排在第二行，跨两列，长度变长)
    # ==========================================
    ax1 = fig.add_subplot(gs[1, :])
    
    # 1. 模拟用户输入的稀疏控制点
    user_points = np.array([[1, 2], [3, 6], [7, 5], [10, 8], [13, 3]])
    
    # 2. B-Spline 平滑
    tck, u = splprep(user_points.T, u=None, s=0.0, k=3) 
    u_fine = np.linspace(u.min(), u.max(), 200)
    smooth_curve = np.array(splev(u_fine, tck)).T
    
    # 3. 弧长参数化重采样
    diffs = np.linalg.norm(smooth_curve[1:] - smooth_curve[:-1], axis=1)
    cum_dist = np.concatenate(([0], np.cumsum(diffs)))
    total_len = cum_dist[-1]
    
    num_frames = 15
    target_dists = np.linspace(0, total_len, num_frames)
    
    resampled_points = []
    for d in target_dists:
        rx = np.interp(d, cum_dist, smooth_curve[:, 0])
        ry = np.interp(d, cum_dist, smooth_curve[:, 1])
        resampled_points.append([rx, ry])
    resampled_points = np.array(resampled_points)

    # --- 绘制子图 (a) ---
    ax1.plot(user_points[:, 0], user_points[:, 1], 'o--', color='gray', alpha=0.5, label='用户输入 (稀疏)', markersize=8)
    ax1.plot(smooth_curve[:, 0], smooth_curve[:, 1], 'k-', linewidth=2.5, label='B样条曲线 $\mathcal{C}(u)$')
    ax1.scatter(resampled_points[:, 0], resampled_points[:, 1], c='#FF5733', s=120, zorder=5, label='重采样帧 $P_{target}(t)$', edgecolors='white', linewidth=1.5)
    
    # 标注
    ax1.text(user_points[0,0], user_points[0,1]-0.8, "起点", fontsize=12, fontweight='bold')
    ax1.text(resampled_points[7,0], resampled_points[7,1]-0.8, "对齐帧", fontsize=11, color='#FF5733', ha='center')
    
    ax1.set_title("(a) 轨迹处理管线\n(B样条平滑 & 弧长重采样)", fontsize=14, pad=15)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_aspect('equal')
    
    plt.tight_layout()
    
    # 保存为 SVG (方便 PPT 编辑) 和 PNG
    plt.savefig("method_diagram.png", dpi=300)
    plt.savefig("method_diagram.svg")
    print("图像已保存为 method_diagram.png 和 method_diagram.svg")
    plt.show()

# 运行函数
create_method_diagrams()