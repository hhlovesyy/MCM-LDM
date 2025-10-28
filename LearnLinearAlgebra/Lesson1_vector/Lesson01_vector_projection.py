import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D # 引入 Line2D 用于手动创建图例

# ----------------------------------------
# 1. 定义向量和计算投影
# ----------------------------------------

# 向量 b (要被投影的向量)
b = np.array([4, 4])
# 向量 a (投影的方向)
a = np.array([5, 1])

# 1. 计算投影向量 p
# 投影公式: p = (b . a / ||a||^2) * a
a_dot_b = np.dot(a, b)
a_sq_norm = np.dot(a, a) # 等价于 ||a||^2

# 投影向量 p
p = (a_dot_b / a_sq_norm) * a

# 2. 计算误差向量 e (误差向量 e 垂直于 a)
e = b - p

# 3. 验证正交性 (误差向量 e . 向量 a 应该接近于 0)
dot_product_test = np.dot(e, a)
print(f"投影向量 p: {p}")
print(f"误差向量 e: {e}")
print(f"正交性测试 (e . a): {dot_product_test:.2e} (接近于 0 即正交)")

# ----------------------------------------
# 2. 可视化投影
# ----------------------------------------

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_title("Vector Projection Visualization", fontsize=14, fontweight='bold')
ax.grid(linestyle=':', alpha=0.6)
ax.set_aspect('equal', adjustable='box')

# 确定坐标轴范围
all_coords = np.concatenate([a, b, p])
max_coord = np.max(np.abs(all_coords)) + 1
ax.set_xlim([-1, max_coord])
ax.set_ylim([-1, max_coord])
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)

# 绘制向量 a (投影方向) - 黑色虚线表示 a 的延直线
# 使用 plot 绘制直线
ax.plot([-max_coord * a[0], max_coord * a[0]], 
        [-max_coord * a[1], max_coord * a[1]], 
        'k--', alpha=0.3) # 注意：这里不加 label，让图例手动处理
ax.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', scale=1, 
          color='k', width=0.015, headwidth=5) # 注意：这里不加 label

# 绘制向量 b (被投影向量)
ax.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', scale=1, 
          color='blue', width=0.015, headwidth=5) # 注意：这里不加 label

# 绘制投影向量 p (红色粗线，落在 a 的直线上)
ax.quiver(0, 0, p[0], p[1], angles='xy', scale_units='xy', scale=1, 
          color='red', width=0.02, headwidth=6) # 注意：这里不加 label

# 绘制误差向量 e (绿色虚线，从 p 终点到 b 终点)
ax.quiver(p[0], p[1], e[0], e[1], angles='xy', scale_units='xy', scale=1, 
          color='green', width=0.01, linestyle='dashed', headwidth=5) # 注意：这里不加 label

# ----------------------------------------
# 3. 正交性标记 (直角符号)
# ----------------------------------------
# 绘制直角标记，表示 e 垂直于 a
# 找到从 p 沿着 a 和 e 方向的一小段距离来画标记
scale = 0.5 
corner_a = p + scale * a / np.linalg.norm(a) # 沿着 a 的方向
corner_e = p + scale * e / np.linalg.norm(e) # 沿着 e 的方向
corner_diag = p + scale * a / np.linalg.norm(a) + scale * e / np.linalg.norm(e) # 对角线

# 绘制两条构成直角的线段
ax.plot([corner_a[0], corner_diag[0]], [corner_a[1], corner_diag[1]], 'm--', alpha=0.7)
ax.plot([corner_e[0], corner_diag[0]], [corner_e[1], corner_diag[1]], 'm--', alpha=0.7)


# ----------------------------------------
# 4. 手动创建图例对象 (加固绘图)
# ----------------------------------------
legend_elements = [
    # 投影线
    Line2D([0], [0], color='k', linestyle='dashed', lw=1, alpha=0.5, label='Line of a'),
    # 向量 a
    Line2D([0], [0], color='k', lw=3, label='Vector a (Direction)'),
    # 向量 b
    Line2D([0], [0], color='blue', lw=3, label='Vector b (Original)'),
    # 投影向量 p
    Line2D([0], [0], color='red', lw=4, label='Projection p'),
    # 误差向量 e
    Line2D([0], [0], color='green', lw=3, linestyle='dashed', label='Error e (Orthogonal)')
]

leg = ax.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True)

# 确保虚线/点线在图例中可以显示 (尽管 Line2D 比 quiver 稳定)
for i, line in enumerate(leg.get_lines()):
    line.set_linestyle(legend_elements[i].get_linestyle())


# ----------------------------------------
# 5. 保存和显示
# ----------------------------------------
# 将保存路径改为更通用的相对路径，以提高代码的可移植性
plt.savefig("/root/autodl-tmp/MyRepository/MCM-LDM/LearnLinearAlgebra/vector_projection_visualization.png", dpi=300, bbox_inches='tight')
# plt.show()