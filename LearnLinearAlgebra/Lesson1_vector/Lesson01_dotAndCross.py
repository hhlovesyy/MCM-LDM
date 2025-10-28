import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# ----------------------------------------
# 1. 定义向量和计算
# ----------------------------------------

# 定义两个 3D 向量
v = np.array([3, 0, 1])
w = np.array([0, 2, 2])

# 计算点积 (Dot Product)
dot_product = np.dot(v, w)
# 计算叉积 (Cross Product)
cross_product = np.cross(v, w)

# ----------------------------------------
# 2. 计算几何辅助量
# ----------------------------------------
# 计算夹角 (用于验证点积)
magnitude_v = np.linalg.norm(v)
magnitude_w = np.linalg.norm(w)
cos_theta = dot_product / (magnitude_v * magnitude_w)
angle_degrees = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))) # clip 避免浮点误差

# ----------------------------------------
# 3. 3D 可视化
# ----------------------------------------

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f"3D Vector Operations: Dot Product & Cross Product\nDot Product: {dot_product:.2f} (Angle: {angle_degrees:.1f}°)", fontsize=14, fontweight='bold')

# 设置坐标轴范围
max_val = max(np.max(np.abs(v)), np.max(np.abs(w)), np.max(np.abs(cross_product))) + 1
ax.set_xlim([-max_val, max_val])
ax.set_ylim([-max_val, max_val])
ax.set_zlim([-max_val, max_val])

# 设置坐标轴标签
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

# 绘制原点
ax.scatter([0], [0], [0], color='k', marker='o', s=50)

# 绘制向量 v (蓝色)
ax.quiver(0, 0, 0, v[0], v[1], v[2], color='blue', arrow_length_ratio=0.1, linewidth=2, label='v')
ax.text(v[0], v[1], v[2], f'v({v[0]}, {v[1]}, {v[2]})', color='blue')

# 绘制向量 w (绿色)
ax.quiver(0, 0, 0, w[0], w[1], w[2], color='green', arrow_length_ratio=0.1, linewidth=2, label='w')
ax.text(w[0], w[1], w[2], f'w({w[0]}, {w[1]}, {w[2]})', color='green')

# 绘制叉积向量 v x w (红色，垂直于 v 和 w 所在的平面)
ax.quiver(0, 0, 0, cross_product[0], cross_product[1], cross_product[2],
          color='red', arrow_length_ratio=0.1, linewidth=3, label='v x w (Cross Product)')
ax.text(cross_product[0], cross_product[1], cross_product[2], 'v x w (Normal)', color='red', fontweight='bold')


# ----------------------------------------
# 4. 辅助线：张成的平行四边形 (仅用于可视化叉积的平面)
# ----------------------------------------
# 绘制平行四边形的顶点
verts = np.array([
    [0, 0, 0],
    v,
    v + w,
    w,
    [0, 0, 0] # 闭合
])
# 绘制平行四边形的边界
ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], color='gray', linestyle='dashed', alpha=0.5)

ax.legend()
plt.savefig("/root/autodl-tmp/MyRepository/MCM-LDM/LearnLinearAlgebra/dot_cross_product_visualization.png", dpi=300, bbox_inches='tight')
# plt.show() # 如果在服务器上运行，请注释掉 show()