import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----------------------------------------
# 0. 向量定义和运算
# ----------------------------------------
# 定义两个 2D 向量
v = np.array([3, 1])
w = np.array([-1, 2])

# 定义运算
v_plus_w = v + w
v_minus_w = v - w  # 重新添加减法
v_scaled = 2 * v
neg_w = -w         # 明确定义 -w 用于减法可视化

# ----------------------------------------
# 1. 向量运算的可视化 (优化版)
# ----------------------------------------

# 设置绘图环境
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_title("2D Vector Operations Visualization", fontsize=14, fontweight='bold')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')

ax.grid(linestyle=':', alpha=0.6) # 使用更细致的网格线
ax.set_aspect('equal', adjustable='box') # 确保 x/y 轴比例一致

# 设置坐标轴范围
all_coords = np.concatenate([v, w, v_plus_w, v_minus_w, v_scaled, neg_w])
max_coord = np.max(np.abs(all_coords)) + 1
ax.set_xlim([-max_coord, max_coord])
ax.set_ylim([-max_coord, max_coord])
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)

# ----------------------------------------
# 2. 绘制向量
# ----------------------------------------

# 绘制原始向量 v (蓝色实线，较粗)
ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
          color='blue', width=0.015, headwidth=5, headlength=7, label='v = [3, 1]')

# 绘制原始向量 w (绿色实线，较粗)
ax.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1,
          color='green', width=0.015, headwidth=5, headlength=7, label='w = [-1, 2]')

# 绘制向量加法 v + w (红色实线，最粗)
ax.quiver(0, 0, v_plus_w[0], v_plus_w[1], angles='xy', scale_units='xy', scale=1,
          color='red', width=0.02, headwidth=6, headlength=8, label='v + w (Sum)')

# 平行四边形辅助线 (从 v 终点到 v+w，浅红色虚线)
ax.quiver(v[0], v[1], w[0], w[1], angles='xy', scale_units='xy', scale=1,
          color='lightcoral', linestyle='dashed', width=0.008, headwidth=0, headlength=0, alpha=0.7)
# 平行四边形辅助线 (从 w 终点到 v+w，浅红色虚线)
ax.quiver(w[0], w[1], v[0], v[1], angles='xy', scale_units='xy', scale=1,
          color='lightcoral', linestyle='dashed', width=0.008, headwidth=0, headlength=0, alpha=0.7)


# 绘制向量数乘 2v (紫色实线，较粗，点线样式)
ax.quiver(0, 0, v_scaled[0], v_scaled[1], angles='xy', scale_units='xy', scale=1,
          color='purple', width=0.015, headwidth=5, headlength=7, linestyle='dotted', label='2v (Scaled)')

# 绘制向量减法 v - w
# 先绘制 -w (橙色虚线，轻微透明)
ax.quiver(0, 0, neg_w[0], neg_w[1], angles='xy', scale_units='xy', scale=1,
          color='orange', width=0.01, headwidth=4, headlength=6, linestyle='dotted', alpha=0.6, label='-w')
# 再绘制 v - w (深蓝色实线，较粗)
ax.quiver(0, 0, v_minus_w[0], v_minus_w[1], angles='xy', scale_units='xy', scale=1,
          color='darkblue', width=0.015, headwidth=5, headlength=7, label='v - w (Difference)')
# 辅助线 (从 w 的终点指向 v 的终点，浅蓝色虚线)
ax.quiver(w[0], w[1], v_minus_w[0] - w[0], v_minus_w[1] - w[1],
          angles='xy', scale_units='xy', scale=1, color='lightskyblue', linestyle='dashed',
          width=0.008, headwidth=0, headlength=0, alpha=0.7)


# ----------------------------------------
# 3. 手动创建图例对象 (用于绕过 quiver 的 bug)
# ----------------------------------------

# 创建代表每个向量/运算的 Line2D 代理对象 (英文标签)
legend_elements = [
    Line2D([0], [0], color='blue', lw=3, label='v = [3, 1]'),
    Line2D([0], [0], color='green', lw=3, label='w = [-1, 2]'),
    Line2D([0], [0], color='red', lw=3, label='v + w (Sum)'),
    Line2D([0], [0], color='purple', lw=3, linestyle='dotted', label='2v (Scaled)'),
    Line2D([0], [0], color='orange', lw=3, linestyle='dotted', label='-w'),
    Line2D([0], [0], color='darkblue', lw=3, label='v - w (Difference)')
]

# 使用 handles 和 labels 参数调用 legend
leg = ax.legend(handles=legend_elements, loc='upper left', frameon=True, fancybox=True, shadow=True, borderpad=1)

# 之前用于修复 ValueError 的代码：确保图例中的线条使用稳定的实线样式
# 这里我们对图例中的 Line2D 代理设置样式，所以它们可以有不同的 linestyle
# 但为了图例的一致性，通常这里还是保持实线，因为 Line2D 的虚线显示比 quiver 稳定
# 为了更好地反映实际线条样式，我们可以根据 legend_elements 重新设置
for i, line in enumerate(leg.get_lines()):
    if 'dotted' in legend_elements[i].get_linestyle():
        line.set_linestyle('dotted')
    elif 'dashed' in legend_elements[i].get_linestyle():
        line.set_linestyle('dashed')
    else:
        line.set_linestyle('-')


# ----------------------------------------
# 4. 最后的保存和显示
# ----------------------------------------
plt.savefig("/root/autodl-tmp/MyRepository/MCM-LDM/LearnLinearAlgebra/vector_operations_visualization.png", dpi=300, bbox_inches='tight') # 提高分辨率并裁剪空白
plt.show()