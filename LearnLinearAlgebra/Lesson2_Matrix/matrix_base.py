import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------
# 1. 定义变换矩阵
# ----------------------------------------

# 1.1. 旋转 90 度矩阵 (将 i 映射到 [0, 1]，将 j 映射到 [-1, 0])
R_90 = np.array([
    [0, -1],
    [1,  0]
])

# 1.2. 剪切/切变矩阵 (Shear, 将 i 保持不变，将 j 推向右上方)
# [1, 0] -> [1, 0]
# [0, 1] -> [1, 1]
S = np.array([
    [1, 1],
    [0, 1]
])

# 1.3. 复合变换：先剪切，再旋转
C = R_90 @ S  # 矩阵乘法，注意顺序：先作用 S，后作用 R_90

# ----------------------------------------
# 2. 定义绘图元素 (基向量和要变换的向量)
# ----------------------------------------

# 标准基向量
i = np.array([1, 0])
j = np.array([0, 1])
# 一个目标向量
v = np.array([2, 1])

# ----------------------------------------
# 3. 变换操作
# ----------------------------------------

# 变换向量 v
v_rotated = R_90 @ v
v_compounded = C @ v

# ----------------------------------------
# 4. 可视化
# ----------------------------------------

def plot_transformation(ax, i_vec, j_vec, v_vec, title):
    """可视化一个 2D 线性变换"""
    ax.set_title(title, fontsize=12)
    ax.grid(linestyle=':', alpha=0.6)
    ax.set_aspect('equal', adjustable='box')
    
    max_c = max(np.max(np.abs([i_vec, j_vec, v_vec])), 3)
    ax.set_xlim([-max_c, max_c])
    ax.set_ylim([-max_c, max_c])
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)

    # 绘制变换后的基向量 (矩阵的列)
    ax.quiver(0, 0, i_vec[0], i_vec[1], angles='xy', scale_units='xy', scale=1, 
              color='red', width=0.015, label='i\' (New Basis)')
    ax.quiver(0, 0, j_vec[0], j_vec[1], angles='xy', scale_units='xy', scale=1, 
              color='green', width=0.015, label='j\' (New Basis)')
    
    # 绘制变换后的目标向量
    ax.quiver(0, 0, v_vec[0], v_vec[1], angles='xy', scale_units='xy', scale=1, 
              color='blue', width=0.02, headwidth=6, label='v\' (Transformed)')
    
    ax.legend(loc='lower right')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# ---- 4.1 原始空间 ----
plot_transformation(axes[0], i, j, v, "Original Space (Identity Matrix)")

# ---- 4.2 旋转变换 ----
# 变换后的基向量是 R_90 的列
i_r = R_90[:, 0] 
j_r = R_90[:, 1]
plot_transformation(axes[1], i_r, j_r, v_rotated, "1. Rotation (R_90)")

# ---- 4.3 复合变换 ----
# 变换后的基向量是 C 的列
i_c = C[:, 0]
j_c = C[:, 1]
plot_transformation(axes[2], i_c, j_c, v_compounded, "2. Compound (S then R_90)")

plt.savefig("/root/autodl-tmp/MyRepository/MCM-LDM/LearnLinearAlgebra/Lesson2_Matrix/matrix_linear_transformation.png", dpi=300, bbox_inches='tight')
# plt.show()