import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------
# 1. 核心函数：计算重心坐标 (2D 平面)
# ----------------------------------------
def get_barycentric_coords(P, PA, PB, PC):
    """
    计算点 P 相对于三角形 (PA, PB, PC) 的重心坐标 (alpha, beta, gamma)。
    使用向量叉乘（2D 推广的代数方法）来计算面积。
    """
    v0 = PC - PA
    v1 = PB - PA
    v2 = P - PA

    # 2D 向量的"叉积"（本质是 3D 叉积的 Z 分量，用于计算平行四边形面积）
    # det(A) = ad - bc for A = [[a, c], [b, d]]
    # det = v0[0]*v1[1] - v0[1]*v1[0]
    
    # D00 = v0 dot v0 (v0^2)
    d00 = np.dot(v0, v0)
    # D01 = v0 dot v1 (v0 . v1)
    d01 = np.dot(v0, v1)
    # D11 = v1 dot v1 (v1^2)
    d11 = np.dot(v1, v1)
    # D20 = v2 dot v0 (v2 . v0)
    d20 = np.dot(v2, v0)
    # D21 = v2 dot v1 (v2 . v1)
    d21 = np.dot(v2, v1)

    # 分母（整个三角形面积的两次方）
    denom = d00 * d11 - d01 * d01
    
    if denom == 0:
        return -1, -1, -1 # 退化三角形

    # 解线性方程组 (基于 Cramer's Rule 或预解的代数方法)
    # Note: 这里 alpha, beta, gamma 是标准的 (1-beta-gamma, beta, gamma) 形式
    gamma = (d11 * d20 - d01 * d21) / denom
    beta = (d00 * d21 - d01 * d20) / denom
    alpha = 1.0 - beta - gamma

    return alpha, beta, gamma

# ----------------------------------------
# 2. 定义三角形和颜色
# ----------------------------------------

# 三角形顶点坐标
PA = np.array([1, 1])
PB = np.array([6, 2])
PC = np.array([2, 5])

# 给每个顶点分配一个 RGB 颜色 (0到1)
ColorA = np.array([1.0, 0.0, 0.0]) # 红色
ColorB = np.array([0.0, 1.0, 0.0]) # 绿色
ColorC = np.array([0.0, 0.0, 1.0]) # 蓝色

# ----------------------------------------
# 3. 渲染插值
# ----------------------------------------

# 定义画布范围和分辨率
x_min, x_max = 0, 7
y_min, y_max = 0, 7
resolution = 100 
X, Y = np.meshgrid(np.linspace(x_min, x_max, resolution), 
                   np.linspace(y_min, y_max, resolution))
Image_data = np.zeros((resolution, resolution, 3)) # RGB图像数据

for i in range(resolution):
    for j in range(resolution):
        # 当前像素点坐标
        P = np.array([X[i, j], Y[i, j]])
        
        # 计算重心坐标
        alpha, beta, gamma = get_barycentric_coords(P, PA, PB, PC)
        
        # 判断点是否在三角形内 (alpha, beta, gamma 均非负且和为1)
        if alpha >= -1e-4 and beta >= -1e-4 and gamma >= -1e-4: # 允许微小浮点误差
            # 插值颜色
            interpolated_color = (alpha * ColorA + 
                                  beta * ColorB + 
                                  gamma * ColorC)
            Image_data[i, j, :] = np.clip(interpolated_color, 0.0, 1.0)
        else:
            # 外部区域设为背景色 (黑色)
            Image_data[i, j, :] = [0.0, 0.0, 0.0]

# ----------------------------------------
# 4. 可视化和保存
# ----------------------------------------

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(np.flipud(Image_data), extent=[x_min, x_max, y_min, y_max])
ax.set_title("Barycentric Coordinate Color Interpolation", fontsize=14, fontweight='bold')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')

# 绘制三角形边界和顶点
triangle_coords = np.vstack([PA, PB, PC, PA])
ax.plot(triangle_coords[:, 0], triangle_coords[:, 1], 'w-', linewidth=2)
ax.scatter(PA[0], PA[1], color='red', s=100, label='PA (Red)')
ax.scatter(PB[0], PB[1], color='green', s=100, label='PB (Green)')
ax.scatter(PC[0], PC[1], color='blue', s=100, label='PC (Blue)')

ax.legend()
plt.savefig("/root/autodl-tmp/MyRepository/MCM-LDM/LearnLinearAlgebra/barycentric_color_interpolation.png", dpi=300, bbox_inches='tight')
# plt.show()