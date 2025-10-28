import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# ----------------------------------------
# 辅助类: 3D 箭头绘制 (解决 mplot3d 的 quiver 局限性)
# ----------------------------------------
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    # 关键修改：添加 do_3d_projection 方法
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        
        # 使用 proj3d 计算投影后的 2D 坐标
        # self.axes.M 是当前 3D 坐标系的投影矩阵
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        
        # 更新 FancyArrowPatch 的 2D 位置
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        
        # 返回 z 坐标的均值用于正确的深度排序 (虽然不是强制的，但很关键)
        return np.min(zs)

    def draw(self, renderer):
        # 确保投影被执行 (在现代 Matplotlib 中，这一步可能不是必需的，但为了兼容性保留)
        self.do_3d_projection(renderer) 
        super().draw(renderer)

# ----------------------------------------
# 1. 核心函数: 构造变换矩阵
# ----------------------------------------

def translation_matrix(tx, ty, tz):
    """构造平移矩阵 (4x4)"""
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])

def scale_matrix(sx, sy, sz):
    """构造缩放矩阵 (4x4)"""
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])

def rotation_x_matrix(angle_rad):
    """绕 X 轴旋转矩阵 (4x4)"""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ])

# ----------------------------------------
# 2. 定义几何体 (立方体) 和变换
# ----------------------------------------

# 定义一个简易立方体（8个顶点，齐次坐标）
# 形状：(4, 8) -> (x, y, z, 1) x 8 points
cube_verts_4d = np.array([
    [1, 1, 1, 1], [-1, 1, 1, 1], [-1, -1, 1, 1], [1, -1, 1, 1],   # Z=1 面
    [1, 1, -1, 1], [-1, 1, -1, 1], [-1, -1, -1, 1], [1, -1, -1, 1] # Z=-1 面
]).T

# ----------------------------------------
# 3. 复合变换 (T * R * S)
# ----------------------------------------

# 1. 缩放 (S): 在 Y 轴压缩 0.5
M_S = scale_matrix(1, 0.5, 1)

# 2. 旋转 (R): 绕 X 轴旋转 30 度
M_R = rotation_x_matrix(np.radians(30))

# 3. 平移 (T): 沿 X 轴平移 2, Z 轴平移 1
M_T = translation_matrix(2, 0, 1)

# 复合矩阵：注意顺序，操作从右到左 (S -> R -> T)
M_final = M_T @ M_R @ M_S

# ----------------------------------------
# 4. 执行变换
# ----------------------------------------

# 初始顶点 (3D)
P_initial = cube_verts_4d[:3, :]
# 变换后的顶点
P_transformed_4d = M_final @ cube_verts_4d
P_transformed = P_transformed_4d[:3, :] # 移除齐次 w 分量

# ----------------------------------------
# 5. 可视化
# ----------------------------------------

fig = plt.figure(figsize=(12, 6))

# ---- 5.1 绘制初始状态 ----
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title("Initial Cube (Identity)", fontsize=14)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])
ax1.set_zlim([-3, 3])
# 绘制立方体 (仅绘制边界)
for i in range(8):
    for j in range(i + 1, 8):
        # 仅连接相邻顶点 (欧几里得距离为 2 且仅一个坐标不同)
        if np.sum(np.abs(P_initial[:, i] - P_initial[:, j]) > 1e-6) == 1 and np.sum(np.abs(P_initial[:, i] - P_initial[:, j])) == 2:
            ax1.plot(*zip(P_initial[:, i], P_initial[:, j]), color='k', linestyle='--', alpha=0.5)
ax1.view_init(elev=20, azim=45) # 设置初始视角

# ---- 5.2 绘制变换后状态 ----
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title("Transformed (Scale -> Rotate -> Translate)", fontsize=14)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
# 根据变换后的数据调整轴范围，以更好地展示结果
min_x, max_x = np.min(P_transformed[0, :]), np.max(P_transformed[0, :])
min_y, max_y = np.min(P_transformed[1, :]), np.max(P_transformed[1, :])
min_z, max_z = np.min(P_transformed[2, :]), np.max(P_transformed[2, :])
ax2.set_xlim([min_x - 1, max_x + 1])
ax2.set_ylim([min_y - 1, max_y + 1])
ax2.set_zlim([min_z - 1, max_z + 1])


# 绘制变换后的立方体
for i in range(8):
    for j in range(i + 1, 8):
        # 仅连接相邻顶点 (使用初始状态的连接逻辑)
        if np.sum(np.abs(P_initial[:, i] - P_initial[:, j]) > 1e-6) == 1 and np.sum(np.abs(P_initial[:, i] - P_initial[:, j])) == 2:
            ax2.plot(*zip(P_transformed[:, i], P_transformed[:, j]), color='blue', linewidth=2)

# 绘制平移箭头 (从原点到变换后的中心)
center_init = np.array([0, 0, 0])
center_final = np.mean(P_transformed, axis=1)

arrow = Arrow3D([center_init[0], center_final[0]], 
                [center_init[1], center_final[1]], 
                [center_init[2], center_final[2]], 
                mutation_scale=20, lw=2, arrowstyle="-|>", color="red")
ax2.add_artist(arrow)
# 稍微调整文本位置以避免与箭头头部重叠
ax2.text(center_final[0] + 0.2, center_final[1] + 0.2, center_final[2] + 0.2, 
         "Translation T", color='red', fontsize=10)


ax2.view_init(elev=20, azim=45) # 保持相同视角

plt.tight_layout()
# 使用通用文件名，避免绝对路径问题
plt.savefig("/root/autodl-tmp/MyRepository/MCM-LDM/LearnLinearAlgebra/Lesson2_Matrix/homogeneous_coordinates_transform.png", dpi=300, bbox_inches='tight')
# plt.show()