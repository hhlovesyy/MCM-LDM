import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from tqdm import tqdm

# 尝试导入 BodyModel，如果你的环境没有配置好 human_body_prior，这一步会报错
try:
    from human_body_prior.body_model.body_model import BodyModel
    from human_body_prior.tools.omni_tools import copy2cpu as c2c
except ImportError:
    print("Error: 请确保安装了 human_body_prior 库 (https://github.com/nghorbani/human_body_prior)")
    exit()

# ---------------- 配置区域 ----------------
# 请修改这里的路径为你本地的路径
MALE_BM_PATH = '/root/autodl-tmp/MyRepository/MCM-LDM/body_models/smplh/male/model.npz'
FEMALE_BM_PATH = '/root/autodl-tmp/MyRepository/MCM-LDM/body_models/smplh/female/model.npz'
MALE_DMPL_PATH = '/root/autodl-tmp/MyRepository/MCM-LDM/body_models/dmpls/male/model.npz'
FEMALE_DMPL_PATH = '/root/autodl-tmp/MyRepository/MCM-LDM/body_models/dmpls/female/model.npz'

TARGET_FPS = 20
num_dmpls = 8 # number of DMPL parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 22个关节的连接关系 (用于可视化连线)
# SMPL 22 joints indices usually follow kinematic tree
# 这是一个简化的连接示意，用于画火柴人
KINEMATIC_CHAIN = [
    [0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9],
    [7, 10], [8, 11], [9, 12], [9, 13], [9, 14], [12, 15], [13, 16], [14, 17],
    [16, 18], [17, 19], [18, 20], [19, 21]
]

def load_body_models():
    """加载 SMPL-H 模型"""
    print(f"Loading Body Models on {DEVICE}...")
    male_bm = BodyModel(bm_fname=MALE_BM_PATH, num_betas=10, num_dmpls=8, dmpl_fname=MALE_DMPL_PATH).to(DEVICE)
    female_bm = BodyModel(bm_fname=FEMALE_BM_PATH, num_betas=10, num_dmpls=8, dmpl_fname=FEMALE_DMPL_PATH).to(DEVICE)
    return male_bm, female_bm

def visualize_pose(pose_data, save_path, frame_idx=0):
    """
    可视化某一帧的骨架，自动调整视角范围以包含整个人物
    Input: pose_data (T, 22, 3)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 获取特定帧的数据
    # frame_pose shape: (22, 3)
    frame_pose = pose_data[frame_idx].copy() 
    
    # --- 【关键修改 1】: 打印数据范围进行调试 ---
    print(f"DEBUG: Frame {frame_idx} Range:")
    print(f"  X: {frame_pose[:, 0].min():.2f} to {frame_pose[:, 0].max():.2f}")
    print(f"  Y: {frame_pose[:, 1].min():.2f} to {frame_pose[:, 1].max():.2f}")
    print(f"  Z: {frame_pose[:, 2].min():.2f} to {frame_pose[:, 2].max():.2f}")

    # --- 【关键修改 2】: 临时将这一帧的数据归一化到原点显示 ---
    # 找到这一帧所有关节的中心点 (或者直接用根节点 frame_pose[0])
    center = frame_pose.mean(axis=0) 
    frame_pose = frame_pose - center  # 把人挪到 (0,0,0)
    
    # 绘制关节点
    ax.scatter(frame_pose[:, 0], frame_pose[:, 1], frame_pose[:, 2], c='r', marker='o', s=20)
    
    # 绘制骨骼连线
    # 确保 KINEMATIC_CHAIN 在全局变量里定义了
    for chain in KINEMATIC_CHAIN:
        if chain[0] < frame_pose.shape[0] and chain[1] < frame_pose.shape[0]:
            x_lines = [frame_pose[chain[0], 0], frame_pose[chain[1], 0]]
            y_lines = [frame_pose[chain[0], 1], frame_pose[chain[1], 1]]
            z_lines = [frame_pose[chain[0], 2], frame_pose[chain[1], 2]]
            ax.plot(x_lines, y_lines, z_lines, c='b')

    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # --- 【关键修改 3】: 动态设置坐标轴范围 ---
    # 因为已经把人挪到原点了，现在可以用固定的范围 (比如 -1 到 1)
    # 或者根据人的实际大小动态设定
    radius = 1.2 # 半径1.2米通常足够容纳一个人
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(-radius, radius)
    
    # 调整视角 (Elev, Azim) 让你看得更清楚
    ax.view_init(elev=10, azim=45) 

    plt.title(f"Visualized Frame {frame_idx} (Centered)")
    plt.savefig(save_path)
    plt.close()
    print(f"Visualization saved to {save_path}")

def process_single_file(npz_path, output_dir, male_bm, female_bm):
    """处理单个 .npz 文件"""
    filename = os.path.basename(npz_path).replace('.npz', '')
    
    # 1. 加载数据
    try:
        bdata = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"Failed to load {npz_path}: {e}")
        return

    # 2. 检查基本信息
    try:
        fps = bdata['mocap_framerate']  # 120
        gender = bdata['gender'] # female
        # print(f"Processing {filename}: Gender={gender}, FPS={fps}")
    except KeyError:
        print(f"Skipping {filename}: Missing metadata")
        return

    # 3. 降采样
    down_sample = int(fps / TARGET_FPS) # 6
    if down_sample < 1: down_sample = 1
    
    # 提取 SMPL 参数
    # AMASS 数据格式通常为: poses (T, 156), trans (T, 3), betas (16)
    # 我们只取需要的帧
    poses = bdata['poses'][::down_sample]  # shape:(333, 156) # 156 / 3 = 52 个关节,表示关节旋转的方法：（1）欧拉角（X） （2）四元数  （3）轴角表示法（√） （4）6D轴角表示法
    trans = bdata['trans'][::down_sample]  # 从头到尾，每隔 down_sample 个元素取一个
    num_frames = poses.shape[0] # 333
    
    # 4. 准备 Tensor 输入
    bm = male_bm if gender == 'male' else female_bm
    
    body_parms = {
        'root_orient': torch.Tensor(poses[:, :3]).to(DEVICE), # torch.Size([333, 3]) 在 AMASS 数据集和 SMPL 模型中，它的数学表示通常不是欧拉角，而是 轴角（Axis-Angle）。
        'pose_body': torch.Tensor(poses[:, 3:66]).to(DEVICE), # 只取身体部分 torch.Size([333, 63])
        'pose_hand': torch.Tensor(poses[:, 66:]).to(DEVICE),  # 手部参数 torch.Size([333, 90])
        'trans': torch.Tensor(trans).to(DEVICE), # torch.Size([333, 3]) trans 表示的是 SMPL 模型根关节点（Root Joint，通常位于盆骨位置）在全局坐标系中的绝对位置。
        # betas 需要重复扩充到每一帧
        'betas': torch.Tensor(np.repeat(bdata['betas'][:10][np.newaxis], repeats=num_frames, axis=0)).to(DEVICE), # torch.Size([333, 10])
    } # [np.newaxis]：这是一个增加维度的技巧。它把形状从 (10,) 变成了 (1, 10)。

    # 5. Forward Pass (SMPL Layer) -> 获得 3D 关节坐标
    with torch.no_grad():
        body = bm(**body_parms)
        # body.Jtr 包含了所有关节的位置 (T, 52, 3) 
        # HumanML3D 通常使用 SMPL 的前 22 个关节
        pose_seq_np = body.Jtr.detach().cpu().numpy()[:, :22, :] # (333, 22, 3)

    # 6. 坐标系转换 (根据原脚本逻辑)
    # 原脚本：np.dot(pose, trans_matrix)
    # trans_matrix = [[1, 0, 0], [0, 0, 1], [0, 1, 0]] ->  x=x, y=z, z=y (Y-up to Z-up swap? or vice versa)
    # 通常 AMASS 是 Z-up, HumanML3D 需要 Y-up
    # 原矩阵效果：New X = Old X, New Y = Old Z, New Z = Old Y
    # 让我们明确一下：通常图形学中 Y 是朝上的。
    trans_matrix = np.array([[1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0],
                             [0.0, 1.0, 0.0]])
    pose_seq_np_n = np.dot(pose_seq_np, trans_matrix) # 这个是矩阵右乘，矩阵在右侧，理论上YZ轴交换.仅仅交换 $Y$ 和 $Z$ 可能会导致手性（Chirality）改变
    
    # 7. 减去中心位移 (可选，HumanML3D 后续处理会做，但这里可以先不做)
    # pose_seq_np_n -= pose_seq_np_n[0:1, 0:1, :] * np.array([1, 0, 1]) # Remove initial position on floor?
    
    # 8. 保存数据
    save_path = os.path.join(output_dir, filename + '.npy')
    np.save(save_path, pose_seq_np_n)
    
    # 9. 额外的详细信息打印和可视化 (针对第一个处理的文件)
    return pose_seq_np_n

def main():
    # 模拟命令行参数，你可以直接改这里
    input_folder = "/root/autodl-tmp/MyRepository/MCM-LDM/amass_sample/ACCAD" # 你的解压后的 AMASS 子数据集文件夹
    output_folder = "/root/autodl-tmp/MyRepository/MCM-LDM/amass_sample/output_joints"
    
    os.makedirs(output_folder, exist_ok=True)
    
    # 加载模型
    male_bm, female_bm = load_body_models()
    
    # 查找文件
    files = [
        os.path.join(root, f) 
        for root, dirs, filenames in os.walk(input_folder) 
        for f in filenames 
        if f.endswith('.npz') # npz是字典，npy是数组
    ]
    print(f"Found {len(files)} .npz files in {input_folder}") # files这个list里面是所有AMASS数据集里面的npz文件的绝对路径
    
    if len(files) == 0:
        print("No files found. Please check input path.")
        return

    # 批处理
    first_processed_data = None
    first_filename = ""
    
    for f in tqdm(files):
        data = process_single_file(f, output_folder, male_bm, female_bm) # (333, 22, 3)
        if first_processed_data is None and data is not None:
            first_processed_data = data
            first_filename = os.path.basename(f)

    # ---------------- 可视化与检查 ----------------
    if first_processed_data is not None:
        print("\n" + "="*30)
        print(f"Details for {first_filename}")
        print("="*30)
        print(f"Shape: {first_processed_data.shape} -> (Frames, Joints, XYZ)")
        print(f"Frame count: {first_processed_data.shape[0]}")
        print(f"Joint count: {first_processed_data.shape[1]} (Should be 22)")
        
        # 打印某一帧的数值示例
        print("\nData Sample (Frame 0, First 5 joints):")
        print(first_processed_data[0, :5, :])
        
        # 保存可视化图片
        vis_save_path = os.path.join(output_folder, f"vis_{first_filename}.png")
        visualize_pose(first_processed_data, vis_save_path, frame_idx=0)
        
        print("\nDone! Check the output folder for .npy files and the visualization png.")

if __name__ == "__main__":
    main()