import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os
import sys
from argparse import ArgumentParser
from tqdm import tqdm
from glob import glob
from visual import visual_pos 

# 依赖检查
try:
    from common.skeleton import Skeleton
    from common.quaternion import * # 确保 qrot, qinv, qmul 等都在这里
    from paramUtil import t2m_raw_offsets, t2m_kinematic_chain
except ImportError:
    print("Error: 缺少 HumanML3D 依赖库 (common/..., paramUtil.py)。")
    sys.exit(1)


def augment_content_rotation_numpy(features, angle_degrees=90):
    """
    对 263 维特征进行 Y 轴旋转增强 (Numpy 版)
    features: (N, 263)
    """
    # 复制一份，不修改原始数据
    features = features.copy()
    frames = features.shape[0]
    
    # 转换角度
    theta = np.radians(angle_degrees)
    c, s = np.cos(theta), np.sin(theta)
    
    # 1. 旋转 Root Linear Velocity (Indices 1, 2)
    root_vx = features[:, 1]
    root_vz = features[:, 2]
    
    # 逆时针旋转
    new_root_vx = root_vx * c + root_vz * s
    new_root_vz = -root_vx * s + root_vz * c
    
    features[:, 1] = new_root_vx
    features[:, 2] = new_root_vz
    
    # 2. 旋转 Local Joint Positions (Indices 4 ~ 67)
    # 63 dims = 21 joints * 3
    joint_pos = features[:, 4:67].reshape(frames, 21, 3)
    pos_x = joint_pos[:, :, 0]
    pos_z = joint_pos[:, :, 2]
    
    new_pos_x = pos_x * c + pos_z * s
    new_pos_z = -pos_x * s + pos_z * c
    
    joint_pos[:, :, 0] = new_pos_x
    joint_pos[:, :, 2] = new_pos_z
    features[:, 4:67] = joint_pos.reshape(frames, 63)
    
    # 3. 旋转 Local Joint Velocities (Indices 193 ~ 259)
    # 66 dims = 22 joints * 3
    joint_vel = features[:, 193:259].reshape(frames, 22, 3)
    vel_x = joint_vel[:, :, 0]
    vel_z = joint_vel[:, :, 2]
    
    new_vel_x = vel_x * c + vel_z * s
    new_vel_z = -vel_x * s + vel_z * c
    
    joint_vel[:, :, 0] = new_vel_x
    joint_vel[:, :, 2] = new_vel_z
    features[:, 193:259] = joint_vel.reshape(frames, 66)
    
    return features

# ==========================================
# 0. 可视化工具 (保持不变，用于检查)
# ==========================================
def render_video(pose_data, save_path, title="Motion", fps=20):
    """渲染视频，HumanML3D 坐标系 (Y-Up, Z-Forward)"""
    print(f"Rendering: {save_path}")
    frames_num = pose_data.shape[0]
    chain = [
        [0,1],[1,4],[4,7],[7,10], [0,2],[2,5],[5,8],[8,11],
        [0,3],[3,6],[6,9],[9,12], [9,13],[13,16],[16,18],[18,20],
        [9,14],[14,17],[17,19],[19,21]
    ]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    lines = [ax.plot([], [], [], c='blue', linewidth=2)[0] for _ in chain]
    # Red for joints
    scats = ax.scatter([], [], [], s=20, c='red')
    
    # Arrows: R=X, G=Y, B=Z
    ax.quiver(0,0,0, 0.5,0,0, color='r') # Plot X
    ax.quiver(0,0,0, 0,0,0.5, color='g') # Plot Z (Data Y)
    ax.quiver(0,0,0, 0,0.5,0, color='b') # Plot Y (Data Z)
    
    radius = 1.2
    
    def update(frame_idx):
        p = pose_data[frame_idx]
        # Plot Logic: X=X, Y=Z(depth), Z=Y(height)
        xs, ys, zs = p[:, 0], p[:, 2], p[:, 1]
        
        scats._offsets3d = (xs, ys, zs)
        for line, link in zip(lines, chain):
            line.set_data([xs[link[0]], xs[link[1]]], [ys[link[0]], ys[link[1]]])
            line.set_3d_properties([zs[link[0]], zs[link[1]]])
            
        root_x, root_y, root_z = xs[0], ys[0], zs[0]
        ax.set_xlim(root_x - radius, root_x + radius)
        ax.set_ylim(root_y - radius, root_y + radius)
        ax.set_zlim(0, 2.0)
        ax.set_title(f"{title} | Frame: {frame_idx}")

    ax.set_xlabel('X'); ax.set_ylabel('Z (Fwd)'); ax.set_zlabel('Y (Up)')
    ax.view_init(elev=20, azim=-45)
    ani = FuncAnimation(fig, update, frames=frames_num, interval=1000/fps)
    try:
        ani.save(save_path, writer='ffmpeg', fps=fps)
    except:
        ani.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=fps)
    plt.close()

# ==========================================
# Step 1 & 2 (保持你之前验证正确的逻辑)
# ==========================================
def step1_uniform_skeleton(raw_data, reference_npy_path): # raw data: [N, 22, 3], 3指的是世界空间的位置
    print("\n[Step 1] Uniform Skeleton...")
    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    skel = Skeleton(n_raw_offsets, t2m_kinematic_chain, 'cpu')
    
    if not os.path.exists(reference_npy_path):
        ref_data = raw_data
    else:
        ref_data = np.load(reference_npy_path)
        if ref_data.shape[1] != 22: ref_data = ref_data.reshape(len(ref_data), -1, 3) # (180, 52, 3)
    
    tgt_offsets = skel.get_offsets_joints(torch.from_numpy(ref_data)[0]) # torch.Size([22, 3])
    src_tensor = torch.from_numpy(raw_data)
    src_offset = skel.get_offsets_joints(src_tensor[0])
    
    l_idx1, l_idx2 = 5, 8
    src_leg = torch.abs(src_offset[l_idx1]).max() + torch.abs(src_offset[l_idx2]).max()
    tgt_leg = torch.abs(tgt_offsets[l_idx1]).max() + torch.abs(tgt_offsets[l_idx2]).max()
    scale_rt = tgt_leg / src_leg
    
    tgt_root_pos = src_tensor[:, 0] * scale_rt
    quat_params = skel.inverse_kinematics_np(raw_data, [2, 1, 17, 16]) # (151, 22, 4)
    skel.set_offset(tgt_offsets)
    return skel.forward_kinematics_np(quat_params, tgt_root_pos.numpy())

def step2_canonicalize(positions):
    print("[Step 2] Canonicalize...")
    positions = positions.copy()
    positions[:, :, 1] -= positions.min(axis=0).min(axis=0)[1] # Floor
    
    root_pos_init = positions[0]
    positions = positions - root_pos_init[0] * np.array([1, 0, 1]) # XZ Origin
    
    r_hip, l_hip, sdr_r, sdr_l = 2, 1, 17, 16
    across = (root_pos_init[r_hip] - root_pos_init[l_hip]) + \
             (root_pos_init[sdr_r] - root_pos_init[sdr_l])
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
    
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis] # (1, 3)
    
    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init
    
    return qrot_np(root_quat_init, positions) # position：(151, 22, 3)

# ==========================================
# Step 3: 特征提取 (严格对应 ipynb Cell 3) 【nframes,22,3】->[nframes, 263]
# ==========================================
def step3_extract_features(positions): # shape:(54, 22, 3)
    print("[Step 3] Extracting Features (Notebook Logic)...")
    # 1. Foot Contacts
    fid_r, fid_l = [8, 11], [7, 10]
    velfactor = np.array([0.002, 0.002])
    feet_l = (((positions[1:, fid_l] - positions[:-1, fid_l]) ** 2).sum(axis=-1) < velfactor).astype(np.float32) # shape:(53, 2)
    feet_r = (((positions[1:, fid_r] - positions[:-1, fid_r]) ** 2).sum(axis=-1) < velfactor).astype(np.float32)
    
    # 2. Skeleton & IK
    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    skel = Skeleton(n_raw_offsets, t2m_kinematic_chain, "cpu")
    quat_params = skel.inverse_kinematics_np(positions, [2, 1, 17, 16], smooth_forward=True) # shape:(54, 22, 4)
    
    # 3. Root Rotation & Velocity
    r_rot = quat_params[:, 0].copy() # Root Rotation
    
    # Linear Velocity (XZ) - Rotate global velocity to local
    velocity = (positions[1:, 0] - positions[:-1, 0]).copy() # (53, 3)
    velocity = qrot_np(r_rot[1:], velocity) 
    l_velocity = velocity[:, [0, 2]] 
    
    # Angular Velocity (Y)
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
    r_velocity = np.arcsin(r_velocity[:, 2:3])  # shape:(53, 1)
    
    # Root Height
    root_y = positions[:, 0, 1:2]
    
    # 4. Rot Data (6D) & RIC (Local Positions)
    cont_6d_params = quaternion_to_cont6d_np(quat_params) # (54, 22, 6)
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1) # (54, 126=21x6)
    
    # RIC: Local position relative to root, rotated to local frame
    positions_local = positions.copy() # shape:(54, 22, 3)
    positions_local[..., 0] -= positions[:, 0:1, 0]
    positions_local[..., 2] -= positions[:, 0:1, 2]
    positions_local = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions_local)
    ric_data = positions_local[:, 1:].reshape(len(positions), -1) # shape:(54, 63)
    
    # Local Velocity
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], positions.shape[1], axis=1), positions[1:] - positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1) # shape:(53, 66)
    
    # 5. Concatenation (All T-1 length)
    data = np.concatenate([
        r_velocity,       # 0
        l_velocity,       # 1:3
        root_y[:-1],      # 3
        ric_data[:-1],    # 4:67 (RIC)
        rot_data[:-1],    # 67:193 (Rot)
        local_vel,        # 193:259
        feet_l,           # 259:261
        feet_r            # 261:263
    ], axis=-1)
    
    return data

# ==========================================
# Step 4: 还原逻辑 (严格对应 ipynb Cell 4)
# ==========================================
def recover_root_rot_pos(data):
    """
    完全复制 HumanML3D Cell 4 的 recover_root_rot_pos 函数
    """
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    # Get Y-axis rotation from rotation velocity
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    # 这里的切片 [..., :-1, 1:3] 是正确的，因为 data 是 (T, D)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3] 
    
    # Add Y-axis rotation to root position
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos

def recover_from_ric(data, joints_num):
    """
    完全复制 HumanML3D Cell 4 的 recover_from_ric 函数
    """
    # 确保是 Tensor
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    r_rot_quat, r_pos = recover_root_rot_pos(data)
    
    # 提取 RIC
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    # Add Y-axis rotation to local joints
    # 这里的逻辑是：将 RIC (局部坐标) 转回 世界坐标
    # 原 notebook 使用的是 qinv(r_rot_quat)
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    # Add root XZ to joints
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    # Concate root and joints
    # 将计算好的 Root 位置和 Body 位置拼起来
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions

def old_main():
    RAW_DATA_PATH = "/root/autodl-tmp/MyRepository/MCM-LDM/datasets/humanml3d/new_joints/000011.npy"
    REFERENCE_FILE = "/root/autodl-tmp/HumanML3D/HumanML3D/joints/000021.npy"
    OUTPUT_DIR = "./final_output_vis"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(REFERENCE_FILE):
        print(f"Warning: Reference file {REFERENCE_FILE} not found. Using input as reference.")
        REFERENCE_FILE = RAW_DATA_PATH

    # 1. Load & Viz Raw
    raw_data = np.load(RAW_DATA_PATH) # [N, 22, 3]
    render_video(raw_data, os.path.join(OUTPUT_DIR, "0_raw.mp4"), "0_Raw")

    # 2. Step 1 Uniform
    uniform_data = step1_uniform_skeleton(raw_data, REFERENCE_FILE)
    render_video(uniform_data, os.path.join(OUTPUT_DIR, "1_uniform.mp4"), "1_Uniform")

    # 3. Step 2 Canonical
    aligned_data = step2_canonicalize(uniform_data)
    render_video(aligned_data, os.path.join(OUTPUT_DIR, "2_aligned.mp4"), "2_Aligned")

    # 4. Step 3 Extract (Strict Notebook Logic)
    feature_vec = step3_extract_features(aligned_data)
    print(f"Features: {feature_vec.shape}") # Should be (N-1, 263)

    # ... (Step 4 结束) ...
    # feature_vec = step3_extract_features(aligned_data)
    # print(f"Features: {feature_vec.shape}")

    # ==========================================
    # [NEW] Test Rotation Augmentation
    # ==========================================
    print("\n[Test] Applying Rotation Augmentation (+90 degrees)...")
    # 旋转 90 度：本来向北，现在应该向西 (或东，取决于坐标系)
    aug_feature_vec = augment_content_rotation_numpy(feature_vec, angle_degrees=90.0)
    
    # 还原增强后的动作
    aug_rec_data = recover_from_ric(aug_feature_vec, 22).numpy()
    
    # 渲染增强后的动作
    # 预期效果：动作本身姿态不变(比如还是向前迈步)，但是整体轨迹发生了偏转
    render_video(aug_rec_data, os.path.join(OUTPUT_DIR, "4_augmented_90deg.mp4"), "4_Aug_90deg")
    
    # 再测试一个 -45 度
    print("[Test] Applying Rotation Augmentation (-45 degrees)...")
    aug_feature_vec_2 = augment_content_rotation_numpy(feature_vec, angle_degrees=-45.0)
    aug_rec_data_2 = recover_from_ric(aug_feature_vec_2, 22).numpy()
    render_video(aug_rec_data_2, os.path.join(OUTPUT_DIR, "5_augmented_neg45deg.mp4"), "5_Aug_neg45deg")
    # ==========================================

    # 5. Reconstruct (Strict Notebook Logic)
    # 输入 feature_vec (N-1, 263)
    # 输出 rec_data (N-1, 22, 3) -> 注意：notebook的还原是不补第一帧的，它直接用累计值
    rec_data = recover_from_ric(feature_vec, 22).numpy()
    
    print(f"Rec Data: {rec_data.shape}")

    # 6. Compare
    # aligned_data 是 (N, 22, 3)
    # rec_data 是 (N-1, 22, 3)
    # HumanML3D 的做法是，提取特征时会丢弃第一帧的绝对位置，
    # 还原时，第一帧默认位置可能是根据第一个速度算出来的，或者 notebook 这种写法会有错位
    # 但我们对比 aligned_data[...:N-1] 应该能对上大部分
    
    # 实际上，notebook 的 recover_root_rot_pos 第一帧是 0 加上 速度，所以对应 aligned_data[1]
    # 我们来对比 rec_data 和 aligned_data 的形状和内容
    
    min_len = min(len(rec_data), len(aligned_data))
    # HumanML3D 的还原通常与 aligned_data 存在对齐关系
    # 让我们直接算 rec_data 和 aligned_data (去掉最后一帧或者第一帧) 的误差
    
    # 尝试1: 假设 rec_data 对应 aligned_data 的 [0:N-1]
    diff1 = np.mean(np.linalg.norm(rec_data - aligned_data[:len(rec_data)], axis=-1))
    # 尝试2: 假设 rec_data 对应 aligned_data 的 [1:N]
    diff2 = np.mean(np.linalg.norm(rec_data - aligned_data[1:len(rec_data)+1], axis=-1))
    
    print(f"Diff (Rec vs Aligned[0:-1]): {diff1:.6f}")
    # print(f"Diff (Rec vs Aligned[1:]):  {diff2:.6f}")
    
    render_video(rec_data, os.path.join(OUTPUT_DIR, "3_reconstructed.mp4"), f"3_Rec (Err: {diff1:.4f})")
    
    if diff1 < 0.01:
        print(">>> SUCCESS! Features are correct.")
        np.save("final_features_263.npy", feature_vec)
    else:
        print(">>> FAIL. Still drifting.")

def render_result_mp4(save_path):
    """
    将 npy 文件转换为 mp4 视频并打印绝对路径
    save_path: 传入的 npy 文件路径
    """
    # 1. 确保获取的是绝对路径 (Absolute Path)
    abs_npy_path = os.path.abspath(save_path)
    
    # 2. 生成对应的 mp4 路径
    if abs_npy_path.endswith('.npy'):
        mp4_path = abs_npy_path.replace('.npy', '.mp4')
    else:
        # 预防万一传入的路径没带后缀
        mp4_path = abs_npy_path + '.mp4'
    
    # 3. 调用你的可视化接口
    # 假设 visual_pos 已经在你的全局作用域中定义好了
    print(f"🎬 正在渲染视频: {os.path.basename(mp4_path)} ...")
    visual_pos(abs_npy_path, mp4_path)
    
    # 4. 打印醒目的输出位置
    print("-" * 60)
    print(f"✅ 渲染完成！")
    print(f"📍 视频绝对路径: \033[92m{mp4_path}\033[0m") # 使用绿色高亮显示路径
    print("-" * 60)
    
    return mp4_path

PIPELINE = [
    ("step1", step1_uniform_skeleton),
    ("step2", step2_canonicalize),
    ("step3", step3_extract_features),
    ("step4", recover_from_ric)
]

def run_pipeline(input_data, base_save_path, start_node="step1", end_node="step3"):
    REFERENCE_FILE = "/root/autodl-tmp/HumanML3D/HumanML3D/joints/000021.npy"
    step_names = [name for name, _ in PIPELINE]
    try:
        start_idx = step_names.index(start_node)
        end_idx = step_names.index(end_node)
    except ValueError:
        return None

    executable_steps = PIPELINE[start_idx : end_idx + 1]
    current_data = input_data
    for name, func in executable_steps:
        # 如果你只想要最终结果，可以只在最后一步保存
        if name == "step1": # 骨骼的归一化处理
            current_data = func(current_data, REFERENCE_FILE)
        elif name == "step4": # 263维特征向量转回22维
            current_data = func(current_data, 22)
        else:
            current_data = func(current_data)
    current_save_path = base_save_path.replace(".npy", f"_process.npy")
    os.makedirs(os.path.dirname(current_save_path), exist_ok=True)
    np.save(current_save_path, current_data)  # <--- 核心代码
    print(f"💾 最终结果已保存至: {current_save_path}")
        
    
    return current_data

if __name__ == "__main__":
    parser = ArgumentParser(description="HumanML3D万能转换器 (带进度条版)")
    parser.add_argument("--start_stage", type=str, default="step1")
    parser.add_argument("--end_stage", type=str, default="step3")
    parser.add_argument("--input_dir", type=str, help="输入文件夹")
    parser.add_argument("--input_file", type=str, help="单个输入文件")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--is_feature263", action="store_true", help="是否是263维度的数据输入")
    parser.add_argument("--render_mp4", action="store_true", help="是否在转换之后渲染成mp4（263维度则会转回22joints再渲染成mp4）")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 收集文件
    files_to_process = []
    if args.input_file:
        files_to_process.append(args.input_file)
    elif args.input_dir:
        files_to_process = sorted(glob(os.path.join(args.input_dir, "*.npy")))
    
    if not files_to_process:
        print("❌ 未找到待处理文件，请检查路径。")
        exit()

    # 2. 使用 tqdm 初始化进度条
    # desc: 左侧描述文字, unit: 单位名称
    pbar = tqdm(files_to_process, desc="🚀 Processing Motion Data", unit="file")

    for file_path in pbar:
        file_name = os.path.basename(file_path)
        
        # 更新进度条左侧的动态信息（显示当前正在处理的文件名）
        pbar.set_description(f"📂 Processing: {file_name[:20]}...")

        try:
            raw_data = np.load(file_path)
            save_path = os.path.join(args.output_dir, file_name)
            
            if args.is_feature263:
                run_pipeline(
                    raw_data, save_path, start_node="step4", end_node="step4" 
                )
            else:
                run_pipeline(
                    raw_data, 
                    save_path, 
                    start_node=args.start_stage, 
                    end_node=args.end_stage
                )

            if args.render_mp4:
                render_path = save_path.replace('.npy', '_process.npy')
                render_result_mp4(render_path)
        except Exception as e:
            # 如果出错，使用 tqdm.write 以免破坏进度条的渲染
            pbar.write(f"⚠️ Error in {file_name}: {e}")

    print("\n✅ 所有任务已完成！结果保存在:", args.output_dir)
    
    