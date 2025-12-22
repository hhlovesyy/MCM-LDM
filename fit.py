# borrow from optimization https://github.com/wangsen1312/joints2smpl
import os
import argparse
import pickle

import h5py
import natsort
import smplx

import torch

from mld.transforms.joints2rots import config
from mld.transforms.joints2rots.smplify import SMPLify3D

from mld.transforms.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle

from scipy.signal import savgol_filter

parser = argparse.ArgumentParser()
parser.add_argument("--pkl", type=str, default=None, help="pkl motion file")
parser.add_argument("--dir", type=str, default=None, help="pkl motion folder")
parser.add_argument("--num_smplify_iters", type=int, default=150, help="num of smplify iters") # 原来是150，写论文的时候记得改回去
parser.add_argument("--cuda", type=bool, default=True, help="enables cuda")
parser.add_argument("--gpu_ids", type=int, default=0, help="choose gpu ids")
parser.add_argument("--num_joints", type=int, default=22, help="joint number")
parser.add_argument("--joint_category", type=str, default="AMASS", help="use correspondence")
parser.add_argument("--fix_foot", type=str, default="False", help="fix foot or not")
opt = parser.parse_args()
print(opt)

if opt.pkl:
    paths = [opt.pkl]
elif opt.dir:
    paths = []
    file_list = natsort.natsorted(os.listdir(opt.dir))
    for item in file_list:
        if item.endswith('.pkl') and not item.endswith("_mesh.pkl"):
            paths.append(os.path.join(opt.dir, item))
else:
    raise ValueError(f'{opt.pkl} and {opt.dir} are both None!')

for path in paths:
    # load joints
    if os.path.exists(path.replace('.pkl', '_mesh.pkl')):
        print(f"{path} is rendered! skip!")
        continue

    with open(path, 'rb') as f:
        data = pickle.load(f)

    joints = data['joints']

    # # 试一下如何解决抖动的问题：在调用 SMPLify 之前，对生成的 joints 数据做一个简单的高斯平滑或者 Savitzky-Golay 滤波。
    # # print("Original joints shape:", joints.shape)
    # joints_smoothed = savgol_filter(joints, window_length=9, polyorder=2, axis=0)
    # joints = torch.from_numpy(joints_smoothed)

    # load predefined something
    device = torch.device("cuda:" + str(opt.gpu_ids) if opt.cuda else "cpu")
    print(config.SMPL_MODEL_DIR)
    smplxmodel = smplx.create(
        config.SMPL_MODEL_DIR,
        model_type="smpl",
        gender="neutral",
        ext="pkl",
        batch_size=joints.shape[0],
    ).to(device)

    # load the mean pose as original
    smpl_mean_file = config.SMPL_MEAN_FILE

    file = h5py.File(smpl_mean_file, "r")
    init_mean_pose = (
        torch.from_numpy(file["pose"][:])
        .unsqueeze(0).repeat(joints.shape[0], 1)
        .float()
        .to(device)
    )
    init_mean_shape = (
        torch.from_numpy(file["shape"][:])
        .unsqueeze(0).repeat(joints.shape[0], 1)
        .float()
        .to(device)
    )
    cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).unsqueeze(0).to(device)

    # initialize SMPLify
    smplify = SMPLify3D(
        smplxmodel=smplxmodel,
        batch_size=joints.shape[0],
        joints_category=opt.joint_category,
        num_iters=opt.num_smplify_iters,
        device=device,
    )
    print("initialize SMPLify3D done!")

    print("Start SMPLify!")
    keypoints_3d = torch.Tensor(joints).to(device).float()

    if opt.joint_category == "AMASS":
        confidence_input = torch.ones(opt.num_joints)
        # make sure the foot and ankle
        if opt.fix_foot:
            confidence_input[7] = 1.5
            confidence_input[8] = 1.5
            confidence_input[10] = 1.5
            confidence_input[11] = 1.5
    else:
        print("Such category not settle down!")

    # ----- from initial to fitting -------
    (
        new_opt_vertices,
        new_opt_joints,
        new_opt_pose,
        new_opt_betas,
        new_opt_cam_t,
        new_opt_joint_loss,
    ) = smplify(
        init_mean_pose.detach(),
        init_mean_shape.detach(),
        cam_trans_zero.detach(),
        keypoints_3d,
        conf_3d=confidence_input.to(device)
    )

    # 1. 获取 SMPLify 算出的原始数据
    # new_opt_pose: [Batch, 72] (Axis-Angle 格式，甚至包含非连续跳变)
    # root_trans: [Batch, 3] (位移)
    raw_pose = new_opt_pose.detach()
    raw_root = keypoints_3d[:, 0, :].detach() # 或者从 new_opt_vertices 推算，这里直接用关节的root比较稳
    
    # ==============================================================
    # [PhysiMoS 终极平滑] Axis-Angle -> Matrix -> 6D -> Smooth -> Matrix -> Axis-Angle
    # ==============================================================
    
    # A. 准备数据形状
    T = raw_pose.shape[0] # 帧数
    
    # B. 将不连续的轴角转为连续的旋转矩阵
    # reshape: [T, 72] -> [T*24, 3] -> [T*24, 3, 3]
    rot_mats = axis_angle_to_matrix(raw_pose.reshape(-1, 3)) 
    
    # C. 转为 6D 旋转表示 (取矩阵的前两列)
    # 6D 表示是完全连续的，非常适合线性平滑，不会出现万向节死锁
    rot_6d = rot_mats[..., :2].reshape(T, -1).cpu().numpy() # [T, 24*6 = 144]
    
    # D. 对 6D 数据进行 Savitzky-Golay 平滑
    # window_length: 越大越平滑，建议 15-21
    if T > 21:
        rot_6d_smooth = savgol_filter(rot_6d, window_length=15, polyorder=2, axis=0)
    else:
        rot_6d_smooth = rot_6d
        
    rot_6d_smooth = torch.from_numpy(rot_6d_smooth).to(device).float() # [T, 144]
    
    # E. 将平滑后的 6D 转回旋转矩阵 (Gram-Schmidt 正交化)
    # 这一步非常关键，它能修复平滑带来的矩阵变形，保证旋转矩阵的合法性
    # Reshape back to [T*24, 6] -> [T*24, 3, 2]
    rot_6d_vecs = rot_6d_smooth.view(-1, 3, 2)
    
    # 取出两列
    x_raw = rot_6d_vecs[:, :, 0] # [N, 3]
    y_raw = rot_6d_vecs[:, :, 1] # [N, 3]
    
    # 标准化 X
    x_norm = x_raw / (torch.norm(x_raw, dim=1, keepdim=True) + 1e-8)
    # 叉乘算出 Z (Z 垂直于 X 和 Y)
    z_norm = torch.cross(x_norm, y_raw, dim=1)
    z_norm = z_norm / (torch.norm(z_norm, dim=1, keepdim=True) + 1e-8)
    # 叉乘算出正交的 Y (Y 垂直于 Z 和 X)
    y_norm = torch.cross(z_norm, x_norm, dim=1)
    
    # 拼回去 [N, 3, 3]
    matrix_smooth = torch.stack([x_norm, y_norm, z_norm], dim=2)
    
    # F. 最后转回 Axis-Angle 喂给 SMPL
    # [T*24, 3] -> [T, 72]
    pose_smoothed = matrix_to_axis_angle(matrix_smooth).reshape(T, 72)
    
    # G. 顺便把位移也平滑了
    root_np = raw_root.cpu().numpy()
    if T > 21:
        root_smooth_np = savgol_filter(root_np, window_length=15, polyorder=2, axis=0)
    else:
        root_smooth_np = root_np
    root_smoothed = torch.from_numpy(root_smooth_np).to(device).float()

    # ==============================================================

    # 使用平滑后的参数生成最终 Mesh
    output = smplxmodel(
        betas=torch.zeros_like(new_opt_betas),
        global_orient=pose_smoothed[:, :3],  # 你的 0-3 是全局旋转
        body_pose=pose_smoothed[:, 3:],      # 3-72 是身体姿态
        transl=root_smoothed,                # 平滑后的位移
        return_verts=True
    )
    
    vertices = output.vertices.detach().cpu().numpy()
    floor_height = vertices[..., 1].min()
    vertices[..., 1] -= floor_height
    data['vertices'] = vertices

    # # fix shape
    # betas = torch.zeros_like(new_opt_betas)
    # root = keypoints_3d[:, 0, :]

    # output = smplxmodel(
    #     betas=betas,
    #     global_orient=new_opt_pose[:, :3],
    #     body_pose=new_opt_pose[:, 3:],
    #     transl=root,
    #     return_verts=True
    # )
    # vertices = output.vertices.detach().cpu().numpy()
    # floor_height = vertices[..., 1].min()
    # vertices[..., 1] -= floor_height
    # data['vertices'] = vertices

    save_file = path.replace('.pkl', '_mesh.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(data, f)
    print(f'vertices saved in {save_file}')
