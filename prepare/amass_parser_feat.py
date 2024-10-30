# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de


import joblib
import argparse
from tqdm import tqdm
import json
import os.path as osp
import os
import sys
sys.path.append('.')
from common.quaternion import *
from common.skeleton import Skeleton
from paramUtil import *
from visual import plot_3d_motion
import torch
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel
from src.datasets import smpl_utils
from src import config
import numpy as np
from PIL import Image

comp_device = torch.device("cpu")

dict_keys = ['betas', 'dmpls', 'gender', 'mocap_framerate', 'poses', 'trans']
action2motion_joints = [8, 1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 21, 24, 38]  # [18,]
text2motion_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21]  # [22,]
kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]

def get_joints_to_use(args):
    joints_to_use = np.array([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 37
    ])  # 23 joints + global_orient # 21 base joints + left_index1(22) + right_index1 (37)
    return np.arange(0, len(smpl_utils.SMPLH_JOINT_NAMES) * 3).reshape((-1, 3))[joints_to_use].reshape(-1)

framerate_hist = []

all_sequences = [
    'ACCAD',
    'BioMotionLab_NTroje',
    'CMU',
    'EKUT',
    'Eyes_Japan_Dataset',
    'HumanEva',
    'KIT',
    'MPI_HDM05',
    'MPI_Limits',
    'MPI_mosh',
    'SFU',
    'SSM_synced',
    'TCD_handMocap',
    'TotalCapture',
    'Transitions_mocap',
]
amass_test_split = ['Transitions_mocap', 'SSM_synced']
amass_vald_split = ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh']
# amass_train_split = ['BioMotionLab_NTroje']
amass_train_split = ['BioMotionLab_NTroje', 'Eyes_Japan_Dataset', 'TotalCapture', 'KIT', 'ACCAD', 'CMU', 'MPI_Limits',
                     'TCD_handMocap', 'EKUT']
# Source - https://github.com/nghorbani/amass/blob/08ca36ce9b37969f72d7251eb61564a7fd421e15/src/amass/data/prepare_data.py#L235
amass_splits = {
    'test': amass_test_split,
    'vald': amass_vald_split,
    'train': amass_train_split
}
assert len(amass_splits['train'] + amass_splits['test'] + amass_splits['vald']) == len(all_sequences) == 15

def read_data(folder, split_name,dataset_name, target_fps, max_fps_dist, joints_to_use, quick_run,babel_labels, clip_images_dir=None):
    # sequences = [osp.join(folder, x) for x in sorted(os.listdir(folder)) if osp.isdir(osp.join(folder, x))]
    # target_fps = None -> will not resample (for backward compatibility)

    if dataset_name == "amass":
        sequences = amass_splits[split_name]
    else:
        sequences = all_sequences

    db = {
        'vid_names': [],
        'thetas': [],
        "feats":[],
        'joints3d': [],
        'clip_images': [],
        'clip_pathes': [],
        'text_raw_labels': [],
        'text_proc_labels': [],
        'action_cat': []
    }

    # instance SMPL model
    print('Loading Body Models')
    body_models = {
        'neutral': BodyModel(config.SMPLH_AMASS_MODEL_PATH, num_betas=config.NUM_BETAS).to(comp_device)

    }
    print('DONE! - Loading Body Models')

    clip_images_path = clip_images_dir
    assert os.path.isdir(clip_images_path)


    for seq_name in sequences:
        print(f'Reading {seq_name} sequence...')
        seq_folder = osp.join(folder, seq_name)

        results_dict = read_single_sequence(split_name, dataset_name, seq_folder, seq_name, body_models, target_fps,
                                            max_fps_dist, joints_to_use, quick_run, clip_images_path, babel_labels)

        for k in db.keys(): db[k].extend(results_dict[k])

    return db


def joint_to_263(joints):
  
    pass


def uniform_skeleton(positions, target_offset, n_raw_offsets, kinematic_chain, l_idx1, l_idx2, face_joint_indx):
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()
    # print(src_offset)
    # print(tgt_offset)
    '''Calculate Scale Ratio as the ratio of legs'''
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    # print(scale_rt)
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
    # print(quat_params.shape)

    '''Forward Kinematics'''
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints

def process_file(positions, feet_thre, window=120, window_step=60, divide=True):
    
    
    #get tgt_offsets
    # Lower legs
    l_idx1, l_idx2 = 5, 8
    # Right/Left foot
    fid_r, fid_l = [8, 11], [7, 10]
    # Face direction, r_hip, l_hip, sdr_r, sdr_l
    face_joint_indx = [2, 1, 17, 16]
    # l_hip, r_hip
    r_hip, l_hip = 2, 1
    joints_num = 22

    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    kinematic_chain = t2m_kinematic_chain

    # Get offsets of target skeleton
    example_data = np.load("/root/jxlcode/HumanML3D/joints/000025.npy")
    example_data = example_data.reshape(len(example_data), -1, 3)
    example_data = torch.from_numpy(example_data)
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    # (joints_num, 3)
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])
    # print(tgt_offsets)




    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    '''Uniform Skeleton'''
    positions = uniform_skeleton(positions, tgt_offsets, n_raw_offsets, kinematic_chain, l_idx1, l_idx2, face_joint_indx)

    '''Put on Floor''' #与地面接触
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height
    #     print(floor_height)

    #     plot_3d_motion("./positions_1.mp4", kinematic_chain, positions, 'title', fps=20)

    '''XZ at origin'''
    root_pos_init = positions[0]#root在第一个数据
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])# 初始的帧的root，y设置为0
    positions = positions - root_pose_init_xz

    # '''Move the first pose to origin '''
    # root_pos_init = positions[0]
    # positions = positions - root_pos_init[0]

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    #     print(forward_init)

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions_b = positions.copy()

    positions = qrot_np(root_quat_init, positions)



    '''New ground truth positions'''
    global_positions = positions.copy()



    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    #     trejec = np.cumsum(np.concatenate([np.array([[0, 0, 0]]), velocity], axis=0), axis=0)
    #     r_rotations, r_pos = recover_ric_glo_np(r_velocity, velocity[:, [0, 2]])

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(ground_positions[:, 0, 0], ground_positions[:, 0, 2], marker='o', color='r')
    # plt.plot(trejec[:, 0], trejec[:, 2], marker='^', color='g')
    # plt.plot(r_pos[:, 0], r_pos[:, 2], marker='s', color='y')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)#根关节的Root Linear Velocity，Root Angular Velocity，y坐标

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data  # (F,4) 
    data = np.concatenate([data, ric_data[:-1]], axis=-1) #ric_data[:-1]]: (F,63)去掉rootpose，然后reshape成63，原来应该是(21,3)
    data = np.concatenate([data, rot_data[:-1]], axis=-1) #rot_data[:-1] ：(F, 126), 数据的6Dcont_6d_params，去掉了root，reshape成126，原始(21,6)
    #     print(data.shape, local_vel.shape)
    data = np.concatenate([data, local_vel], axis=-1) #local_vel: (F, 66),关节速度，(22,3)
    data = np.concatenate([data, feet_l, feet_r], axis=-1) #feet_l(F, 2), feet_r(F, 2), 左脚和右脚都有两个关节，每个关节相对于上一帧有没有动，没有动就判定这时候不能滑

    return data, global_positions, positions, l_velocity


def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos



def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


def read_single_sequence(split_name, dataset_name, folder, seq_name, body_models, target_fps, max_fps_dist,
                         joints_to_use, quick_run, clip_images_path, fname_to_babel):
    # target_fps = None -> will not resample (for backward compatibility)
    subjects = os.listdir(folder)

    thetas = []
    vid_names = []
    joints3d = []
    feats = []
    clip_images = []
    clip_pathes = []
    text_raw_labels = []
    text_proc_labels = []
    action_cat = []

    for subject in tqdm(subjects):
        actions = [x for x in os.listdir(osp.join(folder, subject)) if x.endswith('.npz')]

        for action in actions:
            fname = osp.join(folder, subject, action)
            if fname.endswith('shape.npz'):
                continue

            # Remove folder path from fname
            folder_path, sequence_name = os.path.split(folder)
            seq_subj_action = osp.join(sequence_name, subject, action)
            if seq_subj_action in fname_to_babel:
                babel_dict = fname_to_babel[seq_subj_action]
            else:
                print(f"Not in BABEL: {seq_subj_action}")
                continue

            if dataset_name == "babel":
                # # Check if pose belongs to split
                babel_split = babel_dict['split'].replace("val", "vald")  # Fix diff in split name
                if babel_split != split_name:
                    continue

            data = np.load(fname)
            duration_t = babel_dict['dur']
            fps = data['poses'].shape[0] / duration_t

            # Seq. labels
            seq_raw_labels, seq_proc_label, seq_act_cat = [], [], []
            frame_raw_text_labels = np.full(data['poses'].shape[0], "", dtype=np.object)
            frame_proc_text_labels = np.full(data['poses'].shape[0], "", dtype=np.object)
            frame_action_cat = np.full(data['poses'].shape[0], "", dtype=np.object)

            for label_dict in babel_dict['seq_ann']['labels']:
                seq_raw_labels.extend([label_dict['raw_label']])
                seq_proc_label.extend([label_dict['proc_label']])
                if label_dict['act_cat'] is not None:
                    seq_act_cat.extend(label_dict['act_cat'])

            # Frames labels
            if babel_dict['frame_ann'] is None:
                frame_raw_labels = "and ".join(seq_raw_labels)
                frame_proc_labels = "and ".join(seq_proc_label)
                start_frame = 0
                end_frame = data['poses'].shape[0]
                frame_raw_text_labels[start_frame:end_frame] = frame_raw_labels
                frame_proc_text_labels[start_frame:end_frame] = frame_proc_labels
                frame_action_cat[start_frame:end_frame] = ",".join(seq_act_cat)
            else:
                for label_dict in babel_dict['frame_ann']['labels']:
                    start_frame = round(label_dict['start_t'] * fps)
                    end_frame = round(label_dict['end_t'] * fps)
                    frame_raw_text_labels[start_frame:end_frame] = label_dict['raw_label']
                    frame_proc_text_labels[start_frame:end_frame] = label_dict['proc_label']
                    if label_dict['act_cat'] is not None:
                        frame_action_cat[start_frame:end_frame] = str(",".join(label_dict['act_cat']))

            if target_fps is not None:
                mocap_framerate = float(data['mocap_framerate'])
                sampling_freq = round(mocap_framerate / target_fps)
                if abs(mocap_framerate / float(sampling_freq) - target_fps) > max_fps_dist:
                    print('Will not sample [{}]fps seq with sampling_freq [{}], since target_fps=[{}], max_fps_dist=[{}]'
                          .format(mocap_framerate, sampling_freq, target_fps, max_fps_dist))
                    continue
                # pose = data['poses'][:, joints_to_use]
                pose = data['poses'][0::sampling_freq, joints_to_use]
                pose_all = data['poses'][0::sampling_freq, :]
                trans_all = data['trans'][0::sampling_freq, :]
                frame_raw_text_labels = frame_raw_text_labels[0::sampling_freq]
                frame_proc_text_labels = frame_proc_text_labels[0::sampling_freq]

            else:
                # don't sample
                pose = data['poses'][:, joints_to_use]
                pose_all = data['poses'][:, :]

            if pose.shape[0] < 60:
                continue

            theta = pose
            vid_name = np.array([f'{seq_name}_{subject}_{action[:-4]}']*pose.shape[0])

            if quick_run:
                joints = None
                images = None
            else:
                root_orient = torch.Tensor(pose_all[:, :(smpl_utils.JOINTS_SART_INDEX * 3)]).to(comp_device)
                pose_hand = torch.Tensor(pose_all[:, (smpl_utils.L_HAND_START_INDEX * 3):]).to(comp_device)
                pose_body = torch.Tensor(pose_all[:, (smpl_utils.JOINTS_SART_INDEX * 3):(
                            smpl_utils.L_HAND_START_INDEX * 3)]).to(comp_device)
                trans = torch.Tensor(trans_all).to(comp_device)
                # if data['gender'] == 'male':
                #     body_model = body_models['male']
                # else:
                #     body_model = body_models['female']
                body_model = body_models['neutral']
                trans_matrix = np.array([[1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [0.0, 1.0, 0.0]])
                # trans_matrix = torch.Tensor(trans_matrix).to(comp_device)
                body_motion = body_model(pose_body=pose_body, pose_hand=pose_hand, root_orient=root_orient)
                
                # without trajectory
                # joints = c2c(body_motion.Jtr+trans.unsqueeze(1))  # [seq_len, 52, 3]
                joints = c2c(body_motion.Jtr)  # [seq_len, 52, 3]
                joints = joints[:, text2motion_joints]  # [seq_len, 18, 3]
                joints = np.dot(joints, trans_matrix)
                joints[..., 0] *= -1

                # 22*3-----263
                feats1, ground_positions, positions, l_velocity = process_file(joints, 0.002)
                # joints = recover_from_ric(torch.from_numpy(feats1).unsqueeze(0).float(), 22)
                
                # plot_3d_motion('test_add_trans1_after_ric_-1.mp4', kinematic_chain, rec_ric_data.squeeze().numpy(), title='text_line', fps=30, radius=4)


                # # face Z
                # positions = joints
                # root_pos_init = positions[0]
                # # Face direction, r_hip, l_hip, sdr_r, sdr_l
                # face_joint_indx = [2, 1, 17, 16]

                # '''All initially face Z+'''
                # r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
                # across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
                # across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
                # across = across1 + across2
                # across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

                # # forward (3,), rotate around y-axis
                # forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
                # # forward (3,)
                # forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

                # #     print(forward_init)

                # target = np.array([[0, 0, -1]])
                # root_quat_init = qbetween_np(forward_init, target)
                # root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

                # positions_b = positions.copy()

                # positions = qrot_np(root_quat_init, positions)

                # joints = positions





                images = None
                images_path = None
                if clip_images_path is not None:
                    images_path = [os.path.join(clip_images_path, f) for f in os.listdir(clip_images_path) if f.startswith(vid_name[0]) and f.endswith('.png')]
                    images_path.sort(key=lambda x: int(x.replace('.png', '').split('frame')[-1]))
                    images_path = np.array(images_path)
                    images = [np.asarray(Image.open(im)) for im in images_path]
                    images = np.concatenate([np.expand_dims(im, 0) for im in images], axis=0)

            vid_names.append(vid_name)
            thetas.append(theta)
            joints3d.append(joints)
            feats.append(feats1)
            clip_images.append(images)
            clip_pathes.append(images_path)
            text_raw_labels.append(frame_raw_text_labels)
            text_proc_labels.append(frame_proc_text_labels)
            action_cat.append(frame_action_cat)


    # return np.concatenate(thetas, axis=0), np.concatenate(vid_names, axis=0)
    return {
        # 'betas': betas,
        'vid_names': vid_names,
        'thetas': thetas,
        'joints3d': joints3d,
        'feats': feats,
        'clip_images': clip_images,
        'clip_pathes': clip_pathes,
        'text_raw_labels': text_raw_labels,
        'text_proc_labels': text_proc_labels,
        'action_cat': action_cat
    }


def get_babel_labels(babel_dir_path):
    print("Loading babel labels")
    l_babel_dense_files = ['train', 'val']
    # BABEL Dataset
    pose_file_to_babel = {}
    for file in l_babel_dense_files:
        path = os.path.join(babel_dir_path, file + '.json')
        data = json.load(open(path))
        for seq_id, seq_dict in data.items():
            npz_path = os.path.join(*(seq_dict['feat_p'].split(os.path.sep)[1:]))
            seq_dict['split'] = file
            pose_file_to_babel[npz_path] = seq_dict
    print("DONE! - Loading babel labels")
    return pose_file_to_babel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='dataset directory', default='./data/amass')
    parser.add_argument('--output_dir', type=str, help='target directory', default='./data/amass_feats_notrans')
    parser.add_argument('--clip_images_dir', type=str, help='dataset directory', default='./data/render')
    parser.add_argument('--target_fps', type=int, choices=[10, 30, 60], default=30)
    parser.add_argument('--quick_run', action='store_true', help='quick_run wo saving and modeling 3d positions with smpl, just for debug')
    parser.add_argument('--dataset_name', required=True, type=str, choices=['amass', 'babel'], default='amass',
                        help='choose which dataset you want to create')
    parser.add_argument('--babel_dir', type=str, help='path to processed BABEL downloaded dir BABEL file',
                        default='./data/babel_v1.0_release')

    args = parser.parse_args()

    fname_to_babel = get_babel_labels(args.babel_dir)

    joints_to_use = get_joints_to_use(args)

    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    max_fps_dist = 5 # max distance from target fps that can be tolerated
    if args.quick_run:
        print('quick_run mode')

    for split_name in amass_splits.keys():
        db = read_data(args.input_dir,
                       split_name=split_name,
                       dataset_name=args.dataset_name,
                       target_fps=args.target_fps,
                       max_fps_dist=max_fps_dist,
                       joints_to_use=joints_to_use,
                       quick_run=args.quick_run,
                       babel_labels=fname_to_babel,
                       clip_images_dir=args.clip_images_dir
                       )


        db_file = osp.join(args.output_dir, '{}_{}fps'.format(args.dataset_name, args.target_fps))
        db_file += '_{}.pt'.format(split_name)
        if args.quick_run:
            print(f'quick_run mode - file should be saved to {db_file}')
        else:
            print(f'Saving AMASS dataset to {db_file}')
            joblib.dump(db, db_file)

