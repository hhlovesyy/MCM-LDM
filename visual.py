import os
import numpy as np
import torch
import visual_motion.paramUtil as paramUtil
from visual_motion.plot_script import plot_3d_motion

import argparse

parser = argparse.ArgumentParser(description='test')

parser.add_argument('--name', type=str, default='test1')
parser.add_argument('--motion_path', type=str, default=
'results/mld/test111/style_transfer2024-06-17-21-03'
)

def recover_root_rot_pos(data):
    """
    将 HumanML3D 的根节点特征还原为绝对轨迹 (X, Z)。
    data: (Seq_Len, 4) numpy array
    假设 4维特征顺序为: [Rot_Vel_Y, Vel_X, Vel_Z, Height_Y]
    """
    # 1. 提取旋转速度 (r_rot_vel) 并计算绝对朝向
    # HumanML3D 中 index 0 是 Y轴旋转角速度
    rot_vel = data[:, 0]
    r_rot = np.cumsum(rot_vel) # 累加得到每一帧的绝对朝向 (弧度)
    
    # 2. 提取位移速度 (r_vel_x, r_vel_z)
    # HumanML3D 中 index 1 是 X轴速度, 2 是 Z轴速度 (相对于当前朝向的局部速度)
    r_vel_x = data[:, 1]
    r_vel_z = data[:, 2]
    
    # 3. 将局部速度投影到全局坐标系
    # 公式：Global_X = Local_X * cos(rot) + Local_Z * sin(rot)
    #      Global_Z = -Local_X * sin(rot) + Local_Z * cos(rot) (坐标系定义可能略有差异，通常是这样)
    r_rot_quat = np.zeros(data.shape[0]) # 这一步简化，直接用下面的数学公式
    
    # 逐帧计算累加位置
    root_x = np.zeros(data.shape[0])
    root_z = np.zeros(data.shape[0])
    
    current_x, current_z = 0.0, 0.0
    
    # 为了精确，我们通常假设初始朝向是 0
    # 注意：这里做了一个简化，直接用 numpy 广播计算
    # global_vel_x = r_vel_x * np.cos(r_rot) + r_vel_z * np.sin(r_rot) # 这种旋转可能和数据集定义相反，试一下下面这个
    # HumanML3D 常用变换:
    # global_vel_x = r_vel_x * np.cos(r_rot) + r_vel_z * np.sin(r_rot)
    # global_vel_z = r_vel_z * np.cos(r_rot) - r_vel_x * np.sin(r_rot)

    global_vel_x = r_vel_x * np.cos(r_rot) - r_vel_z * np.sin(r_rot) 
    global_vel_z = r_vel_z * np.cos(r_rot) + r_vel_x * np.sin(r_rot)
    
    root_x = np.cumsum(global_vel_x)
    root_z = np.cumsum(global_vel_z)
    
    # 返回 (Seq_Len, 2) 包含 X, Z 坐标
    return np.stack([root_x, root_z], axis=-1)



def visual_pos(motion_path, save_path = './motion_output/59.mp4', caption = ' ', trans_cond=None):

#    motion_path = './datasets/cmu_new/test_file/000059.npy'
    skeleton = paramUtil.t2m_kinematic_chain
    #(F,22,3)
    motion = np.load(motion_path)[:, :22]

    # --- 新增：处理目标轨迹 ---
    target_trajec_2d = None
    if trans_cond is not None:
        # 1. 确保是 numpy 且去掉 batch 维度
        if isinstance(trans_cond, torch.Tensor):
            trans_data = trans_cond[0].detach().cpu().numpy() # [38, 4]
        else:
            trans_data = trans_cond[0]
            
        # 2. 如果数据是归一化的，这里需要反归一化 (根据你的实际情况决定是否取消注释)
        # trans_data = trans_data * std[:4] + mean[:4] 
        
        # 3. 还原轨迹
        target_trajec_2d = recover_root_rot_pos(trans_data)


    print('generate video for', save_path)
    plot_3d_motion(save_path, skeleton, motion, caption, fps=20, gt_trajec=target_trajec_2d)




if __name__ == "__main__":

    #
    args = parser.parse_args()

    output_dir = 'motion_output'
    output_path = os.path.join(output_dir, args.name)

    # text_file_path = "/root/jxlcode/style_latent_diffusion/datasets/humanml3d/texts"
    # style_text_path = "/root/jxlcode/style_latent_diffusion/datasets/humanml3d/style_texts"

    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)

    motion_path = args.motion_path
    file_list = os.listdir(motion_path)
    file_list.sort()

    for file in file_list:
        if '.npy' not in file:
            continue
        file_name = os.path.join(motion_path, file)
        save_path_1 = os.path.join(output_path, motion_path.split("/")[-1])
        if not os.path.exists(save_path_1):
            os.makedirs(save_path_1)
        save_path = os.path.join(save_path_1, '{}.mp4'.format(file.split('.')[0]))
        print('save video in {}'.format(save_path))


        #for origin
        # text_file = os.path.join(motion_path, file.split(".")[0] + '.txt')
        # with open(text_file, 'r') as f:
        #     action_desc = f.readline()



        # #for gt715
        # content = os.path.join(text_file_path, file.split(".")[0] + '.txt')
        # with open(content, "r") as text_file:
        #     text_line = text_file.readline().strip()
        #     text_line = text_line[:text_line.index("#")]
        # style = os.path.join(style_text_path, file.split(".")[0] + '.txt')
        # with open(style, "r") as style_text_file:
        #     style_text = style_text_file.read().strip()

        # caption = text_line + style_text




        visual_pos(file_name, save_path, "")
        # visual_pos(file_name, save_path, '')
        #visual_pos(file_name, save_path, caption)