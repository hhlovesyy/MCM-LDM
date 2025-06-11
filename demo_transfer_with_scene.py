import logging
import os
import time
from builtins import ValueError
from multiprocessing.sharedctypes import Value
from pathlib import Path
import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader
# from torchsummary import summary
from tqdm import tqdm

from mld.config import parse_args
# from mld.datasets.get_dataset import get_datasets
from mld.data.get_data import get_datasets
from mld.data.sampling import subsample, upsample
from mld.models.get_model import get_model
from mld.utils.logger import create_logger

from visual import visual_pos 





def main():
    """
    get input text
    ToDo skip if user input text in command
    current tasks:
         1 text 2 mtion
         2 motion transfer
         3 random sampling
         4 reconstruction

    ToDo 
    1 use one funtion for all expoert
    2 fitting smpl and export fbx in this file
    3 

    """
    # note: 2025.5.23 解决报错：CUDA not avaliable
    import torch
    torch.cuda.init()  # 显式初始化 CUDA 上下文
    print(f"[强制初始化] CUDA 状态: is_available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}")
    # parse options
    cfg = parse_args(phase="demo")
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.Name = "demo--" + cfg.NAME
    logger = create_logger(cfg, phase="demo")




    style_path = cfg.DEMO.style_motion_dir
    content_path = cfg.DEMO.content_motion_dir


    # 
    cfg.DEMO.TIME = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    output_dir = Path(
        os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME),
                    'style_transfer' + cfg.DEMO.TIME))
    output_dir.mkdir(parents=True, exist_ok=True)

    # note： 2025.5.23 添加，因为只有一个GPU
    cfg.DEVICE = [0] 
    # cuda options
    if cfg.ACCELERATOR == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(x) for x in cfg.DEVICE)
        device = torch.device("cuda:0")
    # load dataset to extract nfeats dim of model
    dataset = get_datasets(cfg, logger=logger, phase="test")[0]
    # create mld model
    model = get_model(cfg, dataset)
    # loading checkpoints
    logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))
    state_dict = torch.load(cfg.TEST.CHECKPOINTS,
                            map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    logger.info("model {} loaded".format(cfg.model.model_type)) # Loading checkpoints from checkpoints/denoiser_checkpoint/denoiser.ckpt
    model.sample_mean = cfg.TEST.MEAN # False， sample_mean means that the model is trained with mean normalization 
    model.fact = cfg.TEST.FACT # 1 ， fact means that the model is trained with factorization normalization
    model.to(device)
    model.eval()

    scale = cfg.DEMO.scale

    for content in os.listdir(content_path):
        if not content.endswith('.npy'):
            continue
        # prepare conent motion
        content_file_name = content.split('.')[0] # 'jump'
        content_file_path = os.path.join(content_path, content) # 'demo/content_motion/jump.npy'
        content_motion = np.load(content_file_path) 
        content_motion = np.array([content_motion]) # shape:(1, 38, 263) 
        content_motion = torch.tensor(content_motion).to(device) # shape: torch.Size([1, 38, 263])
        
        # length is same as content motion
        length = content_motion.shape[1] # 38
        lengths = [int(length)]


        for style in os.listdir(style_path):
            if not style.endswith('.npy'):
                continue
            
            # prepare style motion
            style_file_name = style.split('.')[0] # '030001' ；'DrunkStyle' 进不同次
            style_file_path = os.path.join(style_path, style) # 'demo/style_motion/030001.npy'；'demo/style_motion/DrunkStyle.npy'
            style_motion = np.load(style_file_path) # shape: (263, 263)；（345, 263)
            style_motion = np.array([style_motion]) # shape:(1, 263, 263) ； shape:(1, 345, 263)
            style_motion = torch.tensor(style_motion).to(device) # shape:torch.Size([1, 263, 263]) ; ([1, 345, 263])

            # # 2025.05.26把Style本身可视化出来:帮我填写一下
            # # 可视化原始style运动数据
            # style_vis_path = os.path.join(output_dir, f"style_original_{style_file_name}.mp4")
            # style_motion_np = style_motion[0].detach().cpu().numpy()
            # np.save(os.path.join(output_dir, f"style_original_{style_file_name}.npy"), style_motion_np)
            # visual_pos(os.path.join(output_dir, f"style_original_{style_file_name}.npy"), style_vis_path)

            # start
            # 一个值的tensor，值是71，scene_class  71: ShieldedLeft 35:In the dark   56：OnToesBentForward（弯腰）
            # LeanRight: 41 ; star 77; balance 7
            id_to_english_action_map = {
                7: "Balance",# get
                9: "BentForward",# get
                71: "ShieldedLeft",# get
                35: "InTheDark", # # get  inthedark 效果感觉不好 一直没有伸手
                11: "BigSteps",## get 效果其实也一般
                10: "BentKnees",
                14: "Cat",
                23: "Drunk",
                34: "HighKnees",
                16: "CrossOver",
                19: "Depressed", #  get 低矮的房屋  但是这个不渲出来的话是不太明显的 反正火柴人看不太出来
                40: "LeanLeft",# get
                47: "Lunge",
                74: "SlideFeet", # get
                18: "CrowdAvoidance", # 效果不太行
                99: "DarkZo" ,#get 完全不行 这个有问题
                84 : "Swimming", # get 不太行 因为场景是调控
                # 如果有更多，继续添加
            }
            '''
            0: 独木桥
            1：低矮通道
            2：炎热天气
            3：水坑地面
            4：有易碎品的房间
            5：T台
            6：拥挤区域
            7：低矮天花板
            8：酒吧
            9：沙地/雪地
            10：黑暗场景
            11：倾斜的岩壁下
            12：湿滑地面
            13：暴风雨
            14：冰面
            '''
            sceneIdx = 7 # 

            # 2. 从字典中获取英文名称
            # 使用 .get() 方法可以在ID不存在时提供一个默认值，避免KeyError
            action_name_english = id_to_english_action_map.get(sceneIdx)
            
            scene_class = torch.tensor([sceneIdx], dtype=torch.long).to(device) # 71: 代表室内场景
            with torch.no_grad():

                # prepare batch data
                batch = {"length": lengths, "style_motion": style_motion, "tag_scale": scale, "content_motion": content_motion, 
                         "scene_labels":scene_class}
                # joints,latents = model(batch)
                # note：2025.5.23 报错：RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float，下一行为解决方案
                batch = {k: v.float() if torch.is_tensor(v) else v for k, v in batch.items()}
                joints = model(batch) # 0:shape:torch.Size([38, 22, 3]) ； 一样
                # npypath = str(output_dir /
                #             f"{content_file_name}_{style_file_name}_{str(lengths[0])}_scale_{str(scale).replace('.','-')}.npy")
                npypath = str(output_dir /
                            f"{content_file_name}_{style_file_name}_LowCeiling_{str(scale).replace('.','-')}.npy")
                mp4path = npypath.replace('.npy', '.mp4')
                # with open(npypath.replace(".npy", ".txt"), "w") as text_file:
                #     text_file.write('content {}'.format(content_file_name))
                #     text_file.write('#')
                #     text_file.write('style {}'.format(style_file_name))
                motion = joints[0].detach().cpu().numpy() # [38, 22, 3]
                np.save(npypath, motion)

                # visualization
                visual_pos(npypath, mp4path)

                logger.info(f"Motions are generated here:\n{npypath}")



if __name__ == "__main__":
    main()
