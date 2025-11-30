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

SCENE_DESCRIPTIONS = {
    # 1. 独木桥 - 侧重：平衡、张开双臂、小心翼翼
    "Dumuqiao": [
        "A person carefully crossing a narrow single-plank bridge, extending arms sideways to maintain balance.",
        "Walking tightrope-style on a very narrow beam, taking slow and deliberate steps, placing one foot directly in front of the other.",
        "Balancing precariously on a thin bridge, body wobbling slightly, arms outstretched for stability.",
        "Moving cautiously on a high narrow log, looking down at the feet to ensure safe footing."
    ],

    # 2. 低矮通道 - 侧重：弯腰、屈膝、低头
    "DiAiTongDao": [
        "Walking through a low tunnel, crouching down with bent knees and hunched back to avoid hitting the ceiling.",
        "Moving in a confined space with limited vertical clearance, keeping the head low and body compressed.",
        "Stooping forward while walking, maintaining a lowered posture to navigate through a short passage.",
        "Walking with a duck-walk posture, knees bent deeply, to fit through a low-hanging channel."
    ],

    # 3. 水坑地面 - 侧重：躲避、跨步、犹豫
    "ShuiKengDiMian": [
        "Navigating a muddy ground full of puddles, taking irregular steps to jump over or step around water spots.",
        "Walking carefully on uneven terrain with water pools, looking down to choose dry spots for footing.",
        "Dodging puddles on the ground, making sudden lateral adjustments and varying step lengths.",
        "Tip-toeing and hopping occasionally to avoid stepping into dirty water on the ground."
    ],

    # 4. 玻璃房间 - 侧重：摸索、迷茫、手伸向前方
    "BoLiFangJian": [
        "Trapped in a glass room, walking tentatively with hands reaching out to feel for invisible walls.",
        "Moving with hesitation and confusion, exploring the boundaries of a transparent enclosure with hands stretched forward.",
        "Walking blindly but cautiously, palms facing outward to detect potential glass barriers.",
        "Pacing around in a glass maze, testing the air with hands before taking a step."
    ],

    # 5. T台走秀 - 侧重：自信、挺胸、猫步、夸张的跨步
    "T_Stage": [
        "A fashion model walking on a runway, posture upright and confident, with a rhythmic and exaggerated catwalk strut.",
        "Strutting elegantly on a T-stage, shoulders back, hips swaying with each long step.",
        "Performing a high-fashion walk, looking straight ahead with a cool expression, steps aligned in a straight line.",
        "Walking with intense confidence and style, emphasizing the movement of the legs and hips like a supermodel."
    ],

    # 6. 拥挤的场合 - 侧重：侧身、避让、收缩身体
    "CroudedPlace": [
        "Navigating through a dense crowd, constantly turning the torso sideways to squeeze past people.",
        "Walking in a jammed subway station, making small steps and frequently adjusting direction to avoid collisions.",
        "Weaving through a busy street, protecting personal space by keeping arms close to the body.",
        "Shouldering through a thick crowd, stopping and starting abruptly, looking for gaps in the flow of people."
    ],

    # 7. 低矮天花板 - 侧重：此场景与低矮通道类似，但更强调头顶的压迫感
    "DiAiTianhuaban": [
        "Walking under a very low ceiling, head ducked down and neck bent forward to prevent injury.",
        "Moving with a hunched posture due to insufficient headroom, instinctively protecting the top of the head.",
        "Crouching slightly while walking, constantly looking up to check the clearance of the ceiling.",
        "Walking nervously under a low hanging structure, keeping the body low and compact."
    ],

    # 8. 酒吧 (Drunk) - 侧重：踉跄、摇晃、重心不稳
    "Bar": [
        "Stumbling out of a bar, swaying unpredictably from side to side, struggling to maintain a straight line.",
        "Walking with a drunk gait, footsteps heavy and uncoordinated, body leaning dangerously in random directions.",
        "Trying to walk straight while intoxicated, losing balance frequently and taking wide steps to recover.",
        "A tipsy walk, limbs feeling loose and heavy, occasionally tripping over own feet."
    ],

    # 9. 雪地或沙地 - 侧重：拔腿高抬、费力、陷落感
    "WalkInSnowOrSand": [
        "Trudging through deep snow, lifting knees high and stomping down to break the surface.",
        "Walking on soft sand, feet sinking into the ground with every step, requiring extra effort to push off.",
        "Slogging through heavy terrain, movement is slow and laborious, body leaning forward to generate momentum.",
        "Marching through deep powder snow, emphasizing high leg lifts and forceful grounding."
    ],

    # 10. 摸黑 - 侧重：手探路、脚步虚探、缓慢
    "Dark": [
        "Groping in pitch darkness, moving slowly with hands stretched forward to detect obstacles.",
        "Walking blindly in a blacked-out room, shuffling feet cautiously to feel the ground changes.",
        "Navigating without vision, body tense, arms waving slowly in front to protect the face.",
        "Moving hesitantly in the dark, taking small testing steps before committing weight to the foot."
    ],

    # 11. 左倾 - 侧重：非对称、单侧负重感、抗侧风
    "LeanLeft": [
        "Walking while constantly leaning to the left side, as if carrying a heavy weight on the left shoulder.",
        "Moving with a distinct tilt to the left, struggling to keep the body upright against a force.",
        "A gait with a permanent leftward list, body axis shifted off-center.",
        "Walking as if fighting a strong wind blowing from the right, leaning left to compensate."
    ],

    # 12. 潮湿地面 - 侧重：小碎步、脚掌平放、僵硬、防滑
    "WetFloor": [
        "Walking on a freshly mopped wet floor, taking tiny shuffling steps with stiff legs to prevent slipping.",
        "Moving cautiously on a slick surface, keeping feet flat and close to the ground.",
        "Treading on a slippery tiled floor, body stiff and center of gravity kept perfectly vertical.",
        "Walking as if on eggshells due to the wet floor, arms slightly out for emergency balance."
    ],

    # 13. 暴风雨 - 侧重：挡风、身体前倾、顶风
    "BaoFengYu": [
        "Battling against a violent rainstorm, using one arm to shield the face from rain and wind.",
        "Walking into a gale-force wind, leaning body forward significantly to penetrate the air resistance.",
        "Struggling against strong gusts, protecting eyes with hands, steps are heavy and grounded.",
        "Pushing through a storm, head down and shoulders hunched to minimize wind exposure."
    ],

    # 14. 冰面 - 侧重：极其小心、甚至有些滑动、双腿分开
    "IcyRoad": [
        "Walking on a frozen icy road, maintaining a wide stance for stability, sliding feet gently instead of lifting them.",
        "Moving on black ice, extremely cautious, knees bent to lower the center of gravity.",
        "Trying to walk on a skating rink without skates, arms flailing slightly to catch balance, steps are tentative.",
        "Navigating a slippery ice sheet, looking at the ground intently, fearing a fall at any moment."
    ]
}



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
    logger.info("model {} loaded".format(cfg.model.model_type))
    model.sample_mean = cfg.TEST.MEAN
    model.fact = cfg.TEST.FACT
    model.to(device)
    model.eval()

    scale = cfg.DEMO.scale
    target_scene_label = "DiAiTianhuaban"
    # 核心修复：获取对应的长文本 Prompt
    if target_scene_label in SCENE_DESCRIPTIONS:
        # 既然是推理，我们不用随机列表，直接取第一句或者你觉得最典型的一句
        # 如果 SCENE_DESCRIPTIONS 的 value 是 list，取 list[0]
        # 如果是 string，直接取
        val = SCENE_DESCRIPTIONS[target_scene_label]
        scene_prompt = val[0] if isinstance(val, list) else val
    else:
        scene_prompt = f"A person moving in {target_scene_label} environment."
        
    logger.info(f"Inferencing with Scene: {target_scene_label}")
    logger.info(f"Using Prompt: {scene_prompt}") # 打印出来确认一下


    for content in os.listdir(content_path):
        if not content.endswith('.npy'):
            continue
        # prepare conent motion
        content_file_name = content.split('.')[0]
        content_file_path = os.path.join(content_path, content)
        content_motion = np.load(content_file_path)
        content_motion = np.array([content_motion])
        content_motion = torch.tensor(content_motion).to(device)
        
        # length is same as content motion
        length = content_motion.shape[1]
        lengths = [int(length)]


        for style in os.listdir(style_path):
            if not style.endswith('.npy'):
                continue
            # prepare style motion
            style_file_name = style.split('.')[0]
            style_file_path = os.path.join(style_path, style)
            style_motion = np.load(style_file_path)
            style_motion = np.array([style_motion])
            style_motion = torch.tensor(style_motion).to(device)

            # start
            with torch.no_grad():

                # prepare batch data
                batch = {"length": lengths, "style_motion": style_motion, 
                        "tag_scale": scale, "content_motion": content_motion,
                        # 修复 1: 必须是 List，且长度要和 batch size 一致
                        "scene_text": [scene_prompt] * len(lengths),
                        # 修复 2: 加上 Image 占位符 (防止 mld.py 报错)
                        "scene_image": torch.zeros(len(lengths), 3, 224, 224).to(device)}
                # joints,latents = model(batch)
                joints = model(batch)
                npypath = str(output_dir /
                            f"{content_file_name}_{style_file_name}_{target_scene_label}_{str(lengths[0])}_scale_{str(scale).replace('.','-')}.npy")
                mp4path = npypath.replace('.npy', '.mp4')
                # with open(npypath.replace(".npy", ".txt"), "w") as text_file:
                #     text_file.write('content {}'.format(content_file_name))
                #     text_file.write('#')
                #     text_file.write('style {}'.format(style_file_name))
                motion = joints[0].detach().cpu().numpy()
                np.save(npypath, motion)

                # visualization
                visual_pos(npypath, mp4path)

                logger.info(f"Motions are generated here:\n{npypath}")



if __name__ == "__main__":
    main()
