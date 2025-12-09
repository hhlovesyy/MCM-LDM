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
    # 1. 独木桥 (Dumuqiao)
    "Dumuqiao": [
        "A cinematic shot of a person balancing on a narrow beam high in the sky, unstable footing, arms out for balance, fear of falling.", # Visual
        "Walking tightrope-style, extending arms sideways to maintain balance, taking slow and deliberate steps.", # Action
        "Balancing precariously on a thin line, body wobbling, trying to keep center of gravity.", # Physical
        "Walking on a narrow bridge.", # Simple
        "Crossing a single-plank bridge over a deep canyon." # Scenario
    ],

    # 2. 低矮通道 (DiAiTongDao)
    "DiAiTongDao": [
        "Moving through a cramped underground tunnel with a very low ceiling, body crouched and compressed, claustrophobic atmosphere.", # Visual
        "Walking while crouching down, bending knees deeply and hunching back to avoid hitting the head.", # Action
        "Stooping forward in a confined space, keeping the body low.", # Physical
        "Crouching while walking.", # Simple
        "Navigating inside a low ventilation shaft." # Scenario
    ],

    # 3. 水坑地面 (ShuiKengDiMian)
    "ShuiKengDiMian": [
        "Walking on a muddy road filled with dirty water puddles, carefully choosing dry spots, avoiding getting shoes wet.", # Visual
        "Taking irregular steps to jump over puddles, looking down at the ground constantly.", # Action
        "Navigating uneven and wet terrain, dodging water spots.", # Physical
        "Walking on muddy ground.", # Simple
        "A street full of rain puddles." # Scenario
    ],

    # 4. 玻璃房间 (BoLiFangJian)
    "BoLiFangJian": [
        "Trapped inside a transparent glass maze, hands reaching out to feel invisible walls, hesitant and confused movement.", # Visual
        "Walking with hands stretched forward to detect obstacles, moving slowly and cautiously.", # Action
        "Groping in an invisible enclosure, testing the air before stepping.", # Physical
        "Walking in a glass room.", # Simple
        "A mime artist pretending to be trapped in a box." # Scenario
    ],

    # 5. T台走秀 (T_Stage)
    "T_Stage": [
        "A supermodel walking on a fashion runway under spotlight, confident posture, rhythmic stride, elegant and high-fashion.", # Visual
        "Strutting with a cat-walk gait, shoulders back, hips swaying, stepping in a straight line.", # Action
        "Walking with intense confidence and upright posture.", # Physical
        "Fashion model walking.", # Simple
        "A high-fashion runway show." # Scenario
    ],

    # 6. 拥挤场合 (CroudedPlace)
    "CroudedPlace": [
        "Squeezing through a packed subway crowd during rush hour, protecting personal space, turning sideways to fit through gaps.", # Visual
        "Turning the torso sideways while walking, making small steps, arms held close to the body.", # Action
        "Navigating a high-density area, avoiding collisions with others.", # Physical
        "Walking through a crowd.", # Simple
        "A jammed market street." # Scenario
    ],

    # 7. 低矮天花板 (DiAiTianhuaban)
    "DiAiTianhuaban": [
        "Walking in a room with an extremely low roof, head ducked down instinctively to avoid hitting the beams, protective posture.", # Visual
        "Lowering the head and neck while walking, looking upwards occasionally.", # Action
        "Hunched over to fit under a low structure.", # Physical
        "Walking under a low ceiling.", # Simple
        "Moving in a basement with low hanging pipes." # Scenario
    ],

    # 8. 酒吧/醉酒 (Bar)
    "Bar": [
        "A heavily drunk person stumbling home, dizzy and disoriented, losing balance, swaying unpredictably from side to side.", # Visual
        "Walking with a staggering gait, tripping over own feet, unable to walk in a straight line.", # Action
        "Loss of motor control, gravity feels shifting, heavy limbs.", # Physical
        "Drunk walking.", # Simple
        "Leaving a bar late at night wasted." # Scenario
    ],

    # 9. 雪地/沙地 (WalkInSnowOrSand)
    "WalkInSnowOrSand": [
        "Trudging through deep soft snow, feet sinking into the ground, heavy resistance, lifting legs high to move forward.", # Visual
        "Marching with high knees, stomping down to break the surface, moving slowly.", # Action
        "Walking against high ground resistance, feet sinking.", # Physical
        "Walking in deep snow.", # Simple
        "Crossing a desert dune or snowy field." # Scenario
    ],

    # 10. 摸黑 (Dark)
    "Dark": [
        "Walking in a pitch-black room with zero visibility, moving blindly, hands waving in front to detect obstacles, slow testing steps.", # Visual
        "Shuffling feet carefully, arms reached out for protection, head turning to listen.", # Action
        "Navigating without vision, tentative movement.", # Physical
        "Walking in the dark.", # Simple
        "A blackout at night." # Scenario
    ],

    # 11. 左倾 (LeanLeft)
    "LeanLeft": [
        "Walking while carrying a heavy load on the left shoulder, body tilted significantly to the left, fighting to stay upright.", # Visual
        "Walking with the torso leaning to the left side.", # Action
        "Center of gravity shifted to the left, asymmetric gait.", # Physical
        "Leaning left.", # Simple
        "Walking against a strong wind blowing from the right." # Scenario
    ],

    # 12. 潮湿地面 (WetFloor)
    "WetFloor": [
        "Walking on a freshly polished wet floor, extremely slippery, stiff legs, tiny shuffling steps to prevent slipping and falling.", # Visual
        "Taking small, flat-footed steps, keeping the body stiff and vertical.", # Action
        "Zero friction surface, trying to maintain traction.", # Physical
        "Slippery floor.", # Simple
        "Walking on ice or wet tiles." # Scenario
    ],

    # 13. 暴风雨 (BaoFengYu)
    "BaoFengYu": [
        "Struggling against a violent hurricane wind blowing from the front, body leaning forward to penetrate the wind, heavy storm.", # Visual
        "Walking while shielding face with one arm, leaning torso forward, pushing against resistance.", # Action
        "Fighting high wind resistance, unstable balance, heavy steps.", # Physical
        "Walking in strong wind.", # Simple
        "Caught in a typhoon." # Scenario
    ],

    # 14. 冰面 (IcyRoad)
    "IcyRoad": [
        "Trying to walk on a frozen lake surface, zero friction, feet sliding uncontrollably, wide stance to keep center of gravity low.", # Visual
        "Sliding feet instead of lifting them, arms out for balance, knees bent.", # Action
        "Extremely slippery surface, loss of friction, careful balancing.", # Physical
        "Walking on ice.", # Simple
        "A frozen skating rink." # Scenario
    ]
}

SCENE_LIST = sorted([
    "BaoFengYu",        # 暴风雨
    "Bar",              # 酒吧
    "BoLiFangJian",     # 玻璃房间
    "CroudedPlace",     # 拥挤
    "Dark",             # 黑暗
    "DiAiTianhuaban",   # 低矮天花板
    "DiAiTongDao",      # 低矮通道
    "Dumuqiao",         # 独木桥
    "IcyRoad",          # 冰面
    "LeanLeft",         # 左倾
    "ShuiKengDiMian",   # 水坑
    "T_Stage",          # T台
    "WalkInSnowOrSand", # 雪地/沙地
    "WetFloor"          # 湿地
])

from torchvision import transforms
image_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                 (0.26862954, 0.26130258, 0.27577711))
        ])



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
    model.load_state_dict(state_dict, strict=False)
    # model.load_state_dict(state_dict, strict=False)
    logger.info("model {} loaded".format(cfg.model.model_type))
    model.sample_mean = cfg.TEST.MEAN
    model.fact = cfg.TEST.FACT
    model.to(device)
    model.eval()

    scale = cfg.DEMO.scale
    target_scene_label = "BaoFengYu" #  应该是用不上了，但为了保留字段
    scene_prompt = cfg.TEST.MULTI_MODAL_TEXT_PROMPT

    scene2id_dict = {scene: idx for idx, scene in enumerate(SCENE_LIST)}
    # id2scene_dict = {idx: scene for idx, scene in enumerate(SCENE_LIST)}
    # 使用 items() 方法遍历键值对
    for scene_name, scene_id in scene2id_dict.items():
        print(f"键 (Scene Name): {scene_name}, 值 (ID): {scene_id}")

    print("------------------------")
    target_scene_id = scene2id_dict[target_scene_label]
    scene_ids_tensor = torch.tensor([target_scene_id]).to(device)
    print("target scene id = ", scene_ids_tensor, scene_ids_tensor.shape)
    
    use_image_for_inference = (cfg.TEST.MULTI_MODAL_TYPE == 'image')
    print("IF Use Image for inference: ", use_image_for_inference)

    # 读一下scene_image
    has_image = torch.tensor([False]).to(device)
    scene_image = torch.zeros(3, 224, 224) # 默认全黑
    if use_image_for_inference:
        from PIL import Image
        scene_test_image_path = cfg.TEST.MULTI_MODAL_IMAGE_PATH
        if os.path.exists(scene_test_image_path):
            img = Image.open(scene_test_image_path ).convert("RGB")
            # 保存在这个路径下面：output_dir
            img.save(output_dir / "inference_image_visualize.png")
            print("save image to inference_image_visualize.png, check out!")
            scene_image = image_transform(img)
            if use_image_for_inference:
                has_image = torch.tensor([True]).to(device)
        else:
            print("can not find image for multi modal test!!")
    scene_image = scene_image.unsqueeze(0)

    if not has_image.item(): # 如果是文本，则把txt文件放到output_dir的路径下面
        with open(output_dir / "inference_text_prompt.txt", "w") as text_file:
            text_file.write(scene_prompt)
            text_file.write("\nstyle cfg: {}".format(cfg.TEST.CFG_STYLE))
            text_file.write("\nscene cfg: {}".format(cfg.TEST.CFG_SCENE))
            print(f"File saved to: {output_dir / 'inference_text_prompt.txt'}") # "w" 模式的含义是 "write"，覆盖写入，没有的话会创建
        
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
                batch = {"length": lengths, "style_motion": style_motion,  # torch.Size([1, 199, 263])
                        "tag_scale": scale, "content_motion": content_motion,
                        "scene_text": [scene_prompt] * len(lengths),
                        "scene_image": scene_image, # torch.Size([1, 3, 224, 224])
                        "scene_id": scene_ids_tensor,
                        "has_image": has_image} # torch.Size([1])
                # joints,latents = model(batch)
                joints = model(batch)
                npypath = str(output_dir /
                            f"{content_file_name}_{style_file_name}.npy")
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
