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
import itertools # 用来做组合

from visual import visual_pos 
import json
from torch.nn.utils.rnn import pad_sequence # 引入这个神器


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
    start_time = time.perf_counter()
    cfg = parse_args(phase="demo")
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.Name = "demo--" + cfg.NAME
    logger = create_logger(cfg, phase="demo")

    path2 = '/root/autodl-tmp/MyRepository/MCM-LDM/vis_debug'
    import shutil
    for filename in os.listdir(path2):
        file_path = os.path.join(path2, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或链接
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path) # 删除子文件夹
        except Exception as e:
            print(f'❌ 删除 {file_path} 失败，原因: {e}')

    print(f"✨ {path2} 内容已清理完毕")


    style_path = cfg.DEMO.style_motion_dir
    content_path = cfg.DEMO.content_motion_dir

    render_video = cfg.DEMO.render_video
    print("render video ? ", render_video)
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

    json_path = "/root/autodl-tmp/MyRepository/MCM-LDM/task_config.json"

    # 使用 with 语句打开文件（这样会自动关闭文件，更安全）
    with open(json_path, 'r', encoding='utf-8') as f:
        scene_data = json.load(f)

    

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
        
    # -------------------------------------------------------
    # 1. 预加载所有数据 (Load All into RAM)
    # -------------------------------------------------------
    print("🚀 Pre-loading all Contents and Styles...")

    # A. 加载所有 Content
    all_contents = [] 
    content_files = [f for f in os.listdir(content_path) if f.endswith('.npy')]
    for f in content_files:
        path = os.path.join(content_path, f)
        data = np.load(path)
        if len(data.shape) == 3: data = data[0] # (L, D)
        all_contents.append({
            "name": f.split('.')[0],
            "motion": torch.tensor(data).float(),
            "length": data.shape[0]
        })

    # B. 加载所有 Style
    all_styles = []
    style_files = [f for f in os.listdir(style_path) if f.endswith('.npy')]
    for f in style_files:
        path = os.path.join(style_path, f)
        data = np.load(path)
        if len(data.shape) == 3: data = data[0] # (L, D)
        all_styles.append({
            "name": f.split('.')[0],
            "motion": torch.tensor(data).float(),
            "length": data.shape[0]
        })

    print(f"✅ Loaded {len(all_contents)} Contents and {len(all_styles)} Styles.")

    # -------------------------------------------------------
    # 2. 生成任务列表 (Cartesian Product)
    # -------------------------------------------------------
    # 这就是你要的：直接算出所有组合 (Content X Style)
    # 结果类似: [(c1, s1), (c1, s2), ... (c2, s1), ...]
    all_tasks = list(itertools.product(all_contents, all_styles))
    total_tasks = len(all_tasks)
    
    print(f"🔥 Total Combinations (Tasks): {total_tasks}")
    
    # -------------------------------------------------------
    # 3. Batch 推理 (Flattened Batching)
    # -------------------------------------------------------
    BATCH_SIZE = 64 # 只要显存够，越大越快
    print(f"🚀 Inference Batch Size: {BATCH_SIZE}")

    # 这里的 tqdm 进度条显示的就是总进度的百分比了，非常直观
    for i in tqdm(range(0, total_tasks, BATCH_SIZE), desc="Running Batches"):
        
        # A. 获取当前 Batch 的任务对
        current_batch_tasks = all_tasks[i : i + BATCH_SIZE]
        current_bs = len(current_batch_tasks)

        # B. 整理数据 List
        batch_c_motions = []
        batch_c_lengths = []
        batch_s_motions = []
        batch_s_lengths = []
        batch_names = [] # 用来记这组数据叫什么，方便保存

        for (c_item, s_item) in current_batch_tasks:
            batch_c_motions.append(c_item["motion"])
            batch_c_lengths.append(c_item["length"])
            
            batch_s_motions.append(s_item["motion"])
            batch_s_lengths.append(s_item["length"])
            
            # 记录保存文件名: ContentName_StyleName
            batch_names.append(f"{c_item['name']}_{s_item['name']}")

        # C. 双重 Padding (核心！)
        # Content 和 Style 都可能长短不一，必须都 Pad
        # batch_first=True -> [Batch, MaxLen, Dim]
        c_motion_padded = pad_sequence(batch_c_motions, batch_first=True, padding_value=0.0).to(device) # torch.Size([8, 199, 263])
        s_motion_padded = pad_sequence(batch_s_motions, batch_first=True, padding_value=0.0).to(device) # torch.Size([8, 199, 263])
        
        # D. 构造其他条件 (Repeat) 
        scene_image_batch = scene_image.repeat(current_bs, 1, 1, 1) # torch.Size([8, 3, 224, 224])
        scene_text_batch = [scene_prompt] * current_bs
        scene_id_batch = scene_ids_tensor.repeat(current_bs)

        # E. 组装 Batch Dict
        batch = {
            "length": batch_c_lengths,       # Content 真实长度 List
            "content_motion": c_motion_padded, # Pad 过的 Content
            
            "style_length": batch_s_lengths,   # Style 真实长度 List (给 forward 里的 mask 用)
            "style_motion": s_motion_padded,   # Pad 过的 Style
            
            "tag_scale": scale,
            "scene_text": scene_text_batch,
            "scene_image": scene_image_batch,
            "scene_id": scene_id_batch,
            "has_image": has_image,
            "ablation_no_scene": True  # 如果需要场景的话，把这一项改成False就可以了
        }

        # F. 推理
        with torch.no_grad():
            # joints 返回的是一个列表 (remove_padding 后的结果列表) 
            # 或者是 Tensor，具体看你 forward 最后的 remove_padding 实现
            joints, global_pos = model(batch, scene_data) # global_pos: torch.Size([8, 199, 3])

        # G. 保存结果
        # 如果 remove_padding 返回的是 Tensor [B, L, J, 3]，我们需要根据 lengths 切分
        # 如果返回的是 List[Tensor]，直接遍历即可
        
        if isinstance(joints, torch.Tensor):
            joints = joints.detach().cpu().numpy()
        
        hint_trajectory = None
        for idx, save_name in enumerate(batch_names):
            # 获取单个结果
            motion_res = joints[idx]
            if global_pos is not None:
                hint_trajectory = global_pos[idx] # torch.Size([199, 3])
            
            # 如果是 Tensor 且没被 remove_padding 处理成 List，可能需要手动切片
            # 假设你的 forward 已经处理好了，或者在这里处理：
            if isinstance(motion_res, np.ndarray) and len(motion_res.shape) == 3: # [Frame, J, 3]
                 # 截取真实长度 (以防 model 返回的是 pad 过的结果)
                 real_len = batch_c_lengths[idx]
                 motion_res = motion_res[:real_len]
                 if global_pos is not None:
                    hint_trajectory = hint_trajectory[:real_len]
            elif isinstance(motion_res, torch.Tensor):
                 real_len = batch_c_lengths[idx]
                 motion_res = motion_res[:real_len].detach().cpu().numpy()
                 if global_pos is not None:
                    hint_trajectory = hint_trajectory[:real_len].detach().cpu().numpy() # shape:(199, 3)

            # 保存 NPY
            npypath = str(output_dir / f"{save_name}.npy")
            np.save(npypath, motion_res)
            
            # # 保存 Scene JSON
            # scene_info_path = npypath.replace('.npy', '_scene.json')
            # with open(scene_info_path, 'w') as f_json:
            #     json.dump(scene_data, f_json)
            mp4path = npypath.replace('.npy', '.mp4')
            if render_video or True:
                visual_pos(npypath, mp4path)
            traj_npypath = str(output_dir / f"{save_name}_givenTraj.npy")
            np.save(traj_npypath, hint_trajectory)

    print("✅ All Done!")
    # 记录结束时间
    end_time = time.perf_counter()

    # 计算差值
    elapsed_time = end_time - start_time
    print(f"🚀 任务执行耗时: {elapsed_time:.4f} 秒")



if __name__ == "__main__":
    main()
