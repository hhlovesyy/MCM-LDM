import numpy as np
import torch
from torch.utils import data
import random
from os.path import join as pjoin
import logging
import json
import collections

# 初始化日志模块
logger = logging.getLogger(__name__)

class PhysiMoS100StyleDataset(data.Dataset):
    """
    PhysiMoS 项目技术探针专用 Dataset 类。
    功能：
    1. 加载 100Style 数据集动作文件。
    2. 基于 JSON 映射文件，生成"伪"物理参数和场景类别。
    3. 实现物理参数的噪声注入，防止过拟合。
    4. 遵循 Self-Reconstruction (自重建) 代理任务逻辑。
    """
    def __init__(
        self,
        mean,
        std,
        split_file,
        motion_dir,
        max_motion_length,
        min_motion_length,
        unit_length,
        style_dict_path,
        scene_mapping_path,   # 新增的 scenes.json 路径
        style_subset=None,    # 可选：用于快速测试的小样本子集
        phys_noise_scale=0.05, # 新增：物理参数的噪声强度，防止死记硬背
        **kwargs,             # 优雅地忽略其他无关参数
    ):
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length
        self.mean = mean
        self.std = std
        self.phys_noise_scale = phys_noise_scale  # 保存噪声系数

        # --- 1. 加载场景和物理参数映射 (JSON) ---
        with open(scene_mapping_path, 'r') as f:
            scene_data = json.load(f)
        self.style_to_phys = scene_data["style_mapping"]
        # self.scene_categories = scene_data["scene_categories"]
        # # 构建 场景名 -> ID 的映射字典
        # self.scene_to_id = {name: i for i, name in enumerate(self.scene_categories)}
        # self.num_scenes = len(self.scene_categories)

        self.scene_categories = ["Windy"] 
        self.scene_to_id = {"Windy": 0}
        self.num_scenes = 1 # 只有一个场景类别，即“有风”，看一下模型有没有希望从物理参数中学到东西

        self.phys_params_dim = len(scene_data["physical_parameters_desc"])
        
        logger.info(f"PhysiMoS: 已加载场景映射，包含 {len(self.style_to_phys)} 种风格，归属于 {self.num_scenes} 类场景。")
        logger.info(f"PhysiMoS: 物理参数维度为 {self.phys_params_dim}。噪声强度设置为: {self.phys_noise_scale}")

        # --- 2. 加载 文件ID -> 风格名 的映射 ---
        self.id_to_style = {}
        with open(style_dict_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 2:
                    file_id = parts[0]
                    # 处理风格名格式：转小写，去空格，取下划线前缀
                    style_name_key = parts[1].split('_')[0].lower().replace(" ", "")
                    self.id_to_style[file_id] = style_name_key
        
        # 获取当前划分的名称 (TRAIN/TEST) 用于日志
        split_name = split_file.split('/')[-1].split('.')[0].upper()
        logger.info(f"--- [DIAGNOSTIC REPORT FOR {split_name} SPLIT] ---")

        # --- 3. 读取数据集 Split 文件中的 ID 列表 ---
        id_list = []
        with open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        logger.info(f"1. '{split_name}.txt' 中发现总文件 ID 数: {len(id_list)}")

        # --- 4. 定义已知的损坏或需要排除的风格 (黑名单) ---
        styles_to_exclude = { "whirlarms", "widelegs", "wigglehips", "wildarms", "wildlegs", "zombie" }

        # --- 5. [核心逻辑] 数据加载与过滤 ---
        self.data_dict = {}
        self.name_list = []
        
        # 确定需要加载哪些风格
        valid_styles_from_json = set(self.style_to_phys.keys())
        if style_subset:
            # 如果是在做探针测试(Probe Mode)，取交集
            selected_styles = set(style_subset).intersection(valid_styles_from_json)
            logger.info(f"[PROBE MODE] 仅加载 {len(selected_styles)} 种指定风格: {selected_styles}")
        else:
            selected_styles = valid_styles_from_json
        
        # 统计拒绝加载的原因，用于诊断
        rejection_reasons = collections.defaultdict(int)
        
        for name in id_list:
            style_name = self.id_to_style.get(name)
            
            # 过滤条件:
            # 1. 必须有风格名
            # 2. 不能在黑名单中
            # 3. 必须在 selected_styles 范围内
            if not style_name or style_name in styles_to_exclude or style_name not in selected_styles:
                rejection_reasons['unselected_style'] += 1
                continue

            try:
                motion_path = pjoin(motion_dir, name + ".npy")
                motion = np.load(motion_path)

                # 4. 长度过滤
                if not (self.min_motion_length <= len(motion) < self.max_motion_length):
                    rejection_reasons['invalid_length'] += 1
                    continue
                
                # 所有检查通过，存入内存
                self.data_dict[name] = {
                    "motion": motion,
                    "length": len(motion),
                    "style_name": style_name,
                }
                self.name_list.append(name)
            except Exception:
                rejection_reasons['file_not_found_or_corrupt'] += 1
                continue
        
        # --- 打印详细的诊断报告 ---
        total_rejected = sum(rejection_reasons.values())
        total_processed = len(id_list)
        pass_rate = (len(self.name_list) / total_processed) * 100 if total_processed > 0 else 0

        logger.info(f"2. 处理样本总数: {total_processed}")
        logger.info(f"   - 被拒绝样本数: {total_rejected}")
        logger.info(f"     - 原因 '非选定风格': {rejection_reasons['unselected_style']}")
        logger.info(f"     - 原因 '长度不符': {rejection_reasons['invalid_length']}")
        logger.info(f"     - 原因 '文件缺失/损坏': {rejection_reasons['file_not_found_or_corrupt']}")
        logger.info(f"   - 成功加载样本数: {len(self.name_list)}")
        logger.info(f"3. 通过率: {pass_rate:.2f}%")
        logger.info(f"--- [END OF REPORT] ---")

        if not self.name_list:
            raise ValueError(f"PhysiMoS ({split_name}): 未加载到任何有效样本，请检查路径或配置！")
        
        self.nfeats = self.data_dict[self.name_list[0]]['motion'].shape[1]

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        
        motion, m_length, style_name = data["motion"], data["length"], data["style_name"]

        # --- 步骤 1: 获取对应的物理参数和场景标签 ---
        phys_data = self.style_to_phys[style_name]
        scene_name = phys_data["scene_category"]
        raw_phys_params = phys_data["physical_parameters"]

        # 处理物理参数：转 Tensor 并注入噪声
        # [重要] 这里的噪声是为了防止模型单纯记住 "这个数值组合 = 这个动作"
        # 训练时加噪声，推理时(Evaluation)通常不加，但在探针阶段为了鲁棒性可以一直加，或者在这里判断 self.train
        phys_params_tensor = torch.tensor(raw_phys_params, dtype=torch.float32)
        if self.phys_noise_scale > 0:
            noise = torch.randn_like(phys_params_tensor) * self.phys_noise_scale
            phys_params_tensor = phys_params_tensor + noise
            # 可选：如果是归一化参数，可能需要 clamp 到 0-1 之间，视具体物理定义而定
            # phys_params_tensor = torch.clamp(phys_params_tensor, 0.0, 1.0)

        # 处理场景标签：转 One-Hot
        scene_id = self.scene_to_id[scene_name]
        scene_cat_one_hot = np.zeros(self.num_scenes, dtype=np.float32)
        scene_cat_one_hot[scene_id] = 1.0

        # --- 步骤 2: 随机裁剪与标准化 ---
        # 计算裁剪后的长度（必须是 unit_length 的整数倍，适应 VAE/Transformer 的窗口）
        m_length_cropped = (m_length // self.unit_length) * self.unit_length
        
        # 鲁棒性修复：如果原始长度恰好等于 unit_length，避免 randint(0,0) 报错
        if m_length_cropped < self.unit_length:
             # 理论上前面 init 已经过滤了过短的，但为了双重保险
            m_length_cropped = self.unit_length

        if m_length > m_length_cropped:
            idx = random.randint(0, m_length - m_length_cropped)
        else:
            idx = 0
            
        motion_cropped = motion[idx:idx + m_length_cropped]
        
        # 标准化 (Standardization)
        motion_normalized = (motion_cropped - self.mean) / self.std

        # NaN 检测：如果标准化后出现 NaN，递归重试另一个样本
        if np.any(np.isnan(motion_normalized)):
            logger.warning(f"NaN detected in motion sample {name}. Resampling...")
            return self.__getitem__(np.random.randint(0, len(self.name_list)))

        # --- 步骤 3: 构建输入与目标 (Input & Target) ---
        # 目前是自重建任务 (Self-Reconstruction)
        motion_after = motion_normalized
        motion_before = motion_after.copy() # Deep copy

        # 这里的 motion_before 是作为 Condition 输入给模型的 Content
        # 未来我们可能在这里做 mask (随机遮挡) 或者 zero-out，强迫模型关注 phys_params

        # --- 步骤 4: 返回字典 ---
        return {
            "motion_after": torch.from_numpy(motion_after).float(),   # Ground Truth (Target)
            "motion_before": torch.from_numpy(motion_before).float(), # Input Condition (Content)
            "length": m_length_cropped,
            "phys_params": phys_params_tensor,                        # Input Condition (Physics)
            "scene_cat": torch.from_numpy(scene_cat_one_hot),         # Input Condition (Scene)
            "caption": f"Style: {style_name}, Scene: {scene_name}",   # 调试用文本
        }
    


import os
class PhysicsDataset(data.Dataset):
    def __init__(
        self,
        mean,
        std,
        split_file, # 虽然我们可能不需要split文件，但为了兼容接口保留
        motion_dir,
        json_dir,
        max_motion_length=196, # 训练时裁剪的最大长度
        min_motion_length=40,
        unit_length=4,
        max_wind_force=330000.0, # 【关键】根据你的数据统计设定，用于归一化
        max_ceiling_height=220.0,
        is_train=True,# ============0108=================
        **kwargs,
    ):
        self.mean = mean
        self.std = std
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length
        self.max_wind_force = max_wind_force
        self.max_ceiling_height = max_ceiling_height
        
        self.is_train = is_train # <--- 【改动2】把参数存下来！就是缺了这一行导致的报错 

        self.phys_dim = 6 # [wind_x, wind_y, wind_mag, ceiling_height, gap_width, gap_offset]
        
        self.motion_dir = motion_dir
        self.json_dir = json_dir

        self.data_list = []
        all_json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        
        for fname in all_json_files:
            motion_path = pjoin(motion_dir, fname.replace(".json", ".npy"))
            if os.path.exists(motion_path):
                self.data_list.append({"motion_path": motion_path, "json_path": pjoin(json_dir, fname)})
        
        logger.info(f"PhysicsDataset: Loaded {len(self.data_list)} total samples.")
        
    def __len__(self):
        return len(self.data_list)

    # def __getitem__(self, item):
    #     data_item = self.data_list[item]
        
    #     # --- A. 加载动作数据 ---
    #     motion = np.load(data_item["motion_path"]) # (Total_Frames, 263)
    #     total_frames = motion.shape[0]
        
    #     # --- B. 随机裁剪 (Random Crop) ---
    #     # 确保裁剪长度是 unit_length 的倍数 (VAE requirement)
    #     # 策略：如果有足够长度，随机切一段；否则取全部
        
    #     target_len = self.max_motion_length
    #     # 确保 target_len 是 4 的倍数
    #     target_len = (target_len // self.unit_length) * self.unit_length
        
    #     if total_frames > target_len:
    #         # 随机选择起始点
    #         max_start = total_frames - target_len
    #         start_idx = random.randint(0, max_start)
    #         motion_crop = motion[start_idx : start_idx + target_len]
    #     else:
    #         # 如果太短，就裁剪掉尾部多余的帧使其符合 unit_length
    #         valid_len = (total_frames // self.unit_length) * self.unit_length
    #         if valid_len < self.unit_length: valid_len = self.unit_length # 至少留一点
    #         motion_crop = motion[:valid_len]
            
    #     # --- C. 动作归一化 (Motion Normalization) ---
    #     motion_norm = (motion_crop - self.mean) / self.std
        
    #     # --- D. 加载物理参数 (Physics Condition) ---
    #     with open(data_item["json_path"], 'r') as f:
    #         meta = json.load(f)
        
    #     phys_params_np = np.zeros(self.phys_dim, dtype=np.float32)
    #     params = meta.get("parameters", {})
        
    #     # 处理风力
    #     if "wind_force" in params:
    #         wf = params["wind_force"]
    #         wind_vec = np.array([wf.get('x', 0), wf.get('y', 0)])
    #         mag = np.linalg.norm(wind_vec)
    #         phys_params_np[0] = wf.get('x', 0) / self.max_wind_force
    #         phys_params_np[1] = wf.get('y', 0) / self.max_wind_force
    #         phys_params_np[2] = mag / self.max_wind_force
            
    #     # 处理天花板
    #     if "ceiling_height" in params:
    #         ch = params["ceiling_height"]
    #         # 值越低 -> 特征值越高 (1.0)
    #         phys_params_np[3] = max(0, 1.0 - (ch / self.max_ceiling_height))
        
    #     if "gap_width" in params:
    #         gw = params["gap_width"]
    #         phys_params_np[4] = gw / 120.0
    #     if "gap_offset" in params:
    #         go = params["gap_offset"]
    #         phys_params_np[5] = go / 20.0

    #     if self.is_train: # 在训练的时候添加一个随机噪声，避免模型
    #         noise_scale = 0.05
    #         noise = np.random.randn(*phys_params_np.shape) * noise_scale
    #         phys_params_np += noise
    #         # 可以选择性地 clamp 到 [-1, 1]
    #         phys_params_np = np.clip(phys_params_np, -1.0, 1.0)
    #     phys_params = torch.from_numpy(phys_params_np).float()

    #     # [修改] 生成一个虚拟的、长度为1的 scene_cat，以匹配 mld.py 的接口
    #     scene_cat = torch.ones(1).float()

    #     motion_after = torch.from_numpy(motion_norm).float()
    #     motion_before = motion_after.clone()

    #     dummy_caption = "Varsapura"
    #     return {
    #         "motion_after": motion_after,
    #         "motion_before": motion_before,
    #         "length": len(motion_after),
    #         "phys_params": phys_params,
    #         "scene_cat": scene_cat, # 喂一个虚拟值
    #         "caption": dummy_caption,
    #     }
    def __getitem__(self, item):
        print("33333333333Debug: __getitem__ called with item =", item)  # <--- 【改动1】添加调试打印
        data_item = self.data_list[item]
        
        # [新增] 获取文件名，用于判断当前是哪种场景 (Wind/Ceiling/Gap)
        filename = os.path.basename(data_item["json_path"]) 

        # --- A. 加载动作数据 (保持不变) ---
        motion = np.load(data_item["motion_path"]) # (Total_Frames, 263)
        total_frames = motion.shape[0]
        
        # --- B. 随机裁剪 (保持不变) ---
        target_len = self.max_motion_length
        # 确保 target_len 是 4 的倍数
        target_len = (target_len // self.unit_length) * self.unit_length
        
        if total_frames > target_len:
            # 随机选择起始点
            max_start = total_frames - target_len
            start_idx = random.randint(0, max_start)
            motion_crop = motion[start_idx : start_idx + target_len]
        else:
            # 如果太短，就裁剪掉尾部多余的帧使其符合 unit_length
            valid_len = (total_frames // self.unit_length) * self.unit_length
            if valid_len < self.unit_length: valid_len = self.unit_length # 至少留一点
            motion_crop = motion[:valid_len]
            
        # --- C. 动作归一化 (保持不变) ---
        motion_norm = (motion_crop - self.mean) / self.std
        
        # --- D. 加载物理参数 (Physics Condition) ---
        with open(data_item["json_path"], 'r') as f:
            meta = json.load(f)
        
        phys_params_np = np.zeros(self.phys_dim, dtype=np.float32)
        params = meta.get("parameters", {})

        # [新增] 定义场景判断逻辑
        is_wind = filename.startswith("W") or "Wind" in filename
        is_ceiling = "Ceiling" in filename
        is_gap = "Gap" in filename

        # [新增] 定义 Hack 用到的临时常量 (建议根据你的实际数据调整)
        HACK_MAX_WIND = 30000.0      # 风力归一化分母
        HACK_MAX_CEIL = 220.0        # 天花板安全高度 (超过此高度为0)
        HACK_MAX_GAP_WIDTH = 130.0   # [修改] 缝隙宽度阈值 (超过此宽度为0)
        HACK_MAX_GAP_OFFSET = 50.0   # 缝隙偏移分母

        # ============================================================
        # [修改] 核心逻辑：使用互斥判断 (if/elif)，防止不同场景数据打架
        # ============================================================
        
        if is_wind:
            # --- 风力处理 ---
            if "wind_force" in params:
                wf = params["wind_force"]
                raw_x = wf.get('x', 0.0)
                raw_y = wf.get('y', 0.0)
            else:
                raw_x, raw_y = 0.0, 0.0
            
            raw_mag = np.linalg.norm([raw_x, raw_y])

            # [新增] 修正方向：解决 0 风无方向问题
            # 如果原始模长极小(0风)，根据可视化结果，强制设为向右 (1.0, 0.0)
            if raw_mag < 1e-4:
                dir_x, dir_y = 1.0, 0.0
            else:
                dir_x = raw_x / raw_mag
                dir_y = raw_y / raw_mag
            
            # [新增] 偏移强度：将 [0, max] 映射到 [0.5, 1.0]
            # 让 0 风也变成有效的中等风信号 (0.5)，解决"0也挡风"的矛盾
            normalized_mag = 0.5 + 0.5 * (min(raw_mag, HACK_MAX_WIND) / HACK_MAX_WIND)
            
            phys_params_np[0] = dir_x * normalized_mag
            phys_params_np[1] = dir_y * normalized_mag
            phys_params_np[2] = normalized_mag
            
            # [新增] 显式清零其他参数 (防止数据污染)
            phys_params_np[3:] = 0.0

        elif is_ceiling:
            # --- 天花板处理 ---
            if "ceiling_height" in params:
                ch = params["ceiling_height"]
                # [修改] 反向归一化逻辑：
                # 220cm -> 1.0 - 1.0 = 0.0 (Mask掉，无影响)
                # 80cm  -> 1.0 - 0.36 = 0.64 (强信号)
                val = max(0.0, 1.0 - (ch / HACK_MAX_CEIL))
                
                # [新增] 极小值截断：防止 219cm 产生微弱噪音
                if val < 0.01: val = 0.0
                
                phys_params_np[3] = val
            
            # [新增] 显式清零其他参数
            phys_params_np[0:3] = 0.0
            phys_params_np[4:] = 0.0

        elif is_gap:
            # --- 缝隙处理 ---
            if "gap_width" in params:
                gw = params["gap_width"]
                # [修改] Gap Width 反向归一化 (逻辑同天花板)
                # 130cm (宽) -> 0.0 (无影响)
                # 40cm (窄)  -> >0.6 (强信号)
                val = max(0.0, 1.0 - (gw / HACK_MAX_GAP_WIDTH))
                if val < 0.01: val = 0.0
                phys_params_np[4] = val

            if "gap_offset" in params:
                go = params["gap_offset"]
                # Gap Offset 保持正向逻辑 (0就是0)
                phys_params_np[5] = go / HACK_MAX_GAP_OFFSET
            
            # [新增] 显式清零其他参数
            phys_params_np[0:4] = 0.0

        # ============================================================
        # [修改] 噪声注入逻辑：只对非 0 值加噪声 (保护 Mask)
        # ============================================================
        if hasattr(self, 'is_train') and self.is_train: 
            noise_scale = 0.05 
            noise = np.random.randn(*phys_params_np.shape) * noise_scale
            
            # [新增] 制作掩码，只对有值 (>1e-6) 的地方加噪声
            # 这样原本被我们设为 0 的参数（如 Wind 场景下的 Ceiling）会保持纯 0
            mask = np.abs(phys_params_np) > 1e-6
            
            phys_params_np[mask] += noise[mask]
            
            # Clamp
            phys_params_np = np.clip(phys_params_np, -1.0, 1.0)

        # 转 Tensor (保持不变)
        phys_params = torch.from_numpy(phys_params_np).float()

        # [修改] 生成虚拟 scene_cat (保持不变，为了兼容)
        scene_cat = torch.ones(1).float()

        motion_after = torch.from_numpy(motion_norm).float()
        motion_before = motion_after.clone()

        dummy_caption = "Varsapura"
        return {
            "motion_after": motion_after,
            "motion_before": motion_before,
            "length": len(motion_after),
            "phys_params": phys_params,
            "scene_cat": scene_cat, 
            "caption": dummy_caption,
        }