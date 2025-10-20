# mld/data/style100_dataset.py

import numpy as np
import torch
from torch.utils import data
import random
from os.path import join as pjoin
import logging

# 初始化日志记录器
logger = logging.getLogger(__name__)

class Style100Dataset(data.Dataset):
    def __init__(
        self,
        mean,
        std,
        split_file,
        motion_dir,  # 应该是/root/autodl-tmp/MyRepository/MCM-LDM/datasets/100StyleDataset/new_joint_vecs
        text_dir,  # texts 文件夹的路径
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        style_dict_path, # 新增参数：Style_name_dict.txt 的路径
        max_filter_length, # [新增] 从配置中传入最大过滤长度
        min_filter_length, # [新增] 从配置中传入最小过滤长度
        tiny=False,
        debug=False,
        **kwargs,
    ):
        self.w_vectorizer = w_vectorizer
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        self.mean = mean
        self.std = std

        # --- 新的核心加载逻辑 ---
        # 1. 构建 ID -> Style Name 的映射
        self.id_to_style = {} # 比如 {'M0001': 'Aeroplane', 'M0002': 'Old', ...}
        with open(style_dict_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 2:
                    file_id = parts[0]
                    # 从 'Aeroplane_BR_00.bvh' 中提取 'Aeroplane'
                    style_name = parts[1].split('_')[0]
                    self.id_to_style[file_id] = style_name
        logger.info(f"Loaded {len(self.id_to_style)} mappings from style dictionary.")

        # 2. 从 split 文件 (train.txt/val.txt) 加载 ID 列表
        id_list = []  # 比如 ['M0001', 'M0002', ...]
        with open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        # 3. 遍历 ID 列表，加载数据
        self.data_dict = {}
        self.name_list = []
        count = 0
        maxdata = 10 if tiny else(100 if debug else 1e10)
        
        for name in id_list:
            if count >= maxdata:
                break
            try:
                motion_path = pjoin(motion_dir, name + ".npy")  # note：检查motion_dir对不对
                motion = np.load(motion_path)
                # 过滤不符合长度要求的动作
                if not (min_filter_length <= len(motion) < max_filter_length):
                    continue
                # 读取详细的内容描述文本
                text_data = []
                with open(pjoin(text_dir, name + ".txt"), 'r') as f:
                    for line in f.readlines():
                        line_split = line.strip().split('#')
                        text_dict = {
                            "caption": line_split[0],
                            "tokens": line_split[1].split(" ")
                        }
                        # 在 100Style 中，所有文本都描述整个动作
                        text_data.append(text_dict)
                
                style_name = self.id_to_style.get(name)
                if not style_name: continue
                self.data_dict[name] = {
                    "motion": motion,
                    "length": len(motion),
                    "text_list": text_data, # 存储内容描述列表
                    "style_name": style_name # 存储风格名
                }
                self.name_list.append(name)
                count += 1
            except Exception as e:
                logger.warning(f"Skipping sample {name} due to error: {e}")

        logger.info(f"Successfully loaded {len(self.name_list)} samples from Style100.")
        if not self.name_list: raise ValueError("No Style100 samples loaded.")
        self.nfeats = self.data_dict[self.name_list[0]]['motion'].shape[1]  # 应该是263维度
    
    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        motion, m_length = data["motion"], data["length"]
        # --- 文本处理 (V2 - 精确 Token 合并版本) ---
    
        # 1. 随机选择一个内容描述及其【预处理好的 Tokens】
        content_text_data = random.choice(data["text_list"])
        content_caption = content_text_data["caption"]
        content_tokens = content_text_data["tokens"] # -> ['the/DET', 'person/NOUN', 'is/AUX', ...]

        # 2. 获取风格名并创建【风格 Tokens】
        style_name = data["style_name"].lower() # -> "proud"
        # 我们为风格 prompt 创建自己的 token 序列
        style_prompt_tokens = [
            ",/OTHER", "in/ADP", "a/DET", f"{style_name}/NOUN", "style/NOUN"
        ]

        # 3. *** 核心变更：智能合并 Token 序列 ***
        # 将内容 tokens 和风格 tokens 拼接起来
        # 同时加上开始和结束标记
        tokens = ["sos/OTHER"] + content_tokens + style_prompt_tokens + ["eos/OTHER"]
        sent_len = len(tokens)

        # 4. 重新构建一个用于调试的 caption (可选，但推荐)
        final_caption = content_caption + f", in a {style_name} style"

        # 5. 后续的填充、向量化逻辑完全不变
        if sent_len > self.max_text_len + 2:
            tokens = tokens[:self.max_text_len + 2]
            sent_len = len(tokens)
        
        # [重要] 截断后要确保最后一个 token 是 'eos/OTHER'
        if tokens[-1] != 'eos/OTHER':
            tokens[-1] = 'eos/OTHER'

        tokens = tokens + ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)

        pos_one_hots, word_embeddings = [], []
        for token in tokens:
            try:
                word_emb, pos_oh = self.w_vectorizer[token]
            except KeyError:
                word_emb, pos_oh = self.w_vectorizer["unk/OTHER"]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)
        
        # --- 动作处理 (与之前相同) ---
        m_length_cropped = (m_length // self.unit_length) * self.unit_length
        if m_length_cropped < self.unit_length: m_length_cropped = self.unit_length
        idx = random.randint(0, len(motion) - m_length_cropped)
        motion = motion[idx:idx + m_length_cropped]
        motion = (motion - self.mean) / self.std

        # --- 返回值 ---
        # 现在，100Style 的 __getitem__ 返回值与 HumanML3D 的完全一致
        # 我们甚至不需要在 `mixed_datamodule` 中做猴子补丁了
        return (
            word_embeddings,       # (22, 300) 的数组，代表【文本语义】
            pos_one_hots,          # (22, 52) 的数组，代表【文本语法】
            final_caption,         # "a person walks proudly, in a proud style"，供调试和参考
            sent_len,              # 10，文本的【真实长度】
            motion,                # (84, 263) 的数组，代表【动作数据】
            m_length_cropped,      # 84，动作的【真实长度】
            "_".join(tokens),      # 所有 token 的字符串拼接，供调试
        )