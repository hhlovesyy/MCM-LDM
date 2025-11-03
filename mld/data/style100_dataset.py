import numpy as np
import torch
from torch.utils import data
import random
from os.path import join as pjoin
import logging
import csv
import os
import json

# 初始化日志记录器
logger = logging.getLogger(__name__)

class Style100Dataset(data.Dataset):
    def __init__(
        self,
        mean,
        std,
        split_file,
        motion_dir,
        text_dir,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        style_dict_path,
        max_filter_length,
        min_filter_length,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        self.w_vectorizer = w_vectorizer

        try:
            # 随便取一个常见的 token 来查询动态的维度，但实际上目前100Style数据集是不适用 glove 的，而是后面直接用CLIP的tokenizer，吃原始文本即可，这里仅用作返回值和HumanML3D保持统一结构
            example_word_emb, example_pos_oh = self.w_vectorizer["the/DET"]
            self.glove_embedding_dim = example_word_emb.shape[0] # 获取向量的长度.300
            self.pos_one_hot_dim = example_pos_oh.shape[0]     # 获取 one-hot 向量的长度,15
        except Exception as e:
            logger.warning(f"Could not dynamically determine embedding dims, falling back to defaults. Error: {e}")
            self.glove_embedding_dim = 300
            self.pos_one_hot_dim = 27   # 足够大的备用值

        logger.info(f"Dynamically determined Glove dim: {self.glove_embedding_dim}, POS one-hot dim: {self.pos_one_hot_dim}")
        self.max_motion_length = max_motion_length  # 196
        self.min_motion_length = min_motion_length  # 40
        self.max_text_len = max_text_len  # 20
        self.unit_length = unit_length  # 4,
        self.mean = mean
        self.std = std  # 

        # --- [逻辑解读] 风格字典加载 ---
        # 从 style_dict.txt 加载文件 ID 到风格名称的映射。
        # 例如：'000123' -> 'proud'
        self.id_to_style = {}
        with open(style_dict_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 2:
                    file_id = parts[0]
                    style_name = parts[1].split('_')[0]
                    self.id_to_style[file_id] = style_name
        
        # 简化一下，style_desc_path 直接硬编码，后面再考虑重构，传参
        style_desc_path = "/root/autodl-tmp/MyRepository/MCM-LDM/datasets/100StyleDataset/Dataset_List.csv"

        # --- 2. [新代码] 加载纯粹的风格描述 ---
        self.style_to_desc = {}
        with open(style_desc_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader) # 跳过表头 (Style Name,Description,...)
            for row in reader:
                # key 是小写的 style name (e.g., 'aeroplane')
                # value 是纯粹的描述 (e.g., 'Both arms raised as wings...')
                style_name_key = row[0].lower().replace(" ", "") # 将 'Arms Above Head' 转为 'armsabovehead'
                description = row[1]
                self.style_to_desc[style_name_key] = description
        
        logger.info(f"Loaded {len(self.style_to_desc)} pure style descriptions.")


        id_list = []
        with open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        # --- [新增] 诊断与调试逻辑 ---
        logger.info("--- [DEBUG] Starting Diagnostic for Style100Dataset ---")
        logger.info(f"Filter settings: MIN={min_filter_length}, MAX={max_filter_length}")

        total_samples_before_filter = len(id_list)
        passed_samples_count = 0
        # 我们将把通过的样本信息 dump 到这个文件
        dump_file_path = "passed_samples_style100.txt" 
        with open(dump_file_path, "w") as dump_f:
            dump_f.write(f"# Samples passed filter: MIN={min_filter_length}, MAX={max_filter_length}\n")
            dump_f.write("# Format: [Sample Name] [Original Length]\n")

            self.data_dict = {}
            self.name_list = []
            count = 0
            maxdata = 10 if tiny else (100 if debug else 1e10)
            
            for name in id_list:
                # if count >= maxdata: break
                try:
                    motion_path = pjoin(motion_dir, name + ".npy")
                    motion = np.load(motion_path)
                    original_length = len(motion)  # shape:(31, 263)
                    # --- [核心风险点 1] 长度过滤 ---
                    # 这是关键的过滤步骤。如果 min/max_filter_length 设置不当，
                    # 结合数据分布图来看，这里可能会过滤掉大量本就稀有的 100Style 样本。
                    if not (min_filter_length <= original_length < max_filter_length): 
                        continue
                    
                    # --- [核心修复] 数据清洗 ---
                    text_data = []
                    with open(pjoin(text_dir, name + ".txt"), 'r') as f:
                        for line in f.readlines():
                            line_split = line.strip().split('#')
                            
                            # 1. 检查 caption 和 tokens 是否都存在
                            if len(line_split) < 2 or not line_split[1].strip():
                                logger.warning(f"Skipping empty or malformed token line in {name}.txt")
                                continue

                            tokens_str = line_split[1].strip()
                            raw_tokens = tokens_str.split(" ")
                            
                            # 2. 过滤掉所有格式不正确的 token,确保它们是 'word/POS' 格式
                            cleaned_tokens = [t for t in raw_tokens if '/' in t and len(t.split('/')) == 2]

                            # 3. 如果清洗后 token 列表为空，则跳过此行
                            if not cleaned_tokens:
                                logger.warning(f"No valid tokens found in line of {name}.txt after cleaning.")
                                continue

                            text_dict = {
                                "caption": line_split[0],
                                "tokens": cleaned_tokens # 使用清洗后的 tokens
                            }
                            text_data.append(text_dict)

                    # 如果该文件没有任何有效的文本行，则跳过整个样本
                    if not text_data:
                        logger.warning(f"Skipping sample {name} because it has no valid text entries.")
                        continue
                    # -------------------------

                    style_name = self.id_to_style.get(name)
                    if not style_name: continue
                    
                    # --- 样本完全有效，现在记录它 ---
                    passed_samples_count += 1
                    dump_f.write(f"{name} {original_length}\n")

                    self.data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text_list": text_data,
                        "style_name": style_name
                    }
                    self.name_list.append(name)
                    count += 1
                except Exception as e:
                    logger.warning(f"Skipping sample {name} due to an unexpected error: {e}")

        logger.info(f"Successfully loaded {len(self.name_list)} samples from Style100 after cleaning.")
        if not self.name_list: raise ValueError("No Style100 samples loaded. Check data and filter settings.")
        # --- [新增] 打印最终的诊断报告 ---
        pass_rate = (passed_samples_count / total_samples_before_filter) * 100 if total_samples_before_filter > 0 else 0
        logger.info("--- [DEBUG] Diagnostic Report for Style100Dataset ---")
        logger.info(f"Total samples in split file: {total_samples_before_filter}")
        logger.info(f"Samples that passed all filters: {passed_samples_count}")
        logger.info(f"Pass Rate: {pass_rate:.2f}%")
        logger.info(f"Details of passed samples have been dumped to: {dump_file_path}")  # Pass Rate: 92.44%
        logger.info("-----------------------------------------------------")

        # 1. 加载 GPT 扩写的风格描述
        self.gpt_descriptions = {}
        # [硬编码路径]
        descriptions_path = "/root/autodl-tmp/MyRepository/MCM-LDM/datasets/100StyleDataset/style_description.json"
        
        if os.path.exists(descriptions_path):
            with open(descriptions_path, 'r') as f:
                # 将 json 的 key 转为小写，以匹配我们的 style_name
                self.gpt_descriptions = {k.lower(): v for k, v in json.load(f).items()}
            logger.info(f"成功加载了 {len(self.gpt_descriptions)} 种风格的扩写描述。")
        else:
            logger.warning(f"扩写的风格描述文件未找到: {descriptions_path}. 将只使用基础模板。")
            
        # 2. 剔除已知的“烂完了”的风格样本
        styles_to_exclude = [
            "whirlarms",
            "widelegs",
            "wigglehips",
            "wildarms",
            "wildlegs",
            "zombie"
        ]
        # 将风格名统一为小写，去除空格，以便比较
        styles_to_exclude = [s.lower().replace(" ", "") for s in styles_to_exclude]
        
        logger.info(f"将要剔除以下风格的样本: {styles_to_exclude}")
        initial_sample_count = len(self.name_list)

        self.style_to_names = {}
        for name in self.name_list:
            style_name = self.data_dict[name]["style_name"].lower().replace(" ", "")
            if style_name not in self.style_to_names:
                self.style_to_names[style_name] = []
            self.style_to_names[style_name].append(name)
        
        # 剔除坏样本
        new_name_list = []
        for name in self.name_list:
            style_key = self.id_to_style[name].lower().replace(" ", "")
            if style_key not in styles_to_exclude:
                new_name_list.append(name)
                
        self.name_list = new_name_list
        final_sample_count = len(self.name_list)
        logger.info(f"样本剔除完成。原始样本数: {initial_sample_count}, 剔除后剩余: {final_sample_count}")
        
        # 2025.11.3 新增样本的id信息，用于正确计算align_loss中的对比损失
        self.style_list = sorted(list(self.style_to_names.keys()))
        self.style_to_id = {s_name: i for i, s_name in enumerate(self.style_list)}
        logger.info(f"Created mapping for {len(self.style_to_id)} unique styles for contrastive loss.")
        
        self.nfeats = self.data_dict[self.name_list[0]]['motion'].shape[1]  
    
    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):  # item：dtype('int64')，取样本的索引
        # 它现在接收的是已经清洗过的、绝对安全的数据
        name = self.name_list[item]  # 037644
        data = self.data_dict[name] 
        motion, m_length = data["motion"], data["length"]  # motion：(176, 263)， length：176， 特征维度：263

        style_name = data["style_name"].lower().replace(" ", "")  # 'whirlarms'
        final_caption = ""
        style_id = self.style_to_id[style_name]

        if random.random() < 0.5 and style_name in self.gpt_descriptions and self.gpt_descriptions[style_name]:
            final_caption = random.choice(self.gpt_descriptions[style_name])
        else:
            prompt_templates = [
                '{}', # 纯单词
                'a {} style',
                'style of {}'
            ]
            # 这里用 choices 带权重更严谨，但 random.choice 也足够简单有效
            chosen_template = random.choice(prompt_templates)
            original_style_name_for_prompt = self.id_to_style[name]
            final_caption = chosen_template.format(original_style_name_for_prompt)

        if item < 5:  # 仅打印前5个样本以避免日志过多
            print(f"[Debug] Sample {name}: Using caption: '{final_caption}'")
        
        # 不用glove，因为CLIP有tokenizer等的处理
        tokens = ["unk/OTHER"] * (self.max_text_len + 2)
        sent_len = 0 
        
        word_embeddings = np.zeros((self.max_text_len + 2, self.glove_embedding_dim)) # shape:(22, 300)
        pos_one_hots = np.zeros((self.max_text_len + 2, self.pos_one_hot_dim))  # shape:(22, 15)
        
        # --- [逻辑解读] 动作裁剪与标准化 ---
        m_length_cropped = min(m_length, self.max_motion_length)  # 176
        m_length_cropped = (m_length_cropped // self.unit_length) * self.unit_length
        if m_length_cropped < self.min_motion_length:
             m_length_cropped = self.min_motion_length
        
        idx = random.randint(0, m_length - m_length_cropped)
        motion_cropped = motion[idx:idx + m_length_cropped]  # 100Style的很多动作都比较长，基本都要被裁剪，还是随机裁剪，可能会导致数据集的质量低，这个我们后面再优化
        motion = motion_cropped

        motion = (motion - self.mean) / self.std

        if np.any(np.isnan(motion)):
            logger.warning(f"NaN detected in motion sample {name}. Resampling...")
            return self.__getitem__(np.random.randint(0, len(self.name_list)))

        return {
            "word_embeddings": word_embeddings,  # (22, 300)：zero
            "pos_one_hots": pos_one_hots,  # shape:(22, 15)： zero
            "caption": final_caption,  # 'Style of Areoplane'，
            "sent_len": sent_len,  # 0
            "motion": motion,  # (176, 263)
            "m_length": m_length_cropped,  # 176
            "tokens": "_".join(tokens), # 长度22，全是'unk/OTHER'，应该也是为了简化吧
            "style_id": style_id,  
            "source": "style100"  # 来源标识
        }