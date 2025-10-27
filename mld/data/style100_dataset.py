import numpy as np
import torch
from torch.utils import data
import random
from os.path import join as pjoin
import logging
import csv

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
        self.unit_length = unit_length  # 4,【QUESTION】这是什么？self.max_text_len和self.unit_length在我们的100Style数据集里有用么？本来是什么
        self.mean = mean
        self.std = std  # 【QUESTION】这两个目前都是使用HumanML3D的mean和std，这是否make sense？因为目前MCM-LDM是认识100Style的style motion做风格迁移的，所以是不是用humanML3D的mean和std做归一化也可以？这样也比较方便，请你再次仔细评估。

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
        self.nfeats = self.data_dict[self.name_list[0]]['motion'].shape[1]  
    
    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):  # item：dtype('int64')，取样本的索引
        # 它现在接收的是已经清洗过的、绝对安全的数据
        name = self.name_list[item]  # 037644
        data = self.data_dict[name] 
        motion, m_length = data["motion"], data["length"]  # motion：(176, 263)， length：176， 特征维度：263

        style_name = data["style_name"].lower()  # 'whirlarms'
        # --- 直接从映射中获取纯粹的风格描述,这是因为100Style本来动作的text文本部分包含content，我们需要纯粹的style ---
        # ---【QUESTION】是否需要扩充style的风格描述？比如显式把Style的类型也放进去，这样让模型学的更快？能学的更好吗？
        style_name_key = style_name.lower().replace(" ", "")
        final_caption = self.style_to_desc.get(style_name_key)  # 'Windmill arms'，这个风格太短了，不一定靠谱，这能让网络学到东西么？

        # [健壮性检查] 如果在 csv 中找不到对应的风格，提供一个回退方案
        if final_caption is None:
            logger.warning(f"Pure style description for '{style_name_key}' not found. Falling back to simple style name.")
            final_caption = f"in a {style_name} style"
        
        # [简化] 我们不再需要为 glove 构建复杂的 tokens
        tokens = ["unk/OTHER"] * (self.max_text_len + 2)
        sent_len = 0 # 设为0，因为我们不使用它
        
        # [简化] 创建空的占位符以保持返回字典的结构不变
        word_embeddings = np.zeros((self.max_text_len + 2, self.glove_embedding_dim)) # shape:(22, 300)
        pos_one_hots = np.zeros((self.max_text_len + 2, self.pos_one_hot_dim))  # shape:(22, 15)
        
        # --- [逻辑解读] 动作裁剪与标准化 ---
        m_length_cropped = min(m_length, self.max_motion_length)  # 176
        m_length_cropped = (m_length_cropped // self.unit_length) * self.unit_length
        if m_length_cropped < self.min_motion_length:
             m_length_cropped = self.min_motion_length
        
        idx = random.randint(0, m_length - m_length_cropped)
        motion_cropped = motion[idx:idx + m_length_cropped]  # 100Style的很多动作都比较长，基本都要被裁剪，还是随机裁剪，会不会导致数据集的质量非常低
        motion = motion_cropped
        # 标准化
        motion = (motion - self.mean) / self.std

        # --- [代码加固] NaN 值检查 ---
        if np.any(np.isnan(motion)):
            logger.warning(f"NaN detected in motion sample {name}. Resampling...")
            return self.__getitem__(np.random.randint(0, len(self.name_list)))

        return {
            "word_embeddings": word_embeddings,  # (22, 300)：zero
            "pos_one_hots": pos_one_hots,  # shape:(22, 15)： zero
            "caption": final_caption,  # 'Windmill arms'，【Question】这玩意让一个只会文/图对齐的clip来生成word embedding，是不是太难了，cross attention能很好的注意么？
            "sent_len": sent_len,  # 0
            "motion": motion,  # (176, 263)
            "m_length": m_length_cropped,  # 176
            "tokens": "_".join(tokens), # 长度22，全是'unk/OTHER'，应该也是为了简化吧
            "source": "style100"  # 来源标识
        }