import numpy as np
import pickle
from os.path import join as pjoin

POS_enumerator = {
    'VERB': 0,
    'NOUN': 1,
    'DET': 2,
    'ADP': 3,
    'NUM': 4,
    'AUX': 5,
    'PRON': 6,
    'ADJ': 7,
    'ADV': 8,
    'Loc_VIP': 9,
    'Body_VIP': 10,
    'Obj_VIP': 11,
    'Act_VIP': 12,
    'Desc_VIP': 13,
    'OTHER': 14,
}

Loc_list = ('left', 'right', 'clockwise', 'counterclockwise', 'anticlockwise', 'forward', 'back', 'backward',
            'up', 'down', 'straight', 'curve')

Body_list = ('arm', 'chin', 'foot', 'feet', 'face', 'hand', 'mouth', 'leg', 'waist', 'eye', 'knee', 'shoulder', 'thigh')

Obj_List = ('stair', 'dumbbell', 'chair', 'window', 'floor', 'car', 'ball', 'handrail', 'baseball', 'basketball')

Act_list = ('walk', 'run', 'swing', 'pick', 'bring', 'kick', 'put', 'squat', 'throw', 'hop', 'dance', 'jump', 'turn',
            'stumble', 'dance', 'stop', 'sit', 'lift', 'lower', 'raise', 'wash', 'stand', 'kneel', 'stroll',
            'rub', 'bend', 'balance', 'flap', 'jog', 'shuffle', 'lean', 'rotate', 'spin', 'spread', 'climb')

Desc_list = ('slowly', 'carefully', 'fast', 'careful', 'slow', 'quickly', 'happy', 'angry', 'sad', 'happily',
             'angrily', 'sadly')

VIP_dict = {
    'Loc_VIP': Loc_list,
    'Body_VIP': Body_list,
    'Obj_VIP': Obj_List,
    'Act_VIP': Act_list,
    'Desc_VIP': Desc_list,
}


class WordVectorizer(object):
    def __init__(self, meta_root, prefix):
        vectors = np.load(pjoin(meta_root, '%s_data.npy'%prefix))
        words = pickle.load(open(pjoin(meta_root, '%s_words.pkl'%prefix), 'rb'))
        word2idx = pickle.load(open(pjoin(meta_root, '%s_idx.pkl'%prefix), 'rb'))
        self.word2vec = {w: vectors[word2idx[w]] for w in words}

    def _get_pos_ohot(self, pos):
        pos_vec = np.zeros(len(POS_enumerator))
        if pos in POS_enumerator:
            pos_vec[POS_enumerator[pos]] = 1
        else:
            pos_vec[POS_enumerator['OTHER']] = 1
        return pos_vec

    def __len__(self):
        return len(self.word2vec)

    # 修复了 __getitem__ 方法，添加了对 '/' 分隔符的处理，如果因为某种原因没有词性（比如Spicy处理的问题），则添加默认的词性
    def __getitem__(self, item):
        # print(f"--- WV_INPUT --- ITEM_START --- '{item}' (type: {type(item)}) --- ITEM_END ---")

        if not isinstance(item, str):
            print(f"    WV_ERROR: Item is not a string! Got: {type(item)}, value: '{item}'")
            raise TypeError(f"WordVectorizer expected a string item, but got {type(item)} with value '{item}'")

        word_raw = ""
        pos = ""

        if '/' not in item:
            # --- 紧急修复：给孤立的词添加默认词性 ---
            # print(f"    WV_WARNING: Item '{item}' does not contain '/' separator. Assigning default POS 'OTHER'.")
            word_raw = item  # 整个 item 被当作单词部分
            pos = 'OTHER'    # 分配默认词性 'OTHER'
            # -----------------------------------------
        else:
            try:
                # 仍然建议使用 split('/', 1) 来确保只分割一次
                parts = item.split('/', 1)
                if len(parts) == 2:
                    word_raw, pos = parts
                    if not pos.strip(): # 如果分割后 pos 是空字符串 (例如 item 是 "word/")
                        print(f"    WV_WARNING: Item '{item}' resulted in empty POS after split. Defaulting POS to 'OTHER'.")
                        pos = 'OTHER'
                else: # len(parts) == 1, 理论上 'if '/' not in item:' 已经处理了，但作为保险
                    print(f"    WV_WARNING: Item '{item}' after split resulted in unexpected parts: {parts}. Treating as isolated word.")
                    word_raw = item
                    pos = 'OTHER'

            except Exception as e: # 捕获其他可能的分割问题
                print(f"    WV_CRITICAL_ERROR: Unexpected error during item.split('/', 1) for item: '{item}'. Exception: {e}. Defaulting.")
                word_raw = item # 尝试将整个item作为单词
                pos = 'OTHER'   # 分配默认词性

        word = word_raw.replace('#SLASH#', '/') # 还原 #SLASH#

        # --- 原有的 VIP 词处理逻辑 (保持不变) ---
        if word in self.word2vec:
            word_vec = self.word2vec[word]
            vip_pos_override = None
            for key_vip, values_vip in VIP_dict.items(): # 确保 VIP_dict 可访问
                if word in values_vip:
                    vip_pos_override = key_vip
                    break
            if vip_pos_override is not None:
                final_pos_for_ohot = vip_pos_override
            else:
                final_pos_for_ohot = pos # 使用从 item 中得到的 (可能已设为'OTHER'的) pos
        else:
            word_vec = self.word2vec['unk']
            final_pos_for_ohot = 'OTHER' # 未知词的词性也设为 'OTHER'
        
        pos_vec = self._get_pos_ohot(final_pos_for_ohot) # 确保 _get_pos_ohot 能处理 'OTHER'
        return word_vec, pos_vec
