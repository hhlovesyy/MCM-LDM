# import codecs as cs
# import os
# import random
# from os.path import join as pjoin

# import numpy as np
# import spacy
# import torch
# from rich.progress import track
# from torch.utils import data
# from torch.utils.data._utils.collate import default_collate
# from tqdm import tqdm

# from ..utils.get_opt import get_opt
# from ..utils.word_vectorizer import WordVectorizer


# # import spacy
# def collate_fn(batch):
#     batch.sort(key=lambda x: x[3], reverse=True)
#     return default_collate(batch)


# """For use of training text-2-motion generative model"""


# class Text2MotionDataset(data.Dataset):

#     def __init__(self, opt, mean, std, split_file, w_vectorizer):
#         self.opt = opt
#         self.w_vectorizer = w_vectorizer
#         self.max_length = 20
#         self.pointer = 0
#         min_motion_len = 40 if self.opt.dataset_name == "t2m" else 24

#         joints_num = opt.joints_num

#         data_dict = {}
#         id_list = []
#         with cs.open(split_file, "r") as f:
#             for line in f.readlines():
#                 id_list.append(line.strip())

#         new_name_list = []
#         length_list = []
#         for name in tqdm(id_list):
#             try:
#                 motion = np.load(pjoin(opt.motion_dir, name + ".npy"))
#                 if (len(motion)) < min_motion_len or (len(motion) >= 200):
#                     continue
#                 text_data = []
#                 flag = False
#                 with cs.open(pjoin(opt.text_dir, name + ".txt")) as f:
#                     for line in f.readlines():
#                         text_dict = {}
#                         line_split = line.strip().split("#")
#                         caption = line_split[0]
#                         tokens = line_split[1].split(" ")
#                         f_tag = float(line_split[2])
#                         to_tag = float(line_split[3])
#                         f_tag = 0.0 if np.isnan(f_tag) else f_tag
#                         to_tag = 0.0 if np.isnan(to_tag) else to_tag

#                         text_dict["caption"] = caption
#                         text_dict["tokens"] = tokens
#                         if f_tag == 0.0 and to_tag == 0.0:
#                             flag = True
#                             text_data.append(text_dict)
#                         else:
#                             try:
#                                 n_motion = motion[int(f_tag * 20):int(to_tag *
#                                                                       20)]
#                                 if (len(n_motion)) < min_motion_len or (
#                                         len(n_motion) >= 200):
#                                     continue
#                                 new_name = (
#                                     random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
#                                     "_" + name)
#                                 while new_name in data_dict:
#                                     new_name = (random.choice(
#                                         "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
#                                                 name)
#                                 data_dict[new_name] = {
#                                     "motion": n_motion,
#                                     "length": len(n_motion),
#                                     "text": [text_dict],
#                                 }
#                                 new_name_list.append(new_name)
#                                 length_list.append(len(n_motion))
#                             except:
#                                 print(line_split)
#                                 print(line_split[2], line_split[3], f_tag,
#                                       to_tag, name)
#                                 # break

#                 if flag:
#                     data_dict[name] = {
#                         "motion": motion,
#                         "length": len(motion),
#                         "text": text_data,
#                     }
#                     new_name_list.append(name)
#                     length_list.append(len(motion))
#             except:
#                 # Some motion may not exist in KIT dataset
#                 pass

#         name_list, length_list = zip(
#             *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

#         if opt.is_train:
#             # root_rot_velocity (B, seq_len, 1)
#             std[0:1] = std[0:1] / opt.feat_bias
#             # root_linear_velocity (B, seq_len, 2)
#             std[1:3] = std[1:3] / opt.feat_bias
#             # root_y (B, seq_len, 1)
#             std[3:4] = std[3:4] / opt.feat_bias
#             # ric_data (B, seq_len, (joint_num - 1)*3)
#             std[4:4 + (joints_num - 1) * 3] = std[4:4 +
#                                                   (joints_num - 1) * 3] / 1.0
#             # rot_data (B, seq_len, (joint_num - 1)*6)
#             std[4 + (joints_num - 1) * 3:4 +
#                 (joints_num - 1) * 9] = (std[4 + (joints_num - 1) * 3:4 +
#                                              (joints_num - 1) * 9] / 1.0)
#             # local_velocity (B, seq_len, joint_num*3)
#             std[4 + (joints_num - 1) * 9:4 + (joints_num - 1) * 9 +
#                 joints_num * 3] = (std[4 + (joints_num - 1) * 9:4 +
#                                        (joints_num - 1) * 9 + joints_num * 3] /
#                                    1.0)
#             # foot contact (B, seq_len, 4)
#             std[4 + (joints_num - 1) * 9 + joints_num * 3:] = (
#                 std[4 +
#                     (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias)

#             assert 4 + (joints_num -
#                         1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
#             np.save(pjoin(opt.meta_dir, "mean.npy"), mean)
#             np.save(pjoin(opt.meta_dir, "std.npy"), std)

#         self.mean = mean
#         self.std = std
#         self.length_arr = np.array(length_list)
#         self.data_dict = data_dict
#         self.name_list = name_list
#         self.reset_max_len(self.max_length)

#     def reset_max_len(self, length):
#         assert length <= self.opt.max_motion_length
#         self.pointer = np.searchsorted(self.length_arr, length)
#         print("Pointer Pointing at %d" % self.pointer)
#         self.max_length = length

#     def inv_transform(self, data):
#         return data * self.std + self.mean

#     def __len__(self):
#         return len(self.data_dict) - self.pointer

#     def __getitem__(self, item):
#         idx = self.pointer + item
#         data = self.data_dict[self.name_list[idx]]
#         motion, m_length, text_list = data["motion"], data["length"], data[
#             "text"]
#         # Randomly select a caption
#         text_data = random.choice(text_list)
#         caption, tokens = text_data["caption"], text_data["tokens"]

#         if len(tokens) < self.opt.max_text_len:
#             # pad with "unk"
#             tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
#             sent_len = len(tokens)
#             tokens = tokens + ["unk/OTHER"
#                                ] * (self.opt.max_text_len + 2 - sent_len)
#         else:
#             # crop
#             tokens = tokens[:self.opt.max_text_len]
#             tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
#             sent_len = len(tokens)
#         pos_one_hots = []
#         word_embeddings = []
#         for token in tokens:
#             word_emb, pos_oh = self.w_vectorizer[token]
#             pos_one_hots.append(pos_oh[None, :])
#             word_embeddings.append(word_emb[None, :])
#         pos_one_hots = np.concatenate(pos_one_hots, axis=0)
#         word_embeddings = np.concatenate(word_embeddings, axis=0)

#         len_gap = (m_length - self.max_length) // self.opt.unit_length

#         if self.opt.is_train:
#             if m_length != self.max_length:
#                 # print("Motion original length:%d_%d"%(m_length, len(motion)))
#                 if self.opt.unit_length < 10:
#                     coin2 = np.random.choice(["single", "single", "double"])
#                 else:
#                     coin2 = "single"
#                 if len_gap == 0 or (len_gap == 1 and coin2 == "double"):
#                     m_length = self.max_length
#                     idx = random.randint(0, m_length - self.max_length)
#                     motion = motion[idx:idx + self.max_length]
#                 else:
#                     if coin2 == "single":
#                         n_m_length = self.max_length + self.opt.unit_length * len_gap
#                     else:
#                         n_m_length = self.max_length + self.opt.unit_length * (
#                             len_gap - 1)
#                     idx = random.randint(0, m_length - n_m_length)
#                     motion = motion[idx:idx + self.max_length]
#                     m_length = n_m_length
#                 # print(len_gap, idx, coin2)
#         else:
#             if self.opt.unit_length < 10:
#                 coin2 = np.random.choice(["single", "single", "double"])
#             else:
#                 coin2 = "single"

#             if coin2 == "double":
#                 m_length = (m_length // self.opt.unit_length -
#                             1) * self.opt.unit_length
#             elif coin2 == "single":
#                 m_length = (m_length //
#                             self.opt.unit_length) * self.opt.unit_length
#             idx = random.randint(0, len(motion) - m_length)
#             motion = motion[idx:idx + m_length]
#         "Z Normalization"
#         motion = (motion - self.mean) / self.std

#         return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length

# ###########################
# """For use of training text motion matching model, and evaluations"""
# #########################


# # # 最终使用的类

# # class Text2MotionDatasetV2(data.Dataset):

# #     def __init__(
# #         self,
# #         mean,
# #         std,
# #         split_file,
# #         w_vectorizer,
# #         max_motion_length,
# #         min_motion_length,
# #         max_text_len,
# #         unit_length,
# #         motion_dir,
# #         text_dir,
# #         style_text_dir,
# #         tiny=False,
# #         debug=False,
# #         progress_bar=True,
# #         **kwargs,
# #     ):
# #         # for rot dataset
# #         rot_motion_dir = "/root/autodl-tmp/sc_motion/datasets/humanml3d/rot_joints"
# #         self.w_vectorizer = w_vectorizer
# #         self.max_length = 20
# #         self.pointer = 0
# #         self.max_motion_length = max_motion_length
# #         # min_motion_len = 40 if dataset_name =='t2m' else 24
# #         self.min_motion_length = min_motion_length
# #         self.max_text_len = max_text_len
# #         self.unit_length = unit_length

# #         data_dict = {}
# #         id_list = []
# #         with cs.open(split_file, "r") as f:
# #             for line in f.readlines():
# #                 id_list.append(line.strip())
# #         self.id_list = id_list

# #         if tiny or debug:
# #             progress_bar = False
# #             maxdata = 10 if tiny else 100
# #         else:
# #             maxdata = 1e10

# #         if progress_bar:
# #             enumerator = enumerate(
# #                 track(
# #                     id_list,
# #                     f"Loading HumanML3D {split_file.split('/')[-1].split('.')[0]}",
# #                 ))
# #         else:
# #             enumerator = enumerate(id_list)
# #         count = 0
# #         bad_count = 0
# #         new_name_list = []
# #         length_list = []
# #         for i, name in enumerator:
# #             if count > maxdata:
# #                 break
# #             try:
# #                 if not os.path.exists(pjoin(rot_motion_dir, name + ".npy")):
# #                     continue
# #                 motion_rot = np.load(pjoin(rot_motion_dir, name + ".npy"))
# #                 motion = np.load(pjoin(motion_dir, name + ".npy"))
# #                 if (len(motion)) < self.min_motion_length or (len(motion) >=
# #                                                               200):
# #                     bad_count += 1
# #                     continue
# #                 text_data = []
# #                 style_text_data = []
# #                 flag = False


# #                 with cs.open(pjoin(text_dir, name + ".txt")) as f:
# #                     with cs.open(pjoin(style_text_dir, name + ".txt")) as g:
# #                         style_text_dict = {}
# #                         style = g.readline().strip().split("#")
# #                         style_caption = style[0]
# #                         style_tokens = style[1].split(" ")
# #                         style_tokens = style_tokens[:-1]
# #                         style_label = style[2]

# #                         style_text_dict["caption"] = style_caption


# #                         style_text_data.append(style_text_dict)
# #                         for line in f.readlines():
# #                             text_dict = {}
# #                             line_split = line.strip().split("#")
# #                             caption = line_split[0]
# #                             tokens = line_split[1].split(" ")
# #                             #这里注释掉，去掉style token
# #                             # tokens = tokens+style_tokens
# #                             f_tag = float(line_split[2])
# #                             to_tag = float(line_split[3])
# #                             f_tag = 0.0 if np.isnan(f_tag) else f_tag
# #                             to_tag = 0.0 if np.isnan(to_tag) else to_tag

# #                             text_dict["caption"] = caption
# #                             text_dict["tokens"] = tokens
# #                             if f_tag == 0.0 and to_tag == 0.0:
# #                                 flag = True
# #                                 text_data.append(text_dict)
# #                             else:
# #                                 try:
# #                                     n_motion = motion[int(f_tag * 20):int(to_tag *
# #                                                                         20)]
# #                                     n_motion_rot = motion_rot[int(f_tag * 20):int(to_tag *
# #                                                                         20)]
# #                                     if (len(n_motion)
# #                                         ) < self.min_motion_length or (
# #                                             (len(n_motion) >= 200)):
# #                                         continue
# #                                     new_name = (
# #                                         random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
# #                                         "_" + name)
# #                                     while new_name in data_dict:
# #                                         new_name = (random.choice(
# #                                             "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
# #                                                     name)
# #                                     data_dict[new_name] = {
# #                                         "motion": n_motion,
# #                                         "motion_rot": n_motion_rot,
# #                                         "length": len(n_motion),
# #                                         "text": [text_dict],
# #                                         "style_text": [style_text_dict],
# #                                     }
# #                                     new_name_list.append(new_name)
# #                                     length_list.append(len(n_motion))
# #                                 except:
# #                                     # None
# #                                     print(line_split)
# #                                     print(line_split[2], line_split[3], f_tag,
# #                                         to_tag, name)
# #                                     # break


# #                 if flag:
# #                     data_dict[name] = {
# #                         "motion": motion,
# #                         "motion_rot": motion_rot[:-1,...],
# #                         "length": len(motion),
# #                         "text": text_data,
# #                         "style_text": [style_text_dict],
# #                     }
# #                     new_name_list.append(name)
# #                     length_list.append(len(motion))
# #                     # print(count)
# #                     count += 1
# #                     # print(name)
# #             except:
# #                 pass

# #         name_list, length_list = zip(
# #             *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

# #         self.mean = mean
# #         self.std = std
# #         self.length_arr = np.array(length_list)
# #         self.data_dict = data_dict
# #         self.nfeats = motion.shape[1]
# #         self.name_list = name_list
# #         self.reset_max_len(self.max_length)

# #     def reset_max_len(self, length):
# #         assert length <= self.max_motion_length
# #         self.pointer = np.searchsorted(self.length_arr, length)
# #         print("Pointer Pointing at %d" % self.pointer)
# #         self.max_length = length

# #     def inv_transform(self, data):
# #         return data * self.std + self.mean

# #     def __len__(self):
# #         return len(self.name_list) - self.pointer

# #     def __getitem__(self, item):
# #         idx = self.pointer + item
# #         data = self.data_dict[self.name_list[idx]]
# #         motion, motion_rot, m_length, text_list, style_text_list = data["motion"], data["motion_rot"],data["length"], data[
# #             "text"], data["style_text"]
# #         # 随机选一个caption
# #         # Randomly select a caption
# #         #
# #         text_data = random.choice(text_list)
# #         caption, tokens = text_data["caption"], text_data["tokens"]
# #         style_caption = style_text_list[0]["caption"]

# #         if len(tokens) < self.max_text_len:
# #             # pad with "unk"
# #             tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
# #             sent_len = len(tokens)
# #             tokens = tokens + ["unk/OTHER"
# #                                ] * (self.max_text_len + 2 - sent_len)
# #         else:
# #             # crop
# #             tokens = tokens[:self.max_text_len]
# #             tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
# #             sent_len = len(tokens)
# #         pos_one_hots = []
# #         word_embeddings = []
# #         for token in tokens:
# #             word_emb, pos_oh = self.w_vectorizer[token]
# #             pos_one_hots.append(pos_oh[None, :])
# #             word_embeddings.append(word_emb[None, :])
# #         pos_one_hots = np.concatenate(pos_one_hots, axis=0)
# #         word_embeddings = np.concatenate(word_embeddings, axis=0)

# #         # Crop the motions in to times of 4, and introduce small variations
# #         if self.unit_length < 10:
# #             coin2 = np.random.choice(["single", "single", "double"])
# #         else:
# #             coin2 = "single"

# #         if coin2 == "double":
# #             m_length = (m_length // self.unit_length - 1) * self.unit_length
# #         elif coin2 == "single":
# #             m_length = (m_length // self.unit_length) * self.unit_length
# #         idx = random.randint(0, len(motion) - m_length)
# #         motion = motion[idx:idx + m_length]
# #         motion_rot = motion_rot[idx:idx + m_length]
# #         "Z Normalization"
# #         motion = (motion - self.mean) / self.std

# #         # # padding
# #         # if m_length < self.max_motion_length:
# #         #     motion = np.concatenate(
# #         #         [
# #         #             motion,
# #         #             np.zeros((self.max_motion_length - m_length, motion.shape[1])),
# #         #         ],
# #         #         axis=0,
# #         #     )
# #         # print(word_embeddings.shape, motion.shape, m_length)
# #         # print(tokens)

# #         # debug check nan
# #         if np.any(np.isnan(motion)):
# #             raise ValueError("nan in motion")
# # #这里concat到一起了，style_text保存在
# #         return (
# #             word_embeddings,
# #             pos_one_hots,
# #             caption,
# #             sent_len,
# #             motion,
# #             m_length,
# #             "_".join(tokens),
# #             style_caption,
# #             motion_rot,
# #         )
# #         # return caption, motion, m_length


# # back储存
# class Text2MotionDatasetV2(data.Dataset):

#     def __init__(
#         self,
#         mean,
#         std,
#         split_file,
#         w_vectorizer,
#         max_motion_length,
#         min_motion_length,
#         max_text_len,
#         unit_length,
#         motion_dir,
#         text_dir,
#         style_text_dir,
#         tiny=False,
#         debug=False,
#         progress_bar=True,
#         **kwargs,
#     ):
#         self.w_vectorizer = w_vectorizer
#         self.max_length = 20
#         self.pointer = 0
#         self.max_motion_length = max_motion_length
#         # min_motion_len = 40 if dataset_name =='t2m' else 24
#         self.min_motion_length = min_motion_length
#         self.max_text_len = max_text_len
#         self.unit_length = unit_length

#         data_dict = {}
#         id_list = []
#         with cs.open(split_file, "r") as f:
#             for line in f.readlines():
#                 id_list.append(line.strip())
#         self.id_list = id_list

#         if tiny or debug:
#             progress_bar = False
#             maxdata = 10 if tiny else 100
#         else:
#             maxdata = 1e10

#         if progress_bar:
#             enumerator = enumerate(
#                 track(
#                     id_list,
#                     f"Loading HumanML3D {split_file.split('/')[-1].split('.')[0]}",
#                 ))
#         else:
#             enumerator = enumerate(id_list)
#         count = 0
#         bad_count = 0
#         new_name_list = []
#         length_list = []
#         for i, name in enumerator:
#             if count > maxdata:
#                 break
#             try:
#                 motion = np.load(pjoin(motion_dir, name + ".npy"))
#                 if (len(motion)) < self.min_motion_length or (len(motion) >=
#                                                               200):
#                     bad_count += 1
#                     continue
#                 text_data = []
#                 style_text_data = []
#                 flag = False


#                 with cs.open(pjoin(text_dir, name + ".txt")) as f:
#                     with cs.open(pjoin(style_text_dir, name + ".txt")) as g:
#                         style_text_dict = {}
#                         style = g.readline().strip().split("#")
#                         style_caption = style[0]
#                         style_tokens = style[1].split(" ")
#                         style_tokens = style_tokens[:-1]
#                         style_label = style[2]

#                         style_text_dict["caption"] = style_caption


#                         style_text_data.append(style_text_dict)
#                         for line in f.readlines():
#                             text_dict = {}
#                             line_split = line.strip().split("#")
#                             caption = line_split[0]
#                             tokens = line_split[1].split(" ")
#                             #这里注释掉，去掉style token
#                             # tokens = tokens+style_tokens
#                             f_tag = float(line_split[2])
#                             to_tag = float(line_split[3])
#                             f_tag = 0.0 if np.isnan(f_tag) else f_tag
#                             to_tag = 0.0 if np.isnan(to_tag) else to_tag

#                             text_dict["caption"] = caption
#                             text_dict["tokens"] = tokens
#                             if f_tag == 0.0 and to_tag == 0.0:
#                                 flag = True
#                                 text_data.append(text_dict)
#                             else:
#                                 try:
#                                     n_motion = motion[int(f_tag * 20):int(to_tag *
#                                                                         20)]
#                                     if (len(n_motion)
#                                         ) < self.min_motion_length or (
#                                             (len(n_motion) >= 200)):
#                                         continue
#                                     new_name = (
#                                         random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
#                                         "_" + name)
#                                     while new_name in data_dict:
#                                         new_name = (random.choice(
#                                             "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
#                                                     name)
#                                     data_dict[new_name] = {
#                                         "motion": n_motion,
#                                         "length": len(n_motion),
#                                         "text": [text_dict],
#                                         "style_text": [style_text_dict],
#                                     }
#                                     new_name_list.append(new_name)
#                                     length_list.append(len(n_motion))
#                                 except:
#                                     # None
#                                     print(line_split)
#                                     print(line_split[2], line_split[3], f_tag,
#                                         to_tag, name)
#                                     # break


#                 if flag:
#                     data_dict[name] = {
#                         "motion": motion,
#                         "length": len(motion),
#                         "text": text_data,
#                         "style_text": [style_text_dict],
#                     }
#                     new_name_list.append(name)
#                     length_list.append(len(motion))
#                     # print(count)
#                     count += 1
#                     # print(name)
#             except:
#                 pass

#         name_list, length_list = zip(
#             *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

#         self.mean = mean
#         self.std = std
#         self.length_arr = np.array(length_list)
#         self.data_dict = data_dict
#         self.nfeats = motion.shape[1]
#         self.name_list = name_list
#         self.reset_max_len(self.max_length)

#     def reset_max_len(self, length):
#         assert length <= self.max_motion_length
#         self.pointer = np.searchsorted(self.length_arr, length)
#         print("Pointer Pointing at %d" % self.pointer)
#         self.max_length = length

#     def inv_transform(self, data):
#         return data * self.std + self.mean

#     def __len__(self):
#         return len(self.name_list) - self.pointer

#     def __getitem__(self, item):
#         idx = self.pointer + item
#         data = self.data_dict[self.name_list[idx]]
#         motion, m_length, text_list, style_text_list = data["motion"], data["length"], data[
#             "text"], data["style_text"]
#         # 随机选一个caption
#         # Randomly select a caption
#         #
#         text_data = random.choice(text_list)
#         caption, tokens = text_data["caption"], text_data["tokens"]
#         style_caption = style_text_list[0]["caption"]

#         if len(tokens) < self.max_text_len:
#             # pad with "unk"
#             tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
#             sent_len = len(tokens)
#             tokens = tokens + ["unk/OTHER"
#                                ] * (self.max_text_len + 2 - sent_len)
#         else:
#             # crop
#             tokens = tokens[:self.max_text_len]
#             tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
#             sent_len = len(tokens)
#         pos_one_hots = []
#         word_embeddings = []
#         for token in tokens:
#             word_emb, pos_oh = self.w_vectorizer[token]
#             pos_one_hots.append(pos_oh[None, :])
#             word_embeddings.append(word_emb[None, :])
#         pos_one_hots = np.concatenate(pos_one_hots, axis=0)
#         word_embeddings = np.concatenate(word_embeddings, axis=0)

#         # Crop the motions in to times of 4, and introduce small variations
#         if self.unit_length < 10:
#             coin2 = np.random.choice(["single", "single", "double"])
#         else:
#             coin2 = "single"

#         if coin2 == "double":
#             m_length = (m_length // self.unit_length - 1) * self.unit_length
#         elif coin2 == "single":
#             m_length = (m_length // self.unit_length) * self.unit_length
#         idx = random.randint(0, len(motion) - m_length)
#         motion = motion[idx:idx + m_length]
#         "Z Normalization"
#         motion = (motion - self.mean) / self.std

#         # # padding
#         # if m_length < self.max_motion_length:
#         #     motion = np.concatenate(
#         #         [
#         #             motion,
#         #             np.zeros((self.max_motion_length - m_length, motion.shape[1])),
#         #         ],
#         #         axis=0,
#         #     )
#         # print(word_embeddings.shape, motion.shape, m_length)
#         # print(tokens)

#         # debug check nan
#         if np.any(np.isnan(motion)):
#             raise ValueError("nan in motion")
# #这里concat到一起了，style_text保存在
#         return (
#             word_embeddings,
#             pos_one_hots,
#             caption,
#             sent_len,
#             motion,
#             m_length,
#             "_".join(tokens),
#             style_caption,
#         )
#         # return caption, motion, m_length


# """For use of training baseline"""


# class Text2MotionDatasetBaseline(data.Dataset):

#     def __init__(self, opt, mean, std, split_file, w_vectorizer):
#         self.opt = opt
#         self.w_vectorizer = w_vectorizer
#         self.max_length = 20
#         self.pointer = 0
#         self.max_motion_length = opt.max_motion_length
#         min_motion_len = 40 if self.opt.dataset_name == "t2m" else 24

#         data_dict = {}
#         id_list = []
#         with cs.open(split_file, "r") as f:
#             for line in f.readlines():
#                 id_list.append(line.strip())
#         # id_list = id_list[:200]

#         new_name_list = []
#         length_list = []
#         for name in tqdm(id_list):
#             try:
#                 motion = np.load(pjoin(opt.motion_dir, name + ".npy"))
#                 if (len(motion)) < min_motion_len or (len(motion) >= 200):
#                     continue
#                 text_data = []
#                 flag = False
#                 with cs.open(pjoin(opt.text_dir, name + ".txt")) as f:
#                     for line in f.readlines():
#                         text_dict = {}
#                         line_split = line.strip().split("#")
#                         caption = line_split[0]
#                         tokens = line_split[1].split(" ")
#                         f_tag = float(line_split[2])
#                         to_tag = float(line_split[3])
#                         f_tag = 0.0 if np.isnan(f_tag) else f_tag
#                         to_tag = 0.0 if np.isnan(to_tag) else to_tag

#                         text_dict["caption"] = caption
#                         text_dict["tokens"] = tokens
#                         if f_tag == 0.0 and to_tag == 0.0:
#                             flag = True
#                             text_data.append(text_dict)
#                         else:
#                             try:
#                                 n_motion = motion[int(f_tag * 20):int(to_tag *
#                                                                       20)]
#                                 if (len(n_motion)) < min_motion_len or (
#                                         len(n_motion) >= 200):
#                                     continue
#                                 new_name = (
#                                     random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
#                                     "_" + name)
#                                 while new_name in data_dict:
#                                     new_name = (random.choice(
#                                         "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
#                                                 name)
#                                 data_dict[new_name] = {
#                                     "motion": n_motion,
#                                     "length": len(n_motion),
#                                     "text": [text_dict],
#                                 }
#                                 new_name_list.append(new_name)
#                                 length_list.append(len(n_motion))
#                             except:
#                                 print(line_split)
#                                 print(line_split[2], line_split[3], f_tag,
#                                       to_tag, name)
#                                 # break

#                 if flag:
#                     data_dict[name] = {
#                         "motion": motion,
#                         "length": len(motion),
#                         "text": text_data,
#                     }
#                     new_name_list.append(name)
#                     length_list.append(len(motion))
#             except:
#                 pass

#         name_list, length_list = zip(
#             *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

#         self.mean = mean
#         self.std = std
#         self.length_arr = np.array(length_list)
#         self.data_dict = data_dict
#         self.nfeats = motion.shape[1]
#         self.name_list = name_list
#         self.reset_max_len(self.max_length)

#     def reset_max_len(self, length):
#         assert length <= self.max_motion_length
#         self.pointer = np.searchsorted(self.length_arr, length)
#         print("Pointer Pointing at %d" % self.pointer)
#         self.max_length = length

#     def inv_transform(self, data):
#         return data * self.std + self.mean

#     def __len__(self):
#         return len(self.data_dict) - self.pointer

#     def __getitem__(self, item):
#         idx = self.pointer + item
#         data = self.data_dict[self.name_list[idx]]
#         motion, m_length, text_list = data["motion"], data["length"], data[
#             "text"]
#         # Randomly select a caption
#         text_data = random.choice(text_list)
#         caption, tokens = text_data["caption"], text_data["tokens"]

#         if len(tokens) < self.opt.max_text_len:
#             # pad with "unk"
#             tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
#             sent_len = len(tokens)
#             tokens = tokens + ["unk/OTHER"
#                                ] * (self.opt.max_text_len + 2 - sent_len)
#         else:
#             # crop
#             tokens = tokens[:self.opt.max_text_len]
#             tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
#             sent_len = len(tokens)
#         pos_one_hots = []
#         word_embeddings = []
#         for token in tokens:
#             word_emb, pos_oh = self.w_vectorizer[token]
#             pos_one_hots.append(pos_oh[None, :])
#             word_embeddings.append(word_emb[None, :])
#         pos_one_hots = np.concatenate(pos_one_hots, axis=0)
#         word_embeddings = np.concatenate(word_embeddings, axis=0)

#         len_gap = (m_length - self.max_length) // self.opt.unit_length

#         if m_length != self.max_length:
#             # print("Motion original length:%d_%d"%(m_length, len(motion)))
#             if self.opt.unit_length < 10:
#                 coin2 = np.random.choice(["single", "single", "double"])
#             else:
#                 coin2 = "single"
#             if len_gap == 0 or (len_gap == 1 and coin2 == "double"):
#                 m_length = self.max_length
#                 s_idx = random.randint(0, m_length - self.max_length)
#             else:
#                 if coin2 == "single":
#                     n_m_length = self.max_length + self.opt.unit_length * len_gap
#                 else:
#                     n_m_length = self.max_length + self.opt.unit_length * (
#                         len_gap - 1)
#                 s_idx = random.randint(0, m_length - n_m_length)
#                 m_length = n_m_length
#         else:
#             s_idx = 0

#         src_motion = motion[s_idx:s_idx + m_length]
#         tgt_motion = motion[s_idx:s_idx + self.max_length]
#         "Z Normalization"
#         src_motion = (src_motion - self.mean) / self.std
#         tgt_motion = (tgt_motion - self.mean) / self.std

#         # padding
#         if m_length < self.max_motion_length:
#             src_motion = np.concatenate(
#                 [
#                     src_motion,
#                     np.zeros(
#                         (self.max_motion_length - m_length, motion.shape[1])),
#                 ],
#                 axis=0,
#             )
#         # print(m_length, src_motion.shape, tgt_motion.shape)
#         # print(word_embeddings.shape, motion.shape)
#         # print(tokens)
#         return word_embeddings, caption, sent_len, src_motion, tgt_motion, m_length


# class MotionDatasetV2(data.Dataset):

#     def __init__(self, opt, mean, std, split_file):
#         self.opt = opt
#         joints_num = opt.joints_num

#         self.data = []
#         self.lengths = []
#         id_list = []
#         with cs.open(split_file, "r") as f:
#             for line in f.readlines():
#                 id_list.append(line.strip())

#         for name in tqdm(id_list):
#             try:
#                 motion = np.load(pjoin(opt.motion_dir, name + ".npy"))
#                 if motion.shape[0] < opt.window_size:
#                     continue
#                 self.lengths.append(motion.shape[0] - opt.window_size)
#                 self.data.append(motion)
#             except:
#                 # Some motion may not exist in KIT dataset
#                 pass

#         self.cumsum = np.cumsum([0] + self.lengths)

#         if opt.is_train:
#             # root_rot_velocity (B, seq_len, 1)
#             std[0:1] = std[0:1] / opt.feat_bias
#             # root_linear_velocity (B, seq_len, 2)
#             std[1:3] = std[1:3] / opt.feat_bias
#             # root_y (B, seq_len, 1)
#             std[3:4] = std[3:4] / opt.feat_bias
#             # ric_data (B, seq_len, (joint_num - 1)*3)
#             std[4:4 + (joints_num - 1) * 3] = std[4:4 +
#                                                   (joints_num - 1) * 3] / 1.0
#             # rot_data (B, seq_len, (joint_num - 1)*6)
#             std[4 + (joints_num - 1) * 3:4 +
#                 (joints_num - 1) * 9] = (std[4 + (joints_num - 1) * 3:4 +
#                                              (joints_num - 1) * 9] / 1.0)
#             # local_velocity (B, seq_len, joint_num*3)
#             std[4 + (joints_num - 1) * 9:4 + (joints_num - 1) * 9 +
#                 joints_num * 3] = (std[4 + (joints_num - 1) * 9:4 +
#                                        (joints_num - 1) * 9 + joints_num * 3] /
#                                    1.0)
#             # foot contact (B, seq_len, 4)
#             std[4 + (joints_num - 1) * 9 + joints_num * 3:] = (
#                 std[4 +
#                     (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias)

#             assert 4 + (joints_num -
#                         1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
#             np.save(pjoin(opt.meta_dir, "mean.npy"), mean)
#             np.save(pjoin(opt.meta_dir, "std.npy"), std)

#         self.mean = mean
#         self.std = std
#         print("Total number of motions {}, snippets {}".format(
#             len(self.data), self.cumsum[-1]))

#     def inv_transform(self, data):
#         return data * self.std + self.mean

#     def __len__(self):
#         return self.cumsum[-1]

#     def __getitem__(self, item):
#         if item != 0:
#             motion_id = np.searchsorted(self.cumsum, item) - 1
#             idx = item - self.cumsum[motion_id] - 1
#         else:
#             motion_id = 0
#             idx = 0
#         motion = self.data[motion_id][idx:idx + self.opt.window_size]
#         "Z Normalization"
#         motion = (motion - self.mean) / self.std

#         return motion


# class RawTextDataset(data.Dataset):

#     def __init__(self, opt, mean, std, text_file, w_vectorizer):
#         self.mean = mean
#         self.std = std
#         self.opt = opt
#         self.data_dict = []
#         self.nlp = spacy.load("en_core_web_sm")

#         with cs.open(text_file) as f:
#             for line in f.readlines():
#                 word_list, pos_list = self.process_text(line.strip())
#                 tokens = [
#                     "%s/%s" % (word_list[i], pos_list[i])
#                     for i in range(len(word_list))
#                 ]
#                 self.data_dict.append({
#                     "caption": line.strip(),
#                     "tokens": tokens
#                 })

#         self.w_vectorizer = w_vectorizer
#         print("Total number of descriptions {}".format(len(self.data_dict)))

#     def process_text(self, sentence):
#         sentence = sentence.replace("-", "")
#         doc = self.nlp(sentence)
#         word_list = []
#         pos_list = []
#         for token in doc:
#             word = token.text
#             if not word.isalpha():
#                 continue
#             if (token.pos_ == "NOUN"
#                     or token.pos_ == "VERB") and (word != "left"):
#                 word_list.append(token.lemma_)
#             else:
#                 word_list.append(word)
#             pos_list.append(token.pos_)
#         return word_list, pos_list

#     def inv_transform(self, data):
#         return data * self.std + self.mean

#     def __len__(self):
#         return len(self.data_dict)

#     def __getitem__(self, item):
#         data = self.data_dict[item]
#         caption, tokens = data["caption"], data["tokens"]

#         if len(tokens) < self.opt.max_text_len:
#             # pad with "unk"
#             tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
#             sent_len = len(tokens)
#             tokens = tokens + ["unk/OTHER"
#                                ] * (self.opt.max_text_len + 2 - sent_len)
#         else:
#             # crop
#             tokens = tokens[:self.opt.max_text_len]
#             tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
#             sent_len = len(tokens)
#         pos_one_hots = []
#         word_embeddings = []
#         for token in tokens:
#             word_emb, pos_oh = self.w_vectorizer[token]
#             pos_one_hots.append(pos_oh[None, :])
#             word_embeddings.append(word_emb[None, :])
#         pos_one_hots = np.concatenate(pos_one_hots, axis=0)
#         word_embeddings = np.concatenate(word_embeddings, axis=0)

#         return word_embeddings, pos_one_hots, caption, sent_len


# class TextOnlyDataset(data.Dataset):

#     def __init__(self, opt, mean, std, split_file, text_dir, **kwargs):
#         self.mean = mean
#         self.std = std
#         self.opt = opt
#         self.data_dict = []
#         self.max_length = 20
#         self.pointer = 0
#         self.fixed_length = 120

#         data_dict = {}
#         id_list = []
#         with cs.open(split_file, "r") as f:
#             for line in f.readlines():
#                 id_list.append(line.strip())
#         # id_list = id_list[:200]

#         new_name_list = []
#         length_list = []
#         for name in tqdm(id_list):
#             try:
#                 text_data = []
#                 flag = False
#                 with cs.open(pjoin(text_dir, name + ".txt")) as f:
#                     for line in f.readlines():
#                         text_dict = {}
#                         line_split = line.strip().split("#")
#                         caption = line_split[0]
#                         tokens = line_split[1].split(" ")
#                         f_tag = float(line_split[2])
#                         to_tag = float(line_split[3])
#                         f_tag = 0.0 if np.isnan(f_tag) else f_tag
#                         to_tag = 0.0 if np.isnan(to_tag) else to_tag

#                         text_dict["caption"] = caption
#                         text_dict["tokens"] = tokens
#                         if f_tag == 0.0 and to_tag == 0.0:
#                             flag = True
#                             text_data.append(text_dict)
#                         else:
#                             try:
#                                 new_name = (
#                                     random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
#                                     "_" + name)
#                                 while new_name in data_dict:
#                                     new_name = (random.choice(
#                                         "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
#                                                 name)
#                                 data_dict[new_name] = {"text": [text_dict]}
#                                 new_name_list.append(new_name)
#                             except:
#                                 print(line_split)
#                                 print(line_split[2], line_split[3], f_tag,
#                                       to_tag, name)
#                                 # break

#                 if flag:
#                     data_dict[name] = {"text": text_data}
#                     new_name_list.append(name)
#             except:
#                 pass

#         self.length_arr = np.array(length_list)
#         self.data_dict = data_dict
#         self.name_list = new_name_list

#     def inv_transform(self, data):
#         return data * self.std + self.mean

#     def __len__(self):
#         return len(self.data_dict)

#     def __getitem__(self, item):
#         idx = self.pointer + item
#         data = self.data_dict[self.name_list[idx]]
#         text_list = data["text"]

#         # Randomly select a caption
#         text_data = random.choice(text_list)
#         caption, tokens = text_data["caption"], text_data["tokens"]
#         return None, None, caption, None, np.array([0
#                                                     ]), self.fixed_length, None
#         # fixed_length can be set from outside before sampling


# # A wrapper class for t2m original dataset for MDM purposes
# class HumanML3D(data.Dataset):

#     def __init__(self,
#                  mode,
#                  datapath="./dataset/humanml_opt.txt",
#                  split="train",
#                  **kwargs):
#         self.mode = mode

#         self.dataset_name = "t2m"
#         self.dataname = "t2m"

#         # Configurations of T2M dataset and KIT dataset is almost the same
#         abs_base_path = f"."
#         dataset_opt_path = pjoin(abs_base_path, datapath)
#         device = (
#             None  # torch.device('cuda:4') # This param is not in use in this context
#         )
#         opt = get_opt(dataset_opt_path, device)
#         opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
#         opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
#         opt.text_dir = pjoin(abs_base_path, opt.text_dir)
#         opt.model_dir = pjoin(abs_base_path, opt.model_dir)
#         opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
#         opt.data_root = pjoin(abs_base_path, opt.data_root)
#         opt.save_root = pjoin(abs_base_path, opt.save_root)
#         self.opt = opt
#         print("Loading dataset %s ..." % opt.dataset_name)

#         if mode == "gt":
#             # used by T2M models (including evaluators)
#             self.mean = np.load(pjoin(opt.meta_dir, "mean.npy"))
#             self.std = np.load(pjoin(opt.meta_dir, "std.npy"))
#         elif mode in ["train", "eval", "text_only"]:
#             # used by our models
#             self.mean = np.load(pjoin(opt.data_root, "Mean.npy"))
#             self.std = np.load(pjoin(opt.data_root, "Std.npy"))

#         if mode == "eval":
#             # used by T2M models (including evaluators)
#             # this is to translate their norms to ours
#             self.mean_for_eval = np.load(pjoin(opt.meta_dir, "mean.npy"))
#             self.std_for_eval = np.load(pjoin(opt.meta_dir, "std.npy"))

#         self.split_file = pjoin(opt.data_root, f"{split}.txt")
#         if mode == "text_only":
#             self.t2m_dataset = TextOnlyDataset(self.opt, self.mean, self.std,
#                                                self.split_file)
#         else:
#             self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, "glove"),
#                                                "our_vab")
#             self.t2m_dataset = Text2MotionDatasetV2(self.opt, self.mean,
#                                                     self.std, self.split_file,
#                                                     self.w_vectorizer)
#             self.num_actions = 1  # dummy placeholder

#     def __getitem__(self, item):
#         return self.t2m_dataset.__getitem__(item)

#     def __len__(self):
#         return self.t2m_dataset.__len__()


# # A wrapper class for t2m original dataset for MDM purposes
# class KIT(HumanML3D):

#     def __init__(self,
#                  mode,
#                  datapath="./dataset/kit_opt.txt",
#                  split="train",
#                  **kwargs):
#         super(KIT, self).__init__(mode, datapath, split, **kwargs)






import codecs as cs
import os
import random
from os.path import join as pjoin

import numpy as np
import spacy
import torch
from rich.progress import track
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from ..utils.get_opt import get_opt
from ..utils.word_vectorizer import WordVectorizer


# import spacy
def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


class HumanML3DSceneDataset(data.Dataset):
    def __init__(
        self,
        # Args from original Text2MotionDatasetV2
        mean, # Path to Mean.npy or the loaded numpy array
        std,  # Path to Std.npy or the loaded numpy array
        split_file, # Path to your train/val/test split file (e.g., "train.txt") for the scene dataset
        w_vectorizer, # The word vectorizer instance
        max_motion_length, # Max motion length after padding (usually handled by collate_fn)
        min_motion_length, # Min motion length to consider a sample valid
        max_text_len,      # Max number of tokens in a text description
        unit_length,       # To ensure motion length is a multiple of this
        motion_dir,        # Path to 'HumanML3D_100Style/new_joint_vecs'
        text_dir,          # Path to 'HumanML3D_100Style/texts'

        # New arguments for scene data
        scene_label_filepath, # Path to your scene label mapping file (e.g., "scene_mappings.txt")
        num_scene_classes=100, # Number of unique scene categories

        # Other args
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs, # To catch any other arguments like dataset_name if used by w_vectorizer etc.
    ):
        self.w_vectorizer = w_vectorizer
        self.max_motion_length = max_motion_length # This is more like a cap, actual padding in collate
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.num_scene_classes = num_scene_classes

        # --- 1. Load motion IDs from split file ---
        id_list = []
        # Assuming split_file is now relative to some root or an absolute path
        with open(split_file, "r") as f:
            for line in f.readlines():
                # Assuming each line in split_file is a motion_id like "000001"
                id_list.append(line.strip())
        
        # if tiny: # note：应该是用不到的
        #     id_list = id_list[:10] if tiny else id_list[:100]
        
        # --- 2. Load scene label mapping ---
        self.scene_label_map = {}
        with open(scene_label_filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: 
                    continue # Skip empty lines
                motion_identifier_in_file = parts[0] # e.g., "030001" or "030001XYZ.bvh"
                scene_id = int(parts[-1])

                # Extract the actual motion_id used for file lookup (e.g., "030001")
                # This needs to be robust. If filenames are "XXXXXX.npy", we need "XXXXXX"
                # If motion_identifier_in_file is "030001 Aeroplane_BR_00.bvh", extract "030001"
                # If it's already "030001", then no change.
                # A common pattern is to take the part before the first underscore or space if complex.
                # For simplicity, assuming it's the first part if split by non-alphanumeric, or already the ID.
                # Example: if it's like "030001_extra_stuff" -> "030001"
                # actual_motion_id = motion_identifier_in_file.split('_')[0]
                # Let's assume for HumanML3D the IDs are numeric strings like "000001"
                # And that your scene_label_filepath contains these directly or can be easily extracted.
                # If your scene_label_filepath's first column IS the motion_id (e.g., "000001"), this is simpler:
                actual_motion_id = motion_identifier_in_file # MODIFY IF NEEDED based on your scene_label_file format

                self.scene_label_map[actual_motion_id] = scene_id

        # --- 3. Load data (motion, text) into data_dict ---
        self.data_dict = {}
        new_name_list = [] # This will be our self.name_list later
        length_list = []   # For sorting by motion length

        # Setup progress bar if enabled
        # enumerator = enumerate(id_list)
        # if progress_bar:
        #     try:
        #         from rich.progress import track
        #         enumerator = enumerate(track(id_list, f"Loading SceneDataset {split_file.split('/')[-1].split('.')[0]}"))
        #     except ImportError:
        #         print("rich.progress not found, proceeding without progress bar.")
        # Using a simple print for progress here to reduce external deps for example
        print(f"Loading SceneDataset {split_file.split('/')[-1].split('.')[0]}...")
        valid_sample_count = 0
        for i, name in enumerate(id_list): # 'name' here is the motion_id like "000001"
            if progress_bar and i % 100 == 0 and i > 0:
                print(f"Processed {i}/{len(id_list)} samples...")
            try:
                # Check if scene label exists for this ID. If not, skip.
                if name not in self.scene_label_map:
                    # print(f"Warning: Scene label for motion ID '{name}' not found in scene_label_file. Skipping.")
                    continue

                motion_path = os.path.join(motion_dir, name + ".npy")
                motion = np.load(motion_path)

                if not (self.min_motion_length <= len(motion) < 200): # Original condition: < 200
                    continue

                text_data_list = [] # List to hold text dicts for this motion
                text_file_path = os.path.join(text_dir, name + ".txt") # datasets/humanml3d_scene/texts/032887.txt
                with open(text_file_path, 'r') as f:
                    for line in f.readlines():
                        line_split = line.strip().split("#")
                        if len(line_split) < 2: # Basic check for malformed lines
                            # print(f"Skipping malformed line in {text_file_path}: {line.strip()}")
                            continue
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        
                        # Original dataset had f_tag, to_tag for segments.
                        # Assuming for HumanML3D_100Style, each .txt file corresponds to the whole .npy
                        # and we don't need to sub-segment based on f_tag/to_tag here.
                        # If your .txt files still have f_tag/to_tag and you want to use them,
                        # that logic would need to be re-introduced.
                        # For now, assume each text line is a valid description for the whole motion.
                        text_dict = {"caption": caption, "tokens": tokens}
                        text_data_list.append(text_dict)
                
                if not text_data_list: # If no valid text entries found for this motion
                    # print(f"Warning: No valid text data found for motion ID '{name}'. Skipping.")
                    continue

                self.data_dict[name] = {
                    "motion": motion,
                    "length": len(motion),
                    "text": text_data_list, # Store list of text dicts
                    "scene_label": self.scene_label_map[name] # Store the scene label
                }
                new_name_list.append(name)
                length_list.append(len(motion))
                valid_sample_count +=1

            except FileNotFoundError:
                # print(f"Warning: Motion or text file not found for ID '{name}'. Skipping.")
                pass # Or log this
            except Exception as e:
                # print(f"Error processing ID '{name}': {e}. Skipping.")
                pass # Or log this
        
        print(f"Finished loading. Found {valid_sample_count} valid samples.")

        if not new_name_list:
            raise ValueError("No valid data loaded. Check paths, split_file, and scene_label_file.")

        # Sort by motion length (similar to original dataset)
        # This helps in batching if using 'reset_max_len' logic, though that part is optional for finetuning.
        if length_list: # Ensure not empty
             # Ensure new_name_list and length_list are converted to np.array for advanced indexing if needed
            name_list_sorted, length_list_sorted = zip(
                *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
            self.name_list = list(name_list_sorted)
            self.length_arr = np.array(length_list_sorted)
        else: # Handle case where no data was loaded, though previous check should catch this
            self.name_list = []
            self.length_arr = np.array([])
        

        # --- 4. Load mean and std ---
        # Allow mean/std to be pre-loaded numpy arrays or paths to .npy files
        if isinstance(mean, str):
            self.mean = np.load(mean)
        else:
            self.mean = mean
        if isinstance(std, str):
            self.std = np.load(std)
        else:
            self.std = std
            
        self.nfeats = self.mean.shape[-1] # Infer nfeats from mean/std shape

        # The 'pointer' and 'reset_max_len' logic was for a specific sampling strategy.
        # For standard finetuning, you might not need it, or adapt it.
        # If you want to keep it:
        self.pointer = 0
        # self.reset_max_len(max_motion_length) # Or some initial max_length for batching
    
    # If keeping the reset_max_len dynamic batching strategy:
    # def reset_max_len(self, length):
    #     assert length <= self.max_motion_length
    #     if len(self.length_arr) == 0:
    #         self.pointer = 0
    #         print("Warning: length_arr is empty, pointer set to 0.")
    #     else:
    #         self.pointer = np.searchsorted(self.length_arr, length)
    #     print(f"Pointer set to {self.pointer} for max_length {length}")
    #     # self.current_max_length_for_sampling = length # Renaming for clarity
    
    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        # If using pointer logic: return len(self.name_list) - self.pointer
        return len(self.name_list) # Standard length

    def __getitem__(self, item):
        # If using pointer logic: idx_in_name_list = self.pointer + item
        # else:
        idx_in_name_list = item
        
        motion_id = self.name_list[idx_in_name_list]
        data = self.data_dict[motion_id]
        
        motion, m_length = data["motion"], data["length"]
        text_list = data["text"]
        scene_label = data["scene_label"] # Get the scene label

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        # Text processing (same as original)
        if len(tokens) < self.max_text_len:
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)
        else:
            tokens = tokens[:self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        
        # print(f"--- DEBUG: Dataset __getitem__ (item: {item}) ---")
        # print(f"Final 'tokens' list before passing to WordVectorizer (length: {len(tokens)}):")
        # for i, t in enumerate(tokens):
        #     print(f"  tokens[{i}]: '{t}' (type: {type(t)})")
        # # 额外检查一下，如果t不是字符串，或者包含奇怪的字符
        # if not isinstance(t, str) or not t.strip():
        #     print(f"    WARNING: Invalid token found: '{t}'")
        
        pos_one_hots = []
        word_embeddings = []
        for token_text in tokens: # Renamed from 'token' to 'token_text' to avoid confusion
            word_emb, pos_oh = self.w_vectorizer[token_text]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0).astype(np.float32)
        word_embeddings = np.concatenate(word_embeddings, axis=0).astype(np.float32)

        # Motion processing (adjust length and randomly crop)
        # Original had 'coin2' logic, simplifying here for typical finetuning
        # Ensure m_length is a multiple of unit_length
        processed_m_length = (m_length // self.unit_length) * self.unit_length
        if processed_m_length == 0 and m_length > 0: # Ensure at least one unit if original had length
             processed_m_length = self.unit_length
        if processed_m_length < self.min_motion_length and m_length >= self.min_motion_length: # If unit quantization made it too short
            processed_m_length = (m_length // self.unit_length + (1 if m_length % self.unit_length > 0 else 0)) * self.unit_length
            if processed_m_length == 0 : processed_m_length = self.unit_length # fallback
            while processed_m_length > m_length : # Ensure we don't exceed original length due to rounding up
                processed_m_length -= self.unit_length
            if processed_m_length <=0 : processed_m_length = self.unit_length # final fallback


        if processed_m_length <= 0 : # If motion is too short even for one unit
            # This case should ideally be filtered out in __init__ or handled by padding if allowed
            # For now, if it's extremely short, we might take the whole thing if it's > min_motion_length
            # or it might cause issues if processed_m_length is 0.
            # Fallback: use original m_length if it was valid, or skip.
            # This part needs careful thought based on how short motions are handled.
            # print(f"Warning: Motion {motion_id} has m_length {m_length}, processed_m_length became {processed_m_length}. Taking original segment if > min_len.")
            if m_length >= self.min_motion_length:
                processed_m_length = m_length
            else: # Should have been filtered
                 # Fallback to a small valid length if absolutely necessary, or raise error
                 # This indicates an issue with min_motion_length filter or unit_length choice
                 processed_m_length = self.min_motion_length if self.min_motion_length <= m_length else m_length
                 if processed_m_length == 0 and m_length > 0: processed_m_length = self.unit_length
                 if processed_m_length == 0: raise ValueError(f"Motion {motion_id} too short.")


        if len(motion) > processed_m_length:
            start_idx = random.randint(0, len(motion) - processed_m_length)
            motion_segment = motion[start_idx : start_idx + processed_m_length]
        elif len(motion) == processed_m_length:
            motion_segment = motion
        else: # len(motion) < processed_m_length. This can happen if processed_m_length logic makes it longer.
              # This should be avoided by careful `processed_m_length` calculation.
              # Fallback: take the whole motion. Padding will be handled by collate_fn.
            motion_segment = motion
            processed_m_length = len(motion_segment) # Update length to actual segment length

        # Z Normalization
        motion_normalized = (motion_segment - self.mean) / self.std

        if np.any(np.isnan(motion_normalized)):
            # This can happen if std is zero for some features.
            # print(f"Warning: NaN found in normalized motion for {motion_id}. Check mean/std values.")
            # A common fix: replace NaNs with 0, or investigate std.
            motion_normalized = np.nan_to_num(motion_normalized)


        # Return tuple in the order expected by your mld_collate function
        # (word_embs, pos_ohot, caption, sent_len, motion, m_length, tokens_str, scene_label)
        return (
            word_embeddings,    # b[0]
            pos_one_hots,       # b[1]
            caption,            # b[2]
            sent_len,           # b[3]
            motion_normalized,  # b[4]
            processed_m_length, # b[5] (actual length of the motion segment being returned)
            "_".join(tokens),   # b[6]
            scene_label         # b[7] (new)
        )
    

"""For use of training text-2-motion generative model"""


class Text2MotionDataset(data.Dataset):

    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        min_motion_len = 40 if self.opt.dataset_name == "t2m" else 24

        joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + ".npy"))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20):int(to_tag *
                                                                      20)]
                                if (len(n_motion)) < min_motion_len or (
                                        len(n_motion) >= 200):
                                    continue
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                    "_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice(
                                        "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                name)
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                      to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4:4 + (joints_num - 1) * 3] = std[4:4 +
                                                  (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3:4 +
                (joints_num - 1) * 9] = (std[4 + (joints_num - 1) * 3:4 +
                                             (joints_num - 1) * 9] / 1.0)
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9:4 + (joints_num - 1) * 9 +
                joints_num * 3] = (std[4 + (joints_num - 1) * 9:4 +
                                       (joints_num - 1) * 9 + joints_num * 3] /
                                   1.0)
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = (
                std[4 +
                    (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias)

            assert 4 + (joints_num -
                        1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, "mean.npy"), mean)
            np.save(pjoin(opt.meta_dir, "std.npy"), std)

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.opt.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data[
            "text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if self.opt.is_train:
            if m_length != self.max_length:
                # print("Motion original length:%d_%d"%(m_length, len(motion)))
                if self.opt.unit_length < 10:
                    coin2 = np.random.choice(["single", "single", "double"])
                else:
                    coin2 = "single"
                if len_gap == 0 or (len_gap == 1 and coin2 == "double"):
                    m_length = self.max_length
                    idx = random.randint(0, m_length - self.max_length)
                    motion = motion[idx:idx + self.max_length]
                else:
                    if coin2 == "single":
                        n_m_length = self.max_length + self.opt.unit_length * len_gap
                    else:
                        n_m_length = self.max_length + self.opt.unit_length * (
                            len_gap - 1)
                    idx = random.randint(0, m_length - n_m_length)
                    motion = motion[idx:idx + self.max_length]
                    m_length = n_m_length
                # print(len_gap, idx, coin2)
        else:
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(["single", "single", "double"])
            else:
                coin2 = "single"

            if coin2 == "double":
                m_length = (m_length // self.opt.unit_length -
                            1) * self.opt.unit_length
            elif coin2 == "single":
                m_length = (m_length //
                            self.opt.unit_length) * self.opt.unit_length
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx + m_length]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length


"""For use of training text motion matching model, and evaluations"""


class Text2MotionDatasetV2(data.Dataset):

    def __init__(
        self,
        mean,
        std,
        split_file,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir,
        text_dir,
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs,
    ):
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        # min_motion_len = 40 if dataset_name =='t2m' else 24
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.id_list = id_list

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10

        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading HumanML3D {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)
        count = 0
        bad_count = 0
        new_name_list = []
        length_list = []
        for i, name in enumerator:
            if count > maxdata:
                break
            try:
                motion = np.load(pjoin(motion_dir, name + ".npy"))
                if (len(motion)) < self.min_motion_length or (len(motion) >=
                                                              200):
                    bad_count += 1
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20):int(to_tag *
                                                                      20)]
                                if (len(n_motion)
                                    ) < self.min_motion_length or (
                                        (len(n_motion) >= 200)):
                                    continue
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                    "_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice(
                                        "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                name)
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                # None
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                      to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    # print(count)
                    count += 1
                    # print(name)
            except:
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data[
            "text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        # # padding
        # if m_length < self.max_motion_length:
        #     motion = np.concatenate(
        #         [
        #             motion,
        #             np.zeros((self.max_motion_length - m_length, motion.shape[1])),
        #         ],
        #         axis=0,
        #     )
        # print(word_embeddings.shape, motion.shape, m_length)
        # print(tokens)

        # debug check nan
        if np.any(np.isnan(motion)):
            raise ValueError("nan in motion")

        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            "_".join(tokens),
        )
        # return caption, motion, m_length


"""For use of training baseline"""


class Text2MotionDatasetBaseline(data.Dataset):

    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name == "t2m" else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + ".npy"))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20):int(to_tag *
                                                                      20)]
                                if (len(n_motion)) < min_motion_len or (
                                        len(n_motion) >= 200):
                                    continue
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                    "_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice(
                                        "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                name)
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                      to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data[
            "text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if m_length != self.max_length:
            # print("Motion original length:%d_%d"%(m_length, len(motion)))
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(["single", "single", "double"])
            else:
                coin2 = "single"
            if len_gap == 0 or (len_gap == 1 and coin2 == "double"):
                m_length = self.max_length
                s_idx = random.randint(0, m_length - self.max_length)
            else:
                if coin2 == "single":
                    n_m_length = self.max_length + self.opt.unit_length * len_gap
                else:
                    n_m_length = self.max_length + self.opt.unit_length * (
                        len_gap - 1)
                s_idx = random.randint(0, m_length - n_m_length)
                m_length = n_m_length
        else:
            s_idx = 0

        src_motion = motion[s_idx:s_idx + m_length]
        tgt_motion = motion[s_idx:s_idx + self.max_length]
        "Z Normalization"
        src_motion = (src_motion - self.mean) / self.std
        tgt_motion = (tgt_motion - self.mean) / self.std

        # padding
        if m_length < self.max_motion_length:
            src_motion = np.concatenate(
                [
                    src_motion,
                    np.zeros(
                        (self.max_motion_length - m_length, motion.shape[1])),
                ],
                axis=0,
            )
        # print(m_length, src_motion.shape, tgt_motion.shape)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, caption, sent_len, src_motion, tgt_motion, m_length


class MotionDatasetV2(data.Dataset):

    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + ".npy"))
                if motion.shape[0] < opt.window_size:
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size)
                self.data.append(motion)
            except:
                # Some motion may not exist in KIT dataset
                pass

        self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4:4 + (joints_num - 1) * 3] = std[4:4 +
                                                  (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3:4 +
                (joints_num - 1) * 9] = (std[4 + (joints_num - 1) * 3:4 +
                                             (joints_num - 1) * 9] / 1.0)
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9:4 + (joints_num - 1) * 9 +
                joints_num * 3] = (std[4 + (joints_num - 1) * 9:4 +
                                       (joints_num - 1) * 9 + joints_num * 3] /
                                   1.0)
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = (
                std[4 +
                    (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias)

            assert 4 + (joints_num -
                        1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, "mean.npy"), mean)
            np.save(pjoin(opt.meta_dir, "std.npy"), std)

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(
            len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.opt.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion


class RawTextDataset(data.Dataset):

    def __init__(self, opt, mean, std, text_file, w_vectorizer):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.nlp = spacy.load("en_core_web_sm")

        with cs.open(text_file) as f:
            for line in f.readlines():
                word_list, pos_list = self.process_text(line.strip())
                tokens = [
                    "%s/%s" % (word_list[i], pos_list[i])
                    for i in range(len(word_list))
                ]
                self.data_dict.append({
                    "caption": line.strip(),
                    "tokens": tokens
                })

        self.w_vectorizer = w_vectorizer
        print("Total number of descriptions {}".format(len(self.data_dict)))

    def process_text(self, sentence):
        sentence = sentence.replace("-", "")
        doc = self.nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == "NOUN"
                    or token.pos_ == "VERB") and (word != "left"):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[item]
        caption, tokens = data["caption"], data["tokens"]

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len


class TextOnlyDataset(data.Dataset):

    def __init__(self, opt, mean, std, split_file, text_dir, **kwargs):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.max_length = 20
        self.pointer = 0
        self.fixed_length = 120

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                    "_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice(
                                        "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                name)
                                data_dict[new_name] = {"text": [text_dict]}
                                new_name_list.append(new_name)
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                      to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {"text": text_data}
                    new_name_list.append(name)
            except:
                pass

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        text_list = data["text"]

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]
        return None, None, caption, None, np.array([0
                                                    ]), self.fixed_length, None
        # fixed_length can be set from outside before sampling


# A wrapper class for t2m original dataset for MDM purposes
class HumanML3D(data.Dataset):

    def __init__(self,
                 mode,
                 datapath="./dataset/humanml_opt.txt",
                 split="train",
                 **kwargs):
        self.mode = mode

        self.dataset_name = "t2m"
        self.dataname = "t2m"

        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = f"."
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = (
            None  # torch.device('cuda:4') # This param is not in use in this context
        )
        opt = get_opt(dataset_opt_path, device)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        self.opt = opt
        print("Loading dataset %s ..." % opt.dataset_name)

        if mode == "gt":
            # used by T2M models (including evaluators)
            self.mean = np.load(pjoin(opt.meta_dir, "mean.npy"))
            self.std = np.load(pjoin(opt.meta_dir, "std.npy"))
        elif mode in ["train", "eval", "text_only"]:
            # used by our models
            self.mean = np.load(pjoin(opt.data_root, "Mean.npy"))
            self.std = np.load(pjoin(opt.data_root, "Std.npy"))

        if mode == "eval":
            # used by T2M models (including evaluators)
            # this is to translate their norms to ours
            self.mean_for_eval = np.load(pjoin(opt.meta_dir, "mean.npy"))
            self.std_for_eval = np.load(pjoin(opt.meta_dir, "std.npy"))

        self.split_file = pjoin(opt.data_root, f"{split}.txt")
        if mode == "text_only":
            self.t2m_dataset = TextOnlyDataset(self.opt, self.mean, self.std,
                                               self.split_file)
        else:
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, "glove"),
                                               "our_vab")
            self.t2m_dataset = Text2MotionDatasetV2(self.opt, self.mean,
                                                    self.std, self.split_file,
                                                    self.w_vectorizer)
            self.num_actions = 1  # dummy placeholder

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()


# A wrapper class for t2m original dataset for MDM purposes
class KIT(HumanML3D):

    def __init__(self,
                 mode,
                 datapath="./dataset/kit_opt.txt",
                 split="train",
                 **kwargs):
        super(KIT, self).__init__(mode, datapath, split, **kwargs)