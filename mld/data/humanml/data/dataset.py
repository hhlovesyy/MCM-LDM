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

# -----------------------------------------------------------------------
# 【ICME V3】Prompt Ensemble Strategy
# 包含：视觉描述、动作指令、物理状态、极简描述、具体场景
# -----------------------------------------------------------------------
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

from PIL import Image
from torchvision import transforms

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

class Scene100StyleDataset(data.Dataset):
    def __init__(
        self,
        mean,
        std,
        split_file, # e.g., "train.txt"
        motion_dir, # e.g., ".../new_joint_vecs"
        scene_dict_path, # e.g., ".../Scene_name_dict.txt"
        scene_image_dir, # e.g., ".../scene_images/" (存放15张场景图的文件夹)
        max_motion_length=450, # NOTE:这个先用450，不然到时候VAE的pe长度会溢出报错，先简单训练一轮看看效果
        min_motion_length=20,
        unit_length=4,
        **kwargs,
    ):
        self.mean = mean
        self.std = std
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length
        self.scene_image_dir = scene_image_dir
        
        # 1. 读取 Split File (决定是训练集还是测试集)
        self.id_list = []
        with open(split_file, "r") as f:
            for line in f.readlines():
                self.id_list.append(line.strip())
        
        # 2. 读取 Scene Mapping (Motion ID -> Scene Label)
        # Scene_name_dict.txt 格式: 030561 Dumuqiao_00 0
        self.motion2scene = {}
        unique_scenes = set()
        
        print(f"Loading Scene Dictionary from {scene_dict_path}...")
        total_dataset_id_cnt = 0
        with open(scene_dict_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 2:
                    m_id = parts[0]
                    # 处理 Label: "Dumuqiao_00" -> "Dumuqiao"
                    raw_label = parts[1]
                    # 假设 Label 都是 Name_Number 的格式，我们只取 Name
                    scene_label = raw_label.split('_')[0] 
                    # 暂时写死一个T_Stage这个，代码不是很优雅，但是能跑
                    if scene_label == 'T':
                        scene_label = 'T_Stage'
                    
                    self.motion2scene[m_id] = scene_label
                    total_dataset_id_cnt += 1
                    unique_scenes.add(scene_label)
        
        self.SCENE_TO_ID = {scene: idx for idx, scene in enumerate(SCENE_LIST)}
        self.ID_TO_SCENE = {idx: scene for idx, scene in enumerate(SCENE_LIST)}
        print("--- 字典的所有键和值 ---")

        # 使用 items() 方法遍历键值对
        for scene_name, scene_id in self.SCENE_TO_ID.items():
            print(f"键 (Scene Name): {scene_name}, 值 (ID): {scene_id}")

        print("------------------------")
        print(f"Found {len(unique_scenes)} unique scenes: {unique_scenes}")
        print("Total dataset motion IDs in scene dict:", total_dataset_id_cnt)
        
        # 3. 过滤数据 (只保留在 Split 中 且 有 Scene 标签 且 长度合适 的动作)
        self.data = []
        
        print(f"Indexing motion data from {motion_dir}...")
        nameNotInSceneDict_cnt = 0
        pathNotExit_cnt = 0
        lenNotValid_cnt = 0
        for name in self.id_list:
            # 检查是否有场景标签
            if name not in self.motion2scene:
                nameNotInSceneDict_cnt += 1
                continue
            
            motion_path = pjoin(motion_dir, name + ".npy")
            if not os.path.exists(motion_path):
                pathNotExit_cnt += 1
                continue
                
            try:
                motion = np.load(motion_path)
                # 长度过滤
                if len(motion) < self.min_motion_length or len(motion) >= 450:
                    lenNotValid_cnt += 1
                    print("debug lenNotValid name:", name, " len:", len(motion))
                    continue
                
                # 成功添加
                self.data.append({
                    "name": name,
                    "motion_path": motion_path,
                    "scene_label": self.motion2scene[name]
                })
            except Exception as e:
                print(f"Error loading {name}: {e}")
                pass
                
        print(f"Dataset Loaded: {len(self.data)} samples ready for training.")
        print(f"  - Names not in Scene Dict: {nameNotInSceneDict_cnt}")
        print(f"  - Motion files not found: {pathNotExit_cnt}")
        print(f"  - Motions with invalid length: {lenNotValid_cnt}")
        
        # 1. 图像预处理 (CLIP 标准)
        self.image_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                 (0.26862954, 0.26130258, 0.27577711))
        ])
        self.scene_image_dir = scene_image_dir
        print("On Dataset class __init__ function end")
        pass_dataset_cnt = len(self.data)
        # 打印一下数据集里面的数据的通过率，pass_dataset_cnt / total_dataset_id_cnt
        print("pass dataset cnt:", pass_dataset_cnt, " total dataset id cnt:", total_dataset_id_cnt)
        print("Pass rate :", pass_dataset_cnt / total_dataset_id_cnt)
        # NOTE: 现在训练集里面只有957个动作，但是train.txt的split文件里则有2400个文件，平均一个场景应该是68个动作，这个比例有点少，后面看看需不需要处理一下
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data[item]
        name = sample['name']
        scene_label = sample['scene_label']
        
        # 1. 加载 Motion & 归一化
        motion = np.load(sample['motion_path'])
        
        # 随机裁剪逻辑 (参考 MCM-LDM)
        m_length = len(motion)
        if self.unit_length < 10:
             # MCM-LDM 的奇怪逻辑，保留它
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"
        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
            
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        
        # Z-Normalization
        motion = (motion - self.mean) / self.std
        motion = torch.tensor(motion).float() # 转 Tensor

        scene_id = self.SCENE_TO_ID[scene_label]

        # 2. 获取 Scene Text (LLM Description)
        # 如果字典里没找到，就回退到原始 Label
        if scene_label in SCENE_DESCRIPTIONS:
            # 80%的概率是SCENE_DESCRIPTIONS[scene_label][-2]，20%的概率是随机选
            random_prob = random.random() # [0.0, 1.0)
            if  random_prob < 0.8:
                scene_text_raw = SCENE_DESCRIPTIONS[scene_label][-2]
            else:
                scene_text_raw = random.choice(SCENE_DESCRIPTIONS[scene_label])
            # 随机选一条，增加数据的多样性
            # scene_text_raw = random.choice(SCENE_DESCRIPTIONS[scene_label])
            # scene_text_raw = scene_text_raw
        else:
            # Fallback
            scene_text_raw = f"A person moving in {scene_label} environment."
          
        # 4. 为了兼容 MCM-LDM 的 collate_fn 和 pipeline，我们需要填充一些 Dummy 数据
        # MCM-LDM 需要: word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, tokens
        # 我们的 Scene Text 用 CLIP 处理，不需要 word_embeddings (GloVe)
        # 但为了不报错，我们填一些假的
        
        dummy_w_emb = torch.zeros(1, 300) # 假设 dim=300
        dummy_pos_oh = torch.zeros(1, 15)
        dummy_tokens = "sos/OTHER eos/OTHER"

        scene_dir = os.path.join(self.scene_image_dir, scene_label)
        has_image = False
        scene_image = torch.zeros(3, 224, 224) # 默认全黑
        
        # if os.path.exists(scene_dir):
        #     files = [f for f in os.listdir(scene_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        #     if len(files) > 0:
        #         # 随机选一张
        #         img_name = random.choice(files)
        #         try:
        #             img = Image.open(os.path.join(scene_dir, img_name)).convert("RGB")
        #             scene_image = self.image_transform(img)
        #             has_image = True
        #         except:
        #             pass # 读取失败就还是全黑
        has_image = False
        
        # -----------------------------------------------------------
        # 返回值 (Tuple) - 注意顺序！
        # 前 7 个是 MCM-LDM 原始需要的 (我们尽量给它填上，虽然可能不用)
        # 后 2 个是我们新增的 (Scene Info)
        # -----------------------------------------------------------
        return (
            dummy_w_emb,       # 0. word_embeddings (Unused for Scene)
            dummy_pos_oh,      # 1. pos_one_hots (Unused)
            scene_text_raw,        # 2. caption (这里我们直接把 Scene Text 给进去，方便查看)
            len(scene_text_raw),   # 3. sent_len (String length, not token length, but fine)
            motion,            # 4. motion (The Real Data)
            m_length,          # 5. m_length
            dummy_tokens,      # 6. tokens (Unused)
            scene_text_raw,        # 7. [NEW] scene_text_raw (Explicitly for CLIP)
            scene_image,        # 8. [NEW] scene_image_tensor
            scene_id,
            has_image,
        )


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
        current_motion_name = self.name_list[idx] 
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

        # --- 增加安全性检查：防止 m_length 变成 0 或负数 ---
        if m_length <= 0: 
            raise ValueError("Motion length is too short after crop, motion is: ", current_motion_name)
        
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
        if np.any(np.isnan(motion)) or np.any(np.isinf(motion)):
            print(f"[Warning] NaN/Inf found in motion: {current_motion_name}, skipping and retrying...")
            # 随机换一个索引重试
            return self.__getitem__(random.randint(0, len(self) - 1))

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