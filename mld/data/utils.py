# import torch


# def lengths_to_mask(lengths):
#     max_len = max(lengths)
#     mask = torch.arange(max_len, device=lengths.device).expand(
#         len(lengths), max_len) < lengths.unsqueeze(1)
#     return mask


# # padding to max length in one batch
# def collate_tensors(batch):
#     dims = batch[0].dim()
#     max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
#     size = (len(batch), ) + tuple(max_size)
#     canvas = batch[0].new_zeros(size=size)
#     for i, b in enumerate(batch):
#         sub_tensor = canvas[i]
#         for d in range(dims):
#             sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
#         sub_tensor.add_(b)
#     return canvas


# def all_collate(batch):
#     notnone_batches = [b for b in batch if b is not None]
#     databatch = [b["motion"] for b in notnone_batches]
#     # labelbatch = [b['target'] for b in notnone_batches]
#     if "lengths" in notnone_batches[0]:
#         lenbatch = [b["lengths"] for b in notnone_batches]
#     else:
#         lenbatch = [len(b["inp"][0][0]) for b in notnone_batches]

#     databatchTensor = collate_tensors(databatch)
#     # labelbatchTensor = torch.as_tensor(labelbatch)
#     lenbatchTensor = torch.as_tensor(lenbatch)
#     maskbatchTensor = (lengths_to_mask(
#         lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1)
#                        )  # unqueeze for broadcasting

#     motion = databatchTensor
#     cond = {"y": {"mask": maskbatchTensor, "lengths": lenbatchTensor}}

#     if "text" in notnone_batches[0]:
#         textbatch = [b["text"] for b in notnone_batches]
#         cond["y"].update({"text": textbatch})

#     # collate action textual names
#     if "action_text" in notnone_batches[0]:
#         action_text = [b["action_text"] for b in notnone_batches]
#         cond["y"].update({"action_text": action_text})

#     return motion, cond


# # an adapter to our collate func
# def mld_collate(batch):
#     notnone_batches = [b for b in batch if b is not None]
#     notnone_batches.sort(key=lambda x: x[3], reverse=True)
#     # batch.sort(key=lambda x: x[3], reverse=True)
#     adapted_batch = {
#         "motion":
#         collate_tensors([torch.tensor(b[4]).float() for b in notnone_batches]),
#         "text": [b[2] for b in notnone_batches],
#         # "style_text": [b[-2] for b in notnone_batches],
#         # "motion_rot": collate_tensors([torch.tensor(b[-1]).float() for b in notnone_batches]),
#         "length": [b[5] for b in notnone_batches],
#         "word_embs":
#         collate_tensors([torch.tensor(b[0]).float() for b in notnone_batches]),
#         "pos_ohot":
#         collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
#         "text_len":
#         collate_tensors([torch.tensor(b[3]) for b in notnone_batches]),
#         "tokens": [b[6] for b in notnone_batches],
#     }
#     return adapted_batch


# def a2m_collate(batch):

#     databatch = [b[0] for b in batch]
#     labelbatch = [b[1] for b in batch]
#     lenbatch = [len(b[0][0][0]) for b in batch]
#     labeltextbatch = [b[3] for b in batch]

#     databatchTensor = collate_tensors(databatch)
#     labelbatchTensor = torch.as_tensor(labelbatch).unsqueeze(1)
#     lenbatchTensor = torch.as_tensor(lenbatch)

#     maskbatchTensor = lengths_to_mask(lenbatchTensor)
#     adapted_batch = {
#         "motion": databatchTensor.permute(0, 3, 2, 1).flatten(start_dim=2),
#         "action": labelbatchTensor,
#         "action_text": labeltextbatch,
#         "mask": maskbatchTensor,
#         "length": lenbatchTensor
#     }
#     return adapted_batch



import torch


def lengths_to_mask(lengths):
    max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


# padding to max length in one batch
def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def all_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b["motion"] for b in notnone_batches]
    # labelbatch = [b['target'] for b in notnone_batches]
    if "lengths" in notnone_batches[0]:
        lenbatch = [b["lengths"] for b in notnone_batches]
    else:
        lenbatch = [len(b["inp"][0][0]) for b in notnone_batches]

    databatchTensor = collate_tensors(databatch)
    # labelbatchTensor = torch.as_tensor(labelbatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = (lengths_to_mask(
        lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1)
                       )  # unqueeze for broadcasting

    motion = databatchTensor
    cond = {"y": {"mask": maskbatchTensor, "lengths": lenbatchTensor}}

    if "text" in notnone_batches[0]:
        textbatch = [b["text"] for b in notnone_batches]
        cond["y"].update({"text": textbatch})

    # collate action textual names
    if "action_text" in notnone_batches[0]:
        action_text = [b["action_text"] for b in notnone_batches]
        cond["y"].update({"action_text": action_text})

    return motion, cond

def scene_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    # 按动作长度排序
    notnone_batches.sort(key=lambda x: x[5], reverse=True)
    adapted_batch = {
        # 【修改点 1】：去掉 torch.tensor()，直接用 .float()
        # 因为在 Dataset.__getitem__ 里我们已经转成 Tensor 了
        "motion": collate_tensors([b[4].float() for b in notnone_batches]),
        
        "text": [b[2] for b in notnone_batches],
        "length": [b[5] for b in notnone_batches],
        
        # 【修改点 2】：同样去掉 torch.tensor()
        "word_embs": collate_tensors([b[0].float() for b in notnone_batches]),
        "pos_ohot": collate_tensors([b[1].float() for b in notnone_batches]),
        
        # text_len 是 int，这里可以用 torch.tensor 转一下，或者直接 stack
        "text_len": torch.tensor([b[3] for b in notnone_batches]),
        "tokens": [b[6] for b in notnone_batches],

        # -------------------------------------------------------
        # 【ICME 新增字段】：确保这些也在！不要漏了！
        # -------------------------------------------------------
        "scene_text": [b[7] for b in notnone_batches],  # List of strings
        "scene_image": torch.stack([b[8] for b in notnone_batches]), # Tensor stack
    }
    return adapted_batch

# an adapter to our collate func
def mld_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    notnone_batches.sort(key=lambda x: x[3], reverse=True)
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = {
        "motion":
        collate_tensors([torch.tensor(b[4]).float() for b in notnone_batches]),
        "text": [b[2] for b in notnone_batches],
        "length": [b[5] for b in notnone_batches],
        "word_embs":
        collate_tensors([torch.tensor(b[0]).float() for b in notnone_batches]),
        "pos_ohot":
        collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
        "text_len":
        collate_tensors([torch.tensor(b[3]) for b in notnone_batches]),
        "tokens": [b[6] for b in notnone_batches],
    }
    return adapted_batch


def a2m_collate(batch):

    databatch = [b[0] for b in batch]
    labelbatch = [b[1] for b in batch]
    lenbatch = [len(b[0][0][0]) for b in batch]
    labeltextbatch = [b[3] for b in batch]

    databatchTensor = collate_tensors(databatch)
    labelbatchTensor = torch.as_tensor(labelbatch).unsqueeze(1)
    lenbatchTensor = torch.as_tensor(lenbatch)

    maskbatchTensor = lengths_to_mask(lenbatchTensor)
    adapted_batch = {
        "motion": databatchTensor.permute(0, 3, 2, 1).flatten(start_dim=2),
        "action": labelbatchTensor,
        "action_text": labeltextbatch,
        "mask": maskbatchTensor,
        "length": lenbatchTensor
    }
    return adapted_batch