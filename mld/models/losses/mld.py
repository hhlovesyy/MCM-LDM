import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Metric
import torch.nn.functional as F
from mld.data.humanml.scripts.motion_process import (qrot,
                                                     recover_root_rot_pos)


class MLDLosses(Metric):
    """
    MLD Loss
    """

    def __init__(self, vae, mode, cfg):
        super().__init__(dist_sync_on_step=cfg.LOSS.DIST_SYNC_ON_STEP)

        # Save parameters
        # self.vae = vae
        self.mode = mode
        self.cfg = cfg
        self.stage = cfg.TRAIN.STAGE

        losses = []

        # diffusion loss
        if self.stage in ['diffusion', 'vae_diffusion']:
            # instance noise loss
            losses.append("inst_loss")
            losses.append("align_loss")
            losses.append("align_loss_text") # text -> motion 方向
            losses.append("align_loss_motion") # motion -> text 方向
            # losses.append("x_loss")
            # losses.append("style_loss")
            if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
                # prior noise loss
                losses.append("prior_loss")

        if self.stage in ['vae', 'vae_diffusion']:
            # reconstruction loss
            losses.append("recons_feature")
            losses.append("recons_verts")
            losses.append("recons_joints")
            losses.append("recons_limb")

            losses.append("gen_feature")
            losses.append("gen_joints")

            # KL loss
            losses.append("kl_motion")

        if self.stage not in ['vae', 'diffusion', 'vae_diffusion']:
            raise ValueError(f"Stage {self.stage} not supported")

        losses.append("total")

        for loss in losses:
            self.add_state(loss,
                           default=torch.tensor(0.0),
                           dist_reduce_fx="sum")
            # self.register_buffer(loss, torch.tensor(0.0))
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")
        self.losses = losses

        self._losses_func = {}
        self._params = {}
        for loss in losses:
            if loss.split('_')[0] == 'inst':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = 1.0
            # [核心修改] 添加 align_loss 的定义
            elif loss.split('_')[0] == 'align':
                # 核心思路：我们使用 CLIP-style 的对比损失，它由两个交叉熵损失构成。
                self._losses_func[loss] = nn.CrossEntropyLoss()
                # 从配置文件中读取 align_loss 的权重 lambda
                self._params[loss] = self.cfg.LOSS.get('LAMBDA_ALIGN', 1.0)
            elif loss.split('_')[0] == 'x':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = 1
            #设置style loss使用的函数以及权重
            elif loss.split('_')[0] == 'style':
                self._losses_func[loss] = nn.CosineEmbeddingLoss(reduction='mean')
                self._params[loss] = 0.07
            elif loss.split('_')[0] == 'prior':
                self._losses_func[loss] = nn.MSELoss(reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_PRIOR
            if loss.split('_')[0] == 'kl':
                if cfg.LOSS.LAMBDA_KL != 0.0:
                    self._losses_func[loss] = KLLoss()
                    self._params[loss] = cfg.LOSS.LAMBDA_KL
            elif loss.split('_')[0] == 'recons':
                self._losses_func[loss] = torch.nn.SmoothL1Loss(
                    reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_REC
            elif loss.split('_')[0] == 'gen':
                self._losses_func[loss] = torch.nn.SmoothL1Loss(
                    reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_GEN
            elif loss.split('_')[0] == 'latent':
                self._losses_func[loss] = torch.nn.SmoothL1Loss(
                    reduction='mean')
                self._params[loss] = cfg.LOSS.LAMBDA_LATENT
            else:
                ValueError("This loss is not recognized.")
            if loss.split('_')[-1] == 'joints':
                self._params[loss] = cfg.LOSS.LAMBDA_JOINT
        # self._params["align_loss_text"] = self._params["align_loss"] # _params是保存的参数，不是loss值，是一些比如lambda值之类的
        # self._params["align_loss_motion"] = self._params["align_loss"]

    def update(self, rs_set):
        total: float = 0.0
        diffusion_loss = torch.tensor(0.0, device=self.device)
        # Compute the losses
        # Compute instance loss
        if self.stage in ["vae", "vae_diffusion"]:
            total += self._update_loss("recons_feature", rs_set['m_rst'],
                                       rs_set['m_ref'])
            total += self._update_loss("recons_joints", rs_set['joints_rst'],
                                       rs_set['joints_ref'])
            total += self._update_loss("kl_motion", rs_set['dist_m'], rs_set['dist_ref'])

        if self.stage in ["diffusion", "vae_diffusion"]:
            total += self._update_loss("inst_loss", rs_set['noise_pred'], rs_set['noise'])

            # 2. [核心修改] 计算 align_loss (对齐损失)
            # 核心思路：只有在 rs_set 中包含了我们从 mld.py 传递过来的、用于对齐的两个 embedding 时，才计算此损失。
            # 这确保了只有在处理 100Style 的文本引导数据时，才会激活对齐损失。
            if 'text_style_emb' in rs_set and rs_set['text_style_emb'] is not None:
                text_embeddings = rs_set['text_style_emb']
                motion_embeddings = rs_set['motion_style_emb_for_text']
                
                # [安全检查] 确保我们有超过1个样本，否则对比学习没有意义
                if text_embeddings.shape[0] > 1:
                    # a. 温度参数 (logit_scale)
                    logit_scale = torch.exp(torch.tensor(2.659, device=self.device))

                    # b. 归一化 (Normalization)
                    text_embeds_norm = F.normalize(text_embeddings, p=2, dim=-1)
                    motion_embeds_norm = F.normalize(motion_embeddings, p=2, dim=-1)

                    # c. 计算 N_text x N_text 的相似度矩阵 (Logits Matrix)
                    logits_per_text = torch.matmul(text_embeds_norm, motion_embeds_norm.t()) * logit_scale
                    logits_per_motion = logits_per_text.t()

                    # d. 创建 Ground Truth 标签
                    num_text_samples = text_embeddings.shape[0]
                    ground_truth = torch.arange(num_text_samples, dtype=torch.long, device=self.device)

                    # e. 计算交叉熵损失
                    loss_text = self._losses_func["align_loss"](logits_per_text, ground_truth)
                    loss_motion = self._losses_func["align_loss"](logits_per_motion, ground_truth)
                    contrastive_loss = (loss_text + loss_motion) / 2.0
                    
                    # f. 更新和累加损失
                    self.align_loss += contrastive_loss.detach()
                    self.align_loss_text += loss_text.detach()       # <-- 新增
                    self.align_loss_motion += loss_motion.detach()   # <-- 新增
                    total += self._params["align_loss"] * contrastive_loss
            # ===> END: 替换代码 (最终修正版) <===
            # style loss
            # total += self._update_loss("style_loss", rs_set['gen_motion_feature'],
            #                                rs_set['gt_motion_feature'])
            if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
                # loss - prior loss
                total += self._update_loss("prior_loss", rs_set['noise_prior'],
                                           rs_set['dist_m1'])

        if self.stage in ["vae_diffusion"]:
            # loss
            # noise+text_emb => diff_reverse => latent => decode => motion
            total += self._update_loss("gen_feature", rs_set['gen_m_rst'],
                                       rs_set['m_ref'])
            total += self._update_loss("gen_joints", rs_set['gen_joints_rst'],
                                       rs_set['joints_ref'])

        self.total += total.detach()
        self.count += 1

        # [修改] 返回一个包含 'total' 键的字典，以确保与 allsplit_step 的日志记录代码兼容
        # [核心修改] 返回一个包含所有已计算损失项的字典
        log_dict = {"total": self.total / self.count}
        
        # 将其他你关心的损失也加入字典
        if 'inst_loss' in self.losses and self.inst_loss.item() != 0:
            log_dict['inst'] = self.inst_loss.detach() / self.count # 计算平均值
        if 'align_loss' in self.losses and self.align_loss.item() != 0:
            log_dict['align_total'] = self.align_loss.detach() / self.count
            log_dict['align_text'] = self.align_loss_text.detach() / self.count
            log_dict['align_motion'] = self.align_loss_motion.detach() / self.count

        # update 函数本身返回用于反向传播的总损失张量
        return total, log_dict

    def compute(self, split):
        count = getattr(self, "count")
        # 避免除以零的错误
        if count == 0:
            print("出现了除以0的错误，请检查代码有什么问题，或者训练有什么安全隐患")
            return {loss: torch.tensor(0.0) for loss in self.losses}
        return {loss: getattr(self, loss) / count for loss in self.losses}

    def _update_loss(self, loss: str, outputs, inputs):
        # Update the loss
        if loss == 'style_loss':
            tar = torch.ones(inputs.shape[0]).to(inputs.device)
            val = self._losses_func[loss](outputs, inputs, tar)
        else:
            val = self._losses_func[loss](outputs, inputs)
        getattr(self, loss).__iadd__(val.detach())
        # Return a weighted sum
        weighted_loss = self._params[loss] * val
        return weighted_loss
    
    def _update_loss_cosine(self, loss: str, output1, output2, target):
        # 核心思路：这个函数专门处理需要三个输入的损失，如 CosineEmbeddingLoss。
        # 它的逻辑与 _update_loss 完全相同，只是调用损失函数时传递的参数不同。
        
        # 1. 计算损失值
        val = self._losses_func[loss](output1, output2, target)
        
        # 2. 更新累积状态 (用于日志记录)
        getattr(self, loss).__iadd__(val.detach())
        
        # 3. 返回加权后的损失值，用于累加到 total_loss
        weighted_loss = self._params[loss] * val
        return weighted_loss

    def loss2logname(self, loss: str, split: str):
        if loss == "total":
            log_name = f"{loss}/{split}"
        else:
            loss_type, name = loss.split("_")
            log_name = f"{loss_type}/{name}/{split}"
        return log_name

class KLLoss:

    def __init__(self):
        pass

    def __call__(self, q, p):
        div = torch.distributions.kl_divergence(q, p)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"


class KLLossMulti:

    def __init__(self):
        self.klloss = KLLoss()

    def __call__(self, qlist, plist):
        return sum([self.klloss(q, p) for q, p in zip(qlist, plist)])

    def __repr__(self):
        return "KLLossMulti()"
