import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Metric

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
            losses.append("x_loss")
            losses.append("style_loss")
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
                self._params[loss] = 1
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
            # predict noise
            # --- [核心修复] 在这里实现损失加权 ---
        
            # 1. 检查 rs_set 中是否存在我们传递的掩码
            if "is_text_guided_mask" in rs_set:
                mask = rs_set["is_text_guided_mask"]
                
                # 只有当 batch 中确实存在两种类型的样本时，才进行加权
                if mask.any() and (~mask).any():
                    # a. 提取需要的数据
                    noise_pred = rs_set['noise_pred']
                    noise = rs_set['noise']
                    
                    # b. 分别计算两种任务的 MSE Loss (不加权)
                    loss_text_guided = self._losses_func["inst_loss"](noise_pred[mask], noise[mask])
                    loss_motion_guided = self._losses_func["inst_loss"](noise_pred[~mask], noise[~mask])

                    # c. 从配置中获取权重 lambda
                    lambda_text_style = self.cfg.LOSS.get('LAMBDA_TEXT_STYLE', 1.0)
                    
                    # d. [关键] 计算加权后的总 diffusion loss
                    # 我们只对 text-guided 部分的损失应用额外的权重
                    # _params["inst_loss"] 通常是 1.0
                    diffusion_loss = self._params["inst_loss"] * (loss_motion_guided + lambda_text_style * loss_text_guided)
                    
                    # e. 将计算好的损失累加到 total，并手动更新 inst_loss 的状态用于日志记录
                    total += diffusion_loss
                    self.inst_loss += diffusion_loss.detach()
                    
                else:
                    diffusion_loss = self._losses_func["inst_loss"](rs_set['noise_pred'], rs_set['noise'])
                    # 如果 batch 中只有一种数据，则走原始的、不加权的逻辑
                    total += self._update_loss("inst_loss", rs_set['noise_pred'], rs_set['noise'])

            else:
                # [核心修复] 在这个分支中也计算并赋值 diffusion_loss
                diffusion_loss = self._losses_func["inst_loss"](rs_set['noise_pred'], rs_set['noise'])
                # 如果没有掩码 (例如在验证时)，则走原始的、不加权的逻辑
                total += self._update_loss("inst_loss", rs_set['noise_pred'], rs_set['noise'])
            # total += self._update_loss("inst_loss", rs_set['noise_pred'],
            #                                rs_set['noise'])

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
        return {"total": total, "diffusion": diffusion_loss.detach()}

    def compute(self, split):
        count = getattr(self, "count")
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
