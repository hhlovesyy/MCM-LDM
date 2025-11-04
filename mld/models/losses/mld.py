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
            if self.cfg.LOSS.get('LAMBDA_STYLE_RECON', 0.0) > 0.0:
                losses.append("style_recon_loss")
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
            elif loss.split('_')[0] == 'style' and loss.split('_')[1] == 'recon':
                # 感知损失通常使用 L1 或 L2 loss。L1 (SmoothL1) 对异常值更鲁棒。
                self._losses_func[loss] = nn.SmoothL1Loss(reduction='mean')
                self._params[loss] = self.cfg.LOSS.LAMBDA_STYLE_RECON
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
        
        # Compute instance loss
        if self.stage in ["vae", "vae_diffusion"]:
            total += self._update_loss("recons_feature", rs_set['m_rst'],
                                       rs_set['m_ref'])
            total += self._update_loss("recons_joints", rs_set['joints_rst'],
                                       rs_set['joints_ref'])
            total += self._update_loss("kl_motion", rs_set['dist_m'], rs_set['dist_ref'])

        if self.stage in ["diffusion", "vae_diffusion"]:
            weighted_inst_loss = self._update_loss("inst_loss", rs_set['noise_pred'], rs_set['noise'])
            total += weighted_inst_loss

            if 'text_style_emb' in rs_set and 'style_ids' in rs_set and rs_set['text_style_emb'] is not None:
                text_embeddings = rs_set['text_style_emb']
                motion_embeddings = rs_set['motion_style_emb_for_text']
                style_ids = rs_set['style_ids']   # torch.Size([batch_size / 2])
                
                # 安全检查，确保有样本进行对比
                if text_embeddings.shape[0] > 1:  
                    # 1. 温度参数 (可学习或固定)
                    # 你也可以在这里定义 self.logit_scale = nn.Parameter(...)
                    logit_scale = torch.exp(torch.tensor(2.659, device=self.device))

                    # 2. 特征归一化
                    text_embeds_norm = F.normalize(text_embeddings, p=2, dim=-1)
                    motion_embeds_norm = F.normalize(motion_embeddings, p=2, dim=-1)

                    # 3. 计算 text-motion 相似度矩阵 (Logits)
                    logits_per_text = torch.matmul(text_embeds_norm, motion_embeds_norm.t()) * logit_scale

                    # 4. 新逻辑: 创建 Ground Truth 正样本掩码
                    #    如果样本i和样本j的style_id相同，则 mask[i, j] = 1.0
                    ground_truth_mask = (style_ids.view(-1, 1) == style_ids.view(1, -1)).float()

                    # # --- 将 Tensor 保存到文件的代码从这里开始 ---

                    # output_filepath = "dumptensor.txt"

                    # # 1. 将 PyTorch Tensor 转移到 CPU 并转换为 NumPy 数组
                    # #    如果 Tensor 在 GPU 上，必须先调用 .cpu()
                    # mask_np = ground_truth_mask.cpu().numpy()

                    # print(f"Ground Truth Mask Shape: {mask_np.shape}")
                    # print(f"Saving data to {output_filepath}...")

                    # # 2. 将矩阵内容按行写入文件
                    # with open(output_filepath, 'w') as f:
                    #     # 遍历矩阵的每一行
                    #     for i in range(mask_np.shape[0]):
                            
                    #         # 将当前行（都是 0 或 1）的元素格式化为不带小数点的整数字符串 ('{:.0f}')，并用空格分隔
                    #         row_str = ' '.join(f"{x:.0f}" for x in mask_np[i])
                            
                    #         # 写入文件，并在行尾添加换行符
                    #         f.write(row_str + '\n')

                    # print(f"Successfully saved the tensor to {output_filepath}")

                    # 5. 新逻辑: 手动计算对称交叉熵损失
                    #    - F.log_softmax(logits, dim=1) 是一种数值稳定的计算方式
                    loss_t2m = - F.log_softmax(logits_per_text, dim=1)
                    #    用掩码过滤，只对正样本求和，然后除以每行的正样本数量
                    loss_t2m = (loss_t2m * ground_truth_mask).sum(dim=1) / ground_truth_mask.sum(dim=1)
                    
                    # 计算 motion-to-text 的损失
                    loss_m2t = - F.log_softmax(logits_per_text.t(), dim=1)
                    loss_m2t = (loss_m2t * ground_truth_mask.t()).sum(dim=1) / ground_truth_mask.t().sum(dim=1)
                    
                    # 计算最终的对比损失
                    contrastive_loss = (loss_t2m.mean() + loss_m2t.mean()) / 2.0
                    
                    # 6. 更新 metric 状态并累加到 total loss
                    self.align_loss += contrastive_loss.detach()
                    self.align_loss_text += loss_t2m.mean().detach()
                    self.align_loss_motion += loss_m2t.mean().detach()
                    
                    weighted_align_loss = self._params["align_loss"] * contrastive_loss
                    total += weighted_align_loss
            # ===> END: 替换代码 (最终修正版) <===
            # style loss
            # total += self._update_loss("style_loss", rs_set['gen_motion_feature'],
            #                                rs_set['gt_motion_feature'])
                    
            # --- [PLAN C - Step 3] 计算 style_recon_loss ---
            weighted_style_recon_loss = torch.tensor(0.0, device=self.device)
            # 只有当 mld.py 传递了这些键时，才进行计算
            if 'gt_features' in rs_set and 'pred_features' in rs_set:
                print(f"[DEBUG PROBE] --- Loss function received features for style_recon_loss. ---")
                gt_features = rs_set['gt_features']
                pred_features = rs_set['pred_features']
                
                # 初始化一个累加器
                recon_loss = torch.tensor(0.0, device=self.device)
                num_layers = len(gt_features)

                # 逐层计算 L1 损失并累加
                for i in range(num_layers):
                    # 我们不需要应用掩码，因为在 mld.py 中，送入 teacher 的动作
                    # 已经是去除了 padding 的，或者即使有 padding，L1 loss 对零填充区域的贡献也是零。
                    # 为了安全，也可以在这里应用 rs_set['style_recon_mask']
                    recon_loss += self._losses_func["style_recon_loss"](pred_features[i], gt_features[i])
                
                # 对所有层的损失取平均
                recon_loss = recon_loss / num_layers
                
                # 更新 metric 状态并获取加权损失
                # self._update_loss 内部会自动处理 detach 和加权
                weighted_style_recon_loss = self._update_loss("style_recon_loss", recon_loss, torch.tensor(0.0, device=self.device))
                total += weighted_style_recon_loss
            # ------------------------------------------------
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

        # 3. 准备用于实时日志 (step-level logging) 的字典
        log_dict = {"total": total.detach()} # 直接记录当前 step 的 total loss，比用累加/平均更实时

        # 添加独立的、未加权的损失项
        if self.inst_loss > 0: # 使用 self.inst_loss 而不是 rs_set['noise']
            log_dict['inst_unweighted'] = self.inst_loss / self.count

        if 'contrastive_loss' in locals():
            log_dict['align_unweighted'] = contrastive_loss.detach()
            
        # 添加加权的损失项
        log_dict['inst_weighted'] = weighted_inst_loss.detach()
        if 'weighted_align_loss' in locals():
            log_dict['align_weighted'] = weighted_align_loss.detach()
        
        # [新增] 将 style_recon_loss 也添加到日志
        if 'recon_loss' in locals():
            log_dict['style_recon_unweighted'] = recon_loss.detach()
            log_dict['style_recon_weighted'] = weighted_style_recon_loss.detach()

        # 返回总损失和日志字典
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
