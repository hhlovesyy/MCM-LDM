import inspect
import os
from mld.transforms.rotation2xyz import Rotation2xyz
import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torchmetrics import MetricCollection
import time
from mld.config import instantiate_from_config
from os.path import join as pjoin
from mld.models.architectures import (
    mld_denoiser,
    mld_vae,
    t2m_motionenc,
    t2m_textenc,
)
from mld.models.losses.mld import MLDLosses
from mld.models.modeltype.base import BaseModel
from mld.utils.temos_utils import remove_padding
from mld.utils.temos_utils import lengths_to_mask
# from animate import plot_3d_motion
# for motionclip
import clip
from ..motionclip_263.utils.get_model_and_data import get_model_and_data
from ..motionclip_263.parser.visualize import parser
from ..motionclip_263.visualize.visualize import viz_clip_text, get_gpu_device
from ..motionclip_263.utils.misc import load_model_wo_clip
import yaml
def read_yaml_to_dict(yaml_path: str, ):
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value
    
# for skeleton transform
from datasets.utils.common.skeleton import Skeleton
import numpy as np
import os
from datasets.utils.common.quaternion import *
from datasets.utils.paramUtil import *
# import evaluate.utils.rotation_conversions as geometry

from transformers import CLIPTokenizer, CLIPTextModel
import torch.nn as nn



from .base import BaseModel


class MLD(BaseModel):
    """
    Stage 1 vae
    Stage 2 diffusion
    """

    def __init__(self, cfg, datamodule, **kwargs):
        super().__init__()

        self.cfg = cfg

        self.stage = cfg.TRAIN.STAGE
        self.is_vae = cfg.model.vae
        # self.predict_epsilon = cfg.TRAIN.ABLATION.PREDICT_EPSILON
        self.nfeats = cfg.DATASET.NFEATS
        self.njoints = cfg.DATASET.NJOINTS
        self.debug = cfg.DEBUG
        self.latent_dim = cfg.model.latent_dim
        self.guidance_scale = cfg.model.guidance_scale
        self.guidance_uncodp = cfg.model.guidance_uncondp
        self.datamodule = datamodule




        # self.text_encoder = instantiate_from_config(cfg.model.text_encoder)

        parameters = read_yaml_to_dict("configs/motionclip_config/motionclip_params_263.yaml")
        parameters["device"] = 'cuda:{}'.format(cfg["DEVICE"][0])        
        self.motionclip = get_model_and_data(parameters, split='vald')
        print("load motion clip-xyz-263")
        print("Restore weights..")
        checkpointpath = "checkpoints/motionclip_checkpoint/motionclip.pth.tar"
        state_dict = torch.load(checkpointpath, map_location=parameters["device"])
        load_model_wo_clip(self.motionclip, state_dict)

        self.mean = torch.tensor(self.datamodule.hparams.mean).to(parameters["device"])
        self.std = torch.tensor(self.datamodule.hparams.std).to(parameters["device"])

        #don't train motionclip
        self.motionclip.training = False
        for p in self.motionclip.parameters():
            p.requires_grad = False


        print("Loading CLIP for Scene Guidance...")
        import os
        local_clip_path = "/root/autodl-tmp/MyRepository/MCM-LDM/local_clip_model" 
        # 或者如果是相对路径: local_clip_path = "./local_clip_model"

        print(f"Loading CLIP from local path: {local_clip_path}")
        # 加载预训练模型
        self.scene_tokenizer = CLIPTokenizer.from_pretrained(local_clip_path)
        self.scene_text_encoder = CLIPTextModel.from_pretrained(local_clip_path)

        # 冻结参数（这一点非常重要，否则显存会爆，且难以训练）
        self.scene_text_encoder.eval()
        self.scene_text_encoder.requires_grad_(False)
        for p in self.scene_text_encoder.parameters():
            p.requires_grad = False
        
        # 作用：把 CLIP 的 512 维特征，映射到适合做 Motion FiLM 的 512 维空间
        # 这就是你说的 "CLIP 之后加几层 MLP"
        self.scene_projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512), # 输出给 FiLM MLP 用
            nn.LayerNorm(512)    # 再次 Norm 保证稳定
        )
        # 初始化 Projector
        # 让它初始接近 Identity，或者稍微有波动
        nn.init.xavier_uniform_(self.scene_projector[0].weight)
        nn.init.zeros_(self.scene_projector[-2].weight) # 最后一层 Linear 权重置0
        nn.init.zeros_(self.scene_projector[-2].bias)   # 这样初始输出接近 0 (经过Norm后会变)


        self.vae = instantiate_from_config(cfg.model.motion_vae)
        # Don't train the motion encoder and decoder
        if self.stage == "diffusion":
            self.vae.training = False
            for p in self.vae.parameters():
                p.requires_grad = False

        self.denoiser = instantiate_from_config(cfg.model.denoiser)

        self.scheduler = instantiate_from_config(cfg.model.scheduler)
        self.noise_scheduler = instantiate_from_config(
            cfg.model.noise_scheduler)

        self.num_scenes = 14
        self.scene_embedding_table = torch.nn.Embedding(self.num_scenes, 512)
        # 2. FiLM MLP (输入 512，输出 1024)
        self.film_mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 1024)
        )
        # 初始化为 Identity
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)

        self.scene_norm = nn.LayerNorm(512)
        # Output Norm (最关键！对齐分布！)
        self.style_norm = nn.LayerNorm(512)

        # self._get_t2m_evaluator(cfg)

        # if cfg.TRAIN.OPTIM.TYPE.lower() == "adamw":
        #     self.optimizer = AdamW(lr=cfg.TRAIN.OPTIM.LR,
        #                            params=self.parameters())
        # else:
        #     raise NotImplementedError(
        #         "Do not support other optimizer for now.")

        if cfg.LOSS.TYPE == "mld":
            self._losses = MetricCollection({
                split: MLDLosses(vae=self.is_vae, mode="xyz", cfg=cfg)
                for split in ["losses_train", "losses_test", "losses_val"]
            })
        else:
            raise NotImplementedError(
                "MotionCross model only supports mld losses.")

        self.losses = {
            key: self._losses["losses_" + key]
            for key in ["train", "test", "val"]
        }

        self.metrics_dict = cfg.METRIC.TYPE
        self.configure_metrics()

        # If we want to overide it at testing time
        self.sample_mean = False
        self.fact = None
        self.do_classifier_free_guidance = True

        self.feats2joints = datamodule.feats2joints
        self.joints2feats = datamodule.joints2feats

    def configure_optimizers(self):
        # 1. 慢速组：Denoiser (预训练好的)
        denoiser_params = list(self.denoiser.parameters())
        
        # 2. 快速组：新加入的所有模块 (需要大火猛炒)
        # 包括: FiLM MLP, Embedding, 以及两个 LayerNorm
        adapter_params = []

        # 添加 Projector
        if hasattr(self, "scene_projector"):
            adapter_params.extend(list(self.scene_projector.parameters()))
        
        # 显式添加模块，防止漏掉
        if hasattr(self, "film_mlp"):
            adapter_params.extend(list(self.film_mlp.parameters()))
            print(f"DEBUG: Added film_mlp params ({len(list(self.film_mlp.parameters()))} tensors)")
            
        if hasattr(self, "scene_embedding_table"):
            adapter_params.extend(list(self.scene_embedding_table.parameters()))
            print("DEBUG: Added scene_embedding_table params")

        if hasattr(self, "scene_norm"):
            adapter_params.extend(list(self.scene_norm.parameters()))
            
        if hasattr(self, "style_norm"):
            adapter_params.extend(list(self.style_norm.parameters()))

        # 3. 构造优化器
        optimizer = torch.optim.AdamW([
            # 旧参数：微火慢炖
            {"params": denoiser_params, "lr": 1e-5}, 
            
            # 新参数：大火爆炒 (这里放心地给大 LR)
            {"params": adapter_params, "lr": 5e-4}, 
        ], weight_decay=0.0)
        
        # 打印一下参数组长度，确认是否加上了
        print(f"Optimizer initialized. Fast Group Size: {len(adapter_params)}")
        
        return {"optimizer": optimizer}

    def _get_t2m_evaluator(self, cfg):
        """
        load T2M text encoder and motion encoder for evaluating
        """
        # init module
        self.t2m_textencoder = t2m_textenc.TextEncoderBiGRUCo(
            word_size=cfg.model.t2m_textencoder.dim_word,
            pos_size=cfg.model.t2m_textencoder.dim_pos_ohot,
            hidden_size=cfg.model.t2m_textencoder.dim_text_hidden,
            output_size=cfg.model.t2m_textencoder.dim_coemb_hidden,
        )

        self.t2m_moveencoder = t2m_motionenc.MovementConvEncoder(
            input_size=cfg.DATASET.NFEATS - 4,
            hidden_size=cfg.model.t2m_motionencoder.dim_move_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_move_latent,
        )

        self.t2m_motionencoder = t2m_motionenc.MotionEncoderBiGRUCo(
            input_size=cfg.model.t2m_motionencoder.dim_move_latent,
            hidden_size=cfg.model.t2m_motionencoder.dim_motion_hidden,
            output_size=cfg.model.t2m_motionencoder.dim_motion_latent,
        )
        # load pretrianed
        dataname = cfg.TEST.DATASETS[0]
        dataname = "t2m" if dataname == "humanml3d" else dataname
        t2m_checkpoint = torch.load(
            os.path.join(cfg.model.t2m_path, dataname,
                         "text_mot_match/model/finest.tar"))
        self.t2m_textencoder.load_state_dict(t2m_checkpoint["text_encoder"])
        self.t2m_moveencoder.load_state_dict(
            t2m_checkpoint["movement_encoder"])
        self.t2m_motionencoder.load_state_dict(
            t2m_checkpoint["motion_encoder"])

        # freeze params
        self.t2m_textencoder.eval()
        self.t2m_moveencoder.eval()
        self.t2m_motionencoder.eval()
        for p in self.t2m_textencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_moveencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_motionencoder.parameters():
            p.requires_grad = False










# 


    def sample_from_distribution(
        self,
        dist,
        *,
        fact=None,
        sample_mean=False,
    ) -> Tensor:
        fact = fact if fact is not None else self.fact
        sample_mean = sample_mean if sample_mean is not None else self.sample_mean

        if sample_mean:
            return dist.loc.unsqueeze(0)

        # Reparameterization trick
        if fact is None:
            return dist.rsample().unsqueeze(0)

        # Resclale the eps
        eps = dist.rsample() - dist.loc
        z = dist.loc + fact * eps

        # add latent size
        z = z.unsqueeze(0)
        return z
    
# test
    def forward(self, batch):

        lengths = batch["length"]
        # style
        motion = batch["style_motion"].clone()
        motion[...,:3] = 0


        # content
        content_motion = batch['content_motion']
        content_motion = (content_motion - self.mean.to(content_motion.device))/self.std.to(content_motion.device)

        # trajectory
        trans_motion = content_motion.clone() # [NOTE]原来的仓库也是把归一化之后的轨迹做处理，但是归一化之后的轨迹是不是丢失了本来的一些信息？用还原之后的原来动作的trajectory会不会更好？训练和推理都是
        # 
        content_motion[...,:3] = 0


        scale = batch["tag_scale"]
        lengths1 = [content_motion.shape[1]]* content_motion.shape[0]
        
        if self.cfg.TEST.COUNT_TIME:
            self.starttime = time.time()
            
        if self.stage in ['diffusion', 'vae_diffusion']:\
            #add style text in test
            
            
            # content motion
            with torch.no_grad():
                z, dist_m = self.vae.encode(content_motion.float(), lengths1)
            uncond_tokens = torch.cat([z, z], dim = 1).permute(1,0,2)
            motion_emb_content = uncond_tokens

            # style motion
            lengths11 = [motion.shape[1]]* motion.shape[0]

# for motion input (bs,60,22,3)->(bs,22,3,60)
            # motion_seq = feats_ref*std + mean
            motion_seq = motion.unsqueeze(-1).permute(0,2,3,1)


            motion_emb = self.motionclip.encoder({'x': motion_seq.float(),
                            'y': torch.zeros(motion_seq.shape[0], dtype=int, device=motion_seq.device),
                            'mask': lengths_to_mask(lengths11, device=motion_seq.device)})["mu"]
            motion_emb = motion_emb.unsqueeze(1)

            # trajectory
            trans_cond = trans_motion[...,:3]
            trans_uncond = torch.zeros(trans_cond.shape).to(motion_seq.device)
            # uncond_trans = torch.cat([trans_cond, trans_cond], dim = 0)
            uncond_trans = torch.cat([trans_uncond, trans_uncond], dim=0)
            # uncond_trans = torch.cat([trans_uncond, trans_cond], dim=0)

            scene_texts = batch.get("scene_text", ["A person moving in a normal environment"] * len(lengths))
            with torch.no_grad():
                text_inputs = self.scene_tokenizer(scene_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                clip_feat = self.scene_text_encoder(**text_inputs).pooler_output # [B, 512]
            scene_feat = self.scene_projector(clip_feat) # [Batch, 512]
            scene_feat_norm = self.scene_norm(scene_feat)

            # scene_feat_norm = self.scene_norm(scene_feat)
            film_params = self.film_mlp(scene_feat_norm)
            gamma_raw, beta_raw = film_params.chunk(2, dim=-1)
            gamma = (1.0 + torch.tanh(gamma_raw)).unsqueeze(1) # [B, 1, 512]
            beta = beta_raw.unsqueeze(1)                       # [B, 1, 512]
            motion_emb = torch.zeros_like(motion_emb) # 先试试不要style
            
            filmed_emb = gamma * motion_emb + beta
            adapted_style_normed = self.style_norm(filmed_emb)

            # D. 构造 CFG 输入
            # Uncond 分支：给全 0 (代表"无风格")
            # Cond 分支：给 Adapted Style
            uncond_style = torch.zeros_like(adapted_style_normed)
            
            # 拼接顺序：[Uncond, Cond], 这个是场景指导后的style
            motion_emb_cfg = torch.cat([uncond_style, adapted_style_normed], dim=0)
 

            scene_feat_reshaped = scene_feat.unsqueeze(1)
            uncond_scene = torch.zeros_like(scene_feat_reshaped)
            scene_emb_cfg = torch.cat([uncond_scene, scene_feat_reshaped], dim=0)  # 其实目前的去噪网络用不到这一项，以防后面可能需要就放进来了

            # three conditions
            multi_cond_emb = [motion_emb_content, motion_emb_cfg, uncond_trans, scene_emb_cfg]


            z = self._diffusion_reverse(multi_cond_emb, lengths, scale)

        elif self.stage in ['vae']:
            motions = batch['motion']
            z, dist_m = self.vae.encode(motions, lengths)

        with torch.no_grad():
            feats_rst = self.vae.decode(z, lengths)
            # feats_rst[...,:3] = trans_motion[...,:3] # if copy trajectory

        joints = self.feats2joints(feats_rst.detach().cpu())

        return remove_padding(joints, lengths)
    


    def _diffusion_reverse(self, encoder_hidden_states, lengths=None, scale=None):
        # init latents
        bsz = encoder_hidden_states[0].shape[0]
        if self.do_classifier_free_guidance:
            bsz = bsz // 2

        latents = torch.randn(
            (bsz, self.latent_dim[0], self.latent_dim[-1]),
            device=encoder_hidden_states[0].device,
            dtype=torch.float,
        )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(
            self.cfg.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(encoder_hidden_states[0].device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta
        

        # reverse
        for i, t in enumerate(timesteps):
            # if i // 10 ==0:
            #     latent_feature.append()
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (torch.cat(
                [latents] *
                2) if self.do_classifier_free_guidance else latents)
            lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance
                               else lengths)
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                lengths=lengths_reverse,
            )[0]
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + scale * (
                    noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents,
                                              **extra_step_kwargs).prev_sample

        latents = latents.permute(1, 0, 2)
        return latents





    def _diffusion_process(self, latents, encoder_hidden_states, lengths=None):
        """
        heavily from https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
        """
        # our latent   [batch_size, n_token=1 or 5 or 10, latent_dim=256]
        # sd  latent   [batch_size, [n_token0=64,n_token1=64], latent_dim=4]
        # [n_token, batch_size, latent_dim] -> [batch_size, n_token, latent_dim]
        latents = latents.permute(1, 0, 2) # torch.Size([32, 7, 256])

        # Sample noise that we'll add to the latents
        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz, ),
            device=latents.device,
        )
        timesteps = timesteps.long()  # torch.Size([32])
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents.clone(), noise,
                                                       timesteps)  # torch.Size([32, 7, 256])
        # Predict the noise residual
        noise_pred = self.denoiser(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            lengths=lengths,
            return_dict=False,
        )[0] # torch.Size([32, 7, 256])
        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
        if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
            noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
            noise, noise_prior = torch.chunk(noise, 2, dim=0)
        else:
            noise_pred_prior = 0
            noise_prior = 0


        n_set = {
            "noise": noise, # torch.Size([32, 7, 256])
            "noise_prior": noise_prior,
            "noise_pred": noise_pred, # torch.Size([32, 7, 256])
            "noise_pred_prior": noise_pred_prior,
        }

        return n_set

    def train_vae_forward(self, batch):
        feats_ref = batch["motion"]
        lengths = batch["length"]


        motion_z, dist_m = self.vae.encode(feats_ref, lengths)##########(1,128,256)/
        feats_rst = self.vae.decode(motion_z, lengths)


        # prepare for metric
        recons_z, dist_rm = self.vae.encode(feats_rst, lengths)

        # joints recover
        joints_rst = self.feats2joints(feats_rst)
        joints_ref = self.feats2joints(feats_ref)

        if dist_m is not None:
            if self.is_vae:
                # Create a centred normal distribution to compare with
                mu_ref = torch.zeros_like(dist_m.loc)
                scale_ref = torch.ones_like(dist_m.scale)
                dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
            else:
                dist_ref = dist_m

        # cut longer part over max length
        min_len = min(feats_ref.shape[1], feats_rst.shape[1])
        rs_set = {
            "m_ref": feats_ref[:, :min_len, :],
            "m_rst": feats_rst[:, :min_len, :],
            # [bs, ntoken, nfeats]<= [ntoken, bs, nfeats]
            "lat_m": motion_z.permute(1, 0, 2),
            "lat_rm": recons_z.permute(1, 0, 2),
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "dist_m": dist_m,
            "dist_ref": dist_ref,
        }
        return rs_set
# train
    def train_diffusion_forward(self, batch):
        feats_ref = batch["motion"] # torch.Size([32, 40, 263])
        feats_content = batch["motion"].clone() # torch.Size([32, 40, 263])
        feats_content[...,:3] = 0.0
        lengths = batch["length"]
        
        # content condition
        with torch.no_grad():
            z, dist = self.vae.encode(feats_ref, lengths) # z:torch.Size([7, 32, 256]), dist: torch.Size([7, 32, 256])
            z_content, dist = self.vae.encode(feats_content, lengths)
            cond_emb = z_content.permute(1,0,2)  # torch.Size([32, 7, 256])          
        # style condition
        motion_seq = feats_ref*self.std + self.mean  # 【QUESTION】看起来风格有做一步反归一化处理，而content并没有被反归一化，是这样么？
        motion_seq[...,:3]=0.0
        motion_seq = motion_seq.unsqueeze(-1).permute(0,2,3,1) # torch.Size([32, 263, 1, 40])
        motion_emb = self.motionclip.encoder({'x': motion_seq,
                        'y': torch.zeros(motion_seq.shape[0], dtype=int, device='cuda:{}'.format(self.cfg["DEVICE"][0])),
                        'mask': lengths_to_mask(lengths, device='cuda:{}'.format(self.cfg["DEVICE"][0]))})["mu"] # 一个style被提取成了512维的tensor，torch.Size([32, 512])
        motion_emb = motion_emb.unsqueeze(1) # torch.Size([32, 1, 512])
        mask_uncond = torch.rand(motion_emb.shape[0]) < self.guidance_uncodp # [F,F,...,F,T,F,T,...]
        motion_emb[mask_uncond, ...] = 0 # 1.没有风格的学习（无style motion）
        

        # trans condition
        trans_cond = batch["motion"][...,:3]  # torch.Size([32, 40, 3])
        traj_drop_mask = torch.rand(trans_cond.shape[0], 1, 1, device=trans_cond.device) > 0.5
        trans_cond = trans_cond * traj_drop_mask  # 尝试让生成的动作“偏离轨迹”，看一下效果# 2.没有轨迹的学习

        scene_texts = batch.get("scene_text", [""] * len(batch["length"]))
        # 2. CLIP 编码
        with torch.no_grad():
            text_inputs = self.scene_tokenizer(scene_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            clip_feat = self.scene_text_encoder(**text_inputs).pooler_output 
            # 取 [EOS] token 特征 [Batch, 512]
        scene_feat = self.scene_projector(clip_feat) # [Batch, 512]

        # scene_ids = batch["scene_id"] # [Batch]
    
        # # 直接查表
        # scene_feat = self.scene_embedding_table(scene_ids) # [Batch, 512]
        scene_feat = scene_feat.unsqueeze(1)  # [Batch, 1, 512]
        mask_scene = torch.rand(scene_feat.shape[0], device=scene_feat.device) < 0.2 # 大部分都是False，少量是True
        scene_feat[mask_scene] = 0  # 3.还有一定概率，场景没有

        # =================================================================
        # 【调试代码】打印统计信息 (只在第0个Epoch的前5个Step打印)
        # =================================================================
        if self.current_epoch == 100 and self.global_step < 5:
            print(f"\n[DEBUG STATS] Step {self.global_step}")
            
            # 1. 打印原始 MotionCLIP 输出 (Style)
            # 看看"老大哥"原本长什么样
            m_mean = motion_emb.mean().item()
            m_std = motion_emb.std().item()
            m_min, m_max = motion_emb.min().item(), motion_emb.max().item()
            print(f"  > [Style Raw]  Mean: {m_mean:.4f} | Std: {m_std:.4f} | Range: [{m_min:.4f}, {m_max:.4f}]")
            
            # 2. 打印原始 Scene Embedding
            # 看看"新来的"是不是太弱了
            s_mean = scene_feat.mean().item()
            s_std = scene_feat.std().item()
            print(f"  > [Scene Raw]  Mean: {s_mean:.4f} | Std: {s_std:.4f} (Is this too small?)")

        # =================================================================
        
        # 3. 生成 FiLM 参数
        scene_feat_norm = self.scene_norm(scene_feat)  # 这句归一化应该是必不可少的，不然尺度都对不上

        # =================================================================
        # 【调试代码】打印 Norm 后的 Scene
        # =================================================================
        if self.current_epoch == 100 and self.global_step < 5:
            sn_mean = scene_feat_norm.mean().item()
            sn_std = scene_feat_norm.std().item()
            print(f"  > [Scene Norm] Mean: {sn_mean:.4f} | Std: {sn_std:.4f} (Magic happened here!)")
        # =================================================================

        film_params = self.film_mlp(scene_feat_norm)  # torch.Size([32, 1, 1024])
        gamma_raw, beta_raw = film_params.chunk(2, dim=-1)  # 两个都是torch.Size([32, 1, 512])
        
        # 维度对齐 [Batch, 1, 512]，在刚开始train的时候，gamma是1，beta是0，相当于一个zero映射，不破坏网络本来的学习
        gamma = (1.0 + torch.tanh(gamma_raw)) # [Batch, 1, 512]
        beta = beta_raw                 # [Batch, 1, 512]

        # 执行融合 (先 Mask Style，后 FiLM)
        # 1. 如果 Style 被 Mask，Scene 没被 Mask -> Output = Beta (纯场景)
        # 2. 如果 Style 没 Mask，Scene 被 Mask -> Output = Style (纯风格)
        # 3. 如果都存在 -> Output = Modulated Style (融合)
        # 4. 如果都 Mask -> Output = 0 (无条件)
        adapted_style_emb = gamma * motion_emb + beta  # torch.Size([32, 1, 512])
        adapted_style_emb = self.style_norm(adapted_style_emb)

        # =================================================================
        # 【调试代码】打印最终融合后的 Style
        # =================================================================
        if self.current_epoch == 100 and self.global_step < 5:
            final_mean = adapted_style_emb.mean().item()
            final_std = adapted_style_emb.std().item()
            print(f"  > [Style Final] Mean: {final_mean:.4f} | Std: {final_std:.4f}")
            print(f"  > (Check if Style Final's Std is close to Style Raw's Std)")
            print("-----------------------------------------------------------\n")
        # =================================================================

        # three condition
        # multi_cond_emb = [cond_emb, motion_emb, trans_cond] # 复习一下： cond_emb：内容（torch.Size([32, 7, 256])），motion_emb：风格（torch.Size([32, 1, 512])），trans_cond：轨迹（torch.Size([32, 40, 3])）
        # scene_emb: 我们新增的场景的自然语言描述：torch.Size([32, 1, 512])
        multi_cond_emb = [cond_emb, adapted_style_emb, trans_cond, adapted_style_emb]  

        # diffusion process return with noise and noise_pred
        n_set = self._diffusion_process(z, multi_cond_emb, lengths) # 返回的n_set是一个字段，包含计算loss的时候pytorch_lightning所关心的内容
        return {**n_set}



# 
# evaluate the reconstruction in training time
# 

    def t2m_eval(self, batch):
        texts = batch["text"]
        motions = batch["motion"].detach().clone()

        # content 
        content_motions = batch["motion"].detach().clone()
        content_motions[...,:3] = 0.0

        lengths = batch["length"]
        word_embs = batch["word_embs"].detach().clone()
        pos_ohot = batch["pos_ohot"].detach().clone()
        text_lengths = batch["text_len"].detach().clone()

        motion = batch["motion"].detach().clone()
        

        # start
        start = time.time()

        if self.trainer.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            style_texts = style_texts * self.cfg.TEST.MM_NUM_REPEATS
            motions = motions.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            motion = motion.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS
            word_embs = word_embs.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS,
                                                  dim=0)
            text_lengths = text_lengths.repeat_interleave(
                self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        if self.stage in ['diffusion', 'vae_diffusion']:
            # diffusion reverse
          
            # style 
            motion_seq = motion*self.std + self.mean
            motion_seq[...,:3]=0.0
            motion_seq = motion_seq.unsqueeze(-1).permute(0,2,3,1)

            motion_emb = self.motionclip.encoder({'x': motion_seq,
                          'y': torch.zeros(motion_seq.shape[0], dtype=int, device='cuda:{}'.format(self.cfg["DEVICE"][0])),
                          'mask': lengths_to_mask(lengths, device='cuda:{}'.format(self.cfg["DEVICE"][0]))})["mu"]
            motion_emb = motion_emb.unsqueeze(1)
            # uncond set feature = 0
            uncond_motion_emb = torch.zeros(motion_emb.shape).to('cuda:{}'.format(self.cfg["DEVICE"][0]))
            motion_emb = torch.cat([uncond_motion_emb, motion_emb], dim=0)

            # content condition
            with torch.no_grad():
                z, dist_m = self.vae.encode(content_motions, lengths)
            uncond_tokens = torch.cat([z, z], dim = 1).permute(1,0,2)
            motion_emb_content = uncond_tokens

            # trans
            trans_cond = batch["motion"][...,:3]
            uncond_trans = torch.cat([trans_cond, trans_cond], dim = 0)

            multi_cond_emb = [motion_emb_content, motion_emb, uncond_trans]
            z = self._diffusion_reverse(multi_cond_emb, lengths,scale=self.guidance_scale)
        elif self.stage in ['vae']:
            z, dist_m = self.vae.encode(motions, lengths)

        with torch.no_grad():
            feats_rst = self.vae.decode(z, lengths)


        # end time
        end = time.time()
        self.times.append(end - start)

        # joints recover
        joints_rst = self.feats2joints(feats_rst)
        joints_ref = self.feats2joints(motions)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        motions = self.datamodule.renorm4t2m(motions)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot,
                                        text_lengths)[align_idx]

        rs_set = {
            "m_ref": motions,
            "m_rst": feats_rst,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "lat_rm": recons_emb,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
        }
        return rs_set


    def a2m_gt(self, batch):
        actions = batch["action"]
        actiontexts = batch["action_text"]
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]
        mask = batch["mask"]

        joints_ref = self.feats2joints(motions.to('cuda'), mask.to('cuda'))

        rs_set = {
            "m_action": actions,
            "m_text": actiontexts,
            "m_ref": motions,
            "m_lens": lengths,
            "joints_ref": joints_ref,
        }
        return rs_set

    def eval_gt(self, batch, renoem=True):
        motions = batch["motion"].detach().clone()
        lengths = batch["length"]

        # feats_rst = self.datamodule.renorm4t2m(feats_rst)
        if renoem:
            motions = self.datamodule.renorm4t2m(motions)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=motions.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens,
                           self.cfg.DATASET.HUMANML3D.UNIT_LEN,
                           rounding_mode="floor")

        word_embs = batch["word_embs"].detach()
        pos_ohot = batch["pos_ohot"].detach()
        text_lengths = batch["text_len"].detach()

        motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        text_emb = self.t2m_textencoder(word_embs, pos_ohot,
                                        text_lengths)[align_idx]

        # joints recover
        joints_ref = self.feats2joints(motions)

        rs_set = {
            "m_ref": motions,
            "lat_t": text_emb,
            "lat_m": motion_emb,
            "joints_ref": joints_ref,
        }
        return rs_set

    def allsplit_step(self, split: str, batch, batch_idx):
        if split in ["train", "val"]:



            if self.stage == "vae":
                rs_set = self.train_vae_forward(batch)
                rs_set["lat_t"] = rs_set["lat_m"]



            elif self.stage == "diffusion":#
                rs_set = self.train_diffusion_forward(batch)


            elif self.stage == "vae_diffusion":
                vae_rs_set = self.train_vae_forward(batch)
                diff_rs_set = self.train_diffusion_forward(batch)
                t2m_rs_set = self.test_diffusion_forward(batch,
                                                         finetune_decoder=True)
                # merge results
                rs_set = {
                    **vae_rs_set,
                    **diff_rs_set,
                    "gen_m_rst": t2m_rs_set["m_rst"],
                    "gen_joints_rst": t2m_rs_set["joints_rst"],
                    "lat_t": t2m_rs_set["lat_t"],
                }
            else:
                raise ValueError(f"Not support this stage {self.stage}!")

            loss = self.losses[split].update(rs_set)
            if loss is None:
                raise ValueError(
                    "Loss is None, this happend with torchmetrics > 0.7")

        # # Compute the metrics - currently evaluate results from text to motion
        # if split in ["val", "test"]:
        #     # use t2m evaluators
        #     rs_set = self.t2m_eval(batch)

        #     # MultiModality evaluation sperately
        #     if self.trainer.datamodule.is_mm:
        #         metrics_dicts = ['MMMetrics']
        #     else:
        #         metrics_dicts = self.metrics_dict
        #     # metric = 'TemosMetric' 'TM2TMetrics'
        #     for metric in metrics_dicts:
        #         if metric == "TemosMetric":
        #             phase = split if split != "val" else "eval"
        #             if eval(f"self.cfg.{phase.upper()}.DATASETS")[0].lower(
        #             ) not in [
        #                     "humanml3d",
        #                     "kit",
        #             ]:
        #                 raise TypeError(
        #                     "APE and AVE metrics only support humanml3d and kit datasets now"
        #                 )

        #             getattr(self, metric).update(rs_set["joints_rst"],
        #                                          rs_set["joints_ref"],
        #                                          batch["length"])
        #         elif metric == "TM2TMetrics":
        #             getattr(self, metric).update(
        #                 # lat_t, latent encoded from diffusion-based text
        #                 # lat_rm, latent encoded from reconstructed motion
        #                 # lat_m, latent encoded from gt motion
        #                 # rs_set['lat_t'], rs_set['lat_rm'], rs_set['lat_m'], batch["length"])
        #                 rs_set["lat_t"],
        #                 rs_set["lat_rm"],
        #                 rs_set["lat_m"],
        #                 batch["length"],
        #             )
        #         elif metric == "UncondMetrics":
        #             getattr(self, metric).update(
        #                 recmotion_embeddings=rs_set["lat_rm"],
        #                 gtmotion_embeddings=rs_set["lat_m"],
        #                 lengths=batch["length"],
        #             )
        #         elif metric == "MRMetrics":
        #             getattr(self, metric).update(rs_set["joints_rst"],
        #                                          rs_set["joints_ref"],
        #                                          batch["length"])
        #         elif metric == "MMMetrics":
        #             getattr(self, metric).update(rs_set["lat_rm"].unsqueeze(0),
        #                                          batch["length"])
        #         elif metric == "HUMANACTMetrics":
        #             getattr(self, metric).update(rs_set["m_action"],
        #                                          rs_set["joints_eval_rst"],
        #                                          rs_set["joints_eval_ref"],
        #                                          rs_set["m_lens"])
        #         elif metric == "UESTCMetrics":
        #             # the stgcn model expects rotations only
        #             getattr(self, metric).update(
        #                 rs_set["m_action"],
        #                 rs_set["m_rst"].view(*rs_set["m_rst"].shape[:-1], 6,
        #                                      25).permute(0, 3, 2, 1)[:, :-1],
        #                 rs_set["m_ref"].view(*rs_set["m_ref"].shape[:-1], 6,
        #                                      25).permute(0, 3, 2, 1)[:, :-1],
        #                 rs_set["m_lens"])
        #         else:
        #             raise TypeError(f"Not support this metric {metric}")

        # return forward output rather than loss during test
        if split in ["test"]:
            return rs_set["joints_rst"], batch["length"]
        # tensorboard 现在看不到loss，log一下?
        self.log('train/loss', 
             loss, 
             on_step=True,       # 在每一步 (Batch) 结束时记录
             on_epoch=True,      # 在每轮 (Epoch) 结束时计算平均值并记录
             prog_bar=True,      # 显示在进度条上 (可选)
             logger=True,        # 记录到 TensorBoard
             sync_dist=True)     # 在多 GPU 环境下同步损失 (重要)
        return loss
