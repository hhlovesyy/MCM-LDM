# dce_lightning_module.py

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf # 用于类型提示和配置处理
from mld.config import instantiate_from_config
import os
import numpy as np

# 从我们之前定义的文件中导入模型
from style_classifier_model import StyleClassifierTransformer # 我们的 Cstyle
from dce_model import DisentangledContentExtractor, GradientReversalLayer # 我们的 DCE 和 GRL

class DCETrainingModule(pl.LightningModule):
    def __init__(self,
                 cfg_dce: DictConfig,
                 cfg_cstyle: DictConfig,
                 cstyle_checkpoint_path: str,
                 cfg_vae: DictConfig,
                 vae_checkpoint_path: str,
                 # 新增：直接传入均值和标准差的路径
                 mean_path_for_vae_input: str, # Path to HumanML3D Mean.npy
                 std_path_for_vae_input: str,  # Path to HumanML3D Std.npy
                 learning_rate: float = 1e-4,
                 lambda_style: float = 0.5,
                 lambda_content: float = 0.5,
                 lambda_grl: float = 1.0
                ):
        super().__init__()
        # 忽略大的配置对象，但保存路径等关键参数
        hparams_to_save = {
            'learning_rate': learning_rate,
            'lambda_style': lambda_style,
            'lambda_content': lambda_content,
            'lambda_grl': lambda_grl,
            'cstyle_checkpoint_path': cstyle_checkpoint_path,
            'vae_checkpoint_path': vae_checkpoint_path,
            'mean_path_for_vae_input': mean_path_for_vae_input, # 保存路径
            'std_path_for_vae_input': std_path_for_vae_input   # 保存路径
        }
        # 将 cfg_dce, cfg_cstyle, cfg_vae 的关键参数也提取出来保存到hparams，如果它们包含需要记录的超参
        # 例如: hparams_to_save.update(OmegaConf.to_container(cfg_dce.params, resolve=True))
        self.save_hyperparameters(hparams_to_save)

        # 1. 实例化 VAE (从MCM-LDM加载，并冻结权重)
        # 我们需要能够从 cfg_vae 和 vae_checkpoint_path 加载它
        # 假设有一个函数可以做到这点，或者直接在这里实例化并加载
        # 这里用一个占位符，实际加载逻辑可能更复杂，取决于MLD项目中VAE的定义方式
        # 1. 实例化 VAE (使用 cfg_vae)
        print("--- VAE Initialization ---")
        print("Instantiating VAE from config...")
        try:
            # from mld.config import instantiate_from_config # 假设这个函数可用
            # 我们需要确保 cfg_vae 是正确的 DictConfig 对象
            if not isinstance(cfg_vae, DictConfig):
                cfg_vae = OmegaConf.create(cfg_vae) # 如果传入的是普通字典，转为DictConfig

            print("Using VAE configuration:")
            print(OmegaConf.to_yaml(cfg_vae))
            
            self.vae = instantiate_from_config(cfg_vae) # 核心实例化步骤
            print("VAE instantiated successfully.")

            # 打印实例化后的 self.vae 的键名，用于对比
            current_vae_model_keys = list(self.vae.state_dict().keys())
            print(f"Total keys in instantiated self.vae: {len(current_vae_model_keys)}")
            print("Sample keys from instantiated self.vae (before loading weights):")
            for i, key in enumerate(current_vae_model_keys[:10]): # 打印前10个
                print(f"  {i+1}. {key}")

        except ImportError as e:
            print(f"ERROR: Could not import or instantiate VAE. Target: {cfg_vae.get('target', 'N/A')}. Error: {e}")
            raise
        except Exception as e:
            print(f"ERROR: An unexpected error occurred during VAE instantiation: {e}")
            import traceback
            traceback.print_exc()
            raise

        # 2. 加载预训练的 VAE 权重 (从 vae_checkpoint_path)
        if vae_checkpoint_path and os.path.exists(vae_checkpoint_path):
            print(f"\nAttempting to load VAE weights from: {vae_checkpoint_path}")
            try:
                checkpoint_data = torch.load(vae_checkpoint_path, map_location='cpu')
                
                if 'state_dict' not in checkpoint_data:
                    raise KeyError(f"Checkpoint file at {vae_checkpoint_path} does not have a 'state_dict' key. It might not be a PyTorch Lightning checkpoint or has an unexpected format.")
                
                full_model_state_dict = checkpoint_data['state_dict']
                # print(f"  Successfully loaded 'state_dict' from checkpoint. It has {len(full_model_state_dict)} keys.")
                # print("  Sample keys from checkpoint's full_model_state_dict (before filtering):")
                # for i, key in enumerate(list(full_model_state_dict.keys())[:10]):
                #      print(f"    {i+1}. {key}")


                vae_weights_to_load = OrderedDict()
                prefix_to_strip = "vae." # <--- 你确认了这个前缀是正确的！
                
                print(f"  Attempting to strip prefix: '{prefix_to_strip}' to extract VAE weights.")
                
                found_keys_with_prefix = False
                for k, v in full_model_state_dict.items():
                    if k.startswith(prefix_to_strip):
                        found_keys_with_prefix = True
                        new_key = k.replace(prefix_to_strip, "", 1)
                        vae_weights_to_load[new_key] = v
                
                if not found_keys_with_prefix:
                    print(f"  WARNING: No keys found in checkpoint state_dict starting with the prefix '{prefix_to_strip}'.")
                    print("  This means VAE weights cannot be loaded with this prefix. VAE will remain randomly initialized or partially initialized.")
                
                if vae_weights_to_load:
                    # print(f"  Extracted {len(vae_weights_to_load)} VAE-specific keys after stripping prefix '{prefix_to_strip}'.")
                    # print("  Sample keys extracted for VAE (after stripping prefix):")
                    # for i, key in enumerate(list(vae_weights_to_load.keys())[:10]):
                    #     print(f"    {i+1}. {key}")

                    print("\n  Comparing extracted keys with instantiated VAE model keys...")
                    # 尝试使用 strict=True 进行加载，以便立即知道是否有不匹配
                    print("  Attempting to load VAE weights with strict=True...")
                    try:
                        missing_keys, unexpected_keys = self.vae.load_state_dict(vae_weights_to_load, strict=True)
                        # 如果 strict=True 成功，missing_keys 和 unexpected_keys 应该是空的列表
                        print("  VAE weights loaded successfully with strict=True!")
                        if missing_keys: print(f"    (Strict mode) Missing keys (should be empty): {missing_keys}") # 理论上不应发生
                        if unexpected_keys: print(f"    (Strict mode) Unexpected keys (should be empty): {unexpected_keys}") # 理论上不应发生
                    except RuntimeError as e_strict:
                        print(f"  ERROR: load_state_dict with strict=True failed: {e_strict}")
                        print("  This indicates a mismatch between the keys in the checkpoint (after stripping prefix)")
                        print("  and the keys expected by your instantiated VAE model (from cfg_vae).")
                        print("  Common reasons: cfg_vae defines a different VAE structure than what's in the checkpoint's 'vae.*' part,")
                        print("  or the prefix_to_strip is still not perfectly aligning the keys.")
                        print("  Attempting to load with strict=False to see details...")
                        
                        missing_keys, unexpected_keys = self.vae.load_state_dict(vae_weights_to_load, strict=False)
                        if missing_keys:
                            print(f"    (Strict=False) Missing keys: {missing_keys}")
                        else:
                            print("    (Strict=False) No missing keys.")
                        if unexpected_keys:
                            print(f"    (Strict=False) Unexpected keys: {unexpected_keys}")
                        else:
                            print("    (Strict=False) No unexpected keys.")
                        
                        if not missing_keys:
                            print("    INFO: With strict=False, all expected VAE keys were found. The strict error was likely due to unexpected_keys.")
                            print("    This might be acceptable if the unexpected_keys are not critical.")
                        else:
                            print("    CRITICAL: Even with strict=False, VAE has missing keys. Weights not loaded correctly.")
                else: # vae_weights_to_load is empty
                    print(f"  CRITICAL: No VAE weights extracted after attempting to strip prefix '{prefix_to_strip}'.")
                    print("  VAE will use random weights. This is highly problematic.")

            except FileNotFoundError:
                print(f"ERROR: VAE checkpoint file not found at {vae_checkpoint_path}.")
            except KeyError as e:
                if "'state_dict'" in str(e): # 更通用的KeyError检查
                    print(f"ERROR: Checkpoint file at {vae_checkpoint_path} does not seem to have a 'state_dict' key or is malformed.")
                else:
                    print(f"ERROR: A KeyError occurred: {e}")
            except Exception as e:
                print(f"ERROR: An unexpected exception occurred while loading VAE weights: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Warning: VAE checkpoint path '{vae_checkpoint_path}' not found or not provided. VAE will use random weights.")

        # 冻结VAE参数
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        print("VAE model frozen for training.")
        print("--- VAE Initialization and Weight Loading Complete ---")


        # 冻结VAE参数
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        print("VAE model setup complete (weights loaded if path valid, and frozen).")
                
        # 2. 实例化 Style Classifier (Cstyle) (从我们训练好的加载，并冻结权重)
        # StyleClassifierTransformer.load_from_checkpoint 会加载模型和hparams
        try:
            self.cstyle = StyleClassifierTransformer.load_from_checkpoint(
                cstyle_checkpoint_path,
                map_location='cpu' # 加载到CPU，后续会根据trainer移动到GPU
            )
        except Exception as e:
            raise RuntimeError(f"Error loading Cstyle from checkpoint {cstyle_checkpoint_path}: {e}")
            
        self.cstyle.eval()
        for param in self.cstyle.parameters():
            param.requires_grad = False
        print(f"Cstyle model loaded from {cstyle_checkpoint_path} and frozen.")

        # 3. 实例化 DCE 模型
        # cfg_dce.params 应该包含 DisentangledContentExtractor 的所有初始化参数
        dce_params = OmegaConf.to_container(cfg_dce.params, resolve=True)
        self.dce = DisentangledContentExtractor(**dce_params)
        print("DCE model instantiated.")

        # 4. 实例化 Gradient Reversal Layer (GRL)
        self.grl = GradientReversalLayer(lambda_val=lambda_grl)
        print(f"GRL instantiated with lambda_val={lambda_grl}.")

        # 5. 损失函数
        self.style_loss_fn = nn.CrossEntropyLoss() # Cstyle的输出是logits
        self.content_loss_fn = nn.MSELoss()       # 内容保留损失用MSE

        # 6. 超参数
        self.lr = learning_rate
        self.lambda_style = lambda_style
        self.lambda_content = lambda_content
        
        # 6. 加载均值和标准差，并设为不需要梯度的 buffer
        #    Buffer 会自动随模型移动到设备，且不参与优化
        try:
            mean_np = np.load(mean_path_for_vae_input)
            std_np = np.load(std_path_for_vae_input)
            # 将它们注册为buffer
            self.register_buffer("mean_for_norm", torch.tensor(mean_np, dtype=torch.float32), persistent=False)
            self.register_buffer("std_for_norm", torch.tensor(std_np, dtype=torch.float32), persistent=False)
            print(f"Mean and Std for VAE input loaded from {mean_path_for_vae_input} and {std_path_for_vae_input}.")
        except Exception as e:
            raise FileNotFoundError(f"Could not load Mean/Std from paths: {mean_path_for_vae_input}, {std_path_for_vae_input}. Error: {e}")


    def set_mean_std(self, mean, std):
        # 确保它们是tensor并移到正确的设备
        if not isinstance(mean, torch.Tensor): mean = torch.tensor(mean, dtype=torch.float32)
        if not isinstance(std, torch.Tensor): std = torch.tensor(std, dtype=torch.float32)
        self.mean = nn.Parameter(mean, requires_grad=False) # 使用 nn.Parameter 确保移到正确设备
        self.std = nn.Parameter(std, requires_grad=False)   # 使用 nn.Parameter 确保移到正确设备


    def forward(self, original_motion, lengths):
        """
        Defines the forward pass for DCE training.
        Args:
            original_motion (Tensor): Batch of motions (Batch, Seq_len, Num_feats=263)
            lengths (list): List of actual lengths for each motion in the batch.
        Returns:
            fc (Tensor): Disentangled content features from DCE. (Batch, Num_VAE_Tokens, Dim_VAE_Latent)
            z_raw (Tensor): Latent codes from VAE. (Batch, Num_VAE_Tokens, Dim_VAE_Latent)
            z_content_target (Tensor): Target for content loss. (Batch, Num_VAE_Tokens, Dim_VAE_Latent)
        """
        mean_val = self.mean_for_norm
        std_val = self.std_for_norm

        if mean_val is None or std_val is None: # 理论上 __init__ 中会加载，不应为None
            raise RuntimeError("Mean and Std for normalization are not loaded properly.")

        normalized_motion = (original_motion - mean_val) / (std_val + 1e-8)

        # 1. Encode original motion with VAE to get z_raw
        # VAE的 .encode() 通常返回 (z, distribution_params)
        # VAE的输出是 (Num_Tokens_VAE, Batch, Dim_VAE_Latent)
        # 1. Encode original motion with VAE to get z_raw
        with torch.no_grad():
            z_raw_vae_output, _ = self.vae.encode(normalized_motion, lengths)
        z_raw_for_dce = z_raw_vae_output.permute(1, 0, 2) # (Batch, 7, 256)

        # 2. Pass z_raw through DCE to get fc
        fc = self.dce(z_raw_for_dce, src_key_padding_mask=None) # (Batch, 7, 256)
        # fc: (Batch, 7, 256)

        # 3. Prepare target for content loss: z_raw after heuristic style removal
        #    (This is one way to define content target)
        # with torch.no_grad(): # All ops here are for target generation
        #     motion_for_content_target = original_motion.clone()
        #     # Apply heuristic style removal (e.g., zero out root translation/orientation)
        #     # ** 这里的去风格方式需要和MLD原始方法中获取 cond_emb 的方式对齐 **
        #     # ** 如果MLD是用 batch["motion"] (原始未归一化) 置零，然后归一化，再进VAE **
        #     # ** 那么这里也应该类似。但更简单的是直接对 normalized_motion 操作 **
        #     normalized_motion_style_removed_heuristic = normalized_motion.clone()
        #     normalized_motion_style_removed_heuristic[..., :3] = 0.0 # 假设前3维是root相关
            
        #     z_content_target_vae_output, _ = self.vae.encode(normalized_motion_style_removed_heuristic, lengths)
        #     # z_content_target_vae_output: (7, Batch, 256)
        
        # # Permute for consistency: (Batch, 7, 256)
        # z_content_target = z_content_target_vae_output.permute(1, 0, 2).detach()

        # return fc, z_raw_for_dce, z_content_target
        return fc, z_raw_for_dce


    def _common_step(self, batch, batch_idx, stage: str):
        original_motion = batch["motion"]    # (Batch, Seq_len, 263)
        style_labels = batch["style_labels"] # (Batch,) - 真实风格标签
        lengths = batch["length"]            # List of lengths

        current_batch_size = original_motion.size(0)

        # 修改这里：forward 现在只返回 fc 和 z_raw_for_dce
        # fc, _, z_content_target = self.forward(original_motion, lengths) # 旧的，期望3个
        fc, z_raw_for_dce = self.forward(original_motion, lengths) # 新的，接收2个
        # fc: (Batch, 7, 256)
        # z_raw_for_dce: (Batch, 7, 256)

        # --- Style Invariability Loss ---
        fc_grl = self.grl(fc)
        cstyle_padding_mask = torch.zeros(current_batch_size, fc.size(1), dtype=torch.bool, device=fc.device)
        style_logits_from_fc = self.cstyle(fc_grl, src_key_padding_mask=cstyle_padding_mask)
        loss_style = self.style_loss_fn(style_logits_from_fc, style_labels)

        # --- Content Preservation Loss ---
        # 修改这里：loss_content 的目标是 z_raw_for_dce.detach()
        # loss_content = self.content_loss_fn(fc, z_content_target) # 旧的，使用 z_content_target
        loss_content = self.content_loss_fn(fc, z_raw_for_dce.detach()) # 新的，使用 z_raw_for_dce 并 detach

        # --- Total Loss ---
        total_loss = self.lambda_style * loss_style + self.lambda_content * loss_content

        self.log(f'{stage}_loss_style', loss_style, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=current_batch_size)
        self.log(f'{stage}_loss_content', loss_content, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=current_batch_size)
        self.log(f'{stage}_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=current_batch_size)
        
        if stage != 'train':
            with torch.no_grad():
                style_logits_fc_no_grl = self.cstyle(fc, src_key_padding_mask=cstyle_padding_mask)
                preds_fc_no_grl = torch.argmax(style_logits_fc_no_grl, dim=1)
                acc_fc_no_grl = (preds_fc_no_grl == style_labels).float().mean()
                self.log(f'{stage}_cstyle_acc_on_fc', acc_fc_no_grl, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=current_batch_size)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        # Optimize ONLY the parameters of the DCE model
        optimizer = optim.Adam(self.dce.parameters(), lr=self.lr, weight_decay=1e-5)
        return optimizer

    # Optional: Hook to update GRL lambda during training
    def on_train_epoch_start(self):
        # Example: Anneal lambda for GRL (common practice)
        p = float(self.current_epoch) / float(self.trainer.max_epochs)
        # new_lambda_grl_factor is an annealing factor, typically from 0 to 1 as p goes from 0 to 1
        # The original formula 2. / (1. + np.exp(-10. * p)) - 1 goes from -1 to 1 if p goes from 0 to large
        # A more common annealing schedule for a factor that multiplies a base lambda:
        # For example, linear increase:
        # new_lambda_grl_factor = min(1.0, p * scaling_factor) # e.g. scaling_factor = 2, caps at 1
        # Or using the DANN paper's schedule (often cited):
        gamma = 10.0 # A parameter for DANN schedule
        new_lambda_grl_factor = (2. / (1. + np.exp(-gamma * p))) - 1 # This factor goes from 0 to 1 as p goes from 0 to 1 (if gamma*p is large enough for exp(-gamma*p) to be small)
                                                                # Let's re-check DANN: lambda_p = 2 / (1 + exp(-gamma * p)) - 1
                                                                # No, DANN uses: lambda_p = (2 / (1 + exp(-gamma * p))) - 1
                                                                # This factor actually goes from 0 to 1.
                                                                # So if lambda_p is this factor, then final_lambda = lambda_p * base_lambda

        # Ensure the factor is non-negative if it's a multiplier
        # The original DANN factor (2. / (1. + np.exp(-gamma * p))) - 1 indeed goes from 0 to ~1.
        # So, new_lambda_grl_factor is suitable as a multiplier.

        # 使用 self.hparams.lambda_grl 作为基础值 (这是初始化时传入的 lambda_grl)
        base_lambda_for_grl = self.hparams.lambda_grl
        
        final_grl_lambda = new_lambda_grl_factor * base_lambda_for_grl
        self.grl.update_lambda(final_grl_lambda)
        self.log('grl_lambda_annealed', self.grl.lambda_val, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)