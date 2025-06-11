# for 813adain_loss
import torch
import torch.nn as nn
from torch import  nn # This is redundant, nn is already imported
import torch.nn.functional as F
from mld.models.architectures.tools.embeddings import (TimestepEmbedding,
                                                       Timesteps)
from mld.models.operator import PositionalEncoding
from mld.models.operator.cross_attention import (SkipTransformerEncoder_concat,
                                                 TransformerDecoder,
                                                 TransformerDecoderLayer,
                                                 TransformerEncoder,
                                                 TransformerEncoderLayer_concat,
                                                 TransformerEncoderLayer)
from mld.models.operator.position_encoding import build_position_encoding
from mld.utils.temos_utils import lengths_to_mask
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp



# trans encoder
class TransEncoder(nn.Module): # This class remains unchanged

    def __init__(self, d_model=256, num_heads=4, position_embedding: str = "learned", **block_kwargs):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        seqTransEncoderLayer = nn.TransformerEncoderLayer(self.d_model, self.num_heads)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=2)
        self.global_motion_token = nn.Parameter(
                torch.randn(1, self.d_model))
        self.pe = build_position_encoding(
                self.d_model, position_embedding=position_embedding)

        self.emb_proj_st = nn.Sequential(
            nn.ReLU(), nn.Linear(3, self.d_model))


    def forward(self, x, lengths):
        if lengths is None:
            # 'features' is not defined in this scope, assuming it was a typo for 'x' or handled elsewhere
            # For safety, if lengths is None, we might not be able to create mask correctly.
            # This part of TransEncoder logic is specific to how it's used for trajectory.
            # We are not changing TransEncoder itself.
            pass # Or raise error if lengths are critical and None

        device = x.device
        x = x.float()
        bs, nframes, nfeats = x.shape
        mask = lengths_to_mask(lengths, device)
        x = self.emb_proj_st(x)
        x = x.permute(1, 0, 2)
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1))
        dist_masks = torch.ones((bs, dist.shape[0]), dtype=bool, device=x.device)
        aug_mask = torch.cat((dist_masks, mask), 1)
        xseq = torch.cat((dist, x), 0)
        xseq = self.pe(xseq)
        dist = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)[:dist.shape[0]]
        return dist


# adaln-zero in dit
def modulate(x, shift, scale): # This function remains unchanged
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module): # This class remains unchanged
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
        self.adaLN_modulation_trans = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )

    def forward(self, x, c, t): # c调制MSA, t调制MLP
        shift_msa, scale_msa, gate_msa = self.adaLN_modulation(c).chunk(3, dim=1)
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation_trans(t).chunk(3, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class MldDenoiser(nn.Module):

    def __init__(self,
                 ablation,
                 nfeats: int = 263,
                 condition: str = "text",
                 latent_dim: list = [1, 256], # This is actually [num_tokens, feature_dim] for VAE output
                 ff_size: int = 1024,
                 num_layers: int = 6,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 flip_sin_to_cos: bool = True,
                 return_intermediate_dec: bool = False,
                 position_embedding: str = "learned",
                 arch: str = "trans_enc",
                 freq_shift: int = 0,
                 guidance_scale: float = 7.5,
                 guidance_uncondp: float = 0.1,
                 text_encoded_dim: int = 256, # This seems to be for time embedding base dim
                 motion_encoded_dim: int = 512, # This is style feature dim from MotionCLIP
                 nclasses: int = 10,
                 # --- START NEW PARAMETERS ---
                 diffusion_T: int = 1000, # Total diffusion timesteps
                 W_max_scene: float = 1.0,  # Max weight for scene influence
                 p_scene_weight: float = 1.0, # Power for scene weight curve
                 scene_feature_input_dim: int = 512, # Dimension of f_scene_indep from mld.py
                 # --- END NEW PARAMETERS ---
                 **kwargs) -> None:

        super().__init__()

        # self.latent_dim is the feature dimension used internally by DiT blocks, e.g., 256
        self.latent_dim = latent_dim[-1] # This correctly extracts 256
        # text_encoded_dim = 256 # This was a local variable, not used. self.text_encoded_dim below.
        self.text_encoded_dim = text_encoded_dim # Storing the param, used for time_proj, 256
        self.condition = condition # 'text_scene'
        self.abl_plus = False
        self.arch = arch  # trans_enc
        self.motion_encoded_dim = motion_encoded_dim # 512

        # --- START: Store new parameters ---
        self.diffusion_T = diffusion_T # 1000
        self.W_max_scene = W_max_scene # 1.0
        self.p_scene_weight = p_scene_weight # 1.0
        # --- END: Store new parameters ---

        # emb proj
        self.time_proj = Timesteps(self.text_encoded_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(self.text_encoded_dim, self.latent_dim)

        self.emb_proj = nn.Sequential( # This seems unused in the current forward path
            nn.ReLU(), nn.Linear(self.text_encoded_dim, self.latent_dim))
        # emb_proj_st projects style_emb (motion_encoded_dim=512) to self.latent_dim (256)
        self.emb_proj_st = nn.Sequential(
            nn.ReLU(), nn.Linear(self.motion_encoded_dim, self.latent_dim))

        # --- START: Define MLP_scene_den ---
        # Input: scene_feature_input_dim (e.g., 512 from f_scene_indep)
        # Output: self.latent_dim (e.g., 256)
        self.mlp_scene_den = nn.Sequential(
            nn.Linear(scene_feature_input_dim, self.latent_dim * 2), # Intermediate projection
            nn.SiLU(), # Or nn.GELU()
            nn.Linear(self.latent_dim * 2, self.latent_dim)
        )
        # --- END: Define MLP_scene_den ---

        self.query_pos = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
        self.mem_pos = build_position_encoding( # This seems unused in current forward
                self.latent_dim, position_embedding=position_embedding)

        self.blocks = nn.ModuleList([
            DiTBlock( hidden_size=self.latent_dim, num_heads=num_heads, mlp_ratio=4.0) for _ in range(num_layers)
        ])

        self.IN = nn.InstanceNorm1d(self.latent_dim, affine=True) # Input to IN should be latent_dim for content_emb after VAE
                                                                    # Original: text_encoded_dim. Corrected to self.latent_dim.
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=num_heads) # Use num_heads param
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=1)
        self.pe_content = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
        
        # The VAE output content_emb is [bs, 7, 256] (num_tokens_vae=7, latent_dim=256)
        # So linear input should be num_tokens_vae * latent_dim
        num_tokens_vae = latent_dim[0] if isinstance(latent_dim, list) and len(latent_dim) > 1 else 7 # Default to 7 if not specified well
        self.linear = nn.Linear(num_tokens_vae * self.latent_dim, 6 * self.latent_dim) # Reshape to 6 tokens

        self.trans_Encoder = TransEncoder(d_model=self.latent_dim, num_heads=num_heads) # Pass num_heads

    def forward(self,
                sample,
                timestep,
                encoder_hidden_states,
                lengths=None,
                **kwargs):

        sample = sample.permute(1, 0, 2) # [seq_len_sample, bs, latent_dim]

        timesteps = timestep.expand(sample.shape[1]).clone() # torch.Size([32])
        time_emb = self.time_proj(timesteps) # [bs, text_encoded_dim]
        time_emb = time_emb.to(dtype=sample.dtype)
        time_emb = self.time_embedding(time_emb).unsqueeze(0) # [1, bs, latent_dim] torch.Size([1, 32, 256])

        # --- MODIFIED: Parse 4 conditions ---
        content_emb = encoder_hidden_states[0].permute(1, 0, 2)  # f_c: [7, bs, 256]
        style_emb = encoder_hidden_states[1].permute(1, 0, 2)    # f_s_adapted: [1, bs, 512]
        trans_cond = encoder_hidden_states[2]                    # f_t: [bs, L_traj, 3]
        # NEW: Parse independent scene feature
        scene_feat_indep = encoder_hidden_states[3].permute(1, 0, 2) # f_scene_indep: [1, bs, D_scene_in=512]
        # --- END MODIFIED ---
        
        # content processing
        content_emb_latent = content_emb # [7, bs, 256]
        # InstanceNorm expects (N, C, L) so (bs, latent_dim, 7)
        content_emb_latent = self.IN(content_emb_latent.permute(1,2,0)).permute(2,0,1) # Back to [7, bs, 256]
        content_emb_latent = content_emb_latent + time_emb # Add time_emb (broadcasts over token dim)
        content_emb_latent = self.pe_content(content_emb_latent)
        content_emb_latent = self.seqTransEncoder(content_emb_latent) # Output [7, bs, 256]
        # Reshape from 7 tokens to 6 tokens
        bs_content = content_emb_latent.shape[1]
        content_emb_latent = content_emb_latent.permute(1,0,2).reshape(bs_content, -1) # [bs, 7*256]
        content_emb_latent = self.linear(content_emb_latent).reshape(bs_content, 6 , self.latent_dim) # [bs, 6, 256]
        content_emb_latent = content_emb_latent.permute(1,0,2) # [6, bs, 256]
        
        xseq = torch.cat((content_emb_latent, sample), axis=0) # [6+L_sample, bs, 256] torch.Size([13, 32, 256])

        # style encoder (for MSA modulation in DiTBlock)
        # style_emb is [1, bs, 512]
        style_emb_latent = self.emb_proj_st(style_emb) # Projects 512 -> 256. Output: [1, bs, 256]
        style_emb_latent = time_emb + style_emb_latent # Add time_emb. Output: [1, bs, 256]
        style_emb_latent = style_emb_latent.squeeze(0) # Squeeze token dim. Output: [bs, 256]

        # trajectory encoder (original part of FFN modulation in DiTBlock)
        # trans_cond is [bs, L_traj, 3]
        trans_emb = self.trans_Encoder(trans_cond, lengths) # Output: [1, bs, 256] (global token from TransEncoder)
        trans_emb = trans_emb + time_emb # Add time_emb. Output: [1, bs, 256]
        trans_emb = trans_emb.squeeze(0) # Squeeze token dim. Output: [bs, 256]
        
        # --- START: NEW - Process independent scene feature and combine with trajectory ---
        # scene_feat_indep is [1, bs, D_scene_in=512]
        # Squeeze token dim for mlp_scene_den, then unsqueeze later if needed by DiTBlock's modulator
        processed_scene_feat = self.mlp_scene_den(scene_feat_indep.squeeze(0)) # Output: [bs, latent_dim=256]
        
        # Calculate time-dependent weight w_scene(t)
        # timestep is [bs], self.diffusion_T is scalar
        # t_idx (0 to T-1, reverse) = timestep.float()
        current_t_values = timestep.float() # Shape: [bs] or scalar (0D). weight 缩放
        
        weight_factor = (self.diffusion_T - current_t_values) / self.diffusion_T
        w_scene_t = self.W_max_scene * torch.pow(weight_factor.clamp(min=0.0), self.p_scene_weight) # Shape: [bs] or scalar (0D)
        
        # --- START MODIFIED BLOCK for w_scene_t shaping ---
        if w_scene_t.ndim == 0: # If it's a scalar
            # Expand to [bs, 1] for multiplication with processed_scene_feat [bs, latent_dim]
            _bs_for_weight = processed_scene_feat.shape[0]
            w_scene_t_shaped = w_scene_t.expand(_bs_for_weight, 1)
        elif w_scene_t.ndim == 1: # If it's already [bs]
            w_scene_t_shaped = w_scene_t.unsqueeze(1) # Make it [bs, 1]
        else: # Should not happen, error or handle
            raise ValueError(f"Unexpected shape for w_scene_t: {w_scene_t.shape}")
            
        weighted_scene_feat = processed_scene_feat * w_scene_t_shaped 
        # --- END MODIFIED BLOCK ---
        
        # Combine with trajectory embedding for FFN modulation
        # trans_emb is [bs, 256], weighted_scene_feat is [bs, 256]
        combined_ffn_mod_source = trans_emb + weighted_scene_feat # [bs, 256]
        # --- END: NEW ---
        
        # to dit blocks
        xseq = self.query_pos(xseq).permute(1,0,2) # [bs, 6+L_sample, 256] # content+z_noisy（z_noisy=batch["motion"]）
        for block in self.blocks:
            # MODIFIED: Pass combined_ffn_mod_source as 't' for DiTBlock
            xseq = block(xseq, c=style_emb_latent, t=combined_ffn_mod_source) 
            # --- END MODIFIED ---
            
        sample = xseq[:,content_emb_latent.shape[0]:,:] # Extract the part corresponding to original sample
       
        return (sample.permute(1,0,2), ) # Return sample in original [L_sample, bs, 256] like shape if needed by caller, or [bs, L_sample, 256]


class EmbedAction(nn.Module): # This class remains unchanged
    # ... (original EmbedAction code) ...
    def __init__(self,
                 num_actions,
                 latent_dim, # Here latent_dim is likely just the feature dim, not a list
                 guidance_scale=7.5,
                 guidance_uncodp=0.1,
                 force_mask=False):
        super().__init__()
        self.nclasses = num_actions
        self.guidance_scale = guidance_scale
        self.action_embedding = nn.Parameter(
            torch.randn(num_actions, latent_dim)) # Use the passed latent_dim directly

        self.guidance_uncodp = guidance_uncodp
        self.force_mask = force_mask
        self._reset_parameters()

    def forward(self, input):
        idx = input[:, 0].to(torch.long) 
        output = self.action_embedding[idx]
        # CFG logic specific to EmbedAction, seems fine.
        if not self.training and self.guidance_scale > 1.0: # Check if it's actually self.cfg.TEST.GUIDANCE_SCALE etc.
            if output.shape[0] % 2 == 0: # Ensure it's even for chunking
                uncond, output_cond = output.chunk(2) # output_cond to avoid reassigning 'output'
                uncond_out = self.mask_cond(uncond, force=True)
                out_cond = self.mask_cond(output_cond) # Use output_cond here
                output = torch.cat((uncond_out, out_cond))
            # else: # Batch size is not even, CFG logic might be problematic or intended for non-CFG path
                 # output = self.mask_cond(output) # Default masking if not CFG-structured
        else: # Training or guidance_scale <= 1.0
            output = self.mask_cond(output)

        return output.unsqueeze(0) # Returns [1, bs, latent_dim]

    def mask_cond(self, output, force=False):
        bs, d = output.shape
        if self.force_mask or force:
            return torch.zeros_like(output)
        elif self.training and self.guidance_uncodp > 0.:
            mask = torch.bernoulli(
                torch.ones(bs, device=output.device) *
                self.guidance_uncodp).view(bs, 1)
            return output * (1. - mask)
        else:
            return output

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)