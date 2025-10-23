# diag_vae.py (Version 2 - Robust Key Finding)

import torch

# --- 配置区 ---
# 确保这个路径是正确的
vae_ckpt_path = "checkpoints/vae_checkpoint/vae7.ckpt" 
# ----------------

try:
    state_dict_full = torch.load(vae_ckpt_path, map_location='cpu')

    print("--- Checkpoint Top-Level Keys ---")
    print(list(state_dict_full.keys()))
    print("---------------------------------")

    # PyTorch Lightning 通常将模型权重保存在 'state_dict' 键下
    if 'state_dict' in state_dict_full:
        model_state_dict = state_dict_full['state_dict']
        
        # 权重键名通常带有模块前缀，如 'vae.skel_embedding.weight'
        # 我们来查找包含 'skel_embedding.weight' 的键
        target_key = None
        for key in model_state_dict.keys():
            if 'skel_embedding.weight' in key:
                target_key = key
                break
        
        if target_key:
            skel_embedding_weight = model_state_dict[target_key]
            # 权重的形状是 (output_dim, input_dim)
            vae_input_dim = skel_embedding_weight.shape[1]
            
            print("\n--- Diagnostic Result ---")
            print(f"Found weight key: '{target_key}'")
            print(f"Weight shape: {skel_embedding_weight.shape}")
            print(f"==> The pre-trained VAE expects an input dimension of: {vae_input_dim}")
            print("-------------------------")
        else:
            print("\n[ERROR] Could not find 'skel_embedding.weight' in the checkpoint's state_dict.")
            print("Available keys in state_dict:")
            for key in model_state_dict.keys():
                print(f"- {key}")

    else:
        print(f"\n[ERROR] The checkpoint file does not contain a 'state_dict' key.")
        print("This might be a simple PyTorch model save, not a PyTorch Lightning checkpoint.")
        print("Try inspecting the keys printed above to find the model weights.")

except FileNotFoundError:
    print(f"[ERROR] Checkpoint file not found at: {vae_ckpt_path}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")