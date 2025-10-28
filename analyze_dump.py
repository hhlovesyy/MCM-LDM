# ===> START: analyze_dump_reinforced.py (终极加固版) <===
import torch
import os

def analyze_tensor_reinforced(name, tensor):
    """一个辅助函数，用于打印张量的超详细分析信息"""
    print(f"--- Reinforced Analysis for tensor: '{name}' ---")
    
    if tensor is None:
        print("  Tensor is None.")
        print("-" * (len(name) + 33))
        return

    # 1. 打印基本信息
    print(f"  Shape: {tensor.shape}")
    print(f"  Data type: {tensor.dtype}")
    
    # 2. 打印数值统计信息 (与之前相同)
    tensor_float = tensor.float()
    mean_val = tensor_float.mean().item()
    std_val = tensor_float.std().item()
    min_val = tensor_float.min().item()
    max_val = tensor_float.max().item()
    print(f"  Overall Stats: Mean={mean_val:.4f}, Std={std_val:.4f}, Min={min_val:.4f}, Max={max_val:.4f}")

    # 3. [加固诊断] 检查“一半是零”的问题
    if tensor.dim() >= 2 and tensor.shape[-1] == 512:
        print("  [512-dim Reinforced Check]:")
        
        first_half = tensor_float[..., :256]
        second_half = tensor_float[..., 256:]
        
        # a. 比较前半部分和后半部分的绝对值均值
        first_half_abs_mean = first_half.abs().mean().item()
        second_half_abs_mean = second_half.abs().mean().item()
        print(f"    Mean of Absolute Values (First Half):  {first_half_abs_mean:.7f}")
        print(f"    Mean of Absolute Values (Second Half): {second_half_abs_mean:.7f}")

        # 设置一个阈值来判断是否“接近于零”
        zero_threshold = 1e-7
        if second_half_abs_mean < zero_threshold and first_half_abs_mean > zero_threshold * 100:
            print("    CRITICAL VERDICT: The second half IS effectively zero, while the first half is not.")
        elif first_half_abs_mean < zero_threshold and second_half_abs_mean < zero_threshold:
            print("    WARNING VERDICT: The entire vector is effectively zero.")
        else:
            print("    OK VERDICT: Both halves of the vector contain non-trivial values.")

        # b. [新增] 分析有效维度
        # 计算每一维的标准差
        dim_std = tensor_float.std(dim=list(range(tensor.dim() - 1))) # 在所有非特征维度上计算标准差
        
        # 设置一个阈值来判断维度是否“有效”（即有变化）
        active_dim_threshold = 0.01 
        num_active_dims = (dim_std > active_dim_threshold).sum().item()
        
        print(f"  [Effective Dimensionality Analysis]:")
        print(f"    Number of dimensions with Std > {active_dim_threshold}: {num_active_dims} / 512")
        
        if num_active_dims < 400 and num_active_dims > 100: #  heuristic check
             print(f"    INFO: The number of active dimensions ({num_active_dims}) is significantly less than 512, suggesting a potential information bottleneck.")

    # 4. 动作序列检查 (与之前相同)
    if name.startswith("motion_seq"):
        num_zeros = (tensor == 0).sum().item()
        num_ones = (tensor == 1).sum().item()
        total_elements = tensor.numel()
        print(f"  [Motion Seq Check]:")
        print(f"    Percentage of zeros: {100 * num_zeros / total_elements:.2f}%")
        print(f"    Percentage of ones:  {100 * num_ones / total_elements:.2f}%")
        
    print("-" * (len(name) + 33))


def main():
    dump_path = os.path.join("debug_dumps", "tensors_first_batch.pt")

    if not os.path.exists(dump_path):
        print(f"Error: Dump file not found at '{dump_path}'")
        return

    print(f"Loading tensors from '{dump_path}' for REINFORCED analysis...\n")
    data = torch.load(dump_path)

    # 按照之前的顺序分析
    analyze_tensor_reinforced("gt_motion_emb_for_text (from Teacher)", data.get("gt_motion_emb_for_text"))
    analyze_tensor_reinforced("motion_emb (from Student)", data.get("motion_emb"))
    analyze_tensor_reinforced("text_emb_for_align (for loss)", data.get("text_emb_for_align"))
    analyze_tensor_reinforced("text_features_for_denoiser (for denoiser)", data.get("text_features_for_denoiser"))
    analyze_tensor_reinforced("motion_seq_for_text_unnormalized", data.get("motion_seq_for_text_unnormalized"))
    analyze_tensor_reinforced("motion_seq_unnormalized", data.get("motion_seq_unnormalized"))

    print("\nReinforced analysis complete.")


if __name__ == "__main__":
    main()
# ===> END: analyze_dump_reinforced.py <===