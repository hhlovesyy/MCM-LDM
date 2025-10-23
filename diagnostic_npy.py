# diagnose_npy.py
import numpy as np
import sys

def analyze_motion_npy(file_path, label=""):
    """
    加载并分析一个动作 .npy 文件，打印其关键统计信息。
    """
    print("-" * 50)
    print(f"🔬 Analysing: {label} ({file_path})")
    print("-" * 50)

    try:
        motion_data = np.load(file_path)
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at '{file_path}'")
        return
    except Exception as e:
        print(f"❌ ERROR: Failed to load the file. Reason: {e}")
        return

    # 1. 形状 (Shape)
    # 期望的形状通常是 (num_frames, num_joints, num_dims)，例如 (196, 22, 3)
    print(f"🔹 Shape: {motion_data.shape}")
    if len(motion_data.shape) != 3 or motion_data.shape[2] != 3:
        print(f"   ⚠️ WARNING: Expected shape like (frames, joints, 3), but got {motion_data.shape}. "
              "This might be a major issue.")

    # 2. 数据类型 (Data Type)
    print(f"🔹 Data Type: {motion_data.dtype}")

    # 3. 检查 NaN 或 Inf 值
    nan_count = np.isnan(motion_data).sum()
    inf_count = np.isinf(motion_data).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"   🚨 CRITICAL ERROR: Found {nan_count} NaN values and {inf_count} Inf values. "
              "This will definitely cause rendering issues.")
    else:
        print("🔹 NaN / Inf Check: OK (No invalid values found)")

    # 4. 数值范围 (Min/Max per axis)
    # 这能帮助我们了解动作的空间范围
    min_vals = motion_data.min(axis=(0, 1))
    max_vals = motion_data.max(axis=(0, 1))
    print("🔹 Value Range (Min/Max):")
    print(f"   - X-axis: [{min_vals[0]:.4f}, {max_vals[0]:.4f}]")
    print(f"   - Y-axis: [{min_vals[1]:.4f}, {max_vals[1]:.4f}]")
    print(f"   - Z-axis: [{min_vals[2]:.4f}, {max_vals[2]:.4f}]")

    # 5. 整体统计数据 (Mean, Std, Variance)
    mean_val = motion_data.mean()
    std_val = motion_data.std()
    var_val = motion_data.var()
    print("🔹 Overall Statistics:")
    print(f"   - Mean: {mean_val:.4f}")
    print(f"   - Standard Deviation: {std_val:.4f}")
    print(f"   - Variance: {var_val:.4f}")
    
    # 6. 逐帧位移分析 (Frame-to-Frame Displacement)
    # 这可以帮助判断动作是否“飞了”或者“卡住了”
    if motion_data.shape[0] > 1:
        # 计算 root joint (通常是第 0 个关节) 的逐帧位移
        root_motion = motion_data[:, 0, :]
        displacements = np.linalg.norm(np.diff(root_motion, axis=0), axis=1)
        print("🔹 Root Joint Displacement Analysis:")
        print(f"   - Max Frame-to-Frame Distance: {displacements.max():.4f}")
        print(f"   - Avg Frame-to-Frame Distance: {displacements.mean():.4f}")
        if displacements.max() > 5.0: # 这是一个经验阈值，如果一帧移动了5个单位，可能太快了
            print("   ⚠️ WARNING: Maximum frame displacement is very high. The motion might be 'exploding'.")
        if displacements.mean() < 1e-4:
             print("   ⚠️ WARNING: Average frame displacement is very low. The motion might be 'frozen'.")

    print("-" * 50)
    print("\n")


if __name__ == "__main__":
    # --- [你需要修改这里] ---
    # 请将下面的路径替换为你的两个 .npy 文件的实际路径

    # 路径1: 原始 MCM-LDM 生成的、可以正常可视化的 .npy 文件
    original_good_npy = "/root/autodl-tmp/MyRepository/MCM-LDM/results/mld/testBaseline/style_transfer2025-06-10-17-42/jump_hands_high_38_scale_2-5.npy"
    
    # 路径2: 你的新模型生成的、渲染出纯白视频的 .npy 文件
    our_new_problematic_npy = "/root/autodl-tmp/MyRepository/MCM-LDM/experiments/mld/DualMode_MLD_Finetune_v1/motion_guidance_2025-10-22-20-58-45/jump_styled_by_hands_high.npy"
    
    # --- [修改结束] ---

    # 检查路径是否被修改过
    if "path/to/your" in original_good_npy or "path/to/your" in our_new_problematic_npy:
        print("🚨 Please edit the script `diagnose_npy.py` to set the correct file paths for your .npy files.")
        sys.exit(1)

    analyze_motion_npy(original_good_npy, label="✅ Original (Good)")
    analyze_motion_npy(our_new_problematic_npy, label="❓ Our New (Problematic)")