import numpy as np
import os
from tqdm import tqdm
import warnings

# ==============================================================================
# HumanML3D 263-dim Feature Definition (Deduced from statistics)
# ==============================================================================
# Based on statistical analysis of typical HumanML3D data, we can deduce a likely structure.
# This may not be perfectly semantically accurate but is sufficient for comparing distributions.
FEATURE_SLICES = {
    # The first 4 features often represent root velocity (X, Z), angular velocity (Y), and height (Y pos).
    # Index 3 is very likely the root height from the ground.
    "root_summary": slice(0, 4),
    
    # The vast majority of features representing the body's pose and movement.
    "body_motion": slice(4, 259),
    
    # The last 4 features are consistently the binary foot contact labels.
    "foot_contact": slice(259, 263),
}


def calculate_statistics(data_dir):
    """
    Calculates mean, std, min, max for all .npy files in a directory,
    while automatically skipping files containing NaN values.
    """
    print(f"Calculating statistics for: {data_dir}")
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    if not file_list:
        raise ValueError(f"No .npy files found in {data_dir}")

    all_frames = []
    nan_files = []
    
    for filename in tqdm(file_list, desc=f"Loading motions from {os.path.basename(data_dir)}"):
        file_path = os.path.join(data_dir, filename)
        try:
            motion_data = np.load(file_path)

            # --- 【核心修改】检查 NaN 值 ---
            # np.isnan(motion_data).any() 会检查整个数组中是否存在任何 NaN 值。
            if np.isnan(motion_data).any():
                nan_files.append(filename)
                # 如果发现 NaN，则跳过这个文件，不将其加入计算
                continue
            
            if motion_data.ndim == 2 and motion_data.shape[1] == 263:
                all_frames.append(motion_data)
            else:
                warnings.warn(f"Skipping file {filename} with unexpected shape {motion_data.shape}")

        except Exception as e:
            warnings.warn(f"Could not load or process file {filename}: {e}")

    # 如果有文件因为 NaN 被跳过，打印一个汇总报告
    if nan_files:
        print("\n" + "*"*50)
        print(f"WARNING: Skipped {len(nan_files)} files containing NaN values:")
        # 只打印前 10 个文件名，防止刷屏
        for i, fname in enumerate(nan_files[:10]):
            print(f"  - {fname}")
        if len(nan_files) > 10:
            print(f"  ... and {len(nan_files) - 10} more.")
        print("*"*50 + "\n")

    if not all_frames:
        raise ValueError("No valid motion data could be loaded (all files might be corrupted or have wrong shape).")
        
    all_frames = np.concatenate(all_frames, axis=0)
    
    stats = {
        'mean': all_frames.mean(axis=0),
        'std': all_frames.std(axis=0),
        'min': all_frames.min(axis=0),
        'max': all_frames.max(axis=0)
    }
    stats['std'][stats['std'] == 0] = 1.0
    return stats

def print_comparison_table(stats1, stats2, name1="Dataset 1", name2="Dataset 2"):
    """Prints a formatted comparison table of the statistics."""
    
    print("\n" + "="*80)
    print(f"STATISTICAL COMPARISON: {name1} vs. {name2}")
    print("="*80)

    for feature_name, feature_slice in FEATURE_SLICES.items():
        print(f"\n--- Feature: {feature_name} (slice: {feature_slice.start}:{feature_slice.stop}) ---\n")
        
        headers = ["Stat", name1, name2, "Abs. Difference", "Ratio (D1/D2)"]
        print(f"{headers[0]:<10} | {headers[1]:<20} | {headers[2]:<20} | {headers[3]:<20} | {headers[4]:<20}")
        print("-"*95)
        
        for stat_name in ['mean', 'std', 'min', 'max']:
            # 使用 np.nanmean 来安全地计算，尽管我们已经过滤了NaN文件
            val1 = np.mean(stats1[stat_name][feature_slice])
            val2 = np.mean(stats2[stat_name][feature_slice])
            diff = np.abs(val1 - val2)
            ratio = val1 / (val2 + 1e-8)
            
            print(f"{stat_name:<10} | {val1:<20.6f} | {val2:<20.6f} | {diff:<20.6f} | {ratio:<20.2f}")
    
    print("\n" + "="*80)


def main():
    # --- Configuration ---
    new_dataset_path = "/root/autodl-tmp/MyRepository/MCM-LDM/datasets/humanml3d_scene/new_joint_vecs"
    hml3d_dataset_path = "/root/autodl-tmp/MyRepository/MCM-LDM/datasets/humanml3d/new_joint_vecs"
    
    # --- Execution ---
    try:
        print("Processing New Dataset...")
        stats_new = calculate_statistics(new_dataset_path)
        
        print("\nProcessing HumanML3D Dataset...")
        stats_hml3d = calculate_statistics(hml3d_dataset_path)

        print_comparison_table(stats_new, stats_hml3d, name1="Your New Dataset", name2="HumanML3D")
        
        print("\n--- FORENSIC ANALYSIS ---")
        print("1. Check Root Height (part of 'root_summary'): Does the 'mean' value differ significantly? (e.g., ~1 vs ~100) => Unit mismatch (m vs cm).")
        print("2. Check Root Motion (part of 'root_summary'): Are the 'mean' signs opposite for velocities? => Flipped coordinate axis (Z-axis forward vs. backward).")
        print("3. Check Foot Contact: Are 'min' and 'max' values for both datasets close to 0 and 1? => Representation mismatch.")
        print("4. Check Body Motion: Is the 'std' Ratio very large or small (e.g., >10 or <0.1)? => Strong indicator of unit mismatch for positions/rotations.")

    except (FileNotFoundError, ValueError) as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        print("Please ensure the dataset paths are correct and the directories contain valid .npy files.")

if __name__ == "__main__":
    main()