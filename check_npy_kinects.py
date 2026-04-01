import numpy as np

NPY_PATH = r"/root/autodl-tmp/MyRepository/MCM-LDM/results/mld/SceneMoDiff_onlyTraj/style_transfer2026-03-26-21-52/000710-0_hands_high.npy"

JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'
]

PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19
]

def analyze_motion(path: str):
    x = np.load(path)
    print("=" * 60)
    print(f"File: {path}")
    print(f"Shape: {x.shape}")
    print(f"Dtype: {x.dtype}")
    print(f"Min/Max/Mean: {x.min():.4f} / {x.max():.4f} / {x.mean():.4f}")
    print(f"Has NaN: {np.isnan(x).any()} | Has Inf: {np.isinf(x).any()}")

    assert x.ndim == 3 and x.shape[1:] == (22, 3), "不是 (T,22,3)"

    root = x[:, 0, :]
    print("-" * 60)
    print("First frame pelvis:", root[0])
    print("Root X range:", float(root[:, 0].min()), "->", float(root[:, 0].max()))
    print("Root Y range:", float(root[:, 1].min()), "->", float(root[:, 1].max()))
    print("Root Z range:", float(root[:, 2].min()), "->", float(root[:, 2].max()))

    floor_y = x[:, :, 1].min()
    print("Global floor candidate (min Y):", float(floor_y))

    xz0 = root[0, [0, 2]]
    print("First frame pelvis XZ:", xz0)
    if np.all(np.abs(xz0) < 1e-3):
        print(">>> 看起来像“首帧XZ已归零”的规范化版本")
    else:
        print(">>> 更像“未做首帧XZ归零”的原始/世界空间版本")

    # 骨长稳定性检查
    print("-" * 60)
    bone_stats = []
    for j in range(1, 22):
        p = PARENTS[j]
        vec = x[:, j, :] - x[:, p, :]
        bone_len = np.linalg.norm(vec, axis=1)
        mean_len = bone_len.mean()
        std_len = bone_len.std()
        cv = std_len / (mean_len + 1e-8)
        bone_stats.append((JOINT_NAMES[j], mean_len, std_len, cv))

    bone_stats = sorted(bone_stats, key=lambda t: t[3], reverse=True)
    print("Top 10 bone-length variation:")
    for name, mean_len, std_len, cv in bone_stats[:10]:
        print(f"{name:16s} mean={mean_len:.4f} std={std_len:.4f} cv={cv:.4f}")

    avg_cv = np.mean([t[3] for t in bone_stats])
    print("Average bone-length CV:", float(avg_cv))
    if avg_cv < 0.05:
        print(">>> 骨长整体比较稳定，像是真实骨架关节位置序列")
    else:
        print(">>> 骨长变化偏大，需警惕关节顺序/坐标解释不一致")

    # 朝向粗看：左右髋 + 左右肩的横向向量
    print("-" * 60)
    first = x[0]
    across_hips = first[2] - first[1]
    across_shoulders = first[17] - first[16]
    print("First frame hip-across:", across_hips)
    print("First frame shoulder-across:", across_shoulders)

    print("=" * 60)

if __name__ == "__main__":
    analyze_motion(NPY_PATH)