import os
import shutil

# --- 配置路径 ---
video_dir = "/root/autodl-tmp/MyRepository/MCM-LDM/vis_dataset_debug/forward/best/"
npy_source_dir = "/root/autodl-tmp/MyRepository/MCM-LDM/datasets/humanml3d/new_joint_vecs/"
output_dir = "/root/autodl-tmp/MyRepository/MCM-LDM/ANew113MotionPhy/Tool/selected_npy_forward"

def copy_pairs():
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"🚀 已创建输出目录: {output_dir}")

    # 获取所有视频文件名
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    print(f"🔍 正在扫描视频目录，发现 {len(video_files)} 个视频文件...")

    success_count = 0
    missing_npy_count = 0

    for video_file in video_files:
        # 获取不含后缀的文件名 (例如 '000222')
        base_name = os.path.splitext(video_file)[0]
        
        # 构造完整路径
        video_src = os.path.join(video_dir, video_file)
        npy_src = os.path.join(npy_source_dir, f"{base_name}.npy")
        
        # 检查对应的 npy 是否存在
        if os.path.exists(npy_src):
            # 拷贝 mp4
            shutil.copy2(video_src, os.path.join(output_dir, video_file))
            # 拷贝 npy
            shutil.copy2(npy_src, os.path.join(output_dir, f"{base_name}.npy"))
            success_count += 1
        else:
            print(f"⚠️  警告: 找不到对应的 npy 文件: {base_name}.npy")
            missing_npy_count += 1

    print("-" * 30)
    print(f"✅ 任务完成！")
    print(f"📦 成功拷贝配对数据: {success_count} 对")
    if missing_npy_count > 0:
        print(f"❌ 缺失 npy 文件数: {missing_npy_count}")
    print(f"📂 结果存储在: {output_dir}")

if __name__ == "__main__":
    copy_pairs()