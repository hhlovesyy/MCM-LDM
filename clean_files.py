import os
import shutil
import datetime

def move_files_to_log():
    # ================= 配置区域 (请在此处核对路径) =================
    
    # 1. 源目录：我们要清理的地方
    SOURCE_DIR = "/root/autodl-tmp/MyRepository/MCM-LDM"
    
    # 2. 目标根目录：移动到哪里去
    # 注意：脚本会自动在这个目录下创建时间戳子文件夹
    LOGS_BASE_DIR = os.path.join(SOURCE_DIR, "logs")
    
    # 3. 白名单列表 (绝对路径 或 仅文件名 均可，这里为了保险推荐写文件名)
    # 这些文件绝对不会被移动
    WHITELIST_FILES = [
        "task_config.json",  # 这是你指定的白名单文件
        "run_evaluation.sh", # 建议保留脚本本身(可选)
        "run_evaluation_sca.sh" # 建议保留脚本本身(可选)
    ]
    
    # 4. 指定要移动的文件后缀 (不区分大小写)
    TARGET_EXTENSIONS = ('.json', '.txt', '.png', '.log')

    # ==============================================================

    # 1. 检查源目录是否存在
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ 错误：源目录不存在 -> {SOURCE_DIR}")
        return

    # 2. 生成带时间戳的目标子文件夹名称
    # 格式示例：2023年12月23日_21时30分
    current_time = datetime.datetime.now().strftime("%Y年%m月%d日_%H时%M分")
    target_dir = os.path.join(LOGS_BASE_DIR, current_time)

    # 3. 创建目标文件夹 (如果 logs 不存在也会一并创建)
    try:
        os.makedirs(target_dir, exist_ok=True)
        print(f"📂 目标文件夹已创建/确认: {target_dir}")
        print("-" * 50)
    except Exception as e:
        print(f"❌ 创建文件夹失败: {e}")
        return

    # 4. 开始扫描并移动
    moved_count = 0
    
    # os.listdir 只列出当前目录下的文件和文件夹名，不会递归进入子目录 -> 【符合要求】
    for filename in os.listdir(SOURCE_DIR):
        
        # 拼接完整路径
        file_path = os.path.join(SOURCE_DIR, filename)
        
        # A. 安全检查：必须是文件，不能是目录
        # 这保证了不会移动 'logs' 文件夹或者其他子文件夹 -> 【符合要求】
        if not os.path.isfile(file_path):
            continue

        # B. 白名单检查
        if filename in WHITELIST_FILES:
            print(f"🛡️  [跳过] 白名单文件: {filename}")
            continue

        # C. 后缀名检查
        # lower() 确保 .PNG 和 .png 都能被识别
        if file_path.lower().endswith(TARGET_EXTENSIONS):
            try:
                # D. 执行移动操作
                # shutil.move 会把文件从 source 挪到 target
                shutil.move(file_path, target_dir)
                print(f"✅ [移动] {filename} -> {target_dir}/")
                moved_count += 1
            except Exception as e:
                print(f"❌ [失败] 移动 {filename} 时出错: {e}")
        else:
            # 如果不是 json/txt/png，也不在白名单，默认忽略 (比如 .py, .sh 等)
            pass

    print("-" * 50)
    print(f"🎉 处理完成！共移动了 {moved_count} 个文件。")
    print(f"📍 它们现在位于: {target_dir}")

if __name__ == "__main__":
    move_files_to_log()