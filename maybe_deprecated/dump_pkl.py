import pickle
import numpy as np
import os
import sys

# 定义输出日志文件名
LOG_FILE = "dumpPKL.log"

def analyze_and_dump_pkl(file_path, log_file):
    """
    读取 PKL 文件，解析所有字段，并输出其维度、统计信息和前 N 个值到日志文件。
    """
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"❌ 错误: 文件未找到: {file_path}\n")
        return

    try:
        # 1. 加载 PKL 文件
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
    except Exception as e:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"❌ 错误: 无法加载 PKL 文件。错误信息: {e}\n")
        return

    # 2. 写入日志文件，清空旧内容
    with open(log_file, 'w', encoding='utf-8') as log:
        log.write(f"==================================================\n")
        log.write(f"✅ PKL 文件分析结果: {file_path}\n")
        log.write(f"--------------------------------------------------\n")

        # 检查加载后的数据类型
        if isinstance(data, dict):
            # 如果是字典（常见的 pickle 格式）
            log.write(f"主数据类型: 字典 (包含 {len(data)} 个字段)\n")
            fields_to_analyze = data
        elif isinstance(data, (np.ndarray, list)):
            # 如果是单个数组或列表
            log.write(f"主数据类型: {type(data).__name__}\n")
            # 将其包装成字典方便后续统一处理
            fields_to_analyze = {"ROOT_OBJECT": data}
        else:
            log.write(f"主数据类型: {type(data).__name__}，无法提取结构化数据。\n")
            log.write("==================================================\n")
            return


        # 3. 遍历并分析每个字段
        for key, value in fields_to_analyze.items():
            log.write(f"\n--- 字段: {key} ---\n")
            
            # 尝试将其转换为 NumPy 数组以便分析
            if isinstance(value, list):
                try:
                    # 尝试将列表转换为 NumPy 数组
                    value = np.array(value)
                except:
                    # 如果转换失败（例如列表包含混合类型）
                    log.write(f"类型: 列表 (List), 长度: {len(value)}\n")
                    log.write(f"**无法进行 NumPy 统计分析**\n")
                    
                    # 输出前 N 个值
                    n_to_read = min(5, len(value))
                    log.write(f"前 {n_to_read} 个值: {value[:n_to_read]}\n")
                    # 看看value[0]的尺度
                    if len(value) > 0:
                        log.write(f"第一个元素的类型: {type(value[0]).__name__}\n")
                        # 不能打印一下维度吗？
                        if hasattr(value[0], 'shape'):
                            log.write(f"第一个元素的维度 (Shape): {value[0].shape}\n")
                    continue
            
            # 仅对 NumPy 数组进行维度和统计分析
            if isinstance(value, np.ndarray):
                log.write(f"类型: NumPy 数组 (ndarray)\n")
                log.write(f"维度 (Shape): {value.shape}\n")
                log.write(f"数据类型 (Dtype): {value.dtype}\n")

                if value.size > 0 and np.issubdtype(value.dtype, np.number):
                    # 仅对数值类型计算统计信息
                    log.write(f"统计信息:\n")
                    log.write(f"  - 最小值 (Min): {np.min(value)}\n")
                    log.write(f"  - 最大值 (Max): {np.max(value)}\n")
                    log.write(f"  - 均值 (Mean): {np.mean(value)}\n")
                    log.write(f"  - 标准差 (Std): {np.std(value)}\n")
                elif value.size > 0:
                    log.write(f"统计信息: 非数值类型，跳过 Min/Max/Mean/Std 计算。\n")
                
                # 输出前 N 个值 (切片操作)
                if value.ndim >= 1 and value.shape[0] > 0:
                    n_to_read = min(5, value.shape[0])
                    log.write(f"前 {n_to_read} 个切片 ([:{n_to_read}]):\n")
                    # 使用切片 [0:n_to_read] 提取第一个维度的前 n_to_read 个切片
                    log.write(str(value[:n_to_read]) + "\n")
                elif value.size == 0:
                    log.write("数组为空。\n")
            else:
                # 其他复杂类型，如 torch.Tensor, dict, str 等
                log.write(f"类型: {type(value).__name__}\n")
                log.write(f"无法进行标准 NumPy 维度/统计分析。对象字符串表示:\n")
                # 打印对象的前一部分表示
                output_str = str(value)
                log.write(output_str[:200] + ("..." if len(output_str) > 200 else "") + "\n")

        log.write("\n==================================================\n")
        log.write(f"结果已成功输出到 {log_file}\n")


if __name__ == "__main__":
    # 从命令行参数获取 PKL 文件路径
    if len(sys.argv) < 2:
        print("请在命令行中提供 PKL 文件路径。")
        print(f"用法: python {sys.argv[0]} <path/to/your/file.pkl>")
        sys.exit(1)
    
    pkl_file_path = sys.argv[1]
    
    # 打印到控制台，提示用户结果输出位置
    print(f"开始分析 PKL 文件: {pkl_file_path}")
    print(f"分析结果将输出到文件: {LOG_FILE}")
    
    analyze_and_dump_pkl(pkl_file_path, LOG_FILE)
    print("分析完成。请查看 dumpPKL.log 文件。")