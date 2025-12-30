import sys
import re
from collections import defaultdict
from ast import literal_eval
import io 

def extract_and_rename_metrics(log_file_path, output_file_path, exp_name):
    """
    读取评估日志文件，使用暴力清洗方法去除所有不可见和非字典字符，然后提取关键指标。
    """
    
    # 定义需要提取和重命名的指标映射
    METRIC_MAP = {
        'fid_gen': 'FMD (fid_gen)',
        'accuracy_gen_top1': 'CRA (accuracy_gen_top1)',
        'accuracy_trans_top1': 'SRA (accuracy_trans_top1)',
    }
    
    final_metrics = defaultdict(str)
    metric_found = False

    try:
        # 使用 'rb' 读取二进制，然后解码并忽略编码错误
        with open(log_file_path, 'rb') as f:
            log_content = io.TextIOWrapper(f, encoding='utf-8', errors='ignore').read()
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        return

    # -----------------------------------------------------
    # 🌟 关键：暴力清洗步骤 🌟
    # -----------------------------------------------------

    # 1. 删除所有 ANSI 颜色代码 (最常见的问题)
    # 匹配格式如 \x1b[...m
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    cleaned_content = ansi_escape.sub('', log_content)

    # 2. 清除所有换行符、回车符和制表符
    cleaned_content = cleaned_content.replace('\r', '').replace('\n', '').replace('\t', '')

    # 3. 使用正则表达式查找字典块
    # 匹配 { 到 } 之间的所有内容，并允许任何字符 (非贪婪匹配)
    # re.DOTALL 确保 . 匹配换行符（尽管我们已经清除了换行符，但作为保险）
    pattern = re.compile(r"(\{.*?\})", re.DOTALL)
    metric_blocks = pattern.findall(cleaned_content)
    
    if not metric_blocks:
        print("Warning: No dictionary-like metrics blocks found after cleaning.")

    for block in metric_blocks:
        # 4. 对每个块进行最终的去空格和去引号处理 (确保 literal_eval 友好)
        final_block = block.strip()
        
        try:
            # 尝试将字符串转换为 Python 字典
            # Python literal_eval 允许使用单引号
            data = literal_eval(final_block)
            
            # 提取并重命名关注的指标
            for original_key, new_key in METRIC_MAP.items():
                if original_key in data:
                    final_metrics[new_key] = data[original_key]
                    metric_found = True
            
        except (ValueError, SyntaxError, AttributeError) as e:
            # 忽略无法解析的块
            # print(f"Skipping unparsable block: {final_block[:50]}... Error: {e}")
            pass

    # -----------------------------------------------------
    # 5. 写入和打印结果 (保持不变)
    # -----------------------------------------------------
    
    header_lines = [
        f"--- Evaluation Metrics for {exp_name} ---",
        f"Run Date: {sys.argv[3].strip()}",
        "========================================="
    ]
    
    with open(output_file_path, 'w') as f:
        for line in header_lines:
            f.write(line + '\n')
        
        if metric_found:
            for new_key, value in final_metrics.items():
                f.write(f"{new_key}: {value}\n")
        else:
            f.write("!! WARNING: No target metrics were successfully extracted. !!\n")

    print("-----------------------------------------------------")
    for line in header_lines:
        print(line)
    
    if metric_found:
        for new_key, value in final_metrics.items():
            print(f"{new_key}: {value}")
    else:
        print("!! WARNING: No target metrics were successfully extracted. !!")
        
    print("-----------------------------------------------------")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python extract_metrics.py <temp_log_path> <results_file_path> <exp_name>")
        sys.exit(1)
        
    temp_log_path = sys.argv[1]
    results_file_path = sys.argv[2]
    exp_name = sys.argv[3]
    
    extract_and_rename_metrics(temp_log_path, results_file_path, exp_name)