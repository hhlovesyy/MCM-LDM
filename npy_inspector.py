import numpy as np
import sys
import os

def inspect_npy_file(file_path):
    """
    載入 .npy 文件並打印其內部的數組結構（形狀、維度、數據類型等）。

    Args:
        file_path (str): .npy 文件的完整路徑。
    """
    if not os.path.exists(file_path):
        print(f"錯誤：文件未找到 -> {file_path}")
        return

    if not file_path.lower().endswith('.npy'):
        print(f"警告：'{file_path}' 不是一個典型的 .npy 文件。嘗試載入...")

    print("-" * 50)
    print(f"正在檢查文件: {os.path.basename(file_path)}")
    print("-" * 50)

    try:
        # 嘗試載入文件。
        # 使用 allow_pickle=True 是為了防止載入存儲了 Python 對象的 .npy 文件時報錯。
        data = np.load(file_path, allow_pickle=True)

        if isinstance(data, np.lib.npyio.NpzFile):
            # 如果是 .npz 壓縮文件（可能包含多個數組）
            print("文件類型: NumPy 壓縮文件 (.npz)")
            print("包含數組列表:")
            
            # 遍歷 .npz 中的每個數組
            for name in data.files:
                array = data[name]
                print(f"  數組名稱: '{name}'")
                print(f"    - 結構: {array.shape} (Shape)")
                print(f"    - 維度: {array.ndim} (Ndim)")
                print(f"    - 類型: {array.dtype} (DType)")
                print(f"    - 總元素: {array.size}")
            
            data.close() # 記得關閉 .npz 文件句柄

        else:
            # 如果是單個 .npy 文件或 .npz 中只有一個數組
            print("文件類型: 單個 NumPy 數組 (.npy)")
            print(f"  - 結構: {data.shape} (Shape)")
            print(f"  - 維度: {data.ndim} (Ndim)")
            print(f"  - 類型: {data.dtype} (DType)")
            print(f"  - 總元素: {data.size}")

    except ValueError as e:
        print(f"載入失敗：ValueError - {e}")
        print("這可能是一個格式錯誤的文件，或者需要特定的 `allow_pickle` 設置。")
    except Exception as e:
        print(f"載入過程中發生未知錯誤：{e}")

# --- 命令行執行入口 ---
if __name__ == '__main__':
    # 檢查是否提供了路徑參數
    if len(sys.argv) < 2:
        print("使用方法: python npy_inspector.py <NPY文件路徑>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    inspect_npy_file(file_path)