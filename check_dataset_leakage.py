import os

def check_leakage(train_file_path, val_file_path):
    """
    一个独立的脚本，用于检查训练集和验证集的分割文件是否有重叠。
    """
    try:
        with open(train_file_path, 'r') as f:
            train_ids = set(line.strip() for line in f)
        
        with open(val_file_path, 'r') as f:
            val_ids = set(line.strip() for line in f)
            
        print(f"--- Checking for data leakage between: ---")
        print(f"  - Train file: {os.path.basename(train_file_path)} ({len(train_ids)} unique IDs)")
        print(f"  - Val file:   {os.path.basename(val_file_path)} ({len(val_ids)} unique IDs)")

        intersection = train_ids.intersection(val_ids)

        if not intersection:
            print("\n[SUCCESS] No leakage detected! The train and val sets are perfectly disjoint.")
            return True
        else:
            print(f"\n[!!! DANGER !!!] Data Leakage Detected!")
            print(f"Found {len(intersection)} overlapping IDs between train and val sets.")
            print("Here are the first 5 overlapping IDs:")
            for i, item_id in enumerate(intersection):
                if i >= 5:
                    break
                print(f"  - {item_id}")
            return False

    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}. Please check your paths.")
        return None

if __name__ == '__main__':
    # --- 请在这里填写你的数据集的【绝对路径】 ---
    
    # 1. HumanML3D 的路径
    humanml3d_root = "/root/autodl-tmp/MyRepository/MCM-LDM/datasets/humanml3d"
    hml_train_path = os.path.join(humanml3d_root, 'train.txt')
    hml_val_path = os.path.join(humanml3d_root, 'val.txt')

    # 2. 100Style 的路径
    style100_root = "/root/autodl-tmp/MyRepository/MCM-LDM/datasets/100StyleDataset"
    s100_train_path = os.path.join(style100_root, 'train.txt')
    s100_val_path = os.path.join(style100_root, 'val.txt')
    
    # -----------------------------------------------

    print("\n--- Starting Data Leakage Check ---")
    
    check_leakage(hml_train_path, hml_val_path)
    print("-" * 40)
    check_leakage(s100_train_path, s100_val_path)
    
    print("\n--- Check Complete ---")