import sys
import os
import json
import torch
import numpy as np

# ================= 配置 =================
PROJECT_ROOT = "/root/autodl-tmp/MyRepository/MCM-LDM"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 尝试导入 demo_physics
try:
    from demo_physics import parse_inference_json
    print(">>> 成功导入 demo_physics.parse_inference_json")
except ImportError as e:
    print(f"【错误】导入失败: {e}")
    sys.exit(1)

def run_test_case(name, input_dict):
    # 1. 创建临时 JSON 文件
    temp_file = "temp_test_case.json"
    with open(temp_file, 'w') as f:
        json.dump(input_dict, f)
    
    # 2. 调用解析函数
    try:
        phys_vec, _ = parse_inference_json(temp_file)
        vec = phys_vec.numpy()
        
        # 3. 打印结果
        print("-" * 60)
        print(f"测试用例: {name}")
        print(f"输入 JSON: {json.dumps(input_dict['physical_parameters'])}")
        print(f"输出向量: [Wx: {vec[0]:.2f}, Wy: {vec[1]:.2f}, Mag: {vec[2]:.2f}, Ceil: {vec[3]:.2f}, GapW: {vec[4]:.2f}, GapOff: {vec[5]:.2f}]")
        
        # 4. 自动验证逻辑
        return vec
    except Exception as e:
        print(f"测试出错: {e}")
        return None
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def main():
    print("开始测试 Demo 推理逻辑映射...")
    
    # === Case 1: 0 风 (关键测试) ===
    # 预期：Mag = 0.00
    vec = run_test_case("1. 无风 (Wind=0)", {
        "physical_parameters": {
            "wind_force": {"x": 0.0, "y": 0.0}
        }
    })
    if vec[2] == 0.0: print("✅ 通过：风力成功归零！")
    else: print("❌ 失败：0风没有归零！")

    # === Case 2: 100 风 (最大风) ===
    # 预期：Mag = 1.00
    vec = run_test_case("2. 最大风 (Wind=100)", {
        "physical_parameters": {
            "wind_force": {"x": 100.0, "y": 0.0}
        }
    })
    if abs(vec[2] - 1.0) < 0.01: print("✅ 通过：100风映射为 1.0！")
    else: print("❌ 失败：100风映射错误！")

    # === Case 3: 50 风 (中风) ===
    # 预期：Mag = 0.50
    vec = run_test_case("3. 中风 (Wind=50)", {
        "physical_parameters": {
            "wind_force": {"x": 50.0, "y": 0.0}
        }
    })
    if abs(vec[2] - 0.5) < 0.01: print("✅ 通过：50风映射为 0.5！")
    else: print("❌ 失败：50风映射错误！")

    # === Case 4: 混合场景 (0风 + 低天花板) ===
    # 预期：Mag = 0.00, Ceil > 0.6
    vec = run_test_case("4. 混合 (Wind=0, Ceil=80)", {
        "physical_parameters": {
            "wind_force": {"x": 0.0, "y": 0.0},
            "ceiling_height": 80.0
        }
    })
    if vec[2] == 0.0 and vec[3] > 0.6: print("✅ 通过：风力为0且天花板生效！(互斥逻辑已移除)")
    else: print("❌ 失败：混合逻辑错误！")
    
    vec = run_test_case("4. 混合 (Wind=0, Ceil=80)", {
        "physical_parameters": {
            "wind_force": {"x": 0.0, "y": 0.0},
            "ceiling_height": -220000.0
        }
    })
    if vec[2] == 0.0 and vec[3] > 0.6: print("✅ 通过：风力为0且天花板生效！(互斥逻辑已移除)")
    else: print("❌ 失败：混合逻辑错误！")


    # === Case 5: 窄缝隙 ===
    # 预期：GapW > 0.6
    vec = run_test_case("5. 窄缝隙 (Gap=40)", {
        "physical_parameters": {
            "gap_width": 40.0,
            "gap_offset": 0.0
        }
    })
    if vec[4] > 0.6: print("✅ 通过：窄缝隙产生强信号！")
    else: print("❌ 失败：窄缝隙信号太弱！")
    
    # === Case 5: 窄缝隙 ===
    # 预期：GapW > 0.6
    vec = run_test_case("5. 窄缝隙 (Gap=40)", {
        "physical_parameters": {
            "gap_width": 10.0,
            "gap_offset": 0.0
        }
    })
    if vec[4] > 0.6: print("✅ 通过：窄缝隙产生强信号！")
    else: print("❌ 失败：窄缝隙信号太弱！")

if __name__ == "__main__":
    print("开始..")
    main()