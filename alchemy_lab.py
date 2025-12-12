import streamlit as st
import yaml
import os
import subprocess
import datetime
import copy  # <--- 新增这一行！非常重要！
import time
import glob
import re
import json

# ================= 配置路径 =================
ROOT_DIR = "/root/autodl-tmp/MyRepository/MCM-LDM"
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
BASE_YAML_PATH = os.path.join(CONFIG_DIR, "scenemodiff_train_LiandanBase.yaml")
TRAIN_SCRIPT = os.path.join(ROOT_DIR, "train.py")
ASSETS_FILE = os.path.join(CONFIG_DIR, "assets.yaml")
STATE_FILE = os.path.join(ROOT_DIR, "alchemy_state.json") # 保存状态的文件

# ================= 页面设置 =================
st.set_page_config(page_title="沪爷 智能炼丹炉", layout="wide", page_icon="🔥")

st.title("🔥 沪爷 智能炼丹控制台")
st.markdown("Varsapura！上海萨普！")

# ================= 状态持久化函数 (小功能1) =================
def save_state(key, value):
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                data = json.load(f)
        else:
            data = {}
        data[key] = value
        with open(STATE_FILE, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"State save failed: {e}")

def load_state(key, default=None):
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                data = json.load(f)
            return data.get(key, default)
    except:
        pass
    return default

# 初始化 Session State (从文件加载)
if 'init_loaded' not in st.session_state:
    st.session_state.last_exp_name = load_state('last_exp_name', "SceneMo_Default")
    st.session_state.last_mode = load_state('last_mode', "炼丹 (Training)")
    st.session_state.init_loaded = True

# ================= 左侧导航栏 =================
modes = ["我要炼丹 (Training)", "我要推理 (Inference)", "我要评估 (Eval)", "看看你的：渲染 (Render)", "沪爷工具箱", "我要玩原神！"]
# 自动选中上次的模式
default_mode_idx = modes.index(st.session_state.last_mode) if st.session_state.last_mode in modes else 0
mode = st.sidebar.radio("选择模式", modes, index=default_mode_idx)

# 保存当前模式选择
if mode != st.session_state.last_mode:
    save_state('last_mode', mode)
# ================= 功能函数 =================
def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data, path):
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

def run_in_screen(command, session_name):
    """在 Screen 会话中后台运行命令"""
    # -dmS 创建一个 detached session
    # bash -c '...; exec bash' 让任务跑完后窗口不关闭，方便查看报错
    full_cmd = f"screen -dmS {session_name} bash -c 'cd {ROOT_DIR}; {command}; echo ----------------Task Finished----------------; exec bash'"
    subprocess.Popen(full_cmd, shell=True)



# ================= 小功能2: 进程查看器 =================
st.sidebar.divider()
st.sidebar.subheader("进程监控 (Process Monitor)")
if st.sidebar.checkbox("显示 Python 进程"):
    try:
        # 查找包含 'train.py' 或 'demo' 的 python 进程
        cmd = "ps -ef | grep python | grep -v grep | grep -E 'train.py|demo|render|eval'"
        proc_output = subprocess.check_output(cmd, shell=True).decode("utf-8")
        st.sidebar.code(proc_output if proc_output else "无相关进程运行")
        st.sidebar.caption("提示：使用 `kill -9 PID` 终止进程")
    except:
        st.sidebar.warning("查询进程失败")

# ================= 1. 炼丹模式 =================
if mode == "我要炼丹 (Training)":
    st.header("⚗️ 炼丹配置室")

    # 检查 Base 文件
    if not os.path.exists(BASE_YAML_PATH):
        st.error(f"找不到基准配置文件: {BASE_YAML_PATH}，请先复制一个现有的yaml重命名为该文件名！")
        st.stop()

    # 读取基准配置
    base_config = load_yaml(BASE_YAML_PATH)
    sc_config_base = base_config.get('SCENE_MODIFF_ABLATION', {})

    # === 定义预设配置 ===
    PRESETS = {
        "1. Full Model (FiLM+Loss)": {
            "FUSION": "film", "LOSS": True, "JUST_BASE": False, "LAMBDA": 0.2, "DESC": "完整模型：FiLM融合 + 场景一致性Loss"
        },
        "2. Only FiLM (No Loss)": {
            "FUSION": "film", "LOSS": False, "JUST_BASE": False, "LAMBDA": 0.2, "DESC": "验证Loss作用：保留FiLM，关掉Loss"
        },
        "3. MLP Fusion (No FiLM)": {
            "FUSION": "mlp", "LOSS": True, "JUST_BASE": False, "LAMBDA": 0.2, "DESC": "验证FiLM作用：退化为MLP融合，保留Loss"
        },
        "4. Only Baseline": {
            "FUSION": "mlp", "LOSS": False, "JUST_BASE": True, "LAMBDA": 0.0, "DESC": "纯基线：无FiLM，无Loss，无模块"
        },
        "5. Custom (自定义)": {
            "FUSION": sc_config_base.get('FUSION_MODE', 'film'), 
            "LOSS": sc_config_base.get('USE_SCENE_CLS', True), 
            "JUST_BASE": sc_config_base.get('JUST_FINETUNE_BASELINE', False), 
            "LAMBDA": sc_config_base.get('LAMBDA_SCENE', 0.2),
            "DESC": "自由调整参数，不使用预设模板"
        }
    }

    # === 回调函数：强制更新控件的值 ===
    def update_params_callback():
        selection = st.session_state.preset_selector
        cfg = PRESETS[selection]
        
        # 直接修改控件绑定的 key 的值
        st.session_state.widget_fusion = cfg["FUSION"] # 直接设为 'film' 或 'mlp'
        st.session_state.k_loss = cfg["LOSS"]
        st.session_state.k_just_base = cfg["JUST_BASE"]
        st.session_state.k_lambda = float(cfg["LAMBDA"])
        
        # 更新实验名
        time_str = datetime.datetime.now().strftime("%m%d_%H%M")
        if "Full Model" in selection:
            name_suffix = "Full_FiLM_Loss"
        elif "Only FiLM" in selection:
            name_suffix = "FiLM_NoLoss"
        elif "MLP" in selection:
            name_suffix = "MLP_WithLoss"
        elif "Baseline" in selection:
            name_suffix = "BaselineOnly"
        else:
            name_suffix = "Custom"
        st.session_state.k_exp_name = f"SceneMo_{time_str}_{name_suffix}"

    # === 顶部选择栏 ===
    selected_preset = st.radio(
        "⚡ 快速选择实验配置:",
        options=list(PRESETS.keys()),
        horizontal=True,
        key="preset_selector",
        on_change=update_params_callback
    )
    st.info(f"💡 说明: {PRESETS[selected_preset]['DESC']}")
    st.divider()

    # === 参数编辑区 ===
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("核心超参")
        
        # 初始化默认值 (防止第一次运行报错)
        if 'widget_fusion' not in st.session_state:
            init_cfg = PRESETS["1. Full Model (FiLM+Loss)"]
            st.session_state.widget_fusion = init_cfg["FUSION"]
            st.session_state.k_loss = init_cfg["LOSS"]
            st.session_state.k_just_base = init_cfg["JUST_BASE"]
            st.session_state.k_lambda = init_cfg["LAMBDA"]
            st.session_state.k_exp_name = "SceneMo_Init"

        # 控件定义
        # FUSION_MODE: key='widget_fusion'，回调函数会直接修改这个key的值，从而改变下拉框的选项
        fusion_mode = st.selectbox(
            "FUSION_MODE (融合方式)", 
            options=["film", "mlp"], 
            key="widget_fusion"
        )
        
        use_scene_cls = st.checkbox("USE_SCENE_CLS (使用场景Loss)", key="k_loss")
        just_finetune = st.checkbox("JUST_FINETUNE_BASELINE (只微调基线)", key="k_just_base")
        lambda_scene = st.number_input("LAMBDA_SCENE (Loss权重)", format="%.2f", step=0.1, key="k_lambda")

    with col2:
        st.subheader("训练设置")
        train_config = base_config.get('TRAIN', {})
        optim_config = train_config.get('OPTIM', {})
        
        lr = st.text_input("Learning Rate", value=str(optim_config.get('LR', '2e-5')))
        batch_size = st.number_input("Batch Size", value=int(train_config.get('BATCH_SIZE', 32)))
        end_epoch = st.number_input("End Epoch", value=int(train_config.get('END_EPOCH', 100)))
        
        # exp_name = st.text_input("Experiment NAME (文件夹名)", key="k_exp_name")
        exp_name = st.text_input("Experiment NAME", value=st.session_state.last_exp_name)


    # ================= 预览与执行 =================
    st.divider()
    
    # 构造新的 Config 字典
    new_config = copy.deepcopy(base_config) # Deepcopy 防止修改污染 Base
    new_config['NAME'] = exp_name
    new_config['SCENE_MODIFF_ABLATION']['FUSION_MODE'] = fusion_mode
    new_config['SCENE_MODIFF_ABLATION']['USE_SCENE_CLS'] = use_scene_cls
    new_config['SCENE_MODIFF_ABLATION']['LAMBDA_SCENE'] = lambda_scene
    new_config['SCENE_MODIFF_ABLATION']['JUST_FINETUNE_BASELINE'] = just_finetune
    
    new_config['TRAIN']['BATCH_SIZE'] = batch_size
    new_config['TRAIN']['END_EPOCH'] = end_epoch
    new_config['TRAIN']['OPTIM']['LR'] = float(lr)

    with st.expander("👀 预览生成的 YAML 内容"):
        st.code(yaml.dump(new_config, default_flow_style=False), language='yaml')

    if st.button("🚀 开始炼丹 (Start Training)", type="primary"):
        # 1. 创建实验目录
        exp_dir = os.path.join(ROOT_DIR, "experiments", "mld", exp_name)
        try:
            os.makedirs(exp_dir, exist_ok=True)
            st.success(f"✅ 实验目录已确认: {exp_dir}")
        except Exception as e:
            st.error(f"❌ 创建目录失败: {e}")
            st.stop()

        # 2. 保存 YAML 到实验目录
        new_yaml_path = os.path.join(exp_dir, "launcher_config.yaml")
        save_yaml(new_config, new_yaml_path)
        save_state('last_exp_name', exp_name)
        st.info(f"📄 配置已保存: {new_yaml_path}")
        
        # 3. 构造命令
        screen_name = f"train_{exp_name}"[:25]
        cmd = f"python {TRAIN_SCRIPT} --cfg {new_yaml_path} --cfg_assets {ASSETS_FILE} --batch_size {batch_size} --nodebug"
        
        # 4. 运行
        run_in_screen(cmd, screen_name)
        
        st.balloons()
        st.markdown(f"""
        ### 🎉 任务已启动！
        - **Session**: `{screen_name}`
        - **Config**: `{new_yaml_path}`
        
        查看进度: `screen -D -r {screen_name}`
        """)
    
elif mode == "我要推理 (Inference)":
    st.header("🔮 智能推理控制台")

    # --- 1. 选择实验权重 ---
    st.subheader("1. 选择模型权重 (Checkpoint)")
    
    # 扫描实验目录
    exp_root = os.path.join(ROOT_DIR, "experiments", "mld")
    if os.path.exists(exp_root):
        # 按修改时间排序，最近的在前面
        exps = sorted(os.listdir(exp_root), key=lambda x: os.path.getmtime(os.path.join(exp_root, x)), reverse=True)
    else:
        exps = []

    if not exps:
        st.warning("⚠️ 还没有实验记录，先去炼丹吧！")
        st.stop()

    selected_exp = st.selectbox("选择实验记录", exps)
    exp_path = os.path.join(exp_root, selected_exp)

    # 扫描 Checkpoints
    ckpt_dir = os.path.join(exp_path, "checkpoints")
    if os.path.exists(ckpt_dir):
        ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
        # 只保留文件名
        ckpt_names = [os.path.basename(c) for c in ckpts]
        # 排序：epoch=100 排在 epoch=9 前面
        ckpt_names = sorted(ckpt_names, key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0, reverse=True)
    else:
        ckpt_names = []

    if not ckpt_names:
        st.error(f"❌ 在 {selected_exp} 里找不到 .ckpt 文件！")
        st.stop()

    selected_ckpt_name = st.selectbox("选择 Checkpoint", ckpt_names)
    selected_ckpt_path = os.path.join(ckpt_dir, selected_ckpt_name)
    
    # --- 2. 场景 Prompt 设置 ---
    st.subheader("2. 场景与文本 (Prompt)")
    
    SCENE_DESCRIPTIONS = {
        "Dumuqiao (独木桥)": "Walking on a narrow bridge.",
        "DiAiTongDao (低矮通道)": "Crouching while walking.",
        "ShuiKengDiMian (水坑地面)": "Walking on muddy ground.",
        "BoLiFangJian (玻璃房间)": "Walking in a glass room.",
        "T_Stage (T台走秀)": "Fashion model walking.",
        "CroudedPlace (拥挤场合)": "Walking through a crowd.",
        "DiAiTianhuaban (低矮天花板)": "Walking under a low ceiling.",
        "Bar (酒吧/醉酒)": "Drunk walking.",
        "WalkInSnowOrSand (雪地/沙地)": "Walking in deep snow.",
        "Dark (摸黑)": "Walking in the dark.",
        "LeanLeft (左倾)": "Leaning left.",
        "WetFloor (潮湿地面)": "Slippery floor.",
        "BaoFengYu (暴风雨)": "Walking in strong wind.",
        "IcyRoad (冰面)": "Walking on ice.",
        "Custom (自定义)": ""
    }
    
    scene_key = st.selectbox("选择场景预设", list(SCENE_DESCRIPTIONS.keys()))
    
    if scene_key == "Custom (自定义)":
        prompt_text = st.text_input("输入自定义 Prompt", "Walking carefully.")
    else:
        prompt_text = st.text_input("Prompt 内容", SCENE_DESCRIPTIONS[scene_key])

    # --- 3. 目录与 Hack 设置 ---
    st.subheader("3. 数据与参数注入")
    
    c1, c2, c3 = st.columns(3)
    
    # 扫描 Demo 目录下的子文件夹方便选择
    demo_root = os.path.join(ROOT_DIR, "demo")
    demo_subdirs = [d for d in os.listdir(demo_root) if os.path.isdir(os.path.join(demo_root, d))] if os.path.exists(demo_root) else ["Final_figure_content"]
    
    with c1:
        content_dir_name = st.selectbox("Content Motion Dir", demo_subdirs, index=0)
        content_dir = os.path.join("demo", content_dir_name) # 相对路径
        
    with c2:
        style_dir_name = st.selectbox("Style Motion Dir", demo_subdirs, index=min(1, len(demo_subdirs)-1))
        style_dir = os.path.join("demo", style_dir_name) # 相对路径
        
    with c3:
        # The Hack!
        scene_scalar = st.number_input("🔥 Scene Scalar (注入 mld.py)", value=3.0, step=0.1, help="直接修改源码中的 scene_scalar 变量")

    # --- 执行逻辑 ---
    st.divider()
    
    if st.button("🔮 开始推理 (Start Inference)", type="primary"):
        # A. 准备 Config 文件
        # 尝试读取 launcher_config.yaml, 如果没有就找别的
        launcher_yaml = os.path.join(exp_path, "launcher_config.yaml")
        if not os.path.exists(launcher_yaml):
            # 备选方案：找任何一个yaml
            yamls = glob.glob(os.path.join(exp_path, "*.yaml"))
            if yamls:
                launcher_yaml = yamls[0]
            else:
                st.error("❌ 找不到对应的 yaml 配置文件")
                st.stop()
        
        # 修改 Config
        inf_config = load_yaml(launcher_yaml)
        
        # 强制覆盖测试参数
        if 'TEST' not in inf_config: inf_config['TEST'] = {}
        inf_config['TEST']['CHECKPOINTS'] = selected_ckpt_path
        inf_config['TEST']['MULTI_MODAL_TYPE'] = 'text'
        inf_config['TEST']['MULTI_MODAL_TEXT_PROMPT'] = prompt_text
        
        # 保存临时推理 Config
        temp_inf_yaml = os.path.join(exp_path, f"inference_{scene_key.split(' ')[0]}.yaml")
        save_yaml(inf_config, temp_inf_yaml)
        st.success(f"✅ 推理配置已生成: {temp_inf_yaml}")
        
        # B. 执行源码注入 (The Dirty Hack)
        mld_py_path = os.path.join(ROOT_DIR, "mld/models/modeltype/mld.py")
        try:
            with open(mld_py_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
            
            # 使用正则替换 scene_scalar = x.x
            # 假设源码里是 scene_scalar = 1.0 或 scene_scalar=1.0
            new_code = re.sub(r"DEFAULT_SCALAR_VAL\s*=\s*[\d\.]+", f"DEFAULT_SCALAR_VAL = {scene_scalar}", code_content)
            
            with open(mld_py_path, 'w', encoding='utf-8') as f:
                f.write(new_code)
                
            st.warning(f"⚠️ 已将 mld.py 中的 scene_scalar 修改为 {scene_scalar}")
        except Exception as e:
            st.error(f"❌ 源码注入失败: {e}")
            st.stop()
            
        # C. 构造命令
        # 参考你的 bash 脚本逻辑
        # PYTHON_CMD="python demo_transfer_with_scene.py"
        # CFG_ARGS="--cfg configs/config_mld_humanml3d_with_scene.yaml" -> 这里用我们生成的 temp_inf_yaml
        # ASSETS_ARGS="--cfg_assets configs/assets.yaml"
        # MOTION_ARGS="--content_motion_dir ... --style_motion_dir ..."
        # SCALE_ARGS="--scale 2.5"
        
        script_name = "demo_transfer_with_scene.py"
        cmd = f"python {script_name} --cfg {temp_inf_yaml} --cfg_assets {ASSETS_FILE} --content_motion_dir {content_dir} --style_motion_dir {style_dir} --scale 2.5"
        
        screen_name = f"inf_{scene_key.split(' ')[0]}"[:20]
        
        run_in_screen(cmd, screen_name)
        
        st.balloons()
        st.markdown(f"""
        ### 🔮 推理任务已启动！
        - **Scalar**: `{scene_scalar}` (Injected)
        - **Prompt**: `{prompt_text}`
        - **Ckpt**: `{os.path.basename(selected_ckpt_path)}`
        
        **Check Progress:**
        ```bash
        screen -D -r {screen_name}
        ```
        *(推理结果会保存在 /root/autodl-tmp/MyRepository/MCM-LDM/results/mld 下面，可以去查看)*
        """)

elif mode == "我要评估 (Eval)":
    st.header("📊 智能评估中心 (Two-Stage)")

    # --- 公共组件：选择实验和权重 ---
    st.subheader("0. 基础设置 (Select Experiment)")
    
    exp_root = os.path.join(ROOT_DIR, "experiments", "mld")
    if os.path.exists(exp_root):
        exps = sorted(os.listdir(exp_root), key=lambda x: os.path.getmtime(os.path.join(exp_root, x)), reverse=True)
    else:
        exps = []

    if not exps:
        st.warning("⚠️ 没有实验记录")
        st.stop()

    # 这里选中的 experiment 用于寻找 yaml 和后续寻找 pkl
    eval_exp_name = st.selectbox("选择要评估的实验", exps, key="eval_exp_select")
    eval_exp_path = os.path.join(exp_root, eval_exp_name)

    # 扫描 Checkpoints
    ckpt_dir = os.path.join(eval_exp_path, "checkpoints")
    if os.path.exists(ckpt_dir):
        ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
        ckpt_names = [os.path.basename(c) for c in ckpts]
        # 简单排序
        ckpt_names = sorted(ckpt_names, key=lambda x: len(x), reverse=True) 
    else:
        ckpt_names = []
        
    eval_ckpt_name = st.selectbox("选择 Checkpoint", ckpt_names, key="eval_ckpt_select")
    eval_ckpt_path = os.path.join(ckpt_dir, eval_ckpt_name) if eval_ckpt_name else ""

    st.divider()

    # ================= Stage 1: Standard Evaluation =================
    st.subheader("Stage 1: 生成与标准指标 (FMD/CRA/SRA)")
    st.info("💡 运行 `run_evaluation.sh`。这会生成 motion pkl 文件并计算基础指标。")

    if st.button("🚀 运行 Stage 1 (Standard Eval)", type="primary"):
        if not eval_ckpt_path:
            st.error("请先选择 Checkpoint")
            st.stop()
            
        # 1. 准备临时的 Evaluation Config
        # 读取实验目录下的 launcher_config.yaml
        launcher_yaml = os.path.join(eval_exp_path, "launcher_config.yaml")
        if not os.path.exists(launcher_yaml):
            # 备选：找任何一个yaml
            yamls = glob.glob(os.path.join(eval_exp_path, "*.yaml"))
            launcher_yaml = yamls[0] if yamls else ""
        
        if not launcher_yaml:
            st.error("找不到配置文件 yaml")
            st.stop()
            
        eval_cfg = load_yaml(launcher_yaml)
        # 强制修改 TEST.CHECKPOINTS
        if 'TEST' not in eval_cfg: eval_cfg['TEST'] = {}
        eval_cfg['TEST']['CHECKPOINTS'] = eval_ckpt_path
        
        # 保存为 configs/use_for_evaluation.yaml (脚本里写死读取这个)
        # 或者为了安全，保存为一个新文件，然后修改bash脚本指向它
        temp_eval_yaml_path = os.path.join(CONFIG_DIR, f"eval_temp_{eval_exp_name}.yaml")
        save_yaml(eval_cfg, temp_eval_yaml_path)
        
        # 2. 修改 run_evaluation.sh (Bash Injection)
        bash_script_path = os.path.join(ROOT_DIR, "run_evaluation.sh")
        try:
            with open(bash_script_path, 'r', encoding='utf-8') as f:
                bash_content = f.read()
            
            # 使用正则替换 CONFIG_MLD 和 EXP_NAME
            # 这里的 EXP_NAME 建议加个后缀，方便区分
            target_exp_name = f"{eval_exp_name}_Eval"
            
            # 替换 CONFIG_MLD="..."
            bash_content = re.sub(r'CONFIG_MLD=".*?"', f'CONFIG_MLD="{temp_eval_yaml_path}"', bash_content)
            # 替换 EXP_NAME="..."
            bash_content = re.sub(r'EXP_NAME=".*?"', f'EXP_NAME="{target_exp_name}"', bash_content)
            
            with open(bash_script_path, 'w', encoding='utf-8') as f:
                f.write(bash_content)
                
            st.success(f"✅ Bash 脚本已修改: Target Exp Name = {target_exp_name}")
            
        except Exception as e:
            st.error(f"❌ 修改 Bash 脚本失败: {e}")
            st.stop()
            
        # 3. 运行并重定向日志
        log_file = os.path.join(ROOT_DIR, "stage1_eval.log")
        # 构造命令：运行脚本并将输出同时写入 log_file
        # 注意：这里我们让 screen 把输出吐到 log 文件里
        cmd = f"bash run_evaluation.sh > {log_file} 2>&1"
        screen_name = "stage1_eval"
        
        run_in_screen(cmd, screen_name)
        
        st.balloons()
        st.markdown(f"**任务已启动！** 查看日志文件: `{log_file}`")

    # --- 日志查看器 (Stage 1) ---
    with st.expander("查看 Stage 1 实时日志 (Tail 300)"):
        log_file_s1 = os.path.join(ROOT_DIR, "stage1_eval.log")
        if st.button("刷新日志 (Stage 1)"):
            if os.path.exists(log_file_s1):
                try:
                    # 使用 tail -n 300
                    tail_output = subprocess.check_output(f"tail -n 300 {log_file_s1}", shell=True).decode("utf-8", errors='ignore')
                    st.code(tail_output)
                except Exception as e:
                    st.error(f"读取日志失败: {e}")
            else:
                st.warning("日志文件尚未生成")

    st.divider()

    # ================= Stage 2: SCA Evaluation =================
    st.subheader("Stage 2: 语义一致性 (SCA)")
    st.info("💡 运行 `evaluate_sca.py`。需要 Stage 1 生成的 pkl 文件。")
    
    # 1. 自动扫描 PKL 文件
    # 路径规则：results/mld/{EXP_NAME}
    # 注意：这里的 EXP_NAME 应该是 Stage 1 中设置的 target_exp_name
    # 我们不仅扫描 Eval 的，也扫描原版的，方便灵活选择
    
    results_root = os.path.join(ROOT_DIR, "results", "mld")
    
    # 获取 results 下的所有子文件夹
    if os.path.exists(results_root):
        res_dirs = sorted([d for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, d))], reverse=True)
    else:
        res_dirs = []
        
    c1, c2 = st.columns([1, 2])
    with c1:
        target_res_dir = st.selectbox("选择生成结果文件夹", res_dirs, key="sca_dir_select")
    
    pkl_files = []
    if target_res_dir:
        full_res_path = os.path.join(results_root, target_res_dir)
        # 查找 crafmd 开头的 pkl
        pkl_files = glob.glob(os.path.join(full_res_path, "crafmd*.pkl"))
        pkl_files = [os.path.basename(p) for p in pkl_files]
        
    with c2:
        if pkl_files:
            target_pkl_name = st.selectbox("选择 PKL 文件", pkl_files)
            full_pkl_path = os.path.join(results_root, target_res_dir, target_pkl_name)
        else:
            st.warning("该文件夹下没找到 crafmd*.pkl 文件")
            target_pkl_name = None

    if st.button("🚀 运行 Stage 2 (SCA Eval)", type="primary"):
        if not target_pkl_name:
            st.error("请先选择 pkl 文件")
            st.stop()

        # ================= Step A: 注入 Python 脚本 (Input PKL Path) =================
        sca_script_path = os.path.join(ROOT_DIR, "evaluate_sca.py")
        try:
            with open(sca_script_path, 'r', encoding='utf-8') as f:
                py_content = f.read()
            
            # 正则替换 input_path = "..."
            new_line = f'input_path = "{full_pkl_path}"'
            py_content = re.sub(r'input_path\s*=\s*".*?"', new_line, py_content)
            
            with open(sca_script_path, 'w', encoding='utf-8') as f:
                f.write(py_content)
            
            st.success(f"✅ Python 脚本已修改: Input Path = {target_pkl_name}")
            
        except Exception as e:
            st.error(f"❌ 修改 Python 脚本失败: {e}")
            st.stop()

        # ================= Step B: 注入 Bash 脚本 (Config File) =================
        # 我们需要找到当前实验对应的 yaml 文件
        # 逻辑：使用页面顶部 "0. 基础设置" 中选中的实验 (eval_exp_path)
        
        launcher_yaml = os.path.join(eval_exp_path, "launcher_config.yaml")
        if not os.path.exists(launcher_yaml):
            # 备选：找任何一个 yaml
            yamls = glob.glob(os.path.join(eval_exp_path, "*.yaml"))
            launcher_yaml = yamls[0] if yamls else ""
            
        if not launcher_yaml:
            st.error(f"❌ 在 {eval_exp_name} 中找不到 yaml 配置文件，无法配置 Bash 脚本！")
            st.stop()

        sca_bash_path = os.path.join(ROOT_DIR, "run_evaluation_sca.sh")
        try:
            with open(sca_bash_path, 'r', encoding='utf-8') as f:
                bash_content = f.read()
            
            # 正则替换 CONFIG_FILE="..."
            # 注意：这里我们把 launcher_yaml 的绝对路径填进去
            bash_content = re.sub(r'CONFIG_FILE=".*?"', f'CONFIG_FILE="{launcher_yaml}"', bash_content)
            
            with open(sca_bash_path, 'w', encoding='utf-8') as f:
                f.write(bash_content)
                
            st.success(f"✅ Bash 脚本已修改: CONFIG_FILE = {os.path.basename(launcher_yaml)}")
            
        except Exception as e:
            st.error(f"❌ 修改 Bash 脚本失败: {e}")
            st.stop()
            
        # ================= Step C: 运行 Bash 脚本 =================
        log_file_s2 = os.path.join(ROOT_DIR, "stage2_sca.log")
        
        # 运行 run_evaluation_sca.sh 而不是直接 python
        cmd = f"bash run_evaluation_sca.sh > {log_file_s2} 2>&1"
        screen_name = "stage2_sca"
        
        run_in_screen(cmd, screen_name)
        
        st.balloons()
        st.markdown(f"""
        ### 🚀 SCA 评估任务已启动！
        - **Target PKL**: `{target_pkl_name}`
        - **Config YAML**: `{os.path.basename(launcher_yaml)}`
        
        **查看日志:** `{log_file_s2}` (请点击下方刷新按钮)
        """)

    with st.expander("查看 Stage 2 结果日志 (Tail 300)"):
        log_file_s2 = os.path.join(ROOT_DIR, "stage2_sca.log")
        if st.button("刷新日志 (Stage 2)"):
            if os.path.exists(log_file_s2):
                try:
                    tail_output = subprocess.check_output(f"tail -n 300 {log_file_s2}", shell=True).decode("utf-8", errors='ignore')
                    st.code(tail_output)
                except Exception as e:
                    st.error(f"读取日志失败: {e}")
            else:
                st.warning("日志文件尚未生成")

elif mode == "看看你的：渲染 (Render)":
    st.header("🎬 智能渲染工厂")

    # --- 辅助函数：去除 Shell 日志颜色代码 ---
    def clean_ansi_codes(text):
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    # ================= 布局：左侧配置，右侧预览 =================
    # 使用 2:1 的比例
    col_config, col_preview = st.columns([2, 1])

    target_subdir_path = None # 初始化变量

    with col_config:
        st.subheader("1. 选择输入数据 (Input NPY)")
        
        # 1.1 第一级：选择实验文件夹
        results_root = os.path.join(ROOT_DIR, "results", "mld")
        if os.path.exists(results_root):
            res_dirs = sorted([d for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, d))], reverse=True)
        else:
            res_dirs = []
            
        if not res_dirs:
            st.warning("⚠️ 没有生成结果")
            st.stop()

        selected_exp_name = st.selectbox("Step A: 选择实验大类", res_dirs)
        selected_exp_path = os.path.join(results_root, selected_exp_name)

        # 1.2 第二级：选择具体的子文件夹 (style_transfer...)
        if os.path.exists(selected_exp_path):
            # 扫描所有子文件夹，按修改时间排序
            subdirs = [d for d in os.listdir(selected_exp_path) if os.path.isdir(os.path.join(selected_exp_path, d))]
            subdirs = sorted(subdirs, key=lambda x: os.path.getmtime(os.path.join(selected_exp_path, x)), reverse=True)
        else:
            subdirs = []

        if subdirs:
            selected_subdir_name = st.selectbox("Step B: 选择具体动作序列 (Sub-folder)", subdirs)
            target_subdir_path = os.path.join(selected_exp_path, selected_subdir_name)
            st.success(f"📂 目标路径已锁定: `.../{selected_subdir_name}`")
        else:
            st.error(f"❌ 在 {selected_exp_name} 下没找到子文件夹！")
            st.stop()

        # --- 2. 渲染参数设置 ---
        st.subheader("2. 渲染参数配置")
        
        # 嵌套列布局
        c_p1, c_p2 = st.columns(2)
        
        with c_p1:
            render_mode = st.selectbox("Render Mode (模式)", ["sequence", "video", "frame"])
            smplify_iters = st.number_input("SMPL Iters", value=50)
            
        with c_p2:
            if render_mode == "sequence":
                param_val = st.number_input("Frames Number", value=4, min_value=1)
                param_arg = f"--num {int(param_val)}"
            elif render_mode == "video":
                param_val = st.number_input("FPS", value=20)
                param_arg = f"--fps {int(param_val)}"
            elif render_mode == "frame":
                param_val = st.slider("Exact Frame", 0.0, 1.0, 0.5)
                param_arg = f"--exact_frame {param_val}"
        
        res_quality = st.selectbox("Resolution", ["high", "low"], index=0)
        is_gt = st.checkbox("Is Ground Truth? (Green)", value=False)
        
        # --- 3. 执行按钮 ---
        st.divider()
        render_script_dir = "/root/autodl-tmp/MyRepository/MCM-LDM"
        bash_script_path = "render_result.sh"

        if st.button("🎨 开始渲染 (Run Pipeline)", type="primary"):
            # 构造命令
            cmd_args = f"--input_folder '{target_subdir_path}'"
            cmd_args += f" --iters {smplify_iters}"
            cmd_args += f" --mode {render_mode}"
            cmd_args += f" --res {res_quality}"
            cmd_args += f" {param_arg}" # 加上上面的动态参数
            
            if is_gt:
                cmd_args += " --gt"
                
            full_cmd = f"cd {render_script_dir} && bash {bash_script_path} {cmd_args}"
            
            screen_name = f"render_{selected_subdir_name[:10]}"
            log_file = os.path.join(ROOT_DIR, "render_pipeline.log")
            
            # 运行并记录日志
            final_cmd = f"{full_cmd} > {log_file} 2>&1"
            run_in_screen(final_cmd, screen_name)
            
            st.balloons()
            st.markdown(f"**任务已启动！** Session: `{screen_name}`")

    # ================= 右侧预览区 =================
    with col_preview:
        st.subheader("📺 视频预览")
        st.caption("检测选中文件夹下的 MP4 文件...")
        
        if target_subdir_path and os.path.exists(target_subdir_path):
            # 查找 mp4 文件
            mp4_files = glob.glob(os.path.join(target_subdir_path, "*.mp4"))
            
            if mp4_files:
                st.success(f"发现 {len(mp4_files)} 个视频")
                # 只显示前 3 个，避免页面太卡
                for mp4 in mp4_files[:3]:
                    st.text(os.path.basename(mp4))
                    st.video(mp4)
                if len(mp4_files) > 3:
                    st.info(f"还有 {len(mp4_files)-3} 个视频未显示...")
            else:
                st.warning("当前文件夹下没有 MP4 视频。")
                st.caption("(可能是尚未渲染，或者之前的步骤只生成了 npy)")
        else:
            st.info("请先在左侧选择文件夹")

    # ================= 下方日志区 =================
    st.divider()
    with st.expander("查看渲染日志 (已过滤乱码, Tail 300)"):
        log_file = os.path.join(ROOT_DIR, "render_pipeline.log")
        if st.button("刷新渲染日志"):
            if os.path.exists(log_file):
                try:
                    # 读取 raw content
                    raw_content = subprocess.check_output(f"tail -n 300 {log_file}", shell=True).decode("utf-8", errors='ignore')
                    # 清洗 ANSI 颜色代码
                    clean_content = clean_ansi_codes(raw_content)
                    st.code(clean_content)
                except Exception as e:
                    st.error(f"读取日志失败: {e}")
            else:
                st.warning("日志文件尚未生成")

elif mode ==  "我要玩原神！":
    st.header("☕ 赛博休息室 (Cyber Lounge)")
    st.markdown("炼丹太累了？模型还在跑？不如... **启动！**")
    
    st.info("💡 提示：由于云游戏通常禁止 Iframe 嵌入，如果下方窗口无法加载，请点击【🚀 极速启动】跳转游玩。")

    # 定义游戏列表
    games = [
        {
            "name": "云·原神 (Genshin Cloud)",
            "img": "https://upload-os-bbs.hoyolab.com/upload/2025/10/09/189410651/2bc99939f22736a99711676d19e8481d_7407059174228803606.png", # 官网图
            "url": "https://ys.mihoyo.com/cloud/#/",
            "desc": "异世相遇，尽享美味。网页版直接玩，无需下载。"
        },
        {
            "name": "云·星穹铁道 (SR Cloud)",
            "img": "https://upload-os-bbs.hoyolab.com/upload/2023/04/30/32436303/08665ec33163976368a0d4197d54882e_3608120779189690477.png",
            "url": "https://sr.mihoyo.com/cloud/",
            "desc": "银河列车，即刻出发。但这回合还没结束！"
        },
        {
            "name": "绝区零 (ZZZ)",
            "img": "https://n.sinaimg.cn/spider20241206/650/w1440h810/20241206/b6fa-fcb1051e317effb2a1f9b63e72b4a0dd.jpg",
            "url": "https://zzz.mihoyo.com/", 
            "desc": "好久不见，绳匠。虽然网页云端可能还没全开，先去官网看看？"
        }
    ]

    # 布局：三列展示
    cols = st.columns(3)
    
    for i, game in enumerate(games):
        with cols[i]:
            st.subheader(game["name"])
            # 显示封面图 (如果加载失败也没事，只是装饰)
            try:
                st.image(game["img"], use_container_width=True)
            except:
                st.warning("图片加载失败")
                
            st.caption(game["desc"])
            
            # 跳转按钮 (最稳的方案)
            st.link_button(f"🚀 启动 {game['name']}", game["url"], type="primary")

    st.divider()

    # 尝试嵌入 (For Fun)
    st.subheader("🖥️ 尝试原地嵌入 (Experimental)")
    target_game = st.selectbox("选择要尝试嵌入的游戏", [g["name"] for g in games])
    target_url = next(g["url"] for g in games if g["name"] == target_game)
    
    if st.checkbox("尝试强制加载 (可能会白屏/拒绝连接)", value=False):
        st.markdown(f'<iframe src="{target_url}" width="100%" height="800px" style="border:none;"></iframe>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="padding: 20px; border: 2px dashed #444; border-radius: 10px; text-align: center; color: #666;">
            嵌入模式已关闭 (通常云游戏会拦截嵌入请求)<br>建议使用上方的跳转按钮
        </div>
        """, unsafe_allow_html=True)

    # 甚至可以加一个白噪音/音乐播放器（如果你有mp3链接的话）
    st.divider()
    st.caption("🎵 休息一下，为了更好的炼丹。")

elif mode == "沪爷工具箱":
    st.header("🧰 智能Demo生成工坊 (The Factory)")
    
    st.markdown("""
    这里集成了 **排列组合生成**、**标量插值 (Scalar Interpolation)** 和 **视频拼贴 (Montage)** 功能。
    用于一键产出 Demo 视频或 PPT 素材。
    """)
    
    tab1, tab2 = st.tabs(["🧩 批量排列组合 (Matrix Gen)", "🎞️ 视频拼贴 (Montage)"])
    
    with tab1:
        st.subheader("1. 配置生成矩阵")
        
        # A. 选择 Checkpoint
        exp_root = os.path.join(ROOT_DIR, "experiments", "mld")
        if os.path.exists(exp_root):
            exps = sorted(os.listdir(exp_root), key=lambda x: os.path.getmtime(os.path.join(exp_root, x)), reverse=True)
            sel_exp = st.selectbox("选择实验权重", exps, key="tb_exp")
            ckpt_dir = os.path.join(exp_root, sel_exp, "checkpoints")
            if os.path.exists(ckpt_dir):
                ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
                sel_ckpt = st.selectbox("选择 CKPT", ckpts, format_func=os.path.basename, key="tb_ckpt")
            else:
                sel_ckpt = None
        
        st.divider()
        
        col_c, col_s, col_p = st.columns(3)
        
        # B. 选择 Content & Style
        demo_root = os.path.join(ROOT_DIR, "demo")
        subdirs = [d for d in os.listdir(demo_root) if os.path.isdir(os.path.join(demo_root, d))]
        
        with col_c:
            c_dir = st.selectbox("Content 来源目录", subdirs, index=0, key="tb_cdir")
            c_path = os.path.join(demo_root, c_dir)
            c_files = [f for f in os.listdir(c_path) if f.endswith('.npy')]
            sel_contents = st.multiselect("选择 Content (多选)", c_files, default=c_files[:1], key="tb_c")
            
        with col_s:
            s_dir = st.selectbox("Style 来源目录", subdirs, index=min(1, len(subdirs)-1), key="tb_sdir")
            s_path = os.path.join(demo_root, s_dir)
            s_files = [f for f in os.listdir(s_path) if f.endswith('.npy')]
            sel_styles = st.multiselect("选择 Style (多选)", s_files, default=s_files[:1], key="tb_s")
        
        render_mp4 = st.checkbox("顺便生成 MP4 视频 (会变慢)", value=True, key="tb_render_mp4")

        # C. 选择 Scene
        SCENE_KEYS = [
            "Dumuqiao", "DiAiTongDao", "ShuiKengDiMian", "BoLiFangJian", "T_Stage", 
            "CroudedPlace", "DiAiTianhuaban", "Bar", "WalkInSnowOrSand", "Dark", 
            "LeanLeft", "WetFloor", "BaoFengYu", "IcyRoad"
        ]
        with col_p:
            sel_scenes = st.multiselect("选择 Scene (多选)", SCENE_KEYS, default=["Dumuqiao"], key="tb_sc")
        
        # D. Scalar 设置
        st.divider()
        st.write("🔧 Scene Scalar 设置 (插值实验)")
        do_interpolation = st.checkbox("开启 Scalar 插值 (0.0 -> 3.0)", value=False)
        if do_interpolation:
            scalar_list_str = st.text_input("Scalar 列表 (逗号分隔)", "0.0, 0.5, 1.0, 1.5, 2.0, 3.0")
            scalars = [float(x.strip()) for x in scalar_list_str.split(',')]
        else:
            # 默认只跑 3.0 看效果
            scalars = [3.0]
            
        # E. 生成按钮
        if st.button("🚀 开始批量生成 (Batch Generate)", type="primary"):
            if not (sel_contents and sel_styles and sel_scenes and sel_ckpt):
                st.error("请完善选择！")
                st.stop()
                
            # 1. 自动定位该 Checkpoint 对应的 launcher_config.yaml
            # 这是一个关键修复：必须用训练时的配置，否则模型结构对不上，Scene模块可能会失效
            ckpt_dir_abs = os.path.dirname(sel_ckpt)       # .../checkpoints
            exp_dir_abs = os.path.dirname(ckpt_dir_abs)    # .../ExperimentName
            potential_yaml = os.path.join(exp_dir_abs, "launcher_config.yaml")
            
            if os.path.exists(potential_yaml):
                launcher_yaml = potential_yaml
                st.success(f"✅ 成功锁定实验配置: `{os.path.basename(launcher_yaml)}`")
            else:
                # 兜底
                launcher_yaml = os.path.join(ROOT_DIR, "configs", "scenemodiff_train_LiandanBase.yaml")
                st.warning(f"⚠️ 未找到专属配置，使用兜底配置: `{os.path.basename(launcher_yaml)}` (可能导致效果不佳!)")

            # 2. 生成 Task JSON
            task_config = {
                "checkpoint": sel_ckpt,
                "content_dir": c_path,
                "style_dir": s_path,
                "contents": sel_contents,
                "styles": sel_styles,
                "scenes": sel_scenes,
                "render_mp4": render_mp4,
                "scalars": scalars,
                "output_dir": os.path.join(ROOT_DIR, "results/mld", f"BatchGen_{datetime.datetime.now().strftime('%m%d_%H%M')}")
            }
            
            task_json_path = os.path.join(ROOT_DIR, "batch_task.json")
            with open(task_json_path, 'w') as f:
                json.dump(task_config, f, indent=4)
                
            # 3. 构造命令 (关键：日志重定向)
            batch_script = os.path.join(ROOT_DIR, "batch_gen.py")
            log_file = os.path.join(ROOT_DIR, "batch_gen.log") # 定义日志文件
            
            # 使用 > log_file 2>&1 把所有输出（包括print）写入文件
            # 加上 --nodebug 关闭 tqdm 的动态刷新
            python_cmd = f"python {batch_script} --task_json {task_json_path} --cfg {launcher_yaml} --cfg_assets {ASSETS_FILE}"
            final_cmd = f"{python_cmd} > {log_file} 2>&1"

            session_id = f"batch_gen_{datetime.datetime.now().strftime('%H%M%S')}"
            
            # 4. 运行
            run_in_screen(final_cmd, session_id)
            
            st.success("🚀 任务已启动！")
            st.info(f"读取权重: {sel_ckpt}") # 修复了之前的 st.info 语法错误
            st.info(f"结果输出: `{task_config['output_dir']}`")
            st.warning("⚠️ 请展开下方日志查看器，确认 Scalar 是否正确注入！")

    # --- 日志查看器 (新增) ---
    st.divider()
    with st.expander("🔍 实时日志查看器 (不再乱码)", expanded=True):
        log_file = os.path.join(ROOT_DIR, "batch_gen.log")
        col_l1, col_l2 = st.columns([1, 5])
        with col_l1:
            if st.button("🔄 刷新日志"):
                pass
        with col_l2:
            st.caption(f"正在读取: {log_file}")
            
        if os.path.exists(log_file):
            try:
                # 读取最后 100 行
                lines = subprocess.check_output(f"tail -n 100 {log_file}", shell=True).decode("utf-8", errors='ignore')
                st.code(lines, language="text")
            except Exception as e:
                st.error(f"日志读取失败: {e}")
        else:
            st.info("日志文件尚未生成，请点击开始生成...")

    # --- Tab 2: 视频拼贴 ---
    with tab2:
        st.subheader("2. 视频/图片拼贴 (Montage)")
        st.caption("将多个渲染好的 MP4/PNG 拼成网格，用于 PPT 展示。")
        
        # === 文件夹选择逻辑 (二级联动) ===
        montage_root = os.path.join(ROOT_DIR, "results", "mld")
        
        # 1. 第一级：选择实验文件夹
        if os.path.exists(montage_root):
            exp_dirs = [d for d in os.listdir(montage_root) if os.path.isdir(os.path.join(montage_root, d))]
            exp_dirs = sorted(exp_dirs, key=lambda x: os.path.getmtime(os.path.join(montage_root, x)), reverse=True)
        else:
            exp_dirs = []
            
        if not exp_dirs:
            st.warning("⚠️ 没找到实验结果文件夹")
            st.stop()
            
        target_exp_name = st.selectbox("Step A: 选择实验文件夹", exp_dirs, key="montage_exp")
        target_exp_path = os.path.join(montage_root, target_exp_name)
        
        # 2. 第二级：选择子文件夹 (通常是 _pkl 结尾的)
        if os.path.exists(target_exp_path):
            subdirs = [d for d in os.listdir(target_exp_path) if os.path.isdir(os.path.join(target_exp_path, d))]
            subdirs = sorted(subdirs, key=lambda x: os.path.getmtime(os.path.join(target_exp_path, x)), reverse=True)
            
            # 加上“当前目录”选项，防止视频直接在根目录下
            ROOT_OPTION = "Current Directory (.)"
            subdir_options = [ROOT_OPTION] + subdirs
            
            target_subdir_name = st.selectbox("Step B: 选择素材子文件夹", subdir_options, key="montage_subdir")
            
            if target_subdir_name == ROOT_OPTION:
                target_dir = target_exp_path
            else:
                target_dir = os.path.join(target_exp_path, target_subdir_name)
                
            st.info(f"📂 素材读取路径: `{target_dir}`")
        else:
            st.error("路径不存在")
            st.stop()
        
        if os.path.exists(target_dir):
            files = sorted(glob.glob(os.path.join(target_dir, "*.mp4")) + glob.glob(os.path.join(target_dir, "*.png")))
            if files:
                st.write(f"找到 {len(files)} 个素材文件")
                
                # 选择要拼接的文件
                selected_files = st.multiselect("选择要拼接的文件 (按顺序)", files, default=files[:4], format_func=os.path.basename)
                
                if selected_files:
                    c1, c2 = st.columns(2)
                    with c1:
                        grid_cols = st.number_input("网格列数", 2, 5, 2)
                        padding = st.number_input("页边距 (px)", 0, 100, 10)
                    with c2:
                        title_text = st.text_input("大标题 (可选)", "SceneMoDiff Demo")
                        draw_labels = st.checkbox("自动标注文件名", True)
                    
                    if st.button("🎬 开始拼接 (Compose)", type="primary"):
                        # 生成拼接配置
                        compose_config = {
                            "files": selected_files,
                            "grid_cols": grid_cols,
                            "padding": padding,
                            "title": title_text,
                            "draw_labels": draw_labels,
                            "output_path": os.path.join(target_dir, "Montage_Result.mp4")
                        }
                        
                        compose_json_path = os.path.join(ROOT_DIR, "compose_task.json")
                        with open(compose_json_path, 'w') as f:
                            json.dump(compose_config, f, indent=4)
                            
                        # 调用拼接脚本 (需要安装 moviepy: pip install moviepy)
                        compose_script = os.path.join(ROOT_DIR, "compose_video.py")
                        cmd = f"python {compose_script} --task_json {compose_json_path}"
                        
                        run_in_screen(cmd, "montage_task")
                        st.success("拼接任务已启动！")
            else:
                st.warning("文件夹为空")

# ================= 侧边栏监控 =================
st.sidebar.divider()
st.sidebar.subheader("运行中的任务 (Screen)")
if st.sidebar.button("刷新列表"):
    try:
        result = subprocess.check_output("screen -ls", shell=True).decode("utf-8")
        st.sidebar.code(result)
    except:
        st.sidebar.warning("无后台任务")