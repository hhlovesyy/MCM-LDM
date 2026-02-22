# app.py
import streamlit as st
import os
import yaml
from datetime import datetime
import time
import queue # 导入 queue 模块

# 从同级目录导入我们的模块
from scheduler import GPUScheduler, TaskRunner
from pipelines import OursFullModelPipeline, MCMLDMBaseline, RenderTrajBlenderPipeline

# --- 初始化 ---
st.set_page_config(layout="wide", page_title="统一推理 Demo 生成器")

if "scheduler" not in st.session_state:
    st.session_state.scheduler = GPUScheduler(num_gpus=3)
    st.session_state.runner = TaskRunner(st.session_state.scheduler)
    st.session_state.tasks = {}
    # ############## 核心修改 ##############
    # 初始化一个线程安全的队列，用于通信
    st.session_state.status_queue = queue.Queue()

# --- 全局变量与配置 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS_CFG_PATH = os.path.join(PROJECT_ROOT, "configs/assets.yaml")

# 注册所有可用的 Baseline Pipeline (根据您的日志，我将名称改为 scenemodiff_full)
PIPELINES = [
    OursFullModelPipeline("scenemodiff_full", "我们的 (全量版本)", "configs/ours_full.yaml"),
    MCMLDMBaseline("MCMLDM_baseline", "MCMLDM的Baseline", "configs/MCMLDM.yaml"),
    MCMLDMBaseline("MCMLDM_basline_noguide", "MCMLDM没有轨迹指引", "configs/MCMLDM_noGuide.yaml"),
    OursFullModelPipeline("scenemodiff_noGuide", "我们的（全量版本但是没有轨迹指引）", "configs/ours_no_guide.yaml"),
    MCMLDMBaseline("scenemodiff_noScene", "我们的（没有场景，只有轨迹解耦，有轨迹引导）", "configs/ours_no_scene.yaml"),
    MCMLDMBaseline("scememodiff_noscene_noGuidance", "我们的（没有场景，没有轨迹引导）", "configs/ours_no_scene_noguide.yaml")
    # RenderTrajBlenderPipeline("render_traj", "把轨迹渲染出来（放入第一个文件夹）", "")
]

# --- UI 界面 (这部分无变化) ---
st.title("🎬 统一推理 Demo 生成器")
st.caption("选择输入，勾选要对比的 Baseline，然后点击生成。")

with st.container(border=True):
    st.subheader("1. 全局输入配置")
    col1, col2 = st.columns(2)
    with col1:
        demo_root = os.path.join(PROJECT_ROOT, "demo")
        demo_dirs = [d for d in os.listdir(demo_root) if os.path.isdir(os.path.join(demo_root, d))]
        content_dir = st.selectbox("选择 Content Motion 文件夹", demo_dirs, index=0)
        style_dir = st.selectbox("选择 Style Motion 文件夹", demo_dirs, index=1)
    with col2:
        trajectory_path = st.text_input("输入轨迹文件路径 (.json)", value="/root/autodl-tmp/MyRepository/MCM-LDM/task_config_baseline.json")
        film_scalar = st.slider("场景注入强度 (FiLM Scalar)", 0.0, 10.0, 3.0, 0.1)
    scene_prompt = st.text_area("输入场景文本 (Scene Prompt)", "A person walks on a narrow bridge.")

with st.container(border=True):
    st.subheader("2. 选择要运行的对比实验")
    selected_pipelines = []
    for p in PIPELINES:
        if st.checkbox(p.label, value=True, key=p.name):
            selected_pipelines.append(p)

st.divider()
col_run, _ = st.columns([1, 4])
if col_run.button("🚀 开始并行生成", type="primary", use_container_width=True):
    if not selected_pipelines:
        st.warning("请至少选择一个实验！")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(PROJECT_ROOT, "demo_results", "demo_pipeline", timestamp)
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        st.session_state.tasks = {}

        for pipe in selected_pipelines:
            st.session_state.tasks[pipe.name] = {'status': 'queued', 'msg': '...等待调度...'}
            try:
                template_path = os.path.join(os.path.dirname(__file__), pipe.config_template)
                with open(template_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                if 'TEST' not in config_data: config_data['TEST'] = {}
                config_data['TEST']['MULTI_MODAL_TEXT_PROMPT'] = scene_prompt
                temp_config_filename = f"temp_cfg_{pipe.name}.yaml"
                temp_config_path = os.path.join(output_dir, temp_config_filename)
                with open(temp_config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_data, f)
            except Exception as e:
                st.session_state.tasks[pipe.name] = {'status': 'error', 'msg': f'创建配置文件失败: {e}'}
                continue

            args = {
                "temp_config_path": temp_config_path,
                "assets_cfg_path": ASSETS_CFG_PATH,
                "content_path": os.path.join("demo", content_dir),
                "style_path": os.path.join("demo", style_dir),
                "trajectory_path": trajectory_path,
                "film_scalar": film_scalar,
                "output_dir": output_dir,
            }
            command = pipe.get_command(args)
            
            # ############## 核心修改 ##############
            # 提交任务时，传入 status_queue
            st.session_state.runner.run_task(
                task_name=pipe.name,
                command=command,
                cwd=PROJECT_ROOT,
                log_dir=log_dir,
                status_queue=st.session_state.status_queue
            )
        
        st.success(f"已成功提交 {len(selected_pipelines)} 个任务到后台队列！")
        time.sleep(1)
        st.rerun()

# --- 任务状态展示面板 ---
st.subheader("📊 任务监控面板")

# ############## 核心修改 ##############
# 在渲染UI前，先从队列中取出所有新消息并更新状态
# 这个循环由主线程执行，所以是线程安全的
while not st.session_state.status_queue.empty():
    task_name, status, msg = st.session_state.status_queue.get()
    if task_name in st.session_state.tasks:
        st.session_state.tasks[task_name] = {'status': status, 'msg': msg}

if not st.session_state.tasks:
    st.info("暂无任务，请配置并点击“开始生成”。")
else:
    num_tasks = len(st.session_state.tasks)
    cols = st.columns(min(num_tasks, 3) or 1)
    task_items = list(st.session_state.tasks.items())
    
    is_running = False
    for i, (name, info) in enumerate(task_items):
        with cols[i % 3]:
            status = info.get('status', 'unknown')
            msg = info.get('msg', 'N/A')
            if status in ["running", "pending", "queued"]:
                is_running = True
            
            color_map = {"success": "green", "error": "red", "running": "blue", "pending": "orange"}
            border_color = color_map.get(status, "#888")

            st.markdown(f"""
            <div style="padding: 10px; border-radius: 5px; border: 2px solid {border_color}; margin-bottom: 10px;">
                <strong>{name}</strong><br>
                <small>{msg}</small>
            </div>
            """, unsafe_allow_html=True)

    if is_running:
        time.sleep(3)
        st.rerun()