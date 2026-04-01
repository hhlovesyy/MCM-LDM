from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import subprocess
import uuid
import time
import os

app = FastAPI(title="Motion API")

# ========= 你要改的路径 =========
BASE_DIR = Path("/root/autodl-tmp/MyRepository/MCM-LDM")
RESULTS_ROOT = BASE_DIR / "results" / "mld"
WORK_ROOT = BASE_DIR / "api_jobs"
WORK_ROOT.mkdir(parents=True, exist_ok=True)

BLENDER_EXE = "blender"  # 如果不是这个命令，就改成绝对路径
BLENDER_SCRIPT = BASE_DIR / "auto_fbx.py"
TPOSE_PATH = BASE_DIR / "smplh_tpose_joints_22.npy"

# ========= 简单内存任务表（今天够用） =========
JOBS = {}

class SubmitRequest(BaseModel):
    exp_name: str
    motion_index: int = 0

def init_job(job_id: str, exp_name: str, motion_index: int):
    JOBS[job_id] = {
        "status": "queued",
        "stage": "accepted",
        "message": "job accepted",
        "exp_name": exp_name,
        "motion_index": motion_index,
        "fbx_path": None,
        "file_name": None,
        "created_at": time.time(),
        "elapsed_seconds": 0.0,
        "error_message": "",
        "error_detail": ""
    }

def update_job(job_id: str, *, status=None, stage=None, message=None, file_name=None, fbx_path=None, error_message=None, error_detail=None):
    job = JOBS[job_id]

    if status is not None:
        job["status"] = status
    if stage is not None:
        job["stage"] = stage
    if message is not None:
        job["message"] = message
    if file_name is not None:
        job["file_name"] = file_name
    if fbx_path is not None:
        job["fbx_path"] = fbx_path
    if error_message is not None:
        job["error_message"] = error_message
    if error_detail is not None:
        job["error_detail"] = error_detail

    job["elapsed_seconds"] = time.time() - job["created_at"]

def find_latest_style_transfer_dir(exp_name: str) -> Path:
    exp_root = RESULTS_ROOT / exp_name
    if not exp_root.exists():
        raise FileNotFoundError(f"exp root not found: {exp_root}")

    candidates = [p for p in exp_root.iterdir() if p.is_dir() and p.name.startswith("style_transfer")]
    if not candidates:
        raise FileNotFoundError(f"no style_transfer dir found under {exp_root}")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def find_valid_npy_by_index(result_dir: Path, motion_index: int) -> Path:
    npys = [p for p in result_dir.glob("*.npy") if not p.name.endswith("_givenTraj.npy")]
    npys.sort()

    if not npys:
        raise FileNotFoundError(f"no valid npy found in {result_dir}")

    if motion_index < 0 or motion_index >= len(npys):
        raise IndexError(
            f"motion_index out of range: {motion_index}, total valid npy count = {len(npys)}"
        )

    return npys[motion_index]

def run_pipeline(job_id: str, exp_name: str, motion_index: int):
    job_dir = WORK_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        update_job(job_id, status="running", stage="model_run", message="running model")

        # 1) 跑你的模型命令
        cmd = f"""
cd {BASE_DIR}
python demo_transfer_with_scene.py \
  --cfg ./configs/important_scenemodiff_rebuttal_stage1_schema.yaml \
  --cfg_assets ./configs/assets.yaml \
  --style_motion_dir demo/style_motion \
  --content_motion_dir demo/content_motion \
  --scale 2.5 \
  --render_video \
  --exp_name {exp_name} \
  --trajectory_path /root/autodl-tmp/MyRepository/MCM-LDM/task_config.json
"""
        subprocess.run(["bash", "-lc", cmd], check=True)

        update_job(job_id, stage="find_result", message="finding latest result dir")

        # 2) 找最新 style_transfer 目录
        latest_result_dir = find_latest_style_transfer_dir(exp_name)

        # 3) 找对应index的动作序列
        input_npy = find_valid_npy_by_index(latest_result_dir, motion_index)
        print(f"[{job_id}] selected npy = {input_npy.name}")

        update_job(job_id, stage="blender_convert", message=f"converting {input_npy.name} to fbx")

        # 4) 调 Blender 无头转 FBX
        out_fbx = job_dir / f"{input_npy.stem}.fbx"

        blender_cmd = [
            BLENDER_EXE,
            "-b",
            "-P", str(BLENDER_SCRIPT),
            "--",
            "--npy", str(input_npy),
            "--tpose", str(TPOSE_PATH),
            "--fbx", str(out_fbx),
        ]
        subprocess.run(blender_cmd, check=True)

        if not out_fbx.exists():
            raise FileNotFoundError(f"fbx not created: {out_fbx}")

        update_job(
            job_id,
            status="done",
            stage="finished",
            message="finished",
            fbx_path=str(out_fbx),
            file_name=out_fbx.name
        )

    except Exception as e:
        import traceback

        update_job(
            job_id,
            status="failed",
            stage="failed",
            message="pipeline failed",
            error_message=str(e),
            error_detail=traceback.format_exc()
        )

@app.post("/submit")
def submit_job(req: SubmitRequest, background_tasks: BackgroundTasks):
    job_id = f"job_{uuid.uuid4().hex[:8]}"

    init_job(job_id, req.exp_name, req.motion_index)

    background_tasks.add_task(run_pipeline, job_id, req.exp_name, req.motion_index)

    return {
        "ok": True,
        "job_id": job_id,
        "status": "queued",
        "stage": "accepted",
        "message": "job accepted",
        "exp_name": req.exp_name,
        "motion_index": req.motion_index
    }

@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    job["elapsed_seconds"] = time.time() - job["created_at"]

    return {
        "ok": job["status"] != "failed",
        "job_id": job_id,
        "status": job["status"],
        "stage": job["stage"],
        "message": job["message"],
        "elapsed_seconds": job["elapsed_seconds"],
        "file_name": job["file_name"],
        "error_message": job["error_message"],
        "error_detail": job["error_detail"]
    }

@app.get("/download/{job_id}")
def download_result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    if job["status"] != "done" or not job["fbx_path"]:
        raise HTTPException(status_code=400, detail="job not finished")

    fbx_path = Path(job["fbx_path"])
    if not fbx_path.exists():
        raise HTTPException(status_code=404, detail="fbx file missing")

    return FileResponse(
        path=str(fbx_path),
        filename=fbx_path.name,
        media_type="application/octet-stream"
    )