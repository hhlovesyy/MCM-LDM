# scheduler.py
import os
import time
import shlex
import subprocess
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

class GPUScheduler:
    """
    一个简单的 GPU 资源池管理器。
    它确保在任何时候，每张 GPU 只会分配给一个任务。
    (此文件无变化)
    """
    def __init__(self, num_gpus=3):
        self.gpu_ids = [str(i) for i in range(num_gpus)]
        self.lock = Lock()
        self.status = {gid: True for gid in self.gpu_ids}

    def acquire(self):
        with self.lock:
            for gid, is_free in self.status.items():
                if is_free:
                    self.status[gid] = False
                    return gid
        return None

    def release(self, gid):
        with self.lock:
            if gid in self.status:
                self.status[gid] = True

class TaskRunner:
    """
    任务执行器。
    接收任务，并将其放入线程池中，等待 GPU 资源并执行。
    """
    def __init__(self, scheduler, max_workers=5):
        self.scheduler = scheduler
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    # ############## 核心修改 ##############
    # 不再接收 callback_fn，而是接收一个 status_queue
    def run_task(self, task_name, command, cwd, log_dir, status_queue):
        """提交一个任务到执行队列。"""
        self.executor.submit(self._worker, task_name, command, cwd, log_dir, status_queue)

    def _worker(self, task_name, command, cwd, log_dir, status_queue):
        """这是一个独立的线程，负责完整地执行一个推理任务。"""
        gpu_id = None
        while gpu_id is None:
            gpu_id = self.scheduler.acquire()
            if gpu_id is None:
                # 把状态更新 'put' 进队列，而不是直接调用函数
                status_queue.put((task_name, "pending", "⏳ 等待空闲 GPU..."))
                time.sleep(3)
        
        try:
            # 把状态更新 'put' 进队列
            status_queue.put((task_name, "running", f"🚀 分配至 GPU:{gpu_id}，正在启动..."))
            
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            
            log_file_path = os.path.join(log_dir, f"{task_name}.log")
            
            # 这里现在应该可以被正确执行了
            with open(log_file_path, "w", encoding="utf-8") as f_log:
                process = subprocess.Popen(
                    shlex.split(command),
                    cwd=cwd,
                    env=env,
                    stdout=f_log,
                    stderr=subprocess.STDOUT,
                    encoding='utf-8'
                )
                process.wait()

            if process.returncode == 0:
                # 把状态更新 'put' 进队列
                status_queue.put((task_name, "success", f"✅ 成功！日志: {os.path.basename(log_file_path)}"))
            else:
                status_queue.put((task_name, "error", f"❌ 失败！请检查日志: {os.path.basename(log_file_path)}"))
                
        except Exception as e:
            status_queue.put((task_name, "error", f"💥 严重错误: {e}"))
        finally:
            self.scheduler.release(gpu_id)