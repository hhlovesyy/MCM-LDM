import os
import pickle
import sys
import math
import json
from argparse import ArgumentParser

try:
    import bpy
    sys.path.append(os.path.dirname(bpy.data.filepath))
except ImportError:
    raise ImportError(
        "Blender is not properly installed or not launch properly. See README.md to have instruction on how to install and use blender.")

import mld.launch.blender  # noqa
from mld.render.blender import render

# ==============================================================
# [NEW] 新增：读取 JSON 并绘制障碍物和轨迹点的函数
# ==============================================================
def create_material(name, color_rgba):
    """创建一个简单的材质"""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = False # 简单模式
        mat.diffuse_color = color_rgba # RGBA
    return mat

def load_and_draw_scene(pkl_path, trajectory):
    """
    尝试寻找同名的 _scene.json 并画出障碍物
    """
    # print("shape!!!!", trajectory.shape) # [161,3]
    # 1. 推断 JSON 路径
    # 假设 pkl 是 ".../name_mesh.pkl"，我们要找 ".../name_scene.json"
    # 或者 ".../name.pkl" -> ".../name_scene.json"
    json_path = "/root/autodl-tmp/MyRepository/MCM-LDM/task_config.json"
    if not os.path.exists(json_path):
        print(f"[Blender] No scene json found at {json_path}")
        return

    print(f"[Blender] Loading scene from {json_path}")
    try:
        with open(json_path, 'r') as f:
            scene_data = json.load(f)
    except Exception as e:
        print(f"[Error] Failed to load json: {e}")
        return

    # 材质
    mat_obs = create_material("ObstacleMat", (0.8, 0.1, 0.1, 1.0)) 
    mat_way = create_material("WaypointMat", (0.1, 0.8, 0.1, 1.0))

    # 1. 绘制障碍物 (Cylinders)
    env = scene_data.get('environment', {})
    obstacles = env.get('obstacles', [])
    # originX, originY, originZ = trajectory[0, 0], trajectory[0, 1], trajectory[0, 2] # 是Y轴向上的
    # print("originXYZ", originX, originY, originZ)

    
    for obs in obstacles:
        if obs['type'] == 'cylinder':
            cx, cz = obs['center']

            r = obs['radius']
            h = obs['height']
            h = 0.3
            
            # 【修正1】坐标映射
            # 数据 (x, z) -> Blender地面 (x, y)
            # 数据高度 h -> Blender高度 z
            # Blender 圆柱体中心点位置 = (cx, -cz, h/2) 
            # 注意：SMPL数据通常是 Y-up, Z-forward。转到 Blender Z-up 时，
            # Z 轴通常映射为 -Y (负Y) 轴，这样画面方向才一致。
            # 如果你发现位置前后反了，把下面的 -cz 改成 cz 即可。
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=64, 
                radius=r, 
                depth=h, 
                location=(cz, cx, (h)/2.0) 
            )
            
            cyl = bpy.context.object
            cyl.name = f"Obs_{obs.get('id', 'unk')}"
            
            # 【修正2】删除旋转代码
            # Blender 默认圆柱就是沿 Z 轴竖立的，不需要 rotation_euler = (pi/2, ...)
            # cyl.rotation_euler = (0, 0, 0) # 保持默认即可
            
            if cyl.data.materials:
                cyl.data.materials[0] = mat_obs
            else:
                cyl.data.materials.append(mat_obs)

    # 2. 绘制指引点 (Waypoints)
    traj = scene_data.get('trajectory', {})
    waypoints = traj.get('points', [])
    
    for i, pt in enumerate(waypoints):
        wx, wz = pt
        # 【修正3】坐标映射
        # 原来是 (wx, 0.1, wz) -> 导致 wz 被当成了高度，所以飞出去了
        # 现在改成 (wx, -wz, 0.1) -> 高度固定为 0.1
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.1,
            location=(wz, wx, 0.1)  # 
        )
        sphere = bpy.context.object
        sphere.name = f"Waypoint_{i}"
        
        if sphere.data.materials:
            sphere.data.materials[0] = mat_way
        else:
            sphere.data.materials.append(mat_way)
            
    print("[Blender] Scene objects added (Z-up corrected).")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--pkl", type=str, default=None, help="pkl motion file")
    parser.add_argument("--dir", type=str, default=None, help="pkl motion folder")
    parser.add_argument("--mode", type=str, default="sequence", help="render target: video, sequence, frame")
    parser.add_argument("--res", type=str, default="high")
    parser.add_argument("--denoising", type=bool, default=True)
    parser.add_argument("--oldrender", type=bool, default=True)
    parser.add_argument("--accelerator", type=str, default='gpu', help='accelerator device')
    parser.add_argument("--device", type=int, nargs='+', default=[0], help='gpu ids')
    parser.add_argument("--faces_path", type=str, default='./deps/smpl_models/smplh/smplh.faces')
    parser.add_argument("--always_on_floor", action="store_true", help='put all the body on the floor (not recommended)')
    parser.add_argument("--gt", type=str, default=False, help='green for gt, otherwise orange')
    parser.add_argument("--fps", type=int, default=20, help="the frame rate of the rendered video")
    parser.add_argument("--num", type=int, default=8, help="the number of frames rendered in 'sequence' mode")
    parser.add_argument("--exact_frame", type=float, default=0.5, help="the frame id selected under 'frame' mode ([0, 1])")
    cfg = parser.parse_args()
    return cfg

def clean_scene():
    """
    【修复版】只删除场景中的物体，保留材质数据，防止 ReferenceError。
    """
    # 1. 确保在 Object 模式
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # 2. 取消所有选择
    bpy.ops.object.select_all(action='DESELECT')

    # 3. 选择除了摄像机和灯光以外的所有物体
    # (保留摄像机和灯光是为了保持光照一致，不需要每次重建)
    for obj in bpy.data.objects:
        if obj.type not in ['CAMERA', 'LIGHT']:
            obj.select_set(True)

    # 4. 删除选中的物体
    bpy.ops.object.delete()
    
    # 【重点】绝对不要在这里调用 orphans_purge() !!!
    # 它会把你的材质删掉，导致下一次循环找不到材质报错。

def render_cli() -> None:
    cfg = parse_args()

    if cfg.pkl:
        paths = [cfg.pkl]
    elif cfg.dir:
        paths = []
        file_list = os.listdir(cfg.dir)
        for item in file_list:
            if item.endswith("_mesh.pkl"):
                paths.append(os.path.join(cfg.dir, item))
    else:
        raise ValueError(f'{cfg.pkl} and {cfg.dir} are both None!')

    for path in paths:
        clean_scene()
    
        try:
            with open(path, 'rb') as f:
                pkl = pickle.load(f)
                data = pkl['vertices']
                # trajectory = pkl['hint']
                # print("trajectory", trajectory)
                # if trajectory == None:
                #     trajectory = pkl['ground_trajectory']
                trajectory = pkl['ground_trajectory']
                # trajectory = None

        except FileNotFoundError:
            print(f"{path} not found")
            continue

        load_and_draw_scene(path, trajectory)
        render(
            data,
            trajectory,
            path,
            exact_frame=cfg.exact_frame,
            num=cfg.num,
            mode=cfg.mode,
            faces_path=cfg.faces_path,
            always_on_floor=cfg.always_on_floor,
            oldrender=cfg.oldrender,
            res=cfg.res,
            gt=cfg.gt,
            accelerator=cfg.accelerator,
            device=cfg.device,
            fps=cfg.fps)


if __name__ == "__main__":
    render_cli()
