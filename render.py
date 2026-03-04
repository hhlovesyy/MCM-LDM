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

# def load_and_draw_scene_ceil(pkl_path, trajectory):
#     # 【新增】智能寻找同名 JSON（比如 walk_mesh.pkl 找 walk_scene.json）
#     # json_path = pkl_path.replace("_mesh.pkl", "_scene.json").replace(".pkl", "_scene.json")
#     # if not os.path.exists(json_path):
#     #     # 如果没有同名的，就退回队友写死的固定路径
#     #     json_path = "/root/autodl-tmp/MyRepository/MCM-LDM/task_config.json"
#     json_path = "/root/autodl-tmp/MyRepository/MCM-LDM/results/mld/AMCMLDMRES/03031724TestCeilP/walk_test_scene.json" #暂时

#     if not os.path.exists(json_path):
#         print(f"[Blender] No scene json found at {json_path}")
#         return {} # <--- 【修改这里】找不到就返回空字典

#     print(f"[Blender] Loading scene from {json_path}")
#     try:
#         with open(json_path, 'r') as f:
#             scene_data = json.load(f)
#     except Exception as e:
#         print(f"[Error] Failed to load json: {e}")
#         return {} # <--- 【修改这里】报错也返回空字典

            
#     print("[Blender] Scene objects added (Z-up corrected).")
#     return scene_data # <--- 【修改这里】在函数最后一行把读到的字典返回出去



"""
这段代码现在的表现：
如果你正在渲染的文件是：
...A_pkl/B.pkl

它会聪明地识别出：它的目标不在当前 _pkl 文件夹里。
它会定位到：...A/。

它会按顺序搜索：
B_scene.json (首选)
B.json (备选)
scene.json (保底)

"""
def load_and_draw_scene_ceil(pkl_path, trajectory=None):  # <--- 【修复关键】加了 trajectory 参数来接住它
    """
    智能读取与原始 npy 同级的定制化 JSON 文件。
    """
    # 1. 获取当前 pkl 所在的文件夹和文件名
    dir_name = os.path.dirname(pkl_path)       
    base_name = os.path.basename(pkl_path)     
    
    # 2. 核心逻辑：回溯到原始的 npy 文件夹
    if dir_name.endswith("_pkl"):
        original_dir = dir_name[:-4]  # 截掉最后的 "_pkl"
    else:
        original_dir = dir_name
        
    # 3. 剥离后缀，提取纯动作名
    name_without_ext = base_name.replace('_mesh.pkl', '').replace('.pkl', '')
    
    # 4. 构造可能存在的 JSON 文件名列表 (按优先级查找)
    json_candidates =[
        os.path.join(original_dir, f"{name_without_ext}_scene.json"), # 首选: 比如 000118_scene.json
        os.path.join(original_dir, f"{name_without_ext}.json"),       # 备选: 比如 000118.json
        os.path.join(original_dir, "scene.json")                      # 保底: 比如 scene.json
    ]
    
    # 5. 遍历寻找真正的文件
    json_path = None
    for candidate in json_candidates:
        if os.path.exists(candidate):
            json_path = candidate
            break
            
    # 6. 如果全都没找到，返回空字典 (平滑降级)
    if not json_path:
        print(f"[Blender - Ceil] No specific json found in {original_dir}, using auto-ceiling.")
        return {}
        
    # 7. 读取并返回数据
    print(f"[Blender - Ceil] Bingo! Loading custom scene from {json_path}")
    try:
        with open(json_path, 'r') as f:
            scene_data = json.load(f)
        return scene_data
    except Exception as e:
        print(f"[Error] Failed to load json {json_path}: {e}")
        return {}
    
    
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
    parser.add_argument("--scene_name", type=str, default="default")
    parser.add_argument("--use_guide_hint", type=str, default=False, help="render user-define trajectory hint")
    cfg = parser.parse_args()
    return cfg

def clean_scene():
    """
    【修复版】不依赖 select_set 的安全清理函数。
    直接操作数据块，不会因为物体未链接到场景而报错。
    """
    # 1. 确保在 Object 模式
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # 2. 遍历所有物体，直接删除数据
    # 使用 list(...) 是因为我们在遍历过程中会删除元素，必须操作副本
    for obj in list(bpy.data.objects):
        # 保留摄像机和灯光
        if obj.type in ['CAMERA', 'LIGHT']:
            continue
        
        # 【关键修改】直接移除物体，不需要 select_set
        # do_unlink=True 会确保它从所有场景中解绑
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except Exception as e:
            print(f"Warning: Failed to remove {obj.name}: {e}")

    # 3. (可选) 清理残留的网格和曲线数据，防止内存泄漏
    # 删除没有用户 (users=0) 的 Mesh 数据
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    
    # 删除没有用户的 Curve 数据
    for curve in list(bpy.data.curves):
        if curve.users == 0:
            bpy.data.curves.remove(curve)


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
                hint = pkl['hint']
                # trajectory = None

        except FileNotFoundError:
            print(f"{path} not found")
            continue

        load_and_draw_scene(path, trajectory)
        scene_data = load_and_draw_scene_ceil(path, trajectory) # 
        
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
            fps=cfg.fps,
            hint=hint,
            cfg=cfg,
            scene_data=scene_data  # <--- 【新增这一行】把数据传给核心渲染器
        )

if __name__ == "__main__":
    render_cli()
