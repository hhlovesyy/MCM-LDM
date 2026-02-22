import bpy
import numpy as np
import os
import sys
import argparse
import math
import json

# ==========================================
# 1. 核心处理函数
# ==========================================

def get_gradient_color(t, color_name):
    """手动实现颜色渐变"""
    if color_name == 'Greens':
        c_start, c_end = np.array([0.6, 0.9, 0.6, 1.0]), np.array([0.0, 0.4, 0.0, 1.0])
    elif color_name == 'Reds':
        c_start, c_end = np.array([1.0, 0.6, 0.6, 1.0]), np.array([0.5, 0.0, 0.0, 1.0])
    else:
        c_start, c_end = np.array([0.8, 0.8, 0.8, 1.0]), np.array([0.2, 0.2, 0.2, 1.0])
    return tuple(c_start * (1 - t) + c_end * t)

def create_material(name, color):
    """创建材质"""
    mat = bpy.data.materials.get(name)
    if mat is None: mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    shader = nodes.new(type='ShaderNodeBsdfPrincipled')
    shader.inputs['Base Color'].default_value = color
    shader.inputs['Roughness'].default_value = 1.0
    output = nodes.new(type='ShaderNodeOutputMaterial')
    mat.node_tree.links.new(shader.outputs['BSDF'], output.inputs['Surface'])
    return mat

def prepare_meshes(data):
    """调整坐标轴以适应 Blender (Y-up -> Z-up)"""
    # 假设输入数据是 (T, J, 3) 或 (T, 3)
    # 你的置换: Z->X, X->Y, Y->Z
    data = data[..., [2, 0, 1]]
    # 移除地面偏移
    if data.ndim == 3:
        height_offset = data[..., 2].min()
    elif data.ndim == 2:
        height_offset = data[..., 2].min()
    data[..., 2] -= height_offset
    return data

def show_trajectory(coords, color_map_name, radius, mat_prefix):
    """在 Blender 中生成轨迹球"""
    collection = bpy.data.collections.new(mat_prefix)
    bpy.context.scene.collection.children.link(collection)
    
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=(0, 0, 0))
    base_sphere = bpy.context.active_object
    base_sphere.name = f"{mat_prefix}_Base"
    bpy.context.collection.objects.unlink(base_sphere)
    
    total_points = len(coords)
    begin_t, end_t = 0.3, 1.0
    
    for i, coord in enumerate(coords):
        if np.abs(coord).sum() < 0.001: continue
        x, y, z = coord
        ob = base_sphere.copy()
        ob.data = base_sphere.data.copy()
        ob.location = (x, y, z)
        collection.objects.link(ob)
        
        mapped_t = begin_t + (end_t - begin_t) * (i / total_points)
        rgba_color = get_gradient_color(mapped_t, color_map_name)
        mat = create_material(f"{mat_prefix}_{i}", rgba_color)
        ob.data.materials.append(mat)

def load_and_draw_scene(json_path):
    """
    加载并绘制 task_config.json 中的障碍物和航点
    """
    json_path = "/root/autodl-tmp/MyRepository/MCM-LDM/task_config.json"
    if not os.path.exists(json_path):
        print(f"[Warning] Scene config not found at {json_path}")
        return
        
    print(f"[Info] Loading scene objects from {json_path}")
    with open(json_path, 'r') as f:
        scene_data = json.load(f)

    mat_obs = create_material("ObstacleMat", (0.8, 0.1, 0.1, 1.0)) # Red
    mat_way = create_material("WaypointMat", (0.1, 0.8, 0.1, 1.0)) # Green

    # 绘制障碍物
    obstacles = scene_data.get('environment', {}).get('obstacles', [])
    for obs in obstacles:
        if obs['type'] == 'cylinder':
            cx, cz = obs['center']
            r, h = obs['radius'], obs['height']
            # 坐标转换: (x, z) from data -> (y, x) in Blender
            # 高度 h -> z in Blender
            # Blender 圆柱体中心在 h/2
            bpy.ops.mesh.primitive_cylinder_add(
                radius=r, depth=h, location=(cz, cx, h/2.0)
            )
            cyl = bpy.context.object
            cyl.data.materials.append(mat_obs)
    
    # 绘制航点
    waypoints = scene_data.get('trajectory', {}).get('points', [])
    for pt in waypoints:
        wx, wz = pt
        # 坐标转换: (x, z) -> (y, x)
        # 高度固定在地面上方一点点
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.1, location=(wz, wx, 0.1)
        )
        sphere = bpy.context.object
        sphere.data.materials.append(mat_way)
            
    print("[Info] Scene objects added.")

# ==========================================
# 2. 场景设置与渲染逻辑
# ==========================================

def setup_scene():
    """清理场景"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    world = bpy.context.scene.world
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.8, 0.8, 0.8, 1)
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    bpy.context.active_object.data.energy = 2.0

def setup_top_down_camera(center_target):
    """设置正交顶视相机"""
    bpy.ops.object.camera_add(location=(center_target[0], center_target[1], 20))
    cam = bpy.context.active_object
    bpy.context.scene.camera = cam
    cam.rotation_euler = (0, 0, 0)
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = 12.0 
    return cam

def adjust_camera_fit(camera, coords):
    """自动调整相机视野"""
    if len(coords) == 0: return
    min_x, max_x = coords[:, 0].min(), coords[:, 0].max()
    min_y, max_y = coords[:, 1].min(), coords[:, 1].max()
    camera.location.x = (max_x + min_x) / 2
    camera.location.y = (max_y + min_y) / 2
    camera.data.ortho_scale = max(max_x - min_x, max_y - min_y) * 1.2

# ==========================================
# 3. 主函数
# ==========================================
def main():
    if "--" not in sys.argv:
        print("Usage: blender -b -P script.py -- --input file.npy")
        return
    
    argv = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to input .npy file')
    args = parser.parse_args(argv)

    input_path = args.input
    output_path = input_path.replace(".npy", "_topdown.png")
    
    # 自动推断 JSON 路径
    json_path = os.path.join(os.path.dirname(input_path), "task_config.json")
    
    # 1. 加载数据
    raw_data = np.load(input_path)
    if raw_data.ndim == 3: root_traj = raw_data[:, 0, :]
    elif raw_data.ndim == 2: root_traj = raw_data
    else: print("Error: Unknown data shape"); return

    hint_path = input_path.replace('.npy', '_givenTraj.npy')
    hint_traj = np.load(hint_path) if os.path.exists(hint_path) else None

    # 2. 坐标转换
    proc_root = prepare_meshes(root_traj.copy())
    proc_hint = prepare_meshes(hint_traj.copy()) if hint_traj is not None else None

    # 3. 设置 Blender 场景
    setup_scene()
    
    # 4. 绘制所有元素
    show_trajectory(proc_root, 'Greens', 0.05, "GenTraj")
    if proc_hint is not None:
        proc_hint_offset = proc_hint.copy()
        proc_hint_offset[:, 2] -= 0.02 
        show_trajectory(proc_hint_offset, 'Reds', 0.04, "HintTraj")
    # load_and_draw_scene(json_path)

    # 5. 设置相机
    all_points = np.concatenate([p for p in [proc_root, proc_hint] if p is not None], axis=0)
    cam = setup_top_down_camera((0,0))
    adjust_camera_fit(cam, all_points)

    # 6. 渲染
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES' # 使用 Cycles 避免显示器问题
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = True
    scene.render.filepath = output_path
    
    print(f"Rendering to {output_path}...")
    bpy.ops.render.render(write_still=True)
    print("Done.")

if __name__ == "__main__":
    main()