import bpy
import numpy as np
import os
import math
import mathutils
import sys
import argparse

# ================= 配置区域 =================
# Blender 运行脚本时，会把自身的系统参数也塞进 sys.argv
# 我们需要截取 "--" 之后的内容，交给 argparse 处理
if "--" in sys.argv:
    argv = sys.argv[sys.argv.index("--") + 1:]
else:
    argv = []

parser = argparse.ArgumentParser(description="Batch Convert NPY to FBX via Blender")
parser.add_argument('-n', '--npy', required=True, help="输入的动作 NPY 文件路径")
parser.add_argument('-t', '--tpose', required=True, help="输入的 T-Pose NPY 文件路径")
parser.add_argument('-f', '--fbx', required=True, help="输出的 FBX 文件路径")

# 解析参数
args = parser.parse_args(argv)

NPY_FILE_PATH = args.npy
TPOSE_FILE_PATH = args.tpose
FBX_EXPORT_PATH = args.fbx

# 你当前这批数据在 Blender 里看起来 20 FPS 更合理
FPS = 20

# 旧链路里 Blender 内部放大 100 倍，UE 导入再缩 0.01
SCALE_FACTOR = 100.0

JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'
]

HIERARCHY = {
    'pelvis': None, 'left_hip': 'pelvis', 'right_hip': 'pelvis', 'spine1': 'pelvis',
    'left_knee': 'left_hip', 'right_knee': 'right_hip', 'spine2': 'spine1',
    'left_ankle': 'left_knee', 'right_ankle': 'right_knee', 'spine3': 'spine2',
    'left_foot': 'left_ankle', 'right_foot': 'right_ankle', 'neck': 'spine3',
    'left_collar': 'spine3', 'right_collar': 'spine3', 'head': 'neck',
    'left_shoulder': 'left_collar', 'right_shoulder': 'right_collar',
    'left_elbow': 'left_shoulder', 'right_elbow': 'right_shoulder',
    'left_wrist': 'left_elbow', 'right_wrist': 'right_elbow'
}

CHAIN_MAP = {
    'pelvis': 'spine1',
    'left_hip': 'left_knee', 'left_knee': 'left_ankle', 'left_ankle': 'left_foot',
    'right_hip': 'right_knee', 'right_knee': 'right_ankle', 'right_ankle': 'right_foot',
    'spine1': 'spine2', 'spine2': 'spine3', 'spine3': 'neck', 'neck': 'head',
    'left_collar': 'left_shoulder', 'left_shoulder': 'left_elbow', 'left_elbow': 'left_wrist',
    'right_collar': 'right_shoulder', 'right_shoulder': 'right_elbow', 'right_elbow': 'right_wrist'
}

def clear_scene():
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type in ['ARMATURE', 'EMPTY', 'MESH']:
            obj.select_set(True)
    bpy.ops.object.delete()

def to_blender_coords(pos):
    return (pos[0] * SCALE_FACTOR, -pos[2] * SCALE_FACTOR, pos[1] * SCALE_FACTOR)

def create_armature_from_npy(anim_data, tpose_data):
    # --- 1. 数据处理 ---
    if tpose_data.ndim == 2: tpose_data = tpose_data[np.newaxis, ...]
    min_y = np.min(tpose_data[0, :, 1])
    tpose_data[0, :, 1] -= min_y
    
    combined_data = np.vstack([tpose_data, anim_data])
    n_frames, n_joints, _ = combined_data.shape
    rest_pose = combined_data[0] 

    # --- 2. 创建骨架 (Visual Fix) ---
    bpy.ops.object.armature_add(enter_editmode=True)
    armature_obj = bpy.context.active_object
    armature_obj.name = "SMPLH_Skeleton_Clean"
    amt = armature_obj.data
    
    for bone in amt.edit_bones: amt.edit_bones.remove(bone)
    edit_bones = {}
    
    for i, name in enumerate(JOINT_NAMES):
        bone = amt.edit_bones.new(name)
        bone.head = to_blender_coords(rest_pose[i])
        
        if name in CHAIN_MAP:
            child_idx = JOINT_NAMES.index(CHAIN_MAP[name])
            bone.tail = to_blender_coords(rest_pose[child_idx])
            
        elif 'foot' in name:
            bone.tail = bone.head + mathutils.Vector((0, -0.2 * SCALE_FACTOR, 0))
            
        elif 'wrist' in name:
            parent_name = HIERARCHY[name]
            parent_head = to_blender_coords(rest_pose[JOINT_NAMES.index(parent_name)])
            direction = (mathutils.Vector(bone.head) - mathutils.Vector(parent_head)).normalized()
            bone.tail = bone.head + direction * (0.15 * SCALE_FACTOR)
            
        elif name == 'head':
            bone.tail = bone.head + mathutils.Vector((0, 0, 0.2 * SCALE_FACTOR))
            
        else:
            bone.tail = (bone.head[0], bone.head[1], bone.head[2] + 5.0)
            
        edit_bones[name] = bone

    for name, parent_name in HIERARCHY.items():
        if parent_name: edit_bones[name].parent = edit_bones[parent_name]

    bpy.ops.armature.calculate_roll(type='GLOBAL_POS_Z')
    bpy.ops.object.mode_set(mode='OBJECT')

    # --- 3. 强制 A-Pose ---
    bpy.ops.object.mode_set(mode='POSE')
    
    pbone_l = armature_obj.pose.bones.get('left_shoulder')
    if pbone_l:
        pbone_l.rotation_mode = 'XYZ'
        pbone_l.rotation_euler.x = -math.radians(45) 
        
    pbone_r = armature_obj.pose.bones.get('right_shoulder')
    if pbone_r:
        pbone_r.rotation_mode = 'XYZ'
        pbone_r.rotation_euler.x = -math.radians(45)

    bpy.ops.pose.armature_apply(selected=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    print("✅ Rest Pose -> A-Pose applied.")

    # --- 4. 驱动与约束 ---
    collection = bpy.data.collections.new("Drivers")
    bpy.context.scene.collection.children.link(collection)
    empties = []
    
    # 【核心修改1】：跳过 combined_data，直接获取纯净动画的帧数
    n_anim_frames = anim_data.shape[0] 
    
    for i, name in enumerate(JOINT_NAMES):
        bpy.ops.object.empty_add(type='PLAIN_AXES', radius=0.05 * SCALE_FACTOR)
        empty = bpy.context.active_object
        empty.name = f"DRV_{name}"
        collection.objects.link(empty)
        try: bpy.context.scene.collection.objects.unlink(empty)
        except: pass
        empties.append(empty)
        
        # 【核心修改2】：仅遍历 anim_data，并且让它的第一帧写在 Blender 的第 1 帧
        for frame in range(n_anim_frames):
            empty.location = to_blender_coords(anim_data[frame, i])
            empty.keyframe_insert(data_path="location", frame=frame + 1)

    # --- 5. 约束设置 ---
    bpy.ops.object.select_all(action='DESELECT')
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    for i, name in enumerate(JOINT_NAMES):
        pbone = armature_obj.pose.bones.get(name)
        if not pbone: continue

        if name == 'pelvis':
            c_loc = pbone.constraints.new('COPY_LOCATION')
            c_loc.target = empties[i]
            
            child_idx = JOINT_NAMES.index('spine1')
            c_stretch = pbone.constraints.new('STRETCH_TO')
            c_stretch.target = empties[child_idx]
            c_stretch.volume = 'NO_VOLUME'
            
            l_hip_idx = JOINT_NAMES.index('left_hip')
            c_track = pbone.constraints.new('LOCKED_TRACK')
            c_track.target = empties[l_hip_idx]
            c_track.track_axis = 'TRACK_X' 
            c_track.lock_axis = 'LOCK_Y'   
            
        elif name in CHAIN_MAP:
            child_name = CHAIN_MAP[name]
            child_idx = JOINT_NAMES.index(child_name)
            
            c_track = pbone.constraints.new('DAMPED_TRACK')
            c_track.target = empties[child_idx]
            c_track.track_axis = 'TRACK_Y'

    # --- 6. 烘焙 ---
    print("Baking Animation...")
    
    # 【救命级修正】：强制把 Blender 的场景起始帧改为 0 ！！！
    # 否则 FBX 导出时会直接把第 0 帧切掉丢弃！
    bpy.context.scene.frame_start = 0 
    bpy.context.scene.frame_end = n_anim_frames

    # 烘焙也从 0 开始，确保整个轨道连续
    bpy.ops.nla.bake(
        frame_start=0, frame_end=n_anim_frames,
        only_selected=False, visual_keying=True,
        clear_constraints=True, use_current_action=True,
        bake_types={'POSE'}
    )
    
    # 烘焙完成后，强行将第 0 帧覆盖为完美的 A-Pose
    bpy.context.scene.frame_set(0)
    bpy.ops.object.mode_set(mode='POSE')
    
    for pbone in armature_obj.pose.bones:
        pbone.matrix_basis = mathutils.Matrix.Identity(4)
        # 强制打上关键帧
        pbone.keyframe_insert(data_path="location", frame=0)
        if pbone.rotation_mode == 'QUATERNION':
            pbone.keyframe_insert(data_path="rotation_quaternion", frame=0)
        else:
            pbone.keyframe_insert(data_path="rotation_euler", frame=0)

    bpy.ops.object.mode_set(mode='OBJECT')
    for e in empties: bpy.data.objects.remove(e)
    bpy.data.collections.remove(collection)

    bpy.context.scene.render.fps = FPS
    
    # 在导出 FBX 之前，刷新视图，确保骨架停留在完美的第 0 帧 A-Pose 上
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()
    
    # ================= 新增：自动化生成占位网格与导出 =================
    
    print("Creating Dummy Mesh and Binding...")
    
    # 1. 创建占位立方体并缩放
    bpy.ops.mesh.primitive_cube_add(size=1.0) # 初始大小
    dummy_mesh = bpy.context.active_object
    dummy_mesh.name = "DummyMesh_UE5"
    dummy_mesh.scale = (0.001, 0.001, 0.001) # 缩小到看不见
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True) # 应用缩放
    
    # 2. 建立父子关系与蒙皮 (代码直接操作，杜绝上下文选择错误)
    dummy_mesh.parent = armature_obj
    
    # 添加骨架修改器
    modifier = dummy_mesh.modifiers.new(type='ARMATURE', name="Armature")
    modifier.object = armature_obj
    
    # 创建所有骨骼对应的空顶点组 (相当于 Ctrl+P -> With Empty Groups)
    for bone in armature_obj.data.bones:
        dummy_mesh.vertex_groups.new(name=bone.name)
        
    # ================= 新增：在导出前强行将时间轴归零 =================
    # 这一步极其关键！确保导出FBX瞬间，骨架处于我们设定的完美 A-Pose
    bpy.context.scene.frame_set(0) 
    bpy.context.view_layer.update() # 刷新视图层，确保矩阵更新
    # ================================================================
        
    # 3. 准备导出 FBX
    print("Exporting to FBX...")
    bpy.ops.object.select_all(action='DESELECT')
    armature_obj.select_set(True)
    dummy_mesh.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj

    # 严格按照你提供的截图设置参数
    bpy.ops.export_scene.fbx(
        filepath=FBX_EXPORT_PATH,
        use_selection=True,                     # Limit to Selected Objects
        global_scale=1.0,                       # Scale: 1.00
        apply_scale_options='FBX_SCALE_ALL',    # Apply Scalings: All Local
        axis_forward='-Z',                      # Forward: -Z Forward
        axis_up='Y',                            # Up: Y Up
        apply_unit_scale=True,                  # Apply Unit
        use_space_transform=True,               # Use Space Transform
        bake_space_transform=False,             # Apply Transform (未勾选)
        
        mesh_smooth_type='OFF',                 # Smoothing: Normals Only (在 API 中 OFF 或 FACE 对应此效果)
        use_mesh_modifiers=True,                # Apply Modifiers
        
        primary_bone_axis='Y',                  # Primary Bone Axis: Y Axis
        secondary_bone_axis='X',                # Secondary Bone Axis: X Axis
        armature_nodetype='NULL',               # Armature FBXNode Type: Null
        use_armature_deform_only=False,         # Only Deform Bones (未勾选)
        add_leaf_bones=False,                   # Add Leaf Bones (红框高亮，强制取消)
        
        bake_anim=True,                         # Bake Animation
        bake_anim_use_all_bones=True,           # Key All Bones (红框高亮，强制开启)
        bake_anim_use_nla_strips=False,         # NLA Strips (红框高亮，强制取消)
        bake_anim_use_all_actions=False,        # All Actions (红框高亮，强制取消)
        bake_anim_force_startend_keying=True    # Force Start/End Keying (红框高亮，强制开启)
    )
    
    print(f"✅ 全自动处理完毕！FBX已导出至: {FBX_EXPORT_PATH}")
    # =================================================================

if __name__ == "__main__":
    try:
        if os.path.exists(NPY_FILE_PATH) and os.path.exists(TPOSE_FILE_PATH):
            clear_scene()
            anim_data = np.load(NPY_FILE_PATH)
            tpose_data = np.load(TPOSE_FILE_PATH)
            create_armature_from_npy(anim_data, tpose_data)
        else:
            print(f"❌ 文件不存在。")
    except Exception as e:
        print(f"❌ 出错: {e}")
        import traceback
        traceback.print_exc()