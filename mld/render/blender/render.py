import os
import shutil

import bpy

from .camera import Camera
from .floor import plot_floor, show_trajectory, show_hint
from .sampler import get_frameidx
from .scene import setup_scene
from .tools import delete_objs

from mld.render.video import Video
import numpy as np
import bmesh

class SceneDecorator:
    def __init__(self, scene_cfg=None):
        self.scene_cfg = scene_cfg
        self.scene_name = self.scene_cfg.scene_name
        self.use_guide_hint = (self.scene_cfg.use_guide_hint == 'True')
        print("乖！让我看看！", self.use_guide_hint, type(self.use_guide_hint))
        
        self.materials = {}

    def get_material(self, name, color, alpha=1.0, emission=0.0):
        if name in self.materials:
            return self.materials[name]
        
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name=name)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            
            nodes.clear()
            
            bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf.inputs['Base Color'].default_value = (*color, 1.0)
            bsdf.inputs['Alpha'].default_value = alpha
            if emission > 0:
                bsdf.inputs['Emission'].default_value = (*color, 1.0)
                bsdf.inputs['Emission Strength'].default_value = emission
            
            output = nodes.new(type='ShaderNodeOutputMaterial')
            nodes.active = bsdf
            mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
            
            # ================= [关键修改：透明设置] =================
            if alpha < 1.0:
                # 设置混合模式为 ALPHA_BLEND，这对于 Eevee 渲染半透明至关重要
                mat.blend_method = 'BLEND' 
                # 阴影模式设为 NONE，防止透明物体投下实心黑影
                mat.shadow_method = 'NONE'
                
                # 开启屏幕空间折射 (可选，增加真实感)
                mat.use_screen_refraction = True
            # =======================================================
        
        self.materials[name] = mat
        return mat

    def create_procedural_wood(self):
        """创建一个程序化木纹材质 (无需贴图文件)"""
        mat = bpy.data.materials.get("ProceduralWood")
        if mat: return mat
        
        mat = bpy.data.materials.new(name="ProceduralWood")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        
        # 输出
        output = nodes.new(type='ShaderNodeOutputMaterial')
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        
        # 纹理坐标 & 映射
        tex_coord = nodes.new(type='ShaderNodeTexCoord')
        mapping = nodes.new(type='ShaderNodeMapping')
        mapping.inputs['Scale'].default_value = (5.0, 1.0, 1.0) # 拉伸产生木纹感
        
        # 噪波纹理
        noise = nodes.new(type='ShaderNodeTexNoise')
        noise.inputs['Scale'].default_value = 10.0
        noise.inputs['Detail'].default_value = 10.0
        noise.inputs['Distortion'].default_value = 1.0
        
        # 颜色渐变 (木色)
        ramp = nodes.new(type='ShaderNodeValToRGB')
        ramp.color_ramp.elements[0].position = 0.4
        ramp.color_ramp.elements[0].color = (0.2, 0.1, 0.05, 1) # 深棕
        ramp.color_ramp.elements[1].position = 0.8
        ramp.color_ramp.elements[1].color = (0.6, 0.4, 0.2, 1)  # 浅棕
        
        # 连接
        links.new(tex_coord.outputs['Object'], mapping.inputs['Vector'])
        links.new(mapping.outputs['Vector'], noise.inputs['Vector'])
        links.new(noise.outputs['Fac'], ramp.inputs['Fac'])
        links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
        
        # 粗糙度
        bsdf.inputs['Roughness'].default_value = 0.7
        # 【修改】兼容不同版本的 Specular 输入名
        if 'Specular IOR Level' in bsdf.inputs:
            bsdf.inputs['Specular IOR Level'].default_value = 0.2
        elif 'Specular' in bsdf.inputs:
            bsdf.inputs['Specular'].default_value = 0.2
        
        return mat

    def create_balance_beam(self, points, width=0.4, thickness=0.05):
        """
        生成 NUBRS 独木桥 + 支架
        """
        # --- A. 制作木板 (Extrude Profile along Curve) ---
        
        # 1. 创建路径曲线 (Trajectory)
        curve_data = bpy.data.curves.new(name='BeamPath', type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.resolution_u = 4
        spline = curve_data.splines.new('NURBS')
        spline.points.add(len(points) - 1)
        
        for i, coord in enumerate(points):
            # 稍微降低一点高度，防止和脚底穿模 (假设脚底是0, 桥面设为 -0.02)
            spline.points[i].co = (coord[0], coord[1], -0.02, 1)
            
        spline.use_endpoint_u = True
        
        # 2. 创建截面曲线 (Profile) - 一个扁平矩形
        profile_data = bpy.data.curves.new(name='BeamProfile', type='CURVE')
        profile_data.dimensions = '2D'
        profile_spline = profile_data.splines.new('POLY')
        profile_spline.points.add(3)
        # 定义矩形形状 (相对于中心)
        w, h = width/2, thickness/2
        coords = [(-w, -h, 0), (-w, h, 0), (w, h, 0), (w, -h, 0)]
        for i, (x, y, z) in enumerate(coords):
            profile_spline.points[i].co = (x, y, z, 1)
        profile_spline.use_cyclic_u = True # 闭合
        
        profile_obj = bpy.data.objects.new("BeamProfileObj", profile_data)
        # 不用链接到场景，只要存在于数据块中即可，或者链接后隐藏
        
        # 3. 应用截面
        curve_data.bevel_mode = 'OBJECT'
        curve_data.bevel_object = profile_obj
        curve_data.use_fill_caps = True # 封口
        
        beam_obj = bpy.data.objects.new("BalanceBeam", curve_data)
        bpy.context.collection.objects.link(beam_obj)
        
        # 材质
        beam_obj.data.materials.append(self.create_procedural_wood())
        
        # --- B. 制作支架 (Supports) ---
        # 简单的做法：每隔一定距离，向下生成一个圆柱体直到深渊
        
        floor_depth = -3.0 # 地面下沉深度
        step = max(1, len(points) // 8) # 生成约8个支架
        
        # 材质：生锈金属
        mat_metal = self.get_material("RustyMetal", (0.2, 0.2, 0.2), alpha=1.0)
        
        for i in range(0, len(points), step):
            pt = points[i]
            # 支架高度
            height = abs(floor_depth) + pt[2] # 从桥面到地下
            
            bpy.ops.mesh.primitive_cylinder_add(
                radius=0.05, 
                depth=height, 
                location=(pt[0], pt[1], floor_depth + height/2) # 中心点
            )
            pole = bpy.context.object
            pole.name = "SupportPole"
            pole.data.materials.append(mat_metal)
            
            # 设为父子关系方便管理
            pole.parent = beam_obj
            
        return beam_obj

    def create_low_ceiling(self, body_data, padding=0.2):
        """
        【低矮天花板生成器】
        自动计算人的最高点，并在其上方生成天花板
        """
        # ================= [修改开始] =================
        # 兼容 Meshes 对象：如果传入的是对象，先取出里面的 vertices 数组
        if hasattr(body_data, 'data'):
            verts = body_data.data # 这里原仓库写的不太直观，verts在data字段里面
        else:
            verts = body_data
        # ================= [修改结束] =================

        # 下面的计算全部把 body_data 替换为 verts
        
        # 1. 计算所有帧、所有顶点的最大 Z 值 (Blender里 Z是高)
        max_height = np.max(verts[..., 2])
        
        ceiling_z = max_height + padding
        
        # 计算范围
        min_x = np.min(verts[..., 0]) - 1.5 # 稍微缩小一点范围，聚焦动作
        max_x = np.max(verts[..., 0]) + 1.5
        min_y = np.min(verts[..., 1]) - 1.5
        max_y = np.max(verts[..., 1]) + 1.5
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        scale_x = (max_x - min_x)
        scale_y = (max_y - min_y)
        
        # 创建 Cube
        bpy.ops.mesh.primitive_cube_add(
            size=1, 
            location=(center_x, center_y, ceiling_z + 0.05) 
        )
        ceiling = bpy.context.object
        ceiling.name = "LowCeiling"
        ceiling.scale = (scale_x, scale_y, 0.05)
        
        # ================= [关键修改：视觉效果] =================
        # 1. 设置半透明材质 (淡蓝色玻璃感，更有科技感)
        # Alpha=0.2 (很透), Color=淡蓝灰
        mat = self.get_material("CeilingGlass", (0.6, 0.8, 1.0), alpha=0.15, emission=0.0)
        
        # 增加透射 (Transmission) 让它像玻璃一样
        if mat.node_tree.nodes.get('Principled BSDF'):
            bsdf = mat.node_tree.nodes['Principled BSDF']
            # Blender 4.0+ 是 'Transmission Weight', 旧版是 'Transmission'
            if 'Transmission Weight' in bsdf.inputs:
                bsdf.inputs['Transmission Weight'].default_value = 0.8
            elif 'Transmission' in bsdf.inputs:
                bsdf.inputs['Transmission'].default_value = 0.8
            # 降低粗糙度
            bsdf.inputs['Roughness'].default_value = 0.1

        ceiling.data.materials.append(mat)

        # 3. 材质2：不透明深色边框 (Slot 1)
        mat_wire = self.get_material("CeilingWire", (0.1, 0.1, 0.1), alpha=1.0)
        ceiling.data.materials.append(mat_wire)
        
        # 4. 【关键】添加 Wireframe 修改器
        mod = ceiling.modifiers.new(name="Wireframe", type='WIREFRAME')
        mod.use_replace = False  # 保留原来的半透明面
        mod.thickness = 0.02     # 边框粗细
        mod.material_offset = 1  # 使用 Slot 1 (深色材质) 渲染线框
        
        return ceiling

    def load_asset_obj(self, path, loc=(0,0,0), rot=(0,0,0), scale=(1,1,1)):
        """
        加载外部 OBJ/FBX 模型
        """
        if not path or path == "": return
        
        import os
        ext = os.path.splitext(path)[-1].lower()
        
        try:
            if ext == '.obj':
                bpy.ops.import_scene.obj(filepath=path)
            elif ext == '.fbx':
                bpy.ops.import_scene.fbx(filepath=path)
            else:
                print(f"Unsupported asset format: {ext}")
                return
            
            # 获取刚导入的物体 (通常是选中的)
            objs = bpy.context.selected_objects
            for obj in objs:
                obj.location = loc
                obj.rotation_euler = rot
                obj.scale = scale
                
        except Exception as e:
            print(f"Failed to load asset {path}: {e}")

    def enhance_scene(self, trajectory_points, body_data):
        """
        主逻辑更新
        """
        print(f"🎨 Enhancing scene for: {self.scene_name}")
        
        # 获取场景中的地面物体 (假设之前的代码生成了叫 "floor" 的物体)
        floor_obj = bpy.data.objects.get("SmallPlane")
        if not floor_obj:
            floor_obj = bpy.data.objects.get("BigPlane")
        
        if self.scene_name == "Dumuqiao" or self.scene_name == "Balance":
            print("Constructing Balance Beam...")
            self.create_balance_beam(trajectory_points, width=0.4, thickness=0.08)
            
            # 【环境调整】让地面下沉，营造高空感
            if floor_obj:
                floor_obj.location.z = -3.0 # 下沉 3 米
                floor_obj.scale = (50, 50, 1) # 扩大一点防止穿帮
                
                # 给地面换个暗色材质，像深渊
                mat_floor = self.get_material("AbyssFloor", (0.05, 0.05, 0.1), alpha=1.0)
                if floor_obj.data.materials:
                    floor_obj.data.materials[0] = mat_floor
                else:
                    floor_obj.data.materials.append(mat_floor)

        elif self.scene_name in ["DiAiTianhuaban", "DiAiTongDao", "Crouch"]:
            print("Constructing Low Ceiling...")
            self.create_low_ceiling(body_data, padding=0.2)
            if self.use_guide_hint:
                self.create_guidance_ribbon(trajectory_points)
            
            # 保持地面不动，或者给地面加一点光泽
            if floor_obj:
                mat_floor = self.get_material("GlossyFloor", (0.3, 0.3, 0.3), alpha=1.0)
                # 增加反光
                if mat_floor.node_tree.nodes.get('Principled BSDF'):
                     mat_floor.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.2
                
                if floor_obj.data.materials:
                    floor_obj.data.materials[0] = mat_floor

        else:
            if self.use_guide_hint:
                self.create_guidance_ribbon(trajectory_points)

    def create_guidance_ribbon(self, points):
        """
        【优化需求1】把指引轨迹画成半透明的带子/箭头，而不是 Zigzag 的线条
        """
        # 创建一个 Curve
        curve_data = bpy.data.curves.new(name='GuideRibbon', type='CURVE')
        curve_data.dimensions = '3D'
        spline = curve_data.splines.new('NURBS')
        spline.points.add(len(points) - 1)
        for i, coord in enumerate(points):
            spline.points[i].co = (coord[0], coord[1], 0.02, 1) # 稍微浮在地面上
        
        obj = bpy.data.objects.new("GuidanceRibbon", curve_data)
        bpy.context.collection.objects.link(obj)
        
        # 变成扁平带子
        curve_data.bevel_depth = 0.10 # 宽度
        curve_data.extrude = 0.005 # 极薄
        
        # 材质：发光、半透明青色
        mat = self.get_material("GuideMat", (1.0, 0.5, 0.0), alpha=0.2, emission=0.6)
        obj.data.materials.append(mat)


def prune_begin_end(data, perc):
    to_remove = int(len(data) * perc)
    if to_remove == 0:
        return data
    return data[to_remove:-to_remove]


def render_current_frame(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(use_viewport=True, write_still=True)


def render(npydata, trajectory, path, mode, faces_path, gt=False,
           exact_frame=None, num=8, always_on_floor=False, denoising=True,
           oldrender=True, res="high", accelerator='gpu', device=[0], fps=20, hint=None, cfg=None):

    if mode == 'video':
        if always_on_floor:
            frames_folder = path.replace(".pkl", "_of_frames")
        else:
            frames_folder = path.replace(".pkl", "_frames")

        if os.path.exists(frames_folder.replace("_frames", ".mp4")) or os.path.exists(frames_folder):
            print(f"pkl is rendered or under rendering {path}")
            return

        os.makedirs(frames_folder, exist_ok=False)

    elif mode == 'sequence':
        path = path.replace('.pkl', '.png')
        img_name, ext = os.path.splitext(path)
        if always_on_floor:
            img_name += "_of"
        img_path = f"{img_name}{ext}"
        if os.path.exists(img_path):
            print(f"pkl is rendered or under rendering {img_path}")
            return

    elif mode == 'frame':
        path = path.replace('.pkl', '.png')
        img_name, ext = os.path.splitext(path)
        if always_on_floor:
            img_name += "_of"
        img_path = f"{img_name}_{exact_frame}{ext}"
        if os.path.exists(img_path):
            print(f"pkl is rendered or under rendering {img_path}")
            return
    else:
        raise ValueError(f'Invalid mode: {mode}')

    # Setup the scene (lights / render engine / resolution etc)
    setup_scene(res=res, denoising=denoising, oldrender=oldrender, accelerator=accelerator, device=device)

    # remove X% of beginning and end
    # as it is almost always static
    # in this part
    # if mode == "sequence":
    #     perc = 0.2
    #     npydata = prune_begin_end(npydata, perc)

    from .meshes import Meshes
    data = Meshes(npydata, gt=gt, mode=mode, trajectory=trajectory,
                  faces_path=faces_path, always_on_floor=always_on_floor)
    
    # Create a floor
    plot_floor(data.data, big_plane=False)

    # scene_name = "DiAiTianhuaban"
    if hint is not None:
        hint = hint[..., [2, 0, 1]]
    decorator = SceneDecorator(scene_cfg = cfg)
    decorator.enhance_scene(hint, data)

    # Number of frames possible to render
    nframes = len(data)

    # Show the trajectory
    if trajectory is not None:
        show_trajectory(data.trajectory)

    # initialize the camera
    camera = Camera(first_root=data.get_root(0), mode=mode)

    frameidx = get_frameidx(mode=mode, nframes=nframes,
                            exact_frame=exact_frame,
                            frames_to_keep=num)

    nframes_to_render = len(frameidx)

    # center the camera to the middle
    if mode == "sequence":
        camera.update(data.get_mean_root())

    imported_obj_names = []
    for index, frameidx in enumerate(frameidx):
        if mode == "sequence":
            frac = index / (nframes_to_render - 1)
            mat = data.get_sequence_mat(frac)
        else:
            mat = data.mat
            camera.update(data.get_root(frameidx))

        islast = index == (nframes_to_render - 1)

        obj_name = data.load_in_blender(frameidx, mat)
        name = f"{str(index).zfill(4)}"

        if mode == "video":
            path = os.path.join(frames_folder, f"frame_{name}.png")
        else:
            path = img_path

        if mode == "sequence":
            imported_obj_names.extend(obj_name)
        elif mode == "frame":
            camera.update(data.get_root(frameidx))

        if mode != "sequence" or islast:
            render_current_frame(path)
            delete_objs(obj_name)

    # remove every object created
    delete_objs(imported_obj_names)
    delete_objs(["Plane", "myCurve", "Cylinder"])

    if mode == "video":
        video = Video(frames_folder, fps=fps)
        vid_path = frames_folder.replace("_frames", ".mp4")
        video.save(out_path=vid_path)
        shutil.rmtree(frames_folder)
        print(f"remove tmp fig folder and save video in {vid_path}")

    else:
        print(f"Frame generated at: {img_path}")
