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
import math
import mathutils

# --- 抽象基类: 策略模式+注册表 ---
class BaseSceneStrategy:
    def __init__(self, context):
        self.ctx = context  # 持有 SceneDecorator 实例，以便调用它的 create_xxx 方法及材质

    def execute(self, trajectory_points, body_data):
        """每个策略必须实现的方法"""
        # 默认行为：如果开了开关，就画指引线
        if self.ctx.use_guide_hint:
            self.ctx.create_guidance_ribbon(trajectory_points)

# --- 具体策略：独木桥 ---
class BalanceStrategy(BaseSceneStrategy):
    def execute(self, trajectory_points, body_data):
        print("🏗️ Executing Balance Strategy...")
        # 1. 生成独木桥
        self.ctx.create_balance_beam(trajectory_points, width=0.4, thickness=0.08)
        
        # 2. 地面下沉 (获取名为 SmallPlane 或 BigPlane 的地面)
        floor_obj = bpy.data.objects.get("SmallPlane") or bpy.data.objects.get("BigPlane") or bpy.data.objects.get("floor")
        
        if floor_obj:
            floor_obj.location.z = -3.0 # 下沉
            floor_obj.scale = (50, 50, 1)
            # 换个深渊材质
            mat_floor = self.ctx.get_material("AbyssFloor", (0.02, 0.02, 0.05), alpha=1.0)
            if floor_obj.data.materials: floor_obj.data.materials[0] = mat_floor
            else: floor_obj.data.materials.append(mat_floor)


# --- 具体策略：低矮天花板 ---
class CeilingStrategy(BaseSceneStrategy):
    def execute(self, trajectory_points, body_data):
        print("🏗️ Executing Ceiling Strategy...")
        # 1. 生成天花板
        self.ctx.create_low_ceiling(body_data, padding=0.2)
        
        # 2. 画指引线 (调用基类逻辑或自己画)
        super().execute(trajectory_points, body_data)
        
        # 3. 地面光泽感
        floor_obj = bpy.data.objects.get("SmallPlane") or bpy.data.objects.get("BigPlane")
        if floor_obj:
            mat_floor = self.ctx.get_material("GlossyFloor", (0.3, 0.3, 0.3), alpha=1.0)
            # 增加反光
            if mat_floor.node_tree.nodes.get('Principled BSDF'):
                mat_floor.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.2
            if floor_obj.data.materials: floor_obj.data.materials[0] = mat_floor
            else: floor_obj.data.materials.append(mat_floor)
    

class StormStrategy(BaseSceneStrategy):
    def execute(self, trajectory_points, body_data):
        print("⛈️ Executing Storm Strategy (Back to Basics)...")

        # ================= [1. 环境与光影] =================
        bpy.context.scene.render.film_transparent = False
        
        # 清理
        for obj in bpy.data.objects:
            if obj.type == 'LIGHT' or "Sun" in obj.name:
                bpy.data.objects.remove(obj, do_unlink=True)

        # HDRI 压暗
        self.ctx.setup_ibl("/root/autodl-tmp/MyRepository/MCM-LDM/assets/overcast_soil_puresky_4k.exr", strength=0.3, rotation=0)

        # 强背光 (保留，为了照亮雪花)
        bpy.ops.object.light_add(type='SPOT', location=(0, -5, 8))
        key_light = bpy.context.object
        key_light.data.energy = 3000
        key_light.data.color = (0.7, 0.8, 1.0)
        key_light.rotation_euler = (0.8, 0, 0)

        # ================= [2. 地面] =================
        if "floor" in bpy.data.objects: bpy.data.objects.remove(bpy.data.objects["floor"])
        if "SmallPlane" in bpy.data.objects: bpy.data.objects.remove(bpy.data.objects["SmallPlane"])
        if "BigPlane" in bpy.data.objects: bpy.data.objects.remove(bpy.data.objects["BigPlane"])
        
        bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, 0))
        ground = bpy.context.object
        ground.name = "SnowGround"
        
        mat_snow = self.ctx.get_material("RoughSnow", (0.8, 0.85, 0.9), alpha=1.0)
        if mat_snow.node_tree.nodes.get('Principled BSDF'):
            mat_snow.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 1.0
        ground.data.materials.append(mat_snow)

        # ================= [3. 枯树 (拼接法 - 拒绝光杆)] =================
        import random
        # 定义造树函数：用圆锥体(Cone)拼，上面细下面粗，就像树
        def spawn_tree(x, y):
            # 主干
            bpy.ops.mesh.primitive_cone_add(radius1=0.2, radius2=0.08, depth=5.0, location=(x, y, 2.5))
            trunk = bpy.context.object
            trunk.rotation_euler = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), 0)
            mat_wood = self.ctx.get_material("DarkWood", (0.05, 0.04, 0.03))
            trunk.data.materials.append(mat_wood)
            
            # 分叉1
            bpy.ops.mesh.primitive_cone_add(radius1=0.08, radius2=0.02, depth=2.0, location=(x, y, 3.5))
            branch1 = bpy.context.object
            branch1.rotation_euler = (0.6, 0, random.uniform(0, 6))
            branch1.data.materials.append(mat_wood)
            
            # 分叉2
            bpy.ops.mesh.primitive_cone_add(radius1=0.06, radius2=0.01, depth=1.5, location=(x, y, 4.0))
            branch2 = bpy.context.object
            branch2.rotation_euler = (-0.7, 0.3, random.uniform(0, 6))
            branch2.data.materials.append(mat_wood)

        # 在轨迹周围种树
        center_idx = len(trajectory_points) // 2
        # 选3个点种树
        refs = [trajectory_points[0], trajectory_points[center_idx], trajectory_points[-1]]
        for pt in refs:
            offset_x = random.choice([-3.0, 3.0])
            # 随机偏移一点，防止太整齐
            spawn_tree(pt[0] + offset_x + random.uniform(-0.5, 0.5), pt[1])

        # ================= [4. 暴风雪 (严格复刻你的成功代码)] =================
        # 1. 创建平面发射器 (Z=10)
        bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 9)) # 稍微大一点，低一点
        emitter = bpy.context.object
        emitter.name = "SnowEmitter"
        
        # 【核心修正】确保物体本身是可见的，只是渲染时隐藏实例器
        emitter.hide_render = False 
        
        # 2. 粒子系统
        pset = emitter.modifiers.new(name="SnowParticles", type='PARTICLE_SYSTEM').particle_system
        settings = pset.settings
        
        settings.count = 80000  # 你的代码是 10000，我稍微加点
        settings.frame_start = -100 # 你的代码逻辑
        settings.frame_end = 200 
        settings.lifetime = 200
        
        settings.physics_type = 'NEWTON'
        settings.mass = 0.05
        settings.brownian_factor = 1.0 # 稍微乱一点
        settings.drag_factor = 0.1
        
        # 3. 渲染设置
        # 必须设为 False (Blender逻辑：不渲染发射器平面，只渲染粒子)
        emitter.show_instancer_for_render = False
        emitter.show_instancer_for_viewport = False
        
        # 4. 雪花实体
        bpy.ops.mesh.primitive_ico_sphere_add(radius=0.025, subdivisions=1) 
        snowflake = bpy.context.object

        # 3. 【关键】开启平滑着色，消除棱角感
        bpy.ops.object.shade_smooth()
        snowflake.name = "SnowFlakePrototype"
        snowflake.hide_render = True 
        snowflake.hide_viewport = True
        snowflake.location.z = -100
        
        # 材质：稍微亮一点，确保看见
        mat_snow = self.ctx.get_material("BrightSnow", (1.0, 1.0, 1.0), alpha=1.0, emission=3.0)
        snowflake.data.materials.append(mat_snow)
        
        settings.render_type = 'OBJECT'
        settings.instance_object = snowflake
        settings.particle_size = 0.7 # 你的代码是 0.6
        settings.size_random = 0.8 
        
        # 5. 风场 (保留你的逻辑)
        bpy.ops.object.effector_add(type='WIND', location=(-10, 0, 5), rotation=(0, 1.2, 0))
        wind = bpy.context.object
        wind.field.strength = 20.0 
        wind.field.noise = 5.0

        # ================= [5. 雾气 (淡淡的)] =================
        world = bpy.context.scene.world
        if world.node_tree:
            nodes = world.node_tree.nodes
            links = world.node_tree.links
            output = nodes.get('World Output')
            if output and not output.inputs['Volume'].is_linked:
                volume = nodes.new(type='ShaderNodeVolumeScatter')
                volume.inputs['Density'].default_value = 0.02
                volume.inputs['Color'].default_value = (0.9, 0.9, 0.95, 1)
                links.new(volume.outputs['Volume'], output.inputs['Volume'])

        # 6. 指引线
        super().execute(trajectory_points, body_data)
        
        # ================= [强制刷新] =================
        bpy.context.view_layer.update()
        scene = bpy.context.scene
        # 往前推几帧，激活粒子
        current = scene.frame_current
        for f in range(current-2, current+1):
            scene.frame_set(f)
            bpy.context.view_layer.update()

import math

class DarkStrategy(BaseSceneStrategy):
    def execute(self, trajectory_points, body_data):
        print("🌑 Executing Dark Strategy (Solid Volume Box)...")

        # ================= [1. 强制渲染参数 (专为光束优化)] =================
        scn = bpy.context.scene
        scn.render.film_transparent = False
        
        # 恢复曝光
        if hasattr(scn.view_settings, "exposure"):
            scn.view_settings.exposure = 0.0 
            
        # 【关键】体积光采样精度必须高，不然光束是断的
        # 覆盖掉 setup_renderer 里的优化设置
        scn.cycles.volume_step_rate = 0.5 # 越小越精细，0.5 保证光柱细腻
        scn.cycles.volume_bounces = 1     # 至少反弹1次，让雾气有点自我照亮
        
        # 清理
        for obj in bpy.data.objects:
            if obj.type == 'LIGHT' or "Sun" in obj.name:
                bpy.data.objects.remove(obj, do_unlink=True)

        # ================= [2. 环境 (纯黑)] =================
        world = bpy.context.scene.world
        if world.node_tree:
            nodes = world.node_tree.nodes
            links = world.node_tree.links
            nodes.clear()
            output = nodes.new(type='ShaderNodeOutputWorld')
            bg = nodes.new(type='ShaderNodeBackground')
            bg.inputs['Color'].default_value = (0, 0, 0, 1)
            bg.inputs['Strength'].default_value = 0.0 # 彻底关掉环境光
            links.new(bg.outputs['Background'], output.inputs['Surface'])
            # 注意：这里不再连接 World Volume，我们改用实体盒子

        # ================= [3. 实体雾气盒子 (必现光束)] =================
        # 创建一个巨大的盒子罩住场景
        bpy.ops.mesh.primitive_cube_add(size=50, location=(0, 0, 10))
        fog_box = bpy.context.object
        fog_box.name = "FogVolumeBox"
        fog_box.display_type = 'WIRE' # 视口只看线框，不挡视线
        
        # 创建体积材质
        mat_fog = bpy.data.materials.new(name="VolumetricFog")
        mat_fog.use_nodes = True
        nodes = mat_fog.node_tree.nodes
        links = mat_fog.node_tree.links
        nodes.clear()
        
        output = nodes.new(type='ShaderNodeOutputMaterial')
        volume = nodes.new(type='ShaderNodeVolumeScatter')
        
        # 【参数微调】
        # 密度：0.05 (太浓会黑，太淡没光束，0.05是黄金值)
        volume.inputs['Density'].default_value = 0.05 
        # 各向异性：0.7 (既有聚光感，侧面又能看见) -> 之前0.9太高了
        volume.inputs['Anisotropy'].default_value = 0.7
        # 颜色：纯白或微蓝
        volume.inputs['Color'].default_value = (0.05, 0.05, 0.06, 1) 
        
        # 连接到 Volume 插槽 (Surface 插槽留空=透明)
        links.new(volume.outputs['Volume'], output.inputs['Volume'])
        fog_box.data.materials.append(mat_fog)

        # ================= [4. 地面 (黑镜)] =================
        if "floor" in bpy.data.objects: bpy.data.objects.remove(bpy.data.objects["floor"])
        if "SmallPlane" in bpy.data.objects: bpy.data.objects.remove(bpy.data.objects["SmallPlane"])
        if "BigPlane" in bpy.data.objects: bpy.data.objects.remove(bpy.data.objects["BigPlane"])
        
        bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -0.01))
        ground = bpy.context.object
        ground.name = "DarkFloor"
        
        mat_floor = self.ctx.get_material("DarkMirror", (0.0, 0.0, 0.0), alpha=1.0)
        if mat_floor.node_tree.nodes.get('Principled BSDF'):
            bsdf = mat_floor.node_tree.nodes['Principled BSDF']
            bsdf.inputs['Base Color'].default_value = (0.0, 0.0, 0.0, 1)
            bsdf.inputs['Roughness'].default_value = 0.05 # 稍微有一点点磨砂，拉长倒影
            bsdf.inputs['Specular'].default_value = 0.5
        ground.data.materials.append(mat_floor)

        # ================= [4. 灯光 (论文可视化专用 - 清晰暗夜版)] =================
        
        # --- Light 1: 轮廓背光 (Atmosphere) ---
        # 放在左后方，用来把人和背景分离开
        bpy.ops.object.light_add(type='SPOT', location=(-7, 7, 6)) 
        rim_light = bpy.context.object
        rim_light.name = "RimLight"
        
        # 蓝色强光
        rim_light.data.energy = 80000 
        rim_light.data.color = (0.1, 0.3, 1.0) # 纯蓝
        rim_light.data.spot_size = 0.6
        rim_light.data.spot_blend = 0.2
        # 瞄准中心
        rim_light.rotation_euler = (math.radians(50), 0, math.radians(-45)) 

        # --- Light 2: 侧前补光 (Visibility - 关键！) ---
        # 放在右前方，照亮动作细节
        bpy.ops.object.light_add(type='AREA', location=(6, -4, 4))
        fill_light = bpy.context.object
        fill_light.name = "MoonFill"
        
        # 【关键设置】
        # 1. 颜色：淡蓝灰色 (模拟月光)，不要用暖色，否则像白天
        fill_light.data.color = (0.7, 0.8, 0.9) 
        
        # 2. 强度：适中 (2000-3000)。
        # 既能看清黄色小人的四肢，又不会把地面照得惨白
        fill_light.data.energy = 2500 
        
        # 3. 形状：大面积柔光，让阴影不那么黑
        fill_light.data.shape = 'RECTANGLE'
        fill_light.data.size = 8.0
        
        # 瞄准人物侧面
        fill_light.rotation_euler = (math.radians(60), 0, math.radians(50))

        # --- Light 3: 顶部微光 (Top Separator) ---
        # 防止头顶和肩膀死黑
        bpy.ops.object.light_add(type='POINT', location=(0, 0, 5))
        top_light = bpy.context.object
        top_light.data.energy = 500
        top_light.data.color = (1.0, 1.0, 1.0)

        # ================= [6. 指引线] =================
        self.ctx.create_guidance_ribbon(trajectory_points)
        ribbon = bpy.data.objects.get("GuidanceRibbon")
        if ribbon and ribbon.data.materials:
            mat = ribbon.data.materials[0]
            if mat.node_tree.nodes.get('Principled BSDF'):
                # 荧光绿
                mat.node_tree.nodes['Principled BSDF'].inputs['Emission'].default_value = (0.0, 0.2, 1.0, 1)
                mat.node_tree.nodes['Principled BSDF'].inputs['Emission Strength'].default_value = 15.0

        # 强制刷新
        bpy.context.view_layer.update()

# class DarkStrategy(BaseSceneStrategy):
#     def execute(self, trajectory_points, body_data):
#         print("🌑 Executing Dark Strategy (The Void Version)...")

#         # ================= [1. 制造绝对黑暗背景] =================
#         scn = bpy.context.scene
#         scn.render.film_transparent = False
        
#         # 曝光度调回正常，我们靠灯光强度来控制明暗
#         if hasattr(scn.view_settings, "exposure"):
#             scn.view_settings.exposure = 0.0 
        
#         # 清理
#         for obj in bpy.data.objects:
#             if obj.type == 'LIGHT' or "Sun" in obj.name:
#                 bpy.data.objects.remove(obj, do_unlink=True)

#         # World 设置：纯黑背景 + 稀薄暗雾
#         world = bpy.context.scene.world
#         if world.node_tree:
#             nodes = world.node_tree.nodes
#             links = world.node_tree.links
#             nodes.clear()
            
#             output = nodes.new(type='ShaderNodeOutputWorld')
#             bg = nodes.new(type='ShaderNodeBackground')
            
#             # 背景绝对纯黑
#             bg.inputs['Color'].default_value = (0, 0, 0, 1)
#             bg.inputs['Strength'].default_value = 0.0 
#             links.new(bg.outputs['Background'], output.inputs['Surface'])

#             # 体积雾 (关键修改：颜色变暗，密度降低)
#             volume = nodes.new(type='ShaderNodeVolumeScatter')
#             # 密度降低：防止画面发白
#             volume.inputs['Density'].default_value = 0.02 
#             # 颜色变暗：雾本身不应该太亮，要靠光打亮
#             volume.inputs['Color'].default_value = (0.3, 0.3, 0.3, 1) 
#             # 强各向异性：只在逆光方向显示光束
#             volume.inputs['Anisotropy'].default_value = 0.9 
#             links.new(volume.outputs['Volume'], output.inputs['Volume'])

#         # ================= [2. 地面 (黑镜)] =================
#         if "floor" in bpy.data.objects: bpy.data.objects.remove(bpy.data.objects["floor"])
#         if "SmallPlane" in bpy.data.objects: bpy.data.objects.remove(bpy.data.objects["SmallPlane"])
#         if "BigPlane" in bpy.data.objects: bpy.data.objects.remove(bpy.data.objects["BigPlane"])
        
#         bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -0.01))
#         ground = bpy.context.object
#         ground.name = "DarkFloor"
        
#         mat_floor = self.ctx.get_material("DarkMirror", (0.0, 0.0, 0.0), alpha=1.0)
#         if mat_floor.node_tree.nodes.get('Principled BSDF'):
#             bsdf = mat_floor.node_tree.nodes['Principled BSDF']
#             bsdf.inputs['Base Color'].default_value = (0.0, 0.0, 0.0, 1) # 纯黑
#             bsdf.inputs['Roughness'].default_value = 0.2 # 完美镜面
#             bsdf.inputs['Specular'].default_value = 1.0  # 高反射
#         ground.data.materials.append(mat_floor)

#         # ================= [3. 灯光 (极端对比度)] =================
        
#         # --- Light 1: 探照灯 (Rim Light) ---
#         # 放在正后方高处 (Y+)
#         bpy.ops.object.light_add(type='SPOT', location=(0, 12, 6)) 
#         rim_light = bpy.context.object
#         rim_light.name = "SearchLight"
        
#         # 参数调整：更聚光，更硬
#         rim_light.data.energy = 5000000 # 500万瓦 (配合暗雾)
#         rim_light.data.color = (0.6, 0.8, 1.0) # 冷白
        
#         # 【关键】光束极窄，像舞台追光灯
#         rim_light.data.spot_size = 0.4 # 约20度角，非常窄
#         rim_light.data.spot_blend = 0.0 # 边缘像刀切一样硬，强化光束感
        
#         # 瞄准原点
#         rim_light.rotation_euler = (math.radians(115), 0, 0) 

#         # --- Light 2: 补光 (几乎移除) ---
#         # 之前是 3000，太亮了导致像正面光。
#         # 现在降到 50，只为了让你勉强能看到一点点黄色的皮肤，而不是纯黑剪影
#         bpy.ops.object.light_add(type='AREA', location=(5, -5, 2))
#         fill_light = bpy.context.object
#         fill_light.data.energy = 500 
#         fill_light.data.color = (1.0, 0.5, 0.2) # 极弱的暖光
#         fill_light.data.size = 10.0 # 大面积柔光
#         fill_light.rotation_euler = (math.radians(60), 0, math.radians(45))

#         # ================= [4. 指引线 (自发光)] =================
#         self.ctx.create_guidance_ribbon(trajectory_points)
#         ribbon = bpy.data.objects.get("GuidanceRibbon")
#         if ribbon and ribbon.data.materials:
#             mat = ribbon.data.materials[0]
#             if mat.node_tree.nodes.get('Principled BSDF'):
#                 # 绿色激光感
#                 mat.node_tree.nodes['Principled BSDF'].inputs['Emission'].default_value = (0.0, 0.1, 1.0, 1)
#                 mat.node_tree.nodes['Principled BSDF'].inputs['Emission Strength'].default_value = 20.0

#         # 强制刷新
#         bpy.context.view_layer.update()

class SceneDecorator:
    def __init__(self, scene_cfg=None):
        self.scene_cfg = scene_cfg
        self.scene_name = self.scene_cfg.scene_name
        self.use_guide_hint = (str(self.scene_cfg.use_guide_hint) == 'True') # 防御性转换
        
        self.materials = {}

        # ================= [注册表] =================
        # 这里的 Key 对应你的 scene_name，Value 是上面的策略类
        self.strategies = {
            "Dumuqiao": BalanceStrategy,
            "Balance": BalanceStrategy,
            
            "DiAiTianhuaban": CeilingStrategy,
            "DiAiTongDao": CeilingStrategy,
            "Crouch": CeilingStrategy,
            
            "BaoFengYu": StormStrategy, # [新场景]
            "Snow": StormStrategy,
            
            "Dark": DarkStrategy,       # [新场景]
            "Darkness": DarkStrategy
        }
        # ===========================================

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

    def create_balance_beam(self, points, width=0.4, thickness=0.05, extend_length=1.0):
        """
        生成 NUBRS 独木桥 + 支架
        """
        # --- A. 制作木板 (Extrude Profile along Curve) ---
        # 路径 (Path)：就是你传入的 points（轨迹点）。这决定了独木桥的“走向”。代码里对应 curve_data 和 spline。
        # 截面 (Profile)：就是代码里生成的 BeamProfileObj（那个扁平的矩形）。这决定了独木桥的“形状”和“粗细”。
        # 生成 (Bevel Object)：代码 curve_data.bevel_object = profile_obj 告诉 Blender：“把这个矩形截面，沿着路径拉伸一遍”。
        '''
        Blender 的技巧点：
            NURBS Spline：使用 NURBS 算法让折线点变成平滑的曲线，这样独木桥也是弯曲的。
            Bevel Object：这是非破坏性建模，你不需要操作成千上万个顶点，只需要调整截面矩形的大小，整个桥的厚度就会变。
        '''
        
        # --- A. 制作木板 ---
        
        # 1. 预处理点：计算延伸
        # 转换为 numpy 方便计算
        pts = np.array(points)
        
        # 计算起点的延伸点 (方向: P1 -> P0)
        # 向量 P1_to_P0 = P0 - P1
        start_vec = pts[0] - pts[1]
        # 归一化 (变成长度为1的单位向量)
        start_vec = start_vec / (np.linalg.norm(start_vec) + 1e-8)
        # 新起点 = 旧起点 + 方向 * 长度
        new_start = pts[0] + start_vec * extend_length
        
        # 计算终点的延伸点 (方向: P_last-1 -> P_last)
        end_vec = pts[-1] - pts[-2]
        end_vec = end_vec / (np.linalg.norm(end_vec) + 1e-8)
        new_end = pts[-1] + end_vec * extend_length
        
        # 拼接新的点集：[新起点, 旧P0, ..., 旧Pn, 新终点]
        # 注意：为了保持 NURBS 在连接处的平滑，最好保留原始端点作为控制点
        extended_points = np.vstack(([new_start], pts, [new_end]))

        # 2. 创建路径曲线
        curve_data = bpy.data.curves.new(name='BeamPath', type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.resolution_u = 4
        spline = curve_data.splines.new('NURBS')
        
        # 使用延伸后的点集
        spline.points.add(len(extended_points) - 1)
        
        for i, coord in enumerate(extended_points):
            # 稍微降低高度 -0.02
            spline.points[i].co = (coord[0], coord[1], -0.02, 1)
            
        spline.use_endpoint_u = True # 让曲线强制经过首尾端点
        
        # 3. 创建截面 (Profile) - 保持不变
        profile_data = bpy.data.curves.new(name='BeamProfile', type='CURVE')
        profile_data.dimensions = '2D'
        profile_spline = profile_data.splines.new('POLY')
        profile_spline.points.add(3)
        w, h = width/2, thickness/2
        coords = [(-w, -h, 0), (-w, h, 0), (w, h, 0), (w, -h, 0)]
        for i, (x, y, z) in enumerate(coords):
            profile_spline.points[i].co = (x, y, z, 1)
        profile_spline.use_cyclic_u = True 
        profile_obj = bpy.data.objects.new("BeamProfileObj", profile_data)
        
        # 4. 应用截面
        curve_data.bevel_mode = 'OBJECT'
        curve_data.bevel_object = profile_obj
        curve_data.use_fill_caps = True 
        
        beam_obj = bpy.data.objects.new("BalanceBeam", curve_data)
        bpy.context.collection.objects.link(beam_obj)
        beam_obj.data.materials.append(self.create_procedural_wood())
        
        # --- B. 制作支架 (Supports) ---
        floor_depth = -2.5
        # 只需要在原始路段生成支架即可，延伸段悬空也没关系，或者全生成
        # 这里我们用 extended_points 生成，间距稍微大点
        step = max(1, len(extended_points) // 10) 
        mat_metal = self.get_material("RustyMetal", (0.2, 0.2, 0.2), alpha=1.0)
        
        for i in range(0, len(extended_points), step):
            pt = extended_points[i]
            height = abs(floor_depth) + pt[2] 
            bpy.ops.mesh.primitive_cylinder_add(
                radius=0.05, 
                depth=height, 
                location=(pt[0], pt[1], floor_depth + height/2)
            )
            pole = bpy.context.object
            pole.name = "SupportPole"
            pole.data.materials.append(mat_metal)
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

    def setup_ibl(self, hdri_path, strength=1.0, rotation=0.0):
            """
            设置基于图像的照明 (IBL) / Skybox
            """
            import os
            if not os.path.exists(hdri_path):
                print(f"⚠️ HDRI not found at: {hdri_path}, skipping IBL.")
                return

            world = bpy.context.scene.world
            world.use_nodes = True
            nodes = world.node_tree.nodes
            links = world.node_tree.links
            
            # 1. 清理旧的环境节点
            nodes.clear()
            
            # 2. 创建节点链
            # Texture Environment (加载 HDRI)
            tex_env = nodes.new(type='ShaderNodeTexEnvironment')
            tex_env.image = bpy.data.images.load(hdri_path)
            
            # Mapping & Coordinates (用于旋转天空盒)
            mapping = nodes.new(type='ShaderNodeMapping')
            tex_coord = nodes.new(type='ShaderNodeTexCoord')
            
            # Background (控制强度)
            bg = nodes.new(type='ShaderNodeBackground')
            bg.inputs['Strength'].default_value = strength
            
            # Output
            output = nodes.new(type='ShaderNodeOutputWorld')
            
            # 3. 连接
            links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
            links.new(mapping.outputs['Vector'], tex_env.inputs['Vector'])
            links.new(tex_env.outputs['Color'], bg.inputs['Color'])
            links.new(bg.outputs['Background'], output.inputs['Surface'])
            
            # 4. 设置旋转 (Z轴)
            import math
            mapping.inputs['Rotation'].default_value[2] = math.radians(rotation)
            
            print(f"🌍 IBL setup with {os.path.basename(hdri_path)}, Strength={strength}")

    # 【核心修改】：enhance_scene 变得非常简洁
    # -----------------------------------------------------------
    def enhance_scene(self, trajectory_points, body_data):
        """
        主入口：根据场景名分发给对应的策略
        """
        print(f"🎨 Enhancing scene for: {self.scene_name}")
        
        # 1. 查找策略 (如果没有匹配，使用默认策略)
        strategy_cls = self.strategies.get(self.scene_name, BaseSceneStrategy)
        
        # 2. 实例化策略 (传入 self 作为 context)
        strategy = strategy_cls(self)
        
        # 3. 执行
        try:
            strategy.execute(trajectory_points, body_data)
        except Exception as e:
            print(f"❌ Error executing scene strategy: {e}")
            import traceback
            traceback.print_exc()


    def create_guidance_ribbon(self, points):
        # TODO: 暂时改成绘制箭头，有需要的话再改回去
        self.draw_guidance_path_with_arrow(points)
        return
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
    
    def draw_guidance_path_with_arrow(self, coords, color=(1.0, 0.1, 0.1, 1.0)):
        """
        绘制游戏UI级别的指引轨迹：密集的发光点阵 + 终点方向箭头
        
        :param coords: shape (N, 3) 的轨迹点数组
        :param color: RGBA 颜色，默认亮红色，形成高对比度
        """
        if len(coords) < 2:
            return

        # 1. 创建全局发光材质
        mat_name = "GuidancePathMat"
        mat = bpy.data.materials.get(mat_name)
        if not mat:
            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes["Principled BSDF"]
            bsdf.inputs['Base Color'].default_value = color
            
            if 'Emission' in bsdf.inputs:
                bsdf.inputs['Emission'].default_value = color
                bsdf.inputs['Emission Strength'].default_value = 3.0 # 提高发光强度，使其像霓虹灯

        # 2. 绘制密集的路径点阵 (小一点，贴地)
        # 这里 step 设为 2 或 3，保证密集度，又能稍微节省一点渲染资源
        for i in range(0, len(coords) - 1, 2):
            pt = coords[i]
            # 画极小的扁平球体作为路径点，Z轴贴地 (0.02)
            bpy.ops.mesh.primitive_uv_sphere_add(
                radius=0.03, 
                location=(pt[0], pt[1], 0.02)
            )
            dot_obj = bpy.context.active_object
            dot_obj.name = f"GuideDot_{i}"
            
            if dot_obj.data.materials:
                dot_obj.data.materials[0] = mat
            else:
                dot_obj.data.materials.append(mat)

        # 3. 在轨迹的最末端，计算方向并画一个醒目的终点箭头
        last_pt = mathutils.Vector((coords[-1][0], coords[-1][1], 0.05))
        prev_pt = mathutils.Vector((coords[-5][0], coords[-5][1], 0.05)) # 取倒数第5个点算切线更稳定
        
        direction = last_pt - prev_pt
        if direction.length > 0.001:
            rot_quat = direction.to_track_quat('Z', 'Y')
            
            # 画一个比路径点大得多的箭头
            bpy.ops.mesh.primitive_cone_add(
                vertices=32,
                radius1=0.08,   # 箭头够宽
                depth=0.4,      # 箭头够长
                location=last_pt
            )
            
            arrow_obj = bpy.context.active_object
            arrow_obj.name = "GuideTerminalArrow"
            arrow_obj.rotation_mode = 'QUATERNION'
            arrow_obj.rotation_quaternion = rot_quat
            
            if arrow_obj.data.materials:
                arrow_obj.data.materials[0] = mat
            else:
                arrow_obj.data.materials.append(mat)
                
        bpy.ops.object.mode_set(mode='OBJECT')

    def create_fractal_tree(self, location=(0,0,0), scale=1.0):
        """
        【修复版】极简枯树生成器 (Mesh Extrude 方案)
        不再使用递归，防止坐标爆炸。
        """
        import random
        import math
        
        # 1. 创建一个圆柱体作为树干基底
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.15 * scale, 
            depth=1.0, 
            location=location
        )
        tree = bpy.context.object
        tree.name = "DeadTree"
        
        # 2. 进入编辑模式，进行随机挤出 (Extrude)
        bpy.ops.object.mode_set(mode='EDIT')
        
        # 往上挤出 4-5 段，每段随机扭曲
        current_height = 0
        for i in range(5):
            # 挤出
            bpy.ops.mesh.extrude_region_move(
                TRANSFORM_OT_translate={"value": (
                    (random.random()-0.5)*0.5, # X 偏移
                    (random.random()-0.5)*0.5, # Y 偏移
                    1.5 * scale                # Z 向上长
                )}
            )
            # 缩放 (树梢变细)
            bpy.ops.transform.resize(value=(0.8, 0.8, 0.8))
            
            # 随机做一个分叉 (简单的复制面并旋转)
            if i == 2 or i == 3: # 在中间分叉
                # 再次挤出 creating a branch
                bpy.ops.mesh.extrude_region_move(
                    TRANSFORM_OT_translate={"value": (
                        (random.random()-0.5) * 2.0, 
                        (random.random()-0.5) * 2.0, 
                        1.0 * scale
                    )}
                )
                # 此时选中的是分叉末端，我们需要选回主干稍微麻烦
                # 为了简化，我们只做单根扭曲的枯木，氛围感到位即可
                # 或者：只做主干扭曲，不做分叉，避免拓扑错误
                
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # 3. 材质
        mat_bark = self.get_material("DeadBark", (0.05, 0.02, 0.01), alpha=1.0)
        # 稍微粗糙一点
        if mat_bark.node_tree.nodes.get('Principled BSDF'):
            mat_bark.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.9
            
        tree.data.materials.append(mat_bark)
        
        return tree

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

     # === [关键修复：提升渲染质量] ===
    scn = bpy.context.scene
    
    # 1. 必须开启 Cycles 渲染器 (体积光在 Eevee 下表现不同，论文图通常用 Cycles)
    scn.render.engine = 'CYCLES'
    
    # 2. 启用 GPU (如果可用)
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA' # 或 'OPTIX'
        prefs.get_devices()
    except:
        pass
    scn.cycles.device = 'GPU'

    # 3. 【核心】开启降噪 (Denoising)
    # 这能把噪点抹平，是解决体积光噪点的神器
    scn.cycles.use_denoising = True 
    # 如果有 OptiX (N卡)，可以用 'OPTIX'，否则用 'OPENIMAGEDENOISE' (Intel，CPU跑但质量好)
    scn.cycles.denoiser = 'OPENIMAGEDENOISE' 

    # 4. 增加采样数 (Samples)
    # 默认可能只有 32 或 64，体积光至少需要 128 或 256
    scn.cycles.samples = 128 
    
    # 5. 调整体积光参数 (降低计算压力)
    # 减少体积光的反弹次数，虽然真实感稍微降低，但噪点会少很多
    scn.cycles.max_bounces = 4
    scn.cycles.volume_bounces = 2 

    # scene_name = "DiAiTianhuaban"
    if hint is not None:
        print("hint is not None!")
        hint = hint[..., [2, 0, 1]]
        decorator = SceneDecorator(scene_cfg = cfg)
        decorator.enhance_scene(hint, data)

    # Number of frames possible to render
    nframes = len(data)

    # Show the trajectory
    if trajectory is not None:
        show_trajectory(data.trajectory)

    # initialize the camera
    camera = Camera(first_root=data.get_root(0), mode=mode, scene_name=cfg.scene_name)

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
