import bpy


# def setup_renderer(denoising=True, oldrender=True, accelerator="gpu", device=[0]):
#     bpy.context.scene.render.engine = "CYCLES"
#     bpy.data.scenes[0].render.engine = "CYCLES"
#     if accelerator.lower() == "gpu":
#         bpy.context.preferences.addons[
#             "cycles"
#         ].preferences.compute_device_type = "CUDA"
#         bpy.context.scene.cycles.device = "GPU"
#         i = 0
#         bpy.context.preferences.addons["cycles"].preferences.get_devices()
#         for d in bpy.context.preferences.addons["cycles"].preferences.devices:
#             if i in device:  # gpu id
#                 d["use"] = 1
#                 print(d["name"], "".join(str(i) for i in device))
#             else:
#                 d["use"] = 0
#             i += 1

#     if denoising:
#         bpy.context.scene.cycles.use_denoising = True

#     bpy.context.scene.render.tile_x = 256
#     bpy.context.scene.render.tile_y = 256
#     bpy.context.scene.cycles.samples = 64

#     if not oldrender:
#         bpy.context.scene.view_settings.view_transform = "Standard"
#         bpy.context.scene.render.film_transparent = True
#         bpy.context.scene.display_settings.display_device = "sRGB"
#         bpy.context.scene.view_settings.gamma = 1.2
#         bpy.context.scene.view_settings.exposure = -0.75

def setup_renderer(denoising=True, oldrender=True, accelerator="gpu", device=[0]):
    scn = bpy.context.scene
    scn.render.engine = "CYCLES"
    c = scn.cycles
    
    # 1. 【核心】强制使用 OptiX (3090/4090 提速关键)
    if accelerator.lower() == "gpu":
        cprefs = bpy.context.preferences.addons["cycles"].preferences
        try:
            # 3090 支持 OptiX，这比 CUDA 快很多
            cprefs.compute_device_type = "OPTIX"
            cprefs.get_devices()
            print("🚀 Switched to OptiX backend.")
        except:
            print("⚠️ OptiX not available, falling back to CUDA.")
            cprefs.compute_device_type = "CUDA"
            cprefs.get_devices()

        # 激活显卡
        for i, dev in enumerate(cprefs.devices):
            if i in device:
                dev.use = True
                print(f"✅ Active GPU: {dev.name}")
            else:
                dev.use = False
        
        c.device = "GPU"

    # 2. 【核心】降低体积光计算量 (性能杀手)
    # 默认 Step Rate 是 1.0，太密了。改成 3.0 对暴风雪这种雾气几乎看不出区别，但速度快3倍。
    c.volume_step_rate = 3.0  
    # 限制体积光反弹，设为 0 (只照亮一次，不计算雾气内部反弹)
    c.volume_bounces = 0      
    
    # 3. 限制其他光程
    c.max_bounces = 6         # 总反弹降到 6
    c.transparent_max_bounces = 16 # 透明材质(雪花)需要多一点
    
    # 4. 降噪
    if denoising:
        c.use_denoising = True
        try:
            # 3090 同样支持 OptiX 降噪，速度极快
            c.denoiser = 'OPTIX' 
        except:
            c.denoiser = 'OPENIMAGEDENOISE'

    # 5. 分块大小 (Tile Size)
    # 3090 显存大 (24G)，可以直接渲染大块。
    # 设为 2048 可以减少 CPU-GPU 交互开销
    scn.render.tile_x = 2048
    scn.render.tile_y = 2048
    
    # 6. 采样数
    # 有了降噪，128 足够了。如果还是慢，降到 64 (配合降噪也能看)
    c.samples = 128 

    # 色彩管理
    if not oldrender:
        scn.view_settings.view_transform = "Standard"
        scn.render.film_transparent = True
        scn.display_settings.display_device = "sRGB"
        scn.view_settings.gamma = 1.2
        scn.view_settings.exposure = -0.75


# Setup scene
def setup_scene(
        res="high", denoising=True, oldrender=True, accelerator="gpu", device=[0]
):
    scene = bpy.data.scenes["Scene"]
    assert res in ["ultra", "high", "med", "low"]
    if res == "high":
        scene.render.resolution_x = 1280
        scene.render.resolution_y = 1024
    elif res == "med":
        scene.render.resolution_x = 1280 // 2
        scene.render.resolution_y = 1024 // 2
    elif res == "low":
        scene.render.resolution_x = 1280 // 4
        scene.render.resolution_y = 1024 // 4
    elif res == "ultra":
        scene.render.resolution_x = 1280 * 2
        scene.render.resolution_y = 1024 * 2

    scene.render.film_transparent = True
    world = bpy.data.worlds["World"]
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value[:3] = (1.0, 1.0, 1.0)
    bg.inputs[1].default_value = 1.0

    # Remove default cube
    if "Cube" in bpy.data.objects:
        bpy.data.objects["Cube"].select_set(True)
        bpy.ops.object.delete()

    bpy.ops.object.light_add(
        type="SUN", align="WORLD", location=(0, 0, 0), scale=(1, 1, 1)
    )
    bpy.data.objects["Sun"].data.energy = 1.5

    # rotate camera
    bpy.ops.object.empty_add(
        type="PLAIN_AXES", align="WORLD", location=(0, 0, 0), scale=(1, 1, 1)
    )
    bpy.ops.transform.resize(
        value=(10, 10, 10),
        orient_type="GLOBAL",
        orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        orient_matrix_type="GLOBAL",
        mirror=True,
        use_proportional_edit=False,
        proportional_edit_falloff="SMOOTH",
        proportional_size=1,
        use_proportional_connected=False,
        use_proportional_projected=False,
    )
    bpy.ops.object.select_all(action="DESELECT")

    setup_renderer(
        denoising=denoising, oldrender=oldrender, accelerator=accelerator, device=device
    )
    return scene
