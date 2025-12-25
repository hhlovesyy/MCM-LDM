import bpy


class Camera:
    def __init__(self, first_root, mode, scene_name):
        self.scene_name = scene_name
        camera = bpy.data.objects['Camera']

        # 基础位置 (标准场景)
        base_x, base_y, base_z = 7.36, -6.93, 5.6
        
        # 默认焦距
        lens_val = 65

        # ================= [关键修改：针对大场景的适配] =================
        # 如果是独木桥或暴风雨，我们需要相机退得更远，视野更广
        is_large_scene = False
        if scene_name and any(s in scene_name for s in ["Dumuqiao", "Balance", "Dark"]): # , "BaoFengYu", "Storm"
            is_large_scene = True
            
        if is_large_scene:
            # 1. 拉远距离 (乘以一个系数，比如 1.4倍)
            scale_factor = 1.2 
            base_x *= scale_factor
            base_y *= scale_factor
            base_z *= scale_factor
            
            # 2. 广角镜头 (数值越小视野越宽)
            # 65 -> 45 (更广)
            if mode == "sequence":
                lens_val = 60 
        # ==========================================================

        # 应用位置
        camera.location.x = base_x
        camera.location.y = base_y
        camera.location.z = base_z

        # 应用焦距
        if mode == "sequence":
            camera.data.lens = lens_val
        elif mode == "frame":
            camera.data.lens = 130
        elif mode == "video":
            camera.data.lens = 110

        self.mode = mode
        self.camera = camera

        self.camera.location.x += first_root[0]
        self.camera.location.y += first_root[1]

        self._root = first_root

    def update(self, new_root):
        delta_root = new_root - self._root
        self.camera.location.x += delta_root[0]
        self.camera.location.y += delta_root[1]
        self._root = new_root
