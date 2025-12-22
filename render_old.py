import os
import pickle
import sys
from argparse import ArgumentParser

try:
    import bpy
    sys.path.append(os.path.dirname(bpy.data.filepath))
except ImportError:
    raise ImportError(
        "Blender is not properly installed or not launch properly. See README.md to have instruction on how to install and use blender.")

import mld.launch.blender  # noqa
from mld.render.blender import render


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
                trajectory = pkl['hint']
                print("trajectory", trajectory)
                if trajectory == None:
                    trajectory = pkl['ground_trajectory']
                # trajectory = None

        except FileNotFoundError:
            print(f"{path} not found")
            continue

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
