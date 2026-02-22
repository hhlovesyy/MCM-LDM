# pipelines.py
import os

class BasePipeline:
    """
    所有 Baseline 的抽象基类。
    定义了每个 Baseline 必须提供的信息和行为。
    """
    def __init__(self, name, label, config_template):
        self.name = name  # 内部唯一 ID, e.g., "ours_full"
        self.label = label # UI 上显示的名称, e.g., "我们的 (全量版本)"
        self.config_template = config_template # 模板 YAML 文件的路径

    def get_command(self, args: dict) -> str:
        """
        根据传入的参数，生成最终的 shell 命令行。
        这是一个模板方法，子类必须实现它。
        """
        raise NotImplementedError

class OursFullModelPipeline(BasePipeline):
    """我们的方法的 Pipeline 定义"""
    def get_command(self, args: dict) -> str:
        # **核心**：从 `args` 字典中获取所有需要的参数，构建命令行
        # 注意 --film_scalar 和 --trajectory_path 是我们新增的命令行参数
        # --exp_name 用于在输出目录中区分不同 baseline 的结果
        cmd = f"""python demo_transfer_with_scene.py \
            --cfg {args['temp_config_path']} \
            --cfg_assets {args['assets_cfg_path']} \
            --content_motion_dir {args['content_path']} \
            --style_motion_dir {args['style_path']} \
            --film_scalar {args['film_scalar']} \
            --trajectory_path {args['trajectory_path']} \
            --demo_out_dir {args['output_dir']} \
            --exp_name {self.name} \
            --render_video \
            --use_scene \
            --scale 2.5
        """
        # 使用 .strip() 和替换换行符来清理命令字符串
        return " ".join(cmd.strip().split())

class MCMLDMBaseline(BasePipeline):
    """我们的方法的 Pipeline 定义"""
    def get_command(self, args: dict) -> str:
        # **核心**：从 `args` 字典中获取所有需要的参数，构建命令行
        # 注意 --film_scalar 和 --trajectory_path 是我们新增的命令行参数
        # --exp_name 用于在输出目录中区分不同 baseline 的结果
        cmd = f"""python demo_transfer_with_scene.py \
            --cfg {args['temp_config_path']} \
            --cfg_assets {args['assets_cfg_path']} \
            --content_motion_dir {args['content_path']} \
            --style_motion_dir {args['style_path']} \
            --film_scalar {args['film_scalar']} \
            --trajectory_path {args['trajectory_path']} \
            --demo_out_dir {args['output_dir']} \
            --exp_name {self.name} \
            --render_video \
            --scale 2.5
        """
        # 使用 .strip() 和替换换行符来清理命令字符串
        return " ".join(cmd.strip().split())

class RenderTrajBlenderPipeline(BasePipeline):
    def get_command(self, args: dict) -> str:
        # blender -b -P script.py -- --input file.npy
        cmd = f"""blender -b -P python render_traj_tip.py \
            -- -- input {args['traj_render_dir']}
        """
        return "".join(cmd.strip().split())