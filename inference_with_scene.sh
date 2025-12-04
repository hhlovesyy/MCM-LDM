# inference_with_scene.sh (示例)
#!/bin/bash

PYTHON_CMD="python demo_transfer_with_scene.py"
CFG_ARGS="--cfg configs/config_mld_humanml3d_with_scene.yaml"
ASSETS_ARGS="--cfg_assets configs/assets.yaml"
MOTION_ARGS="--content_motion_dir demo/content_motion --style_motion_dir demo/style_motion"
SCALE_ARGS="--scale 12.5"

# 执行命令
$PYTHON_CMD $CFG_ARGS $ASSETS_ARGS $MOTION_ARGS $SCALE_ARGS