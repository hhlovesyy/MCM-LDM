# Note：For our new features

# 一、原来的指令
对应的机器是：Stage1_Final，需要到这个目录下：`/root/autodl-tmp/MyRepository/MCM-LDM`
- 1.首先，切换到main分支：git checkout main
- 2.跑demo的指令：`python demo_transfer.py --cfg ./configs/config_mld_humanml3d.yaml --cfg_assets ./configs/assets.yaml --style_motion_dir demo/style_motion --content_motion_dir demo/content_motion --scale 2.5`
- 3.跑train的指令：`python -m train --cfg configs/config_mld_humanml3d.yaml --cfg_assets configs/assets.yaml --batch_size 128 --nodebug`

# 三、数据集相关的指令
需要配一下VIBE的环境，这一点后面看看能否优化一下，比如用比较新的一些单目相机动捕方案。对于VIBE的环境来说，需要在`/root/autodl-tmp/MyRepository/MCM-LDM/AddingCodes/Dataset_process`这个文件夹下面运行`git clone https://github.com/mkocabas/VIBE.git`，这里有做.gitignore的操作，因此不会传到github上。需要在VIBE/lib文件夹下面加一个空的`__init__.py`文件.

跑数据集处理的流程需要下面的指令：
```PY
python /root/autodl-tmp/MyRepository/MCM-LDM/AddingCodes/Dataset_process/pipeline_video_to_action.py --input_folder "/root/autodl-tmp/MyRepository/MCM-LDM/dataset_process_demo" --smpl_models_dir "/root/autodl-tmp/MyRepository/MCM-LDM/deps/smpl_models"  --output_root_folder "/root/autodl-tmp/MyRepository/MCM-LDM/dataset_process_demo/outputs" --render_smpl
    # 可选参数可以放在最后，或者如果想注释掉中间的，要小心处理续行符
    # --render_vibe
    # --target_fps 30
```