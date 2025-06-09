# Note：For our new features

# 一、原来的指令
对应的机器是：Stage1_Final，需要到这个目录下：`/root/autodl-tmp/MyRepository/MCM-LDM`
- 1.首先，切换到main分支：git checkout main
- 2.跑demo的指令：`python demo_transfer.py --cfg ./configs/config_mld_humanml3d.yaml --cfg_assets ./configs/assets.yaml --style_motion_dir demo/style_motion --content_motion_dir demo/content_motion --scale 2.5`
- 3.跑train的指令：`python -m train --cfg configs/config_mld_humanml3d.yaml --cfg_assets configs/assets.yaml --batch_size 128 --nodebug`