# Note：For our new features

# 一、原来的指令
对应的机器是：Stage1_Final，需要到这个目录下：`/root/autodl-tmp/MyRepository/MCM-LDM`
- 1.首先，切换到main分支：git checkout main
- 2.跑demo的指令：`python demo_transfer.py --cfg ./configs/config_mld_humanml3d.yaml --cfg_assets ./configs/assets.yaml --style_motion_dir demo/style_motion --content_motion_dir demo/content_motion --scale 2.5`
- 3.跑train的指令：`python -m train --cfg configs/config_mld_humanml3d.yaml --cfg_assets configs/assets.yaml --batch_size 128 --nodebug`

# 二、新的指令
对应的机器是：Stage1_Final3，新增的与DCE模块训练相关的代码都在这个路径下面：`/root/autodl-tmp/MyRepository/MCM-LDM/AddingCodes/DCE_Module`
- 1.训练Cstyle分类器：`python /root/autodl-tmp/MyRepository/MCM-LDM/AddingCodes/DCE_Module/train_style_classifier.py`
- 2.训练DCE模块：`python /root/autodl-tmp/MyRepository/MCM-LDM/AddingCodes/DCE_Module/train_dce.py`