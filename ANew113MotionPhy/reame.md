
cd /root/autodl-tmp/MyRepository/MCM-LDM
跑推理：

python demo_physics.py --cfg ./configs/config_physimos_probe.yaml --cfg_assets ./configs/assets.yaml --content_motion_dir demo/content_motion --scale 2.5

跑训练：
python -m train --cfg configs/config_physimos_probe.yaml --cfg_assets configs/assets.yaml
# batch_size是16，在yaml中有记录

数据集分布，
(base) root@autodl-container-1421458302-b0bada58:~/autodl-tmp/MyRepository/MCM-LDM/ANew113MotionPhy/Tool# python check_dataset_distribution.py 
正在读取数据目录: /root/autodl-tmp/MyRepository/MCM-LDM/datasets/PhysicsDataset/json_files ...
找到 749 个样本。正在解析...

==============================
【分类统计报告】
总文件数: 749
  - LowCeiling: 275 个
  - NarrowGap: 306 个
  - Windy: 168 个
==============================

统计图表已保存至: dataset_distribution_v2.png

python visualize_zero_wind_v2.py --single /root/autodl-tmp/MyRepository/MCM-LDM/datasets/humanml3d/new_joint_vecs/000021.npy

/root/autodl-tmp/MyRepository/MCM-LDM/datasets/humanml3d/new_joint_vecs/000021.npy

python visualize_trusted.py --single /root/autodl-tmp/MyRepository/MCM-LDM/datasets/humanml3d/new_joint_vecs/000021.npy


python call_viz.py --single /root/autodl-tmp/MyRepository/MCM-LDM/datasets/humanml3d/new_joint_vecs/000021.npy

/root/autodl-tmp/MyRepository/MCM-LDM/datasets/PhysicsDataset/new_joint_vecs/W_0p1_FrontLeft_300k_0112.npy


python call_viz.py --single /root/autodl-tmp/MyRepository/MCM-LDM/datasets/PhysicsDataset/new_joint_vecs/W_0p1_FrontLeft_300k_0112.npy


W_0p1_BackRight_200k_0082

python call_viz.py --single /root/autodl-tmp/MyRepository/MCM-LDM/datasets/PhysicsDataset/new_joint_vecs/W_0p1_BackRight_200k_0082.npy