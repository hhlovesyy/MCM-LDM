# tests/test_data_pipeline.py

import pytest
import torch
import numpy as np
import os
import yaml
from omegaconf import OmegaConf
import logging
from os.path import join as pjoin
import sys
from torch.utils.data import DataLoader

# 将项目根目录添加到 python 路径，以便导入 mld
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mld.data.mixed_datamodule import MixedDataModule

# --- 配置日志记录器，将所有日志输出到文件 ---
# 创建一个专门的日志文件用于测试
LOG_FILE = "test_data_pipeline.log"
# 如果文件已存在，先删除，确保每次测试都是全新的日志
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

# 配置根 logger
logging.basicConfig(
    level=logging.DEBUG, # 记录所有级别的日志
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'), # 写入文件
        logging.StreamHandler() # 同时输出到控制台
    ]
)
logger = logging.getLogger(__name__)

# --- Mock 对象，用于在没有真实文件的情况下进行测试 ---
class MockVectorizer:
    """一个假的 WordVectorizer，避免对真实 glove 文件的依赖。"""
    def __getitem__(self, key):
        # 返回形状正确的随机数据
        return np.random.randn(300).astype(np.float32), np.random.randn(52).astype(np.float32)

@pytest.fixture(scope="module")
def mock_data_and_config(tmp_path_factory):
    """
    一个 pytest fixture，用于创建一次性的、隔离的虚拟数据集和配置文件。
    scope="module" 表示这个 fixture 在整个测试文件中只运行一次。
    """
    tmp_path = tmp_path_factory.mktemp("data")
    logger.info(f"Creating mock data and config in temporary directory: {tmp_path}")

    # --- 1. 创建虚拟数据目录和文件 ---
    humanml_dir = tmp_path / "humanml3d"
    style100_dir = tmp_path / "100style"
    for d in [humanml_dir, style100_dir]:
        d.mkdir()
        (d / "new_joint_vecs").mkdir()
        (d / "texts").mkdir()

    # --- 创建 HumanML3D 的虚拟文件 ---
    logger.info("Creating mock HumanML3D files...")
    np.save(humanml_dir / "Mean.npy", np.random.randn(263))
    np.save(humanml_dir / "Std.npy", np.ones(263))
    with open(humanml_dir / "train.txt", "w") as f:
        f.write("hml_train_01\nhml_train_02\n")
    with open(humanml_dir / "val.txt", "w") as f:
        # f.write("hml_val_01\n")
        f.write("hml_val_01\nhml_val_02\n") # 之前只有一个
    # 创建 train 和 val 对应的动作和文本文件
    np.save(humanml_dir / "new_joint_vecs" / "hml_train_01.npy", np.random.randn(150, 263))
    np.save(humanml_dir / "new_joint_vecs" / "hml_train_02.npy", np.random.randn(180, 263))
    np.save(humanml_dir / "new_joint_vecs" / "hml_val_01.npy", np.random.randn(160, 263))
    np.save(humanml_dir / "new_joint_vecs" / "hml_val_02.npy", np.random.randn(170, 263)) # 新增
    with open(humanml_dir / "texts" / "hml_train_01.txt", "w") as f: f.write("humanml train sample 1#humanml/NOUN train/NOUN sample/NOUN 1/NUM#0.0#0.0\n")
    with open(humanml_dir / "texts" / "hml_train_02.txt", "w") as f: f.write("humanml train sample 2#humanml/NOUN train/NOUN sample/NOUN 2/NUM#0.0#0.0\n")
    with open(humanml_dir / "texts" / "hml_val_01.txt", "w") as f: f.write("humanml val sample 1#humanml/NOUN val/NOUN sample/NOUN 1/NUM#0.0#0.0\n")
    with open(humanml_dir / "texts" / "hml_val_02.txt", "w") as f: f.write("humanml val sample 2#humanml/NOUN val/NOUN sample/NOUN 1/NUM#0.0#0.0\n") # 新增

    # --- 创建 100Style 的虚拟文件 ---
    logger.info("Creating mock 100Style files...")
    np.save(style100_dir / "Mean.npy", np.random.randn(263))
    np.save(style100_dir / "Std.npy", np.ones(263))
    with open(style100_dir / "train.txt", "w") as f:
        f.write("style_train_01\n")
    with open(style100_dir / "Style_name_dict.txt", "w") as f:
        f.write("style_train_01 Proud_Action.bvh 0\n")
    np.save(style100_dir / "new_joint_vecs" / "style_train_01.npy", np.random.randn(250, 263))
    with open(style100_dir / "texts" / "style_train_01.txt", "w") as f: f.write("style train sample 1#style/NOUN train/NOUN sample/NOUN 1/NUM#0.0#0.0\n")

    # --- 2. 创建用于测试的虚拟 YAML 配置 ---
    logger.info("Creating mock OmegaConf config object...")
    config_dict = {
        'DEBUG': True,
        'TRAIN': {'BATCH_SIZE': 4, 'NUM_WORKERS': 0},
        'EVAL': {'BATCH_SIZE': 2},
        'DATASET': {
            'TYPE': 'mixed',
            'WORD_VERTILIZER_PATH': 'dummy_path',
            'MIXED': {'BATCH_RATIO': [2, 2]}, # 1:1 比例，方便验证
            'HUMANML3D': {
                'ROOT': str(humanml_dir),
                'SAMPLER': {'MAX_LEN': 196, 'MIN_LEN': 40, 'MAX_TEXT_LEN': 20},
                'UNIT_LEN': 4,
                'MIN_FILTER_LEN': 40, 'MAX_FILTER_LEN': 200,
            },
            'STYLE100': {
                'ROOT': str(style100_dir),
                'SAMPLER': {'MAX_LEN': 196, 'MIN_LEN': 40, 'MAX_TEXT_LEN': 20},
                'UNIT_LEN': 4,
                'MIN_FILTER_LEN': 40, 'MAX_FILTER_LEN': 300,
            }
        }
    }
    cfg = OmegaConf.create(config_dict)
    
    return cfg, MockVectorizer()

# --- 测试用例 ---

def test_datamodule_setup(mock_data_and_config):
    """测试 MixedDataModule 能否成功初始化和 setup。"""
    logger.info("--- Running Test: test_datamodule_setup ---")
    cfg, mock_vectorizer = mock_data_and_config
    
    dm = MixedDataModule(cfg)
    dm.w_vectorizer = mock_vectorizer # 手动注入 mock vectorizer
    
    dm.setup(stage='fit')
    
    logger.info("Asserting dataset lengths...")
    assert len(dm.humanml3d_train) == 2, "HumanML3D train set size should be 2"
    assert len(dm.style100_train) == 1, "100Style train set size should be 1"
    assert len(dm.train_dataset) == 3, "Total concatenated dataset size should be 3"
    assert len(dm.val_dataset) == 2, "Validation dataset size should be 2"
    
    logger.info("Asserting feature dimension...")
    assert dm.nfeats == 263, f"nfeats should be 263, but got {dm.nfeats}"
    
    logger.info("=> PASSED: test_datamodule_setup")

def test_train_dataloader_and_batch(mock_data_and_config):
    """测试 train_dataloader 能否产出结构正确、比例正确的 batch。"""
    logger.info("--- Running Test: test_train_dataloader_and_batch ---")
    cfg, mock_vectorizer = mock_data_and_config
    
    dm = MixedDataModule(cfg)
    dm.w_vectorizer = mock_vectorizer
    dm.setup(stage='fit')
    
    train_loader = dm.train_dataloader()
    
    logger.info("Asserting dataloader is created...")
    assert isinstance(train_loader, DataLoader), "train_dataloader should return a DataLoader instance"
    
    # 因为总样本数(3)小于 batch_size(4)，所以 sampler 只会生成 0 个 batch
    # 我们需要手动调整 batch size 来进行测试
    cfg.TRAIN.BATCH_SIZE = 2
    cfg.DATASET.MIXED.BATCH_RATIO = [1, 1]
    dm.batch_size = 2
    train_loader_small_batch = dm.train_dataloader()
    
    logger.info(f"Fetching one batch from train_dataloader (batch size={dm.batch_size})...")
    batch = next(iter(train_loader_small_batch))
    
    logger.info(f"Batch keys: {batch.keys()}")
    expected_keys = ['motion', 'text', 'length', 'word_embs', 'pos_ohot', 'text_len', 'tokens']
    for key in expected_keys:
        assert key in batch, f"Batch should contain key '{key}'"
    
    logger.info("Asserting batch tensor shapes...")
    # (batch_size, seq_len, n_feats)
    assert batch['motion'].shape[0] == 2, "Batch size of motion tensor is incorrect"
    assert batch['motion'].shape[2] == 263, "Feature dimension of motion tensor is incorrect"
    
    # 验证 sampler 的比例 (由于样本太少，这里无法精确验证，但在真实数据上 sampler 会起作用)
    # 这是一个集成测试，验证了从 setup 到 dataloader 的整个流程
    logger.info("=> PASSED: test_train_dataloader_and_batch")

def test_val_dataloader(mock_data_and_config):
    """测试 val_dataloader 能否正常工作。"""
    logger.info("--- Running Test: test_val_dataloader ---")
    cfg, mock_vectorizer = mock_data_and_config
    
    dm = MixedDataModule(cfg)
    dm.w_vectorizer = mock_vectorizer
    dm.setup(stage='fit')
    
    val_loader = dm.val_dataloader()
    
    logger.info("Asserting val_dataloader is created...")
    assert isinstance(val_loader, DataLoader), "val_dataloader should return a DataLoader instance"
    
    logger.info("Fetching one batch from val_dataloader...")
    batch = next(iter(val_loader))
    
    logger.info(f"Validation batch size: {batch['motion'].shape[0]}")
    assert batch['motion'].shape[0] == cfg.EVAL.BATCH_SIZE, "Validation batch size is incorrect"
    
    logger.info("=> PASSED: test_val_dataloader")