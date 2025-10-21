# tests/test_data_pipeline.py (Absolutely Complete and Runnable Version)

import pytest
import torch
import numpy as np
import os
import sys
from omegaconf import OmegaConf
import logging
from torch.utils.data import DataLoader

# ===================================================================
# 1. 设置 Python 路径，确保可以从测试脚本中导入 'mld' 包
# ===================================================================
# 将项目根目录（'tests'文件夹的父目录）添加到 sys.path
# 这使得 'import mld.data...' 这样的语句能够正常工作
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ===================================================================
# 2. 导入所有必需的模块和函数
# ===================================================================
# 导入我们自己编写的、需要被测试的模块
from mld.data.get_data import get_datasets

# 假设 mld_collate 和 collate_tensors 位于 mld/data/utils.py
# 如果它们在别处，需要修改这里的导入路径
# 为了让这个测试脚本自包含，我们将在这里定义一个 collate_tensors 的 mock 实现
def collate_tensors(batch):
    """一个简单的 collate 函数，用于堆叠已经填充好的张量。
    在真实代码中，这会处理填充。在测试中，我们的数据长度各不相同，
    所以我们需要一个能处理填充的版本。
    """
    dims = batch[0].dim()
    max_size = [0] * dims
    for b in batch:
        for i in range(dims):
            max_size[i] = max(max_size[i], b.size(i))
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.copy_(b)
    return canvas

# 将我们自己实现的 collate_tensors 注入到 mld.data.utils 模块中，
# 以便 mld_collate 可以找到它。这是一个测试技巧。
import mld.data.utils
mld.data.utils.collate_tensors = collate_tensors

# ===================================================================
# 3. 配置日志系统
# ===================================================================
LOG_FILE = "test_data_pipeline.log"
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

# 配置根 logger，使其同时输出到文件和控制台
logging.basicConfig(
    level=logging.INFO, # 设置为 INFO，避免过多的 DEBUG 信息刷屏
    format='%(asctime)s [%(levelname)s] [%(name)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout) # 确保日志输出到标准输出
    ]
)
logger = logging.getLogger(__name__)

# ===================================================================
# 4. Pytest Fixture: 创建隔离的测试环境
# ===================================================================

@pytest.fixture(scope="module")
def mock_data_and_config(tmp_path_factory):
    """
    一个 pytest fixture，它只在整个测试会话中运行一次，
    用于创建所有必需的虚拟数据文件和配置对象。
    """
    # tmp_path_factory 是 pytest 提供的内置 fixture，用于创建临时目录
    tmp_path = tmp_path_factory.mktemp("mock_data_root")
    logger.info(f"Creating mock data and config in temporary directory: {tmp_path}")

    # --- 1. 创建虚拟数据目录结构 ---
    humanml_dir = tmp_path / "humanml3d"
    style100_dir = tmp_path / "100style"
    for d in [humanml_dir, style100_dir]:
        d.mkdir()
        (d / "new_joint_vecs").mkdir()
        (d / "texts").mkdir()

    # --- 2. 创建 HumanML3D 的虚拟文件 ---
    logger.info("Creating mock HumanML3D files...")
    np.save(str(humanml_dir / "Mean.npy"), np.random.randn(263).astype(np.float32))
    np.save(str(humanml_dir / "Std.npy"), np.ones(263).astype(np.float32))
    
    with open(str(humanml_dir / "train.txt"), "w") as f:
        f.write("hml_train_01\nhml_train_02\n")
    with open(str(humanml_dir / "val.txt"), "w") as f:
        f.write("hml_val_01\nhml_val_02\n")
    
    # 创建动作和文本文件 (确保长度在过滤范围内)
    np.save(str(humanml_dir / "new_joint_vecs" / "hml_train_01.npy"), np.random.randn(150, 263).astype(np.float32))
    np.save(str(humanml_dir / "new_joint_vecs" / "hml_train_02.npy"), np.random.randn(180, 263).astype(np.float32))
    np.save(str(humanml_dir / "new_joint_vecs" / "hml_val_01.npy"), np.random.randn(160, 263).astype(np.float32))
    np.save(str(humanml_dir / "new_joint_vecs" / "hml_val_02.npy"), np.random.randn(170, 263).astype(np.float32))
    
    with open(str(humanml_dir / "texts" / "hml_train_01.txt"), "w") as f: f.write("a person is walking#a/DET person/NOUN is/VERB walking/VERB#0.0#0.0\n")
    with open(str(humanml_dir / "texts" / "hml_train_02.txt"), "w") as f: f.write("the character runs forward#the/DET character/NOUN runs/VERB forward/ADV#0.0#0.0\n")
    with open(str(humanml_dir / "texts" / "hml_val_01.txt"), "w") as f: f.write("someone is jumping#someone/NOUN is/VERB jumping/VERB#0.0#0.0\n")
    with open(str(humanml_dir / "texts" / "hml_val_02.txt"), "w") as f: f.write("a person sits down#a/DET person/NOUN sits/VERB down/ADV#0.0#0.0\n")

    # --- 3. 创建 100Style 的虚拟文件 ---
    logger.info("Creating mock 100Style files...")
    np.save(str(style100_dir / "Mean.npy"), np.random.randn(263).astype(np.float32))
    np.save(str(style100_dir / "Std.npy"), np.ones(263).astype(np.float32))
    
    with open(str(style100_dir / "train.txt"), "w") as f:
        f.write("style_train_01\n")
    with open(str(style100_dir / "Style_name_dict.txt"), "w") as f:
        f.write("style_train_01 Proud_SomeAction.bvh 0\n")
    
    np.save(str(style100_dir / "new_joint_vecs" / "style_train_01.npy"), np.random.randn(250, 263).astype(np.float32))
    with open(str(style100_dir / "texts" / "style_train_01.txt"), "w") as f:
        f.write("a person is moving#a/DET person/NOUN is/VERB moving/VERB#0.0#0.0\n")

    # --- 4. 创建用于测试的虚拟 OmegaConf 配置对象 ---
    logger.info("Creating mock OmegaConf config object...")
    config_dict = {
        'NAME': 'test_run',
        'DEBUG': False,
        'ACCELERATOR': 'cpu',
        'DEVICE': [0],
        'TRAIN': {
            'STAGE': 'diffusion',
            'DATASETS': ['mixed'],
            'NUM_WORKERS': 0,
            'BATCH_SIZE': 3,
            'END_EPOCH': 1,
            'PRETRAINED': '',
            'PRETRAINED_VAE': '',
        },
        'EVAL': {
            'DATASETS': ['humanml3d'],
            'BATCH_SIZE': 2,
            'SPLIT': 'val'
        },
        'DATASET': {
            'JOINT_TYPE': 'humanml3d',
            'NFEATS': 263, # 预设一个值
            'NJOINTS': 22,
            'TYPE': 'mixed',
            'WORD_VERTILIZER_PATH': str(tmp_path), # 指向一个存在的目录，尽管我们用 mock 替代
            'MIXED': {
                'BATCH_RATIO': [2, 1] # 2 个 HumanML3D, 1 个 100Style
            },
            'HUMANML3D': {
                'ROOT': str(humanml_dir),
                'SAMPLER': {'MAX_LEN': 196, 'MIN_LEN': 40, 'MAX_TEXT_LEN': 20},
                'UNIT_LEN': 4,
                'MIN_FILTER_LEN': 40,
                'MAX_FILTER_LEN': 200,
            },
            'STYLE100': {
                'ROOT': str(style100_dir),
                'SAMPLER': {'MAX_LEN': 196, 'MIN_LEN': 40, 'MAX_TEXT_LEN': 20},
                'UNIT_LEN': 4,
                'MIN_FILTER_LEN': 40,
                'MAX_FILTER_LEN': 300,
            }
        },
        'model': {
            't2m_path': '/root/autodl-tmp/MyRepository/MCM-LDM/deps/t2m/t2m' # 提供一个假的路径
        }
    }
    cfg = OmegaConf.create(config_dict)
    
    # fixture 会将这个配置对象 yield 给测试函数
    yield cfg

@pytest.fixture(autouse=True)
def mock_word_vectorizer(monkeypatch):
    """
    一个自动使用的 fixture，用于在整个测试期间用 MockVectorizer 替换真实的 WordVectorizer。
    autouse=True 表示所有测试函数都会自动使用这个 fixture，无需显式声明。
    monkeypatch 是 pytest 的内置 fixture，用于安全地修改类、方法等。
    """
    class MockVectorizer:
        def __init__(self, path, vab_name):
            logger.info(f"MockVectorizer initialized with path='{path}', vab_name='{vab_name}'")
        
        def __getitem__(self, key):
            # 返回形状正确的、类型正确的随机 numpy 数组
            word_emb = np.random.randn(300).astype(np.float32)
            pos_oh = np.random.randn(52).astype(np.float32)
            return word_emb, pos_oh
            
    # 使用 monkeypatch 来替换 mld.data.humanml.utils.word_vectorizer.WordVectorizer
    # 注意，路径是相对于调用者的，所以是 'mld.data...'
    monkeypatch.setattr("mld.data.humanml.utils.word_vectorizer.WordVectorizer", MockVectorizer)
    logger.info("Replaced 'mld.data.humanml.utils.word_vectorizer.WordVectorizer' with MockVectorizer.")


# ===================================================================
# 5. 测试用例
# ===================================================================

def test_full_data_pipeline_integration(mock_data_and_config):
    """
    一个完整的集成测试，覆盖从配置到批次产出的整个数据管道。
    """
    logger.info("--- Running Test: test_full_data_pipeline_integration ---")
    cfg = mock_data_and_config

    # --- Step 1, 2 (保持不变) ---
    logger.info("Step 1: Calling get_datasets factory to get the DataModule...")
    datamodules = get_datasets(cfg, logger, phase="train")
    dm = datamodules[0]
    logger.info("Step 1 PASSED: get_datasets factory works as expected.")

    logger.info("Step 2: Testing train_dataloader...")
    train_loader = dm.train_dataloader()
    assert len(train_loader) == 1
    logger.info("Step 2 PASSED: train_dataloader created with correct length.")

    # --- Step 3: 验证训练批次 (train batch) 的结构和内容 ---
    logger.info("Step 3: Fetching and verifying the training batch...")
    batch = next(iter(train_loader))

    expected_keys = ['motion', 'text', 'length', 'word_embs', 'pos_ohot', 'text_len', 'tokens']
    for key in expected_keys:
        assert key in batch, f"Batch is missing the expected key '{key}'."
    logger.info(f"Batch contains all expected keys: {list(batch.keys())}")

    # --- [核心修复] 修正单元测试的断言 ---
    logger.info("Asserting batch tensor shapes based on correct collate_fn output...")
    
    # mld_collate 的输出形状是 (BatchSize, SeqLen, FeatureDim)
    # 所以 batch['motion'].shape 应该是 (3, 196, 263)
    
    motion_shape = batch['motion'].shape
    assert isinstance(batch['motion'], torch.Tensor)
    
    assert motion_shape[0] == 3, \
        f"Batch size (dim 0) should be 3, but got {motion_shape[0]}."
        
    # SeqLen (dim 1) 会被填充到批内最大值，我们只检查它存在
    assert motion_shape[1] > 0, \
        f"Sequence length (dim 1) should be > 0, but got {motion_shape[1]}."
        
    # FeatureDim (dim 2) 必须是 263
    assert motion_shape[2] == 263, \
        f"Feature dimension (dim 2) should be 263, but got {motion_shape[2]}."

    assert isinstance(batch['length'], list) and len(batch['length']) == 3

    logger.info(f"Step 3 PASSED: Training batch has correct structure and shape {motion_shape}.")
    
    # --- Step 4 (保持不变) ---
    logger.info("Step 4: Testing val_dataloader...")
    val_loader = dm.val_dataloader()
    val_batch = next(iter(val_loader))
    assert val_batch['motion'].shape[0] == 2
    # 验证验证集批次的维度顺序也符合 collate_fn 的输出
    assert val_batch['motion'].shape[2] == 263
    logger.info("Step 4 PASSED: val_dataloader works correctly.")

    logger.info("\n[SUCCESS] All data pipeline tests passed!")