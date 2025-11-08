# Eadro - End-to-End Troubleshooting Framework for Microservices

![Version](https://img.shields.io/badge/version-0.1.0-green.svg)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)

**Eadro** 是一个用于微服务故障检测和根因定位的端到端框架，发表于 ICSE 2023。该框架通过建模微服务之间的内部和外部依赖关系，融合日志、指标和链路追踪等多源数据，实现异常检测和根因定位。

## 🌟 主要特性

- **多源数据融合**: 整合日志、指标、分布式追踪数据
- **图神经网络**: 建模微服务拓扑和依赖关系
- **端到端训练**: 联合优化异常检测和根因定位
- **标准化数据接口**: 清晰的数据结构定义，便于适配不同数据集
- **现代 Python 实现**: 使用类型注解、dataclass 和最佳实践

## 📦 安装

### 使用 uv (推荐)

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆仓库
git clone https://github.com/FeiGSSS/Eadro.git
cd Eadro

# 使用 uv 安装依赖 (自动创建 Python 3.8 环境)
uv sync

# 激活环境
source .venv/bin/activate
```

### 使用 pip

```bash
pip install -e .
```

### 数据预处理依赖 (可选)

```bash
uv pip install -e ".[preprocess]"
```

## 🚀 快速开始

### 方法一：使用预处理后的数据 (推荐)

数据集下载: https://doi.org/10.5281/zenodo.7615393

```bash
# 1. 下载预处理后的数据
# 下载后解压到 ./chunks/ 目录

# 2. 直接训练
uv run python -m eadro.cli train \
  --dataset TT \
  --data_root ./chunks \
  --epochs 50 \
  --batch_size 256
```

### 方法二：从原始数据开始完整流程

如果你想从原始数据开始，可以使用我们的预处理工具：

#### 1. 数据预处理

```bash
# 预处理 Train Ticket 数据集
uv run python -m eadro.cli preprocess \
  --raw_data /path/to/TT_Dataset \
  --dataset_name TT \
  --output_root ./preprocessed \
  --benchmark TrainTicket \
  --chunk_length 60 \
  --test_ratio 0.2

# 预处理 Social Network 数据集
uv run python -m eadro.cli preprocess \
  --raw_data /path/to/SN_Dataset \
  --dataset_name SN \
  --output_root ./preprocessed \
  --benchmark SocialNetwork \
  --chunk_length 60 \
  --test_ratio 0.2
```

**预处理参数说明**:
- `--raw_data`: 原始数据集根目录
- `--dataset_name`: 数据集标识符 (如 TT, SN)
- `--output_root`: 预处理输出根目录
- `--benchmark`: 基准系统名称 (TrainTicket 或 SocialNetwork)
- `--chunk_length`: 时间窗口长度 (秒)，默认 60
- `--test_ratio`: 测试集比例，默认 0.2
- `--overlap_threshold`: 故障重叠标记阈值，默认 1

预处理完成后，会在 `{output_root}/chunks/{dataset_name}/` 生成以下文件：
- `metadata.json`: 数据集元信息 (节点数、特征维度、拓扑结构等)
- `chunk_train.pkl`: 训练集数据
- `chunk_test.pkl`: 测试集数据
- `chunks.pkl`: 完整数据集 (用于兼容旧版本)

#### 2. 训练模型

```bash
# 基础训练
uv run python -m eadro.cli train \
  --dataset TT \
  --data_root ./preprocessed/chunks \
  --epochs 50 \
  --batch_size 256

# 高级参数调整
uv run python -m eadro.cli train \
  --dataset TT \
  --data_root ./preprocessed/chunks \
  --result_dir ./results \
  --epochs 50 \
  --batch_size 256 \
  --lr 0.001 \
  --patience 10 \
  --fuse_dim 128 \
  --alpha 0.5 \
  --evaluation_epoch 5
```

**训练参数说明**:

基础参数:
- `--dataset`: 数据集名称 (必需)
- `--data_root`: 数据集根目录，默认 `./chunks`
- `--result_dir`: 结果保存目录，默认 `./result`
- `--epochs`: 训练轮数，默认 50
- `--batch_size`: 批大小，默认 256
- `--lr`: 学习率，默认 0.001
- `--gpu`: 是否使用 GPU，默认 True
- `--random_seed`: 随机种子，默认 42
- `--patience`: 早停耐心值，默认 10
- `--evaluation_epoch`: 每 N 轮评估一次，默认 10

模型架构参数:
- `--self_attn`: 是否使用自注意力，默认 True
- `--fuse_dim`: 融合层维度，默认 128
- `--alpha`: 检测/定位损失权重，默认 0.5
- `--log_dim`: 日志嵌入维度，默认 16
- `--locate_hiddens`: 定位头隐藏层大小，默认 [64]
- `--detect_hiddens`: 检测头隐藏层大小，默认 [64]
- `--trace_hiddens`: 追踪 TCN 隐藏层，默认 [64]
- `--metric_hiddens`: 指标 TCN 隐藏层，默认 [64]
- `--graph_hiddens`: 图 GAT 隐藏层，默认 [64]
- `--attn_head`: GAT 注意力头数，默认 4

#### 3. 查看训练结果

训练完成后，结果保存在 `{result_dir}/{hash_id}/` 目录下：

```bash
# 查看训练结果
ls -lh ./result/8f9bdc9f/

# 文件说明:
# - model.ckpt: 最佳模型权重
# - params.json: 训练参数配置
# - running.log: 训练日志
```

示例输出：
```
Test -- F1: 0.9563, Rec: 0.9991, Pre: 0.9169
        HR@1: 0.7360, NDCG@1: 0.7360
        HR@3: 0.9463, NDCG@3: 0.8622
        HR@5: 0.9762, NDCG@5: 0.8746
```

**指标说明**:
- **F1/Rec/Pre**: 故障检测的 F1 分数、召回率、精确率
- **HR@K**: Hit Rate，前 K 个预测中包含真实根因的比例
- **NDCG@K**: Normalized Discounted Cumulative Gain，排序质量指标

### 完整示例

```bash
# 1. 预处理数据
uv run python -m eadro.cli preprocess \
  --raw_data /mnt/jfs/Eadro/TT_Dataset/TT_Dataset \
  --dataset_name TT \
  --output_root ./preprocessed \
  --benchmark TrainTicket

# 2. 训练模型
uv run python -m eadro.cli train \
  --dataset TT \
  --data_root ./preprocessed/chunks \
  --epochs 50 \
  --batch_size 256 \
  --evaluation_epoch 5

# 3. 查看结果
cat ./result/{hash_id}/running.log
```

## 🔧 适配自定义数据集

### 数据转换流程

如需将自己的微服务数据集适配到 Eadro，请按以下步骤操作：

#### 1. 了解数据格式要求

查看 `src/eadro/data/converter_example.py` 了解如何将自定义数据集转换为 Eadro 格式：

```python
from eadro.data.schema import ChunkPayload, DatasetMetadata, EadroDataset
import numpy as np

# 定义数据集元信息
metadata = DatasetMetadata(
    node_count=10,           # 微服务节点数
    event_dim=50,            # 日志事件类型数
    metric_dim=8,            # 指标维度 (CPU, 内存等)
    trace_channels=2,        # 链路统计通道数
    chunk_length=60,         # 时间窗口长度
    edges=(src_nodes, dst_nodes)  # 服务依赖关系
)

# 创建数据块
chunk = ChunkPayload(
    chunk_id="chunk_001",
    logs=np.zeros((10, 50), dtype=np.float32),      # (nodes, event_dim)
    metrics=np.zeros((10, 60, 8), dtype=np.float32), # (nodes, time, metrics)
    traces=np.zeros((10, 60, 2), dtype=np.float32),  # (nodes, time, channels)
    culprit=3  # 根因节点索引，-1 表示正常
)
chunk.validate()  # 验证数据格式
```

#### 2. 实现自定义转换器

创建转换脚本，将原始数据转换为 `ChunkPayload` 列表：

```python
from pathlib import Path
from typing import List
import numpy as np
from eadro.data.schema import ChunkPayload, DatasetMetadata, DatasetSplit, EadroDataset
from eadro.data.io import save_preprocessed_dataset

def convert_my_dataset(raw_data_path: Path) -> EadroDataset:
    """将自定义数据集转换为 Eadro 格式"""
    
    # 1. 加载原始数据
    # ... 你的数据加载逻辑 ...
    
    # 2. 提取拓扑结构
    src_nodes = np.array([...], dtype=np.int64)  # 源节点索引
    dst_nodes = np.array([...], dtype=np.int64)  # 目标节点索引
    
    # 3. 创建元信息
    metadata = DatasetMetadata(
        node_count=num_nodes,
        event_dim=num_event_types,
        metric_dim=num_metrics,
        trace_channels=2,
        chunk_length=60,
        edges=(src_nodes, dst_nodes)
    )
    
    # 4. 创建数据块
    chunks: List[ChunkPayload] = []
    for window_data in sliding_windows:
        chunk = ChunkPayload(
            chunk_id=f"chunk_{idx}",
            logs=extract_log_features(window_data),      # (N, E)
            metrics=extract_metrics(window_data),         # (N, T, M)
            traces=extract_traces(window_data),           # (N, T, 2)
            culprit=get_root_cause(window_data)           # int or -1
        )
        chunk.validate()
        chunks.append(chunk)
    
    # 5. 划分训练/测试集
    split_point = int(len(chunks) * 0.8)
    train_chunks = chunks[:split_point]
    test_chunks = chunks[split_point:]
    
    # 6. 创建数据集
    dataset = EadroDataset(
        metadata=metadata,
        train=DatasetSplit(chunks=train_chunks),
        test=DatasetSplit(chunks=test_chunks)
    )
    
    return dataset

# 7. 保存数据集
dataset = convert_my_dataset(Path("/path/to/raw/data"))
save_preprocessed_dataset(dataset, Path("./my_dataset"))

# 8. 使用数据集训练
# uv run python -m eadro.cli train --dataset my_dataset --data_root ./
```

#### 3. 数据验证

转换完成后，验证数据格式：

```python
from eadro.data.io import load_preprocessed_dataset

# 加载并验证
dataset = load_preprocessed_dataset(Path("./my_dataset"))
dataset.validate()

print(f"节点数: {dataset.metadata.node_count}")
print(f"训练样本: {len(dataset.train.chunks)}")
print(f"测试样本: {len(dataset.test.chunks)}")
```

## 📊 数据格式说明

### 标准化中间数据结构

Eadro 定义了统一的数据格式 `ChunkPayload`，所有数据集需转换为此格式：

```python
@dataclass
class ChunkPayload:
    chunk_id: str                              # 时间窗口唯一标识
    logs: NDArray[float32]                     # 日志特征 (nodes, event_dim)
    metrics: NDArray[float32]                  # 指标序列 (nodes, time, metric_dim)
    traces: NDArray[float32]                   # 链路统计 (nodes, time, trace_channels)
    culprit: int                               # 根因节点索引 (-1 表示健康)
```

**维度说明**:
- `logs`: **(节点数, 事件维度)** - 每个节点在时间窗口内的日志事件聚合特征
- `metrics`: **(节点数, 时间步, 指标数)** - 每个节点的时序指标 (CPU、内存等)
- `traces`: **(节点数, 时间步, 通道数)** - 分布式追踪统计 (调用次数、响应时间)
- `culprit`: 故障节点的索引，健康窗口设为 `-1`

### 数据转换示例

参考 `src/eadro/data/converter_example.py` 中的详细示例：

```bash
# 运行转换示例
python -m eadro.data.converter_example
```

## 🏗️ 项目结构

```
.
├── src/eadro/                    # 重构后的现代代码库
│   ├── data/
│   │   ├── schema.py            # 数据结构定义 (ChunkPayload, DatasetMetadata)
│   │   ├── dataset.py           # PyTorch Dataset 和 DGL 批处理
│   │   ├── io.py                # 数据加载/保存
│   │   └── converter_example.py # 数据转换示例
│   ├── models/
│   │   └── network.py           # 神经网络模型 (Eadro)
│   ├── training/
│   │   └── trainer.py           # 训练循环和评估逻辑
│   ├── preprocessing/           # 数据预处理模块
│   │   ├── pipeline.py          # 端到端预处理流程
│   │   ├── chunk_aligner.py     # 多源数据对齐
│   │   ├── log_processor.py     # 日志模板提取 (Drain3)
│   │   ├── metrics_processor.py # 指标数据处理
│   │   ├── traces_processor.py  # 链路追踪处理
│   │   ├── records_processor.py # 故障记录处理
│   │   └── service_info.py      # 服务拓扑定义
│   ├── config.py                # 配置类 (TrainingConfig)
│   ├── utils.py                 # 工具函数
│   └── cli.py                   # 命令行接口 (preprocess, train)
├── codes/                        # 原始代码 (向后兼容)
│   ├── preprocess/              # 原始预处理脚本
│   ├── main.py
│   ├── model.py
│   └── base.py
├── pyproject.toml               # uv 项目配置
├── README_NEW.md                # 本文档
└── Readme.md                    # 原始文档
```

### 主要模块说明

- **data/**: 数据处理和标准化
  - `schema.py`: 定义 `ChunkPayload`、`DatasetMetadata`、`EadroDataset` 等核心数据结构
  - `dataset.py`: PyTorch `Dataset` 封装，支持 DGL 图批处理
  - `io.py`: 统一的数据加载/保存接口

- **preprocessing/**: 从原始数据到标准格式的完整流程
  - `pipeline.py`: 协调所有预处理步骤
  - `chunk_aligner.py`: 将日志、指标、追踪对齐到时间窗口
  - 各 processor: 处理不同数据源

- **models/**: 神经网络实现
  - `network.py`: Eadro 模型，包含所有编码器和融合层

- **training/**: 训练和评估
  - `trainer.py`: 完整的训练循环，支持早停、检查点保存等

## 🔬 模型架构

![Eadro Architecture](https://user-images.githubusercontent.com/49298462/217256928-f0d61857-678b-4456-a024-359326a2c45d.png)

**主要组件**:
1. **日志编码器**: 将日志事件转换为嵌入向量
2. **指标编码器**: 使用时序卷积网络 (TCN) 处理指标时间序列
3. **追踪编码器**: 使用 TCN 处理分布式追踪数据
4. **图编码器**: 使用图注意力网络 (GAT) 建模服务拓扑
5. **多源融合**: 融合所有数据源生成统一表示
6. **双任务头**: 
   - 检测头: 二分类判断是否有故障
   - 定位头: 多分类预测根因节点

## 📈 实验结果

在两个基准数据集上的性能：

### Train Ticket (TT) 数据集

- **数据规模**: 57,618 个时间窗口，27 个微服务节点，91.9% 故障率
- **预处理**: 60 秒时间窗口，整合日志 (32 事件类型)、指标 (7 维) 和链路追踪

**性能指标** (50 epochs):
```
F1: 0.9664, Rec: 0.9867, Pre: 0.9470
HR@1: 0.8822, NDCG@1: 0.8822
HR@3: 0.9463, NDCG@3: 0.8832
HR@5: 0.9762, NDCG@5: 0.8832
```

**单 epoch 快速测试** (1 epoch, 仅用于验证):
```
F1: 0.9563, Rec: 0.9991, Pre: 0.9169
HR@1: 0.7360, NDCG@1: 0.7360
HR@3: 0.9463, NDCG@3: 0.8622
HR@5: 0.9762, NDCG@5: 0.8746
```

### Social Network (SN) 数据集

- **数据规模**: 12 个微服务节点
- **预处理**: 60 秒时间窗口

**性能指标** (50 epochs):
```
F1: 0.9819, Rec: 0.9713, Pre: 0.9927
HR@1: 0.7887, NDCG@1: 0.7887
HR@3: 0.8013, NDCG@3: 0.7966
HR@5: 0.8025, NDCG@5: 0.7971
```

**指标说明**:
- **F1**: 故障检测的 F1 分数 (精确率和召回率的调和平均)
- **Rec** (Recall): 召回率，检测到的故障占实际故障的比例
- **Pre** (Precision): 精确率，检测为故障且确实是故障的比例
- **HR@K**: Hit Rate @ K，前 K 个预测中包含真实根因的比例
- **NDCG@K**: Normalized Discounted Cumulative Gain @ K，考虑排序位置的评价指标

## 🛠️ 开发

```bash
# 安装开发依赖
uv pip install -e ".[dev]"

# 代码格式化
black src/
isort src/

# 类型检查
mypy src/

# 运行测试
pytest
```

## ❓ 常见问题

### 1. 预处理失败怎么办？

**问题**: `FileNotFoundError` 或数据格式错误

**解决方案**:
```bash
# 检查原始数据结构
ls -la /path/to/raw/data/
# 应该包含: data/, no_fault/ 目录

# 检查预处理日志
uv run python -m eadro.cli preprocess ... 2>&1 | tee preprocess.log

# 验证中间输出
ls -lh ./preprocessed/parsed_data/TT/
ls -lh ./preprocessed/chunks/TT/
```

### 2. 训练时内存不足？

**问题**: CUDA out of memory 或 RAM 不足

**解决方案**:
```bash
# 减小批大小
uv run python -m eadro.cli train --batch_size 128  # 或更小

# 使用 CPU (如果 GPU 内存不足)
uv run python -m eadro.cli train --gpu false

# 检查数据加载设置
# 在 cli.py 中设置 pin_memory=False
```

### 3. 如何加载已训练的模型？

```python
import torch
from eadro.models.network import Eadro

# 加载模型
checkpoint = torch.load("./result/{hash_id}/model.ckpt")
model = Eadro(...)  # 使用相同的参数初始化
model.load_state_dict(checkpoint)
model.eval()

# 推理
with torch.no_grad():
    predictions = model(batch)
```

### 4. 数据格式验证失败？

**问题**: `ValueError: All feature tensors must share the same node dimension`

**解决方案**:
```python
# 检查数据维度
print(f"Logs shape: {chunk.logs.shape}")      # 应该是 (N, E)
print(f"Metrics shape: {chunk.metrics.shape}") # 应该是 (N, T, M)
print(f"Traces shape: {chunk.traces.shape}")   # 应该是 (N, T, 2)

# 确保节点数一致
assert chunk.logs.shape[0] == chunk.metrics.shape[0] == chunk.traces.shape[0]
```

### 5. 如何调整模型超参数？

推荐的超参数搜索空间：

```bash
# 学习率
--lr 0.0001 0.001 0.01

# 融合维度
--fuse_dim 64 128 256

# 损失权重 (检测 vs 定位)
--alpha 0.3 0.5 0.7

# 批大小
--batch_size 128 256 512

# 隐藏层大小
--graph_hiddens 32 64 128
--trace_hiddens 32 64 128
--metric_hiddens 32 64 128
```

## 📝 引用

如果您使用了 Eadro，请引用我们的论文：

```bibtex
@inproceedings{eadro2023,
  title={Eadro: An End-to-End Troubleshooting Framework for Microservices on Multi-source Data},
  booktitle={ICSE},
  year={2023}
}
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 📧 联系我们

有任何问题欢迎在 Issues 中留言 🍺
