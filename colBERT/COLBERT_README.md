# ColBERT 增强型密集检索模型

## 简介

ColBERT (Enhanced Dense Retrieval) 是一个基于 BERT 的密集检索模型，专为知识库问答系统设计。本实现采用了多种优化策略，显著提高了问答检索的准确性和效率，具有以下特点：

- **增强表示学习**：结合 CLS 表示和平均池化，捕获更全面的语义信息
- **多层次投影网络**：使用多层感知机替代简单线性投影，增强表示能力
- **复合损失函数**：组合对比学习、聚类和多样性损失，优化嵌入空间
- **高效索引结构**：集成 FAISS 库实现快速向量检索
- **混合重排序**：结合向量相似度和关键词匹配进行二阶段检索

## 安装依赖

```bash
pip install torch transformers tqdm numpy faiss-cpu
# 如果使用 GPU 加速，可以安装 faiss-gpu
# pip install faiss-gpu
```

## 使用方法

### 命令行参数

```bash
python predict_colbert.py [参数]
```

主要参数:

- `--docfile`: 文档文件路径，默认 "tmp/document_chunked_cleaned.jsonl"
- `--trainfile`: 训练文件路径，默认 "tmp/train_cleaned.jsonl"
- `--valfile`: 验证文件路径，默认 "tmp/val_predict_cleaned.jsonl"
- `--max_doc_length`: 文档最大长度，默认 384
- `--batch_size`: 批次大小，默认 8
- `--epochs`: 训练轮数，默认 5
- `--lr`: 学习率，默认 1e-5
- `--dim`: 向量维度，默认 128
- `--output`: 输出文件路径，默认 "tmp/colbert/predict.jsonl"
- `--load_corpus`: 要加载的语料库编码文件名
- `--notsave_corpus`: 设置该标志不保存语料库编码

### 数据格式

#### 文档文件 (document_chunked_cleaned.jsonl)

```json
{
  "document_id": 12345,
  "title": "文档标题",
  "chunks": [
    {"content": "这是第一个文本块的内容..."},
    {"content": "这是第二个文本块的内容..."}
  ]
}
```

#### 训练/验证文件 (train_cleaned.jsonl, val_predict_cleaned.jsonl)

```json
{
  "question": "这是一个问题?",
  "answer": "答案",
  "document_id": 12345
}
```

#### 输出文件格式 (predict.jsonl)

```json
{
  "question": "这是一个问题?",
  "answer": "答案",
  "document_id": [12345, 67890, ...],
  "scores": [0.95, 0.85, ...]
}
```

## 模型架构

模型采用了创新的架构设计：

```
┌─────────────┐
│  BERT 编码器  │
└──────┬──────┘
       │
┌──────┴──────┐
│  表示增强层   │ ← 结合 CLS + 平均池化
└──────┬──────┘
       │
┌──────┴──────┐
│  多层投影网络 │ ← 非线性变换 + Dropout
└──────┬──────┘
       │
┌──────┴──────┐
│  L2 归一化   │
└──────┬──────┘
       │
┌──────┴──────┐
│ 密集向量表示  │
└─────────────┘
```

## 优化策略

### 1. 表示学习优化

- **混合表示**：结合 CLS token 表示和全序列平均池化，捕获更丰富的语义信息
- **丰富特征提取**：多层网络结构替代简单线性投影
- **正则化**：增强的 Dropout 策略防止过拟合

### 2. 训练策略优化

- **复合损失函数**：
  - InfoNCE 对比损失：增强查询与文档匹配度
  - 聚类损失：促进相似问题嵌入的聚集
  - 多样性损失：优化嵌入空间的利用率
- **学习率策略**：余弦退火调度，避免局部最优

### 3. 检索优化

- **IVF 索引**：大型语料库使用 FAISS IVF 索引加速检索
- **二阶段检索**：
  1. 召回阶段：检索更多候选文档
  2. 重排序阶段：结合向量相似度和关键词匹配进行混合评分

## 文件结构

运行后生成的文件结构:

```
tmp/
├── colbert/
│   ├── model/
│   │   └── total/
│   │       └── colbert_model.pt      # 模型参数
│   ├── corpus_encodings/
│   │   └── total/                    # 语料库编码
│   ├── log/                          # 日志文件
│   ├── predict.jsonl                 # 预测结果
│   └── enhanced_retrieval.py         # 检索示例代码
├── vector_corpus/
│   └── total/                        # 语料库向量
│       ├── corpus_vectors.npy        # 向量数据
│       └── corpus_ids.json           # 向量ID映射
├── vector_train/
│   └── 384/                          # 训练查询向量
├── vector_val_predict/
│   └── 384/                          # 验证查询向量
└── faiss_index/
    └── total/                        # FAISS索引
        └── corpus_index.faiss        # 向量索引文件
```

## 性能提升技巧

1. **模型调优**：
   - 增大向量维度 (`--dim`) 可提高表示能力，但会增加计算开销
   - 增加 Dropout 率可减轻过拟合，对小数据集特别有效

2. **检索优化**：
   - 对于大型语料库 (>10K 文档)，使用 IVF 索引提高检索速度
   - 调整 `INITIAL_K` 参数控制候选文档数量，权衡效率和准确性

3. **重排序优化**：
   - 调整向量相似度与关键词匹配的权重比例（当前为 0.7:0.3）
   - 增加更复杂的重排序策略，如 BM25 或基于 BERT 的交互式重排序

## 实时检索应用示例

`enhanced_retrieval.py` 提供了一个完整的检索应用示例，演示如何:

1. 加载预训练的模型和索引
2. 将用户查询编码为向量
3. 使用 FAISS 执行高效检索
4. 结合向量相似度和关键词匹配进行重排序
5. 返回最相关的文档及其内容

## 常见问题

1. **内存错误**：
   - 减小 `batch_size` 参数
   - 降低 `max_doc_length` 参数
   - 考虑使用梯度累积技术

2. **检索准确率低**：
   - 增加训练轮数 (`--epochs`)
   - 尝试不同的学习率和权重衰减
   - 调整混合重排序的权重比例

3. **检索速度慢**：
   - 对于大型语料库，确保使用 IVF 索引
   - 调整 IVF 参数 (nlist, nprobe) 平衡速度和精度
   - 考虑使用 GPU 加速 FAISS (`faiss-gpu`) 