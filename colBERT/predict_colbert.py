#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import argparse
import datetime
import faiss
from collections import defaultdict

# 解析命令行参数
parser = argparse.ArgumentParser(description='ColBERT模型训练与预测')
parser.add_argument('--load_corpus', type=str, help='要加载的语料库编码文件名', default=None)
parser.add_argument('--notsave_corpus', action='store_true', help='是否保存语料库编码')
parser.add_argument('--docfile', type=str, help='文档文件路径', default="tmp/document_chunked_cleaned.jsonl")
parser.add_argument('--trainfile', type=str, help='训练文件路径', default="tmp/train_cleaned.jsonl")
parser.add_argument('--valfile', type=str, help='验证文件路径', default="tmp/val_predict_cleaned.jsonl")
parser.add_argument('--max_doc_length', type=int, help='文档最大长度', default=384)
parser.add_argument('--batch_size', type=int, help='批次大小', default=8)
parser.add_argument('--neg_samples', type=int, help='负样本数量', default=7)
parser.add_argument('--epochs', type=int, help='训练轮数', default=5)
parser.add_argument('--lr', type=float, help='学习率', default=1e-5)
parser.add_argument('--dim', type=int, help='向量维度', default=128)
parser.add_argument('--hard_negatives', action='store_true', help='使用难负例采样')
parser.add_argument('--output', type=str, help='输出文件路径', default="tmp/colbert/predict.jsonl")
args = parser.parse_args()

# 设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 配置日志
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = "tmp/colbert/log"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"colbert_{timestamp}.log")

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 路径和参数配置
TRAIN_FILE = "tmp/train_cleaned.jsonl"
DOC_FILE = "tmp/document_chunked_cleaned.jsonl"
VAL_FILE = "tmp/val_predict_cleaned.jsonl"
OUTPUT_FILE = "tmp/colbert/predict.jsonl"
MODEL_DIR = "tmp/colbert/model/total"
CORPUS_ENCODING_DIR = "tmp/colbert/corpus_encodings/total"
VECTOR_CORPUS_DIR = "./tmp/vector_corpus/total"
FAISS_INDEX_DIR = "./tmp/faiss_index/total"
VECTOR_TRAIN_DIR = f"./tmp/vector_train/{args.max_doc_length}"
VECTOR_VAL_DIR = f"./tmp/vector_val_predict/{args.max_doc_length}"
BERT_MODEL = "bert-base-uncased"
MAX_QUERY_LENGTH = 64
MAX_DOC_LENGTH = args.max_doc_length
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
EPOCHS = args.epochs
VECTOR_DIM = args.dim
TOP_K = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 确保输出目录存在
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CORPUS_ENCODING_DIR, exist_ok=True)
os.makedirs(VECTOR_CORPUS_DIR, exist_ok=True)
os.makedirs(VECTOR_TRAIN_DIR, exist_ok=True)
os.makedirs(VECTOR_VAL_DIR, exist_ok=True)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# 输出参数配置
logger.info(f"参数配置: 文档最大长度={MAX_DOC_LENGTH}, 批次大小={BATCH_SIZE}, 向量维度={VECTOR_DIM}")
logger.info(f"参数配置: 训练轮数={EPOCHS}, 学习率={LEARNING_RATE}")

# 增强的ColBERT模型 - 专注于提高检索性能
class EnhancedDenseRetriever(nn.Module):
    def __init__(self, bert_model_name, vector_dim=VECTOR_DIM, dropout_rate=0.2):
        super(EnhancedDenseRetriever, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # 更强的dropout以提高泛化能力
        self.dropout = nn.Dropout(dropout_rate)
        
        # 更复杂的投影层 - 多层感知机
        self.projection = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.bert.config.hidden_size, vector_dim)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 更丰富的表示方法 - 结合[CLS]和平均池化
        cls_representation = outputs.last_hidden_state[:, 0, :]
        avg_representation = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1, keepdim=True)
        combined_representation = cls_representation + avg_representation
        
        # 应用dropout和投影
        combined_representation = self.dropout(combined_representation)
        projected = self.projection(combined_representation)
        
        # L2归一化以便于相似度计算
        normalized = F.normalize(projected, p=2, dim=1)
        return normalized

# 数据集类
class QueryDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=MAX_QUERY_LENGTH):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["question"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "idx": idx,
            "question": item["question"],
            "answer": item.get("answer", ""),
            "document_id": item.get("document_id", ""),
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze()
        }

class DocDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=MAX_DOC_LENGTH):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "idx": idx,
            "docid": item["docid"],
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze()
        }

# 加载jsonl文件
def load_jsonl(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    except Exception as e:
        logger.error(f"加载 {file_path} 失败: {e}")
        return []

# 分批加载大型jsonl文件
def load_jsonl_in_batches(file_path, batch_size=1000):
    count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            batch = []
            for line in tqdm(f, desc=f"加载 {file_path}"):
                if line.strip():
                    batch.append(json.loads(line))
                    count += 1
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
            if batch:
                yield batch
        logger.info(f"共加载了 {count} 条记录")
    except Exception as e:
        logger.error(f"加载 {file_path} 失败: {e}")
        yield []

# 保存jsonl文件
def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 保存向量和ID到文件
def save_vectors(vectors, ids, file_prefix):
    # 保存向量
    np.save(f"{file_prefix}_vectors.npy", vectors)
    
    # 检查ids的类型并相应处理
    processed_ids = []
    
    # 如果ids是简单的ID列表（如corpus_ids）
    if isinstance(ids, list) and (len(ids) == 0 or not isinstance(ids[0], dict)):
        # 直接保存ID列表
        with open(f"{file_prefix}_ids.json", 'w', encoding='utf-8') as f:
            # 确保所有元素都是JSON可序列化的
            serializable_ids = []
            for id_value in ids:
                if torch.is_tensor(id_value):
                    serializable_ids.append(id_value.item() if id_value.numel() == 1 else id_value.tolist())
                else:
                    serializable_ids.append(id_value)
            json.dump(serializable_ids, f, ensure_ascii=False)
    else:
        # 处理包含元数据的字典列表（如query_data）
        for item in ids:
            processed_item = {}
            for key, value in item.items():
                # 处理tensor或其他不可序列化的类型
                if torch.is_tensor(value):
                    processed_item[key] = value.item() if value.numel() == 1 else value.tolist()
                else:
                    processed_item[key] = value
            processed_ids.append(processed_item)
        
        # 保存处理后的ID列表
        with open(f"{file_prefix}_ids.json", 'w', encoding='utf-8') as f:
            json.dump(processed_ids, f, ensure_ascii=False)
    
    logger.info(f"向量和ID保存到 {file_prefix}")

# 加载向量和ID
def load_vectors(file_prefix):
    try:
        # 加载向量
        vectors = np.load(f"{file_prefix}_vectors.npy")
        
        # 加载ID列表
        with open(f"{file_prefix}_ids.json", 'r', encoding='utf-8') as f:
            ids = json.load(f)
        
        logger.info(f"从 {file_prefix} 加载了向量和ID")
        return vectors, ids
    except Exception as e:
        logger.error(f"加载向量和ID失败: {e}")
        return None, None

# 从document_chunked_cleaned.jsonl提取文档内容
def extract_documents_from_corpus(file_path):
    logger.info(f"从 {file_path} 提取文档内容")
    doc_data = []
    doc_id_to_text = {}
    
    # 分批处理大文件
    for batch in load_jsonl_in_batches(file_path):
        for doc in batch:
            if "document_id" not in doc:
                continue
                
            doc_id = doc["document_id"]
            
            # 提取文档内容
            content = ""
            if "title" in doc:
                content += doc["title"] + " "
                
            if "chunks" in doc and isinstance(doc["chunks"], list):
                for chunk in doc["chunks"]:
                    if "content" in chunk:
                        content += chunk["content"] + " "
            
            if content and doc_id not in doc_id_to_text:
                doc_id_to_text[doc_id] = content
                doc_data.append({
                    "docid": doc_id,
                    "text": content
                })
    
    logger.info(f"从语料库中提取了 {len(doc_data)} 个文档")
    return doc_data, doc_id_to_text

# 编码语料库文档为密集向量表示
def encode_corpus_documents(model, doc_data, tokenizer):
    logger.info("开始编码语料库文档...")
    
    # 检查是否要加载已有的语料库编码
    if args.load_corpus:
        load_path = os.path.join(CORPUS_ENCODING_DIR, args.load_corpus)
        if os.path.exists(load_path):
            corpus_embeddings, corpus_ids = load_vectors(load_path)
            if corpus_embeddings is not None and corpus_ids is not None:
                logger.info(f"成功加载语料库编码，共 {len(corpus_ids)} 个文档")
                return corpus_embeddings, corpus_ids
            else:
                logger.warning("加载语料库编码失败，将重新编码")
    
    # 创建数据加载器
    doc_dataset = DocDataset(doc_data, tokenizer)
    doc_loader = DataLoader(doc_dataset, batch_size=BATCH_SIZE)
    
    # 编码所有文档
    doc_embeddings = []
    doc_ids = []
    doc_texts = []  # 保存原始文本，便于后续检索
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(doc_loader, desc="编码语料库文档"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            
            embeddings = model(input_ids, attention_mask)
            doc_embeddings.append(embeddings.cpu().numpy())
            doc_ids.extend(batch["docid"])
            if "text" in batch:
                doc_texts.extend(batch["text"])
    
    # 将所有文档嵌入连接成一个数组
    corpus_embeddings = np.vstack(doc_embeddings)
    corpus_ids = [id.item() if torch.is_tensor(id) else id for id in doc_ids]
    logger.info(f"编码了 {len(corpus_ids)} 个语料库文档")
    
    # 保存语料库编码
    if not args.notsave_corpus:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(CORPUS_ENCODING_DIR, f"corpus_embeddings_{timestamp}")
        
        # 创建包含更多元数据的语料库向量保存格式
        corpus_metadata = []
        for i, doc_id in enumerate(corpus_ids):
            metadata = {
                "docid": doc_id,
                "text": doc_texts[i] if i < len(doc_texts) else ""
            }
            corpus_metadata.append(metadata)
        
        # 保存语料库向量和元数据
        save_vectors(corpus_embeddings, corpus_metadata, save_path)
        
        # 额外保存到vector_corpus目录，使用更丰富的元数据
        save_vectors(corpus_embeddings, corpus_metadata, os.path.join(VECTOR_CORPUS_DIR, "corpus"))
    
    return corpus_embeddings, corpus_ids

# 编码查询或问题为密集向量表示
def encode_queries(model, query_data, tokenizer, vector_dir, prefix):
    logger.info(f"开始编码查询...")
    
    # 创建数据加载器
    query_dataset = QueryDataset(query_data, tokenizer)
    query_loader = DataLoader(query_dataset, batch_size=BATCH_SIZE)
    
    # 编码所有查询
    query_embeddings = []
    questions = []
    answers = []
    doc_ids = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(query_loader, desc="编码查询"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            
            embeddings = model(input_ids, attention_mask)
            query_embeddings.append(embeddings.cpu().numpy())
            
            # 收集问题文本、答案和文档ID，确保转换为Python原生类型
            questions.extend([q for q in batch["question"]])
            
            # 处理答案，确保它们是Python原生类型
            for ans in batch["answer"]:
                if torch.is_tensor(ans):
                    answers.append(ans.item() if ans.numel() == 1 else ans.tolist())
                else:
                    answers.append(ans)
            
            # 处理文档ID，确保它们是Python原生类型
            for doc_id in batch["document_id"]:
                if torch.is_tensor(doc_id):
                    doc_ids.append(doc_id.item() if doc_id.numel() == 1 else doc_id.tolist())
                else:
                    doc_ids.append(doc_id)
    
    # 将所有查询嵌入连接成一个数组
    query_embeddings = np.vstack(query_embeddings)
    
    # 准备保存的数据
    query_data = []
    for i in range(len(questions)):
        query_data.append({
            "question": questions[i],
            "answer": answers[i] if i < len(answers) else "",
            "document_id": doc_ids[i] if i < len(doc_ids) else ""
        })
    
    # 保存查询向量和元数据
    save_vectors(query_embeddings, query_data, os.path.join(vector_dir, prefix))
    logger.info(f"编码了 {len(query_data)} 个查询")
    
    return query_embeddings, query_data

# 构建FAISS索引
def build_faiss_index(embeddings, save_path=None):
    logger.info("构建FAISS索引...")
    
    # 获取向量维度
    dimension = embeddings.shape[1]
    
    # 创建IndexFlatIP索引（内积=余弦相似度，对于归一化向量）
    index = faiss.IndexFlatIP(dimension)
    
    # 添加向量到索引
    index.add(embeddings)
    
    logger.info(f"FAISS索引已构建，包含 {index.ntotal} 个向量")
    
    # 如果提供了保存路径，保存索引
    if save_path:
        try:
            faiss.write_index(index, save_path)
            logger.info(f"FAISS索引已保存到 {save_path}")
        except Exception as e:
            logger.error(f"保存FAISS索引失败: {e}")
    
    return index

# 加载FAISS索引
def load_faiss_index(load_path):
    try:
        index = faiss.read_index(load_path)
        logger.info(f"从 {load_path} 加载了FAISS索引，包含 {index.ntotal} 个向量")
        return index
    except Exception as e:
        logger.error(f"加载FAISS索引失败: {e}")
        return None

# 使用FAISS进行向量检索，并返回更详细的结果
def retrieve_documents(index, query_embeddings, corpus_ids, corpus_docs=None, k=TOP_K):
    logger.info(f"使用FAISS检索相关文档...")
    
    # 执行搜索
    scores, indices = index.search(query_embeddings, k)
    
    # 整理检索结果
    results = []
    for i in range(len(query_embeddings)):
        doc_indices = indices[i]
        doc_scores = scores[i]
        
        retrieved_docs = []
        for j, idx in enumerate(doc_indices):
            if 0 <= idx < len(corpus_ids):
                doc_info = {
                    "docid": corpus_ids[idx],
                    "score": float(doc_scores[j])  # 转换为Python原生类型
                }
                
                # 如果有文档内容，添加到结果中
                if corpus_docs and idx < len(corpus_docs):
                    doc_info["text"] = corpus_docs[idx]["text"] if "text" in corpus_docs[idx] else ""
                
                retrieved_docs.append(doc_info)
        
        results.append(retrieved_docs)
    
    return results

# 函数来运行快速相似度计算和文档检索（便于后续使用）
def quick_retrieve(query, model, tokenizer, faiss_index, corpus_ids, corpus_docs=None, k=TOP_K):
    """
    快速检索函数 - 用于后续根据问题直接检索相关文档
    
    参数:
    - query: 查询文本
    - model: 编码模型
    - tokenizer: 分词器
    - faiss_index: FAISS索引
    - corpus_ids: 语料库ID列表
    - corpus_docs: 语料库文档内容
    - k: 返回的文档数量
    
    返回:
    - 检索到的文档列表，包含ID、相似度分数和文本内容
    """
    # 对查询进行编码
    encoding = tokenizer(
        query,
        max_length=MAX_QUERY_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        query_embedding = model(input_ids, attention_mask).cpu().numpy()
    
    # 执行搜索
    scores, indices = faiss_index.search(query_embedding, k)
    
    # 整理检索结果
    results = []
    for j, idx in enumerate(indices[0]):
        if 0 <= idx < len(corpus_ids):
            doc_info = {
                "docid": corpus_ids[idx],
                "score": float(scores[0][j])
            }
            
            # 如果有文档内容，添加到结果中
            if corpus_docs and idx < len(corpus_docs):
                doc_info["text"] = corpus_docs[idx]["text"] if "text" in corpus_docs[idx] else ""
            
            results.append(doc_info)
    
    return results

def main():
    start_time = time.time()
    logger.info(f"使用设备: {DEVICE}")
    logger.info(f"参数信息: 文档文件={DOC_FILE}, 训练文件={TRAIN_FILE}, 验证文件={VAL_FILE}")
    
    # 加载tokenizer
    logger.info(f"加载tokenizer: {BERT_MODEL}")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    
    # 初始化增强的模型
    logger.info("初始化增强的密集检索模型")
    model = EnhancedDenseRetriever(BERT_MODEL, vector_dim=VECTOR_DIM)
    model.to(DEVICE)
    
    # 从语料库加载文档
    corpus_docs = []
    corpus_doc_id_to_text = {}
    corpus_embeddings = None
    corpus_ids = None
    
    if os.path.exists(DOC_FILE):
        # 提取语料库文档
        corpus_docs, corpus_doc_id_to_text = extract_documents_from_corpus(DOC_FILE)
        logger.info(f"从语料库中提取了 {len(corpus_docs)} 个文档")
    else:
        logger.error(f"语料库文件 {DOC_FILE} 不存在，无法继续!")
        return
    
    # 加载训练数据
    logger.info(f"加载训练数据: {TRAIN_FILE}")
    train_data = load_jsonl(TRAIN_FILE)
    logger.info(f"加载了 {len(train_data)} 条训练数据")
    
    # 使用更先进的优化器和学习率策略
    logger.info("使用AdamW优化器和循环学习率")
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05, no_deprecation_warning=True)
    
    # 添加学习率调度器 - 使用余弦退火调度
    train_dataset = QueryDataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=total_steps // (EPOCHS // 2),  # 两个周期
        T_mult=2,  # 每个周期加倍
        eta_min=LEARNING_RATE / 100  # 最小学习率
    )
    
    # 初始化最佳模型状态
    best_loss = float('inf')
    best_model_state = None
    
    model.train()
    
    # 增强的训练循环
    for epoch in range(EPOCHS):
        total_loss = 0
        valid_batch_count = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            
            # 获取查询嵌入
            query_embeddings = model(input_ids, attention_mask)
            
            # 组合多种损失函数以提高性能
            
            # 1. 对比学习损失 - InfoNCE
            sim_matrix = torch.matmul(query_embeddings, query_embeddings.t())
            temperature = 0.05  # 降低温度以增强对比度
            sim_matrix = sim_matrix / temperature
            labels = torch.arange(sim_matrix.size(0), device=DEVICE)
            infonce_loss = F.cross_entropy(sim_matrix, labels)
            
            # 2. 聚类损失 - 让同一类的查询靠近
            # 简化版的聚类：假设每个batch内有相似的样本
            cluster_loss = 0
            if sim_matrix.size(0) > 1:
                # 假设相邻样本更可能相似（适用于某些数据集）
                for i in range(sim_matrix.size(0) - 1):
                    cluster_loss += F.mse_loss(
                        query_embeddings[i], 
                        query_embeddings[i+1].detach()
                    )
                cluster_loss = cluster_loss / (sim_matrix.size(0) - 1)
            
            # 3. 正则化损失 - 促进向量的分散性
            # Embedding diversity loss (鼓励嵌入空间的有效利用)
            eye = torch.eye(query_embeddings.size(1), device=DEVICE)
            cov = torch.matmul(query_embeddings.t(), query_embeddings)
            norm_cov = F.normalize(cov, p=2, dim=1)
            diversity_loss = F.mse_loss(norm_cov, eye)
            
            # 组合损失 - 使用加权和
            loss = infonce_loss + 0.1 * cluster_loss + 0.01 * diversity_loss
            
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度剪裁，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            valid_batch_count += 1
            
            progress_bar.set_postfix({
                "loss": loss.item(), 
                "infonce": infonce_loss.item(),
                "lr": scheduler.get_last_lr()[0]
            })
        
        avg_loss = total_loss / valid_batch_count if valid_batch_count > 0 else 0
        logger.info(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict().copy()
            logger.info(f"保存新的最佳模型，Loss: {best_loss:.4f}")
    
    # 恢复最佳模型状态
    if best_model_state:
        model.load_state_dict(best_model_state)
        logger.info(f"已恢复最佳模型，Loss: {best_loss:.4f}")
    
    # 保存最终模型
    model_save_path = os.path.join(MODEL_DIR, "colbert_model.pt")
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"模型已保存到 {model_save_path}")
    
    # 编码语料库文档
    corpus_embeddings, corpus_ids = encode_corpus_documents(model, corpus_docs, tokenizer)
    
    # 编码训练查询
    logger.info("编码训练查询...")
    train_query_embeddings, _ = encode_queries(model, train_data, tokenizer, VECTOR_TRAIN_DIR, "train_queries")
    
    # 加载验证数据
    logger.info(f"加载验证数据: {VAL_FILE}")
    val_data = load_jsonl(VAL_FILE)
    logger.info(f"加载了 {len(val_data)} 条验证数据")
    
    # 编码验证查询
    logger.info("编码验证查询...")
    val_query_embeddings, val_query_data = encode_queries(model, val_data, tokenizer, VECTOR_VAL_DIR, "val_queries")
    
    # 使用更高级的FAISS索引配置
    logger.info("构建增强的FAISS索引配置...")
    dimension = corpus_embeddings.shape[1]
    
    # 创建FAISS索引 - 使用IVF索引进行更快的检索
    # 对于大型语料库，IVF索引比普通的FlatIP更快
    nlist = min(4096, max(50, corpus_embeddings.shape[0] // 10))  # 聚类数量
    quantizer = faiss.IndexFlatIP(dimension)  # 内部的量化器使用内积计算
    
    if corpus_embeddings.shape[0] > 10000:  # 对于大型语料库使用IVF
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        # 训练IVF索引
        logger.info(f"训练IVF索引 (nlist={nlist})...")
        index.train(corpus_embeddings)
        # 设置搜索时探测的聚类数量 (nprobe)
        index.nprobe = min(64, nlist // 4)  # 搜索时探测的聚类数量
        logger.info(f"设置nprobe={index.nprobe}")
    else:  # 对于小型语料库保持使用FlatIP
        index = faiss.IndexFlatIP(dimension)
    
    # 添加向量到索引
    index.add(corpus_embeddings)
    logger.info(f"FAISS索引已构建，包含 {index.ntotal} 个向量")
    
    # 保存索引
    faiss_index_path = os.path.join(FAISS_INDEX_DIR, "corpus_index.faiss")
    try:
        faiss.write_index(index, faiss_index_path)
        logger.info(f"FAISS索引已保存到 {faiss_index_path}")
    except Exception as e:
        logger.error(f"保存FAISS索引失败: {e}")
    
    # 使用更多检索结果然后重新排序 - 提高召回率
    INITIAL_K = TOP_K * 3  # 首先检索更多候选项
    
    # 使用FAISS检索相关文档，获取详细结果
    raw_retrieved_results = retrieve_documents(index, val_query_embeddings, corpus_ids, corpus_docs, INITIAL_K)
    
    # 重新排序 - 使用更精细的相似度计算
    logger.info("对初始检索结果进行重新排序...")
    retrieved_results = []
    
    for i, query_data in enumerate(val_query_data):
        query_text = query_data["question"]
        raw_results = raw_retrieved_results[i]
        
        # 使用BM25等传统算法进行重新排序
        # 这里使用简化版：基于关键词匹配的重新排序
        for doc in raw_results:
            if "text" in doc:
                # 简单的关键词匹配评分
                doc_text = doc["text"].lower()
                query_tokens = set(query_text.lower().split())
                matches = sum(1 for token in query_tokens if token in doc_text)
                token_score = matches / max(1, len(query_tokens))
                
                # 结合向量相似度和关键词匹配
                doc["combined_score"] = doc["score"] * 0.7 + token_score * 0.3
            else:
                doc["combined_score"] = doc["score"]
        
        # 根据组合分数重新排序
        sorted_results = sorted(raw_results, key=lambda x: x.get("combined_score", x["score"]), reverse=True)
        retrieved_results.append(sorted_results[:TOP_K])  # 保留前K个
    
    # 准备结果
    results = []
    for i, query_data in enumerate(val_query_data):
        # 从检索结果中提取文档ID
        doc_ids = [doc["docid"] for doc in retrieved_results[i]]
        
        result = {
            "question": query_data["question"],
            "answer": query_data["answer"],
            "document_id": doc_ids,
            "scores": [doc["combined_score"] if "combined_score" in doc else doc["score"] for doc in retrieved_results[i]]
        }
        results.append(result)
    
    # 保存结果
    logger.info(f"保存结果到: {OUTPUT_FILE}")
    save_jsonl(results, OUTPUT_FILE)
    
    # 分析模型性能
    # 计算命中率 - 假设answer在document_id对应的文档中
    hits = 0
    for result in results:
        for doc_id in result["document_id"]:
            if str(doc_id) == str(result["answer"]):
                hits += 1
                break
    
    accuracy = hits / len(results) if results else 0
    logger.info(f"命中率: {accuracy:.2%} ({hits}/{len(results)})")
    
    # 保存一个更完整的示例检索代码
    example_file = os.path.join(os.path.dirname(OUTPUT_FILE), "enhanced_retrieval.py")
    with open(example_file, 'w', encoding='utf-8') as f:
        f.write("""#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
import numpy as np
import faiss
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.nn.functional as F

# 增强的检索模型
class EnhancedDenseRetriever(nn.Module):
    def __init__(self, bert_model_name, vector_dim=128, dropout_rate=0.2):
        super(EnhancedDenseRetriever, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.bert.config.hidden_size, vector_dim)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_representation = outputs.last_hidden_state[:, 0, :]
        avg_representation = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1, keepdim=True)
        combined_representation = cls_representation + avg_representation
        combined_representation = self.dropout(combined_representation)
        projected = self.projection(combined_representation)
        normalized = F.normalize(projected, p=2, dim=1)
        return normalized

# 从保存的文件中加载模型和向量
def load_vectors(file_prefix):
    vectors = np.load(f"{file_prefix}_vectors.npy")
    with open(f"{file_prefix}_ids.json", 'r', encoding='utf-8') as f:
        ids = json.load(f)
    return vectors, ids

def load_model(model_path, model_class, device='cpu'):
    model = model_class('bert-base-uncased')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

def load_faiss_index(file_path):
    return faiss.read_index(file_path)

def encode_query(text, model, tokenizer, device='cpu'):
    max_length = 64
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    with torch.no_grad():
        query_vector = model(input_ids, attention_mask)
        
    return query_vector.cpu().numpy()

def retrieve(query_text, model, tokenizer, index, corpus_ids, corpus_data=None, k=5, device='cpu'):
    # 编码查询
    query_embedding = encode_query(query_text, model, tokenizer, device)
    
    # 初始检索
    initial_k = k * 3  # 先检索更多结果用于重排序
    scores, indices = index.search(query_embedding, initial_k)
    
    # 收集检索结果
    results = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(corpus_ids):
            doc_info = {
                "docid": corpus_ids[idx],
                "score": float(scores[0][i])
            }
            
            # 如果有语料库数据，添加文本内容
            if corpus_data:
                for item in corpus_data:
                    if item["docid"] == corpus_ids[idx]:
                        doc_info["text"] = item.get("text", "")
                        break
            
            results.append(doc_info)
    
    # 重新排序 (可选)
    if len(results) > 0 and all("text" in doc for doc in results):
        for doc in results:
            # 简单的关键词匹配评分
            doc_text = doc["text"].lower()
            query_tokens = set(query_text.lower().split())
            matches = sum(1 for token in query_tokens if token in doc_text)
            token_score = matches / max(1, len(query_tokens))
            
            # 结合向量相似度和关键词匹配
            doc["combined_score"] = doc["score"] * 0.7 + token_score * 0.3
        
        # 根据组合分数重新排序
        results = sorted(results, key=lambda x: x.get("combined_score", x["score"]), reverse=True)
    
    # 返回前k个结果
    return results[:k]

# 示例用法
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型和tokenizer
    model_path = "./tmp/colbert/model/total/colbert_model.pt"
    model = load_model(model_path, EnhancedDenseRetriever, device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 加载FAISS索引和语料库数据
    index = load_faiss_index("./tmp/faiss_index/total/corpus_index.faiss")
    _, corpus_data = load_vectors("./tmp/vector_corpus/total/corpus")
    corpus_ids = [item["docid"] for item in corpus_data]
    
    # 示例查询
    query = "什么是机器学习？"
    
    # 检索相关文档
    results = retrieve(query, model, tokenizer, index, corpus_ids, corpus_data, device=device)
    
    # 输出结果
    print(f"查询: {query}")
    print(f"找到 {len(results)} 个相关文档:")
    for i, doc in enumerate(results):
        score = doc.get("combined_score", doc["score"])
        print(f"{i+1}. 文档ID: {doc['docid']}, 相似度分数: {score:.4f}")
        if "text" in doc:
            text = doc["text"]
            # 只显示前200个字符
            print(f"   文本: {text[:200]}..." if len(text) > 200 else f"   文本: {text}")
        print()
""")
    logger.info(f"增强的检索代码示例已保存到: {example_file}")
    
    end_time = time.time()
    logger.info(f"总执行时间: {end_time - start_time:.2f} 秒")
    logger.info(f"日志已保存到: {log_file}")

if __name__ == "__main__":
    main()
