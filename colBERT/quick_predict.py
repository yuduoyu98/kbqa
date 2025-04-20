#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
import numpy as np
import faiss
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm

# 解析命令行参数
parser = argparse.ArgumentParser(description='使用预训练ColBERT模型直接进行预测')
parser.add_argument('--model_path', type=str, default="tmp/colbert/model/total/colbert_model.pt", help='模型文件路径')
parser.add_argument('--faiss_index', type=str, default="tmp/faiss_index/total/corpus_index.faiss", help='FAISS索引文件路径')
parser.add_argument('--corpus_vectors', type=str, default="tmp/vector_corpus/total/corpus", help='语料库向量文件路径（不含后缀）')
parser.add_argument('--valfile', type=str, default="tmp/val_predict_cleaned.jsonl", help='验证文件路径')
parser.add_argument('--output', type=str, default="tmp/colbert/quick_predict_results.jsonl", help='输出文件路径')
parser.add_argument('--top_k', type=int, default=5, help='返回的文档数量')
parser.add_argument('--vector_dim', type=int, default=128, help='向量维度')
args = parser.parse_args()

# 确保输出目录存在
os.makedirs(os.path.dirname(args.output), exist_ok=True)

# 增强的ColBERT模型
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

# 加载jsonl文件
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

# 保存jsonl文件
def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 加载向量和ID
def load_vectors(file_prefix):
    vectors = np.load(f"{file_prefix}_vectors.npy")
    with open(f"{file_prefix}_ids.json", 'r', encoding='utf-8') as f:
        ids = json.load(f)
    return vectors, ids

# 编码查询
def encode_query(text, model, tokenizer, device):
    encoding = tokenizer(
        text,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    with torch.no_grad():
        query_vector = model(input_ids, attention_mask)
        
    return query_vector.cpu().numpy()

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载FAISS索引
    print(f"加载FAISS索引: {args.faiss_index}")
    index = faiss.read_index(args.faiss_index)
    print(f"FAISS索引包含 {index.ntotal} 个向量")
    
    # 加载语料库向量和ID
    print(f"加载语料库向量: {args.corpus_vectors}")
    _, corpus_data = load_vectors(args.corpus_vectors)
    if isinstance(corpus_data[0], dict):
        corpus_ids = [item["docid"] for item in corpus_data]
    else:
        corpus_ids = corpus_data
    print(f"语料库包含 {len(corpus_ids)} 个文档")
    
    # 加载验证数据
    print(f"加载验证数据: {args.valfile}")
    val_data = load_jsonl(args.valfile)
    print(f"加载了 {len(val_data)} 条验证数据")
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    model = EnhancedDenseRetriever('bert-base-uncased', vector_dim=args.vector_dim)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 处理查询并检索相关文档
    results = []
    
    for query_data in tqdm(val_data, desc="处理查询"):
        question = query_data["question"]
        answer = query_data.get("answer", "")
        
        # 编码查询
        query_embedding = encode_query(question, model, tokenizer, device)
        
        # 执行检索
        scores, indices = index.search(query_embedding, args.top_k)
        
        # 获取文档ID
        doc_ids = [corpus_ids[idx] for idx in indices[0] if 0 <= idx < len(corpus_ids)]
        
        # 添加到结果
        result = {
            "question": question,
            "answer": answer,
            "document_id": doc_ids,
            "scores": [float(score) for score in scores[0]]
        }
        results.append(result)
    
    # 保存结果
    print(f"保存结果到: {args.output}")
    save_jsonl(results, args.output)
    
    # 计算命中率
    hits = 0
    for result in results:
        answer = result.get("answer", "")
        if answer:
            for doc_id in result["document_id"]:
                if str(doc_id) == str(answer):
                    hits += 1
                    break
    
    accuracy = hits / len(results) if results else 0
    print(f"命中率: {accuracy:.2%} ({hits}/{len(results)})")

if __name__ == "__main__":
    main() 