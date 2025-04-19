import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import nltk
import numpy as np
import tomli
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# 导入API客户端
from api_client import SiliconFlowClient

# 确保下载必要的NLTK资源
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)


class Word2VecRetriever:
    """Word2Vec/GloVe检索模型实现，使用FAISS索引"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化Word2Vec检索器

        Args:
            config: 配置参数
        """
        self.config = config
        self.word2vec_config = config.get("word2vec", {})
        self.common_config = config.get("common", {})

        # 获取配置参数
        self.glove_path = self.word2vec_config.get(
            "glove_path", "model/glove.6B.300d.txt"
        )
        self.vector_dim = self.word2vec_config.get("vector_dim", 300)
        self.tfidf_weighting = self.word2vec_config.get("tfidf_weighting", True)
        self.save_index_path = self.word2vec_config.get(
            "save_index_path", "indexes/word2vec_index.faiss"
        )
        self.save_metadata_path = self.word2vec_config.get(
            "save_metadata_path", "indexes/word2vec_metadata.json"
        )

        # 是否使用文本预处理
        self.use_preprocessing = self.word2vec_config.get("use_preprocessing", False)

        # 初始化NLTK工具
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))

        # 加载GloVe词向量
        self.word_vectors = self._load_glove()

        # 索引和元数据
        self.index = None  # FAISS索引
        self.doc_ids = []  # 文档ID
        self.chunk_ids = []  # 分块ID
        self.titles = []  # 标题
        self.idf = {}  # 逆文档频率
        self.total_docs = 0  # 文档总数

        self.index_built = False

    def _load_glove(self):
        """加载GloVe预训练词向量"""
        print(f"Loading GloVe vectors from {self.glove_path}...")

        # 转换GloVe格式为Word2Vec格式
        try:
            # 创建model目录
            os.makedirs("model", exist_ok=True)

            # 指定保存转换后文件的路径
            word2vec_output_file = os.path.join("model", "glove.word2vec.300d.txt")

            if not os.path.exists(word2vec_output_file):
                glove2word2vec(self.glove_path, word2vec_output_file)

            # 加载词向量
            model = KeyedVectors.load_word2vec_format(word2vec_output_file)
            print(
                f"Loaded {len(model.key_to_index)} word vectors with dimension {self.vector_dim}"
            )
            return model
        except Exception as e:
            print(f"Error loading GloVe vectors: {e}")
            print("Using random vectors as fallback")
            # 创建一个假的词向量模型，用于测试
            from gensim.models import Word2Vec

            return Word2Vec(vector_size=self.vector_dim, min_count=1).wv

    def preprocess_query(self, text: str) -> List[str]:
        """
        预处理查询文本

        Args:
            text: 查询文本

        Returns:
            处理后的词tokens
        """
        # 如果不使用预处理，直接分词返回
        if not self.use_preprocessing:
            print("query not preprocessed")
            return text.split()

        # 否则进行完整预处理
        # 转为小写
        text = text.lower()

        # 移除特殊字符和数字
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # 分词
        tokens = word_tokenize(text)

        # 移除停用词并应用词干提取
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words:
                # 应用词干提取
                stemmed = self.stemmer.stem(token)
                processed_tokens.append(stemmed)

        return processed_tokens

    def _calculate_idf(self, corpus: List[List[str]]):
        """
        计算语料库中词汇的IDF值

        Args:
            corpus: 分词后的文档列表
        """
        self.total_docs = len(corpus)
        doc_freq = Counter()

        # 计算每个词在多少文档中出现
        for doc in corpus:
            words = set(doc)  # 去重，每个文档中的词只计算一次
            for word in words:
                doc_freq[word] += 1

        # 计算IDF
        self.idf = {}
        for word, freq in doc_freq.items():
            self.idf[word] = np.log((self.total_docs + 1) / (freq + 1)) + 1  # 平滑处理

    def _get_document_vector(self, doc_tokens: List[str]) -> np.ndarray:
        """
        计算文档的向量表示，使用加权平均

        Args:
            doc_tokens: 文档分词列表

        Returns:
            文档向量
        """
        if not doc_tokens:
            return np.zeros(self.vector_dim)

        # 初始化文档向量
        doc_vector = np.zeros(self.vector_dim)
        word_count = 0

        # 使用TF-IDF加权平均
        if self.tfidf_weighting:
            # 计算词频
            term_freq = Counter(doc_tokens)
            total_terms = len(doc_tokens)

            for token in doc_tokens:
                if token in self.word_vectors:
                    # 计算TF-IDF权重
                    tf = term_freq[token] / total_terms
                    idf = self.idf.get(token, 1.0)  # 默认为1.0
                    weight = tf * idf

                    # 加权求和
                    doc_vector += weight * self.word_vectors[token]
                    word_count += 1
        else:
            # 简单平均
            for token in doc_tokens:
                if token in self.word_vectors:
                    doc_vector += self.word_vectors[token]
                    word_count += 1

        # 归一化
        if word_count > 0:
            doc_vector /= word_count

        # L2归一化
        norm = np.linalg.norm(doc_vector)
        if norm > 0:
            doc_vector /= norm

        return doc_vector

    def build_index(self, documents: List[Dict], verbose=True):
        """
        构建基于FAISS的检索索引

        Args:
            documents: 文档列表
            verbose: 是否打印详细信息
        """
        if verbose:
            print("Building Word2Vec/FAISS index...")
            start_time = time.time()

        # 重置索引
        self.doc_ids = []
        self.chunk_ids = []
        self.titles = []
        corpus = []

        # 处理每个文档
        for doc in documents:
            doc_id = doc["document_id"]
            title = doc["title"]

            for chunk in doc["chunks"]:
                chunk_id = chunk["chunk_id"]
                chunk_title = chunk["title"]
                text = chunk["content"]

                # 文档分词 - 使用简单的空格分词
                tokens = text.split()

                # 存储元数据
                self.doc_ids.append(doc_id)
                self.chunk_ids.append(chunk_id)
                self.titles.append(chunk_title)
                corpus.append(tokens)

        # 计算IDF值（用于TF-IDF加权）
        if self.tfidf_weighting:
            self._calculate_idf(corpus)

        # 对每个文档生成向量
        doc_vectors = []
        for i, tokens in enumerate(corpus):
            if verbose and (i + 1) % 1000 == 0:
                print(f"  Vectorizing document {i+1}/{len(corpus)}")

            vector = self._get_document_vector(tokens)
            doc_vectors.append(vector)

        # 转换为numpy数组
        vectors_array = np.array(doc_vectors).astype("float32")

        # 构建FAISS索引
        self.index = faiss.IndexFlatIP(
            self.vector_dim
        )  # 内积（对于L2归一化的向量等同于余弦相似度）
        self.index.add(vectors_array)

        self.index_built = True

        if verbose:
            build_time = time.time() - start_time
            print(f"Word2Vec/FAISS index built in {build_time:.2f} seconds.")
            print(f"  Documents: {len(self.doc_ids)}")
            print(f"  Vector dimension: {self.vector_dim}")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, int, float, str]]:
        """
        搜索文档

        Args:
            query: 查询文本
            top_k: 返回的文档数量

        Returns:
            (doc_id, chunk_id, score, title)元组列表
        """
        if not self.index_built:
            raise ValueError("Index not built. Call build_index first.")

        # 预处理查询文本
        query_tokens = self.preprocess_query(query)

        # 生成查询向量
        query_vector = self._get_document_vector(query_tokens)
        query_vector = np.array([query_vector]).astype("float32")

        # 搜索相似文档
        scores, indices = self.index.search(query_vector, top_k)

        # 格式化结果
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(
                self.doc_ids
            ):  # FAISS可能返回-1表示找不到足够的结果
                continue
            doc_id = self.doc_ids[idx]
            chunk_id = self.chunk_ids[idx]
            title = self.titles[idx]
            score = float(scores[0][i])
            results.append((doc_id, chunk_id, score, title))

        return results

    def save(self):
        """保存索引和元数据"""
        if not self.index_built:
            raise ValueError("Index not built. Call build_index first.")

        # 创建索引目录
        os.makedirs(os.path.dirname(self.save_index_path), exist_ok=True)

        # 保存FAISS索引
        print(f"Saving FAISS index to {self.save_index_path}")
        faiss.write_index(self.index, self.save_index_path)

        # 保存元数据
        metadata = {
            "doc_ids": self.doc_ids,
            "chunk_ids": self.chunk_ids,
            "titles": self.titles,
            "vector_dim": self.vector_dim,
            "idf": self.idf,
            "total_docs": self.total_docs,
        }

        with open(self.save_metadata_path, "w") as f:
            json.dump(metadata, f)

        print(f"Metadata saved to {self.save_metadata_path}")

    def load(self):
        """加载索引和元数据"""
        if not os.path.exists(self.save_index_path) or not os.path.exists(
            self.save_metadata_path
        ):
            raise FileNotFoundError(f"Index files not found at {self.save_index_path}")

        # 加载FAISS索引
        print(f"Loading FAISS index from {self.save_index_path}")
        self.index = faiss.read_index(self.save_index_path)

        # 加载元数据
        with open(self.save_metadata_path, "r") as f:
            metadata = json.load(f)

        self.doc_ids = metadata["doc_ids"]
        self.chunk_ids = metadata["chunk_ids"]
        self.titles = metadata["titles"]
        self.vector_dim = metadata["vector_dim"]
        self.idf = metadata["idf"]
        self.total_docs = metadata["total_docs"]

        self.index_built = True
        print(f"Loaded {len(self.doc_ids)} documents with dimension {self.vector_dim}")


def load_jsonl(filepath):
    """从JSONL文件加载数据"""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def calculate_metrics(predictions):
    """
    计算Recall@k和MRR@k指标

    Args:
        predictions: 预测结果列表

    Returns:
        指标字典
    """
    total = len(predictions)
    recall_sum = 0
    mrr_sum = 0
    not_found = []  # 记录未找到的查询

    for pred in predictions:
        true_doc_id = pred["true_doc_id"]
        pred_doc_ids = pred["document_id"]
        query = pred["question"]

        # Recall@k
        if true_doc_id in pred_doc_ids:
            recall_sum += 1

            # MRR@k
            rank = pred_doc_ids.index(true_doc_id) + 1
            mrr_sum += 1.0 / rank
        else:
            # 记录未找到的查询
            not_found.append(
                {
                    "question": query,
                    "true_doc_id": true_doc_id,
                    "predicted_docs": pred_doc_ids,
                }
            )

    recall_k = recall_sum / total
    mrr_k = mrr_sum / total

    return {
        "recall@k": recall_k,
        "mrr@k": mrr_k,
        "not_found": not_found,
        "not_found_count": len(not_found),
    }


def main():
    # 加载配置
    with open("config.toml", "rb") as f:
        config = tomli.load(f)

    # 创建必要的目录
    os.makedirs("indexes", exist_ok=True)
    os.makedirs("result", exist_ok=True)
    os.makedirs("model", exist_ok=True)

    # 初始化Word2Vec检索器
    retriever = Word2VecRetriever(config)

    # 文档路径 - 使用原始未处理的文档
    docs_path = "tmp/document_chunked_cleaned.jsonl"
    original_docs_path = "tmp/document_chunked_original.jsonl"

    # 加载原始文档以获取chunk内容
    print(f"Loading original documents from {original_docs_path}...")
    original_docs = {}
    original_chunks = {}

    try:
        # 加载原始文档数据
        original_doc_data = load_jsonl(original_docs_path)
        for doc in original_doc_data:
            doc_id = doc["document_id"]
            original_docs[doc_id] = doc
            # 建立doc_id和chunk_id到chunk内容的映射
            for chunk in doc["chunks"]:
                chunk_key = (doc_id, chunk["chunk_id"])
                original_chunks[chunk_key] = chunk
    except FileNotFoundError:
        print(f"Warning: Original document file {original_docs_path} not found.")
        print("Using example/document_chunked_original.jsonl as fallback.")
        # 尝试使用样例数据作为回退选项
        try:
            original_doc_data = load_jsonl("example/document_chunked_original.jsonl")
            for doc in original_doc_data:
                doc_id = doc["document_id"]
                original_docs[doc_id] = doc
                # 建立doc_id和chunk_id到chunk内容的映射
                for chunk in doc["chunks"]:
                    chunk_key = (doc_id, chunk["chunk_id"])
                    original_chunks[chunk_key] = chunk
        except FileNotFoundError:
            print("Error: Could not find any original document files.")
            return

    # 加载或构建索引
    if os.path.exists(retriever.save_index_path) and os.path.exists(
        retriever.save_metadata_path
    ):
        print(f"Loading Word2Vec/FAISS index...")
        retriever.load()
    else:
        print(f"Building Word2Vec/FAISS index from {docs_path}...")
        documents = load_jsonl(docs_path)
        retriever.build_index(documents)

        # 保存索引
        print(f"Saving Word2Vec/FAISS index...")
        retriever.save()

    # 初始化SiliconFlow API客户端
    print("Initializing SiliconFlow API client...")
    api_client = SiliconFlowClient()

    # 加载验证数据
    val_path = config["common"]["val_data_path"]
    val_data = load_jsonl(val_path)

    # 获取要检索的文档数量
    top_k = config["common"]["top_k"]

    # 生成预测结果
    predictions = []
    full_predictions = []  # 用于评估的完整预测
    query_results = []  # 存储所有查询的详细结果

    print(f"Generating predictions for {len(val_data)} validation queries...")
    for i, item in enumerate(val_data):
        query = item["question"]
        original_answer = item["answer"]
        true_doc_id = item["document_id"]

        # 搜索文档
        results = retriever.search(
            query, top_k=top_k * 2
        )  # 获取更多结果,以便去重后仍有足够数量

        # 获取文档ID（去重）
        document_ids = []
        seen_ids = set()
        result_details = []

        for doc_id, chunk_id, score, title in results:
            if doc_id not in seen_ids:
                document_ids.append(doc_id)
                seen_ids.add(doc_id)
                result_details.append(
                    {
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "score": score,
                        "title": title,
                    }
                )

        # 确保不超过top_k个文档
        document_ids = document_ids[:top_k]
        result_details = result_details[:top_k]

        # 收集前top_k个chunk的内容作为上下文
        context_chunks = []
        for detail in result_details:
            doc_id = detail["doc_id"]
            chunk_id = detail["chunk_id"]
            chunk_key = (doc_id, chunk_id)
            if chunk_key in original_chunks:
                chunk = original_chunks[chunk_key]
                context_chunks.append(chunk["content"])

        # 使用SiliconFlow API生成回答
        if context_chunks:
            print(f"Generating answer for query {i+1}: {query}")
            generated_answer = api_client.generate_answer(
                question=query,
                context=context_chunks[:3],  # 使用前3个最相关的chunk
                max_tokens=150,
                temperature=0.3,
            )
        else:
            generated_answer = "抱歉，我无法回答这个问题。"

        # 格式化预测结果（符合val_predict.jsonl格式）
        prediction = {
            "question": query,
            "answer": generated_answer,  # 使用生成的回答
            "document_id": document_ids,
        }

        # 完整预测结果（用于评估）
        full_prediction = {
            "question": query,
            "original_answer": original_answer,  # 保存原始答案
            "generated_answer": generated_answer,  # 添加生成的答案
            "true_doc_id": true_doc_id,
            "document_id": document_ids,
        }

        # 详细查询结果（包含所有信息）
        query_result = {
            "question": query,
            "original_answer": original_answer,
            "generated_answer": generated_answer,
            "true_doc_id": true_doc_id,
            "predict": result_details,
        }

        predictions.append(prediction)
        full_predictions.append(full_prediction)
        query_results.append(query_result)

        # 打印进度
        if (i + 1) % 10 == 0:  # 改为每10条打印一次
            print(f"Processed {i+1}/{len(val_data)} queries")

    # 计算评估指标 (使用document_id)
    metrics = calculate_metrics(full_predictions)

    print("\n===== Evaluation Results =====")
    print(f"Document Retrieval Recall@{top_k}: {metrics['recall@k']:.4f}")
    print(f"Document Retrieval MRR@{top_k}:    {metrics['mrr@k']:.4f}")
    print(f"Documents not found: {metrics['not_found_count']}/{len(val_data)}")

    # 保存未找到的查询
    not_found_path = "result/word2vec_not_found.json"
    with open(not_found_path, "w", encoding="utf-8") as f:
        json.dump(metrics["not_found"], f, indent=2)
    print(f"Queries with documents not found saved to {not_found_path}")

    # 保存所有查询的详细结果
    query_results_path = "result/word2vec_query_results_with_llm.jsonl"
    with open(query_results_path, "w", encoding="utf-8") as f:
        for result in query_results:
            f.write(json.dumps(result) + "\n")
    print(f"Detailed results with generated answers saved to {query_results_path}")

    # 保存预测结果
    output_path = "result/word2vec_val_predict.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    print(f"Predictions with LLM-generated answers saved to {output_path}")


if __name__ == "__main__":
    main()
