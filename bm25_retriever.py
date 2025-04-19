import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import nltk
import numpy as np
import tomli
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# 确保下载必要的NLTK资源
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)


class BM25:
    """BM25 retrieval model implementation."""

    def __init__(self, k1=1.5, b=0.75, epsilon=0.25):
        """
        Initialize BM25 with parameters.

        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
            epsilon: Additional smoothing parameter
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        self.doc_freqs = defaultdict(int)  # DF: Document frequency per term
        self.idf = {}  # IDF: Inverse document frequency per term
        self.doc_len = []  # Document lengths
        self.avgdl = 0  # Average document length
        self.doc_vectors = []  # Term frequencies per document
        self.doc_ids = []  # Original document IDs
        self.chunk_ids = []  # Original chunk IDs
        self.corpus_size = 0  # Number of documents
        self.titles = []  # Document titles

        # 初始化NLTK工具
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))

        self.index_built = False

    def _calc_idf(self):
        """Calculate inverse document frequency."""
        self.idf = {}
        for term, freq in self.doc_freqs.items():
            # Standard BM25 IDF formula
            idf = np.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf[term] = idf

    def preprocess_document(self, text: str) -> List[str]:
        """
        Basic tokenization for documents.
        Note: Documents are already preprocessed with cleaning and stemming.
        """
        return text.split()

    def preprocess_query(self, text: str) -> List[str]:
        """
        Preprocess query text with the same steps as documents:
        - Lowercase
        - Remove special characters
        - Remove stopwords
        - Apply stemming

        Args:
            text: Query text

        Returns:
            List of preprocessed tokens
        """
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

    def build_index(self, documents: List[Dict], verbose=True):
        """
        Build BM25 index from documents.

        Args:
            documents: List of preprocessed document dictionaries
        """
        if verbose:
            print("Building BM25 index...")
            start_time = time.time()

        # Reset index
        self.doc_vectors = []
        self.doc_len = []
        self.doc_ids = []
        self.chunk_ids = []
        self.titles = []
        self.doc_freqs = defaultdict(int)

        # Process each document
        for doc in documents:
            doc_id = doc["document_id"]
            title = doc["title"]

            for chunk in doc["chunks"]:
                chunk_id = chunk["chunk_id"]
                chunk_title = chunk["title"]
                text = chunk["content"]

                # Tokenize and count term frequencies
                terms = self.preprocess_document(text)
                term_freqs = Counter(terms)

                # Store document data
                self.doc_ids.append(doc_id)
                self.chunk_ids.append(chunk_id)
                self.titles.append(chunk_title)
                self.doc_vectors.append(term_freqs)
                self.doc_len.append(len(terms))

                # Update document frequencies
                for term in set(terms):
                    self.doc_freqs[term] += 1

        # Calculate average document length and corpus size
        self.corpus_size = len(self.doc_vectors)
        self.avgdl = sum(self.doc_len) / self.corpus_size

        # Calculate IDF for each term
        self._calc_idf()

        self.index_built = True

        if verbose:
            build_time = time.time() - start_time
            print(f"BM25 index built in {build_time:.2f} seconds.")
            print(f"  Documents: {self.corpus_size}")
            print(f"  Unique terms: {len(self.doc_freqs)}")
            print(f"  Average document length: {self.avgdl:.1f}")

    def get_score(self, query_terms: List[str], doc_idx: int) -> float:
        """
        Calculate BM25 score for a document.

        Args:
            query_terms: Preprocessed query terms
            doc_idx: Document index

        Returns:
            BM25 score
        """
        if not self.index_built:
            raise ValueError("Index not built. Call build_index first.")

        score = 0.0
        doc_len = self.doc_len[doc_idx]
        doc_freqs = self.doc_vectors[doc_idx]

        # BM25 scoring formula
        for term in query_terms:
            if term not in self.idf:
                continue

            # Get term frequency in document
            freq = doc_freqs.get(term, 0)

            # Apply BM25 formula
            numerator = self.idf[term] * freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += (numerator / denominator) + self.epsilon

        return score

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, int, float, str]]:
        """
        Search documents for the given query.

        Args:
            query: Search query
            top_k: Number of top documents to return

        Returns:
            List of (doc_id, chunk_id, score, title) tuples
        """
        if not self.index_built:
            raise ValueError("Index not built. Call build_index first.")

        # 预处理查询文本
        query_terms = self.preprocess_query(query)

        # 打印预处理后的查询词
        # print(f"Preprocessed query terms: {query_terms}")

        scores = []
        for i in range(self.corpus_size):
            score = self.get_score(query_terms, i)
            scores.append((self.doc_ids[i], self.chunk_ids[i], score, self.titles[i]))

        # Sort by score in descending order
        scores.sort(key=lambda x: x[2], reverse=True)

        return scores[:top_k]

    def get_all_scores(self, query: str) -> Dict[int, List[Tuple[int, float, str]]]:
        """
        Get scores for all documents grouped by document ID.

        Args:
            query: Search query

        Returns:
            Dictionary mapping document_id to list of (chunk_id, score, title) tuples
        """
        if not self.index_built:
            raise ValueError("Index not built. Call build_index first.")

        # 预处理查询文本
        query_terms = self.preprocess_query(query)

        # 按文档ID组织所有chunk的得分
        doc_chunks = defaultdict(list)

        for i in range(self.corpus_size):
            doc_id = self.doc_ids[i]
            chunk_id = self.chunk_ids[i]
            title = self.titles[i]
            score = self.get_score(query_terms, i)

            doc_chunks[doc_id].append((chunk_id, score, title))

        return doc_chunks

    def save(self, path: str):
        """Save the BM25 index to disk."""
        if not self.index_built:
            raise ValueError("Index not built. Call build_index first.")

        data = {
            "k1": self.k1,
            "b": self.b,
            "epsilon": self.epsilon,
            "doc_freqs": dict(self.doc_freqs),
            "idf": self.idf,
            "doc_len": self.doc_len,
            "avgdl": self.avgdl,
            "doc_vectors": [dict(dv) for dv in self.doc_vectors],
            "doc_ids": self.doc_ids,
            "chunk_ids": self.chunk_ids,
            "titles": self.titles,
            "corpus_size": self.corpus_size,
        }

        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        """Load a BM25 index from disk."""
        with open(path, "r") as f:
            data = json.load(f)

        self.k1 = data["k1"]
        self.b = data["b"]
        self.epsilon = data["epsilon"]
        self.doc_freqs = defaultdict(int, data["doc_freqs"])
        self.idf = data["idf"]
        self.doc_len = data["doc_len"]
        self.avgdl = data["avgdl"]
        self.doc_vectors = [Counter(dv) for dv in data["doc_vectors"]]
        self.doc_ids = data["doc_ids"]
        self.chunk_ids = data["chunk_ids"]
        self.titles = data["titles"]
        self.corpus_size = data["corpus_size"]

        self.index_built = True


def load_jsonl(filepath):
    """Load data from a JSONL file."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def calculate_metrics(predictions, top_k=5):
    """
    Calculate Recall@k and MRR@k metrics.

    Args:
        predictions: List of prediction dictionaries
        top_k: K value for metrics

    Returns:
        Dictionary with metrics
    """
    total = len(predictions)
    recall_sum = 0
    mrr_sum = 0
    not_found = []

    for pred in predictions:
        query = pred["question"]
        true_doc_id = pred["true_doc_id"]
        pred_doc_ids = pred["document_id"]

        # Recall@k
        if true_doc_id in pred_doc_ids:
            recall_sum += 1
        else:
            not_found.append((query, true_doc_id, pred_doc_ids))

        # MRR@k
        if true_doc_id in pred_doc_ids:
            rank = pred_doc_ids.index(true_doc_id) + 1
            mrr_sum += 1.0 / rank

    recall_k = recall_sum / total
    mrr_k = mrr_sum / total

    return {"recall@k": recall_k, "mrr@k": mrr_k, "not_found": not_found}


def main():
    # Load configuration
    with open("config.toml", "rb") as f:
        config = tomli.load(f)
    bm25_config = config["bm25"]
    common_config = config["common"]

    # Create directories if they don't exist
    os.makedirs("indexes", exist_ok=True)
    os.makedirs("result", exist_ok=True)

    # Initialize BM25 model with parameters from config
    bm25 = BM25(
        k1=bm25_config.get("k1", 1.5),
        b=bm25_config.get("b", 0.75),
        epsilon=bm25_config.get("epsilon", 0.25),
    )

    # Path to documents
    docs_path = "tmp/document_chunked_cleaned.jsonl"

    # Check if index file exists
    index_path = bm25_config.get("save_index_path", "indexes/bm25_index")

    # Load or build the index
    if os.path.exists(index_path):
        print(f"Loading BM25 index from {index_path}...")
        bm25.load(index_path)
    else:
        print(f"Building BM25 index from {docs_path}...")
        documents = load_jsonl(docs_path)
        bm25.build_index(documents)

        # Save the index
        print(f"Saving BM25 index to {index_path}...")
        bm25.save(index_path)

    # Load validation data
    val_path = common_config.get("val_data_path", "data/val.jsonl")
    val_data = load_jsonl(val_path)

    # Number of top documents to retrieve
    top_k = common_config.get("top_k", 5)

    # Generate predictions
    predictions = []
    query_results = []  # 存储所有查询的详细结果

    print(f"Generating predictions for {len(val_data)} validation queries...")
    for i, item in enumerate(val_data):
        query = item["question"]
        answer = item["answer"]
        true_doc_id = item["document_id"]

        # 预处理查询词
        query_terms = bm25.preprocess_query(query)

        # 搜索初始的大量结果
        all_results = bm25.search(query, top_k=50)

        # 获取每个文档的所有chunk得分
        doc_scores = {}
        for doc_id, chunk_id, score, title in all_results:
            if doc_id not in doc_scores or score > doc_scores[doc_id][1]:
                doc_scores[doc_id] = (chunk_id, score, title)

        # 按得分排序文档
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1][1], reverse=True)

        # 获取前top_k个不同的文档ID
        top_docs = sorted_docs[:top_k]
        document_ids = [doc_id for doc_id, _ in top_docs]

        # 格式化预测结果
        prediction = {"question": query, "answer": answer, "document_id": document_ids}
        predictions.append(prediction)

        # 为每个查询保存详细结果（移除found字段，添加query_terms字段）
        result_entry = {
            "question": query,
            "query_terms": query_terms,  # 添加预处理后的查询词
            "doc_id": true_doc_id,
            "predict": [
                {"doc_id": doc_id, "chunk_id": chunk_id, "score": score, "title": title}
                for doc_id, (chunk_id, score, title) in top_docs
            ],
        }
        query_results.append(result_entry)

        # 打印进度
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(val_data)} queries")

    # 计算指标时需要调整
    recall_count = 0
    mrr_sum = 0
    for r in query_results:
        true_doc_id = r["doc_id"]
        doc_ids = [item["doc_id"] for item in r["predict"]]

        # Recall@k
        if true_doc_id in doc_ids:
            recall_count += 1

            # MRR@k
            rank = doc_ids.index(true_doc_id) + 1
            mrr_sum += 1.0 / rank

    recall_k = recall_count / len(query_results)
    mrr_k = mrr_sum / len(query_results)

    print("\n===== Evaluation Results =====")
    print(f"Document Retrieval Recall@{top_k}: {recall_k:.4f}")
    print(f"Document Retrieval MRR@{top_k}:    {mrr_k:.4f}")
    print(f"Documents not found: {len(query_results) - recall_count}/{len(val_data)}")

    # 保存所有查询的详细结果
    results_path = "result/query_results.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for result in query_results:
            f.write(json.dumps(result) + "\n")

    print(f"Detailed results for all queries saved to {results_path}")

    # 保存预测结果
    output_path = "result/val_predict.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
