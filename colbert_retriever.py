#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import os

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Enhanced ColBERT model
class EnhancedDenseRetriever(nn.Module):
    def __init__(
        self, bert_model_name="bert-base-uncased", vector_dim=128, dropout_rate=0.2
    ):
        super(EnhancedDenseRetriever, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.bert.config.hidden_size, vector_dim),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_representation = outputs.last_hidden_state[:, 0, :]
        avg_representation = (
            outputs.last_hidden_state * attention_mask.unsqueeze(-1)
        ).sum(1) / attention_mask.sum(-1, keepdim=True)
        combined_representation = cls_representation + avg_representation
        combined_representation = self.dropout(combined_representation)
        projected = self.projection(combined_representation)
        normalized = F.normalize(projected, p=2, dim=1)
        return normalized


class ColBERTRetriever:
    def __init__(self, config=None):
        """
        Initialize the ColBERT retriever

        Args:
            config: configuration dictionary containing paths to model, index, and corpus
        """
        self.config = config or {}
        self.colbert_config = self.config.get("colbert", {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.index = None
        self.corpus_ids = []
        self.top_k = self.colbert_config.get("top_k", 3)
        self.vector_dim = 128

        # Get paths from config
        self.model_path = self.colbert_config.get(
            "model_path", "indexes/colbert/colbert_model.pt"
        )
        self.faiss_index_path = self.colbert_config.get(
            "faiss_index_path", "indexes/colbert/corpus_index.faiss"
        )

        # Get the corpus vector directory
        self.corpus_vector_dir = self.colbert_config.get(
            "colbert_vector_dir", "indexes/colbert/corpus"
        )

        # Default corpus embeddings filenames
        self.corpus_ids_path = os.path.join(
            self.corpus_vector_dir, "corpus_embeddings_ids.json"
        )
        self.corpus_vectors_path = os.path.join(
            self.corpus_vector_dir, "corpus_embeddings_vectors.npy"
        )

        # Check if specific corpus_ids_path is provided in the config
        if "corpus_ids_path" in self.colbert_config:
            self.corpus_ids_path = self.colbert_config.get("corpus_ids_path")

        logger.info(f"ColBERT Retriever initialized with device: {self.device}")
        logger.info(f"Using model path: {self.model_path}")
        logger.info(f"Using FAISS index path: {self.faiss_index_path}")
        logger.info(f"Using corpus directory: {self.corpus_vector_dir}")
        logger.info(f"Using corpus IDs path: {self.corpus_ids_path}")
        logger.info(f"Using fixed vector dimension: {self.vector_dim}")

    def load(self):
        """
        Load the model, tokenizer, index, and corpus data

        Returns:
            bool: True if everything loaded successfully, False otherwise
        """
        try:
            # Load tokenizer
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            logger.info("Tokenizer loaded")

            # Load model with fixed vector dimension
            logger.info(f"Creating model with vector dimension: {self.vector_dim}")
            self.model = EnhancedDenseRetriever(vector_dim=self.vector_dim)

            # Load the model state dict
            logger.info(f"Loading model state from: {self.model_path}")
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"ColBERT model loaded from {self.model_path}")

            # Load FAISS index
            self.index = faiss.read_index(self.faiss_index_path)
            logger.info(
                f"FAISS index loaded from {self.faiss_index_path} with {self.index.ntotal} vectors"
            )

            # Check index dimension matches model
            if hasattr(self.index, "d"):
                index_dim = self.index.d
                if index_dim != self.vector_dim:
                    logger.warning(
                        f"FAISS index dimension ({index_dim}) doesn't match model dimension ({self.vector_dim})!"
                    )

            # Load corpus IDs
            with open(self.corpus_ids_path, "r", encoding="utf-8") as f:
                corpus_data = json.load(f)

            # Handle different corpus data formats
            if isinstance(corpus_data[0], dict):
                self.corpus_ids = [item["docid"] for item in corpus_data]
            else:
                self.corpus_ids = corpus_data

            logger.info(
                f"Corpus IDs loaded from {self.corpus_ids_path}, total: {len(self.corpus_ids)}"
            )

            return True

        except Exception as e:
            logger.error(f"Error loading ColBERT retriever: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False

    def encode_query(self, text):
        """
        Encode a query text into a vector

        Args:
            text: query text to encode

        Returns:
            numpy array: the encoded query vector
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Model or tokenizer not loaded. Call load() first.")
            return None

        # Tokenize the query
        encoding = self.tokenizer(
            text,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Encode the query
        with torch.no_grad():
            query_vector = self.model(input_ids, attention_mask)

        return query_vector.cpu().numpy()

    def search(self, query, top_k=None):
        """
        Search for the most similar documents to the query

        Args:
            query: the query text
            top_k: number of results to return (defaults to self.top_k)

        Returns:
            list: the top_k most similar documents, each as (doc_id, chunk_id, score, title)
        """
        if top_k is None:
            top_k = self.top_k

        if self.model is None or self.index is None:
            logger.error("Model or index not loaded. Call load() first.")
            return []

        # Encode the query
        query_embedding = self.encode_query(query)

        if query_embedding is None:
            return []

        # Double-check dimensions
        expected_dim = self.index.d
        actual_dim = query_embedding.shape[1]

        if expected_dim != actual_dim:
            logger.error(
                f"Dimension mismatch: FAISS index expects {expected_dim}, but query embedding has {actual_dim} dimensions"
            )
            logger.info("Attempting to adapt query embedding dimension...")

            # Try to adapt the embedding if possible
            if expected_dim < actual_dim:
                # Truncate the vector
                query_embedding = query_embedding[:, :expected_dim]
                logger.info(
                    f"Truncated query embedding from {actual_dim} to {expected_dim} dimensions"
                )
            else:
                # Can't easily expand dimensions, so return an error
                return []

        # Search the index
        scores, indices = self.index.search(query_embedding, top_k)

        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.corpus_ids):
                doc_id = self.corpus_ids[idx]

                # Handle case where doc_id could be a string like "doc_123|chunk_45"
                if isinstance(doc_id, str) and "|" in doc_id:
                    parts = doc_id.split("|")
                    if len(parts) >= 2:
                        doc_id_part = parts[0]
                        # Extract numeric ID if in format "doc_123"
                        if doc_id_part.startswith("doc_"):
                            doc_id_part = doc_id_part[4:]

                        chunk_id_part = parts[1]
                        # Extract numeric ID if in format "chunk_45"
                        if chunk_id_part.startswith("chunk_"):
                            chunk_id_part = chunk_id_part[6:]

                        try:
                            doc_id = int(doc_id_part)
                            chunk_id = int(chunk_id_part)
                            # Add to results: (doc_id, chunk_id, score, title)
                            results.append((doc_id, chunk_id, float(scores[0][i]), ""))
                        except ValueError:
                            # If conversion fails, use original format
                            results.append((doc_id, 0, float(scores[0][i]), ""))
                else:
                    # If doc_id is not in expected format, use as is with default chunk_id=0
                    results.append((doc_id, 0, float(scores[0][i]), ""))

        return results


# For testing
if __name__ == "__main__":
    import tomli

    # Load config
    with open("config.toml", "rb") as f:
        config = tomli.load(f)

    # Initialize and load retriever
    retriever = ColBERTRetriever(config)
    if retriever.load():
        # Test search
        query = "What is machine learning?"
        results = retriever.search(query, top_k=3)

        print(f"Query: {query}")
        print(f"Results: {len(results)}")
        for doc_id, chunk_id, score, title in results:
            print(f"  Doc ID: {doc_id}, Chunk ID: {chunk_id}, Score: {score:.4f}")


