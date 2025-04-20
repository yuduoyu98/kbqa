#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import tomli

from colbert_retriever import ColBERTRetriever


def main():
    # Load configuration
    try:
        with open("config.toml", "rb") as f:
            config = tomli.load(f)
    except Exception as e:
        print(f"Error loading config.toml: {e}")
        return

    print("Configuration loaded successfully")

    # Print ColBERT configuration
    colbert_config = config.get("colbert", {})
    model_path = colbert_config.get("model_path", "indexes/colbert/colbert_model.pt")
    index_path = colbert_config.get(
        "faiss_index_path", "indexes/colbert/corpus_index.faiss"
    )
    vector_dir = colbert_config.get("colbert_vector_dir", "indexes/colbert/corpus")

    print(f"ColBERT Config:")
    print(f"  Model path: {model_path} (exists: {os.path.exists(model_path)})")
    print(f"  Index path: {index_path} (exists: {os.path.exists(index_path)})")
    print(f"  Vector dir: {vector_dir} (exists: {os.path.exists(vector_dir)})")

    ids_file = os.path.join(vector_dir, "corpus_embeddings_ids.json")
    vectors_file = os.path.join(vector_dir, "corpus_embeddings_vectors.npy")

    print(f"  IDs file: {ids_file} (exists: {os.path.exists(ids_file)})")
    print(f"  Vectors file: {vectors_file} (exists: {os.path.exists(vectors_file)})")
    print(f"  Using fixed vector dimension: 128 (matching the saved model)")

    # Initialize and load retriever
    print("\nInitializing ColBERTRetriever...")
    retriever = ColBERTRetriever(config)

    print("\nLoading model, index, and corpus data...")
    if retriever.load():
        print("\nRetriever loaded successfully.")
        print(f"Vector dimension used: {retriever.vector_dim}")

        # Test search with a sample query
        query = "What is machine learning?"
        print(f"\nSearching for: '{query}'")

        try:
            results = retriever.search(query, top_k=3)
            print(f"Found {len(results)} results:")

            for i, (doc_id, chunk_id, score, title) in enumerate(results):
                print(f"  Result {i+1}:")
                print(f"    Doc ID: {doc_id}")
                print(f"    Chunk ID: {chunk_id}")
                print(f"    Score: {score:.4f}")
        except Exception as e:
            print(f"Error during search: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("Failed to load the retriever")


if __name__ == "__main__":
    main()
