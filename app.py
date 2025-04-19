import json
import os
import sys

from flask import Flask, jsonify, request
from flask_cors import CORS

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tomli

from api_client import SiliconFlowClient

# Import the retrievers and API client
from bm25_retriever import BM25
from word2vec_retriever import Word2VecRetriever

# Load configuration
with open("config.toml", "rb") as f:
    config = tomli.load(f)

# Initialize the BM25 retriever
bm25_index_path = config["bm25"].get("save_index_path", "indexes/bm25_index")
bm25_retriever = None

# Initialize the Word2Vec (GloVe) retriever
word2vec_retriever = None

# Original documents data
original_docs = {}
original_chunks = {}


def create_app():
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes

    # Initialize retrievers and data
    initialize_retrievers()
    load_original_documents()

    return app


def initialize_retrievers():
    global bm25_retriever, word2vec_retriever

    # Check if we should use parent paths
    if not os.path.exists(bm25_index_path) and os.path.exists(f"../{bm25_index_path}"):
        parent_bm25_path = f"../{bm25_index_path}"
    else:
        parent_bm25_path = bm25_index_path

    # Initialize BM25 retriever
    if os.path.exists(parent_bm25_path):
        bm25_retriever = BM25()
        bm25_retriever.load(parent_bm25_path)
        print(f"BM25 retriever initialized successfully from {parent_bm25_path}")
    else:
        print(f"Warning: BM25 index not found at {parent_bm25_path}")

    # Initialize Word2Vec (GloVe) retriever if index exists
    word2vec_index_path = config["word2vec"].get(
        "save_index_path", "indexes/word2vec_index.faiss"
    )
    word2vec_metadata_path = config["word2vec"].get(
        "save_metadata_path", "indexes/word2vec_metadata.json"
    )

    # Check if we should use parent paths
    if (
        not os.path.exists(word2vec_index_path)
        and os.path.exists(f"../{word2vec_index_path}")
        and os.path.exists(f"../{word2vec_metadata_path}")
    ):
        word2vec_index_path = f"../{word2vec_index_path}"
        word2vec_metadata_path = f"../{word2vec_metadata_path}"

    if os.path.exists(word2vec_index_path) and os.path.exists(word2vec_metadata_path):
        try:
            # Initialize Word2Vec retriever with config
            word2vec_retriever = Word2VecRetriever(config)
            word2vec_retriever.load()
            print(
                f"Word2Vec (GloVe) retriever initialized successfully from {word2vec_index_path}"
            )
        except Exception as e:
            print(f"Error initializing Word2Vec retriever: {e}")
    else:
        print(f"Warning: Word2Vec index not found at {word2vec_index_path}")


def load_original_documents():
    """Load original documents to get chunk content"""
    global original_docs, original_chunks

    original_docs_path = "tmp/document_chunked_original.jsonl"

    # Check if we should use parent path
    if not os.path.exists(original_docs_path) and os.path.exists(
        f"../{original_docs_path}"
    ):
        original_docs_path = f"../{original_docs_path}"

    try:
        # Load original document data
        with open(original_docs_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                doc_id = doc["document_id"]
                original_docs[doc_id] = doc
                # Map doc_id and chunk_id to chunk content
                for chunk in doc["chunks"]:
                    chunk_key = (doc_id, chunk["chunk_id"])
                    original_chunks[chunk_key] = chunk
        print(
            f"Loaded {len(original_docs)} documents and {len(original_chunks)} chunks"
        )
    except FileNotFoundError:
        print(f"Warning: Original document file {original_docs_path} not found.")
        print("Using example/document_chunked_original.jsonl as fallback.")
        try:
            # Try fallback to example directory
            fallback_path = "example/document_chunked_original.jsonl"
            if not os.path.exists(fallback_path) and os.path.exists(
                f"../{fallback_path}"
            ):
                fallback_path = f"../{fallback_path}"

            with open(fallback_path, "r", encoding="utf-8") as f:
                for line in f:
                    doc = json.loads(line)
                    doc_id = doc["document_id"]
                    original_docs[doc_id] = doc
                    # Map doc_id and chunk_id to chunk content
                    for chunk in doc["chunks"]:
                        chunk_key = (doc_id, chunk["chunk_id"])
                        original_chunks[chunk_key] = chunk
            print(
                f"Loaded {len(original_docs)} documents and {len(original_chunks)} chunks from example directory"
            )
        except FileNotFoundError:
            print("Error: Could not find any original document files.")


# Initialize API client
api_client = SiliconFlowClient(
    "../config.toml" if os.path.exists("../config.toml") else "config.toml"
)

# Create the Flask app
app = create_app()


@app.route("/api/search", methods=["POST"])
def search():
    data = request.json
    question = data.get("question", "")
    model_type = data.get("mode", "bm25")  # Default to BM25

    if not question:
        return jsonify({"error": "Question is required"}), 400

    if model_type == "bm25":
        if bm25_retriever is None:
            return jsonify({"error": "BM25 retriever not initialized"}), 500

        # Search with BM25
        results = bm25_retriever.search(question, top_k=50)

    elif model_type == "glove":
        if word2vec_retriever is None:
            return jsonify({"error": "Word2Vec (GloVe) retriever not initialized"}), 500

        # Search with Word2Vec (GloVe)
        results = word2vec_retriever.search(question, top_k=50)

    elif model_type == "colbert":
        # To be implemented
        return jsonify({"error": "ColBERT search not implemented yet"}), 501

    else:
        return jsonify({"error": f"Unknown search mode: {model_type}"}), 400

    # Get each document's highest scoring chunk
    doc_scores = {}
    for doc_id, chunk_id, score, title in results:
        if doc_id not in doc_scores or score > doc_scores[doc_id][1]:
            doc_scores[doc_id] = (chunk_id, score, title)

    # Sort by score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1][1], reverse=True)

    # Get top 5 results
    top_docs = sorted_docs[:5]

    # Collect context chunks for LLM
    context_chunks = []
    documents = []

    for doc_id, (chunk_id, score, title) in top_docs:
        chunk_key = (doc_id, chunk_id)
        if chunk_key in original_chunks:
            chunk = original_chunks[chunk_key]
            context_chunks.append(chunk["content"])

            # Format document for response
            documents.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "title": title,
                    "score": score,
                    "content": chunk["content"],
                }
            )

    # Generate answer with LLM
    if context_chunks:
        generated_answer = api_client.generate_answer(
            question=question,
            context=context_chunks[:3],  # Use top 3 most relevant chunks
            max_tokens=150,
            temperature=0.3,
        )
    else:
        generated_answer = (
            "Sorry, I couldn't find relevant information to answer this question."
        )

    # Format response like mockData.js
    response = {"answer": generated_answer, "documents": documents}

    return jsonify(response)


if __name__ == "__main__":
    # Make sure the app is created and everything is initialized before running
    app.run(debug=True, host="0.0.0.0", port=5000)
