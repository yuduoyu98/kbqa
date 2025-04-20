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
from colbert_retriever import ColBERTRetriever
from word2vec_retriever import Word2VecRetriever

# Load configuration
with open("config.toml", "rb") as f:
    config = tomli.load(f)

# Get backend configuration
backend_config = config.get("backend", {})

# Get model loading controls
load_bm25 = backend_config.get(
    "load_bm25", True
)  # Default to True for backward compatibility
load_glove = backend_config.get("load_glove", True)
load_colbert = backend_config.get("load_colbert", True)

# Get index paths from backend configuration
bm25_index_path = backend_config.get("bm25_index_path", "indexes/bm25_index")
word2vec_index_path = backend_config.get(
    "word2vec_index_path", "indexes/word2vec_index.faiss"
)
word2vec_metadata_path = backend_config.get(
    "word2vec_metadata_path", "indexes/word2vec_metadata.json"
)
colbert_model_path = backend_config.get(
    "colbert_model_path", "indexes/colbert/colbert_model.pt"
)
colbert_index_path = backend_config.get(
    "colbert_index_path", "indexes/colbert/corpus_index.faiss"
)
colbert_ids_path = backend_config.get(
    "colbert_ids_path",
    "indexes/corpus_encodings/corpus_embeddings_20250420_114811_ids.json",
)

# Server configuration
host = backend_config.get("host", "0.0.0.0")
port = backend_config.get("port", 5000)
debug = backend_config.get("debug", True)

# Initialize retrievers
bm25_retriever = None
word2vec_retriever = None
colbert_retriever = None

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
    global bm25_retriever, word2vec_retriever, colbert_retriever

    # Initialize BM25 retriever if enabled
    if load_bm25:
        if os.path.exists(bm25_index_path):
            try:
                bm25_retriever = BM25()
                bm25_retriever.load(bm25_index_path)
                print(f"BM25 retriever initialized successfully from {bm25_index_path}")
            except Exception as e:
                print(f"Error initializing BM25 retriever: {e}")
        else:
            print(f"Warning: BM25 index not found at {bm25_index_path}")
    else:
        print("BM25 retriever loading disabled in config")

    # Initialize Word2Vec (GloVe) retriever if enabled
    if load_glove:
        if os.path.exists(word2vec_index_path) and os.path.exists(
            word2vec_metadata_path
        ):
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
    else:
        print("Word2Vec (GloVe) retriever loading disabled in config")

    # Initialize ColBERT retriever if enabled
    if load_colbert:
        if os.path.exists(colbert_index_path) and os.path.exists(colbert_ids_path):
            try:
                # Update config with correct paths
                config.setdefault("colbert", {})
                config["colbert"]["model_path"] = colbert_model_path
                config["colbert"]["faiss_index_path"] = colbert_index_path
                config["colbert"]["corpus_ids_path"] = colbert_ids_path

                # Initialize ColBERT retriever with config
                colbert_retriever = ColBERTRetriever(config)
                if colbert_retriever.load():
                    print(
                        f"ColBERT retriever initialized successfully from {colbert_index_path}"
                    )
                else:
                    colbert_retriever = None
                    print("Failed to initialize ColBERT retriever")
            except Exception as e:
                print(f"Error initializing ColBERT retriever: {e}")
                colbert_retriever = None
        else:
            print(f"Warning: ColBERT index not found at {colbert_index_path}")
    else:
        print("ColBERT retriever loading disabled in config")


def load_original_documents():
    """Load original documents to get chunk content"""
    global original_docs, original_chunks

    original_docs_path = "tmp/document_chunked_original.jsonl"

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


# Initialize API client
api_client = SiliconFlowClient("config.toml")

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
        results = bm25_retriever.search(question, top_k=3)

    elif model_type == "glove":
        if word2vec_retriever is None:
            return jsonify({"error": "Word2Vec (GloVe) retriever not initialized"}), 500

        # Search with Word2Vec (GloVe)
        results = word2vec_retriever.search(question, top_k=3)

    elif model_type == "colbert":
        if colbert_retriever is None:
            return jsonify({"error": "ColBERT retriever not initialized"}), 500

        # Search with ColBERT
        results = colbert_retriever.search(question, top_k=3)
    else:
        return jsonify({"error": f"Unknown search mode: {model_type}"}), 400

    # Get top 3 chunks directly, no need to group by document
    top_chunks = results[:3]  # Take the first 3 chunks directly

    # Collect context chunks for LLM
    context_chunks = []
    documents = []

    for doc_id, chunk_id, score, title in top_chunks:
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
        try:
            generated_answer = api_client.generate_answer(
                question=question,
                context=context_chunks,  # Use all top 3 chunks
                max_tokens=150,
                temperature=0.3,
            )
        except Exception as e:
            print(f"Error generating answer: {e}")
            generated_answer = "Sorry, there was an error generating an answer."
    else:
        generated_answer = (
            "Sorry, I couldn't find relevant information to answer this question."
        )

    # Format response like mockData.js
    response = {"answer": generated_answer, "documents": documents}

    return jsonify(response)


if __name__ == "__main__":
    # Use server configuration
    app.run(debug=debug, host=host, port=port)
