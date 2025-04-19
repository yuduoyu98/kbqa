# KBQA Backend

This is the backend server for the Knowledge Base Question Answering (KBQA) system.

## Features

The backend supports three different retrieval models:

1. **BM25**: Term-based retrieval using BM25 algorithm
2. **GloVe**: Vector-based retrieval using GloVe embeddings 
3. **ColBERT**: (Coming soon) Dense retrieval using the ColBERT model

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure the necessary indexes are built:

   - BM25 index at `indexes/bm25_index`
   - GloVe index at `indexes/word2vec_index.faiss` and `indexes/word2vec_metadata.json`

   If the indexes are not built, you can build them using the commands:

   ```bash
   # Build BM25 index
   python ../bm25_retriever.py
   
   # Build GloVe index
   python ../word2vec_retriever.py
   ```

3. Make sure your `config.toml` has the correct API key for SiliconFlow:

```toml
[api]
API_KEY = "your-api-key-here"
```

## Running the Server

### On Linux/Mac:

```bash
./run.sh
```

### On Windows:

```
run.bat
```

Or start the Flask development server directly:

```bash
python app.py
```

This will start the server at http://localhost:5000.

## API Endpoints

### Search Endpoint

**URL**: `/api/search`

**Method**: `POST`

**Request Body**:

```json
{
  "question": "What is email marketing?",
  "mode": "bm25"  // Options: "bm25", "glove", "colbert"
}
```

**Response**:

```json
{
  "answer": "Generated answer from LLM",
  "documents": [
    {
      "doc_id": 0,
      "chunk_id": 1,
      "title": "Document Title",
      "score": 0.785,
      "content": "Document content..."
    },
    // More document chunks...
  ]
}
```

## Retrieval Models

### BM25

BM25 is a term-based retrieval model that ranks documents based on the occurrence of query terms in each document. It considers both term frequency and document length.

### GloVe (Word2Vec)

The GloVe model uses word embeddings to capture semantic relationships between words. Documents are represented as the weighted average of word vectors.

### ColBERT (Coming Soon)

ColBERT is a neural retrieval model that performs late interaction between queries and documents using contextualized embeddings.

## Testing

You can test the BM25 retrieval and API integration using:

```bash
python test_bm25.py
```

This will test loading the BM25 model, searching for a sample query, and generating an answer with the LLM. 