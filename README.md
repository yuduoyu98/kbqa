# KBQA - Knowledge Base Question Answering System

A question answering system with multiple retrieval models and a modern React frontend.

## Deployment Instructions

### Prerequisites

- Python 3.10.16
- Node.js 22.13.1
- npm 10.9.2+

### Installation

#### Backend Setup

1. Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure your `config.toml` has the correct API key for SiliconFlow:

```toml
[api]
API_KEY = "your-api-key-here"
```

#### Frontend Setup

1. Navigate to the frontend directory:

```bash
cd frontend
```

2. Install the required npm packages:

```bash
npm install
```

### Starting the Application

#### 1. Start the Backend Server

```bash
python app.py
```

This will start the Flask server at http://localhost:5000.

Note: you need to wait until model loaded which will take a few minitus.  

#### 2. Start the Frontend Development Server

In a separate terminal:

```bash
cd frontend
npm start
```

This will start the React development server at http://localhost:3000.

## Retrieval Models and Prediction Files

The system supports three different retrieval models:

### 1. BM25 (Term-based retrieval)

- **Index Location**: [bm25 index](indexes/bm25_index)
- **Predict result**: [bm25 prediction](result/bm25_val_predict.jsonl)
- **Reproduce Prediction**:

```bash
python bm25_retriever.py
```

### 2. GloVe (Word2Vec embedding-based retrieval)

- **Index Locations** (6B 300d Glove): 
  - [FAISS index](indexes/word2vec_index.faiss)
  - [metadata](indexes/word2vec_metadata.json)
- **Predict result**: [GloVe 6B 300d prediction](result/300_word2vec_val_predict.jsonl)  
- **Reproduce Prediction**:

```bash
python word2vec_retriever.py
```

### 3. ColBERT (Dense retrieval with contextualized embeddings)

- **Index Locations**:
  - [FAISS index](indexes/colbert/corpus_index.faiss)
  - [model](indexes/colbert/colbert_model.pt)
  - [corpus embeddings](indexes/colbert/corpus/)
- **Predict result**: [ColBERT prediction](result/colbert_val_predict.jsonl)
- **Reproduce Prediction**:

```bash
python colBERT/predict_colbert.py
```

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
    }
  ]
}
``` 