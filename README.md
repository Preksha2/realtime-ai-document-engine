# Real-Time AI Document Engine

A production-style RAG (Retrieval-Augmented Generation) pipeline for querying large document collections via natural language. Built with FAISS for vector search, FastAPI for serving, and WebSocket support for real-time interaction.

## Architecture

```
Documents -> Chunking -> Embeddings -> FAISS Index
                                        |
User Query -> Embedding -> Vector Search -> Top-K Chunks -> LLM Prompt -> Response
                                                                        |
                                                              Evaluation Module
                                                        (relevance, groundedness)
```

## Tech Stack

- **Retrieval**: FAISS (Facebook AI Similarity Search) for approximate nearest neighbor lookup over 50K+ document chunks
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2) via PyTorch
- **Generation**: LLM integration via LangChain for grounded, context-aware responses
- **API**: FastAPI with WebSocket support for real-time querying
- **Evaluation**: Custom retrieval relevance, groundedness, and safety scoring
- **Deployment**: Dockerized, designed for horizontal scaling across compute nodes

## Project Structure

```
src/
+-- ingestion/      # Document loading, chunking, preprocessing
+-- indexing/        # Embedding generation + FAISS index management
+-- query/          # RAG pipeline: retrieval -> prompt -> LLM response
+-- evaluation/     # Response quality metrics
+-- api/            # FastAPI server + WebSocket handlers
```

## Setup

### Prerequisites
- Python 3.10+
- Docker (optional, for containerized deployment)

### Installation

```bash
git clone https://github.com/Preksha2/realtime-ai-document-engine.git
cd realtime-ai-document-engine
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
```

### Running

```bash
# Start the API server
uvicorn src.api.server:app --reload --port 8000

# Or with Docker
docker-compose up --build
```

## Usage

```python
import requests

response = requests.post("http://localhost:8000/query", json={
    "question": "What were the key findings in the Q3 report?",
    "top_k": 5
})
print(response.json()["answer"])
```

## Evaluation

The pipeline includes built-in evaluation for:
- **Retrieval Relevance**: Measures cosine similarity between query and retrieved chunks
- **Groundedness**: Checks if the LLM response is supported by retrieved context
- **Safety**: Filters for harmful or out-of-scope responses

## License

MIT
