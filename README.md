# Real-Time AI Document Engine

A production-style RAG (Retrieval-Augmented Generation) pipeline for querying large document collections via natural language. Built with FAISS for vector search, FastAPI for serving, and WebSocket support for real-time interaction.

## Architecture

`
Documents --> Chunking --> Embeddings --> FAISS Index
                                            |
                                            v
User Query --> Embedding --> Vector Search --> Top-K Chunks --> LLM Prompt --> Response
                                                                                |
                                                                                v
                                                                    Evaluation Module
                                                              (relevance, groundedness, safety)
`

## Tech Stack

- **Retrieval**: FAISS (Facebook AI Similarity Search) for approximate nearest neighbor lookup over 50K+ document chunks
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2) via PyTorch
- **Generation**: LLM integration via LangChain for grounded, context-aware responses
- **API**: FastAPI with WebSocket support for real-time querying
- **Evaluation**: Custom retrieval relevance, groundedness, and safety scoring
- **Deployment**: Dockerized, designed for horizontal scaling across compute nodes

## Project Structure

`
src/
+-- ingestion/          # Document loading, chunking, preprocessing
¦   +-- loader.py       # Multi-format document loader (PDF, TXT, DOCX)
¦   +-- chunker.py      # Token-aware text chunking with overlap
¦   +-- preprocessor.py # Text cleaning and normalization
+-- indexing/           # Embedding generation + FAISS index management
¦   +-- embedder.py     # Sentence-Transformer embedding with GPU support
¦   +-- faiss_index.py  # FAISS index build, search, save/load
¦   +-- pipeline.py     # End-to-end indexing orchestrator
+-- query/              # RAG pipeline: retrieval --> prompt --> LLM response
¦   +-- prompt_builder.py  # Context-aware prompt construction
¦   +-- llm_client.py      # OpenAI API client with error handling
¦   +-- rag_engine.py      # Core RAG engine with similarity filtering
+-- evaluation/         # Response quality metrics
¦   +-- relevance.py    # Retrieval relevance scoring (precision@k)
¦   +-- groundedness.py # Hallucination detection via chunk-sentence matching
¦   +-- safety.py       # PII detection, response anomaly flagging
¦   +-- evaluator.py    # Unified evaluation pipeline
+-- api/                # FastAPI server + WebSocket handlers
    +-- server.py       # REST + WebSocket endpoints with lifecycle management
    +-- schemas.py      # Pydantic request/response models
`

## Setup

### Prerequisites
- Python 3.10+
- Docker (optional, for containerized deployment)

### Installation

`ash
git clone https://github.com/Preksha2/realtime-ai-document-engine.git
cd realtime-ai-document-engine
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # Add your OpenAI API key
`

### Running

`ash
# Start the API server
uvicorn src.api.server:app --reload --port 8000

# Or with Docker
docker-compose up --build
`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check and index status |
| POST | /query | Query documents via natural language |
| POST | /index | Build index from a document directory |
| WS | /ws/query | Real-time querying via WebSocket |

## Usage

### REST API
`python
import requests

# Build index
requests.post("http://localhost:8000/index", json={
    "source_dir": "./data/documents",
    "save_path": "data/faiss_index.bin"
})

# Query documents
response = requests.post("http://localhost:8000/query", json={
    "question": "What were the key findings in the Q3 report?",
    "top_k": 5
})
print(response.json()["answer"])
`

### WebSocket
`python
import asyncio
import websockets
import json

async def query():
    async with websockets.connect("ws://localhost:8000/ws/query") as ws:
        await ws.send(json.dumps({
            "question": "Summarize the main risks identified",
            "top_k": 5
        }))
        response = json.loads(await ws.recv())
        print(response["answer"])

asyncio.run(query())
`

## Testing

`ash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
`

## Evaluation

The pipeline includes built-in evaluation measuring three dimensions:

- **Retrieval Relevance** (40%): Cosine similarity between query and retrieved chunks, precision@k
- **Groundedness** (40%): Per-sentence verification against source chunks to detect hallucination
- **Safety** (20%): PII leak detection, hallucination signal flagging, response anomaly checks

Overall quality score is a weighted combination; responses must score >= 0.5 and pass safety checks.

## License

MIT
