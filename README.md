# Real-Time AI Document Engine

A production-style RAG (Retrieval-Augmented Generation) pipeline for querying event logs and operational documents via natural language. Built with FAISS for vector search, FastAPI for serving, and WebSocket support for real-time interaction.

Designed for operations and SRE teams to query incident logs, deployment records, and pipeline outputs without manual log review.

## Architecture

`
Event Logs --> Chunking --> Embeddings --> FAISS Index
                                             |
                                             v
User Query --> Embedding --> Vector Search --> Top-K Chunks --> LLM Prompt --> Response
                                                                                 |
                                                                                 v
                                                                   Evaluation Module
                                                        (relevance, groundedness, reliability, safety)
`

## Tech Stack

- **Retrieval**: FAISS (Facebook AI Similarity Search) over 50K+ document chunks
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2) via PyTorch
- **Generation**: Dual LLM backend -- HuggingFace (free, local) or OpenAI (optional)
- **API**: FastAPI with WebSocket support for real-time querying
- **Evaluation**: Retrieval relevance, groundedness, reliability, and safety scoring
- **Deployment**: Dockerized with Nginx load balancer for horizontal scaling across multiple compute nodes

## Quick Start (Demo)

Run the full pipeline end-to-end with sample event logs -- no API keys needed:

`ash
git clone https://github.com/Preksha2/realtime-ai-document-engine.git
cd realtime-ai-document-engine
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python scripts/demo.py
`

This will:
1. Ingest 5 sample event log files (payment, auth, deployment, ML pipeline, reconciliation)
2. Chunk and embed them into a FAISS index
3. Run 5 natural language queries against the logs
4. Evaluate each response for relevance, groundedness, and safety
5. Print a results summary

To use OpenAI instead:
`ash
export OPENAI_API_KEY=your_key
python scripts/demo.py --backend openai
`

## Scaling to 50K+ Chunks

Generate a large synthetic dataset for stress-testing:

`ash
python scripts/generate_large_dataset.py --num-entries 55000 --output data/large_events
`

This creates 55,000 realistic event log entries across 11 files, covering payment processing, authentication, deployments, ML pipelines, infrastructure monitoring, and security events.

## Benchmark: RAG vs Baseline

Compare RAG pipeline accuracy against a no-retrieval baseline:

`ash
python scripts/benchmark.py
`

This runs 8 queries with expected keyword matching, comparing:
- **Baseline**: LLM answers without any document context
- **RAG**: LLM answers grounded in retrieved event log chunks

Demonstrates significant accuracy improvement from retrieval augmentation.

## Horizontal Scaling

Deploy across multiple compute nodes with Docker + Nginx load balancer:

`ash
# Single node
docker-compose up --build

# Multi-node (4 workers behind Nginx)
docker-compose -f docker-compose.scale.yaml up --build --scale worker=4
`

Architecture:
`
Client --> Nginx (port 8000)
              |
              +--> Worker 1 (FastAPI + RAG engine)
              +--> Worker 2 (FastAPI + RAG engine)
              +--> Worker 3 (FastAPI + RAG engine)
              +--> Worker 4 (FastAPI + RAG engine)
              |
              +--> Shared FAISS index (mounted volume)
`

Each worker loads the FAISS index independently and handles queries. Nginx distributes requests via round-robin. WebSocket connections are proxied with upgrade support.

## Project Structure

`
src/
+-- ingestion/              # Document loading, chunking, preprocessing
¦   +-- loader.py           # Multi-format loader (PDF, TXT, DOCX)
¦   +-- chunker.py          # Token-aware chunking with overlap
¦   +-- preprocessor.py     # Text cleaning and normalization
+-- indexing/               # Embedding generation + FAISS index
¦   +-- embedder.py         # Sentence-Transformer embeddings (GPU support)
¦   +-- faiss_index.py      # FAISS build, search, save/load
¦   +-- pipeline.py         # End-to-end indexing orchestrator
+-- query/                  # RAG pipeline
¦   +-- prompt_builder.py   # Context-aware prompt construction
¦   +-- llm_client.py       # Multi-backend LLM client (HuggingFace + OpenAI)
¦   +-- rag_engine.py       # Core retrieval-augmented generation engine
+-- evaluation/             # Response quality metrics
¦   +-- relevance.py        # Retrieval relevance (precision@k, cosine similarity)
¦   +-- groundedness.py     # Hallucination detection via chunk-sentence matching
¦   +-- reliability.py      # Response consistency across multiple runs
¦   +-- safety.py           # PII detection, anomaly flagging
¦   +-- evaluator.py        # Unified evaluation pipeline
+-- api/                    # FastAPI server
    +-- server.py           # REST + WebSocket endpoints
    +-- schemas.py          # Pydantic request/response models
scripts/
+-- demo.py                 # End-to-end demo with sample event logs
+-- generate_large_dataset.py  # Synthetic 50K+ event log generator
+-- benchmark.py            # RAG vs baseline accuracy comparison
configs/
+-- config.yaml             # Application configuration
+-- nginx.conf              # Nginx load balancer config
data/
+-- sample_events/          # Sample event logs for demo
`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check and index status |
| POST | /query | Query event logs via natural language |
| POST | /index | Build index from a document directory |
| WS | /ws/query | Real-time querying via WebSocket |

## Usage

### REST API
`python
import requests

# Build index from event logs
requests.post("http://localhost:8000/index", json={
    "source_dir": "./data/sample_events",
    "save_path": "data/faiss_index.bin"
})

# Query
response = requests.post("http://localhost:8000/query", json={
    "question": "What payment failures occurred and how were they resolved?",
    "top_k": 5
})
print(response.json()["answer"])
`

### WebSocket
`python
import asyncio, websockets, json

async def query():
    async with websockets.connect("ws://localhost:8000/ws/query") as ws:
        await ws.send(json.dumps({
            "question": "Were there any security incidents?",
            "top_k": 5
        }))
        response = json.loads(await ws.recv())
        print(response["answer"])

asyncio.run(query())
`

## Evaluation

Four-dimensional evaluation measuring response quality:

- **Relevance** (30%): Cosine similarity between query and retrieved chunks
- **Groundedness** (30%): Per-sentence verification against source chunks
- **Reliability** (20%): Response consistency across repeated queries
- **Safety** (20%): PII leak detection, hallucination signals, response anomalies

Overall quality score is a weighted combination; responses must score >= 0.5 and pass safety checks.

## Testing

`ash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
`

## Configuration

Edit configs/config.yaml to customize:
- Chunk size and overlap
- Embedding model
- FAISS index type (Flat vs IVFFlat)
- LLM backend (huggingface or openai)
- Evaluation thresholds
- API settings

## License

MIT
