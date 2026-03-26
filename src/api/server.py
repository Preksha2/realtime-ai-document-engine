"""
FastAPI application with REST and WebSocket endpoints
for real-time document querying.
"""
import os
import json
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.schemas import (
    QueryRequest, QueryResponse, HealthResponse,
    IndexRequest, IndexResponse,
)
from src.indexing import Embedder, FAISSIndex, IndexingPipeline
from src.query import RAGEngine


# --- Global state ---
rag_engine: Optional[RAGEngine] = None
faiss_index: Optional[FAISSIndex] = None
embedder: Optional[Embedder] = None


def load_config(path: str = "configs/config.yaml") -> dict:
    """Load application configuration from YAML."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    global rag_engine, faiss_index, embedder

    config = load_config()
    index_cfg = config["indexing"]
    query_cfg = config["query"]
    index_path = index_cfg["index_path"]

    # Initialize embedder
    embedder = Embedder(model_name=index_cfg["embedding_model"])

    # Load existing index if available
    if os.path.exists(index_path):
        faiss_index = FAISSIndex(embedding_dim=embedder.embedding_dim)
        faiss_index.load(index_path)
        rag_engine = RAGEngine(
            faiss_index=faiss_index,
            embedder=embedder,
            llm_model=query_cfg["llm_model"],
            temperature=query_cfg["temperature"],
            top_k=query_cfg["top_k"],
            similarity_threshold=query_cfg["similarity_threshold"],
            max_context_tokens=query_cfg["max_context_tokens"],
        )
        logger.info("RAG engine loaded with existing index")
    else:
        logger.warning(f"No index found at {index_path}. Use /index endpoint to build one.")

    yield

    logger.info("Shutting down application")


# --- App setup ---
app = FastAPI(
    title="Real-Time AI Document Engine",
    description="RAG pipeline for querying documents via natural language",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- REST endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and index status."""
    return HealthResponse(
        status="healthy",
        index_loaded=faiss_index is not None and faiss_index.is_trained,
        total_chunks=faiss_index.index.ntotal if faiss_index and faiss_index.index else 0,
    )


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents using natural language."""
    if rag_engine is None:
        raise HTTPException(
            status_code=503,
            detail="RAG engine not initialized. Build an index first via /index."
        )

    result = rag_engine.query(question=request.question, top_k=request.top_k)
    return QueryResponse(**result)


@app.post("/index", response_model=IndexResponse)
async def build_index(request: IndexRequest):
    """Trigger document indexing from a source directory."""
    global rag_engine, faiss_index

    config = load_config()
    index_cfg = config["indexing"]
    query_cfg = config["query"]

    try:
        pipeline = IndexingPipeline(
            source_dir=request.source_dir,
            chunk_size=config["ingestion"]["chunk_size"],
            chunk_overlap=config["ingestion"]["chunk_overlap"],
            embedding_model=index_cfg["embedding_model"],
            index_type=index_cfg["index_type"],
            nlist=index_cfg["nlist"],
        )
        faiss_index = pipeline.run(save_path=request.save_path)

        rag_engine = RAGEngine(
            faiss_index=faiss_index,
            embedder=pipeline.embedder,
            llm_model=query_cfg["llm_model"],
            temperature=query_cfg["temperature"],
            top_k=query_cfg["top_k"],
            similarity_threshold=query_cfg["similarity_threshold"],
            max_context_tokens=query_cfg["max_context_tokens"],
        )

        return IndexResponse(
            status="success",
            chunks_indexed=faiss_index.index.ntotal,
            index_path=request.save_path,
        )

    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


# --- WebSocket endpoint ---

class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self, max_connections: int = 50):
        self.active_connections: list[WebSocket] = []
        self.max_connections = max_connections

    async def connect(self, websocket: WebSocket):
        if len(self.active_connections) >= self.max_connections:
            await websocket.close(code=1013, reason="Max connections reached")
            return False
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected ({len(self.active_connections)} active)")
        return True

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected ({len(self.active_connections)} active)")

    async def send_json(self, websocket: WebSocket, data: dict):
        await websocket.send_text(json.dumps(data))


manager = ConnectionManager()


@app.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    """
    WebSocket endpoint for real-time document querying.

    Expects JSON messages: {"question": "...", "top_k": 5}
    Returns JSON responses with answer and sources.
    """
    connected = await manager.connect(websocket)
    if not connected:
        return

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if rag_engine is None:
                await manager.send_json(websocket, {
                    "error": "RAG engine not initialized. Build an index first."
                })
                continue

            question = message.get("question", "")
            top_k = message.get("top_k", 5)

            if not question:
                await manager.send_json(websocket, {"error": "Question cannot be empty"})
                continue

            # Run query in thread pool to avoid blocking
            result = await asyncio.to_thread(
                rag_engine.query, question=question, top_k=top_k
            )

            await manager.send_json(websocket, result)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except json.JSONDecodeError:
        await manager.send_json(websocket, {"error": "Invalid JSON format"})
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)
