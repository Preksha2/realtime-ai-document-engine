"""
End-to-end demo of the Real-Time AI Document Engine.
Runs the full pipeline: ingest sample event logs -> build index -> query -> evaluate.

Usage:
    python scripts/demo.py                    # Uses HuggingFace (free, no API key)
    python scripts/demo.py --backend openai   # Uses OpenAI (requires OPENAI_API_KEY)
"""
import os
import sys
import argparse
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from src.ingestion import DocumentLoader, TextChunker, TextPreprocessor
from src.indexing import Embedder, FAISSIndex, IndexingPipeline
from src.query import RAGEngine
from src.evaluation import ResponseEvaluator


SAMPLE_QUERIES = [
    "What payment failures occurred and how were they handled?",
    "Were there any security incidents or suspicious activity?",
    "What was the result of the ML model training job?",
    "Describe the deployment process and any issues encountered.",
    "How many transactions were processed in the reconciliation batch?",
]


def run_demo(backend: str = "huggingface", model: str = None):
    """Run the full RAG pipeline demo."""

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_events")

    if not os.path.exists(data_dir):
        logger.error(f"Sample data not found at {data_dir}")
        return

    # --- Step 1: Ingest ---
    logger.info("=" * 60)
    logger.info("STEP 1: Document Ingestion")
    logger.info("=" * 60)

    loader = DocumentLoader(data_dir)
    documents = loader.load_all()
    documents = TextPreprocessor.batch_clean(documents)

    chunker = TextChunker(chunk_size=256, chunk_overlap=30)
    chunks = chunker.chunk_documents(documents)

    logger.info(f"Ingested {len(documents)} event log files into {len(chunks)} chunks")

    # --- Step 2: Index ---
    logger.info("=" * 60)
    logger.info("STEP 2: Embedding + FAISS Indexing")
    logger.info("=" * 60)

    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.embed_texts(texts)

    faiss_index = FAISSIndex(embedding_dim=embedder.embedding_dim, index_type="Flat")
    faiss_index.build(embeddings, chunks)

    # Save index
    index_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "faiss_index.bin")
    faiss_index.save(index_path)

    # --- Step 3: Query ---
    logger.info("=" * 60)
    logger.info("STEP 3: RAG Querying")
    logger.info("=" * 60)

    # Determine LLM backend
    if backend == "openai" and not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set, falling back to HuggingFace")
        backend = "huggingface"

    default_model = model or ("gpt-3.5-turbo" if backend == "openai" else "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    rag_engine = RAGEngine(
        faiss_index=faiss_index,
        embedder=embedder,
        llm_model=default_model,
        llm_backend=backend,
        temperature=0.2,
        top_k=5,
        similarity_threshold=0.3,
        max_context_tokens=1024,
    )

    results = []
    for i, query in enumerate(SAMPLE_QUERIES, 1):
        logger.info(f"\n--- Query {i}/{len(SAMPLE_QUERIES)} ---")
        logger.info(f"Q: {query}")
        result = rag_engine.query(query)
        logger.info(f"A: {result['answer'][:200]}...")
        logger.info(f"   Sources: {result['chunks_used']} chunks used from {result['chunks_retrieved']} retrieved")
        results.append({"query": query, **result})

    # --- Step 4: Evaluate ---
    logger.info("=" * 60)
    logger.info("STEP 4: Evaluation")
    logger.info("=" * 60)

    evaluator = ResponseEvaluator(
        embedder=embedder,
        relevance_threshold=0.5,
        groundedness_threshold=0.4,
        enable_safety=True,
        reliability_runs=2,
    )

    eval_results = []
    for r in results:
        # Reconstruct retrieved chunks for evaluation
        query_embedding = embedder.embed_query(r["query"])
        retrieved = faiss_index.search(query_embedding, top_k=5)

        evaluation = evaluator.evaluate(
            query=r["query"],
            answer=r["answer"],
            retrieved_chunks=retrieved,
        )
        eval_results.append({
            "query": r["query"],
            "quality_score": evaluation["quality_score"],
            "relevance": evaluation["relevance"]["mean_score"],
            "groundedness": evaluation["groundedness"]["overall_score"],
            "safety": evaluation["safety"]["is_safe"],
            "pass": evaluation["pass"],
        })

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    for er in eval_results:
        status = "PASS" if er["pass"] else "FAIL"
        logger.info(
            f"[{status}] Q: {er['query'][:60]}... | "
            f"quality={er['quality_score']:.3f} | "
            f"relevance={er['relevance']:.3f} | "
            f"groundedness={er['groundedness']:.3f} | "
            f"safe={er['safety']}"
        )

    avg_quality = sum(er["quality_score"] for er in eval_results) / len(eval_results)
    pass_rate = sum(1 for er in eval_results if er["pass"]) / len(eval_results)

    logger.info(f"\nAverage Quality Score: {avg_quality:.4f}")
    logger.info(f"Pass Rate: {pass_rate:.0%}")
    logger.info(f"Total Chunks Indexed: {faiss_index.index.ntotal}")
    logger.info(f"LLM Backend: {backend} ({default_model})")

    # Save results
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "demo_results.json")
    with open(output_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG Document Engine demo")
    parser.add_argument("--backend", choices=["huggingface", "openai"], default="huggingface",
                        help="LLM backend (default: huggingface)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model name")
    args = parser.parse_args()
    run_demo(backend=args.backend, model=args.model)
