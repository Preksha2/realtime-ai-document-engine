"""
Benchmark script comparing RAG pipeline vs naive (no-retrieval) baseline.
Demonstrates accuracy improvement from retrieval-augmented generation.

Usage:
    python scripts/benchmark.py
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from src.ingestion import DocumentLoader, TextChunker, TextPreprocessor
from src.indexing import Embedder, FAISSIndex
from src.query import RAGEngine, PromptBuilder, LLMClient
from src.evaluation import ResponseEvaluator


BENCHMARK_QUERIES = [
    {
        "question": "What payment failures occurred and how were they handled?",
        "expected_keywords": ["timeout", "retry", "fallback", "secondary", "TXN"],
    },
    {
        "question": "Were there any security incidents or suspicious activity?",
        "expected_keywords": ["brute force", "locked", "suspicious", "rate limit", "blocked"],
    },
    {
        "question": "What was the result of the ML model training job?",
        "expected_keywords": ["AUC", "training", "epoch", "XGBoost", "fraud_detector"],
    },
    {
        "question": "Describe the deployment process and any issues encountered.",
        "expected_keywords": ["rolling", "pod", "readiness", "deployment", "user-service"],
    },
    {
        "question": "How many transactions were processed in the reconciliation batch?",
        "expected_keywords": ["48,291", "reconciliation", "matched", "discrepancy"],
    },
    {
        "question": "What auto-scaling events occurred and why?",
        "expected_keywords": ["scale-up", "CPU", "replicas", "threshold", "scale-down"],
    },
    {
        "question": "Which users had failed login attempts?",
        "expected_keywords": ["U-3301", "failed", "login", "invalid_password", "locked"],
    },
    {
        "question": "What was the system uptime and health status?",
        "expected_keywords": ["healthy", "uptime", "99.97", "active_connections"],
    },
]


def keyword_accuracy(answer: str, expected_keywords: list) -> float:
    """Calculate what fraction of expected keywords appear in the answer."""
    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return found / len(expected_keywords) if expected_keywords else 0.0


def run_benchmark():
    """Run RAG vs baseline benchmark."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_events")

    # --- Setup: Build index ---
    logger.info("Setting up benchmark...")

    loader = DocumentLoader(data_dir)
    documents = loader.load_all()
    documents = TextPreprocessor.batch_clean(documents)

    chunker = TextChunker(chunk_size=256, chunk_overlap=30)
    chunks = chunker.chunk_documents(documents)

    embedder = Embedder(model_name="all-MiniLM-L6-v2")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.embed_texts(texts)

    faiss_index = FAISSIndex(embedding_dim=embedder.embedding_dim, index_type="Flat")
    faiss_index.build(embeddings, chunks)

    # --- Baseline: LLM without retrieval ---
    logger.info("=" * 60)
    logger.info("BASELINE: LLM without retrieval (no RAG)")
    logger.info("=" * 60)

    llm_client = LLMClient(backend="huggingface", model="Qwen/Qwen2.5-0.5B-Instruct")
    prompt_builder = PromptBuilder(max_context_tokens=1024)

    baseline_results = []
    for q in BENCHMARK_QUERIES:
        # Query LLM directly without any context
        messages = [
            {"role": "system", "content": "Answer the question based on your knowledge. If you don't know, say so."},
            {"role": "user", "content": q["question"]},
        ]
        start = time.time()
        response = llm_client.generate(messages)
        elapsed = time.time() - start

        accuracy = keyword_accuracy(response["answer"], q["expected_keywords"])
        baseline_results.append({
            "question": q["question"],
            "answer": response["answer"][:200],
            "keyword_accuracy": round(accuracy, 4),
            "latency_s": round(elapsed, 2),
        })
        logger.info(f"  Q: {q['question'][:60]}... | accuracy={accuracy:.2f} | latency={elapsed:.1f}s")

    baseline_avg = sum(r["keyword_accuracy"] for r in baseline_results) / len(baseline_results)

    # --- RAG Pipeline ---
    logger.info("=" * 60)
    logger.info("RAG PIPELINE: LLM with retrieval")
    logger.info("=" * 60)

    rag_engine = RAGEngine(
        faiss_index=faiss_index,
        embedder=embedder,
        llm_model="Qwen/Qwen2.5-0.5B-Instruct",
        llm_backend="huggingface",
        temperature=0.2,
        top_k=5,
        similarity_threshold=0.3,
        max_context_tokens=1024,
    )

    rag_results = []
    for q in BENCHMARK_QUERIES:
        start = time.time()
        result = rag_engine.query(q["question"])
        elapsed = time.time() - start

        accuracy = keyword_accuracy(result["answer"], q["expected_keywords"])
        rag_results.append({
            "question": q["question"],
            "answer": result["answer"][:200],
            "keyword_accuracy": round(accuracy, 4),
            "chunks_used": result["chunks_used"],
            "latency_s": round(elapsed, 2),
        })
        logger.info(f"  Q: {q['question'][:60]}... | accuracy={accuracy:.2f} | chunks={result['chunks_used']} | latency={elapsed:.1f}s")

    rag_avg = sum(r["keyword_accuracy"] for r in rag_results) / len(rag_results)

    # --- Evaluation comparison ---
    logger.info("=" * 60)
    logger.info("RAG EVALUATION METRICS")
    logger.info("=" * 60)

    evaluator = ResponseEvaluator(
        embedder=embedder,
        relevance_threshold=0.5,
        groundedness_threshold=0.4,
        enable_safety=True,
        reliability_runs=2,
    )

    rag_eval_scores = []
    for q, r in zip(BENCHMARK_QUERIES, rag_results):
        query_embedding = embedder.embed_query(q["question"])
        retrieved = faiss_index.search(query_embedding, top_k=5)
        full_result = rag_engine.query(q["question"])
        evaluation = evaluator.evaluate(
            query=q["question"],
            answer=full_result["answer"],
            retrieved_chunks=retrieved,
        )
        rag_eval_scores.append(evaluation["quality_score"])

    avg_quality = sum(rag_eval_scores) / len(rag_eval_scores)

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 60)

    improvement = rag_avg - baseline_avg
    improvement_pct = (improvement / baseline_avg * 100) if baseline_avg > 0 else 0

    logger.info(f"Baseline (no retrieval) avg keyword accuracy: {baseline_avg:.4f}")
    logger.info(f"RAG pipeline avg keyword accuracy:            {rag_avg:.4f}")
    logger.info(f"Absolute improvement:                         +{improvement:.4f}")
    logger.info(f"Relative improvement:                         +{improvement_pct:.1f}%")
    logger.info(f"RAG avg quality score (eval suite):           {avg_quality:.4f}")
    logger.info("")

    logger.info("Per-query comparison:")
    logger.info(f"{'Query':<55} | {'Baseline':>8} | {'RAG':>8} | {'Delta':>8}")
    logger.info("-" * 90)
    for b, r in zip(baseline_results, rag_results):
        delta = r["keyword_accuracy"] - b["keyword_accuracy"]
        logger.info(f"{b['question'][:55]:<55} | {b['keyword_accuracy']:>8.2f} | {r['keyword_accuracy']:>8.2f} | {delta:>+8.2f}")

    # Save results
    output = {
        "baseline_avg_accuracy": round(baseline_avg, 4),
        "rag_avg_accuracy": round(rag_avg, 4),
        "improvement_absolute": round(improvement, 4),
        "improvement_relative_pct": round(improvement_pct, 1),
        "rag_avg_quality_score": round(avg_quality, 4),
        "baseline_results": baseline_results,
        "rag_results": rag_results,
    }

    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    run_benchmark()
