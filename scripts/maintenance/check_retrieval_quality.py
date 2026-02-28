"""
Simple retrieval quality smoke test.

Usage:
  python check_retrieval_quality.py
  python check_retrieval_quality.py --source data/rules/Tembo_Rules_En_C_2026.pdf --query "Tembo rules"
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

from core.rag_engine import RAGEngine


def normalize_source(source: str) -> str:
    return source.replace("\\", "/")


def build_source_chunks(rag_engine: RAGEngine) -> Dict[str, List[str]]:
    by_source: Dict[str, List[str]] = {}
    for doc, meta in zip(rag_engine.vector_store.documents, rag_engine.vector_store.metadatas):
        source = normalize_source(str(meta.get("source", "")))
        by_source.setdefault(source, []).append(doc)
    return by_source


def build_query_from_source(source: str, first_chunk: str) -> str:
    stem = Path(source).stem
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"\b(rules?|rulebook|v\d+(\.\d+)?)\b", " ", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\s+", " ", stem).strip()
    if stem:
        return f"{stem} rules 规则"

    snippet = re.sub(r"\s+", " ", first_chunk[:60]).strip()
    return snippet if snippet else "游戏规则"


def run_one_query(
    rag_engine: RAGEngine,
    query: str,
    expected_source: str,
    top_k: int,
    min_score: float,
) -> Tuple[bool, List[str], float]:
    results, _ = rag_engine.retrieve(query=query, top_k=top_k, min_score=min_score)
    sources = [normalize_source(str(r.get("metadata", {}).get("source", ""))) for r in results]
    top_score = results[0]["score"] if results else 0.0
    hit = expected_source in sources
    return hit, sources, top_score


def main():
    parser = argparse.ArgumentParser(description="Run retrieval smoke checks")
    parser.add_argument("--source", type=str, default="", help="Expected source file path")
    parser.add_argument("--query", type=str, default="", help="Query text for source check")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k for retrieval checks")
    parser.add_argument("--min-score", type=float, default=0.3, help="Relevance threshold")
    args = parser.parse_args()

    rag_engine = RAGEngine()
    by_source = build_source_chunks(rag_engine)

    if not by_source:
        print("Vector store is empty.")
        return

    print(f"Sources: {len(by_source)}")
    print(f"Chunks: {len(rag_engine.vector_store.documents)}")
    print("-" * 72)

    if args.source and args.query:
        expected = normalize_source(args.source)
        hit, sources, top_score = run_one_query(
            rag_engine, args.query, expected, args.top_k, args.min_score
        )
        print(f"Query: {args.query}")
        print(f"Expected source: {expected}")
        print(f"Hit: {'YES' if hit else 'NO'}")
        print(f"Top score: {top_score:.4f}")
        print("Top sources:")
        for s in sources:
            print(f"  - {s}")
        return

    # Auto mode: one generated query per source.
    checks = []
    for source, chunks in sorted(by_source.items()):
        query = build_query_from_source(source, chunks[0] if chunks else "")
        hit, _, top_score = run_one_query(
            rag_engine, query, source, args.top_k, args.min_score
        )
        checks.append((source, len(chunks), query, hit, top_score))

    hit_count = sum(1 for c in checks if c[3])
    total = len(checks)
    print(f"Auto checks: {hit_count}/{total} source hits ({(hit_count / total) * 100:.1f}%)")
    print("-" * 72)
    print("Per-source summary:")
    for source, chunk_count, query, hit, top_score in checks:
        status = "PASS" if hit else "FAIL"
        print(f"[{status}] chunks={chunk_count:4d} score={top_score:6.3f} source={source}")
        print(f"       query={query}")


if __name__ == "__main__":
    main()
