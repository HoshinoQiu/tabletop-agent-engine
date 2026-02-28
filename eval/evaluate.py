"""
Evaluation script for the tabletop game rules Q&A system.
Measures retrieval recall and answer relevance.
"""

import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_dataset(path: str = None) -> dict:
    """Load evaluation dataset."""
    if path is None:
        path = Path(__file__).parent / "eval_dataset.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval(rag_engine, questions: list, top_k: int = 5) -> dict:
    """Evaluate retrieval quality."""
    results = {
        "total": len(questions),
        "relevant_found": 0,
        "avg_top_score": 0.0,
        "avg_results_count": 0.0,
        "keyword_hit_rate": 0.0,
        "details": [],
    }

    total_score = 0.0
    total_results = 0
    keyword_hits = 0

    for q in questions:
        query = q["query"]
        expected_keywords = q.get("expected_keywords", [])

        start_time = time.time()
        retrieved, is_relevant = rag_engine.retrieve(query, top_k=top_k, min_score=0.3)
        elapsed = time.time() - start_time

        top_score = retrieved[0]["score"] if retrieved else 0.0
        total_score += top_score
        total_results += len(retrieved)

        if is_relevant:
            results["relevant_found"] += 1

        # Check keyword hits in retrieved documents
        all_text = " ".join(r["document"] for r in retrieved).lower()
        hits = sum(1 for kw in expected_keywords if kw.lower() in all_text)
        keyword_ratio = hits / len(expected_keywords) if expected_keywords else 0
        keyword_hits += keyword_ratio

        results["details"].append({
            "id": q["id"],
            "query": query,
            "category": q.get("category", ""),
            "top_score": round(top_score, 4),
            "results_count": len(retrieved),
            "is_relevant": is_relevant,
            "keyword_hit_ratio": round(keyword_ratio, 2),
            "latency_ms": round(elapsed * 1000, 1),
        })

    results["avg_top_score"] = round(total_score / len(questions), 4) if questions else 0
    results["avg_results_count"] = round(total_results / len(questions), 1) if questions else 0
    results["keyword_hit_rate"] = round(keyword_hits / len(questions), 4) if questions else 0
    results["recall_at_k"] = round(results["relevant_found"] / len(questions), 4) if questions else 0

    return results


def main():
    """Run evaluation."""
    from core.rag_engine import RAGEngine

    print("=" * 60)
    print("Tabletop Agent Engine - Evaluation")
    print("=" * 60)

    # Load dataset
    dataset = load_dataset()
    questions = dataset["questions"]
    print(f"Loaded {len(questions)} evaluation questions")

    # Initialize RAG engine
    print("Initializing RAG engine...")
    rag_engine = RAGEngine()
    print(f"Vector store has {rag_engine.vector_store.index.ntotal} embeddings")

    if rag_engine.vector_store.index.ntotal == 0:
        print("ERROR: Vector store is empty. Please run init_all_rules.py first.")
        sys.exit(1)

    # Run evaluation
    print("\nRunning retrieval evaluation...")
    results = evaluate_retrieval(rag_engine, questions)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total questions:     {results['total']}")
    print(f"Relevant found:      {results['relevant_found']} ({results['recall_at_k']:.1%})")
    print(f"Avg top score:       {results['avg_top_score']:.4f}")
    print(f"Avg results count:   {results['avg_results_count']:.1f}")
    print(f"Keyword hit rate:    {results['keyword_hit_rate']:.1%}")
    print()

    # Print per-category breakdown
    categories = {}
    for d in results["details"]:
        cat = d["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "relevant": 0, "scores": []}
        categories[cat]["total"] += 1
        if d["is_relevant"]:
            categories[cat]["relevant"] += 1
        categories[cat]["scores"].append(d["top_score"])

    print("Per-category breakdown:")
    for cat, data in sorted(categories.items()):
        avg_score = sum(data["scores"]) / len(data["scores"])
        recall = data["relevant"] / data["total"]
        print(f"  {cat:20s}: recall={recall:.0%}, avg_score={avg_score:.4f} ({data['total']} questions)")

    # Save detailed results
    output_path = Path(__file__).parent / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
