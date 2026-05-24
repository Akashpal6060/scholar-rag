"""Reranker: cross-encoder that re-scores retrieved chunks against the query.

Why rerank?
The bi-encoder (BGE-M3) retrieves fast by comparing pre-computed vectors. But it
compares query and passage INDEPENDENTLY — it never sees them together. A
cross-encoder sees the full (query, passage) pair simultaneously, which is much
more accurate at judging relevance. We use it only on the top-50 candidates
(not the whole corpus) so it's fast enough for real-time use.

This step is what closes the gap between "retrieved the right paper" and "actually
answered correctly" — a common interview question. You'll have numbers to show:
"without reranking, faithfulness was X%; with reranker, it went to Y%."
"""
from __future__ import annotations

import torch
from FlagEmbedding import FlagReranker

from src.config import cfg


_reranker: FlagReranker | None = None


def _get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        use_fp16 = torch.cuda.is_available()
        print(f"[reranker] Loading {cfg.rerank['model_name']} | fp16: {use_fp16}")
        _reranker = FlagReranker(
            cfg.rerank["model_name"],
            use_fp16=use_fp16,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    return _reranker


def rerank(query: str, results: list[dict], top_k: int | None = None) -> list[dict]:
    """Reorder retrieved chunks by cross-encoder relevance score.

    Args:
        query:    the user's original question
        results:  output of hybrid_search() — list of {'score', 'payload', 'id'}
        top_k:    how many to keep after reranking (default from config)

    Returns:
        Reranked + truncated list, same dict format, with 'rerank_score' added.
    """
    if not cfg.rerank.get("enabled", True) or not results:
        return results[:top_k or cfg.rerank["top_k_final"]]

    reranker = _get_reranker()
    k = top_k or cfg.rerank["top_k_final"]

    # Build (query, passage) pairs — the cross-encoder scores these jointly.
    # Use the fullest text available: prefer 'text', fall back to title + snippet.
    def _passage(r: dict) -> str:
        p = r["payload"]
        text = p.get("text") or p.get("snippet") or ""
        title = p.get("title", "")
        return f"{title}. {text}"[:1024]

    pairs = [(query, _passage(r)) for r in results]

    scores = reranker.compute_score(pairs, normalize=True)
    if isinstance(scores, float):
        scores = [scores]

    # Attach rerank score and sort descending
    for i, r in enumerate(results):
        r["rerank_score"] = float(scores[i])

    reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:k]


if __name__ == "__main__":
    # python -m src.retrieve.reranker
    # Quick sanity check — does not need GPU, runs on CPU.
    dummy_results = [
        {"id": "a", "score": 0.9, "rerank_score": 0,
         "payload": {"snippet": "A hyperbolic segmentation framework for autonomous driving.", "title": "HyperProtoSeg"}},
        {"id": "b", "score": 0.85, "rerank_score": 0,
         "payload": {"snippet": "Multimodal open-set recognition using CLIP and BLIP for remote sensing.", "title": "ViSTA-RS"}},
        {"id": "c", "score": 0.8, "rerank_score": 0,
         "payload": {"snippet": "A recipe for chocolate cake with vanilla frosting.", "title": "Cake Recipe"}},
    ]
    query = "What are recent methods for domain adaptation in segmentation?"
    reranked = rerank(query, dummy_results, top_k=2)
    print(f"Query: {query}\n")
    for i, r in enumerate(reranked, 1):
        print(f"  [{i}] {r['payload']['title']} | rerank_score: {r['rerank_score']:.3f}")
