"""Retriever: the full retrieval pipeline in one function.

Orchestrates: embed query → hybrid search → rerank → return top-k chunks.
This is what pipeline.py calls for every user question.
"""
from __future__ import annotations

from src.index.embedder import embed_query
from src.index.vector_store import hybrid_search
from src.retrieve.reranker import rerank
from src.config import cfg


def retrieve(query: str, top_k_final: int | None = None) -> list[dict]:
    """End-to-end retrieval for a single query.

    Steps:
      1. Embed query with BGE-M3 (dense + sparse)
      2. Hybrid search Qdrant (RRF fusion, returns top-50)
      3. Cross-encoder rerank → top-8

    Returns list of result dicts with keys:
      - id, score, rerank_score, payload
        payload has: title, authors, published, abs_url, pdf_url, snippet
    """
    # Step 1: embed query
    q_embeddings = embed_query(query)
    dense_vec  = q_embeddings["dense"][0]
    sparse_vec = q_embeddings["sparse"][0]

    # Step 2: hybrid search → top-50 candidates
    candidates = hybrid_search(dense_vec=dense_vec, sparse_vec=sparse_vec)

    # Step 3: rerank → top-8
    k = top_k_final or cfg.rerank["top_k_final"]
    results = rerank(query=query, results=candidates, top_k=k)

    return results


if __name__ == "__main__":
    # python -m src.retrieve.retriever
    # Run AFTER you have papers indexed in Qdrant.
    query = "What are recent methods for test-time adaptation in semantic segmentation?"
    print(f"Query: {query}\n")
    results = retrieve(query)
    for i, r in enumerate(results, 1):
        p = r["payload"]
        print(f"[{i}] {p.get('title', 'N/A')}")
        print(f"     {p.get('published','')[:10]} | {', '.join(p.get('categories',[]))}")
        print(f"     Score: {r.get('rerank_score', r['score']):.3f}")
        print(f"     {p.get('abs_url','')}")
        print()
