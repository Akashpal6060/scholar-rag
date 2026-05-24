"""The end-to-end RAG pipeline: question -> retrieve -> generate -> cited answer.

This is the single function the API and UI call. It ties together the retriever
(embed + hybrid search + rerank) and the generator (grounded Groq answer), and
returns both the answer and the source papers so the UI can render citations.
"""
from __future__ import annotations

from src.retrieve.retriever import retrieve
from src.generate.llm import generate, generate_stream


def _format_sources(results: list[dict]) -> list[dict]:
    """Trim retrieval results to what the UI needs for citations."""
    sources = []
    for i, r in enumerate(results, 1):
        p = r["payload"]
        sources.append({
            "n": i,
            "title": p.get("title", "Unknown"),
            "authors": p.get("authors", []),
            "published": p.get("published", "")[:10],
            "categories": p.get("categories", []),
            "url": p.get("abs_url", ""),
            "score": round(r.get("rerank_score", r.get("score", 0)), 4),
        })
    return sources


def answer(query: str, top_k: int | None = None) -> dict:
    """Non-streaming end-to-end answer.

    Returns {"answer": str, "sources": list[dict]}.
    """
    results = retrieve(query, top_k_final=top_k)
    text = generate(query, results)
    return {"answer": text, "sources": _format_sources(results)}


def answer_stream(query: str, top_k: int | None = None):
    """Streaming version for the UI: yields tokens, then a final sources dict."""
    results = retrieve(query, top_k_final=top_k)
    for token in generate_stream(query, results):
        yield {"type": "token", "data": token}
    yield {"type": "sources", "data": _format_sources(results)}


if __name__ == "__main__":
    import json
    q = "What recent work explores test-time search or adaptation?"
    print(f"Q: {q}\n")
    out = answer(q)
    print("ANSWER:\n" + out["answer"] + "\n")
    print("SOURCES:")
    for s in out["sources"]:
        print(f"  [{s['n']}] {s['title']} ({s['score']}) - {s['url']}")
