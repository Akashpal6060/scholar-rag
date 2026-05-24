"""Prompt construction for grounded generation.

The whole point of RAG is that the LLM answers from RETRIEVED CONTEXT, not from
its own (possibly stale, possibly hallucinated) memory. The prompt enforces that:

  1. Answer ONLY from the provided papers - no outside knowledge.
  2. CITE every claim with [n] referring to the numbered source.
  3. If the context doesn't contain the answer, SAY SO - don't make it up.

Rule 3 is the anti-hallucination guardrail. It's also why low reranker scores are
useful: when nothing relevant is retrieved, the model refuses instead of inventing.
"""
from __future__ import annotations


SYSTEM_PROMPT = """You are ScholarRAG, a research assistant that answers questions \
about machine learning papers. Follow these rules strictly:

1. Answer ONLY using the provided paper excerpts. Do not use outside knowledge or \
make up information.
2. Cite every factual claim with a bracketed number like [1], [2] referring to the \
numbered sources.
3. If the provided papers do NOT contain enough information, say exactly: "I don't \
have enough information in the indexed papers to answer that confidently." Then \
optionally suggest what to search for instead.
4. Be concise and technical. Prefer accuracy over completeness.
5. Never invent paper titles, authors, or citation numbers."""


def build_context(results: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block."""
    blocks = []
    for i, r in enumerate(results, 1):
        p = r["payload"]
        title = p.get("title", "Unknown")
        authors_list = p.get("authors", [])
        authors = ", ".join(authors_list[:3])
        if len(authors_list) > 3:
            authors += " et al."
        published = p.get("published", "")[:10]
        text = p.get("text") or p.get("snippet") or ""
        blocks.append(
            f"[{i}] {title}\n"
            f"    Authors: {authors} | {published}\n"
            f"    Abstract: {text}"
        )
    return "\n\n".join(blocks)


def build_messages(query: str, results: list[dict]) -> list[dict]:
    """Build the full chat message list for the Groq API."""
    context = build_context(results) if results else "(No papers were retrieved.)"
    user_content = (
        f"Question: {query}\n\n"
        f"Retrieved paper excerpts:\n\n{context}\n\n"
        f"Answer using only these papers, citing sources with [n]."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


if __name__ == "__main__":
    fake = [{"payload": {
        "title": "HyperProtoSeg", "authors": ["Gole", "Pal"],
        "published": "2025-11-06", "text": "A hyperbolic segmentation framework."}}]
    for m in build_messages("What is HyperProtoSeg?", fake):
        print(f"--- {m['role']} ---\n{m['content']}\n")
