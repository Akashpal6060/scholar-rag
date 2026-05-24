"""Chunking: turn papers into the units we embed and retrieve.

For v1 we ingest ABSTRACTS, so each paper becomes a single chunk. We prepend a
short contextual prefix (title + categories) to each chunk's text before
embedding — a cheap version of "contextual retrieval" that measurably improves
recall, because the embedding then captures what the passage is *about*, not
just the bare abstract text.

When you later switch to full-text (config: ingest_full_text: true), this is
where you'd split a long paper into multiple overlapping chunks. The Chunk
interface stays identical, so nothing downstream changes.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from src.config import cfg
from src.ingest.arxiv_client import Paper


@dataclass
class Chunk:
    """One retrievable unit. `text` is what gets embedded; `payload` is metadata
    stored alongside the vector for filtering and for showing citations."""
    chunk_id: str               # stable unique id (hash) -> enables idempotent upserts
    text: str                   # the text we embed
    payload: dict = field(default_factory=dict)


def _stable_id(paper_id: str, idx: int) -> str:
    """Deterministic id so re-ingesting the same paper overwrites, not duplicates."""
    raw = f"{paper_id}::{idx}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def _contextual_prefix(paper: Paper) -> str:
    """Cheap context injection: tell the embedder what this chunk is about."""
    cats = ", ".join(paper.categories[:3])
    return f"Title: {paper.title}\nCategories: {cats}\n\n"


def chunk_paper(paper: Paper) -> list[Chunk]:
    """Convert one Paper into one or more Chunks."""
    add_prefix = cfg.chunking.get("add_contextual_prefix", True)
    prefix = _contextual_prefix(paper) if add_prefix else ""

    # v1: abstracts only -> a single chunk per paper.
    if not cfg.arxiv.get("ingest_full_text", False):
        text = prefix + paper.abstract
        payload = {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": paper.authors,
            "categories": paper.categories,
            "published": paper.published,
            "abs_url": paper.abs_url,
            "pdf_url": paper.pdf_url,
            "text": paper.abstract,
            "snippet": paper.abstract[:300],
        }
        return [Chunk(chunk_id=_stable_id(paper.arxiv_id, 0), text=text, payload=payload)]

    # v2 placeholder: full-text splitting would go here (recursive/semantic split).
    raise NotImplementedError(
        "Full-text chunking is a v2 feature. Set arxiv.ingest_full_text: false for now."
    )


def chunk_papers(papers: list[Paper]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in papers:
        chunks.extend(chunk_paper(p))
    return chunks


if __name__ == "__main__":
    # python -m src.ingest.chunker  (run the arxiv_client smoke test first)
    from src.ingest.arxiv_client import load_papers

    papers = load_papers()
    chunks = chunk_papers(papers)
    print(f"{len(papers)} papers -> {len(chunks)} chunks\n")
    c = chunks[0]
    print("Example chunk id:", c.chunk_id)
    print("Embedded text (first 200 chars):")
    print(c.text[:200])
    print("\nPayload keys:", list(c.payload.keys()))
