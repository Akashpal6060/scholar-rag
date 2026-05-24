"""Main indexing entrypoint — called by both SLURM scripts.

Usage:
    python -m src.index.build_index --mode full           # fetch all, rebuild
    python -m src.index.build_index --mode incremental --since 1d   # daily update

This is what runs inside the SLURM batch job on the GPU compute node.
On the login node you can run it with --max 10 for a quick smoke test (CPU, slow but works).
"""
from __future__ import annotations

import argparse
import time

from src.ingest.arxiv_client import fetch_papers, save_papers
from src.ingest.chunker import chunk_papers
from src.index.embedder import embed_texts
from src.index.vector_store import create_collection, upsert_chunks, collection_info


def run(mode: str, since_days: int | None, max_papers: int | None):
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"ScholarRAG Indexer | mode={mode} | since_days={since_days} | max={max_papers}")
    print(f"{'='*60}\n")

    # --- Step 1: Fetch papers ---
    print("Step 1/4 — Fetching papers from arXiv...")
    papers = list(fetch_papers(max_results=max_papers, since_days=since_days))
    print(f"  Fetched {len(papers)} papers.")
    if not papers:
        print("  Nothing to index. Exiting.")
        return
    save_papers(papers)

    # --- Step 2: Chunk ---
    print("\nStep 2/4 — Chunking...")
    chunks = chunk_papers(papers)
    print(f"  {len(papers)} papers → {len(chunks)} chunks.")

    # --- Step 3: Embed (GPU-accelerated if available) ---
    print("\nStep 3/4 — Embedding with BGE-M3 (dense + sparse)...")
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts, show_progress=True)
    print(f"  Embedded {len(texts)} chunks.")

    # --- Step 4: Upsert to Qdrant Cloud ---
    print("\nStep 4/4 — Upserting to Qdrant Cloud...")
    create_collection()          # idempotent — safe to call every run
    total = upsert_chunks(chunks, embeddings)

    info = collection_info()
    elapsed = time.time() - t0
    print(f"\n✓ Done in {elapsed:.1f}s")
    print(f"  Upserted: {total} points")
    print(f"  Collection total: {info['points_count']} points")


def _parse_since(s: str) -> int:
    """Convert '1d', '7d', '30d' → int days."""
    return int(s.rstrip("d"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ScholarRAG indexer")
    parser.add_argument("--mode", choices=["full", "incremental"], default="full")
    parser.add_argument("--since", type=str, default=None,
                        help="Only fetch papers this recent, e.g. '1d', '7d'")
    parser.add_argument("--max", type=int, default=None,
                        help="Override max_results from config (useful for quick tests)")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    since_days = _parse_since(args.since) if args.since else None
    run(mode=args.mode, since_days=since_days, max_papers=args.max)
