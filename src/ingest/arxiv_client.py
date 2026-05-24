"""arXiv ingestion client.

Fetches paper metadata (title, abstract, authors, categories, date, links) from
the public arXiv API. No API key needed. Runs on the LOGIN node — pure network +
parsing, no GPU.

Two modes drive the rest of the system:
  - full:        grab a batch of recent papers to build the initial index
  - incremental: grab only papers from the last N days (the daily-freshness job)

Why this matters for the project story: the incremental path is what keeps the
knowledge base current, which is the whole reason RAG beats a fine-tuned model
for "what's the latest work on X" questions.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
from typing import Iterator

import arxiv

from src.config import cfg, ROOT


@dataclass
class Paper:
    """A single arXiv paper's metadata. One Paper -> one (or more) chunks later."""
    arxiv_id: str          # e.g. "2401.12345v1"
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: str         # ISO date string
    pdf_url: str
    abs_url: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _client() -> arxiv.Client:
    # Be polite: small page size, a delay between requests, a few retries.
    return arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)


def fetch_papers(
    categories: list[str] | None = None,
    max_results: int | None = None,
    since_days: int | None = None,
) -> Iterator[Paper]:
    """Yield Paper objects matching the given categories.

    Args:
        categories:  arXiv categories, e.g. ["cs.CV", "cs.LG"]. Defaults to config.
        max_results: cap on number of papers. Defaults to config.
        since_days:  if set, only papers published within the last N days
                     (this powers the daily incremental update).
    """
    categories = categories or cfg.arxiv["categories"]
    max_results = max_results or cfg.arxiv["max_results_per_run"]

    # arXiv query syntax: cat:cs.CV OR cat:cs.LG OR ...
    cat_query = " OR ".join(f"cat:{c}" for c in categories)

    search = arxiv.Search(
        query=cat_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,   # newest first
    )

    cutoff = None
    if since_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)

    for result in _client().results(search):
        # Stop early once we pass the date cutoff (results are newest-first).
        if cutoff is not None and result.published < cutoff:
            break

        yield Paper(
            arxiv_id=result.get_short_id(),
            title=result.title.strip().replace("\n", " "),
            abstract=result.summary.strip().replace("\n", " "),
            authors=[a.name for a in result.authors],
            categories=result.categories,
            published=result.published.isoformat(),
            pdf_url=result.pdf_url,
            abs_url=result.entry_id,
        )


def save_papers(papers: list[Paper], path: Path | None = None) -> Path:
    """Persist fetched papers to a JSONL file (one paper per line)."""
    path = path or (ROOT / "data" / "raw" / "papers.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in papers:
            f.write(p.to_json() + "\n")
    return path


def load_papers(path: Path | None = None) -> list[Paper]:
    """Load papers back from a JSONL file."""
    path = path or (ROOT / "data" / "raw" / "papers.jsonl")
    papers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            papers.append(Paper(**json.loads(line)))
    return papers


if __name__ == "__main__":
    # Smoke test: fetch a few papers and print them.
    #   python -m src.ingest.arxiv_client
    print("Fetching 5 recent papers to test the arXiv connection...\n")
    t0 = time.time()
    papers = list(fetch_papers(max_results=5))
    for i, p in enumerate(papers, 1):
        print(f"[{i}] {p.title}")
        print(f"    {p.arxiv_id} | {', '.join(p.categories)} | {p.published[:10]}")
        print(f"    {p.abstract[:140]}...\n")
    out = save_papers(papers)
    print(f"Fetched {len(papers)} papers in {time.time()-t0:.1f}s -> saved to {out}")
