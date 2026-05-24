"""Groq LLM client for grounded answer generation.

Groq is used because it's free and extremely fast (sub-second for Llama 3.3 70B).
This runs at QUERY time on whatever machine serves the app - including a free CPU
box like HuggingFace Spaces - because the heavy lifting happens on Groq's servers,
not locally. No GPU needed here.

Supports both streaming (for the UI, token-by-token) and non-streaming (for eval).
"""
from __future__ import annotations

from typing import Iterator

from groq import Groq

from src.config import cfg
from src.generate.prompt import build_messages


_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=cfg.groq_api_key)
    return _client


def generate(query: str, results: list[dict]) -> str:
    """Non-streaming: return the full grounded answer as a string."""
    messages = build_messages(query, results)
    resp = _get_client().chat.completions.create(
        model=cfg.generation["model_name"],
        messages=messages,
        temperature=cfg.generation.get("temperature", 0.1),
        max_tokens=cfg.generation.get("max_tokens", 1024),
    )
    return resp.choices[0].message.content


def generate_stream(query: str, results: list[dict]) -> Iterator[str]:
    """Streaming: yield answer tokens as they arrive (for a responsive UI)."""
    messages = build_messages(query, results)
    stream = _get_client().chat.completions.create(
        model=cfg.generation["model_name"],
        messages=messages,
        temperature=cfg.generation.get("temperature", 0.1),
        max_tokens=cfg.generation.get("max_tokens", 1024),
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


if __name__ == "__main__":
    # python -m src.generate.llm
    fake = [{"payload": {
        "title": "HyperProtoSeg: Hyperbolic Prototype Learning",
        "authors": ["Gole", "Pal"], "published": "2025-11-06",
        "text": "We propose a hyperbolic segmentation framework using SegFormer "
                "that learns class prototypes in the Poincare ball for domain-shift robustness."}}]
    print("Answer:\n")
    for tok in generate_stream("What is HyperProtoSeg and what does it use as a backbone?", fake):
        print(tok, end="", flush=True)
    print()
