"""Embedder: wraps BGE-M3 to produce dense + sparse vectors.

BGE-M3 is special — one model gives THREE vector types from one forward pass:
  - dense  (1024-dim float): captures semantic meaning, good for "what is X about"
  - sparse (bag of weighted tokens): captures exact keywords, good for "find paper 2401.xxxx"
  - colbert (multi-vector): most powerful but slow; we skip it for now.

Using BOTH dense + sparse is called hybrid retrieval. It beats either alone because
dense finds semantically similar text while sparse catches exact terms (model names,
numbers, arXiv IDs) that dense embeddings can miss. This is the interview story:
"naive vector search misses exact keyword matches; I measured the gap and fixed it
with hybrid retrieval using BGE-M3's native sparse output."

GPU note: BGE-M3 (~570M params) runs fine on one A100/L40S. The embedder
auto-detects GPU and falls back to CPU gracefully so the same code works on the
login node for small tests and on the compute node for the real index build.
"""
from __future__ import annotations

import torch
from FlagEmbedding import BGEM3FlagModel

from src.config import cfg


_model: BGEM3FlagModel | None = None   # module-level singleton


def _get_model() -> BGEM3FlagModel:
    global _model
    if _model is None:
        use_fp16 = torch.cuda.is_available()   # fp16 on GPU, fp32 on CPU
        print(f"[embedder] Loading {cfg.embedding['model_name']} "
              f"| GPU: {torch.cuda.is_available()} | fp16: {use_fp16}")
        _model = BGEM3FlagModel(
            cfg.embedding["model_name"],
            use_fp16=use_fp16,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    return _model


def embed_texts(
    texts: list[str],
    batch_size: int | None = None,
    show_progress: bool = True,
) -> dict[str, list]:
    """Embed a list of texts. Returns {'dense': [...], 'sparse': [...]}.

    Dense vectors  -> list of list[float], shape (N, 1024)
    Sparse vectors -> list of dict {token_id: weight}, ready for Qdrant sparse format
    """
    model = _get_model()
    bs = batch_size or cfg.embedding.get("batch_size", 32)

    output = model.encode(
        texts,
        batch_size=bs,
        max_length=cfg.embedding.get("max_length", 8192),
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    dense_vecs = output["dense_vecs"].tolist()          # list of 1024-dim lists
    sparse_vecs = output["lexical_weights"]             # list of {token_id: weight}

    return {"dense": dense_vecs, "sparse": sparse_vecs}


def embed_query(query: str) -> dict[str, list]:
    """Embed a single query string. Same output format as embed_texts."""
    return embed_texts([query], batch_size=1, show_progress=False)


if __name__ == "__main__":
    # python -m src.index.embedder
    # Run on a compute node (GPU) or login node (CPU, slower but works).
    texts = [
        "Title: HyperProtoSeg\nCategories: cs.CV\n\nA hyperbolic segmentation framework using SegFormer.",
        "Title: ViSTA-RS\nCategories: cs.CV\n\nMultimodal open-set recognition using CLIP and BLIP.",
        "The cat sat on the mat.",   # should score low against vision queries
    ]
    out = embed_texts(texts, show_progress=True)
    print(f"\nDense shape: {len(out['dense'])} x {len(out['dense'][0])}")
    print(f"Sparse example (first 5 tokens): "
          f"{dict(list(out['sparse'][0].items())[:5])}")

    # Sanity check: cosine similarity between dense vecs
    import numpy as np
    d = np.array(out["dense"])
    def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    print(f"\nSimilarity [HyperProtoSeg ↔ ViSTA-RS]: {cos(d[0], d[1]):.3f}  (should be high ~0.7+)")
    print(f"Similarity [HyperProtoSeg ↔ cat/mat]:  {cos(d[0], d[2]):.3f}  (should be low ~0.2)")
