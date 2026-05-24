"""Vector store: all Qdrant operations in one place.

Three responsibilities:
  1. create_collection()  — idempotent setup (safe to call multiple times)
  2. upsert_chunks()      — store chunks with dense + sparse vectors + metadata payload
  3. hybrid_search()      — retrieve top-k chunks using dense + sparse fusion (RRF)

Why Qdrant?
  - Native hybrid search (dense + sparse in one query) without extra libraries
  - Payload filtering: "only search papers from 2025 onwards in cs.CV"
  - Free cloud tier, production-grade API, Python client is clean

The upsert uses chunk_id as the point ID so re-running the indexer on the same
paper overwrites the existing vector instead of creating a duplicate. This is what
makes the daily incremental update safe: idempotent upserts.
"""
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, SparseIndexParams,
    PointStruct, SparseVector, NamedVector, NamedSparseVector,
    SearchRequest, Query, FusionQuery, Fusion, Prefetch,
)

from src.config import cfg


COLLECTION = cfg.vector_store["collection_name"]
DENSE_DIM  = cfg.vector_store["dense_dim"]          # 1024 for BGE-M3


def _client() -> QdrantClient:
    return QdrantClient(
        url=cfg.qdrant_url,
        api_key=cfg.qdrant_api_key,
        timeout=60,
    )


def create_collection(recreate: bool = False) -> None:
    """Create the Qdrant collection with dense + sparse vector configs.

    Safe to call repeatedly — skips creation if the collection already exists,
    unless recreate=True (which wipes and rebuilds from scratch).
    """
    client = _client()
    exists = any(c.name == COLLECTION for c in client.get_collections().collections)

    if exists and not recreate:
        print(f"[vector_store] Collection '{COLLECTION}' already exists — skipping create.")
        return

    if exists and recreate:
        client.delete_collection(COLLECTION)
        print(f"[vector_store] Deleted existing collection '{COLLECTION}'.")

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)},
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
        },
    )
    print(f"[vector_store] Created collection '{COLLECTION}' "
          f"(dense={DENSE_DIM}d, sparse=enabled).")


def upsert_chunks(chunks, embeddings: dict[str, list]) -> int:
    """Store chunks with their embeddings in Qdrant.

    Args:
        chunks:      list of Chunk objects
        embeddings:  output of embedder.embed_texts() — {'dense': [...], 'sparse': [...]}

    Returns:
        number of points upserted
    """
    client = _client()

    points = []
    for i, chunk in enumerate(chunks):
        dense_vec  = embeddings["dense"][i]
        sparse_raw = embeddings["sparse"][i]   # dict {token_id (str): weight}

        # Qdrant expects integer keys for sparse vectors
        sparse_indices = [int(k) for k in sparse_raw.keys()]
        sparse_values  = [float(v) for v in sparse_raw.values()]

        points.append(PointStruct(
            id=chunk.chunk_id,       # stable MD5 hex string — Qdrant accepts str IDs
            vector={
                "dense":  dense_vec,
                "sparse": SparseVector(indices=sparse_indices, values=sparse_values),
            },
            payload=chunk.payload,
        ))

    # Upsert in batches of 64 to avoid payload size limits
    batch_size = 64
    total = 0
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        client.upsert(collection_name=COLLECTION, points=batch)
        total += len(batch)
        print(f"[vector_store] Upserted {total}/{len(points)} points...")

    return total


def hybrid_search(
    dense_vec: list[float],
    sparse_vec: dict,
    top_k: int | None = None,
    filter_: dict | None = None,
) -> list[dict]:
    """Hybrid search: fuse dense (semantic) + sparse (keyword) with Reciprocal Rank Fusion.

    Why RRF?
    Dense alone: misses exact keyword matches (model names, arXiv IDs, numbers).
    Sparse alone: misses paraphrase / synonym matches.
    RRF fuses the ranked lists from both without needing to tune score weights.

    Returns list of dicts with 'score', 'payload', 'id'.
    """
    client = _client()
    k = top_k or cfg.retrieval["top_k_retrieve"]

    sparse_indices = [int(ki) for ki in sparse_vec.keys()]
    sparse_values  = [float(v) for v in sparse_vec.values()]

    results = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=k),
            Prefetch(
                query=SparseVector(indices=sparse_indices, values=sparse_values),
                using="sparse",
                limit=k,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=k,
        with_payload=True,
    )

    return [
        {"id": str(r.id), "score": r.score, "payload": r.payload}
        for r in results.points
    ]


def collection_info() -> dict:
    client = _client()
    info = client.get_collection(COLLECTION)
    return {
        "points_count": info.points_count,
        "vectors_count": info.vectors_count,
        "status": str(info.status),
    }


if __name__ == "__main__":
    # python -m src.index.vector_store
    print("Creating collection...")
    create_collection()
    info = collection_info()
    print("Collection info:", info)
