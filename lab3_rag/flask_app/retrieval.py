"""Retrieval module -- Dense (ChromaDB) + BM25 + Hybrid with Reciprocal Rank Fusion.

Supports three modes controlled by config.RETRIEVAL_METHOD:
  dense  -- sentence-transformer bi-encoder via ChromaDB
  bm25   -- sparse keyword matching via rank_bm25
  hybrid -- fuse dense + BM25 rankings with RRF
"""
from __future__ import annotations

import json
import os
import pickle
from typing import Dict, List, Optional

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_COLLECTION,
    CHROMA_HOST,
    CHROMA_PORT,
    EMBEDDING_MODEL,
    RETRIEVAL_METHOD,
    TOP_K,
)

_embed_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.HttpClient] = None

BM25_DIR = os.getenv("BM25_DIR", "/data/bm25")


def get_embedding_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


def get_chroma_client() -> chromadb.HttpClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return _chroma_client


def get_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def _bm25_path():
    os.makedirs(BM25_DIR, exist_ok=True)
    return os.path.join(BM25_DIR, "bm25_corpus.pkl")


def _load_bm25_corpus() -> List[Dict]:
    path = _bm25_path()
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return []


def _save_bm25_corpus(corpus: List[Dict]):
    with open(_bm25_path(), "wb") as f:
        pickle.dump(corpus, f)


def index_chunks(doc_id: str, filename: str, chunks: List[str]):
    """Store chunks in ChromaDB (dense) and local BM25 corpus (sparse)."""
    model = get_embedding_model()
    collection = get_collection()

    embeddings = model.encode(chunks).tolist()
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"doc_id": doc_id, "filename": filename, "chunk_index": i}
        for i in range(len(chunks))
    ]

    batch = 500
    for start in range(0, len(ids), batch):
        end = start + batch
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=chunks[start:end],
            metadatas=metadatas[start:end],
        )

    corpus = _load_bm25_corpus()
    for i, chunk in enumerate(chunks):
        corpus.append({
            "id": ids[i],
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": i,
            "text": chunk,
        })
    _save_bm25_corpus(corpus)


def delete_doc_chunks(doc_id: str):
    """Remove all chunks for a document from both indexes."""
    collection = get_collection()
    results = collection.get(where={"doc_id": doc_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])

    corpus = _load_bm25_corpus()
    corpus = [c for c in corpus if c["doc_id"] != doc_id]
    _save_bm25_corpus(corpus)


def _dense_retrieve(query: str, top_k: int) -> List[Dict]:
    model = get_embedding_model()
    collection = get_collection()
    q_emb = model.encode([query]).tolist()

    try:
        count = collection.count()
    except Exception:
        count = 0
    if count == 0:
        return []

    results = collection.query(
        query_embeddings=q_emb,
        n_results=min(top_k, count),
    )

    hits: list[dict] = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i] if results["distances"] else 0.0
        hits.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "doc_id": results["metadatas"][0][i]["doc_id"],
            "filename": results["metadatas"][0][i]["filename"],
            "score": round(1.0 - distance, 4),
        })
    return hits


def _bm25_retrieve(query: str, top_k: int) -> List[Dict]:
    corpus = _load_bm25_corpus()
    if not corpus:
        return []

    tokenized = [c["text"].lower().split() for c in corpus]
    bm25 = BM25Okapi(tokenized)
    q_tokens = query.lower().split()
    scores = bm25.get_scores(q_tokens)

    top_indices = np.argsort(scores)[::-1][:top_k]
    hits: list[dict] = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        c = corpus[idx]
        hits.append({
            "chunk_id": c["id"],
            "text": c["text"],
            "doc_id": c["doc_id"],
            "filename": c["filename"],
            "score": round(float(scores[idx]), 4),
        })
    return hits


def _reciprocal_rank_fusion(
    rankings: List[List[Dict]],
    k: int = 60,
    top_k: int = 5,
) -> List[Dict]:
    """Fuse multiple ranked lists with RRF."""
    fused_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            cid = hit["chunk_id"]
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            chunk_map[cid] = hit

    sorted_ids = sorted(fused_scores, key=lambda x: fused_scores[x], reverse=True)
    results: list[dict] = []
    for cid in sorted_ids[:top_k]:
        entry = chunk_map[cid].copy()
        entry["score"] = round(fused_scores[cid], 4)
        results.append(entry)
    return results


def _hybrid_retrieve(query: str, top_k: int) -> List[Dict]:
    dense_hits = _dense_retrieve(query, top_k=top_k * 2)
    bm25_hits = _bm25_retrieve(query, top_k=top_k * 2)
    return _reciprocal_rank_fusion([dense_hits, bm25_hits], top_k=top_k)


def retrieve(query: str, top_k: int = TOP_K, method: Optional[str] = None) -> List[Dict]:
    """Retrieve the most relevant chunks for *query*."""
    method = method or RETRIEVAL_METHOD
    if method == "dense":
        return _dense_retrieve(query, top_k)
    if method == "bm25":
        return _bm25_retrieve(query, top_k)
    if method == "hybrid":
        return _hybrid_retrieve(query, top_k)
    raise ValueError(f"Unknown retrieval method: {method}")
