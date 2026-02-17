"""Chunking strategies -- split extracted text into retrievable units.

Implements three strategies:
1. Fixed-Size   -- sliding window of N words with overlap
2. Recursive    -- split by paragraph -> sentence -> fixed size
3. Semantic     -- split where embedding similarity drops below a threshold
"""
from __future__ import annotations

import re
from typing import List

from config import CHUNK_SIZE, CHUNK_OVERLAP


def fixed_size_chunk(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Sliding-window chunking by word count."""
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start += max(chunk_size - overlap, 1)
    return chunks


_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def recursive_split(
    text: str,
    max_chunk_size: int = CHUNK_SIZE,
) -> List[str]:
    """Split by paragraph -> sentence -> fixed window (preserves author boundaries)."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks: list[str] = []
    for para in paragraphs:
        words = para.split()
        if len(words) <= max_chunk_size:
            chunks.append(para)
        else:
            sentences = _SENT_RE.split(para)
            current: list[str] = []
            current_len = 0
            for sent in sentences:
                sent_len = len(sent.split())
                if current_len + sent_len > max_chunk_size and current:
                    chunks.append(" ".join(current))
                    current = []
                    current_len = 0
                current.append(sent)
                current_len += sent_len
            if current:
                chunks.append(" ".join(current))
    return chunks


def semantic_chunk(
    text: str,
    threshold: float = 0.5,
    embedding_model=None,
) -> List[str]:
    """Split where consecutive-sentence similarity drops below *threshold*.

    Requires a SentenceTransformer model instance (passed to avoid repeated loading).
    """
    import numpy as np

    sentences = _SENT_RE.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= 1:
        return sentences

    embeddings = embedding_model.encode(sentences)

    chunks: list[str] = []
    current_chunk = [sentences[0]]
    for i in range(1, len(sentences)):
        a, b = embeddings[i - 1], embeddings[i]
        cos_sim = float(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        )
        if cos_sim < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def chunk_text(
    text: str,
    strategy: str = "recursive",
    **kwargs,
) -> List[str]:
    """Public entry point - choose strategy by name."""
    if strategy == "fixed":
        return fixed_size_chunk(text, **kwargs)
    if strategy == "recursive":
        return recursive_split(text, **kwargs)
    if strategy == "semantic":
        return semantic_chunk(text, **kwargs)
    raise ValueError(f"Unknown chunking strategy: {strategy}")
