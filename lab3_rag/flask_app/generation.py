"""LLM answer generation via llama.cpp /completion endpoint."""
from __future__ import annotations

from typing import Dict, List

import requests

from config import LLM_URL

SYSTEM_PROMPT = (
    "You are a helpful document assistant. "
    "Answer the user's question based ONLY on the provided context passages. "
    "Cite the source passage numbers in your answer using [1], [2], etc. "
    "If the context does not contain enough information to answer, say so clearly."
)


def _build_prompt(query: str, chunks: List[Dict]) -> str:
    """Build a Mistral-Instruct formatted prompt with retrieved context."""
    context_parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        score = chunk.get("score", 0)
        fname = chunk.get("filename", "unknown")
        context_parts.append(f"[{i}] (Source: {fname}, score: {score:.3f})\n{chunk['text']}")
    context = "\n\n".join(context_parts)

    prompt = (
        f"[INST] {SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query} [/INST]"
    )
    return prompt


def generate_answer(query: str, chunks: List[Dict]) -> str:
    """Call the local LLM and return the generated answer text."""
    if not chunks:
        return "No relevant context was retrieved. Please upload documents first."

    prompt = _build_prompt(query, chunks)

    try:
        resp = requests.post(
            f"{LLM_URL}/completion",
            json={
                "prompt": prompt,
                "n_predict": 512,
                "temperature": 0.3,
                "top_p": 0.9,
                "stop": ["[INST]", "</s>"],
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("content", "").strip()
    except requests.exceptions.ConnectionError:
        return "[LLM service unreachable -- is the llama container running?]"
    except requests.exceptions.Timeout:
        return "[LLM service timed out -- the model may be loading.]"
    except requests.exceptions.RequestException as exc:
        return f"[LLM error: {exc}]"
