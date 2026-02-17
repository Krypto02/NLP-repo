"""ReAct-style agentic pipeline (Bonus +5%).

The LLM iteratively decides whether to SEARCH, SUMMARIZE, or produce a FINAL ANSWER.
Each loop iteration calls a tool and feeds the observation back to the LLM.
"""
from __future__ import annotations

import re
from typing import Dict, List

import requests

from config import LLM_URL, TOP_K
from retrieval import retrieve

MAX_ITERATIONS = 5

REACT_SYSTEM = """You are a document Q&A agent. You have access to two tools:

SEARCH[query] -- search the document index for relevant passages
ANSWER[text]  -- provide the final answer to the user (this ends the loop)

Think step-by-step. At each step output EXACTLY one action line.
Format examples:
  Thought: I need to find information about X.
  Action: SEARCH[information about X]
  Thought: I now have enough context.
  Action: ANSWER[The answer is ...]

Do NOT include any text after the Action line."""


_ACTION_RE = re.compile(r"Action:\s*(SEARCH|ANSWER)\[(.+?)\]", re.DOTALL)


def _call_llm(prompt: str) -> str:
    try:
        resp = requests.post(
            f"{LLM_URL}/completion",
            json={
                "prompt": prompt,
                "n_predict": 512,
                "temperature": 0.2,
                "stop": ["\nObservation:", "[INST]", "</s>"],
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("content", "").strip()
    except requests.exceptions.RequestException as exc:
        return f"ANSWER[LLM error: {exc}]"


def agent_query(question: str, top_k: int = TOP_K) -> Dict:
    """Run a ReAct loop that alternates between thinking and tool use."""
    all_observations: List[str] = []
    all_chunks: List[Dict] = []
    seen_chunk_ids: set = set()

    prompt = f"[INST] {REACT_SYSTEM}\n\nQuestion: {question}\n[/INST]"

    for iteration in range(MAX_ITERATIONS):
        llm_output = _call_llm(prompt)

        match = _ACTION_RE.search(llm_output)
        if not match:
            return {
                "answer": llm_output,
                "sources": all_chunks,
                "iterations": iteration + 1,
                "mode": "agent",
            }

        action, argument = match.group(1), match.group(2)

        if action == "ANSWER":
            return {
                "answer": argument,
                "sources": all_chunks,
                "iterations": iteration + 1,
                "mode": "agent",
            }

        if action == "SEARCH":
            hits = retrieve(argument, top_k=top_k)
            for h in hits:
                if h["chunk_id"] not in seen_chunk_ids:
                    seen_chunk_ids.add(h["chunk_id"])
                    all_chunks.append(h)

            if hits:
                obs_text = "\n".join(
                    f"[{i}] ({h['filename']}) {h['text'][:300]}"
                    for i, h in enumerate(hits, 1)
                )
            else:
                obs_text = "No results found."

            all_observations.append(obs_text)
            prompt += f"\n{llm_output}\nObservation: {obs_text}\n"

    return {
        "answer": "(Agent reached max iterations) " + (all_observations[-1] if all_observations else "No answer generated."),
        "sources": all_chunks,
        "iterations": MAX_ITERATIONS,
        "mode": "agent",
    }
