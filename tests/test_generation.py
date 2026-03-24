"""Unit tests for the LLM generation module."""

from unittest.mock import MagicMock, patch

import requests as req
from generation import _build_prompt, generate_answer

SAMPLE_CHUNKS = [
    {"text": "The sky is blue.", "filename": "doc1.pdf", "score": 0.95},
    {"text": "Water is wet.", "filename": "doc2.pdf", "score": 0.80},
]


# ── _build_prompt ─────────────────────────────────────────────────────────────


class TestBuildPrompt:
    def test_contains_query(self):
        prompt = _build_prompt("What color is the sky?", SAMPLE_CHUNKS)
        assert "What color is the sky?" in prompt

    def test_contains_chunk_text(self):
        prompt = _build_prompt("question", SAMPLE_CHUNKS)
        assert "The sky is blue." in prompt
        assert "Water is wet." in prompt

    def test_contains_filenames(self):
        prompt = _build_prompt("question", SAMPLE_CHUNKS)
        assert "doc1.pdf" in prompt
        assert "doc2.pdf" in prompt

    def test_numbered_references(self):
        prompt = _build_prompt("question", SAMPLE_CHUNKS)
        assert "[1]" in prompt
        assert "[2]" in prompt

    def test_empty_chunks_still_builds(self):
        prompt = _build_prompt("question", [])
        assert "question" in prompt


# ── generate_answer ───────────────────────────────────────────────────────────


class TestGenerateAnswer:
    def test_returns_no_context_message_when_empty(self):
        result = generate_answer("some question", [])
        assert "No relevant context" in result

    def test_returns_llm_response_on_success(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"content": "The sky is blue."}
        mock_resp.raise_for_status = MagicMock()

        with patch("generation.requests.post", return_value=mock_resp):
            result = generate_answer("What color is the sky?", SAMPLE_CHUNKS)
        assert result == "The sky is blue."

    def test_handles_connection_error(self):
        with patch("generation.requests.post", side_effect=req.exceptions.ConnectionError):
            result = generate_answer("question", SAMPLE_CHUNKS)
        assert "unreachable" in result.lower()

    def test_handles_timeout(self):
        with patch("generation.requests.post", side_effect=req.exceptions.Timeout):
            result = generate_answer("question", SAMPLE_CHUNKS)
        assert "timed out" in result.lower()

    def test_handles_http_error(self):
        with patch("generation.requests.post", side_effect=req.exceptions.RequestException("500")):
            result = generate_answer("question", SAMPLE_CHUNKS)
        assert "LLM error" in result
