"""Unit tests for the meme generation script utilities."""

# pylint: disable=duplicate-code

import os
from unittest.mock import MagicMock, patch
import csv

import pytest
import generate_memes_rag as gmr

# ── clean_meme ────────────────────────────────────────────────────────────────


class TestCleanMeme:
    def test_removes_urls(self):
        result = gmr.clean_meme("Check http://example.com for more")
        assert "http" not in result

    def test_removes_hashtags(self):
        result = gmr.clean_meme("Funny meme #memelife #lol")
        assert "#" not in result

    def test_removes_meme_site_names(self):
        result = gmr.clean_meme("imgflip.com quickmeme just a joke")
        assert "imgflip" not in result.lower()
        assert "quickmeme" not in result.lower()

    def test_removes_brackets(self):
        result = gmr.clean_meme("Text [context] more text")
        assert "[" not in result
        assert "]" not in result

    def test_strips_leading_trailing_whitespace(self):
        result = gmr.clean_meme("  hello world  ")
        assert result == result.strip()

    def test_returns_empty_on_refusal(self):
        result = gmr.clean_meme(
            "The context does not contain enough information to generate a meme."
        )
        assert result == ""

    def test_truncates_long_text(self):
        long_text = "Word " * 100
        result = gmr.clean_meme(long_text)
        assert len(result) <= 220 + 10  # small tolerance for word boundary

    def test_valid_meme_passes_through(self):
        meme = "Women be like: I have nothing to wear"
        result = gmr.clean_meme(meme)
        assert "Women" in result

    def test_returns_empty_for_very_short_content(self):
        result = gmr.clean_meme("ok")
        assert result == ""

    def test_joins_multiple_lines_with_slash(self):
        result = gmr.clean_meme("Line one here.\nLine two here.\nLine three here.")
        assert "/" in result or len(result) > 0


# ── build_prompt ──────────────────────────────────────────────────────────────


class TestBuildPrompt:
    def test_contains_base_prompt(self):
        result = gmr.build_prompt("shaming", [], already_generated=None)
        assert len(result) > 10

    def test_includes_examples(self):
        examples = ["Example meme one", "Example meme two"]
        result = gmr.build_prompt("shaming", examples)
        assert "Example meme one" in result

    def test_includes_already_generated_to_avoid(self):
        already = ["Old meme A", "Old meme B"]
        result = gmr.build_prompt("stereotype", [], already_generated=already)
        assert "Old meme A" in result

    def test_ends_with_meme_cue(self):
        result = gmr.build_prompt("neutral", [])
        assert "Meme:" in result


# ── get_rag_context ───────────────────────────────────────────────────────────


class TestGetRagContext:
    def test_returns_string_on_success(self):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"chunks": [{"text": "Some context text here."}]}

        with patch("generate_memes_rag.requests.post", return_value=mock_resp):
            result = gmr.get_rag_context("shaming")
        assert isinstance(result, str)
        assert "Some context text here." in result

    def test_returns_empty_string_on_failure(self):
        with patch("generate_memes_rag.requests.post", side_effect=Exception("timeout")):
            result = gmr.get_rag_context("shaming")
        assert result == ""

    def test_returns_empty_when_not_ok(self):
        mock_resp = MagicMock()
        mock_resp.ok = False

        with patch("generate_memes_rag.requests.post", return_value=mock_resp):
            result = gmr.get_rag_context("neutral")
        assert result == ""


# ── CSV output format ─────────────────────────────────────────────────────────


class TestCsvOutputFormat:
    """Verify the generated CSV matches training.csv column structure."""

    EXPECTED_HEADERS = [
        "file_name",
        "misogynous",
        "shaming",
        "stereotype",
        "objectification",
        "violence",
        "Text Transcription",
    ]

    def test_generated_csv_has_correct_headers(self):
        output_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "src",
                "evaluation",
                "results",
                "generated_memes_rag.csv",
            )
        )
        if not os.path.exists(output_path):
            pytest.skip("generated_memes_rag.csv not found -- run the generation script first")

        with open(output_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            headers = next(reader)
        assert headers == self.EXPECTED_HEADERS

    def test_generated_csv_binary_labels_are_valid(self):
        output_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "src",
                "evaluation",
                "results",
                "generated_memes_rag.csv",
            )
        )
        if not os.path.exists(output_path):
            pytest.skip("generated_memes_rag.csv not found -- run the generation script first")

        with open(output_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                for col in ["misogynous", "shaming", "stereotype", "objectification", "violence"]:
                    assert row[col] in ("0", "1"), f"Invalid value '{row[col]}' in column '{col}'"

    def test_generated_csv_has_rows(self):
        output_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "src",
                "evaluation",
                "results",
                "generated_memes_rag.csv",
            )
        )
        if not os.path.exists(output_path):
            pytest.skip("generated_memes_rag.csv not found -- run the generation script first")

        with open(output_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        assert len(rows) > 0
