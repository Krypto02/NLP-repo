"""Unit tests for chunking strategies."""

import pytest
from chunking import chunk_text, fixed_size_chunk, recursive_split

# ── fixed_size_chunk ──────────────────────────────────────────────────────────


class TestFixedSizeChunk:
    def test_empty_string_returns_empty_list(self):
        assert not fixed_size_chunk("")

    def test_single_chunk_when_text_fits(self):
        text = "hello world foo bar"
        result = fixed_size_chunk(text, chunk_size=10, overlap=0)
        assert result == [text]

    def test_splits_into_multiple_chunks(self):
        words = " ".join([f"word{i}" for i in range(20)])
        result = fixed_size_chunk(words, chunk_size=5, overlap=0)
        assert len(result) == 4

    def test_overlap_produces_extra_chunks(self):
        words = " ".join([f"w{i}" for i in range(10)])
        no_overlap = fixed_size_chunk(words, chunk_size=5, overlap=0)
        with_overlap = fixed_size_chunk(words, chunk_size=5, overlap=2)
        assert len(with_overlap) > len(no_overlap)

    def test_chunk_content_is_subset_of_original(self):
        text = "the quick brown fox jumps over the lazy dog"
        for chunk in fixed_size_chunk(text, chunk_size=3, overlap=0):
            for word in chunk.split():
                assert word in text

    def test_whitespace_only_returns_empty(self):
        assert not fixed_size_chunk("   \n\t  ")


# ── recursive_split ───────────────────────────────────────────────────────────


class TestRecursiveSplit:
    def test_empty_returns_empty(self):
        assert not recursive_split("")

    def test_short_paragraph_is_one_chunk(self):
        text = "This is a short sentence."
        result = recursive_split(text, max_chunk_size=50)
        assert result == [text]

    def test_multiple_paragraphs_split_correctly(self):
        text = "Para one is here.\n\nPara two is here.\n\nPara three."
        result = recursive_split(text, max_chunk_size=20)
        assert len(result) == 3

    def test_long_paragraph_splits_by_sentence(self):
        long_para = "First sentence here. " * 30
        result = recursive_split(long_para.strip(), max_chunk_size=10)
        assert len(result) > 1

    def test_no_chunk_exceeds_max_size_by_large_margin(self):
        # Build paragraphs with sentence boundaries so recursive_split can split them
        sentence = "This is a sentence that has some words in it. "
        text = "\n\n".join([sentence * 10] * 3)
        result = recursive_split(text, max_chunk_size=10)
        for chunk in result:
            assert len(chunk.split()) <= 30  # each sentence is ~10 words


# ── chunk_text dispatcher ─────────────────────────────────────────────────────


class TestChunkText:
    def test_fixed_strategy(self):
        result = chunk_text("a b c d e f g h i j", strategy="fixed", chunk_size=5, overlap=0)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_recursive_strategy(self):
        result = chunk_text("Hello world.\n\nSecond paragraph.", strategy="recursive")
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            chunk_text("some text", strategy="nonexistent")
