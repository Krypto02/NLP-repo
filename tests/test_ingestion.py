"""Unit tests for document ingestion (PDF / DOCX parsing)."""

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from docx import Document as DocxDocument
from ingestion import parse_document, parse_docx, parse_pdf

# ── parse_pdf ─────────────────────────────────────────────────────────────────


class TestParsePdf:
    def _make_pdf_bytes(self) -> bytes:
        """Actual bytes don't matter -- we mock pdfplumber."""
        return b"FAKE_PDF"

    def test_extracts_text_from_pages(self):
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page one content."
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page two content."

        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [mock_page1, mock_page2]

        with patch("ingestion.pdfplumber.open", return_value=mock_pdf):
            result = parse_pdf(b"fake_bytes")

        assert "Page one content." in result
        assert "Page two content." in result

    def test_raises_on_image_only_pdf(self):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = None

        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [mock_page]

        with patch("ingestion.pdfplumber.open", return_value=mock_pdf):
            with pytest.raises(ValueError, match="scanned"):
                parse_pdf(b"fake_bytes")

    def test_empty_pages_list_raises(self):
        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = []

        with patch("ingestion.pdfplumber.open", return_value=mock_pdf):
            with pytest.raises(ValueError):
                parse_pdf(b"fake_bytes")


# ── parse_docx ────────────────────────────────────────────────────────────────


class TestParseDocx:
    def _make_docx_bytes(self, texts: list[str]) -> bytes:
        """Create a real minimal DOCX in memory."""
        doc = DocxDocument()
        for t in texts:
            doc.add_paragraph(t)
        buf = BytesIO()
        doc.save(buf)
        return buf.getvalue()

    def test_extracts_paragraphs(self):
        docx_bytes = self._make_docx_bytes(["Hello world.", "Second paragraph."])
        result = parse_docx(docx_bytes)
        assert "Hello world." in result
        assert "Second paragraph." in result

    def test_empty_docx_raises(self):
        docx_bytes = self._make_docx_bytes([])
        with pytest.raises(ValueError, match="no extractable text"):
            parse_docx(docx_bytes)

    def test_whitespace_only_raises(self):
        docx_bytes = self._make_docx_bytes(["   ", "\t"])
        with pytest.raises(ValueError, match="no extractable text"):
            parse_docx(docx_bytes)


# ── parse_document dispatcher ─────────────────────────────────────────────────


class TestParseDocument:
    def test_dispatches_to_pdf(self):
        with patch("ingestion.parse_pdf", return_value="pdf text") as mock_pdf:
            result = parse_document(b"bytes", "doc.pdf")
        mock_pdf.assert_called_once_with(b"bytes")
        assert result == "pdf text"

    def test_dispatches_to_docx(self):
        with patch("ingestion.parse_docx", return_value="docx text") as mock_docx:
            result = parse_document(b"bytes", "doc.docx")
        mock_docx.assert_called_once_with(b"bytes")
        assert result == "docx text"

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            parse_document(b"bytes", "file.txt")

    def test_no_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            parse_document(b"bytes", "nodotfile")
