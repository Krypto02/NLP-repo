"""Document parsing -- extract text from PDF and DOCX files."""
import io
import pdfplumber
from docx import Document


def parse_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF. Raises ValueError for scanned/image-only PDFs."""
    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    text = "\n".join(text_parts).strip()
    if not text:
        raise ValueError(
            "PDF appears to be scanned / image-only -- no extractable text found."
        )
    return text


def parse_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX file."""
    doc = Document(io.BytesIO(file_bytes))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if not text.strip():
        raise ValueError("DOCX contains no extractable text.")
    return text.strip()


def parse_document(file_bytes: bytes, filename: str) -> str:
    """Dispatch to the correct parser based on file extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "pdf":
        return parse_pdf(file_bytes)
    if ext in ("docx",):
        return parse_docx(file_bytes)
    raise ValueError(f"Unsupported file type: .{ext}")
