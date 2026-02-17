"""Flask REST API -- central orchestration layer for the RAG pipeline.

Endpoints
---------
POST   /documents       Upload a PDF or DOCX file
GET    /documents       List indexed documents
DELETE /documents/{id}  Remove a document and its chunks
POST   /query           Ask a question and get a grounded answer
GET    /health          Verify all services are reachable
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from io import BytesIO

import requests as http_requests
from flask import Flask, jsonify, request
from minio import Minio

from chunking import chunk_text
from config import (
    LLM_URL,
    MAX_UPLOAD_MB,
    MINIO_ACCESS_KEY,
    MINIO_BUCKET,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    TOP_K,
)
from agent import agent_query
from generation import generate_answer
from ingestion import parse_document
from retrieval import (
    delete_doc_chunks,
    get_chroma_client,
    index_chunks,
    retrieve,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

REGISTRY_PATH = os.getenv("REGISTRY_PATH", "/data/docs_registry.json")


def _load_registry() -> dict:
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _save_registry(registry: dict):
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2)


def _ensure_bucket():
    if not minio_client.bucket_exists(MINIO_BUCKET):
        minio_client.make_bucket(MINIO_BUCKET)


@app.route("/health", methods=["GET"])
def health():
    checks: dict[str, str] = {"flask": "ok"}

    try:
        minio_client.list_buckets()
        checks["minio"] = "ok"
    except Exception as exc:
        checks["minio"] = f"error: {exc}"

    try:
        get_chroma_client().heartbeat()
        checks["chromadb"] = "ok"
    except Exception as exc:
        checks["chromadb"] = f"error: {exc}"

    try:
        r = http_requests.get(f"{LLM_URL}/health", timeout=5)
        checks["llm"] = "ok" if r.status_code == 200 else f"status {r.status_code}"
    except Exception as exc:
        checks["llm"] = f"error: {exc}"

    all_ok = all(v == "ok" for v in checks.values())
    status_code = 200 if all_ok else 503
    return jsonify({"status": "healthy" if all_ok else "degraded", "services": checks}), status_code


@app.route("/documents", methods=["POST"])
def upload_document():
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Use multipart/form-data with key 'file'."}), 400

    file = request.files["file"]
    filename = file.filename or ""
    if not filename:
        return jsonify({"error": "Empty filename."}), 400

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ("pdf", "docx"):
        return jsonify({"error": f"Unsupported file type: .{ext}. Only PDF and DOCX accepted."}), 400

    file_bytes = file.read()
    if not file_bytes:
        return jsonify({"error": "Uploaded file is empty."}), 400

    try:
        text = parse_document(file_bytes, filename)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422

    doc_id = str(uuid.uuid4())

    _ensure_bucket()
    minio_client.put_object(
        MINIO_BUCKET,
        f"{doc_id}/{filename}",
        BytesIO(file_bytes),
        len(file_bytes),
        content_type="application/octet-stream",
    )

    chunks = chunk_text(text, strategy="recursive")
    if not chunks:
        chunks = chunk_text(text, strategy="fixed")

    index_chunks(doc_id, filename, chunks)

    registry = _load_registry()
    registry[doc_id] = {
        "filename": filename,
        "upload_date": datetime.now(timezone.utc).isoformat(),
        "num_chunks": len(chunks),
        "text_length": len(text),
    }
    _save_registry(registry)

    return jsonify({
        "document_id": doc_id,
        "filename": filename,
        "num_chunks": len(chunks),
        "message": "Document uploaded, parsed, and indexed successfully.",
    }), 201


@app.route("/documents", methods=["GET"])
def list_documents():
    registry = _load_registry()
    docs = [
        {
            "id": doc_id,
            "filename": info["filename"],
            "upload_date": info["upload_date"],
            "num_chunks": info["num_chunks"],
        }
        for doc_id, info in registry.items()
    ]
    return jsonify({"documents": docs}), 200


@app.route("/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    registry = _load_registry()
    if doc_id not in registry:
        return jsonify({"error": "Document not found."}), 404

    filename = registry[doc_id]["filename"]

    try:
        minio_client.remove_object(MINIO_BUCKET, f"{doc_id}/{filename}")
    except Exception:
        pass

    delete_doc_chunks(doc_id)

    del registry[doc_id]
    _save_registry(registry)

    return jsonify({"message": f"Document {doc_id} deleted."}), 200


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json(silent=True)
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field in JSON body."}), 400

    question = data["question"]
    top_k = data.get("top_k", TOP_K)
    method = data.get("method", None)
    mode = data.get("mode", "default")

    if mode == "agent":
        result = agent_query(question, top_k=int(top_k))
        sources = [
            {
                "doc_id": c["doc_id"],
                "filename": c["filename"],
                "score": c.get("score"),
                "text": c["text"][:500],
            }
            for c in result["sources"]
        ]
        return jsonify({
            "answer": result["answer"],
            "sources": sources,
            "agent_iterations": result["iterations"],
        }), 200

    chunks = retrieve(question, top_k=int(top_k), method=method)

    if not chunks:
        return jsonify({
            "answer": "No relevant documents found. Please upload documents first.",
            "sources": [],
        }), 200

    answer = generate_answer(question, chunks)

    sources = [
        {
            "doc_id": c["doc_id"],
            "filename": c["filename"],
            "score": c.get("score"),
            "text": c["text"][:500],
        }
        for c in chunks
    ]

    return jsonify({"answer": answer, "sources": sources}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
