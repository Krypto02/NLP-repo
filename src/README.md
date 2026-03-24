# src/

Main source directory for the project.

- `backend/flask_app/` — Flask REST API: document ingestion, chunking, vector retrieval, LLM generation
- `backend/scripts/` — Standalone scripts: upload corpus to RAG (`upload_training_to_rag.py`), generate memes (`generate_memes_rag.py`)
- `frontend/streamlit/` — Streamlit web UI for interacting with the RAG pipeline
- `data/training/` — Original meme dataset (`training.csv`, TSV format)
- `data/examples/` — Source documents loaded into the RAG vector store
- `models/gguf/` — Quantized GGUF model weights (not committed to Git)
- `evaluation/results/` — Generated memes CSV and retrieval evaluation metrics
- `evaluation/datasets/` — Evaluation dataset (`eval_dataset.json`)
