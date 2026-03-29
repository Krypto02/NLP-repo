# Misogynous Meme Generator & Classifier

A two-stage NLP project that (1) **generates** misogynous memes using a Retrieval-Augmented Generation (RAG) pipeline and (2) **classifies** them to evaluate whether the generated content is correctly identified as misogynous.

## Project Overview

### Stage 1 � RAG-Based Meme Generator

A RAG pipeline powered by a local quantized LLM (Mistral 7B Instruct) that retrieves relevant context from a curated meme dataset and generates new misogynous meme captions across multiple categories: shaming, stereotype, objectification, and violence.

### Stage 2 � Misogyny Classifier

A classifier trained on the generated and original meme dataset that identifies whether a given meme caption is misogynous and, if so, which sub-category it belongs to.

## Project Structure

```
.
├── src/
│   ├── app.py                 # Streamlit classifier demo (main entry point)
│   ├── model.py               # MultiTaskModel (binary + multi-label heads)
│   ├── MultiClassifier.ipynb   # Training notebook
│   ├── backend/
│   │   ├── flask_app/         # Flask REST API (RAG orchestration)
│   │   └── scripts/           # Data upload & meme generation scripts
│   ├── frontend/
│   │   └── streamlit/         # Docker-based Streamlit UI (RAG Q&A)
│   ├── data/
│   │   ├── training/          # training.csv (original meme dataset)
│   │   ├── examples/          # examples.txt / examples.docx (RAG source docs)
│   │   └── test_docs/         # Test PDFs and DOCX files
│   ├── models/
│   │   ├── gguf/              # Quantized GGUF model weights (not committed)
│   │   └── trained/           # multitask_model_weights.pt, ml_thresholds.json
│   └── evaluation/
│       ├── results/           # generated_memes_rag.csv, retrieval metrics
│       └── datasets/          # eval_dataset.json
├── notebooks/                 # Jupyter notebooks (Labs 1-3)
├── tests/                     # Unit tests (pytest)
├── requirements.txt
├── docker-compose.yml
└── pyproject.toml
```

## Dataset

`data/training/training.csv` � TSV file with 7,500 meme samples:

| Column | Description |
|--------|-------------|
| `file_name` | Meme image identifier |
| `misogynous` | 1 if the meme is misogynous |
| `shaming` | Body/slut shaming |
| `stereotype` | Gender stereotypes |
| `objectification` | Sexual objectification |
| `violence` | Threats or physical violence |
| `Text Transcription` | OCR-extracted meme text |

## Quick Start

### 1. Download the LLM model weights

> **Do NOT commit model weights to Git.** GGUF files are 4�8 GB.

```bash
mkdir -p src/models/gguf
wget -O src/models/gguf/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
  "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
```

### 2. Launch the RAG stack

```powershell
# Start Docker Desktop first, then:
docker compose up -d minio chromadb llama flask_app
```

Services started: MinIO (9000), ChromaDB (8000), llama.cpp (8080), Flask API (5000).

### 3. Upload the meme corpus to the vector store

```powershell
cd src
python -m backend.scripts.upload_training_to_rag
```

### 4. Launch the classifier demo

```powershell
cd ..   # back to project root
python -m streamlit run src/app.py --server.port 8501
```

Open http://localhost:8501 — four tabs: Examples, Random from Dataset, AI Generator (live Mistral 7B), Write your own.

### 5. Generate memes (batch, optional)

```powershell
cd src
python -m backend.scripts.generate_memes_rag
```

Output: `src/evaluation/results/generated_memes_rag.csv` (same TSV format as `training.csv`).

### 6. Run tests

```bash
pytest tests/ -v
```

## Architecture

```
+----------+   PDF/DOCX    +-----------+    store     +---------+
| Scripts  | ---------->  |  Flask API | ---------->  |  MinIO  |
|          |              |   :5000    |              |  :9000  |
|          |  question    |            |  embeddings  |         |
| Streamlit| ---------->  |  parse --> | ---------->  +---------+
|  :8501   |              |  chunk     |              |ChromaDB |
+----------+              |  retrieve  | <----------  |  :8000  |
                           |  generate  |              +---------+
                           |            |  prompt+ctx  +---------+
                           |            | ---------->  | llama   |
                           |            | <----------  |  :8080  |
                           +-----------+              +---------+
```

| Service   | Port | Purpose |
|-----------|------|---------|
| MinIO     | 9000 | Raw document storage |
| ChromaDB  | 8000 | Vector index for chunk retrieval |
| llama.cpp | 8080 | Local quantized LLM (Mistral 7B, GPU) |
| Flask API | 5000 | REST API orchestration |
| Streamlit | 8501 | Web UI |

## API Reference

```bash
# Upload a document to the RAG corpus
curl -X POST http://localhost:5000/documents -F "file=@memes.pdf"

# Ask a question (retrieval + generation)
curl -X POST http://localhost:5000/query -H "Content-Type: application/json" \
  -d '{"question": "Generate a meme about gender stereotypes"}'

# Health check
curl http://localhost:5000/health
```
## Reproducibility

All random seeds are fixed for full reproducibility:
```python
RANDOM_STATE = 42
random.seed(42)
numpy.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)  # when CUDA available
```
All `train_test_split` calls use `random_state=42` with stratification.
## Code Quality

- **black** � code formatting (line-length 100)
- **pylint** � static analysis (**10.00/10**)
- **pytest** � 61 unit tests, all passing
