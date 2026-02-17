# RAG Document Q&A System -- Assignment 3

## Architecture Overview

```
+----------+   PDF/DOCX    +-----------+    store     +---------+
|  Client   | ---------->  |  Flask API | ---------->  |  MinIO  |
| (curl /   |              |  :5000     |              |  :9000  |
| Streamlit)|  question    |            |  chunks      |         |
|  :8501    | ---------->  |  parse --> | ---------->  |---------|
+----------+              |  chunk     |              |ChromaDB |
                           |  retrieve  | <---------- |  :8000  |
                           |  generate  |              +---------+
                           |            |  prompt+ctx   +---------+
                           |            | ---------->  | llama   |
                           |            | <---------- |  :8080  |
                           +-----------+  answer       +---------+
```

| Service       | Image / Build           | Port  | Purpose                          |
|---------------|-------------------------|-------|----------------------------------|
| MinIO         | `minio/minio:latest`    | 9000  | Raw document storage             |
| ChromaDB      | `chromadb/chroma:0.6.3` | 8000  | Vector index for chunk retrieval |
| llama.cpp     | `ghcr.io/ggml-org/llama.cpp:server-cuda` | 8080 | Local quantized LLM (GPU)  |
| Flask API     | `./flask_app`           | 5000  | REST API orchestration           |
| Streamlit     | `./frontend`            | 8501  | Web UI (bonus)                   |

## Quick Start

### 1. Clone the repository

```bash
git clone <repo-url>
cd lab3_rag
```

### 2. Download the LLM model weights

> **Do NOT commit model weights to Git.** GGUF files are 4-8 GB.

```bash
mkdir -p models

# Option A -- Mistral 7B Instruct (recommended, ~4.4 GB)
wget -O models/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
  "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Option B -- TinyLlama 1.1B Chat (faster, ~700 MB) -- change docker-compose.yml accordingly
# wget -O models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
#   "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
```

### 3. Launch the system

```bash
docker-compose up --build
```

Wait until all health checks pass (the LLM container may take 1-2 minutes to load the model on GPU).

> **GPU support**: The default configuration uses `server-cuda` with `-ngl 35` to offload all model layers to GPU. Requires an NVIDIA GPU with >= 6 GB VRAM and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). To run on CPU only, change the image to `server` and set `-ngl 0` in `docker-compose.yml`.

### 4. Verify

```bash
curl http://localhost:5000/health
```

## API Examples

### Upload a document

```bash
curl -X POST http://localhost:5000/documents \
  -F "file=@my_report.pdf"
```

### List documents

```bash
curl http://localhost:5000/documents
```

### Delete a document

```bash
curl -X DELETE http://localhost:5000/documents/<document-id>
```

### Ask a question

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?", "top_k": 5}'
```

### Ask a question (ReAct agent mode)

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarise the key conclusions.", "mode": "agent"}'
```

## Chunking Strategies

| Strategy      | Description                                             |
|---------------|---------------------------------------------------------|
| `fixed`       | Sliding window of N words with overlap                  |
| `recursive`   | Split by paragraph -> sentence -> fixed window            |
| `semantic`    | Split where cosine similarity between sentences drops   |

Default: **recursive** (best balance of speed and boundary preservation).

## Retrieval Methods

| Method   | Description                                              |
|----------|----------------------------------------------------------|
| `dense`  | Bi-Encoder (`all-MiniLM-L6-v2`) + ChromaDB cosine search |
| `bm25`   | Sparse keyword matching via `rank_bm25`                  |
| `hybrid` | Reciprocal Rank Fusion of dense + BM25 rankings          |

Default: **hybrid**.

## Evaluation

See `eval_dataset.json` for the annotated question set (35 questions across 7 documents).
Run the evaluation notebook (`notebooks/Lab3.ipynb`) to compute Hit Rate, MRR, and Precision @k.
Results are saved to `results/`.

### Results (k=1 / k=3 / k=5)

| Method | Hit Rate @k | MRR | Precision @k |
|--------|-------------|-----|--------------|
| Dense  | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 0.98 / 0.97 |
| BM25   | 0.93 / 1.00 / 1.00 | 0.93 / 0.97 / 0.97 | 0.93 / 0.64 / 0.57 |
| Hybrid | 1.00 / 1.00 / 1.00 | 1.00 / 0.97 / 0.97 | 1.00 / 0.91 / 0.87 |

## Known Limitations

- **Scanned PDFs**: Image-only PDFs are detected and rejected (no OCR support).
- **LLM latency**: With GPU offloading (RTX 5060, `-ngl 35`), inference takes ~3-5 s per query. On CPU-only it is ~10-30 s.
- **Context window**: llama.cpp is configured with a 2048-token context window; very long retrieved contexts may be truncated.
- **BM25 index**: Stored as a pickle file; not suitable for production at scale.
- **No authentication**: The API is unauthenticated -- intended for local development only.

## Project Structure

```
lab3_rag/
|-- docker-compose.yml          # One-command startup
|-- README.md                   # This file
|-- eval_dataset.json           # Annotated evaluation questions
|-- flask_app/
|   |-- Dockerfile
|   |-- requirements.txt
|   |-- config.py               # Centralised env-var configuration
|   |-- ingestion.py            # PDF & DOCX parsing
|   |-- chunking.py             # Fixed-size, recursive, semantic chunking
|   |-- retrieval.py            # Dense + BM25 + hybrid retrieval
|   |-- generation.py           # LLM prompt assembly & generation
|   |-- agent.py                # ReAct agentic pipeline (bonus)
|   +-- app.py                  # Flask REST API
|-- frontend/
|   |-- Dockerfile
|   |-- requirements.txt
|   +-- streamlit_app.py        # Streamlit web UI (bonus)
|-- models/                     # .gitignore -- do NOT commit GGUF files
+-- results/                    # Evaluation output
```
