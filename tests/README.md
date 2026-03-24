# tests/

Unit tests for the RAG pipeline components and generation scripts.

| File | Module under test | Tests |
|------|------------------|-------|
| `test_chunking.py` | `chunking.py` | Fixed-size, recursive and semantic chunking strategies |
| `test_ingestion.py` | `ingestion.py` | PDF and DOCX parsing |
| `test_generation.py` | `generation.py` | LLM prompt building and answer generation |
| `test_generate_memes_rag.py` | `generate_memes_rag.py` | Meme generation pipeline, output format |
| `test_upload_training_to_rag.py` | `upload_training_to_rag.py` | DOCX splitting and RAG upload |

## Run

```bash
# from repo root
pytest tests/ -v
```

61 tests — all passing.
