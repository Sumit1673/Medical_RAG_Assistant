# Quickstart

## Prerequisites

- Python 3.11+ with a virtual environment
- Docker Desktop (running)
- [Ollama](https://ollama.ai) for local LLM

---

## Local Setup (Manual)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start ChromaDB

```bash
docker run -d -p 8001:8000 chromadb/chroma:latest
```

> ChromaDB must be running before the API starts.

### 3. Pull your Ollama model

```bash
ollama run gemma4:e4b
```

Change `model_name` in `config/config.yaml` to use a different model.

### 4. Ingest documents

```bash
python scripts/run_ingest.py
```

Loads all files from `dataset/medical/` into ChromaDB (`medical_rag` collection).

### 5. Start the API

```bash
uvicorn app:app --reload
```

API is available at http://localhost:8000

### 6. Open the demo UI

```bash
open demo/index.html
```

The UI fetches model info from `/pipeline/info` on load and always reflects the current `config.yaml`.

---

## Docker Compose Setup

Starts ChromaDB + the RAG API together:

```bash
docker-compose up --build
```

No API key needed when using Ollama. Services:
- RAG API → http://localhost:8000
- ChromaDB → http://localhost:8001

---

## Quick API Test

```bash
# Health check
curl http://localhost:8000/health

# Pipeline config (shows active model from config.yaml)
curl http://localhost:8000/pipeline/info

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the symptoms of type 2 diabetes?"}'

# Streaming
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain hypertension treatment options"}'
```

---

## Run Tests

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

---

## Key Configuration (`config/config.yaml`)

### Switch LLM model

```yaml
llm:
  provider: "ollama"
  model_name: "llama3"          # any model pulled in Ollama
  model_base_url: "http://localhost:11434"
```

### Switch to OpenAI

```yaml
embedding:
  provider: "openai"
  model_name: "text-embedding-3-small"

llm:
  provider: "openai"
  model_name: "gpt-4o"
```

Set `OPENAI_API_KEY` in your environment.

### Enable top_p nucleus filtering (instead of fixed top_k)

`top_p` selects the minimum set of documents covering `p` of the total
relevance score — useful when query difficulty varies.

```yaml
retrieval:
  # top_p: 0.85       # RRF output: nucleus over normalised RRF scores
  final_top_k: 5      # fallback when top_p is not set

reranker:
  # top_p: 0.90       # reranker output: nucleus over softmax of cross-encoder scores
  top_n: 5            # fallback when top_p is not set
```

> `dense_top_k` and `sparse_top_k` always use top_k (broad initial retrieval).

### Tune retrieval

```yaml
retrieval:
  dense_top_k: 10
  sparse_top_k: 10
  final_top_k: 5
  enable_query_rewriting: true
  enable_reranking: true
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'rag_assistant'`**
Run tests with the `src` path:
```bash
PYTHONPATH=src python -m pytest tests/ -v
```

**ChromaDB client/server version mismatch**
Upgrade the Python client to match the server:
```bash
pip install "chromadb>=1.0.0" "langchain-chroma>=0.2.0"
```

**Port 8000 already in use**
```bash
lsof -ti :8000 | xargs kill -9
```

**API starts but pipeline not initialised**
ChromaDB is not running. Start the Docker container first (step 2).

**Slow responses on CPU**
Disable optional stages in `config/config.yaml`:
```yaml
retrieval:
  enable_query_rewriting: false
  enable_reranking: false
```
