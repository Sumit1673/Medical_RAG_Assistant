# 🧠 Medical RAG Assistant

[![CI](https://github.com/Sumit1673/medical-rag-assistant/.github/workflows/ci.yml/badge.svg)](https://github.com/Sumit1673/medical-rag-assistant/.github/workflows/ci.yml)
[![CD](https://github.com/Sumit1673/medical-rag-assistant/.github/workflows/cd.yml/badge.svg)](https://github.com/Sumit1673/medical-rag-assistant/.github/workflows/cd.yml)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-25%20passed-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **production-grade** Retrieval-Augmented Generation (RAG) system that upgrades a basic vector-search pipeline into an advanced, multi-stage architecture — using the same patterns employed by Netflix, Amazon, and Pinecone at scale. For dataset I have created some sample 
medical dataset.

---

## 🔄 Basic vs. Advanced RAG

| Feature | Basic RAG (v1) | Advanced RAG (v2) |
|---|---|---|
| **Retrieval** | Dense vector search only | **Hybrid: BM25 + Dense + RRF** |
| **Query handling** | Raw query passed directly | **LLM query rewriting** |
| **Ranking** | Cosine similarity score | **Cross-encoder reranking** |
| **LLM** | Ollama (local only) | **OpenAI GPT-4o + Ollama** |
| **Embeddings** | HuggingFace only | **OpenAI + HuggingFace** |
| **API** | Synchronous JSON only | **Sync + Streaming SSE** |
| **Evaluation** | None | **RAGAS metrics** |
| **CI/CD** | None | **GitHub Actions (lint, test, docker)** |
| **Chunking** | Fixed-size only | **Recursive + Semantic** |
| **Document formats** | PDF, TXT | **PDF, TXT, CSV, MD, DOCX** |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE  (Offline)                    │
│                                                                     │
│  Documents  ──►  MultiFormat  ──►  Recursive /  ──►  OpenAI /      │
│  (PDF/TXT/        Loader            Semantic         HuggingFace    │
│   CSV/MD/                           Chunker          Embeddings     │
│   DOCX)                                                    │        │
│                                                            ▼        │
│                                                   ChromaDB Store    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE  (Online)                     │
│                                                                     │
│  User Query                                                         │
│      │                                                              │
│      ▼                                                              │
│  ┌──────────────────┐                                               │
│  │  Query Rewriter  │  ◄── GPT-4o rewrites for better recall       │
│  └────────┬─────────┘                                               │
│           │  rewritten query                                        │
│           ▼                                                         │
│  ┌───────────────────────────────────────────────────────┐         │
│  │                 Hybrid Retriever                       │         │
│  │                                                        │         │
│  │  ┌─────────────────┐    ┌─────────────────────────┐  │         │
│  │  │  Dense Retriever│    │  BM25 Sparse Retriever  │  │         │
│  │  │  (ChromaDB)     │    │  (keyword matching)     │  │         │
│  │  └────────┬────────┘    └────────────┬────────────┘  │         │
│  │           └──────────────────────────┘                │         │
│  │                         ▼                              │         │
│  │          Reciprocal Rank Fusion  (k=60)               │         │
│  │              Top-20 fused candidates                   │         │
│  └─────────────────────────┬─────────────────────────────┘         │
│                             │                                       │
│                             ▼                                       │
│  ┌───────────────────────────────────────────────────────┐         │
│  │           Cross-Encoder Reranker                       │         │
│  │  ms-marco-MiniLM-L-6-v2 scores (query, passage) pairs │         │
│  │              Top-5 precision-ranked docs               │         │
│  └─────────────────────────┬─────────────────────────────┘         │
│                             │                                       │
│                             ▼                                       │
│  ┌───────────────────────────────────────────────────────┐         │
│  │        GPT-4o / Ollama — Grounded Answer Generation    │         │
│  │        (structured prompt + source citations)          │         │
│  └────────────────────┬──────────────────────────────────┘         │
│                        │                                            │
│            ┌───────────┴───────────┐                               │
│            ▼                       ▼                               │
│       JSON Response           SSE Stream                           │
│  (answer + sources +      (token-by-token via                      │
│   rerank_score +           EventSource API)                        │
│   latency_ms)                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        CI / CD PIPELINE                             │
│                                                                     │
│  git push ──► GitHub Actions CI ──► lint + test (Py 3.10 & 3.11)  │
│                                  └──► Docker build validation       │
│                                                                     │
│  git tag v* ──► GitHub Actions CD ──► push GHCR image              │
│                                    └──► create GitHub Release       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Advanced Features

### 1. Hybrid Search with Reciprocal Rank Fusion
Two retrieval paradigms combined into one superior ranked list:
- **Dense retrieval** (ChromaDB + embeddings) finds semantically similar documents even with different words
- **BM25 sparse retrieval** excels at exact keyword matching — critical for medical/technical terms
- **RRF fusion** (k=60, Cormack et al. 2009) merges both ranked lists without needing score normalisation

```python
# RRF score per document across both retrieval methods
# RRF(d) = Σ  1 / (k + rank_i(d))
```

### 2. LLM-Powered Query Rewriting
Before retrieval, GPT-4o rewrites vague queries to be more retrieval-friendly:
```
Input:    "what are the side effects?"
Rewritten: "medication side effects adverse reactions clinical symptoms"
```
This improves recall by ~15-25% (Amazon Kendra research).

### 3. Cross-Encoder Reranking
Standard bi-encoders score query and document independently. Cross-encoders process the *(query, document)* pair **jointly**, producing far more accurate relevance scores. Applied only on the top-20 candidates for efficiency — the same two-stage pattern used by Netflix.

### 4. Streaming Responses (SSE)
```javascript
const es = new EventSource("/query/stream");
es.onmessage = ({ data }) => {
  if (data === "[DONE]") return es.close();
  document.getElementById("output").textContent += data;
};
```

### 5. RAGAS Evaluation
Automated quality measurement using three key metrics:
- **Faithfulness** — Is the answer grounded in context?
- **Answer Relevancy** — Does the answer address the question?
- **Context Precision** — Are retrieved chunks relevant?

---

## 📁 Project Structure

```
advanced-rag-system/
├── .github/workflows/
│   ├── ci.yml          # Lint + multi-Python tests + Docker build check
│   └── cd.yml          # Build & push GHCR image + GitHub Release on tags
│
├── src/rag_assistant/
│   ├── core/
│   │   ├── retriever.py          ⭐ Hybrid BM25 + Dense + RRF fusion
│   │   ├── reranker.py           ⭐ Cross-encoder reranking
│   │   ├── query_handler.py      ⭐ Full advanced pipeline + streaming
│   │   ├── llm_handler.py        ⭐ OpenAI GPT-4o + Ollama + async stream
│   │   ├── embedding_generator.py   OpenAI + HuggingFace
│   │   ├── document_loader.py       PDF / TXT / CSV / MD / DOCX
│   │   ├── document_splitter.py     Recursive + Semantic chunking
│   │   └── vector_store_manager.py  ChromaDB client
│   ├── evaluation/
│   │   └── ragas_eval.py         ⭐ RAGAS metrics pipeline
│   ├── pipeline/
│   │   └── ingestion.py             Full offline ingestion orchestrator
│   └── utils/
│       └── config_loader.py         YAML config parser
│
├── tests/                        25 unit tests — all passing ✅
│   ├── conftest.py
│   ├── test_retriever.py
│   ├── test_reranker.py
│   ├── test_query_handler.py
│   └── test_ingestion.py
│
├── dataset/
│   └── download_dataset.py       Downloads medical Q&A from HuggingFace
├── scripts/
│   ├── run_ingest.py             CLI: ingest documents into ChromaDB
│   └── run_eval.py               CLI: run RAGAS evaluation
│
├── config/config.yaml            All configuration (LLM, embeddings, retrieval)
├── app.py                        FastAPI: /query, /query/stream, /health
├── docker-compose.yml            ChromaDB + RAG API
├── Dockerfile
├── requirements.txt
└── .env.example
```

---