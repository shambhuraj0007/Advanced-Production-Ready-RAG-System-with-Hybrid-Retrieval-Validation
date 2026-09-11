# 🚀 Advanced Production-Ready RAG System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Cloud Run](https://img.shields.io/badge/Deploy-GCP%20Cloud%20Run-4285F4.svg?logo=google-cloud&logoColor=white)](https://cloud.google.com/run)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-green.svg)](https://python.langchain.com/)
[![VectorDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![Reranker](https://img.shields.io/badge/Reranker-Cross--Encoder-orange.svg)](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
[![Evaluation](https://img.shields.io/badge/Evaluation-RAGAS-red.svg)](https://github.com/explodinggradients/ragas)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance, modular **Retrieval-Augmented Generation (RAG)** pipeline engineered with **LangChain (LCEL)**, combining **Hybrid Search (ChromaDB + BM25)**, **Cross-Encoder Re-ranking**, **Dynamic Query Routing**, **Hallucination Verification**, **Router-Aware External Tools**, and **Automated Quantitative Evaluation using RAGAS**.

---

## 🏛️ System Architecture

```
                                  ┌────────────────────────┐
                                  │      User Question     │
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                 ┌──────────────────────────┐
                                 │  🧠 Query Router (LLM)   │ ──► Classify: Factual / Analytical / Multi-hop
                                 └────────────┬─────────────┘
                                              │
                        ┌─────────────────────┴─────────────────────┐
                        ▼                                           ▼
          ┌───────────────────────────┐               ┌───────────────────────────┐
          │  🔄 Contextual Rewriter   │               │   🛠️ External Tools       │
          │   (Expands & Clarifies)   │               │ (Tavily Search / Math)    │
          └─────────────┬─────────────┘               └─────────────┬─────────────┘
                        │                                           │
                        ▼                                           │
         ┌─────────────────────────────┐                            │
         │     🔍 Hybrid Retrieval     │                            │
         ├──────────────┬──────────────┤                            │
         │ Dense Search │ Sparse BM25  │                            │
         │  (ChromaDB)  │   (Okapi)    │                            │
         └──────┬───────┴──────┬───────┘                            │
                └───────┬──────┘                                    │
                        ▼                                           │
         ┌─────────────────────────────┐                            │
         │   🎯 Cross-Encoder Rerank   │                            │
         │ (ms-marco-MiniLM-L-6-v2)    │                            │
         └──────────────┬──────────────┘                            │
                        │                                           │
                        └─────────────────────┬─────────────────────┘
                                              │
                                              ▼
                                 ┌──────────────────────────┐
                                 │  🤖 LLM Synthesis (LCEL) │ (Gemini / Groq / OpenAI)
                                 └────────────┬─────────────┘
                                              │
                        ┌─────────────────────┴─────────────────────┐
                        ▼                                           ▼
          ┌───────────────────────────┐               ┌───────────────────────────┐
          │  🛡️ Answer Validator      │               │  📎 Citation Engine       │
          │ (Hallucination Detection) │               │ (Page, Source, Line Num)  │
          └─────────────┬─────────────┘               └─────────────┬─────────────┘
                        │                                           │
                        └─────────────────────┬─────────────────────┘
                                              │
                                              ▼
                                 ┌──────────────────────────┐
                                 │ 📊 Response + Metadata   │
                                 │ (Modern Web UI / REST SDK│
                                 └──────────────────────────┘
```

---

## ✨ Core Production Features

### 🔍 1. Hybrid Search (Dense + Sparse)
- **Vector Search (ChromaDB)**: Captures deep semantic meaning and synonyms using state-of-the-art embedding models.
- **Keyword Search (BM25 Okapi)**: Captures exact keyword matches, technical codes, product identifiers, and domain terms that embeddings often miss.
- **Tunable Fusion ($\alpha$ Weighting)**: Dynamically balance sparse and dense scores ($\alpha = 0.5$ for balanced retrieval).

### 🎯 2. Cross-Encoder Re-Ranking
- Re-scores retrieved candidate chunks using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Unlike bi-encoders, cross-encoders attend across both the query and document tokens simultaneously, placing the most contextually relevant chunks at the very top of the context window.

### 🧭 3. Query Classification & Dynamic Routing
- Analyzes incoming queries before retrieval:
  - **Factual**: Direct retrieval with strict $k$.
  - **Analytical / Multi-hop**: Expands $k$ and enables multi-step context gathering.
  - **External / Real-time**: Dispatches to search tools when local context is insufficient.

### 🔄 4. Conversational Memory & Query Rewriting
- Maintains multi-turn conversation context using conversation buffer memory.
- Uses LLMs to resolve coreferences and pronouns in follow-up queries before executing retrieval.

### 🛡️ 5. Automated Answer Validation (Hallucination Guard)
- Evaluates the generated answer against the retrieved source passages before presentation.
- Generates a **confidence score (0.0 to 1.0)** and highlights unsupported claims and warnings.

### 🛠️ 6. Router-Aware External Tools
- **Web Search**: Integrates **Tavily Search API** and **DuckDuckGo** for real-time web retrieval.
- **Math Calculator**: Deterministic calculator tool to prevent LLM arithmetic hallucinations.

### 📊 7. Automated Ragas Evaluation Suite
- Built-in evaluation pipeline (`evaluate_ragas.py`) testing the **RAG Triad**:
  - **Faithfulness**: Validates whether the answer is derived strictly from context.
  - **Answer Relevance**: Checks semantic alignment with the prompt.
  - **Context Precision**: Evaluates the signal-to-noise ratio in retrieved context.
  - **Context Recall**: Verifies all required ground-truth facts are captured.

### ⏱️ 8. Real-time Latency & Query Metadata Tracking
- Live telemetry displayed directly in the web interface showing query latency (seconds), query classification type (factual, analytical, multi-hop), and effective $k$.

### 🔀 9. Multi-Provider Flexibility
- Clean, composable LangChain Expression Language (LCEL) supporting hot-swapping between:
  - **Google Gemini** (Gemini 1.5 Flash / Pro, `text-embedding-004`)
  - **Groq Cloud** (Llama 3.3 70B Versatile, high-speed inference)
  - **OpenAI** (GPT-4o, GPT-4o-mini, `text-embedding-3-small`)

---

## 📁 Repository Structure

```
├── src/
│   ├── __init__.py
│   ├── rag_system.py          # Core RAG orchestration pipeline (LCEL)
│   ├── retriever.py           # Hybrid search (ChromaDB + BM25) & Cross-Encoder re-ranker
│   ├── query_router.py        # Intent classification & dynamic strategy router
│   ├── query_rewriter.py      # Coreference resolution & query expansion
│   ├── validator.py           # Hallucination detector & groundedness validator
│   ├── citation_formatter.py  # Source attribution & metadata formatting
│   ├── tool_manager.py        # Router-aware external tool dispatcher
│   ├── tools.py               # Tool implementations (Tavily, DDG, Calculator)
│   └── memory_manager.py      # Conversational context & history management
├── documents/                 # Knowledge base documents (PDF, TXT, MD)
├── static/
│   └── index.html             # Modern HTML5 + TailwindCSS + JS Web UI
├── tests/
│   └── eval_dataset.json      # Golden benchmark dataset for automated testing
├── api.py                     # FastAPI asynchronous REST backend microservice
├── evaluate_ragas.py          # Automated RAGAS evaluation runner
├── requirements.txt           # Production dependencies
├── .env.example              # Environment variables template
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/shambhuraj0007/Advanced-Production-Ready-RAG-System-with-Hybrid-Retrieval-Validation.git
cd Advanced-Production-Ready-RAG-System-with-Hybrid-Retrieval-Validation

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the template and add your API keys:

```bash
cp .env.example .env
```

Edit `.env`:
```env
# Primary LLM Provider (gemini, groq, or openai)
RAG_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Groq for fast inference
GROQ_API_KEY=your_groq_api_key_here

# Optional: Web search
TAVILY_API_KEY=your_tavily_api_key_here
```

---

## 💻 Usage

### 1. Launch the Modern Web Application (FastAPI + TailwindCSS)

```bash
python api.py
```
*Access the decoupled web UI at `http://127.0.0.1:8000` with interactive Swagger OpenAPI documentation at `http://127.0.0.1:8000/docs`.*

---

### 2. Run Automated RAGAS Evaluation

Benchmark the accuracy of your RAG pipeline against ground truth data:

```bash
python evaluate_ragas.py
```

**Sample Evaluation Output:**
```
======================================================================
 RAGAS BENCHMARK RESULTS
======================================================================
 Faithfulness             : 0.9333
 Answer Relevancy         : 0.8872
 Context Precision        : 0.9167
 Context Recall           : 0.8750
======================================================================
 Detailed results saved to benchmark_results.json
```

---

### 3. Programmatic Python SDK

```python
from src.rag_system import SimpleRAG

# Initialize with desired features
rag = SimpleRAG(
    provider="gemini",
    use_hybrid_search=True,
    hybrid_search_alpha=0.6,
    use_reranking=True,
    rerank_top_k=3,
    use_answer_validation=True
)

# Ingest documents
rag.load_from_files(["documents/sample.txt"])

# Query the pipeline
response = rag.query("What is RAG and why is it useful?")

print("Answer:", response["result"])
print("Validation Score:", response["validation"]["confidence"])
print("Citations:", response["citations"])
```

---

## ⚙️ Configuration Reference

All settings can be customized via `.env` or passed directly to `SimpleRAG()`:

| Parameter | Env Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `provider` | `RAG_PROVIDER` | `gemini` | Model provider (`gemini`, `groq`, `openai`) |
| `embedding_model` | `RAG_EMBEDDING_MODEL` | Provider default | Embedding model identifier |
| `llm_model` | `RAG_LLM_MODEL` | Provider default | LLM model identifier |
| `persist_directory` | `RAG_PERSIST_DIRECTORY` | `./chroma_db` | ChromaDB persistence path |
| `use_hybrid_search` | `RAG_USE_HYBRID_SEARCH` | `true` | Combine vector search with BM25 |
| `hybrid_search_alpha` | `RAG_HYBRID_ALPHA` | `0.5` | $0.0 = \text{BM25 only}$, $1.0 = \text{Vector only}$ |
| `use_reranking` | `RAG_USE_RERANKING` | `true` | Cross-encoder re-ranking flag |
| `rerank_top_k` | `RAG_RERANK_TOP_K` | `3` | Documents retained post-rerank |
| `use_query_routing` | `RAG_USE_QUERY_ROUTING` | `true` | Intent-based adaptive routing |
| `use_query_rewriting`| `RAG_USE_QUERY_REWRITING`| `false` | Contextual query expansion |
| `use_answer_validation`| `RAG_USE_ANSWER_VALIDATION`| `false` | Hallucination verification |
| `use_tools` | `RAG_USE_TOOLS` | `true` | Enable web search & calculator |

---

## 🐳 Containerization & Cloud Deployment

This service is fully containerized and production-ready for **Google Cloud Run**, **AWS App Runner**, or any Docker-compatible infrastructure.

### Option A: Run Locally via Docker Compose
```bash
# Build and run the microservice
docker compose up --build

# Access the Web UI and API Docs
# UI:   http://localhost:8080
# Docs: http://localhost:8080/docs
```

### Option B: Deploy to Google Cloud Run (Recommended)
1. **Build & Deploy via Google Cloud CLI:**
```bash
# Authenticate with GCP
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Deploy directly from source to Cloud Run
gcloud run deploy rag-service \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GROQ_API_KEY="your_groq_key",GEMINI_API_KEY="your_gemini_key",TAVILY_API_KEY="your_tavily_key"
```

2. **Or Deploy via GCP Cloud Console (No CLI needed):**
- Go to [Google Cloud Run Console](https://console.cloud.google.com/run).
- Click **Create Service** $\to$ **Continuously deploy from a repository** (Connect your GitHub repo).
- Select branch `main` and choose **Dockerfile**.
- Under **Variables & Secrets**, add:
  - `GROQ_API_KEY`
  - `GEMINI_API_KEY`
  - `TAVILY_API_KEY`
- Click **Create**. Google Cloud automatically builds the container and provisions a global HTTPS endpoint!

---

## 🛡️ License

This project is licensed under the [MIT License](LICENSE). Contributions and pull requests are welcome!
