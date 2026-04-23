# University RAG System - Bachelor Thesis

A **Retrieval-Augmented Generation (RAG)** system for the National University "Lviv Polytechnic" (NULP) that provides an intelligent chatbot answering student questions about university life, regulations, campus, scholarships, and more.

Developed as a **Bachelor Thesis** project comparing semantic search techniques and neural re-ranking approaches.

---

## Project Structure

```
├── src/                                  # Core RAG pipeline
│   ├── technique_1_basic/                # Config A: Semantic → Keyword boosting
│   │   ├── complete_rag_system.py        # Full RAG pipeline (Gemini + LapaLLM)
│   │   ├── hybrid_search.py             # Semantic search + keyword boosting
│   │   ├── fast_query_analyzer.py       # BM25-based query analyzer (no LLM)
│   │   ├── lapa_generator.py            # LapaLLM local model (GPU)
│   │   ├── llm1.py                      # Gemini-based query analyzer
│   │   ├── vector_search_engine.py      # ChromaDB vector interface
│   │   └── prompt/                      # System prompts for LLM
│   │
│   ├── technique_2_enhanced/             # Config B: Keyword → Semantic re-ranking
│   │   ├── improved_complete_rag_system.py
│   │   ├── improved_hybrid_search.py
│   │   └── llm1_enhanced.py
│   │
│   ├── data_processing/                  # Data collection & chunking
│   │   ├── chunk.py                     # Semantic document splitting
│   │   └── create_and_save_embeddings.py
│   │
│   └── run_evaluation.py                # Batch evaluation runner
│
├── ai_bot_web/                           # Web UI (Chrome extension)
│   ├── bot.js                           # Chat widget + Maps + KaTeX
│   └── style.css                        # University-branded styles
│
├── api_server.py                         # Flask API backend (:5001)
├── build_demo_cache.py                   # Pre-build demo cache
│
├── data/                                 # Knowledge base
│   ├── processed_documents/             # 50+ cleaned text documents
│   ├── chunked_documents.json           # Chunks (512 tokens)
│   └── chunked_documents_128.json       # Chunks (128 tokens)
│
├── vector_db/                            # ChromaDB vector database
├── models/                               # Trained re-ranking models
│   ├── kan_model.pth                    # KAN re-ranker
│   ├── xnet_model.pth                   # XNet re-ranker
│   └── mlp_model.pth                    # MLP baseline
│
├── evaluation/                           # Evaluation scripts & results
│   ├── ground_truth/                    # Ground truth Q&A pairs
│   └── results/                         # Metrics & comparison tables
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/bachelor_project_lnu.git
cd bachelor_project_lnu

python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 3. Run the API server

```bash
python api_server.py
```

The server starts at `http://localhost:5001`. Test it:

```bash
# Health check
curl http://localhost:5001/health

# Ask a question
curl -X POST http://localhost:5001/ask \
  -H "Content-Type: application/json" \
  -d '{"user_query": "Де знаходиться 19 корпус?"}'
```

### 4. Web UI (Chrome Extension)

The web interface is injected into the official NULP website via a userscript manager (e.g., Tampermonkey):

1. Install [Tampermonkey](https://www.tampermonkey.net/) in Chrome
2. Create a new script and paste the contents of `ai_bot_web/bot.js`
3. Add `@require` for the CSS from `ai_bot_web/style.css`
4. Navigate to [lpnu.ua](https://lpnu.ua) — the chat widget appears in the bottom-right corner

---

## LLM Backends

The system supports **two LLM backends** for answer generation (Stage 3 of the RAG pipeline):

### Option A: Google Gemini API (default)

Uses `gemini-flash-latest` via the Google Generative AI API. Fast, no GPU required.

```bash
# Set your API key in .env
GOOGLE_API_KEY=your_key_here

# Run normally
python api_server.py
```

### Option B: LapaLLM on GPU (vast.ai)

[LapaLLM](https://huggingface.co/lapa-llm/lapa-v0.1.2-instruct) is a Ukrainian-language instruction-tuned model based on Gemma 3. It was deployed on a **rented NVIDIA GPU via [vast.ai](https://vast.ai)** for this thesis.

#### Running LapaLLM on a GPU server (vast.ai)

```bash
# On the GPU instance (vast.ai with CUDA)
pip install torch transformers accelerate

# Start the server with LapaLLM
python api_server.py --use-lapa
```

#### LapaLLM usage example (standalone)

```python
from transformers import pipeline
import torch

pipe = pipeline(
    "image-text-to-text",
    model="lapa-llm/lapa-v0.1.2-instruct",
    device="cuda",
    torch_dtype=torch.bfloat16
)

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "Ти — університетський асистент."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Як розраховується семестрова рейтингова оцінка?"}
        ]
    }
]

output = pipe(text=messages, max_new_tokens=512)
print(output[0]["generated_text"][-1]["content"])
```

#### LapaLLM integration in the RAG pipeline

The file `src/technique_1_basic/lapa_generator.py` provides a **drop-in replacement** for Gemini. It automatically selects the best device (CUDA > MPS > CPU):

```python
from complete_rag_system import CompleteRAGSystem

rag = CompleteRAGSystem(
    db_path="vector_db",
    collection_name="hybrid_collection",
    use_lapa=True   # ← switches to local LapaLLM
)

result = rag.query("Які умови участі в програмі Erasmus+?")
print(result["answer"])
```

---

## RAG Pipeline Architecture

```
User Query
    │
    ▼
┌──────────────────────┐
│ Stage 1: Query       │  FastQueryAnalyzer (BM25-based keyword extraction)
│ Analysis             │  or LLMQueryAnalyzer (Gemini-based)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Stage 2: Hybrid      │  Semantic search (paraphrase-multilingual-mpnet-base-v2)
│ Document Retrieval   │  + Keyword boosting → top-100 documents
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Stage 3: Answer      │  Gemini Flash API  OR  LapaLLM (local GPU)
│ Generation           │  → Ukrainian-language answer with source citations
└──────────────────────┘
```

### Two Search Configurations Compared

| | **Config A** (technique_1) | **Config B** (technique_2) |
|---|---|---|
| Stage 1 | Semantic search first | Keyword retrieval first |
| Stage 2 | Keyword boosting refines | Semantic re-ranking |
| Best for | Broad conceptual queries | Specific entity lookups |
| Example | "How to connect to Wi-Fi?" | "Where is building 19?" |

---

## Web Interface Features

- **Chat widget** embedded in the university website
- **Google Maps** auto-detection and embedding for address queries
- **KaTeX** rendering for mathematical formulas (e.g., rating calculations)
- **Markdown** formatting with proper lists, bold text, and links
- **Source citations** with clickable links to official documents
- **Typewriter effect** for realistic response animation
- **Deduplication** of sources (max 2 unique relevant documents shown)

---

## Evaluation

Evaluated on **30 test queries** across 5 categories using RAGAS metrics:

1. Navigation and Infrastructure
2. Educational Process and Academic Rules
3. Scholarships
4. Student Services, Events, and Organizations
5. Structure and Institutions

Results and comparison tables are available in `evaluation/results/`.

---

## Technologies

| Component | Technology |
|---|---|
| Embedding Model | `paraphrase-multilingual-mpnet-base-v2` |
| Vector Database | ChromaDB (cosine similarity) |
| LLM (API) | Google Gemini Flash |
| LLM (Local) | LapaLLM v0.1.2 (Gemma 3 fine-tune) |
| Re-ranking | KAN / XNet / MLP neural models |
| Backend | Flask + Flask-CORS |
| Frontend | Vanilla JS (Tampermonkey userscript) |
| Math Rendering | KaTeX |
| Maps | Google Maps Embed API |
| GPU Hosting | vast.ai (NVIDIA) |

---

## License

This project was developed as part of a bachelor thesis at the National University "Lviv Polytechnic".

## Author

Liliana Mirchuk
