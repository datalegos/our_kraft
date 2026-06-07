# RAG-Based AI Chatbot — Project Experience

## Project Overview

Built a production-ready, cost-optimized **AI chatbot** for **DataLegos** — a company information assistant that scrapes the company website, processes documents, and answers queries using a full Retrieval-Augmented Generation (RAG) pipeline. The chatbot was deployed with a web UI and designed to minimize OpenAI API costs through multiple optimization layers.

> **One-liner:** An intelligent company FAQ and knowledge-base chatbot powered by OpenAI GPT-4o-mini, FAISS vector search, and HuggingFace embeddings — with caching, query routing, and conversation memory to keep costs low and responses accurate.

---

## Problem Statement

The client needed a conversational AI assistant that could:
- Answer questions about their company (team, services, contact, offices) accurately
- Use their existing website and internal documents as the knowledge source
- Be cost-efficient — avoiding unnecessary OpenAI API calls
- Work out of the box without requiring users to search through the website manually

---

## What I Built

A full end-to-end AI pipeline consisting of three main phases:

### Phase 1 — Data Ingestion
- **Web Scraper** (`scraper.py`): Crawls the company website, extracts clean text from HTML pages, handles rate limiting, retries, and domain scoping
- **Document Processor** (`document_processor.py`): Ingests PDF, DOCX, TXT, and Markdown files with metadata extraction
- **Semantic Chunker**: Splits content into semantically meaningful chunks (respecting paragraph and sentence boundaries) instead of naive character splits

### Phase 2 — Vector Index
- **Embeddings** (`embeddings.py`): Converts text chunks into dense vectors using the `BAAI/bge-small-en-v1.5` HuggingFace model
- **FAISS Index**: Stores and searches vectors locally for fast semantic similarity retrieval

### Phase 3 — Chatbot Application
- **RAG Pipeline** (`app.py`): Retrieves top relevant chunks, re-ranks them, compresses context, and sends to OpenAI GPT-4o-mini
- **Gradio UI**: Clean web chat interface accessible via browser
- **5 Optimization Modules** (described below)

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **LLM** | OpenAI GPT-4o-mini | Cost-effective, high quality for Q&A tasks |
| **Embeddings** | HuggingFace `BAAI/bge-small-en-v1.5` | Free, fast, high-performing small embedding model — no API cost |
| **Vector Store** | FAISS (CPU) | Local, zero-cost semantic search with fast retrieval |
| **RAG Framework** | LangChain + LangChain Community | Modular pipeline with splitters, vector stores, and document loaders |
| **Web UI** | Gradio | Rapid deployment of chat interface with minimal frontend code |
| **Web Scraping** | BeautifulSoup + Requests | Lightweight, reliable HTML extraction with retry support |
| **Document Parsing** | PyPDF2, python-docx | Handles real-world business documents (PDFs, Word files) |
| **Config Management** | PyYAML | All settings in one clean YAML file — no hardcoded values |
| **Logging** | Python `logging` | Structured logs to file and console for debugging and monitoring |
| **Package** | setuptools + pyproject.toml | Proper Python packaging, installable as a module |
| **Environment** | conda / pip (environment.yml + requirements.txt) | Flexible setup for different team environments |

---

## Why These Technology Choices

**GPT-4o-mini over GPT-4o:**
The application is a FAQ/information retrieval assistant. GPT-4o-mini offers 95%+ of GPT-4's quality at ~10x lower cost for this use case. With the caching and routing layers I added, actual AI calls drop significantly further.

**BAAI/bge-small-en-v1.5 for embeddings:**
This is a top-ranked small embedding model on the MTEB leaderboard. Running it locally means zero API cost for embeddings at every query, and it's fast enough to run on CPU. No dependency on OpenAI embeddings.

**FAISS over cloud vector DBs (Pinecone, Weaviate):**
The knowledge base is static and small-to-medium scale. A local FAISS index gives millisecond retrieval with no monthly SaaS cost and no network latency.

**Gradio over building a custom frontend:**
The chatbot needed a usable UI quickly. Gradio's `ChatInterface` delivers a production-looking web app in under 10 lines, and it can be shared publicly with `share=True` — ideal for demos.

**LangChain:**
Provides the abstractions needed to wire together the vector store, embeddings, and document splitting without reinventing the wheel, while keeping the code readable and modular.

---

## Architecture

```
User Query (Gradio UI)
        │
        ▼
┌─────────────────────┐
│   Response Cache    │ ──── Cache Hit? ──▶ Return Cached Response
│  (similarity match) │
└─────────────────────┘
        │ Cache Miss
        ▼
┌─────────────────────┐
│   Query Router      │ ──── FAQ/Simple? ──▶ Return Pre-computed Response
│  (FAQ + classifier) │
└─────────────────────┘
        │ Complex Query
        ▼
┌─────────────────────┐
│   FAISS Vector DB   │ ◀── HuggingFace Embeddings (Local)
│   Similarity Search │
└─────────────────────┘
        │ Top K Docs
        ▼
┌─────────────────────┐
│  Document Reranker  │
│  (keyword scoring)  │
└─────────────────────┘
        │ Re-ranked Docs
        ▼
┌─────────────────────┐
│  Memory Manager     │
│  (sliding window +  │
│   summarization)    │
└─────────────────────┘
        │ Context + History
        ▼
┌─────────────────────┐
│   Token Controller  │
│   (build prompt,    │
│    enforce limits)  │
└─────────────────────┘
        │ Optimized Prompt
        ▼
┌─────────────────────┐
│   OpenAI API        │
│   GPT-4o-mini       │
└─────────────────────┘
        │ Response
        ▼
┌─────────────────────┐
│   Analytics Tracker │ (logs tokens, cost, cache rate)
└─────────────────────┘
        │
        ▼
     User Answer
```

---

## Key Features Built

### 1. Full RAG Pipeline
Retrieves the top K semantically similar document chunks from FAISS, re-ranks them by keyword relevance and position scoring, then feeds them as context to the LLM. This ensures answers are grounded in real company data — not hallucinated.

### 2. Multi-layer Cost Optimization

| Optimization | How It Works | Impact |
|---|---|---|
| **Response Cache** | Stores query-response pairs on disk; fuzzy similarity matching (85% threshold) catches rephrased repeats | Eliminates API cost for repeated/similar questions |
| **Query Router** | Classifies query as FAQ, simple greeting, or complex — only complex queries hit the LLM | Saves API calls for greetings, thanks, FAQs |
| **Memory Manager** | Sliding window (last 10 messages) + automatic summarization after 20 messages | Prevents prompt token bloat in long conversations |
| **Token Controller** | Enforces input/output token limits, truncates at sentence boundaries | Prevents runaway costs from unexpectedly large inputs |
| **Context Compressor** | Scores and prioritizes document chunks by query relevance before building the prompt | Keeps context within token budget while maximizing relevance |

### 3. Semantic Chunking
Instead of splitting documents at fixed character counts (which can cut mid-sentence), the chunker tries to split at paragraph → sentence → word boundaries in order. This preserves meaning and produces better retrieval results.

### 4. Conversation Memory
Implements a sliding window over conversation history so the chatbot maintains context across turns without blowing up the prompt token count. When history exceeds a threshold, it auto-summarizes older turns using the LLM itself.

### 5. Analytics & Monitoring
Tracks per-session and global metrics: total tokens, costs, cache hit rate, FAQ hit rate, AI call rate. Saved to JSONL logs and JSON summaries for later analysis.

### 6. Document Ingestion Pipeline
Supports PDF, DOCX, TXT, and Markdown out of the box. Also includes a web scraper that can crawl an entire company website, respect rate limits, and extract clean text from HTML.

---

## Project Structure

```
project/
├── src/chatbot/
│   ├── core/
│   │   ├── app.py              # Main chatbot app + Gradio UI
│   │   └── config.py           # All config loaded from config.yaml
│   ├── optimizers/
│   │   ├── analytics.py        # Token/cost/cache tracking
│   │   ├── cost_optimizer.py   # Reranker, compressor, token counter/controller
│   │   ├── memory_manager.py   # Sliding window + summarization
│   │   ├── query_router.py     # FAQ DB + query classifier
│   │   └── response_cache.py   # Disk-based cache with TTL + similarity
│   ├── processors/
│   │   ├── document_processor.py  # PDF/DOCX/TXT parser + semantic chunker
│   │   ├── embeddings.py          # FAISS index creation
│   │   └── scraper.py             # Website crawler
│   └── utils/
│       ├── exceptions.py       # Custom exception classes
│       └── logger.py           # Logging setup
├── scripts/
│   ├── run_chatbot.py          # Launch the app
│   ├── run_scraper.py          # Run web scraper
│   └── create_embeddings.py    # Build FAISS index
├── config.yaml                 # Single source of truth for all settings
└── scraper_config.yaml         # Website-specific scraper settings
```

---

## How to Run (3 Steps)

```bash
# Step 1: Scrape the website (or use existing documents)
python scripts/run_scraper.py

# Step 2: Build the vector index
python scripts/create_embeddings.py

# Step 3: Launch the chatbot
python scripts/run_chatbot.py
# → Opens at http://127.0.0.1:7860
```

---

## Configuration-Driven Design

Everything is controlled via `config.yaml` — no hardcoded values anywhere in the code. This includes:
- OpenAI model, temperature, token limits
- Embedding model and FAISS index path
- Chunk size and overlap
- Cache TTL and similarity threshold
- FAQ matching threshold
- Memory sliding window size
- App title, port, host

This means the same codebase can be redeployed for a different client just by updating the config file.

---

## Challenges & Solutions

**Challenge: Answers were incomplete for multi-part questions (team, address, contact)**
Solution: Increased `k_documents` to 8 and `k_final` to 6, disabled context compression (which was pruning important details), and increased `max_context_tokens` to 4000 so all relevant chunks fit in the prompt.

**Challenge: Repeated questions kept hitting the OpenAI API**
Solution: Built a disk-based response cache with MD5 hashing for exact matches and `SequenceMatcher`-based fuzzy matching for semantically similar questions (85% similarity threshold).

**Challenge: Long conversations bloated the prompt**
Solution: Implemented a sliding window that keeps only the last 10 messages in context, plus automatic LLM-based summarization of older history when conversation exceeds 20 messages.

**Challenge: Some queries were simple greetings or FAQs wasting API calls**
Solution: Built a `QueryRouter` that first checks an FAQ database (fuzzy keyword + string similarity matching), then pattern-matches simple greetings/thanks/byes, and only routes genuinely complex queries to the full RAG + LLM pipeline.

---

## Results / Outcomes

- Chatbot accurately answers questions about team members, company address, services, contact info, and business background
- Multiple optimization layers reduce OpenAI API calls significantly for repeat and FAQ-type questions
- Fully configurable and re-deployable — no code changes needed to adapt to a new client's website
- Clean modular codebase with custom exception handling, structured logging, and proper Python packaging

---

## Skills Demonstrated

- **AI/ML Engineering**: RAG pipeline design, vector embeddings, FAISS, LLM prompt engineering
- **Backend Python**: Modular package architecture, async-ready design, config management
- **Web Scraping**: BeautifulSoup, requests with retry/backoff, rate limiting, HTML text extraction
- **Cost Engineering**: Multi-layer API cost optimization (caching, routing, token control, memory management)
- **DevOps/Packaging**: pyproject.toml, conda environments, YAML-driven configuration
- **System Design**: Separation of concerns across core / optimizers / processors / utils layers

---

## Questions You Might Be Asked (With Answers)

**Q: Why RAG instead of fine-tuning?**
RAG is far more practical for company info use cases. Fine-tuning requires large datasets, is expensive, and the knowledge gets stale. RAG lets you update the knowledge base just by re-running the scraper and embeddings — no retraining needed.

**Q: Why FAISS instead of a cloud vector database?**
For a single-company, relatively small and static knowledge base, FAISS running locally is faster, cheaper (zero cost), and simpler to deploy. There's no need for the scale that Pinecone or Weaviate are designed for.

**Q: How do you handle hallucination?**
The system prompt explicitly instructs the LLM to only answer from the provided context and to say it doesn't know if the information is not present. The RAG retrieval ensures the LLM always has relevant grounding context.

**Q: What's the cost per query?**
With GPT-4o-mini at $0.15/1M input tokens and $0.60/1M output tokens, a typical RAG query costs ~$0.001–$0.003. With caching and routing, a large fraction of queries never hit the API at all.

**Q: Can this be adapted for another company?**
Yes. Update `config.yaml` with the new website URL, API key, and app title. Run the scraper and create new embeddings. The entire pipeline works for any company's website or document set.

**Q: How does the semantic chunking differ from basic splitting?**
Basic splitting cuts at a fixed character count, often mid-sentence. The semantic chunker tries to find natural breakpoints in order: section breaks → paragraphs → sentences → words. This preserves meaning, leading to better retrieval results.
