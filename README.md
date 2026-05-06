# ZenDesk Customer Support Ticket Resolution Agent

> An end-to-end AI agent that resolves customer support tickets using LangChain, ChromaDB, and GPT-4o-mini.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up your API key

```bash
cp .env.example .env
# Then open .env and add your key:
# OPENAI_API_KEY=sk-...
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 🏗️ Project Structure

```
ZenDesk Agent/
├── agent/
│   ├── __init__.py
│   └── agent.py              # LangChain ReAct agent + executor
├── tools/
│   ├── __init__.py
│   ├── knowledge_base.py     # ChromaDB ingestion + KB retrieval tool
│   ├── ticket_tools.py       # Priority classifier + resolution drafter
│   └── sentiment.py          # Customer sentiment analysis tool
├── eval/
│   ├── __init__.py
│   └── evaluate.py           # Evaluation runner (KB precision, quality score)
├── data/
│   ├── knowledge_base.json   # 12 KB articles (billing, account, technical, etc.)
│   └── sample_tickets.json   # 5 labelled tickets for evaluation
├── chroma_db/                # Auto-generated ChromaDB vector store
├── app.py                    # Streamlit UI (UI only — no business logic)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🧠 How It Works

The agent follows a **4-step ReAct protocol** for every ticket:

| Step | Tool | Purpose |
|------|------|---------|
| 1 | `analyse_sentiment` | Detect emotional tone → tailor response empathy |
| 2 | `classify_ticket_priority` | Keyword heuristic → critical / high / medium / low |
| 3 | `search_knowledge_base` | ChromaDB semantic search → relevant KB articles |
| 4 | `draft_resolution` | Structured template → final customer-facing response |

### Embeddings
`all-MiniLM-L6-v2` from `sentence-transformers` is used to embed KB articles and queries — no OpenAI API calls for embeddings.

### Vector Store
ChromaDB persists to `./chroma_db/`. The first run auto-ingests `data/knowledge_base.json`. Subsequent runs use the cached collection.

---

## 📊 Running the Evaluation Suite

```bash
python -m eval.evaluate
```

Evaluates the agent against 5 labelled tickets and reports:
- **KB Citation Precision** — were the right articles referenced?
- **Quality Pass Rate** — does the response include all required sections?

Results are saved to `eval/results.json`.

---

## 🔧 Code Standards

- Python 3.11+ with full **type hints**
- **Docstrings** on all functions (Google style)
- API keys via **python-dotenv** (never hardcoded)
- **Error handling** on all external API calls
- Business logic strictly separated from UI (`app.py` is UI-only)

---

## 📦 Stack

| Component | Library |
|-----------|---------|
| LLM | `gpt-4o-mini` via `langchain-openai` |
| Agent orchestration | `langchain` ReAct |
| Vector store | `chromadb` + `langchain-chroma` |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| UI | `streamlit` |
| Env management | `python-dotenv` |
