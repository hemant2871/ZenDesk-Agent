# 🎫 Customer Support Ticket Resolution Agent

> AI-powered support agent that reads a ticket, searches a FAQ knowledge base via RAG, and either resolves it with a drafted reply or escalates it — with full reasoning logs.

[![Built for Zangoh](https://img.shields.io/badge/Built%20for-Zangoh%20SWE%20Intern-7c3aed?style=flat-square)](https://zangoh.com)

---

## Architecture

```
Customer Ticket
      │
      ▼
┌─────────────────┐
│  categorize_    │  Keyword matching → billing / technical / general
│  ticket (Tool)  │  + confidence: high / medium / low
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  search_faq     │  ChromaDB semantic search (all-MiniLM-L6-v2)
│  (Tool)         │  Top-3 chunks · similarity threshold 0.4
└────────┬────────┘
         │
    ┌────┴─────┐
    │          │
score≥0.4   score<0.4
    │          │
    ▼          ▼
┌───────┐  ┌──────────────┐
│ Draft │  │ escalate_    │
│ Reply │  │ ticket (Tool)│
└───┬───┘  └──────┬───────┘
    │              │
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │  LLM Agent   │  GPT-4o-mini · ZERO_SHOT_REACT_DESCRIPTION
    │  Final Answer│  + JSON metadata block
    └──────┬───────┘
           │
    ┌──────┴───────┐
    │ Reasoning Log│  Full intermediate steps
    └──────────────┘
```

---

## Setup

```bash
# 1. Clone / navigate to the project
cd support-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenAI API key
cp .env.example .env
# Edit .env → OPENAI_API_KEY=sk-...

# 4. Run the app (vectorstore builds automatically on first run)
streamlit run app.py
```

---

## Run Evaluation

```bash
python -m eval.run_eval
# or
python eval/run_eval.py
```

---

## Project Structure

```
support-agent/
├── data/
│   ├── tickets.json          # 25 sample support tickets
│   └── faq_docs/
│       ├── billing.txt       # 16 billing Q&A pairs
│       ├── technical.txt     # 15 technical Q&A pairs
│       └── general.txt       # 11 general Q&A pairs
├── agent/
│   ├── rag_pipeline.py       # ChromaDB + sentence-transformers RAG
│   ├── support_agent.py      # LangChain ZERO_SHOT_REACT agent
│   └── evaluator.py          # LLM-as-judge scoring
├── tools/
│   ├── search_faq.py         # FAQ retrieval tool
│   ├── categorize.py         # Keyword categorization tool
│   └── escalate.py           # Escalation + log tool
├── eval/
│   ├── test_cases.json       # 15 labelled test cases
│   └── run_eval.py           # Evaluation runner
├── chroma_db/                # Auto-generated vector store
├── app.py                    # Streamlit UI
├── .env.example
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Component | Library | Notes |
|-----------|---------|-------|
| LLM | `gpt-4o-mini` | via `langchain-openai` |
| Agent | LangChain `ZERO_SHOT_REACT` | 3 tools registered |
| Vector Store | ChromaDB | Persistent, `./chroma_db/` |
| Embeddings | `all-MiniLM-L6-v2` | sentence-transformers, no API cost |
| UI | Streamlit | Dark glassmorphism theme |
| Evaluation | GPT-4o-mini as judge | Relevance · Tone · Correctness |

---

## Evaluation Results

> Run `python -m eval.run_eval` to populate this section.

| Metric | Score |
|--------|-------|
| Accuracy (correct actions) | TBD / 15 |
| Average Score | TBD / 5.0 |
| Best Score | TBD / 5.0 |
| Avg — Billing | TBD |
| Avg — Technical | TBD |
| Avg — General | TBD |

---

## Screenshots

> _Run `streamlit run app.py` to see the live UI._

---

*Built for Zangoh SWE Intern role preparation.*
