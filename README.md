# 🎫 TicketMind — AI-Powered Customer Support Agent

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2.16-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.5-FF6B35?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white)

**An autonomous AI agent that reads support tickets, searches a RAG-powered knowledge base, and either resolves them with a drafted reply or escalates — with full reasoning logs.**

[Features](#-features) • [Architecture](#-architecture) • [Setup](#-setup) • [Usage](#-usage) • [Evaluation](#-evaluation) • [Tech Stack](#-tech-stack)

</div>

---

## 🎯 What It Does

TicketMind simulates a **Digital Employee** for customer support operations:

- 📥 **Reads** incoming support tickets (billing, technical, general)
- 🔍 **Searches** a FAQ knowledge base using RAG (Retrieval-Augmented Generation)
- 🤔 **Reasons** step-by-step using LangChain ReAct agent
- ✅ **Resolves** tickets with a professional drafted reply (if confident)
- ⚠️ **Escalates** tickets with reasoning (if confidence is low)
- 📊 **Logs** every decision with category, confidence score, and reasoning

> Built as a portfolio project for the **Zangoh SWE Intern – Generative AI** role, directly mirroring their Digital Employee product use case.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **RAG Pipeline** | ChromaDB + sentence-transformers for semantic FAQ retrieval |
| 🤖 **ReAct Agent** | LangChain agent with 3 tools: search, categorize, escalate |
| 📊 **Confidence Scoring** | Low-confidence tickets auto-escalated with reason |
| 🔍 **Reasoning Log** | Full step-by-step agent thinking visible per ticket |
| 📋 **25 Sample Tickets** | Pre-loaded realistic tickets across 3 categories |
| 🧪 **Eval Dashboard** | LLM-as-judge scoring across 15 test cases |
| 📚 **KB Explorer** | Browse FAQ knowledge base + test retrieval live |
| 🎫 **Ticket History** | Session-level history of all resolved/escalated tickets |

---

*Live Demo:* [Click Here](https://zendesk-agent-lbxpxuudys9c7jqdrwevaq.streamlit.app/)

---

## Screenshot
![one](spa.png)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    TICKETMIND FLOW                       │
└─────────────────────────────────────────────────────────┘

  Customer Ticket Input
          │
          ▼
  ┌───────────────┐
  │   CATEGORIZE  │  ──► billing / technical / general
  │     TOOL      │
  └───────┬───────┘
          │
          ▼
  ┌───────────────┐
  │  SEARCH FAQ   │  ──► ChromaDB Vector Search (Top-3 chunks)
  │     TOOL      │      sentence-transformers embeddings
  └───────┬───────┘
          │
          ▼
  ┌───────────────────────────┐
  │   CONFIDENCE CHECK        │
  │   score ≥ 0.4 ?           │
  └────────┬──────────────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
 RESOLVE      ESCALATE
 (GPT reply)  (flag + reason)
    │             │
    └──────┬──────┘
           ▼
   Resolution Output
   + Reasoning Log
   + Ticket History
```

**Key Components:**

```
ticketmind/
├── data/
│   ├── tickets.json          # 25 sample support tickets
│   └── faq_docs/
│       ├── billing.txt       # Billing FAQ (15+ Q&A pairs)
│       ├── technical.txt     # Technical FAQ (15+ Q&A pairs)
│       └── general.txt       # General FAQ (10+ Q&A pairs)
├── agent/
│   ├── rag_pipeline.py       # ChromaDB vectorstore + retrieval
│   ├── support_agent.py      # LangChain ReAct agent
│   └── evaluator.py          # LLM-as-judge scoring
├── tools/
│   ├── search_faq.py         # Tool: semantic FAQ search
│   ├── categorize.py         # Tool: ticket categorization
│   └── escalate.py           # Tool: escalation logic
├── eval/
│   ├── test_cases.json       # 15 labeled test cases
│   └── run_eval.py           # Automated eval script
├── app.py                    # Streamlit UI
├── .env.example              # API key template
└── requirements.txt
```

---

## ⚙️ Setup

### Prerequisites
- Python 3.11+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/ticketmind.git
cd ticketmind
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Key
```bash
# Copy the example env file
cp .env.example .env
```

Open `.env` and add your key:
```
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 4. Run the App
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` 🚀

---

## 🖥️ Usage

### Tab 1 — Resolve Ticket
1. Select a sample ticket from the dropdown **or** type your own
2. Click **"🚀 Resolve Ticket"**
3. View the resolution with:
   - Color-coded status (✅ Resolved / ⚠️ Escalated)
   - Category badge
   - Confidence score
   - Full agent reasoning log

### Tab 2 — Evaluation Dashboard
1. Click **"Run Evaluation"**
2. Agent runs all 15 test cases automatically
3. View accuracy %, per-category scores, and LLM-as-judge feedback

### Tab 3 — Knowledge Base Explorer
1. Browse all 3 FAQ categories
2. Type any query to test live retrieval
3. See top-3 matching chunks with similarity scores

---

## 🧪 Evaluation

The project includes an automated evaluation system:

```bash
python eval/run_eval.py
```

**Metrics tracked:**
| Metric | Description |
|---|---|
| Action Accuracy | % of tickets correctly resolved vs escalated |
| Relevance Score | Does the reply address the ticket? (1-5) |
| Tone Score | Is the response professional? (1-5) |
| Correctness Score | Is the action appropriate? (1-5) |
| Overall Score | Average of all 3 (1-5) |

Sample output:
```
┌──────────┬────────────┬───────────┬───────┬─────────────────────────┐
│ Ticket   │ Category   │ Action    │ Score │ Feedback                │
├──────────┼────────────┼───────────┼───────┼─────────────────────────┤
│ TICK-001 │ technical  │ resolved  │ 4.3   │ Clear and relevant      │
│ TICK-002 │ billing    │ resolved  │ 4.7   │ Professional tone       │
│ TICK-003 │ general    │ escalated │ 4.1   │ Correct escalation      │
└──────────┴────────────┴───────────┴───────┴─────────────────────────┘

✅ Accuracy: 13/15 correct actions
📊 Average Score: 4.2 / 5.0
```

---

## 🛠️ Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| **Python** | 3.11+ | Core language |
| **LangChain** | 0.2.16 | Agent orchestration (ReAct) |
| **OpenAI GPT-4o-mini** | latest | LLM reasoning + reply drafting |
| **ChromaDB** | 0.5.5 | Vector database for FAQ storage |
| **sentence-transformers** | 3.0.1 | Local embeddings (all-MiniLM-L6-v2) |
| **Streamlit** | 1.38.0 | Web UI |
| **Pandas** | 2.2.2 | Eval results processing |
| **Plotly** | 5.24.1 | Eval score charts |
| **python-dotenv** | 1.0.1 | Environment variable management |

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ Yes | OpenAI API key for GPT-4o-mini |

---

## 🚀 Key Design Decisions

**Why ChromaDB over Pinecone?**
ChromaDB runs locally with zero setup — no account, no API key, no cost. For a portfolio project, this means anyone can clone and run immediately.

**Why sentence-transformers for embeddings?**
Avoids an extra OpenAI API call per ticket. `all-MiniLM-L6-v2` is fast, free, and runs locally — reducing latency and cost in production.

**Why ReAct agent over simple RAG chain?**
ReAct forces the LLM to reason step-by-step before acting. This makes the escalation decision traceable and explainable — a critical requirement in real business support systems.

**Why LLM-as-judge for evaluation?**
Rule-based scoring can't measure response quality. Using GPT-4o-mini as an evaluator gives nuanced feedback on tone, relevance, and correctness — similar to how Zangoh's Zing dashboard measures Digital Employee performance.

---

## 📈 Roadmap

- [ ] Add multi-turn conversation memory
- [ ] Integrate with real CRM API (HubSpot free tier)
- [ ] Add PEFT fine-tuning on support ticket dataset
- [ ] Deploy to Hugging Face Spaces
- [ ] Add nightly eval with GitHub Actions
- [ ] Export escalated tickets to Google Sheets

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

1. Fork the repo
2. Create your branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'feat: add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 👤 Author


Reach me here:
🔗 [Linktree](https://linktr.ee/hemantsharma22?fbclid=PAQ0xDSwLbT41leHRuA2FlbQIxMQABp6bFdMywhk2GzbSiCfWfDCb8gXvykT8vF0bZEOt6SykMrXjh5t9-hKWpy3Ak_aem_0I6JJKhw2812C9Gu80zg5A) - All My Links
---

<div align="center">


*Demonstrating RAG pipelines, LangChain agents, and production-grade evaluation systems*

</div>
