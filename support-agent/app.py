"""
Streamlit UI — Customer Support Ticket Resolution Agent.
Pure UI layer: all business logic lives in agent/ and tools/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

# ── sys.path ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Support Agent | Zangoh",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#0d0d18 0%,#11111f 100%);min-height:100vh;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:1.8rem 2.2rem;max-width:1300px;}

/* Sidebar */
section[data-testid="stSidebar"]{background:rgba(255,255,255,0.03);border-right:1px solid rgba(255,255,255,0.07);}
section[data-testid="stSidebar"] .stMarkdown h3{color:#a78bfa;font-size:.8rem;letter-spacing:.09em;text-transform:uppercase;margin:0 0 .5rem;}

/* Hero */
.hero{background:linear-gradient(135deg,rgba(124,58,237,.22) 0%,rgba(59,130,246,.12) 100%);
      border:1px solid rgba(124,58,237,.3);border-radius:18px;padding:1.6rem 2rem;margin-bottom:1.5rem;}
.hero h1{font-size:1.8rem;font-weight:700;background:linear-gradient(135deg,#a78bfa,#60a5fa);
         -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:0;}
.hero p{color:rgba(255,255,255,.5);font-size:.88rem;margin:.3rem 0 0;}

/* Cards */
.glass{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);
       border-radius:14px;padding:1.2rem 1.4rem;margin-bottom:.9rem;}

/* Badges */
.badge{display:inline-block;padding:.22rem .7rem;border-radius:999px;font-size:.72rem;
       font-weight:600;letter-spacing:.05em;text-transform:uppercase;}
.badge-resolved{background:rgba(34,197,94,.18);color:#86efac;border:1px solid rgba(34,197,94,.3);}
.badge-escalated{background:rgba(249,115,22,.18);color:#fdba74;border:1px solid rgba(249,115,22,.3);}
.badge-billing{background:rgba(59,130,246,.18);color:#93c5fd;border:1px solid rgba(59,130,246,.3);}
.badge-technical{background:rgba(168,85,247,.18);color:#d8b4fe;border:1px solid rgba(168,85,247,.3);}
.badge-general{background:rgba(20,184,166,.18);color:#5eead4;border:1px solid rgba(20,184,166,.3);}
.badge-high{background:rgba(34,197,94,.15);color:#86efac;border:1px solid rgba(34,197,94,.25);}
.badge-medium{background:rgba(234,179,8,.15);color:#fde047;border:1px solid rgba(234,179,8,.25);}
.badge-low{background:rgba(100,116,139,.15);color:#94a3b8;border:1px solid rgba(100,116,139,.25);}

/* Inputs */
.stTextInput input,.stTextArea textarea{background:rgba(255,255,255,.05)!important;
  border:1px solid rgba(255,255,255,.1)!important;border-radius:10px!important;color:#f1f5f9!important;}
.stTextInput input:focus,.stTextArea textarea:focus{border-color:rgba(124,58,237,.6)!important;
  box-shadow:0 0 0 3px rgba(124,58,237,.15)!important;}
.stTextInput label,.stTextArea label,.stSelectbox label{color:rgba(255,255,255,.65)!important;font-size:.82rem!important;}

/* Buttons */
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#7c3aed,#3b82f6)!important;
  color:white!important;border:none!important;border-radius:10px!important;
  font-weight:600!important;width:100%!important;transition:all .2s!important;}
.stButton>button[kind="primary"]:hover{transform:translateY(-1px)!important;
  box-shadow:0 8px 25px rgba(124,58,237,.4)!important;}
.stButton>button:not([kind="primary"]){background:rgba(255,255,255,.05)!important;
  color:rgba(255,255,255,.7)!important;border:1px solid rgba(255,255,255,.1)!important;
  border-radius:10px!important;width:100%!important;}

/* Response box */
.resp-box{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);
          border-radius:12px;padding:1.2rem;font-size:.87rem;color:#e2e8f0;
          white-space:pre-wrap;line-height:1.7;}

/* Metric mini card */
.mcard{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);
       border-radius:11px;padding:.8rem 1rem;text-align:center;}
.mcard .val{font-size:1.6rem;font-weight:700;color:#a78bfa;}
.mcard .lbl{font-size:.7rem;color:rgba(255,255,255,.4);margin-top:.1rem;}

/* Misc */
hr{border-color:rgba(255,255,255,.07)!important;margin:1rem 0!important;}
.stSuccess,.stInfo,.stWarning,.stError{border-radius:10px!important;}
.stTabs [data-baseweb="tab"]{color:rgba(255,255,255,.55)!important;font-weight:500;}
.stTabs [aria-selected="true"]{color:#a78bfa!important;}
.stTabs [data-baseweb="tab-border"]{background:#7c3aed!important;}
</style>""", unsafe_allow_html=True)


# ── Data loaders ──────────────────────────────────────────────────────────
@st.cache_data
def load_tickets() -> list[dict[str, Any]]:
    """Load sample tickets from tickets.json.

    Returns:
        List of ticket dicts.
    """
    path = _ROOT / "data" / "tickets.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


@st.cache_data
def load_faq_content() -> dict[str, str]:
    """Load all FAQ documents into a dict keyed by category name.

    Returns:
        Dict mapping category name to file text content.
    """
    faq_dir = _ROOT / "data" / "faq_docs"
    result: dict[str, str] = {}
    for txt in sorted(faq_dir.glob("*.txt")):
        result[txt.stem] = txt.read_text(encoding="utf-8")
    return result


# ── Agent loader (cached across reruns) ───────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_agent() -> Any:
    """Load and cache the LangChain agent executor.

    Returns:
        The process_ticket function, or an Exception if loading failed.
    """
    try:
        from agent.support_agent import process_ticket
        return process_ticket
    except Exception as exc:
        return exc


@st.cache_resource(show_spinner=False)
def load_vectorstore() -> Any:
    """Build/load the ChromaDB vectorstore.

    Returns:
        The Chroma vectorstore, or an Exception.
    """
    try:
        from agent.rag_pipeline import build_vectorstore
        return build_vectorstore()
    except Exception as exc:
        return exc


# ── Session state ─────────────────────────────────────────────────────────
def _init_state() -> None:
    """Initialise all required session state keys."""
    for k, v in {
        "history": [],
        "eval_results": None,
        "prefill_text": "",
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Helpers ───────────────────────────────────────────────────────────────
def _badge(label: str, cls: str) -> str:
    return f'<span class="badge badge-{cls}">{label}</span>'


def _vs_status(vs: Any) -> tuple[str, str]:
    """Return (icon, label) for vectorstore status."""
    if isinstance(vs, Exception):
        return "🔴", f"Error: {vs}"
    try:
        count = vs._collection.count()
        return "🟢", f"Ready — {count} chunks"
    except Exception:
        return "🟡", "Unknown"


# ── Sidebar ───────────────────────────────────────────────────────────────
tickets = load_tickets()
vs = load_vectorstore()
vs_icon, vs_label = _vs_status(vs)

with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:.8rem 0">'
        '<span style="font-size:2.2rem">🎫</span>'
        '<p style="color:#a78bfa;font-weight:700;font-size:1rem;margin:.2rem 0 0">Support Agent</p>'
        '<p style="color:rgba(255,255,255,.35);font-size:.7rem;margin:0">Zangoh Intern Prep</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Sample Tickets dropdown ──
    st.markdown("### 📋 Sample Tickets")
    options = ["— pick a ticket —"] + [
        f"{t['id']} | {t['category'].upper()} | {t['subject'][:40]}"
        for t in tickets
    ]
    sel = st.selectbox("Load sample", options, key="ticket_sel", label_visibility="collapsed")
    if sel != "— pick a ticket —":
        idx = options.index(sel) - 1
        t = tickets[idx]
        st.session_state.prefill_text = f"Subject: {t['subject']}\n\n{t['body']}"

    st.divider()

    # ── Settings ──
    st.markdown("### 🔧 Settings")
    st.markdown(
        f'<div class="glass" style="font-size:.8rem;color:rgba(255,255,255,.7);">'
        f'<b>Model:</b> gpt-4o-mini<br>'
        f'<b>Embeddings:</b> all-MiniLM-L6-v2<br>'
        f'<b>Vector Store:</b> {vs_icon} {vs_label}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── History ──
    st.markdown("### 🕑 Recent")
    if st.session_state.history:
        for h in reversed(st.session_state.history[-5:]):
            action_cls = "resolved" if h["action"] == "resolved" else "escalated"
            st.markdown(
                f'<div class="glass" style="padding:.6rem .9rem;margin-bottom:.4rem;font-size:.78rem;">'
                f'{_badge(h["action"], action_cls)} &nbsp;'
                f'<span style="color:rgba(255,255,255,.6)">{h["subject"][:36]}…</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No tickets processed yet.")


# ── Hero ──────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero">'
    '<h1>🎫 Customer Support Resolution Agent</h1>'
    '<p>LangChain ReAct · ChromaDB RAG · GPT-4o-mini · sentence-transformers</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎫 Resolve Ticket", "📊 Evaluation Dashboard", "📚 Knowledge Base"])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — Resolve Ticket
# ═══════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        ticket_text = st.text_area(
            "Ticket Text",
            value=st.session_state.prefill_text,
            height=220,
            placeholder="Paste or type the support ticket here (subject + body)…",
            key="ticket_input",
        )
        process_btn = st.button("🚀 Process Ticket", type="primary", key="process_btn")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        if process_btn:
            if not ticket_text.strip():
                st.error("Please enter a ticket first.")
            else:
                process_ticket_fn = load_agent()
                if isinstance(process_ticket_fn, Exception):
                    st.error(f"Agent failed to load: {process_ticket_fn}")
                else:
                    with st.spinner("Agent is reasoning… (20–40 seconds)"):
                        result: dict[str, Any] = process_ticket_fn(ticket_text)

                    action = result.get("action", "escalated")
                    category = result.get("category", "general")
                    confidence = result.get("confidence", "low")
                    response = result.get("agent_response", "")
                    reasoning = result.get("reasoning", "")
                    intermediate = result.get("intermediate_steps", [])

                    # Save to history
                    subject_line = ticket_text.split("\n")[0][:50]
                    st.session_state.history.append({
                        "subject": subject_line,
                        "action": action,
                        "category": category,
                        "response": response,
                    })

                    # Meta badges row
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(
                            f'<div class="mcard">{_badge(action.upper(), action)}'
                            f'<div class="lbl">Action</div></div>',
                            unsafe_allow_html=True,
                        )
                    with m2:
                        st.markdown(
                            f'<div class="mcard">{_badge(category, category)}'
                            f'<div class="lbl">Category</div></div>',
                            unsafe_allow_html=True,
                        )
                    with m3:
                        st.markdown(
                            f'<div class="mcard">{_badge(confidence, confidence)}'
                            f'<div class="lbl">Confidence</div></div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("**Agent Response**")
                    st.markdown(
                        f'<div class="resp-box">{response}</div>',
                        unsafe_allow_html=True,
                    )

                    if reasoning:
                        with st.expander("🔍 Reasoning Log"):
                            st.info(reasoning)

        else:
            st.markdown(
                '<div class="glass" style="text-align:center;padding:3.5rem 1.5rem;">'
                '<div style="font-size:2.5rem;margin-bottom:.8rem">📬</div>'
                '<p style="color:rgba(255,255,255,.35);font-size:.88rem;margin:0">'
                'Select a sample ticket from the sidebar or type one, then click '
                '<strong style="color:#a78bfa">Process Ticket</strong>.</p>'
                '</div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════
# TAB 2 — Evaluation Dashboard
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Run the 15-case evaluation suite against the live agent.")

    run_btn = st.button("▶ Run Evaluation (15 test cases)", type="primary", key="run_eval_btn")

    if run_btn:
        try:
            from eval.run_eval import run_evaluation  # type: ignore

            progress_bar = st.progress(0, text="Starting evaluation…")
            status_ph = st.empty()

            with st.spinner("Running all 15 test cases — this takes a few minutes…"):
                # Patch progress: run_evaluation is synchronous; we just run it
                summary = run_evaluation(verbose=False)

            progress_bar.progress(1.0, text="Complete!")
            st.session_state.eval_results = summary

        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")

    if st.session_state.eval_results:
        s = st.session_state.eval_results
        results = s.get("results", [])

        # ── Metrics row ──
        m1, m2, m3, m4 = st.columns(4)
        metrics = [
            ("Accuracy", f"{s['accuracy_percent']}%"),
            ("Avg Score", f"{s['average_score']}/5"),
            ("Best Score", f"{s['best_score']}/5"),
            ("Worst Score", f"{s['worst_score']}/5"),
        ]
        for col, (lbl, val) in zip([m1, m2, m3, m4], metrics):
            with col:
                st.markdown(
                    f'<div class="mcard"><div class="val">{val}</div>'
                    f'<div class="lbl">{lbl}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Results table ──
        rows = []
        for r in results:
            match = r["action_match"]
            rows.append({
                "ID": r["id"],
                "Category": r["expected_category"],
                "Expected": r["expected_action"],
                "Predicted": r["predicted_action"],
                "✓ Match": "✅ Yes" if match else "❌ No",
                "Score": r["eval"]["overall_score"],
                "Feedback": r["eval"]["feedback"][:70] + "…" if len(r["eval"]["feedback"]) > 70 else r["eval"]["feedback"],
            })

        df = pd.DataFrame(rows)

        def _color_match(val: str) -> str:
            return "color: #86efac" if "Yes" in val else "color: #fca5a5"

        styled = df.style.applymap(_color_match, subset=["✓ Match"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # ── Bar chart: scores by category ──
        cat_data = s.get("average_score_by_category", {})
        if cat_data:
            chart_df = pd.DataFrame(
                {"Category": list(cat_data.keys()), "Avg Score": list(cat_data.values())}
            )
            fig = px.bar(
                chart_df,
                x="Category",
                y="Avg Score",
                color="Category",
                color_discrete_map={
                    "billing": "#60a5fa",
                    "technical": "#a78bfa",
                    "general": "#34d399",
                },
                title="Average Score by Category",
                range_y=[0, 5],
                template="plotly_dark",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.03)",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.markdown(
            '<div class="glass" style="text-align:center;padding:2.5rem">'
            '<p style="color:rgba(255,255,255,.35);margin:0">Click <b>Run Evaluation</b> to start.</p>'
            '</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════
# TAB 3 — Knowledge Base
# ═══════════════════════════════════════════════════════════════
with tab3:
    faq_content = load_faq_content()

    col_kb, col_ret = st.columns([1, 1], gap="large")

    with col_kb:
        st.markdown("#### 📄 FAQ Documents")
        cat_icons = {"billing": "💳", "technical": "🔧", "general": "ℹ️"}
        for cat, content in faq_content.items():
            icon = cat_icons.get(cat, "📄")
            qa_count = content.count("\nQ:")
            with st.expander(f"{icon} {cat.capitalize()} FAQ  ({qa_count} Q&A pairs)"):
                st.text(content)

    with col_ret:
        st.markdown("#### 🔍 Test Retrieval")
        ret_query = st.text_input(
            "Query",
            placeholder="e.g. how do I reset my password",
            key="ret_query",
        )
        ret_btn = st.button("Search Knowledge Base", key="ret_btn")

        if ret_btn and ret_query.strip():
            if isinstance(vs, Exception):
                st.error(f"Vectorstore unavailable: {vs}")
            else:
                try:
                    from agent.rag_pipeline import retrieve_context
                    with st.spinner("Searching…"):
                        hits = retrieve_context(ret_query, k=3, vectorstore=vs)
                    if not hits:
                        st.warning("No results above similarity threshold (0.4). This query would trigger escalation.")
                    else:
                        for i, h in enumerate(hits, 1):
                            score_color = "#86efac" if h["score"] >= 0.6 else "#fde047" if h["score"] >= 0.4 else "#fca5a5"
                            st.markdown(
                                f'<div class="glass">'
                                f'<div style="display:flex;justify-content:space-between;margin-bottom:.5rem">'
                                f'<b style="color:#a78bfa">Result {i}</b>'
                                f'<span style="color:{score_color};font-size:.8rem;font-weight:600">'
                                f'Score: {h["score"]:.3f} · {h["source_file"]}</span>'
                                f'</div>'
                                f'<div style="font-size:.83rem;color:rgba(255,255,255,.75);white-space:pre-wrap">'
                                f'{h["content"][:400]}{"…" if len(h["content"]) > 400 else ""}'
                                f'</div></div>',
                                unsafe_allow_html=True,
                            )
                except Exception as exc:
                    st.error(f"Retrieval error: {exc}")
        elif ret_btn:
            st.warning("Please enter a query first.")


# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:rgba(255,255,255,.25);font-size:.78rem;margin:0">'
    'Built for Zangoh SWE Intern preparation | RAG + LangChain + ChromaDB + GPT-4o-mini'
    '</p>',
    unsafe_allow_html=True,
)
