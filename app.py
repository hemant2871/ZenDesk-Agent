"""
Streamlit UI for the Zangoh Customer Support Ticket Resolution Agent.
Pure UI layer — all business logic lives in agent/ and tools/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Zangoh Support Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Inline CSS — dark glassmorphism theme
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root & body ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #13131f 50%, #0d0d18 100%);
    min-height: 100vh;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem; max-width: 1280px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.03);
    border-right: 1px solid rgba(255,255,255,0.07);
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    color: #a78bfa;
    font-size: 0.95rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Glass card ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1rem;
}

/* ── Header banner ── */
.hero-banner {
    background: linear-gradient(135deg, rgba(124,58,237,0.25) 0%, rgba(59,130,246,0.15) 100%);
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(124,58,237,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.hero-subtitle {
    color: rgba(255,255,255,0.55);
    font-size: 0.95rem;
    margin-top: 0.4rem;
}

/* ── Labels & text ── */
.stTextInput label, .stTextArea label, .stSelectbox label {
    color: rgba(255,255,255,0.7) !important;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.02em;
}

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: rgba(124,58,237,0.6) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.15) !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7c3aed, #3b82f6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.4) !important;
}

/* ── Secondary button ── */
.stButton > button:not([kind="primary"]) {
    background: rgba(255,255,255,0.05) !important;
    color: rgba(255,255,255,0.7) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    width: 100% !important;
}

/* ── Priority badges ── */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-critical { background: rgba(239,68,68,0.2); color: #fca5a5; border: 1px solid rgba(239,68,68,0.3); }
.badge-high     { background: rgba(249,115,22,0.2); color: #fdba74; border: 1px solid rgba(249,115,22,0.3); }
.badge-medium   { background: rgba(234,179,8,0.2);  color: #fde047; border: 1px solid rgba(234,179,8,0.3);  }
.badge-low      { background: rgba(34,197,94,0.2);  color: #86efac; border: 1px solid rgba(34,197,94,0.3);  }

/* ── Step tracker ── */
.step-row {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.step-dot {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 2px;
}
.step-dot-done { background: rgba(34,197,94,0.2); color: #86efac; border: 1px solid rgba(34,197,94,0.4); }
.step-dot-run  { background: rgba(124,58,237,0.3); color: #c4b5fd; border: 1px solid rgba(124,58,237,0.5); }
.step-dot-idle { background: rgba(255,255,255,0.05); color: rgba(255,255,255,0.3); border: 1px solid rgba(255,255,255,0.1); }
.step-label { font-size: 0.85rem; color: rgba(255,255,255,0.7); line-height: 1.4; }
.step-label strong { color: rgba(255,255,255,0.9); }

/* ── Resolution output ── */
.resolution-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.5rem;
    font-family: 'Inter', monospace;
    font-size: 0.88rem;
    color: #e2e8f0;
    white-space: pre-wrap;
    line-height: 1.7;
}

/* ── History list ── */
.history-item {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    cursor: pointer;
    transition: all 0.15s ease;
}
.history-item:hover {
    background: rgba(124,58,237,0.1);
    border-color: rgba(124,58,237,0.3);
}
.history-id { font-size: 0.75rem; color: #a78bfa; font-weight: 600; }
.history-subject { font-size: 0.85rem; color: rgba(255,255,255,0.8); margin-top: 0.1rem; }

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.metric-card {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.metric-value { font-size: 1.8rem; font-weight: 700; color: #a78bfa; }
.metric-label { font-size: 0.75rem; color: rgba(255,255,255,0.45); margin-top: 0.2rem; }

/* ── Expander ── */
details summary { color: rgba(255,255,255,0.6) !important; }
.streamlit-expanderHeader { color: rgba(255,255,255,0.6) !important; }

/* ── Selectbox ── */
.stSelectbox div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.07) !important; margin: 1.5rem 0 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #7c3aed !important; }

/* ── Toast / success ── */
.stSuccess { background: rgba(34,197,94,0.1) !important; border: 1px solid rgba(34,197,94,0.2) !important; color: #86efac !important; border-radius: 10px !important; }
.stError   { background: rgba(239,68,68,0.1)  !important; border: 1px solid rgba(239,68,68,0.2)  !important; color: #fca5a5 !important; border-radius: 10px !important; }
.stWarning { background: rgba(249,115,22,0.1) !important; border: 1px solid rgba(249,115,22,0.2) !important; color: #fdba74 !important; border-radius: 10px !important; }
.stInfo    { background: rgba(59,130,246,0.1) !important; border: 1px solid rgba(59,130,246,0.2)  !important; color: #93c5fd !important; border-radius: 10px !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    """Initialise all required Streamlit session-state keys."""
    defaults: dict[str, Any] = {
        "executor": None,
        "history": [],
        "agent_ready": False,
        "last_steps": [],
        "init_error": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


_init_state()


# ---------------------------------------------------------------------------
# Agent loader (cached across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_agent(api_key: str) -> Any:
    """Load and cache the AgentExecutor (runs once per session).

    Args:
        api_key: The Groq API key string (used as cache key).

    Returns:
        AgentExecutor instance, or Exception if loading failed.
    """
    import os
    os.environ["GROQ_API_KEY"] = api_key
    try:
        from agent.agent import build_agent_executor
        return build_agent_executor()
    except Exception as exc:
        return exc


def _try_auto_init(api_key: str) -> None:
    """Attempt to auto-initialize the agent with the given API key.

    Sets session state agent_ready / init_error based on outcome.

    Args:
        api_key: Groq API key to use.
    """
    if st.session_state.agent_ready:
        return
    executor = load_agent(api_key)
    if isinstance(executor, Exception):
        st.session_state.init_error = str(executor)
    else:
        st.session_state.executor = executor
        st.session_state.agent_ready = True
        st.session_state.init_error = None


# ---------------------------------------------------------------------------
# Sample tickets for quick-fill
# ---------------------------------------------------------------------------

SAMPLE_TICKETS: dict[str, dict[str, str]] = {
    "— Select a sample ticket —": {"id": "", "subject": "", "body": ""},
    "🔴 Billing — Double Charge": {
        "id": "TICK-002",
        "subject": "Charged twice this month",
        "body": (
            "I noticed two charges from your company on my credit card statement this month, "
            "both for $49.99. My subscription is monthly and I should only be billed once. "
            "Please refund the duplicate charge immediately."
        ),
    },
    "🟠 Technical — Login Failure": {
        "id": "TICK-001",
        "subject": "Can't log into my account",
        "body": (
            "Hi, I've been trying to log in for the past hour and keep getting an "
            "'invalid credentials' error even though I'm sure my password is correct. "
            "I've tried resetting it but didn't receive the email. Please help!"
        ),
    },
    "🟡 Account — Adding Team Members": {
        "id": "TICK-003",
        "subject": "How do I add my team members?",
        "body": (
            "We just upgraded to Pro and I want to add 5 new team members. "
            "I went to settings but I'm not sure where to find the option. "
            "Can you walk me through it?"
        ),
    },
    "🟠 Technical — API 429 Errors": {
        "id": "TICK-004",
        "subject": "API returning 429 errors",
        "body": (
            "Our integration started getting a lot of 429 errors from your API as of yesterday. "
            "We are on the Pro plan and our usage hasn't changed. Is there an incident?"
        ),
    },
    "🟡 Subscription — Cancel Account": {
        "id": "TICK-005",
        "subject": "Want to cancel subscription",
        "body": (
            "I'd like to cancel my subscription. "
            "Can you tell me if I'll be charged again and what happens to my data?"
        ),
    },
}


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _priority_badge(priority: str) -> str:
    """Return an HTML priority badge for a given priority string.

    Args:
        priority: One of 'critical', 'high', 'medium', 'low'.

    Returns:
        HTML string for the badge.
    """
    p = priority.lower() if priority else "medium"
    icons = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
    icon = icons.get(p, "⚪")
    return f'<span class="badge badge-{p}">{icon} {p.upper()}</span>'


def _extract_priority(text: str) -> str:
    """Extract the priority keyword from agent output.

    Args:
        text: The full agent output text.

    Returns:
        The detected priority string, or 'medium' as fallback.
    """
    for p in ["critical", "high", "medium", "low"]:
        if p in text.lower():
            return p
    return "medium"


def _render_step_tracker(steps: list[tuple[Any, Any]]) -> None:
    """Render a visual tracker of the agent's reasoning steps.

    Args:
        steps: The intermediate_steps list from AgentExecutor output.
    """
    tool_order = [
        ("analyse_sentiment", "Sentiment Analysis"),
        ("classify_ticket_priority", "Priority Classification"),
        ("search_knowledge_base", "Knowledge Base Search"),
        ("draft_resolution", "Draft Resolution"),
    ]
    tools_used: set[str] = {step[0].tool for step in steps if hasattr(step[0], "tool")}

    rows_html = ""
    for tool_name, label in tool_order:
        if tool_name in tools_used:
            dot_class = "step-dot-done"
            status = "✓"
        else:
            dot_class = "step-dot-idle"
            status = str(tool_order.index((tool_name, label)) + 1)

        rows_html += f"""
        <div class="step-row">
            <div class="step-dot {dot_class}">{status}</div>
            <div class="step-label"><strong>{label}</strong></div>
        </div>
        """

    st.markdown(
        f'<div class="glass-card">{rows_html}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

# ── Auto-init: read key from Streamlit secrets (cloud) or .env (local) ────
import os as _os
from dotenv import load_dotenv as _load_dotenv
_load_dotenv()
# Prefer st.secrets (Streamlit Cloud), fall back to .env
try:
    _env_key: str = st.secrets.get("GROQ_API_KEY", "") or _os.environ.get("GROQ_API_KEY", "")
except Exception:
    _env_key: str = _os.environ.get("GROQ_API_KEY", "")

with st.sidebar:
    st.markdown(
        '<div style="text-align:center; padding: 1rem 0;">'
        '<span style="font-size:2.5rem;">🎯</span>'
        '<p style="color:#a78bfa; font-weight:700; font-size:1.1rem; margin:0.3rem 0 0;">Zangoh</p>'
        '<p style="color:rgba(255,255,255,0.4); font-size:0.75rem; margin:0;">Support Agent v1.0</p>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("## ⚙️ API Key")

    # Show masked key if already loaded from .env
    key_from_env = bool(_env_key and not _env_key.startswith("your-"))
    if key_from_env:
        st.success("Key loaded from .env", icon="🔑")
        active_key = _env_key
    else:
        active_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...  (or add to .env file)",
            help="Add GROQ_API_KEY=gsk_... to your .env file to skip this step.",
        )

    # ── Auto-initialize as soon as a valid key is available ──
    if active_key and not st.session_state.agent_ready:
        with st.spinner("🔄 Initialising agent & knowledge base…"):
            _try_auto_init(active_key)
        if st.session_state.agent_ready:
            st.rerun()

    st.divider()
    st.markdown("## 🤖 Agent Status")
    if st.session_state.agent_ready:
        st.success("Agent ready", icon="✅")
        st.caption("Model: llama-3.1-8b-instant | KB: ChromaDB")
    elif st.session_state.init_error:
        st.error(f"Init failed: {st.session_state.init_error[:120]}", icon="❌")
        if st.button("Retry", type="primary"):
            st.cache_resource.clear()
            st.session_state.agent_ready = False
            st.session_state.init_error = None
            st.rerun()
    else:
        st.warning("Enter API key above to start", icon="⚠️")

    st.divider()

    # History panel
    st.markdown("## 📋 Ticket History")
    if st.session_state.history:
        for entry in reversed(st.session_state.history[-10:]):
            priority = _extract_priority(entry.get("output", ""))
            p_icons = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
            icon = p_icons.get(priority, "⚪")
            st.markdown(
                f'<div class="history-item">'
                f'<div class="history-id">{icon} {entry["ticket_id"]}</div>'
                f'<div class="history-subject">{entry["subject"][:48]}…</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No tickets resolved yet.")

    st.divider()
    total = len(st.session_state.history)
    st.markdown(
        f'<div class="metric-card" style="margin-bottom:0.5rem;">'
        f'<div class="metric-value">{total}</div>'
        f'<div class="metric-label">Tickets Resolved</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

# Hero banner
st.markdown(
    '<div class="hero-banner">'
    '<h1 class="hero-title">🎯 Customer Support Ticket Resolution</h1>'
    '<p class="hero-subtitle">Powered by Llama-3.1 · LangChain ReAct · ChromaDB · sentence-transformers</p>'
    "</div>",
    unsafe_allow_html=True,
)

# Two-column layout
col_form, col_output = st.columns([1, 1], gap="large")

# ── LEFT: Ticket input ──────────────────────────────────────────────────────
with col_form:
    st.markdown(
        '<div class="glass-card">'
        '<p style="color:#a78bfa; font-weight:600; font-size:0.8rem; letter-spacing:0.08em; text-transform:uppercase; margin:0 0 1rem;">New Ticket</p>',
        unsafe_allow_html=True,
    )

    # Sample ticket selector
    selected_sample = st.selectbox(
        "Quick-fill with sample ticket",
        options=list(SAMPLE_TICKETS.keys()),
        key="sample_selector",
    )
    sample = SAMPLE_TICKETS[selected_sample]

    # Form fields
    ticket_id = st.text_input(
        "Ticket ID",
        value=sample["id"],
        placeholder="TICK-001",
        key="ticket_id_input",
    )
    subject = st.text_input(
        "Subject",
        value=sample["subject"],
        placeholder="Describe the issue in one line…",
        key="subject_input",
    )
    body = st.text_area(
        "Customer Message",
        value=sample["body"],
        placeholder="Full message from the customer…",
        height=160,
        key="body_input",
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # Submit
    resolve_btn = st.button(
        "🚀 Resolve Ticket",
        type="primary",
        disabled=not st.session_state.agent_ready,
        key="resolve_btn",
    )

    if not st.session_state.agent_ready:
        st.info("Initialise the agent in the sidebar first.", icon="ℹ️")

# ── RIGHT: Output ───────────────────────────────────────────────────────────
with col_output:
    st.markdown(
        '<div style="padding: 0 0 0.75rem;">'
        '<p style="color:#a78bfa; font-weight:600; font-size:0.8rem; letter-spacing:0.08em; text-transform:uppercase; margin:0;">Resolution Output</p>'
        "</div>",
        unsafe_allow_html=True,
    )

    output_placeholder = st.empty()
    steps_placeholder = st.empty()

    # ── Run agent ──
    if resolve_btn:
        if not ticket_id or not subject or not body:
            st.error("Please fill in all fields before resolving.")
        else:
            with output_placeholder.container():
                with st.spinner("Agent is thinking… this may take 20–40 seconds."):
                    start = time.time()
                    result: dict[str, Any] = st.session_state.executor.invoke(
                        {
                            "input": (
                                f"Ticket ID: {ticket_id}\n"
                                f"Subject: {subject}\n\n"
                                f"Customer Message:\n{body}"
                            )
                        }
                    )
                    elapsed = round(time.time() - start, 2)

                agent_output: str = result.get("output", "No output generated.")
                steps: list[Any] = result.get("intermediate_steps", [])

                # Save to history
                st.session_state.history.append(
                    {
                        "ticket_id": ticket_id,
                        "subject": subject,
                        "output": agent_output,
                        "steps": steps,
                        "elapsed": elapsed,
                    }
                )
                st.session_state.last_steps = steps

            # Render output
            output_placeholder.empty()
            priority = _extract_priority(agent_output)

            with output_placeholder.container():
                # Meta row
                meta_cols = st.columns([1, 1, 1])
                with meta_cols[0]:
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-value" style="font-size:1.1rem;">{ticket_id}</div><div class="metric-label">Ticket ID</div></div>',
                        unsafe_allow_html=True,
                    )
                with meta_cols[1]:
                    st.markdown(
                        f'<div class="metric-card">{_priority_badge(priority)}<div class="metric-label" style="margin-top:0.4rem;">Priority</div></div>',
                        unsafe_allow_html=True,
                    )
                with meta_cols[2]:
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-value" style="font-size:1.1rem;">{elapsed}s</div><div class="metric-label">Response Time</div></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    f'<div class="resolution-box">{agent_output}</div>',
                    unsafe_allow_html=True,
                )

                # Reasoning steps expander
                if steps:
                    with st.expander("🔍 View Agent Reasoning Steps"):
                        _render_step_tracker(steps)
                        for i, (action, observation) in enumerate(steps, start=1):
                            tool_name = getattr(action, "tool", "unknown")
                            tool_input = getattr(action, "tool_input", {})
                            st.markdown(
                                f"**Step {i} — `{tool_name}`**",
                            )
                            st.json(tool_input if isinstance(tool_input, dict) else {"input": str(tool_input)})
                            st.caption(f"Observation: {str(observation)[:400]}…")
                            st.divider()

    else:
        # Idle state
        if st.session_state.history:
            # Show last result
            last = st.session_state.history[-1]
            priority = _extract_priority(last["output"])
            with output_placeholder.container():
                st.markdown(
                    f'<div class="resolution-box">{last["output"]}</div>',
                    unsafe_allow_html=True,
                )
        else:
            with output_placeholder.container():
                st.markdown(
                    '<div class="glass-card" style="text-align:center; padding:3rem 2rem;">'
                    '<div style="font-size:3rem; margin-bottom:1rem;">📬</div>'
                    '<p style="color:rgba(255,255,255,0.4); font-size:0.9rem; margin:0;">'
                    "Fill in a ticket on the left and click <strong>Resolve Ticket</strong> to begin."
                    "</p>"
                    "</div>",
                    unsafe_allow_html=True,
                )
