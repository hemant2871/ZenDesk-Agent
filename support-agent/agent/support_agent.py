"""
Support Agent — LangChain ReAct agent that resolves or escalates customer
support tickets using FAQ retrieval, ticket categorization, and escalation tools.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_groq import ChatGroq

# Ensure project root on sys.path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.categorize import categorize_tool  # noqa: E402
from tools.escalate import escalate_tool  # noqa: E402
from tools.search_faq import search_faq_tool  # noqa: E402

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert customer support agent for a SaaS platform. Your job is to \
resolve customer support tickets efficiently, accurately, and with empathy.

RESOLUTION PROTOCOL — follow these steps in order:

1. **categorize_ticket**: Call this first with the full ticket text to classify \
   the issue as billing, technical, or general. Note the category and confidence.

2. **search_faq**: Search the knowledge base using the core issue as your query. \
   Be specific — use the customer's actual problem as the query.

3. **Decide based on search result**:
   - If search returns relevant content (no "NO_RELEVANT_ANSWER_FOUND"):
     Draft a professional, helpful reply that directly addresses the customer's \
     issue using the FAQ content. Be specific — reference exact steps, numbers, \
     and links from the FAQ.
   - If search returns "NO_RELEVANT_ANSWER_FOUND":
     Call **escalate_ticket** with the ticket ID and a clear reason for escalation.

4. **Always end your Final Answer** with this exact JSON block on a new line:
```json
{{"action": "resolved" or "escalated", "category": "billing/technical/general", \
"confidence": "high/medium/low", "reasoning": "one sentence explaining your decision"}}
```

TONE RULES:
- Always be professional, empathetic, and solution-focused
- Address the customer by referencing their specific issue details
- For billing issues, reassure about data security
- For technical issues, provide numbered step-by-step instructions
- For escalations, explain what happens next and expected wait time
"""

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def _build_llm() -> ChatGroq:
    """Build the ChatGroq LLM instance from environment configuration.

    Returns:
        ChatGroq: Configured LLM using llama-3.1-8b-instant.

    Raises:
        ValueError: If GROQ_API_KEY is not set.
    """
    api_key: str | None = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Create a .env file with: "
            "GROQ_API_KEY=gsk_..."
        )
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=api_key,
    )


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------

_agent_executor: Any = None


def _get_agent() -> Any:
    """Build and cache the LangChain ZERO_SHOT_REACT_DESCRIPTION agent.

    Returns:
        AgentExecutor: The initialized agent with all tools attached.
    """
    global _agent_executor
    if _agent_executor is not None:
        return _agent_executor

    llm = _build_llm()
    tools: list[Tool] = [
        categorize_tool,
        search_faq_tool,
        escalate_tool,
    ]

    logger.info("Initialising LangChain ZERO_SHOT_REACT_DESCRIPTION agent...")
    _agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=6,
        handle_parsing_errors=True,
        agent_kwargs={
            "prefix": SYSTEM_PROMPT,
        },
    )
    logger.info("Agent ready.")
    return _agent_executor


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json_block(text: str) -> dict[str, Any]:
    """Extract the trailing JSON metadata block from the agent's response.

    Looks for a ```json ... ``` fenced block or a raw JSON object at the
    end of the response text.

    Args:
        text: The agent's full final answer text.

    Returns:
        A dict with keys: action, category, confidence, reasoning.
        Falls back to defaults if no valid JSON block is found.
    """
    # Try fenced code block first
    pattern = r"```json\s*(\{.*?\})\s*```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try raw JSON object (last {...} in the text)
    raw_matches = list(re.finditer(r"\{[^{}]*\}", text, re.DOTALL))
    for m in reversed(raw_matches):
        try:
            data = json.loads(m.group())
            if "action" in data:
                return data
        except json.JSONDecodeError:
            continue

    # Fallback heuristics
    action = "escalated" if "escalat" in text.lower() else "resolved"
    category = "general"
    for cat in ["billing", "technical", "general"]:
        if cat in text.lower():
            category = cat
            break

    return {
        "action": action,
        "category": category,
        "confidence": "low",
        "reasoning": "Could not extract structured JSON from response.",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_ticket(ticket_text: str) -> dict[str, Any]:
    """Run the support agent on a single ticket and return structured output.

    Args:
        ticket_text: The full ticket text (subject + body, or pre-formatted string).

    Returns:
        A dict with the following keys:
            - agent_response (str): Full agent output text.
            - action (str): 'resolved' or 'escalated'.
            - category (str): 'billing', 'technical', or 'general'.
            - confidence (str): 'high', 'medium', or 'low'.
            - reasoning (str): Agent's reasoning for the decision.
            - timestamp (str): ISO 8601 UTC timestamp of processing.
            - error (str | None): Error message if processing failed.
    """
    logger.info("=" * 60)
    logger.info("Processing ticket: %s...", ticket_text[:80])
    logger.info("=" * 60)

    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        agent = _get_agent()
        result = agent.invoke({"input": ticket_text})
        agent_response: str = result.get("output", "")

        metadata = _extract_json_block(agent_response)

        logger.info(
            "Ticket processed — action: %s | category: %s | confidence: %s",
            metadata.get("action"),
            metadata.get("category"),
            metadata.get("confidence"),
        )

        return {
            "agent_response": agent_response,
            "action": metadata.get("action", "escalated"),
            "category": metadata.get("category", "general"),
            "confidence": metadata.get("confidence", "low"),
            "reasoning": metadata.get("reasoning", ""),
            "timestamp": timestamp,
            "error": None,
        }

    except Exception as exc:
        logger.error("Agent processing failed: %s", exc, exc_info=True)
        return {
            "agent_response": f"Processing failed: {exc}",
            "action": "escalated",
            "category": "general",
            "confidence": "low",
            "reasoning": f"Agent error: {exc}",
            "timestamp": timestamp,
            "error": str(exc),
        }
