"""
LLM-as-Judge Evaluator — scores agent responses on relevance, tone,
and correctness using Llama-3.1-8b-instant as the judge.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are an expert customer support quality evaluator. Score the following \
AI agent response to a support ticket.

TICKET:
{ticket}

AGENT RESPONSE:
{agent_response}

EXPECTED ACTION: {expected_action}

Score the response on these three dimensions (integer 1-5 each):

1. RELEVANCE (1-5): Does the response directly address the customer's actual issue? \
   5 = perfectly targeted, 1 = completely off-topic.

2. TONE (1-5): Is the response professional, empathetic, and customer-friendly? \
   5 = excellent tone, 1 = rude, cold, or robotic.

3. CORRECTNESS (1-5): Was the action (resolved vs escalated) appropriate given the ticket? \
   5 = perfect decision, 1 = completely wrong decision. \
   Expected action was: {expected_action}

Respond with ONLY this JSON (no markdown, no extra text):
{{"relevance": <1-5>, "tone": <1-5>, "correctness": <1-5>, \
"feedback": "<one sentence of constructive feedback>"}}
"""


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def _build_judge_llm() -> ChatGroq:
    """Build the LLM used as the evaluation judge.

    Returns:
        ChatGroq: A llama-3.1-8b-instant instance configured for structured output.

    Raises:
        ValueError: If GROQ_API_KEY is not set.
    """
    api_key: str | None = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set — cannot run evaluator.")
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,  # deterministic for evaluation
        api_key=api_key,
    )


def evaluate_response(
    ticket: str,
    agent_response: str,
    expected_action: str,
) -> dict[str, Any]:
    """Score an agent response using Llama-3.1-8b-instant as judge.

    Evaluates on three dimensions: relevance (does it address the ticket?),
    tone (professional and empathetic?), and correctness (right action taken?).

    Args:
        ticket: The original support ticket text.
        agent_response: The agent's complete response text.
        expected_action: Expected action string: 'resolved' or 'escalated'.

    Returns:
        A dict with keys:
            - relevance (int): Score 1-5.
            - tone (int): Score 1-5.
            - correctness (int): Score 1-5.
            - overall_score (float): Average of the three scores.
            - feedback (str): One-sentence constructive feedback.
            - error (str | None): Error message if evaluation failed.
    """
    logger.info(
        "Evaluating response (expected: %s, response length: %d)",
        expected_action,
        len(agent_response),
    )

    try:
        llm = _build_judge_llm()
        prompt = _JUDGE_PROMPT.format(
            ticket=ticket[:1000],
            agent_response=agent_response[:2000],
            expected_action=expected_action,
        )

        response = llm.invoke(prompt)
        raw_text: str = response.content.strip()

        # Strip markdown fences if present
        raw_text = re.sub(r"```json|```", "", raw_text).strip()

        scores: dict[str, Any] = json.loads(raw_text)
        relevance: int = int(scores.get("relevance", 1))
        tone: int = int(scores.get("tone", 1))
        correctness: int = int(scores.get("correctness", 1))
        feedback: str = scores.get("feedback", "No feedback provided.")

        overall: float = round((relevance + tone + correctness) / 3, 2)

        logger.info(
            "Evaluation scores — relevance: %d, tone: %d, correctness: %d, overall: %.2f",
            relevance,
            tone,
            correctness,
            overall,
        )

        return {
            "relevance": relevance,
            "tone": tone,
            "correctness": correctness,
            "overall_score": overall,
            "feedback": feedback,
            "error": None,
        }

    except json.JSONDecodeError as exc:
        logger.error("Failed to parse judge JSON: %s | raw: %s", exc, raw_text[:200])
        return {
            "relevance": 0,
            "tone": 0,
            "correctness": 0,
            "overall_score": 0.0,
            "feedback": f"Evaluation parse error: {exc}",
            "error": str(exc),
        }
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc, exc_info=True)
        return {
            "relevance": 0,
            "tone": 0,
            "correctness": 0,
            "overall_score": 0.0,
            "feedback": f"Evaluation error: {exc}",
            "error": str(exc),
        }
