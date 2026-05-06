"""
Escalate Tool — marks a ticket as escalated, logs the reason and timestamp,
and returns a confirmation. No LLM call needed.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain.tools import Tool

logger = logging.getLogger(__name__)

# Path where escalation logs are persisted
ESCALATION_LOG_PATH: Path = Path(__file__).parent.parent / "escalation_log.json"

# In-memory store for the current session
_escalation_log: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Core escalation function
# ---------------------------------------------------------------------------

def escalate_ticket(input_text: str) -> str:
    """Escalate a support ticket to a human agent with reason logging.

    Parses the input for ticket ID and reason, records the escalation
    with a timestamp, appends it to the session log and persists it to
    a JSON file.

    Args:
        input_text: A description of why the ticket is being escalated.
            Format: "Ticket ID: <id> | Reason: <reason>" or free-form text.

    Returns:
        A confirmation string with escalation ID and timestamp.
    """
    timestamp: str = datetime.now(timezone.utc).isoformat()

    # Parse structured input if provided
    ticket_id: str = "UNKNOWN"
    reason: str = input_text.strip()

    if "ticket id:" in input_text.lower():
        try:
            parts = input_text.split("|")
            for part in parts:
                if "ticket id" in part.lower():
                    ticket_id = part.split(":", 1)[1].strip()
                elif "reason" in part.lower():
                    reason = part.split(":", 1)[1].strip()
        except Exception:
            reason = input_text.strip()

    escalation_id: str = f"ESC-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    record: dict[str, Any] = {
        "escalation_id": escalation_id,
        "ticket_id": ticket_id,
        "reason": reason,
        "timestamp": timestamp,
        "status": "pending_human_review",
    }

    _escalation_log.append(record)
    _persist_escalation(record)

    logger.warning(
        "ESCALATION: %s | Ticket: %s | Reason: %s",
        escalation_id,
        ticket_id,
        reason[:120],
    )

    return (
        f"✅ Ticket escalated successfully.\n"
        f"Escalation ID: {escalation_id}\n"
        f"Ticket ID: {ticket_id}\n"
        f"Reason: {reason}\n"
        f"Timestamp: {timestamp}\n"
        f"Status: Pending human agent review. A team member will respond within 4 hours."
    )


def _persist_escalation(record: dict[str, Any]) -> None:
    """Append an escalation record to the JSON log file.

    Args:
        record: The escalation record dict to persist.
    """
    try:
        existing: list[dict[str, Any]] = []
        if ESCALATION_LOG_PATH.exists():
            with ESCALATION_LOG_PATH.open(encoding="utf-8") as fh:
                existing = json.load(fh)
        existing.append(record)
        with ESCALATION_LOG_PATH.open("w", encoding="utf-8") as fh:
            json.dump(existing, fh, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.error("Failed to persist escalation log: %s", exc)


def get_escalation_log() -> list[dict[str, Any]]:
    """Return all escalations recorded in the current session.

    Returns:
        List of escalation record dicts.
    """
    return list(_escalation_log)


# ---------------------------------------------------------------------------
# LangChain Tool
# ---------------------------------------------------------------------------

escalate_tool = Tool(
    name="escalate_ticket",
    func=escalate_ticket,
    description=(
        "Escalate a support ticket to a human agent when the FAQ does not contain "
        "a relevant answer or the issue is too complex to resolve automatically. "
        "Input should explain why the ticket is being escalated. "
        "Format: 'Ticket ID: <id> | Reason: <detailed reason for escalation>'. "
        "Returns an escalation confirmation with a unique escalation ID."
    ),
)
