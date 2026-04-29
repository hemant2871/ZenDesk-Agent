"""
Ticket management tools — classify priority, set status, and generate
structured resolution summaries as LangChain StructuredTools.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Priority classifier
# ---------------------------------------------------------------------------

class ClassifyPriorityInput(BaseModel):
    """Input schema for the ticket priority classifier."""

    subject: str = Field(description="The subject line of the support ticket.")
    body: str = Field(description="The full body text of the support ticket.")


def classify_ticket_priority(subject: str, body: str) -> str:
    """Classify the priority of a support ticket using keyword heuristics.

    Applies a simple rule-based heuristic that checks for urgency signals in
    the ticket subject and body before deferring to a default medium priority.

    Args:
        subject: The subject line of the support ticket.
        body: The full body text of the support ticket.

    Returns:
        A string indicating priority: 'critical', 'high', 'medium', or 'low'.
    """
    combined: str = (subject + " " + body).lower()

    critical_keywords = {
        "down", "outage", "breach", "security", "hacked", "data loss",
        "production down", "all users", "everyone affected",
    }
    high_keywords = {
        "can't log in", "cannot log in", "charged twice", "double charge",
        "not working", "broken", "urgent", "asap", "immediately", "api error",
        "payment failed", "refund", "billing issue",
    }
    low_keywords = {
        "question", "curious", "wondering", "feedback", "suggestion",
        "feature request", "how do i", "tutorial", "documentation",
    }

    if any(kw in combined for kw in critical_keywords):
        priority = "critical"
    elif any(kw in combined for kw in high_keywords):
        priority = "high"
    elif any(kw in combined for kw in low_keywords):
        priority = "low"
    else:
        priority = "medium"

    return (
        f"Ticket priority classified as: **{priority.upper()}**\n"
        f"Reasoning: Based on keywords detected in the ticket content."
    )


def build_classify_priority_tool() -> StructuredTool:
    """Build the priority classification LangChain StructuredTool.

    Returns:
        StructuredTool: A ready-to-use LangChain tool for priority classification.
    """
    return StructuredTool.from_function(
        func=classify_ticket_priority,
        name="classify_ticket_priority",
        description=(
            "Classify the priority of a support ticket as critical, high, medium, or low. "
            "Use this tool to determine urgency before crafting the response. "
            "Provide the ticket subject and body as inputs."
        ),
        args_schema=ClassifyPriorityInput,
        return_direct=False,
    )


# ---------------------------------------------------------------------------
# Resolution drafter
# ---------------------------------------------------------------------------

class DraftResolutionInput(BaseModel):
    """Input schema for the resolution draft tool."""

    ticket_id: str = Field(description="The unique ID of the support ticket.")
    customer_issue: str = Field(description="A one-sentence summary of the customer's issue.")
    resolution_steps: str = Field(
        description="The step-by-step resolution or answer to provide to the customer."
    )
    priority: Literal["critical", "high", "medium", "low"] = Field(
        description="The classified priority of the ticket."
    )
    kb_articles_used: str = Field(
        description="Comma-separated list of KB article titles that were referenced.",
        default="None",
    )


def draft_resolution(
    ticket_id: str,
    customer_issue: str,
    resolution_steps: str,
    priority: str,
    kb_articles_used: str = "None",
) -> str:
    """Format a structured resolution response for a support ticket.

    Generates a professional, templated response ready to be sent to the
    customer, including metadata for internal tracking.

    Args:
        ticket_id: The unique identifier of the support ticket.
        customer_issue: A brief summary of the customer's problem.
        resolution_steps: The detailed steps or answer to resolve the issue.
        priority: The ticket priority level (critical/high/medium/low).
        kb_articles_used: KB articles referenced during resolution.

    Returns:
        A formatted string containing the customer-facing response and
        internal metadata block.
    """
    priority_emoji: dict[str, str] = {
        "critical": "🔴",
        "high": "🟠",
        "medium": "🟡",
        "low": "🟢",
    }
    emoji = priority_emoji.get(priority.lower(), "⚪")

    response = f"""
═══════════════════════════════════════════
TICKET RESOLUTION — {ticket_id}
Priority: {emoji} {priority.upper()}
═══════════════════════════════════════════

📋 ISSUE SUMMARY
{customer_issue}

✅ RESOLUTION
{resolution_steps}

📚 KNOWLEDGE BASE REFERENCES
{kb_articles_used}

─────────────────────────────────────────
This response was generated by the Zangoh
Support AI. A human agent will follow up
if further assistance is needed.
═══════════════════════════════════════════
""".strip()

    return response


def build_draft_resolution_tool() -> StructuredTool:
    """Build the resolution drafting LangChain StructuredTool.

    Returns:
        StructuredTool: A ready-to-use LangChain tool for drafting resolutions.
    """
    return StructuredTool.from_function(
        func=draft_resolution,
        name="draft_resolution",
        description=(
            "Draft a structured, professional resolution response for a support ticket. "
            "Use this as the FINAL step after you have searched the knowledge base and "
            "classified the priority. Provide all required fields to generate a formatted response."
        ),
        args_schema=DraftResolutionInput,
        return_direct=False,
    )
