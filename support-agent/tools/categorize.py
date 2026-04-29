"""
Categorize Tool — keyword-based ticket categorization with confidence scoring.
No LLM call required, making it fast and deterministic.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from langchain.tools import Tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword dictionaries
# ---------------------------------------------------------------------------

Category = Literal["billing", "technical", "general"]

_BILLING_KEYWORDS: frozenset[str] = frozenset(
    {
        "charge", "charged", "refund", "invoice", "payment", "pay", "paid",
        "subscription", "cancel", "billing", "credit card", "card", "price",
        "pricing", "cost", "fee", "plan", "upgrade", "downgrade", "receipt",
        "coupon", "discount", "promo", "vat", "tax", "seat", "seats",
        "renewal", "renew", "expire", "expired", "trial", "debit",
        "transaction", "overcharged", "double charge", "refunded",
    }
)

_TECHNICAL_KEYWORDS: frozenset[str] = frozenset(
    {
        "login", "log in", "password", "reset", "crash", "crashes", "error",
        "bug", "broken", "install", "installation", "api", "401", "403",
        "404", "500", "rate limit", "slow", "loading", "performance",
        "sync", "notification", "email", "sms", "2fa", "two-factor",
        "authentication", "upload", "download", "export", "import",
        "app", "mobile", "desktop", "browser", "cache", "cookies",
        "locked", "account locked", "unauthorized", "token", "key",
        "dll", "dependency", "setup", "configure", "connect", "disconnect",
        "timeout", "connection", "websocket", "offline",
    }
)

_GENERAL_KEYWORDS: frozenset[str] = frozenset(
    {
        "how do i", "getting started", "new user", "onboarding", "tutorial",
        "hours", "working hours", "support hours", "contact", "phone",
        "feature request", "suggestion", "dark mode", "roadmap",
        "privacy", "gdpr", "data deletion", "delete account", "security",
        "partnership", "integration", "transfer", "ownership", "team",
        "general", "question", "information", "help", "guide", "faq",
        "where", "when", "who", "what is", "how long",
    }
)


# ---------------------------------------------------------------------------
# Scoring function
# ---------------------------------------------------------------------------

def _count_keyword_hits(text_lower: str, keywords: frozenset[str]) -> int:
    """Count keyword matches in lowercased text using word-boundary matching.

    Args:
        text_lower: Lowercased input text.
        keywords: Set of keywords to match.

    Returns:
        Number of keyword hits found in the text.
    """
    hits = 0
    for kw in keywords:
        # Use word boundaries for single words, substring for phrases
        if " " in kw:
            if kw in text_lower:
                hits += 1
        else:
            if re.search(rf"\b{re.escape(kw)}\b", text_lower):
                hits += 1
    return hits


def _compute_confidence(score: int, second_score: int) -> str:
    """Compute a confidence level based on win margin.

    Args:
        score: The winning category's keyword hit count.
        second_score: The second-highest category's hit count.

    Returns:
        'high' if margin > 3, 'medium' if > 1, else 'low'.
    """
    margin = score - second_score
    if margin > 3:
        return "high"
    if margin > 1:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def categorize_ticket(ticket_text: str) -> str:
    """Categorize a support ticket as billing, technical, or general.

    Uses keyword matching to assign a category and computes a confidence
    level (high/medium/low) based on the margin over competing categories.

    Args:
        ticket_text: The combined subject and body of the support ticket.

    Returns:
        A formatted string: "Category: <category> | Confidence: <level>"
        with a brief explanation of the keyword matches found.
    """
    logger.info("Categorizing ticket (length: %d chars)", len(ticket_text))
    text_lower = ticket_text.lower()

    billing_score = _count_keyword_hits(text_lower, _BILLING_KEYWORDS)
    technical_score = _count_keyword_hits(text_lower, _TECHNICAL_KEYWORDS)
    general_score = _count_keyword_hits(text_lower, _GENERAL_KEYWORDS)

    scores: dict[str, int] = {
        "billing": billing_score,
        "technical": technical_score,
        "general": general_score,
    }

    sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    winner, top_score = sorted_cats[0]
    _, second_score = sorted_cats[1]

    confidence = _compute_confidence(top_score, second_score)

    logger.info(
        "Categorization result: %s (confidence=%s) | scores: billing=%d, technical=%d, general=%d",
        winner,
        confidence,
        billing_score,
        technical_score,
        general_score,
    )

    return (
        f"Category: {winner} | Confidence: {confidence}\n"
        f"Scores — billing: {billing_score}, technical: {technical_score}, "
        f"general: {general_score}"
    )


# ---------------------------------------------------------------------------
# LangChain Tool
# ---------------------------------------------------------------------------

categorize_tool = Tool(
    name="categorize_ticket",
    func=categorize_ticket,
    description=(
        "Categorize a support ticket as 'billing', 'technical', or 'general' "
        "using keyword analysis. Input should be the full ticket text (subject + body). "
        "Returns the category and a confidence level (high/medium/low)."
    ),
)
