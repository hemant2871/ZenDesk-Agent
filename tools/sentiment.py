"""
Sentiment analysis tool — detects customer sentiment from ticket text and
provides tone guidance for the agent's response as a LangChain StructuredTool.
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sentiment scoring (keyword-based, no external model dependency)
# ---------------------------------------------------------------------------

_NEGATIVE_KEYWORDS: frozenset[str] = frozenset({
    "angry", "furious", "disgusted", "horrible", "terrible", "awful",
    "worst", "unacceptable", "ridiculous", "scam", "fraud", "useless",
    "hate", "outraged", "disappointed", "frustrated", "never again",
    "pathetic", "incompetent", "rip off", "ripped off",
})

_FRUSTRATED_KEYWORDS: frozenset[str] = frozenset({
    "frustrated", "annoying", "annoyed", "can't believe", "cannot believe",
    "still not working", "again", "keeps happening", "wasted", "hours",
    "days", "weeks", "no response", "ignored", "waiting", "still waiting",
    "how long", "ridiculous wait",
})

_POSITIVE_KEYWORDS: frozenset[str] = frozenset({
    "thank", "thanks", "appreciate", "love", "great", "amazing",
    "excellent", "wonderful", "happy", "pleased", "satisfied", "helpful",
    "fantastic", "awesome",
})


from langchain_core.tools import tool

@tool
def analyse_sentiment(text: str) -> str:
    """Analyse the sentiment of a customer's ticket to determine their emotional state 
    (negative, frustrated, neutral, or positive) and receive tone guidance for the response. 
    Use this early in the resolution process to tailor your empathy level.
    Input should be the ticket subject and body text as a plain string."""
    lowered: str = text.lower()
    words: list[str] = lowered.split()
    word_set: set[str] = set(words)

    negative_hits: int = len(_NEGATIVE_KEYWORDS & word_set)
    frustrated_hits: int = len(_FRUSTRATED_KEYWORDS & word_set)
    positive_hits: int = len(_POSITIVE_KEYWORDS & word_set)

    # Score-based classification
    if negative_hits >= 2 or (negative_hits >= 1 and frustrated_hits >= 1):
        sentiment = "negative"
        tone_guidance = (
            "The customer appears angry or very upset. Lead with a sincere apology, "
            "acknowledge the inconvenience explicitly, and prioritise swift resolution. "
            "Avoid defensive or generic language."
        )
    elif frustrated_hits >= 2 or (negative_hits == 1 and positive_hits == 0):
        sentiment = "frustrated"
        tone_guidance = (
            "The customer shows signs of frustration. Show empathy and understanding, "
            "validate their experience, and provide clear and direct resolution steps."
        )
    elif positive_hits >= 1 and negative_hits == 0:
        sentiment = "positive"
        tone_guidance = (
            "The customer has a positive or neutral tone. Maintain a friendly, "
            "helpful tone. A brief thank-you for reaching out is appropriate."
        )
    else:
        sentiment = "neutral"
        tone_guidance = (
            "The customer's tone is neutral. Use a professional, clear, and concise "
            "response style."
        )

    sentiment_icons: dict[str, str] = {
        "negative": "😠",
        "frustrated": "😤",
        "positive": "😊",
        "neutral": "😐",
    }
    icon = sentiment_icons.get(sentiment, "😐")

    return (
        f"Detected Sentiment: {icon} {sentiment.upper()}\n\n"
        f"Tone Guidance: {tone_guidance}"
    )
