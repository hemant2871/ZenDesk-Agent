"""
Search FAQ Tool — queries ChromaDB vector store and returns formatted
context chunks with source metadata as a LangChain Tool.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from langchain.tools import Tool

# Ensure project root is on sys.path when imported from various contexts
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent.rag_pipeline import (  # noqa: E402
    build_vectorstore,
    format_context_for_prompt,
    retrieve_context,
)

logger = logging.getLogger(__name__)

# Shared vectorstore instance — built once per process
_vectorstore: Any = None


def _get_vectorstore() -> Any:
    """Lazy-load and cache the vector store.

    Returns:
        Chroma: The initialized vector store.
    """
    global _vectorstore
    if _vectorstore is None:
        logger.info("Initialising vectorstore for search_faq tool...")
        _vectorstore = build_vectorstore()
    return _vectorstore


# ---------------------------------------------------------------------------
# Core search function
# ---------------------------------------------------------------------------

def search_faq(query: str) -> str:
    """Search the FAQ knowledge base for answers relevant to a customer query.

    Retrieves the top-3 FAQ chunks from ChromaDB using semantic similarity.
    If no chunk scores above the similarity threshold (0.4), returns a
    message indicating no relevant answer was found, which should trigger
    ticket escalation.

    Args:
        query: The customer's question or a summary of their issue.

    Returns:
        A formatted string containing matched FAQ content with source
        references and similarity scores, or a "no results" message
        if no relevant content is found.
    """
    logger.info("search_faq called with query: %s", query[:100])

    try:
        vs = _get_vectorstore()
        results = retrieve_context(query=query, k=3, vectorstore=vs)
    except Exception as exc:
        logger.error("search_faq error: %s", exc)
        return f"Knowledge base search failed: {exc}"

    if not results:
        logger.info("No relevant FAQ content found above threshold.")
        return (
            "NO_RELEVANT_ANSWER_FOUND: The knowledge base does not contain a "
            "relevant answer for this query. Confidence is below threshold. "
            "This ticket should be escalated to a human agent."
        )

    # Build formatted response
    lines: list[str] = [
        f"Found {len(results)} relevant FAQ entries:\n",
    ]
    for i, r in enumerate(results, start=1):
        lines.append(
            f"--- Result {i} (Source: {r['source_file']}, Score: {r['score']:.2f}) ---\n"
            f"{r['content']}\n"
        )

    logger.info(
        "search_faq returning %d results, best score: %.3f",
        len(results),
        results[0]["score"] if results else 0.0,
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LangChain Tool
# ---------------------------------------------------------------------------

search_faq_tool = Tool(
    name="search_faq",
    func=search_faq,
    description=(
        "Search the internal FAQ knowledge base for answers to customer support queries. "
        "Input should be a clear query string describing the customer's issue. "
        "Returns relevant FAQ content with source references and similarity scores. "
        "If it returns 'NO_RELEVANT_ANSWER_FOUND', use the escalate_ticket tool instead."
    ),
)
