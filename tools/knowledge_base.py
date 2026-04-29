"""
Knowledge base tool — ingests KB articles into ChromaDB and exposes a
retrieval function as a LangChain StructuredTool.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KB_PATH: Path = Path(__file__).parent.parent / "data" / "knowledge_base.json"
CHROMA_DIR: str = str(Path(__file__).parent.parent / "chroma_db")
COLLECTION_NAME: str = "knowledge_base"
EMBED_MODEL: str = "all-MiniLM-L6-v2"
TOP_K: int = 3


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace embedding function backed by sentence-transformers.

    Returns:
        HuggingFaceEmbeddings: LangChain-compatible embedding wrapper.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ---------------------------------------------------------------------------
# ChromaDB ingestion
# ---------------------------------------------------------------------------

def ingest_knowledge_base(force: bool = False) -> Chroma:
    """Load KB articles from JSON and upsert them into ChromaDB.

    Args:
        force: If True, re-ingest even if the collection already exists.

    Returns:
        Chroma: An initialised LangChain Chroma vector store instance.

    Raises:
        FileNotFoundError: If the knowledge_base.json file is missing.
        json.JSONDecodeError: If the JSON file is malformed.
    """
    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    # Check whether collection is already populated
    existing_count: int = vectorstore._collection.count()
    if existing_count > 0 and not force:
        return vectorstore

    # Load articles
    if not KB_PATH.exists():
        raise FileNotFoundError(f"Knowledge base file not found at {KB_PATH}")

    with KB_PATH.open(encoding="utf-8") as fh:
        articles: list[dict[str, Any]] = json.load(fh)

    texts: list[str] = []
    metadatas: list[dict[str, str]] = []
    ids: list[str] = []

    for article in articles:
        texts.append(f"{article['title']}\n\n{article['content']}")
        metadatas.append(
            {
                "id": article["id"],
                "category": article["category"],
                "title": article["title"],
            }
        )
        ids.append(article["id"])

    # Upsert — safe to call multiple times
    vectorstore._collection.upsert(
        documents=texts,
        metadatas=metadatas,  # type: ignore[arg-type]
        ids=ids,
        embeddings=embeddings.embed_documents(texts),
    )

    return vectorstore


# ---------------------------------------------------------------------------
# Retrieval helper
# ---------------------------------------------------------------------------

def retrieve_kb_articles(query: str, top_k: int = TOP_K) -> str:
    """Retrieve the most relevant KB articles for a given query.

    Args:
        query: The user's support query or question.
        top_k: Maximum number of KB articles to return.

    Returns:
        A formatted string of matching KB articles with their titles and
        content, ready for inclusion in a prompt.
    """
    try:
        vectorstore = ingest_knowledge_base()
        results = vectorstore.similarity_search(query, k=top_k)
        if not results:
            return "No relevant knowledge base articles found."

        output_parts: list[str] = []
        for i, doc in enumerate(results, start=1):
            title = doc.metadata.get("title", "Untitled")
            category = doc.metadata.get("category", "general")
            output_parts.append(
                f"[KB Article {i}] ({category.upper()}) {title}\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(output_parts)

    except Exception as exc:  # noqa: BLE001
        return f"Knowledge base retrieval failed: {exc}"


# ---------------------------------------------------------------------------
# LangChain StructuredTool schema
# ---------------------------------------------------------------------------

class KBSearchInput(BaseModel):
    """Input schema for the knowledge base search tool."""

    query: str = Field(
        description="The customer support query to search the knowledge base for."
    )


def build_kb_tool() -> StructuredTool:
    """Build and return the KB retrieval LangChain StructuredTool.

    Returns:
        StructuredTool: A ready-to-use LangChain tool for KB retrieval.
    """
    return StructuredTool.from_function(
        func=retrieve_kb_articles,
        name="search_knowledge_base",
        description=(
            "Search the internal knowledge base for articles relevant to a customer "
            "support query. Use this tool first before composing any response to a ticket. "
            "Input should be the customer's question or issue description."
        ),
        args_schema=KBSearchInput,
        return_direct=False,
    )
