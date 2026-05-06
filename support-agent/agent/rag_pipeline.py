"""
RAG Pipeline — Builds and queries a persistent ChromaDB vector store from
FAQ documents using local sentence-transformers embeddings (no external API cost).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

BASE_DIR: Path = Path(__file__).parent.parent
FAQ_DIR: Path = BASE_DIR / "data" / "faq_docs"
CHROMA_DIR: str = str(BASE_DIR / "chroma_db")
COLLECTION_NAME: str = "faq_knowledge_base"
EMBED_MODEL: str = "all-MiniLM-L6-v2"
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50
SIMILARITY_THRESHOLD: float = 0.4

# ---------------------------------------------------------------------------
# Embeddings (singleton-ish via module-level lazy init)
# ---------------------------------------------------------------------------

_embeddings: HuggingFaceEmbeddings | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFaceEmbeddings instance using all-MiniLM-L6-v2.

    Returns:
        HuggingFaceEmbeddings: A sentence-transformers embedding wrapper.
    """
    global _embeddings
    if _embeddings is None:
        logger.info("Loading sentence-transformer model: %s", EMBED_MODEL)
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


# ---------------------------------------------------------------------------
# Vector store builder
# ---------------------------------------------------------------------------

def build_vectorstore(force_rebuild: bool = False) -> Chroma:
    """Create or load the ChromaDB vector store from FAQ documents.

    Loads the 3 FAQ text files from data/faq_docs/, splits them into
    chunks, and upserts them into a persistent ChromaDB collection.
    Skips re-ingestion if the collection is already populated (unless
    force_rebuild=True).

    Args:
        force_rebuild: If True, clears and rebuilds the collection from scratch.

    Returns:
        Chroma: A LangChain Chroma vector store ready for querying.

    Raises:
        FileNotFoundError: If any FAQ document is missing.
        RuntimeError: If ChromaDB fails to initialise.
    """
    embeddings = _get_embeddings()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    existing_count: int = vectorstore._collection.count()

    if existing_count > 0 and not force_rebuild:
        logger.info(
            "Loaded existing ChromaDB collection '%s' with %d chunks.",
            COLLECTION_NAME,
            existing_count,
        )
        return vectorstore

    if force_rebuild and existing_count > 0:
        logger.info("Force rebuild requested — clearing existing collection.")
        vectorstore._collection.delete(
            where={"source": {"$ne": "__none__"}}
        )

    logger.info("Building vectorstore from FAQ documents in: %s", FAQ_DIR)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\nQ:", "\n\n", "\n", " "],
    )

    all_texts: list[str] = []
    all_metadatas: list[dict[str, Any]] = []
    all_ids: list[str] = []

    faq_files = list(FAQ_DIR.glob("*.txt"))
    if not faq_files:
        raise FileNotFoundError(f"No FAQ .txt files found in {FAQ_DIR}")

    for faq_file in sorted(faq_files):
        source_name = faq_file.stem  # e.g. "billing"
        logger.info("  Processing: %s", faq_file.name)

        content = faq_file.read_text(encoding="utf-8")
        chunks = splitter.split_text(content)

        for i, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_metadatas.append(
                {
                    "source": source_name,
                    "source_file": faq_file.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
            )
            all_ids.append(f"{source_name}_chunk_{i}")

    # Upsert into ChromaDB
    vectorstore._collection.upsert(
        documents=all_texts,
        metadatas=all_metadatas,  # type: ignore[arg-type]
        ids=all_ids,
        embeddings=embeddings.embed_documents(all_texts),
    )

    logger.info(
        "Vectorstore built: %d chunks from %d files.",
        len(all_texts),
        len(faq_files),
    )
    return vectorstore


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_context(
    query: str,
    k: int = 3,
    vectorstore: Chroma | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the top-k most relevant FAQ chunks for a query.

    Uses cosine similarity. If the best match score is below the
    SIMILARITY_THRESHOLD (0.4), returns an empty list to signal that no
    relevant context was found (triggering escalation in the agent).

    Args:
        query: The search query (e.g. customer ticket text).
        k: Number of top results to retrieve.
        vectorstore: Optional pre-built vectorstore; built if None.

    Returns:
        A list of result dicts, each containing:
            - 'content': The chunk text.
            - 'source': The FAQ category (billing/technical/general).
            - 'source_file': The filename.
            - 'chunk_index': Position of chunk in its source file.
            - 'score': Relevance score in [0.0, 1.0] (higher = more relevant).
        Returns empty list if best score < SIMILARITY_THRESHOLD.
    """
    if vectorstore is None:
        vectorstore = build_vectorstore()

    try:
        results_with_scores = vectorstore.similarity_search_with_relevance_scores(
            query, k=k
        )
    except Exception as exc:
        logger.error("ChromaDB retrieval failed: %s", exc)
        return []

    if not results_with_scores:
        logger.info("No results found for query: %s", query[:80])
        return []

    best_score: float = results_with_scores[0][1]
    logger.info(
        "Retrieved %d chunks — best score: %.3f (threshold: %.1f)",
        len(results_with_scores),
        best_score,
        SIMILARITY_THRESHOLD,
    )

    if best_score < SIMILARITY_THRESHOLD:
        logger.info(
            "Best score %.3f is below threshold %.1f — no useful context found.",
            best_score,
            SIMILARITY_THRESHOLD,
        )
        return []

    output: list[dict[str, Any]] = []
    for doc, score in results_with_scores:
        output.append(
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "source_file": doc.metadata.get("source_file", "unknown"),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "score": round(float(score), 4),
            }
        )

    return output


# ---------------------------------------------------------------------------
# Convenience: formatted string for LLM context
# ---------------------------------------------------------------------------

def format_context_for_prompt(results: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a readable string for LLM prompt injection.

    Args:
        results: List of result dicts from retrieve_context().

    Returns:
        A formatted string with numbered source references, or an empty
        string if results is empty.
    """
    if not results:
        return ""

    parts: list[str] = []
    for i, r in enumerate(results, start=1):
        parts.append(
            f"[Source {i}: {r['source_file']} | Score: {r['score']:.2f}]\n"
            f"{r['content']}"
        )
    return "\n\n---\n\n".join(parts)
