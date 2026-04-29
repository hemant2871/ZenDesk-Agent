"""
Evaluation script — runs the agent against sample tickets and scores
responses on KB citation accuracy and resolution completeness.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path when run directly
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.agent import build_agent_executor, resolve_ticket  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICKETS_PATH: Path = PROJECT_ROOT / "data" / "sample_tickets.json"
RESULTS_PATH: Path = PROJECT_ROOT / "eval" / "results.json"


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_kb_citation(
    response_text: str, expected_kb_ids: list[str], kb_data: list[dict[str, Any]]
) -> dict[str, Any]:
    """Check whether the response mentions expected KB article titles.

    Args:
        response_text: The agent's final output text.
        expected_kb_ids: List of KB article IDs expected to be referenced.
        kb_data: The full knowledge base as a list of article dicts.

    Returns:
        A dict containing 'hits', 'misses', and 'precision' score (0.0–1.0).
    """
    # Build id -> title mapping
    id_to_title: dict[str, str] = {
        article["id"]: article["title"].lower()
        for article in kb_data
    }

    response_lower = response_text.lower()
    hits: list[str] = []
    misses: list[str] = []

    for kb_id in expected_kb_ids:
        title = id_to_title.get(kb_id, "")
        # Check if any significant words from the title appear in the response
        title_words = [w for w in title.split() if len(w) > 4]
        if title_words and any(word in response_lower for word in title_words):
            hits.append(kb_id)
        else:
            misses.append(kb_id)

    precision = len(hits) / len(expected_kb_ids) if expected_kb_ids else 0.0
    return {"hits": hits, "misses": misses, "precision": round(precision, 2)}


def score_resolution_quality(response_text: str) -> dict[str, bool]:
    """Check structural completeness of the resolution response.

    Args:
        response_text: The agent's final output text.

    Returns:
        A dict of boolean quality checks.
    """
    lowered = response_text.lower()
    return {
        "has_ticket_id": "tick-" in lowered or "ticket" in lowered,
        "has_priority": any(
            p in lowered for p in ["critical", "high", "medium", "low"]
        ),
        "has_resolution_steps": any(
            w in lowered for w in ["step", "navigate", "go to", "click", "please", "you can"]
        ),
        "has_kb_reference": "kb article" in lowered or "knowledge base" in lowered,
        "has_closing": any(
            w in lowered for w in ["follow up", "further assistance", "support ai", "human agent"]
        ),
    }


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(verbose: bool = True) -> list[dict[str, Any]]:
    """Run the agent against all sample tickets and compute evaluation metrics.

    Args:
        verbose: If True, print progress and results to stdout.

    Returns:
        A list of per-ticket result dicts containing scores and agent output.

    Raises:
        FileNotFoundError: If sample_tickets.json or knowledge_base.json is missing.
    """
    # Load test data
    with TICKETS_PATH.open(encoding="utf-8") as fh:
        tickets: list[dict[str, Any]] = json.load(fh)

    kb_path = PROJECT_ROOT / "data" / "knowledge_base.json"
    with kb_path.open(encoding="utf-8") as fh:
        kb_data: list[dict[str, Any]] = json.load(fh)

    if verbose:
        print(f"\n{'='*60}")
        print("  Zangoh Support Agent — Evaluation Suite")
        print(f"{'='*60}")
        print(f"  Tickets to evaluate: {len(tickets)}\n")

    executor = build_agent_executor()
    all_results: list[dict[str, Any]] = []

    for i, ticket in enumerate(tickets, start=1):
        ticket_id = ticket["id"]
        if verbose:
            print(f"\n[{i}/{len(tickets)}] Processing {ticket_id}: {ticket['subject']}")
            print("-" * 50)

        start_time = time.time()
        result = resolve_ticket(
            executor=executor,
            ticket_id=ticket_id,
            subject=ticket["subject"],
            body=ticket["body"],
        )
        elapsed = round(time.time() - start_time, 2)

        output_text: str = result.get("output", "")
        kb_score = score_kb_citation(
            output_text, ticket.get("expected_kb_ids", []), kb_data
        )
        quality_score = score_resolution_quality(output_text)
        quality_pass_rate = sum(quality_score.values()) / len(quality_score)

        ticket_result = {
            "ticket_id": ticket_id,
            "subject": ticket["subject"],
            "category": ticket["category"],
            "expected_priority": ticket["priority"],
            "elapsed_seconds": elapsed,
            "kb_citation_score": kb_score,
            "quality_checks": quality_score,
            "quality_pass_rate": round(quality_pass_rate, 2),
            "agent_output": output_text,
        }
        all_results.append(ticket_result)

        if verbose:
            print(f"  ⏱  Elapsed: {elapsed}s")
            print(f"  📚 KB Citation Precision: {kb_score['precision']}")
            print(f"  ✅ Quality Pass Rate: {quality_pass_rate:.0%}")
            if kb_score["misses"]:
                print(f"  ⚠️  Missed KB IDs: {kb_score['misses']}")

    # Aggregate metrics
    avg_kb = round(
        sum(r["kb_citation_score"]["precision"] for r in all_results) / len(all_results), 2
    )
    avg_quality = round(
        sum(r["quality_pass_rate"] for r in all_results) / len(all_results), 2
    )

    summary = {
        "total_tickets": len(tickets),
        "avg_kb_citation_precision": avg_kb,
        "avg_quality_pass_rate": avg_quality,
        "results": all_results,
    }

    # Save results
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\n{'='*60}")
        print("  EVALUATION COMPLETE")
        print(f"  Avg KB Citation Precision : {avg_kb}")
        print(f"  Avg Quality Pass Rate     : {avg_quality:.0%}")
        print(f"  Results saved to          : {RESULTS_PATH}")
        print(f"{'='*60}\n")

    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_evaluation(verbose=True)
