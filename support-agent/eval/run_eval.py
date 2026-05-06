"""
Evaluation Runner — runs all 15 test cases through the support agent,
scores each with LLM-as-judge, prints a summary table, and saves results.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root on sys.path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent.evaluator import evaluate_response  # noqa: E402
from agent.support_agent import process_ticket  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TEST_CASES_PATH: Path = Path(__file__).parent / "test_cases.json"
RESULTS_PATH: Path = Path(__file__).parent / "results.json"

# ANSI colors for terminal output
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_test_cases() -> list[dict[str, Any]]:
    """Load test cases from JSON file.

    Returns:
        List of test case dicts.

    Raises:
        FileNotFoundError: If test_cases.json is missing.
    """
    if not TEST_CASES_PATH.exists():
        raise FileNotFoundError(f"Test cases not found at: {TEST_CASES_PATH}")
    with TEST_CASES_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _action_match(predicted: str, expected: str) -> bool:
    """Check if the predicted action matches the expected action.

    Args:
        predicted: Agent's predicted action string.
        expected: Expected action string.

    Returns:
        True if they match (case-insensitive).
    """
    return predicted.strip().lower() == expected.strip().lower()


def _color_score(score: float) -> str:
    """Return a colored string for a score value.

    Args:
        score: Score value (0-5).

    Returns:
        ANSI-colored string.
    """
    if score >= 4.0:
        return f"{_GREEN}{score:.1f}{_RESET}"
    if score >= 3.0:
        return f"{_YELLOW}{score:.1f}{_RESET}"
    return f"{_RED}{score:.1f}{_RESET}"


def _print_table(results: list[dict[str, Any]]) -> None:
    """Print a formatted summary table to stdout.

    Args:
        results: List of per-test-case result dicts.
    """
    header = (
        f"{'ID':<8} {'Category':<12} {'Expected':<12} {'Got':<12} "
        f"{'Match':<8} {'Score':<8} {'Feedback'}"
    )
    print(f"\n{_BOLD}{header}{_RESET}")
    print("-" * 100)

    for r in results:
        match = r["action_match"]
        match_str = f"{_GREEN}✓ YES{_RESET}" if match else f"{_RED}✗ NO{_RESET}"
        score_str = _color_score(r["eval"]["overall_score"])
        feedback_short = r["eval"]["feedback"][:55] + "…" if len(r["eval"]["feedback"]) > 55 else r["eval"]["feedback"]
        print(
            f"{r['id']:<8} {r['expected_category']:<12} {r['expected_action']:<12} "
            f"{r['predicted_action']:<12} {match_str:<18} {score_str:<18} {feedback_short}"
        )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_evaluation(verbose: bool = True) -> dict[str, Any]:
    """Run all test cases through the agent and evaluate with LLM-as-judge.

    Args:
        verbose: If True, prints progress and final table to stdout.

    Returns:
        A summary dict with accuracy, average score, and per-case results.

    Raises:
        FileNotFoundError: If test_cases.json is missing.
    """
    test_cases = _load_test_cases()
    total = len(test_cases)

    if verbose:
        print(f"\n{_BOLD}{_CYAN}{'='*60}")
        print("  SUPPORT AGENT EVALUATION SUITE")
        print(f"  Running {total} test cases...")
        print(f"{'='*60}{_RESET}\n")

    all_results: list[dict[str, Any]] = []
    correct_actions = 0
    score_by_category: dict[str, list[float]] = {
        "billing": [],
        "technical": [],
        "general": [],
    }

    for i, tc in enumerate(test_cases, start=1):
        tc_id = tc["id"]
        ticket_text = tc["ticket_text"]
        expected_action = tc["expected_action"]
        expected_category = tc["expected_category"]

        if verbose:
            print(f"[{i:>2}/{total}] {_BOLD}{tc_id}{_RESET}: {ticket_text[:60]}…")

        # Run agent
        start = time.time()
        agent_result = process_ticket(ticket_text)
        elapsed = round(time.time() - start, 1)

        predicted_action = agent_result.get("action", "escalated")
        predicted_category = agent_result.get("category", "general")
        agent_response = agent_result.get("agent_response", "")

        # Evaluate
        eval_scores = evaluate_response(
            ticket=ticket_text,
            agent_response=agent_response,
            expected_action=expected_action,
        )

        is_correct = _action_match(predicted_action, expected_action)
        if is_correct:
            correct_actions += 1

        overall = eval_scores.get("overall_score", 0.0)
        score_by_category.setdefault(expected_category, []).append(overall)

        result = {
            "id": tc_id,
            "ticket_text": ticket_text,
            "expected_action": expected_action,
            "expected_category": expected_category,
            "predicted_action": predicted_action,
            "predicted_category": predicted_category,
            "action_match": is_correct,
            "elapsed_seconds": elapsed,
            "eval": eval_scores,
            "agent_response": agent_response,
        }
        all_results.append(result)

        status_icon = "✓" if is_correct else "✗"
        if verbose:
            print(
                f"   {status_icon} Action: {predicted_action} (expected: {expected_action}) | "
                f"Score: {overall:.1f}/5 | {elapsed}s\n"
            )

    # Aggregate metrics
    avg_score = round(
        sum(r["eval"]["overall_score"] for r in all_results) / total, 2
    )
    best_score = round(max(r["eval"]["overall_score"] for r in all_results), 2)
    worst_score = round(min(r["eval"]["overall_score"] for r in all_results), 2)
    accuracy_pct = round(correct_actions / total * 100, 1)

    avg_by_category: dict[str, float] = {
        cat: round(sum(scores) / len(scores), 2) if scores else 0.0
        for cat, scores in score_by_category.items()
    }

    summary: dict[str, Any] = {
        "total_cases": total,
        "correct_actions": correct_actions,
        "accuracy_percent": accuracy_pct,
        "average_score": avg_score,
        "best_score": best_score,
        "worst_score": worst_score,
        "average_score_by_category": avg_by_category,
        "results": all_results,
    }

    # Print table
    if verbose:
        _print_table(all_results)
        print(f"\n{_BOLD}{_CYAN}{'='*60}")
        print(f"  FINAL RESULTS")
        print(f"{'='*60}{_RESET}")
        print(f"  Correct Actions : {_BOLD}{correct_actions}/{total}{_RESET}  ({accuracy_pct}%)")
        print(f"  Average Score   : {_color_score(avg_score)}/5")
        print(f"  Best Score      : {_color_score(best_score)}/5")
        print(f"  Worst Score     : {_color_score(worst_score)}/5")
        print(f"\n  Scores by Category:")
        for cat, score in avg_by_category.items():
            print(f"    {cat:<12}: {_color_score(score)}/5")
        print(f"\n  Results saved → {RESULTS_PATH}")
        print(f"{_CYAN}{'='*60}{_RESET}\n")

    # Save results
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_evaluation(verbose=True)
