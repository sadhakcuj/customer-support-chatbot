"""Measure retrieval quality. The threshold is chosen on the dev split and reported on the test split.

Usage: python -m eval.run_eval [--write]   (--write saves eval/results.json)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.retriever import Retriever

ROOT = Path(__file__).resolve().parent.parent


def evaluate(retriever: Retriever, items: list[dict], threshold: float) -> dict:
    in_scope = [i for i in items if i["expect"]]
    out_scope = [i for i in items if not i["expect"]]
    correct = wrong = abstained = in_top3 = 0
    for i in in_scope:
        top3 = retriever.search(i["q"], 3)
        in_top3 += any(h.id == i["expect"] for h in top3)
        hits = top3[:1]
        if not hits or hits[0].score < threshold:
            abstained += 1
        elif hits[0].id == i["expect"]:
            correct += 1
        else:
            wrong += 1
    false_answers = sum(1 for i in out_scope if (h := retriever.search(i["q"], 1)) and h[0].score >= threshold)
    n_in, n_out = len(in_scope), len(out_scope)
    answered = correct + wrong
    return {
        "threshold": threshold,
        "in_scope_questions": n_in,
        "out_of_scope_questions": n_out,
        "answered_correctly": correct,
        "answered_wrongly": wrong,
        "abstained_in_scope": abstained,
        "in_scope_accuracy": round(correct / n_in, 4),
        "recall_at_3_ignoring_threshold": round(in_top3 / n_in, 4),
        "precision_when_answering": round(correct / answered, 4) if answered else None,
        "out_of_scope_wrongly_answered": false_answers,
        "out_of_scope_rejection_rate": round(1 - false_answers / n_out, 4),
    }


def choose_threshold(retriever: Retriever, dev: list[dict]) -> float:
    """Highest in-scope accuracy subject to rejecting at least 90% of out-of-scope questions."""
    best, best_key = 0.3, (-1.0, 0.0)
    for t in [x / 100 for x in range(10, 71, 1)]:
        r = evaluate(retriever, dev, t)
        if r["out_of_scope_rejection_rate"] < 0.9:
            continue
        key = (r["in_scope_accuracy"], -t)
        if key > best_key:
            best, best_key = t, key
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    retriever = Retriever.from_file(ROOT / "data" / "knowledge_base.json")
    data = json.loads((ROOT / "eval" / "eval_set.json").read_text())
    thr = choose_threshold(retriever, data["dev"])
    out = {"chosen_on": "dev split", "dev": evaluate(retriever, data["dev"], thr), "test": evaluate(retriever, data["test"], thr)}
    print(json.dumps(out, indent=2))
    if args.write:
        (ROOT / "eval" / "results.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
