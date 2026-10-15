"""Label a sample of RAG answers by hand, to validate the LLM judge against humans.

    python -m lab_b.annotate --answers lab-b-nlp-llm/reports/results/b4_rag_answers.csv --n 60

Shows each question, its reference answers and a model answer; type ``y`` (correct),
``n`` (wrong), ``s`` (skip) or ``q`` (quit). Labels are appended to
``lab-b-nlp-llm/annotations/b4_human_labels.csv``; re-running continues where you
stopped. Then re-run ``python -m lab_b.rag`` to get Cohen's κ against your labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lab_b.datasets import load_qa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=Path("lab-b-nlp-llm/annotations/b4_human_labels.csv")
    )
    parser.add_argument("--config", default="rag@3")
    parser.add_argument("--dataset", default="squad")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    answers = pd.read_csv(args.answers)
    sample = (
        answers[answers.config == args.config].sample(frac=1, random_state=args.seed).head(args.n)
    )
    done = (
        pd.read_csv(args.out)
        if args.out.exists()
        else pd.DataFrame(columns=["model", "config", "qid", "human_correct"])
    )
    seen = set(zip(done.model, done.config, done.qid, strict=True))
    questions = {q.id: q for q in load_qa(args.dataset).questions}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    todo = [r for r in sample.itertuples() if (r.model, r.config, r.qid) not in seen]
    print(f"{len(seen)} labelled, {len(todo)} to go. y = correct, n = wrong, s = skip, q = quit\n")
    for i, r in enumerate(todo, 1):
        q = questions[r.qid]
        print(
            f"[{i}/{len(todo)}] {q.text}\n  reference: {' | '.join(q.answers)}\n  answer:    {r.answer}"
        )
        key = ""
        while key not in {"y", "n", "s", "q"}:
            key = input("  correct? [y/n/s/q] ").strip().lower()
        if key == "q":
            break
        if key == "s":
            continue
        row = pd.DataFrame(
            [{"model": r.model, "config": r.config, "qid": r.qid, "human_correct": int(key == "y")}]
        )
        row.to_csv(args.out, mode="a", header=not args.out.exists(), index=False)
        print()
    print(f"Labels saved to {args.out}")


if __name__ == "__main__":
    main()
