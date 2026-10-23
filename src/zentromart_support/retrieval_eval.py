"""Offline retrieval metrics against the dataset's gold evidence labels."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from .retrieval import KnowledgeRetriever


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(os.getenv("DATASET_DIR", _root() / "Capstone_Project_Dataset_FTTT_2026")),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=_root() / "outputs" / "retrieval_eval.jsonl",
    )
    args = parser.parse_args()

    retriever = KnowledgeRetriever(args.dataset)
    questions_path = args.dataset / "practice" / "questions.csv"
    evaluated = 0
    hit_count = 0
    reciprocal_ranks: list[float] = []
    evidence_recall: list[float] = []
    misses: list[str] = []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with questions_path.open(encoding="utf-8-sig", newline="") as source, args.output.open(
        "w", encoding="utf-8", newline=""
    ) as target:
        for row in csv.DictReader(source):
            gold = {part.strip() for part in row["gold_evidence"].split(";") if part.strip()}
            if not gold:
                continue
            results = retriever.retrieve(row["question"], top_k=args.top_k)
            ranked_ids = [item.evidence_id for item in results]
            found = gold.intersection(ranked_ids)
            evaluated += 1
            if found:
                hit_count += 1
                first_rank = min(ranked_ids.index(item) + 1 for item in found)
                reciprocal_ranks.append(1 / first_rank)
            else:
                reciprocal_ranks.append(0.0)
                misses.append(row["id"])
            evidence_recall.append(len(found) / len(gold))
            record = {
                "id": row["id"],
                "question": row["question"],
                "gold_evidence": sorted(gold),
                "retrieved_evidence": [item.to_dict() for item in results],
                "hit_at_k": bool(found),
                "evidence_recall_at_k": len(found) / len(gold),
            }
            target.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Evaluated answerable-evidence cases: {evaluated}")
    print(f"Hit@{args.top_k}: {hit_count}/{evaluated} ({hit_count / max(1, evaluated):.1%})")
    print(f"MRR@{args.top_k}: {sum(reciprocal_ranks) / max(1, evaluated):.3f}")
    print(f"Evidence recall@{args.top_k}: {sum(evidence_recall) / max(1, evaluated):.3f}")
    print(f"Missed question IDs: {', '.join(misses) if misses else 'none'}")
    print(f"Per-question details saved to {args.output}")


if __name__ == "__main__":
    main()
