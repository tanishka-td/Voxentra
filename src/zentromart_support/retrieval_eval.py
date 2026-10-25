"""Evaluate retrieval quality against the practice set's gold evidence."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from .json_records import write_records
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
        "--lexical-only",
        action="store_true",
        help="Skip Gemini embeddings and measure the BM25 fallback only.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_root() / "outputs" / "retrieval_eval.json",
    )
    args = parser.parse_args()

    questions_path = args.dataset / "practice" / "questions.csv"
    with questions_path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))

    retriever = KnowledgeRetriever(args.dataset, use_embeddings=not args.lexical_only)
    retriever.prepare_queries([row["question"] for row in rows])
    evaluated = 0
    total_questions = 0
    hit_count = 0
    reciprocal_ranks: list[float] = []
    evidence_recall: list[float] = []
    misses: list[str] = []
    records = []

    for row in rows:
        gold = {part.strip() for part in row["gold_evidence"].split(";") if part.strip()}
        results = retriever.retrieve(row["question"], top_k=args.top_k)
        ranked_ids = [item.evidence_id for item in results]
        found = gold.intersection(ranked_ids)
        total_questions += 1
        if gold:
            evaluated += 1
            if found:
                hit_count += 1
                first_rank = min(ranked_ids.index(item) + 1 for item in found)
                reciprocal_ranks.append(1 / first_rank)
            else:
                reciprocal_ranks.append(0.0)
                misses.append(row["id"])
            evidence_recall.append(len(found) / len(gold))
        records.append({
            "id": row["id"],
            "question": row["question"],
            "answerable": row.get("answerable") == "True",
            "gold_evidence": sorted(gold),
            "retrieved_evidence": [item.to_dict() for item in results],
            "retrieval_mode": retriever.mode,
            "hit_at_k": bool(found) if gold else None,
            "evidence_recall_at_k": len(found) / len(gold) if gold else None,
        })

    write_records(args.output, records)

    print(f"Retrieval mode: {retriever.mode}")
    if retriever.embedding_error:
        print(f"Embedding fallback reason: {retriever.embedding_error}")
    print(f"Questions recorded: {total_questions}")
    print(f"Evaluated evidence-bearing cases: {evaluated}")
    print(f"Hit@{args.top_k}: {hit_count}/{evaluated} ({hit_count / max(1, evaluated):.1%})")
    print(f"MRR@{args.top_k}: {sum(reciprocal_ranks) / max(1, evaluated):.3f}")
    print(f"Evidence recall@{args.top_k}: {sum(evidence_recall) / max(1, evaluated):.3f}")
    print(f"Missed question IDs: {', '.join(misses) if misses else 'none'}")
    print(f"Per-question details saved to {args.output}")


if __name__ == "__main__":
    main()
