"""Generate RAG answers for the single-question practice set."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from .agent import GroundedTextAgent
from .retrieval import KnowledgeRetriever


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(os.getenv("DATASET_DIR", _root() / "Capstone_Project_Dataset_FTTT_2026")))
    parser.add_argument("--output", type=Path, default=_root() / "outputs" / "step2_rag.jsonl")
    parser.add_argument("--limit", type=int, help="Optional maximum number of questions to send to Gemini.")
    parser.add_argument("--asr-text", action="store_true", help="Evaluate the practice set's ASR-like transcripts instead of typed questions.")
    args = parser.parse_args()

    questions_path = args.dataset / "practice" / "questions.csv"
    if not questions_path.is_file():
        parser.error(f"Practice questions not found: {questions_path}")
    try:
        agent = GroundedTextAgent(KnowledgeRetriever(args.dataset))
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    with questions_path.open(encoding="utf-8-sig", newline="") as source, args.output.open("w", encoding="utf-8") as target:
        for row in csv.DictReader(source):
            if args.limit is not None and processed >= args.limit:
                break
            question = (row["asr_text"] if args.asr_text else row["question"]).strip()
            result = agent.respond(question)
            record = {
                "id": row["id"],
                "question": question,
                "language": row["language"],
                "expected_decision": row["expected_decision"],
                "acceptable_decisions": [item.strip() for item in row["acceptable_decisions"].split(";") if item.strip()],
                "answerable": row["answerable"] == "True",
                "must_include": row["must_include"],
                "must_not_include": row["must_not_include"],
                "gold_evidence": [item.strip() for item in row["gold_evidence"].split(";") if item.strip()],
                "reply": result.reply,
                "decision": result.decision,
                "retrieved_evidence": result.retrieved_evidence,
                "model": agent.model.model_name,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            target.flush()
            processed += 1
            print(f"Processed {processed}: {row['id']}")
    print(f"Saved {processed} grounded responses to {args.output}")


if __name__ == "__main__":
    main()
