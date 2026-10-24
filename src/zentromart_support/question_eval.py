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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard existing output and regenerate responses instead of resuming.",
    )
    parser.add_argument(
        "--lexical-only",
        action="store_true",
        help="Skip Gemini embeddings and use the BM25 fallback only.",
    )
    parser.add_argument("--asr-text", action="store_true", help="Evaluate the practice set's ASR-like transcripts instead of typed questions.")
    args = parser.parse_args()

    questions_path = args.dataset / "practice" / "questions.csv"
    if not questions_path.is_file():
        parser.error(f"Practice questions not found: {questions_path}")
    try:
        retriever = KnowledgeRetriever(args.dataset, use_embeddings=not args.lexical_only)
        agent = GroundedTextAgent(retriever)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    with questions_path.open(encoding="utf-8-sig", newline="") as source:
        all_rows = list(csv.DictReader(source))
    selected_rows = all_rows if args.limit is None else all_rows[:args.limit]
    expected_questions = {
        row["id"]: (row["asr_text"] if args.asr_text else row["question"]).strip()
        for row in all_rows
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict] = {}
    if args.output.is_file() and not args.overwrite:
        try:
            for line in args.output.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                question_id = str(record.get("id", ""))
                if (
                    question_id in expected_questions
                    and record.get("question") == expected_questions[question_id]
                    and record.get("model") == agent.model.model_name
                ):
                    completed[question_id] = record
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Error reading existing output for resume: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

    pending_rows = [row for row in selected_rows if row["id"] not in completed]
    # Deterministic data/rule answers do not need query embeddings. Prepare
    # vectors only for questions that will go through document retrieval.
    retriever.prepare_queries([
        expected_questions[row["id"]]
        for row in pending_rows
        if agent.structured_data.answer_question(expected_questions[row["id"]]) is None
    ])

    # Rewrite the valid checkpoint once to remove duplicate or stale lines,
    # then append each new response immediately so interrupted runs can resume.
    with args.output.open("w", encoding="utf-8") as target:
        for record in completed.values():
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
    processed = len(completed)
    if processed:
        print(f"Resuming with {processed} existing responses; {len(pending_rows)} remain.")

    with args.output.open("a", encoding="utf-8") as target:
        for row in pending_rows:
            question = expected_questions[row["id"]]
            result = agent.respond(question)
            record = {
                "id": row["id"],
                "question": question,
                "language": row["language"],
                "expected_decision": row["expected_decision"],
                "acceptable_decisions": [item.strip() for item in row["acceptable_decisions"].split(";") if item.strip()],
                "answerable": row["answerable"] == "True",
                "input_mode": "asr_text" if args.asr_text else "question",
                "must_include": row["must_include"],
                "must_not_include": row["must_not_include"],
                "gold_evidence": [item.strip() for item in row["gold_evidence"].split(";") if item.strip()],
                "reply": result.reply,
                "decision": result.decision,
                "retrieval_mode": retriever.mode,
                "retrieved_evidence": result.retrieved_evidence,
                "evidence_ids": result.evidence_ids,
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
