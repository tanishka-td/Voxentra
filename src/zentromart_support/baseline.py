"""Prompt-only Step 1 baseline and practice-question runner."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .gemini import GeminiTextModel
from .json_records import write_records

SYSTEM_INSTRUCTION = """You are ZentroMart's customer-support assistant. In your first reply,
briefly introduce yourself as an AI assistant. Be polite, concise, and reply in the
language used by the customer. This baseline has no access to company documents,
catalogs, customer records, orders, or support tools. Do not claim that you checked
an order or completed an action. For ZentroMart-specific policies or facts you cannot
reliably know, say that you cannot confirm them and offer to connect the customer to
support. Never invent a policy, deadline, price, refund, or customer/order detail."""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _questions_path() -> Path:
    configured = os.getenv("PRACTICE_QUESTIONS")
    if configured:
        return Path(configured).expanduser().resolve()
    dataset_dir = Path(
        os.getenv(
            "DATASET_DIR",
            str(_project_root() / "Capstone_Project_Dataset_FTTT_2026"),
        )
    ).expanduser()
    return dataset_dir / "practice" / "questions.csv"


def _default_output_path() -> Path:
    return _project_root() / "outputs" / "step1_baseline.json"


def evaluate(model: GeminiTextModel, questions_path: Path, output_path: Path) -> None:
    if not questions_path.is_file():
        raise FileNotFoundError(f"Practice questions not found: {questions_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    records = []
    with questions_path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            started = time.perf_counter()
            try:
                answer = model.generate(
                    system_instruction=SYSTEM_INSTRUCTION,
                    prompt=row["question"].strip(),
                )
                record = {
                    "id": row["id"],
                    "language": row["language"],
                    "intent": row["intent"],
                    "question": row["question"],
                    "expected_decision": row["expected_decision"],
                    "acceptable_decisions": [
                        item.strip()
                        for item in row["acceptable_decisions"].split(";")
                        if item.strip()
                    ],
                    "answerable": row["answerable"] == "True",
                    "must_include": row["must_include"],
                    "must_not_include": row["must_not_include"],
                    "gold_evidence": [
                        item.strip()
                        for item in row["gold_evidence"].split(";")
                        if item.strip()
                    ],
                    "answer": answer,
                    "model": model.model_name,
                    "latency_seconds": round(time.perf_counter() - started, 3),
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            except Exception as exc:  # Keep an API failure visible in the record.
                record = {
                    "id": row.get("id"),
                    "question": row.get("question"),
                    "error": type(exc).__name__,
                    "model": model.model_name,
                }
            records.append(record)
            write_records(output_path, records)
            count += 1
            print(f"Processed {count}: {row['id']}")

    print(f"Saved {count} responses to {output_path}")
    print("Score these responses with `python -m zentromart_support.score_practice outputs/step1_baseline.json`.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run all practice questions and save responses as formatted JSON.",
    )
    parser.add_argument("--questions", type=Path, default=_questions_path())
    parser.add_argument("--output", type=Path, default=_default_output_path())
    args = parser.parse_args()

    try:
        model = GeminiTextModel()
        if args.evaluate:
            evaluate(model, args.questions, args.output)
        else:
            question = input("Customer: ").strip()
            if not question:
                print("Please enter a question.", file=sys.stderr)
                raise SystemExit(2)
            print("\nAssistant:", model.generate(
                system_instruction=SYSTEM_INSTRUCTION,
                prompt=question,
            ))
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
