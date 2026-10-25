"""Interactive text interface for the grounded single-turn agent."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .agent import GroundedTextAgent
from .retrieval import KnowledgeRetriever


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    dataset_dir = Path(os.getenv("DATASET_DIR", root / "Capstone_Project_Dataset_FTTT_2026"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="One customer question; omit for interactive chat.")
    parser.add_argument("--evidence-json", action="store_true", help="Print the reply and evidence record as JSON.")
    args = parser.parse_args()

    try:
        agent = GroundedTextAgent(KnowledgeRetriever(dataset_dir))
        if args.question:
            result = agent.respond(args.question)
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) if args.evidence_json else result.reply)
            return
        while True:
            try:
                question = input("Customer (or 'quit'): ").strip()
            except EOFError:
                break
            if question.casefold() in {"quit", "exit"}:
                break
            if not question:
                continue
            result = agent.respond(question)
            print("Assistant:", result.reply)
            if args.evidence_json:
                print(json.dumps({"evidence_ids": result.evidence_ids}, ensure_ascii=False))
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
