"""Interactive multi-turn text interface for the support agent."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .agent import GroundedTextAgent
from .capstone_tools import CapstoneToolGateway
from .conversation import ConversationAgent, ConversationState, _redact_proofs
from .retrieval import KnowledgeRetriever


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    dataset_dir = Path(os.getenv("DATASET_DIR", root / "Capstone_Project_Dataset_FTTT_2026"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversation-id", default=f"local-{uuid.uuid4().hex[:8]}")
    parser.add_argument("--output", type=Path, help="Optional JSONL turn log; verification numbers are redacted.")
    args = parser.parse_args()

    try:
        text_agent = GroundedTextAgent(KnowledgeRetriever(dataset_dir))
        agent = ConversationAgent(text_agent, CapstoneToolGateway(dataset_dir), dataset_dir)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    state = ConversationState(conversation_id=args.conversation_id)
    output = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output = args.output.open("a", encoding="utf-8")
    try:
        while True:
            try:
                message = input("Customer (or 'quit'): ").strip()
            except EOFError:
                break
            if message.casefold() in {"quit", "exit"}:
                break
            result = agent.respond(state, message)
            print("Assistant:", result.reply)
            if output:
                record = {
                    "conversation_id": args.conversation_id,
                    "turn": state.turns,
                    "customer_message": _redact_proofs(message),
                    **result.to_dict(),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
    finally:
        if output:
            output.close()


if __name__ == "__main__":
    main()
