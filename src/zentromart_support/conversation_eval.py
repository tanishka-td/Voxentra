"""Run and locally score the supplied multi-turn conversation practice set."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .agent import GroundedTextAgent
from .capstone_tools import CapstoneToolGateway
from .conversation import ConversationAgent, ConversationState, _redact_proofs
from .retrieval import KnowledgeRetriever


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _alternatives(value: str) -> list[str]:
    return [item.strip() for item in value.split("/") if item.strip()]


def _requirement_groups(value: str) -> list[list[str]]:
    return [group for part in value.split(";") if (group := _alternatives(part))]


def _contains_any(text: str, alternatives: list[str]) -> bool:
    folded = text.casefold()
    return any(item.casefold() in folded for item in alternatives)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(os.getenv("DATASET_DIR", _root() / "Capstone_Project_Dataset_FTTT_2026")))
    parser.add_argument("--output", type=Path, default=_root() / "outputs" / "step3_conversations.jsonl")
    parser.add_argument("--limit", type=int, help="Optional maximum number of conversations to run.")
    parser.add_argument("--conversation-id", action="append", help="Run one selected conversation ID; may be repeated.")
    parser.add_argument("--lexical-only", action="store_true", help="Skip Gemini embeddings and use lexical retrieval.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    args = parser.parse_args()

    source_path = args.dataset / "practice" / "conversations.csv"
    if not source_path.is_file():
        parser.error(f"Practice conversations not found: {source_path}")
    if args.output.exists() and not args.overwrite:
        parser.error(f"Output already exists: {args.output}; pass --overwrite to replace it.")

    with source_path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["conversation_id"]].append(row)
    conversation_ids = sorted(groups)
    if args.conversation_id:
        missing = sorted(set(args.conversation_id) - groups.keys())
        if missing:
            parser.error(f"Unknown conversation ID(s): {', '.join(missing)}")
        conversation_ids = [item for item in conversation_ids if item in set(args.conversation_id)]
    elif args.limit is not None:
        conversation_ids = conversation_ids[:args.limit]

    try:
        retriever = KnowledgeRetriever(args.dataset, use_embeddings=not args.lexical_only)
        agent = ConversationAgent(
            GroundedTextAgent(retriever), CapstoneToolGateway(args.dataset), args.dataset
        )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    turn_count = 0
    turn_passes = 0
    conversation_passes = 0
    with args.output.open("w", encoding="utf-8") as target:
        for conversation_id in conversation_ids:
            state = ConversationState(conversation_id=conversation_id)
            turns_ok = True
            for row in sorted(groups[conversation_id], key=lambda item: int(item["turn"])):
                result = agent.respond(state, row["customer_message"])
                expected_decisions = {part.strip() for part in row["expected_decision"].split(";") if part.strip()}
                required = _requirement_groups(row["must_include"])
                forbidden = _requirement_groups(row["must_not_include"])
                actual_tool_names = {str(call.get("tool", "")) for call in result.tool_calls}
                expected_tool_names = set(re.findall(r"[a-z_]+(?=\()", row["expected_tool_calls"]))
                decision_ok = not expected_decisions or result.decision in expected_decisions
                required_ok = not required or all(
                    _contains_any(result.reply, group) for group in required
                )
                forbidden_ok = not forbidden or not any(
                    _contains_any(result.reply, group) for group in forbidden
                )
                tools_ok = expected_tool_names.issubset(actual_tool_names)
                evidence_ids = [item.get("evidence_id", "") for item in result.evidence]
                gold_ids = [item.strip() for item in row["gold_evidence"].split(";") if item.strip()]
                evidence_ok = not gold_ids or all(
                    any(gold in evidence_id for evidence_id in evidence_ids) for gold in gold_ids
                )
                passed = decision_ok and required_ok and forbidden_ok and tools_ok and evidence_ok
                turn_count += 1
                turn_passes += int(passed)
                turns_ok = turns_ok and passed
                record = {
                    "conversation_id": conversation_id,
                    "turn": int(row["turn"]),
                    "scenario": row["scenario"],
                    "customer_message": _redact_proofs(row["customer_message"]),
                    "reply": result.reply,
                    "decision": result.decision,
                    "expected_decision": sorted(expected_decisions),
                    "checks": {
                        "decision": decision_ok,
                        "must_include": required_ok,
                        "must_not_include": forbidden_ok,
                        "tool_calls": tools_ok,
                        "evidence": evidence_ok,
                    },
                    "passed": passed,
                    "tool_calls": result.tool_calls,
                    "evidence_ids": evidence_ids,
                    "model": agent.text_agent.model.model_name,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
                target.flush()
                print(f"Processed {conversation_id} turn {row['turn']}: {'pass' if passed else 'review'}")
            conversation_passes += int(turns_ok)

    turn_rate = 100 * turn_passes / turn_count if turn_count else 0.0
    conversation_rate = 100 * conversation_passes / len(conversation_ids) if conversation_ids else 0.0
    print(f"Turns passing local checks: {turn_passes}/{turn_count} ({turn_rate:.1f}%)")
    print(f"Conversations passing every turn: {conversation_passes}/{len(conversation_ids)} ({conversation_rate:.1f}%)")
    print(f"Per-turn details saved to {args.output}")
    print("These string and routing checks are approximate; review flagged replies manually.")


if __name__ == "__main__":
    main()
