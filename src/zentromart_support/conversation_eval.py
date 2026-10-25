"""Run and locally score the supplied multi-turn conversation practice set."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .agent import GroundedTextAgent
from .capstone_tools import CapstoneToolGateway
from .conversation import ConversationAgent, ConversationState, _redact_proofs
from .json_records import write_records
from .retrieval import KnowledgeRetriever


class _RecordingTools:
    """Capture tool arguments in memory for scoring; never serialize proof values."""

    def __init__(self, wrapped: CapstoneToolGateway):
        self._wrapped = wrapped
        self.calls: list[dict[str, object]] = []

    def lookup_order(self, order_id: str, verification_value: str | None = None):
        call: dict[str, object] = {"tool": "lookup_order", "order_id": str(order_id)}
        if verification_value is not None:
            call["verification_value"] = str(verification_value)
        self.calls.append(call)
        return self._wrapped.lookup_order(order_id, verification_value=verification_value)

    def create_ticket(self, summary: str, *, order_id=None, reason=None, intent=None):
        outcome = self._wrapped.create_ticket(
            summary, order_id=order_id, reason=reason, intent=intent
        )
        self.calls.append({"tool": "create_ticket", "created": bool(outcome.get("created"))})
        return outcome


def _expected_tool_calls(value: str) -> list[dict[str, str]]:
    calls = []
    for name, raw_arguments in re.findall(r"([a-z_]+)\(([^)]*)\)", value):
        arguments = {
            key.strip(): item.strip()
            for part in raw_arguments.split(",")
            if (separator := part.partition("="))[1]
            for key, item in [(separator[0], separator[2])]
        }
        calls.append({"tool": name, **arguments})
    return calls


def _tool_checks(expected: list[dict[str, str]], actual: list[dict[str, object]]) -> tuple[bool, bool]:
    """Return (arguments/order match, expected ticket was successfully created)."""
    calls_match = len(expected) == len(actual)
    for wanted, got in zip(expected, actual):
        calls_match = calls_match and all(str(got.get(key, "")) == value for key, value in wanted.items())
    ticket_expected = any(call["tool"] == "create_ticket" for call in expected)
    ticket_created = not ticket_expected or any(
        call.get("tool") == "create_ticket" and call.get("created") is True for call in actual
    )
    return calls_match, ticket_created


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
    parser.add_argument(
        "--output", type=Path,
        default=_root() / "outputs" / "phase3_conversations" / "step3_conversations.json",
    )
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
        tools = _RecordingTools(CapstoneToolGateway(args.dataset))
        agent = ConversationAgent(GroundedTextAgent(retriever), tools, args.dataset)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    turn_count = 0
    turn_passes = 0
    conversation_passes = 0
    records = []
    for conversation_id in conversation_ids:
        state = ConversationState(conversation_id=conversation_id)
        turns_ok = True
        for row in sorted(groups[conversation_id], key=lambda item: int(item["turn"])):
            tools.calls.clear()
            result = agent.respond(state, row["customer_message"])
            expected_decisions = {
                part.strip() for part in row["expected_decision"].split(";") if part.strip()
            }
            required = _requirement_groups(row["must_include"])
            forbidden = _requirement_groups(row["must_not_include"])
            expected_calls = _expected_tool_calls(row["expected_tool_calls"])
            decision_ok = not expected_decisions or result.decision in expected_decisions
            required_ok = not required or all(
                _contains_any(result.reply, group) for group in required
            )
            forbidden_ok = not forbidden or not any(
                _contains_any(result.reply, group) for group in forbidden
            )
            tool_arguments_ok, ticket_created_ok = _tool_checks(expected_calls, tools.calls)
            tools_ok = tool_arguments_ok and ticket_created_ok and (
                {str(call.get("tool", "")) for call in result.tool_calls}
                == {str(call.get("tool", "")) for call in tools.calls}
            )
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
                    "tool_arguments": tool_arguments_ok,
                    "ticket_created": ticket_created_ok,
                    "evidence": evidence_ok,
                },
                "passed": passed,
                "tool_calls": result.tool_calls,
                "evidence_ids": evidence_ids,
                "model": agent.text_agent.model.model_name,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            records.append(record)
            write_records(args.output, records)
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
