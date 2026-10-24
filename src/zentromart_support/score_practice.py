"""Approximate practice-set scoring for generated JSONL answer records."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5,
    "पांच": 5, "छह": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10,
    "बारह": 12, "पंद्रह": 15, "तीस": 30, "अड़तालीस": 48,
    "उनचास": 49, "साठ": 60, "सत्तर": 70, "उनहत्तर": 69,
}

MONTHS = {
    "january": "01", "jan": "01", "february": "02", "feb": "02",
    "march": "03", "mar": "03", "april": "04", "apr": "04",
    "may": "05", "june": "06", "jun": "06", "july": "07", "jul": "07",
    "august": "08", "aug": "08", "september": "09", "sep": "09",
    "october": "10", "oct": "10", "november": "11", "nov": "11",
    "december": "12", "dec": "12", "जनवरी": "01", "फ़रवरी": "02",
    "फरवरी": "02", "मार्च": "03", "अप्रैल": "04", "मई": "05",
    "जून": "06", "जुलाई": "07", "अगस्त": "08", "सितंबर": "09",
    "अक्टूबर": "10", "नवंबर": "11", "दिसंबर": "12",
}


def _ascii_digits(text: str) -> str:
    output = []
    for char in text:
        try:
            output.append(str(unicodedata.digit(char)))
        except (TypeError, ValueError):
            output.append(char)
    return "".join(output)


def _tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", _ascii_digits(text)).casefold()
    return re.findall(r"[\w₹]+", normalized, re.UNICODE)


def _number_values(text: str) -> list[int]:
    tokens = _tokens(text)
    values: list[int] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.isdigit():
            values.append(int(token))
        elif token in NUMBER_WORDS:
            value = NUMBER_WORDS[token]
            if value in range(20, 100, 10) and index + 1 < len(tokens):
                following = NUMBER_WORDS.get(tokens[index + 1])
                if following is not None and 0 < following < 10:
                    value += following
                    index += 1
            values.append(value)
        index += 1
    return values


def _month_numbers(text: str) -> set[str]:
    return {MONTHS[token] for token in _tokens(text) if token in MONTHS}


def _normalize(text: str) -> str:
    return " ".join(_tokens(text))


def _alternative_matches(alternative: str, answer: str) -> bool:
    expected = _normalize(alternative)
    actual = _normalize(answer)
    if expected and expected in actual:
        return True

    # Accept spoken or native-script numeric equivalents for factual requirements.
    expected_numbers = _number_values(alternative)
    if expected_numbers:
        actual_numbers = _number_values(answer)
        if not all(number in actual_numbers for number in expected_numbers):
            return False
        expected_months = _month_numbers(alternative)
        return not expected_months or bool(expected_months.intersection(_month_numbers(answer)))
    return False


def _fact_groups(field: str) -> list[list[str]]:
    return [
        [alternative.strip() for alternative in group.split("/") if alternative.strip()]
        for group in field.split(";")
        if group.strip()
    ]


def score_record(record: dict[str, Any]) -> dict[str, Any]:
    answer = str(record.get("reply", record.get("answer", "")))
    required = _fact_groups(str(record.get("must_include", "")))
    forbidden = [item.strip() for item in str(record.get("must_not_include", "")).split(";") if item.strip()]
    missing = [
        "/".join(group)
        for group in required
        if not any(_alternative_matches(alternative, answer) for alternative in group)
    ]
    normalized_answer = _normalize(answer)
    forbidden_hits = [item for item in forbidden if _normalize(item) in normalized_answer]
    expected_decisions = set(record.get("acceptable_decisions") or [])
    if not expected_decisions and record.get("expected_decision"):
        expected_decisions = {str(record["expected_decision"])}
    actual_decision = record.get("decision")
    decision_correct = actual_decision in expected_decisions if actual_decision and expected_decisions else None
    evidence_ids = {
        item.get("evidence_id", item.get("section_id", ""))
        for item in record.get("retrieved_evidence", [])
    }
    gold = set(record.get("gold_evidence") or [])
    evidence_hit = (
        bool(gold.intersection(evidence_ids))
        if gold and "retrieved_evidence" in record
        else None
    )
    passed = not missing and not forbidden_hits and decision_correct is not False
    return {
        "id": record.get("id"),
        "fact_groups": len(required),
        "fact_groups_passed": len(required) - len(missing),
        "missing_facts": missing,
        "forbidden_claims": forbidden_hits,
        "decision_correct": decision_correct,
        "evidence_hit": evidence_hit,
        "approximate_pass": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="JSONL from baseline or question_eval.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()

    records = [json.loads(line) for line in args.results.read_text(encoding="utf-8").splitlines() if line.strip()]
    scores = [score_record(record) for record in records if "error" not in record]
    decision_scores = [item["decision_correct"] for item in scores if item["decision_correct"] is not None]
    evidence_scores = [item["evidence_hit"] for item in scores if item["evidence_hit"] is not None]
    summary = {
        "scoring_note": "Approximate local rubric check. It checks listed facts, literal forbidden phrases, decisions when available, and evidence overlap; it is not the organizers' official scorer.",
        "responses_scored": len(scores),
        "approximate_passes": sum(item["approximate_pass"] for item in scores),
        "approximate_pass_rate": sum(item["approximate_pass"] for item in scores) / max(1, len(scores)),
        "fact_coverage": sum(item["fact_groups_passed"] for item in scores) / max(1, sum(item["fact_groups"] for item in scores)),
        "decision_accuracy": (
            sum(value is True for value in decision_scores) / len(decision_scores)
            if decision_scores else None
        ),
        "retrieval_hit_rate": (
            sum(value is True for value in evidence_scores) / len(evidence_scores)
            if evidence_scores else None
        ),
        "per_question": scores,
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(f"Responses scored: {summary['responses_scored']}")
    print(f"Approximate pass rate: {summary['approximate_pass_rate']:.1%}")
    print(f"Fact coverage: {summary['fact_coverage']:.1%}")
    if args.output:
        print(f"Saved score report to {args.output}")


if __name__ == "__main__":
    main()
