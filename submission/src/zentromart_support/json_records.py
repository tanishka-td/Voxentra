"""Read legacy JSONL records and write readable JSON arrays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read a JSON array/object or legacy newline-delimited JSON records."""
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not all(isinstance(record, dict) for record in records):
            raise ValueError(f"Expected JSON objects in {path}")
        return records
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list) and all(isinstance(record, dict) for record in value):
        return value
    raise ValueError(f"Expected a JSON object, array of objects, or JSONL records in {path}")


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    """Atomically write records as an indented JSON array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
