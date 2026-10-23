"""Thin loader for the supplied offline, identity-checked capstone tools."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


def load_capstone_tools(dataset_dir: str | Path) -> ModuleType:
    path = Path(dataset_dir).resolve() / "tools" / "tools.py"
    if not path.is_file():
        raise FileNotFoundError(f"Supplied capstone tools not found: {path}")
    spec = importlib.util.spec_from_file_location("zentromart_capstone_tools", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load supplied tools: {path}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CapstoneToolGateway:
    """Call only the provided lookup and ticket tools."""

    def __init__(self, dataset_dir: str | Path):
        module = load_capstone_tools(dataset_dir)
        self._lookup_order = module.lookup_order
        self._create_ticket = module.create_ticket

    def lookup_order(self, order_id: str, verification_value: str | None = None) -> dict[str, Any]:
        if verification_value is None:
            return self._lookup_order(order_id)
        return self._lookup_order(order_id, verification_value=verification_value)

    def create_ticket(
        self,
        summary: str,
        *,
        order_id: str | None = None,
        reason: str | None = None,
        intent: str | None = None,
    ) -> dict[str, Any]:
        return self._create_ticket(
            summary=summary,
            order_id=order_id,
            reason=reason,
            intent=intent,
        )
