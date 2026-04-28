"""Prediction parsers.

A parser turns a raw model-output JSONL row into a canonical
`Prediction(ts_id, dataset, pred_caption)`.

Different caption models emit different formats:
- OpenTSLM-Flamingo: a single short sentence per row.
- ChatTS: may wrap output in "Caption:" + multi-paragraph bullet points.
- Pure-LLM-prompt: often repeats the question prompt or adds reasoning.
- Custom student models: whatever the student decides.

Students add their own parser by subclassing `Parser` and registering it.
"""

from __future__ import annotations

from typing import Callable

from ..types import Prediction


class Parser:
    """Override `parse_row` in subclasses."""

    name: str = "default"

    def parse_row(self, row: dict) -> Prediction:
        return Prediction(
            ts_id=row["ts_id"],
            dataset=row.get("dataset", ""),
            pred_caption=str(row.get("pred_caption", "")).strip(),
        )

    def parse_all(self, rows: list[dict]) -> list[Prediction]:
        return [self.parse_row(r) for r in rows]


_REGISTRY: dict[str, Callable[[], Parser]] = {}


def register(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        cls.name = name
        return cls

    return deco


def get(name: str) -> Parser:
    if name not in _REGISTRY:
        raise KeyError(
            f"Parser '{name}' not registered. "
            f"Known parsers: {sorted(_REGISTRY)}. "
            f"Register yours with @register('my_name')."
        )
    return _REGISTRY[name]()


def list_parsers() -> list[str]:
    return sorted(_REGISTRY)


# Register built-ins.
from .default import DefaultParser  # noqa: E402, F401
from .chatts import ChatTSParser  # noqa: E402, F401
