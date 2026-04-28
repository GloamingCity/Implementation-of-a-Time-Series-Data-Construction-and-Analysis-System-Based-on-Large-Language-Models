"""Typed containers for predictions, references, raw samples, and QA items."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RawSample:
    ts_id: str
    dataset: str
    series: list[float]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    ts_id: str
    dataset: str
    pred_caption: str


@dataclass
class Reference:
    ts_id: str
    dataset: str
    ref_caption: str
    source: str = ""


@dataclass
class QASample:
    qa_id: str
    ts_id: str
    dataset: str
    question: str
    options: list[str]
    answer: str
    series: list[float]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Output of an Evaluator.

    corpus_scores : macro-averaged scalar metrics (one per metric name).
    per_sample    : dict of lists, aligned with input order, for
                    per-sample drill-down / case studies.
    """

    evaluator: str
    corpus_scores: dict[str, float]
    per_sample: dict[str, list[Any]] = field(default_factory=dict)
    notes: str = ""
