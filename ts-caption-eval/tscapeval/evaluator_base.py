"""Abstract base class all evaluators implement."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .types import EvalResult, Prediction, QASample, RawSample, Reference


class Evaluator(ABC):
    """Evaluators consume aligned (predictions, references, raw) triples.

    `references` and `raw` are optional because some evaluators (LLM judge)
    need raw series but not references; others (reference-based) need
    references but not raw; and the downstream QA evaluator needs QA items
    passed in instead of references.
    """

    name: str = "base"

    @abstractmethod
    def evaluate(
        self,
        predictions: list[Prediction],
        references: list[Reference] | None = None,
        raw: list[RawSample] | None = None,
        qa: list[QASample] | None = None,
    ) -> EvalResult: ...
