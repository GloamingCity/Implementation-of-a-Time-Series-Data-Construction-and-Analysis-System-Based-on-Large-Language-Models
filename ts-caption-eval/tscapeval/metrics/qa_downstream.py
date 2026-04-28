"""Downstream QA evaluator.

Feeds the predicted caption (or a null/negative-control variant) to a frozen
LLM answerer and scores accuracy on the TSShapeQA subset. Supports the
three core conditions:

- meta_only     : answerer sees dataset meta_info only (baseline floor)
- caption       : answerer sees the predicted caption
- wrong_caption : answerer sees a shuffled caption from another sample
                  (negative control; caption info should hurt if valuable)

If a caption model adds real signal, `caption > meta_only > wrong_caption`.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
from tqdm import tqdm

from ..evaluator_base import Evaluator
from ..llm import chat
from ..types import EvalResult, Prediction, QASample


_ANSWER_PROMPT = """You are answering a multiple-choice question about a time series you cannot see directly. You MUST pick exactly one option letter (A / B / C / D).

[Context]
{context}

[Question]
{question}

[Options]
{options}

Respond with the single letter only."""


class DownstreamQAEvaluator(Evaluator):
    name = "downstream_qa"

    def __init__(
        self,
        answerer_model: str | None = None,
        conditions: list[str] | None = None,
        max_samples: int | None = None,
        seed: int = 0,
    ):
        self.answerer_model = answerer_model or os.environ.get(
            "JUDGE_MODEL", "gpt-5.4-mini"
        )
        self.conditions = conditions or ["meta_only", "caption", "wrong_caption"]
        self.max_samples = max_samples
        self.seed = seed

    def evaluate(
        self,
        predictions: list[Prediction],
        references=None,
        raw=None,
        qa: list[QASample] | None = None,
    ) -> EvalResult:
        if not qa:
            raise ValueError("DownstreamQAEvaluator requires `qa` samples.")
        pred_by_ts = {p.ts_id: p.pred_caption for p in predictions}

        items = [q for q in qa if q.ts_id in pred_by_ts]
        if self.max_samples is not None:
            items = items[: self.max_samples]

        rng = random.Random(self.seed)
        shuffled_captions = [pred_by_ts[q.ts_id] for q in items]
        rng.shuffle(shuffled_captions)

        per_sample: dict[str, list[Any]] = {"qa_id": [q.qa_id for q in items]}
        for cond in self.conditions:
            per_sample[f"pred_{cond}"] = []
            per_sample[f"correct_{cond}"] = []

        for idx, q in enumerate(
            tqdm(items, desc=f"downstream-qa[{self.answerer_model}]", leave=False)
        ):
            options_blob = "\n".join(q.options)
            for cond in self.conditions:
                context = _build_context(
                    cond,
                    meta=q.meta.get("meta_info") or q.meta.get("domain", ""),
                    caption=pred_by_ts.get(q.ts_id, ""),
                    wrong_caption=shuffled_captions[idx],
                )
                prompt = _ANSWER_PROMPT.format(
                    context=context, question=q.question, options=options_blob
                )
                try:
                    raw_answer = chat(
                        self.answerer_model,
                        [{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=8192,
                    )
                    pred_letter = _extract_letter(raw_answer)
                except Exception:  # noqa: BLE001
                    pred_letter = ""
                per_sample[f"pred_{cond}"].append(pred_letter)
                per_sample[f"correct_{cond}"].append(int(pred_letter == q.answer))

        corpus = {
            f"acc_{cond}": float(np.mean(per_sample[f"correct_{cond}"]))
            if per_sample[f"correct_{cond}"]
            else 0.0
            for cond in self.conditions
        }
        if "caption" in self.conditions and "meta_only" in self.conditions:
            corpus["delta_vs_meta_only"] = corpus["acc_caption"] - corpus["acc_meta_only"]
        return EvalResult(
            evaluator=self.name,
            corpus_scores=corpus,
            per_sample=per_sample,
        )


def _build_context(cond: str, *, meta: str, caption: str, wrong_caption: str) -> str:
    if cond == "meta_only":
        return f"Dataset description: {meta or '(none)'}"
    if cond == "caption":
        return (
            f"Dataset description: {meta or '(none)'}\n"
            f"Caption of the series: {caption or '(none)'}"
        )
    if cond == "wrong_caption":
        return (
            f"Dataset description: {meta or '(none)'}\n"
            f"Caption of the series: {wrong_caption or '(none)'}"
        )
    raise ValueError(f"Unknown condition: {cond}")


def _extract_letter(raw: str) -> str:
    """Robustly extract an A/B/C/D letter from an LLM answer string."""
    raw = (raw or "").strip().upper()
    for ch in raw:
        if ch in "ABCD":
            return ch
    return ""
