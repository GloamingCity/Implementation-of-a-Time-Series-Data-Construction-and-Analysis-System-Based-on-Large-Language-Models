"""LLM-as-judge evaluator.

The judge sees a compact statistical summary of the series (computed via
`tscapeval.stats`) — not the raw array — and scores the caption on:
- faithfulness: are the numeric/shape claims correct?
- completeness: does the caption cover the key features (trend, extrema,
  volatility regions)?

Both on a 1-5 Likert with a short free-text reason. We average to a scalar
per dimension for the table.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
from tqdm import tqdm

from ..evaluator_base import Evaluator
from ..llm import chat_json
from ..stats import format_stats_for_prompt, summarize_series
from ..types import EvalResult, Prediction, RawSample


_RUBRIC_TEMPLATE = """You are a meticulous evaluator of time series captions.

[Time series summary]
{stats}

[Caption to evaluate]
"{caption}"

Score the caption on a single dimension: {dimension}.

{rubric}

Return a compact JSON object: {{"score": <int 1-5>, "reason": "<one short sentence>"}}.
"""

_RUBRICS = {
    "faithfulness": (
        "Faithfulness = are all numeric/shape claims consistent with the summary?\n"
        "5 = all claims correct; no hallucination\n"
        "4 = minor issues (vague phrasing or one slightly-off number)\n"
        "3 = two to three clearly wrong claims but overall direction correct\n"
        "2 = most claims wrong\n"
        "1 = contradicts the actual series"
    ),
    "completeness": (
        "Completeness = does the caption cover the key features "
        "(overall trend, location of extrema, volatility regions, "
        "notable change points)?\n"
        "5 = covers all key features\n"
        "4 = covers most, minor omission\n"
        "3 = covers about half\n"
        "2 = misses most key features\n"
        "1 = essentially empty or generic"
    ),
}


class LLMJudgeEvaluator(Evaluator):
    name = "llm_judge"

    def __init__(
        self,
        model: str | None = None,
        dimensions: list[str] | None = None,
        max_samples: int | None = None,
    ):
        self.model = model or os.environ.get("JUDGE_MODEL", "gpt-5.4-mini")
        self.dimensions = dimensions or ["faithfulness", "completeness"]
        self.max_samples = max_samples
        for d in self.dimensions:
            if d not in _RUBRICS:
                raise ValueError(f"Unknown judge dimension: {d}")

    def evaluate(
        self,
        predictions: list[Prediction],
        references=None,
        raw: list[RawSample] | None = None,
        qa=None,
    ) -> EvalResult:
        if not raw:
            raise ValueError("LLMJudgeEvaluator requires `raw` samples for stats.")
        raw_by_id = {r.ts_id: r for r in raw}

        items = [p for p in predictions if p.ts_id in raw_by_id]
        if self.max_samples is not None:
            items = items[: self.max_samples]

        per_sample: dict[str, list[Any]] = {"ts_id": [p.ts_id for p in items]}
        for d in self.dimensions:
            per_sample[d] = []
            per_sample[f"{d}_reason"] = []

        for p in tqdm(items, desc=f"llm-judge[{self.model}]", leave=False):
            stats = summarize_series(raw_by_id[p.ts_id].series)
            stats_blob = format_stats_for_prompt(stats)
            for d in self.dimensions:
                prompt = _RUBRIC_TEMPLATE.format(
                    stats=stats_blob,
                    caption=p.pred_caption,
                    dimension=d,
                    rubric=_RUBRICS[d],
                )
                score, reason = self._judge_with_retry(prompt, max_retries=5)
                per_sample[d].append(score)
                per_sample[f"{d}_reason"].append(reason)

    def _judge_with_retry(self, prompt: str, max_retries: int = 5) -> tuple[int, str]:
        """反复请求直到获得有效分数（1-5），而非技术错误导致的0分。"""
        for attempt in range(max_retries):
            try:
                obj = chat_json(
                    self.model,
                    [{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=8192,
                )
                score = int(obj.get("score", 0))
                reason = str(obj.get("reason", ""))[:200]
                
                # 分数在1-5范围内说明是有效评分
                if 1 <= score <= 5:
                    return score, reason
                
                # 分数不在1-5范围内，继续重试
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                
                # 所有重试都失败，返回0
                return 0, f"invalid_score_after_{max_retries}_retries: score={score}"
                
            except Exception as e:  # noqa: BLE001
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                return 0, f"judge_error: {type(e).__name__}"
        
        return 0, "judge_error: all retries failed"

        corpus = {
            d: float(np.mean([s for s in per_sample[d] if s > 0]) if per_sample[d] else 0.0)
            for d in self.dimensions
        }
        return EvalResult(
            evaluator=self.name,
            corpus_scores=corpus,
            per_sample=per_sample,
        )
