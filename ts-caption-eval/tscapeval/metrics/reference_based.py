"""Reference-based text similarity: BLEU-4, ROUGE-L, BERTScore-F1."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..evaluator_base import Evaluator
from ..types import EvalResult, Prediction, Reference


class ReferenceBasedEvaluator(Evaluator):
    name = "reference_based"

    def __init__(
        self,
        metrics: list[str] | None = None,
        bertscore_model: str = "roberta-large",
        bertscore_lang: str = "en",
    ):
        self.metrics = metrics or ["bleu4", "rouge_l", "bertscore_f1"]
        self.bertscore_model = bertscore_model
        self.bertscore_lang = bertscore_lang

    def evaluate(
        self,
        predictions: list[Prediction],
        references: list[Reference] | None = None,
        raw=None,
        qa=None,
    ) -> EvalResult:
        if not references:
            raise ValueError("ReferenceBasedEvaluator requires references.")
        ref_by_id = {r.ts_id: r.ref_caption for r in references}
        pairs: list[tuple[str, str, str]] = []  # (ts_id, pred, ref)
        for p in predictions:
            if p.ts_id not in ref_by_id:
                continue
            pairs.append((p.ts_id, p.pred_caption, ref_by_id[p.ts_id]))

        if not pairs:
            return EvalResult(
                evaluator=self.name,
                corpus_scores={},
                notes="no overlap between predictions and references",
            )

        per_sample: dict[str, list[Any]] = {"ts_id": [t[0] for t in pairs]}
        corpus: dict[str, float] = {}

        if "bleu4" in self.metrics:
            bleu_vals = self._bleu4([p for _, p, _ in pairs], [r for _, _, r in pairs])
            per_sample["bleu4"] = bleu_vals
            corpus["bleu4"] = float(np.mean(bleu_vals))
        if "rouge_l" in self.metrics:
            rouge_vals = self._rouge_l([p for _, p, _ in pairs], [r for _, _, r in pairs])
            per_sample["rouge_l"] = rouge_vals
            corpus["rouge_l"] = float(np.mean(rouge_vals))
        if "bertscore_f1" in self.metrics:
            bs_vals = self._bertscore([p for _, p, _ in pairs], [r for _, _, r in pairs])
            per_sample["bertscore_f1"] = bs_vals
            corpus["bertscore_f1"] = float(np.mean(bs_vals))

        return EvalResult(
            evaluator=self.name,
            corpus_scores=corpus,
            per_sample=per_sample,
        )

    # --- metric back-ends -----------------------------------------------

    @staticmethod
    def _bleu4(preds: list[str], refs: list[str]) -> list[float]:
        import sacrebleu

        scores = []
        for p, r in zip(preds, refs):
            # sentence-level BLEU with 4-gram, smoothing exp (sacrebleu default)
            bleu = sacrebleu.sentence_bleu(p, [r]).score / 100.0
            scores.append(float(bleu))
        return scores

    @staticmethod
    def _rouge_l(preds: list[str], refs: list[str]) -> list[float]:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = []
        for p, r in zip(preds, refs):
            s = scorer.score(r, p)  # rouge_score expects (target, prediction)
            scores.append(float(s["rougeL"].fmeasure))
        return scores

    def _bertscore(self, preds: list[str], refs: list[str]) -> list[float]:
        from bert_score import score as bs_score
        import torch

        # 检测是否有可用的GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device for BERTScore: {device}")

        _P, _R, F1 = bs_score(
            preds,
            refs,
            model_type=self.bertscore_model,
            lang=self.bertscore_lang,
            verbose=False,
            rescale_with_baseline=False,
            device=device,
        )
        return [float(x) for x in F1.tolist()]
