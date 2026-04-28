# Evaluation Design

This short note explains *why* the evaluation is shaped the way it is, so
you can defend each decision in a thesis defense.

## Problem

A time series captioning model takes a 1-D numeric series and produces a
short natural-language description. There is no single obvious way to
score that description:

- BLEU/ROUGE-style metrics reward *text overlap* with a reference, which
  is how image captioning is usually scored.
- But two very different wordings can be equally faithful to the same
  series; text overlap alone penalizes paraphrase.
- And a caption could be perfectly fluent yet hallucinate numbers,
  trends, or change points that are not actually present.
- Finally, a *useful* caption should help a downstream reader (human or
  LLM) answer questions about the series that they could not answer from
  the series meta-info alone.

Rather than pick one metric, `ts-caption-eval` evaluates along three
complementary axes.

## Axis 1 — Reference-based text similarity

Generate a strong reference caption once per sample with `gpt-5.5` (or
any LLM you want to fix as the "oracle"), then score each caption model
against the reference using:

- **BLEU-4** (`sacrebleu`) — n-gram precision with brevity penalty.
- **ROUGE-L** (`rouge-score`) — longest-common-subsequence F1.
- **BERTScore-F1** (`bert-score`, RoBERTa-large backbone) — contextual
  embedding similarity. Catches semantic paraphrase that BLEU misses.

**Limit.** If the caption is very different from the reference but still
correct (different structure, different emphasis), these metrics under-
rate it. Use them as one signal, not the final verdict.

## Axis 2 — LLM-as-judge

For each sample we feed a compact *statistical summary* of the series
(mean, std, argmin, argmax, slope, first-vs-second-half volatility, top
change points) plus the candidate caption to a judge LLM, and ask it to
score two dimensions on a 1–5 Likert scale:

- **Faithfulness** — does the caption contradict anything the summary
  tells us?
- **Completeness** — does it cover the key features (trend, extrema,
  volatility region, notable change points)?

We feed the summary and not the raw array for three reasons:

1. The prompt stays short and cheap (`gpt-5.4-mini` is the default judge).
2. The judge's reasoning is grounded in a deterministic feature
   extraction, so scores are reproducible across runs.
3. A judge that only saw raw numbers would itself hallucinate — the
   same failure mode we are trying to catch.

**Limit.** LLM-judge scores are correlated across dimensions and
sensitive to prompt wording. Report the rubric in full (see
`tscapeval/metrics/llm_judge.py`) so they are reproducible.

## Axis 3 — Downstream QA

Run the caption through a frozen answerer LLM on the TSShapeQA
multiple-choice benchmark, under three conditions:

- `meta_only` — answerer sees only the dataset description (a floor).
- `caption`   — answerer sees the candidate caption on top of that.
- `wrong_caption` — answerer sees a caption from a *different* sample
  (a ceiling for how much of the task is solvable by caption-style
  hallucination alone).

If captions contain real signal, we expect
`caption > meta_only > wrong_caption`. The delta
`acc_caption − acc_meta_only` is the clean utility signal.

**Limit.** This only probes the aspects of the caption the QA benchmark
itself tests (shape primitives: trend, extrema location, volatility
region). Captions that are correct about other aspects do not benefit.

## Why these three together

Each axis has a failure mode that the other two catch:

| A high-scoring caption under … | … can still fail because … |
|---|---|
| Reference metrics only      | … it may be a faithful paraphrase of the wrong series (text can match while facts are hallucinated). |
| LLM judge only              | … judge can mistake verbose caption for completeness; gaming is possible. |
| Downstream QA only          | … a caption that just repeats the question topic may answer well without describing the series. |

A caption model that wins on all three axes on a diverse 4-family
sample is a safer bet than one that wins on any single axis.

## Dataset mix

Four families, 100 samples each, picked to cover the kinds of series a
general-purpose TS caption model will meet:

- **FRED** — economic indicators, real-world, irregular amplitude.
- **ETT** — sensor data, strong daily/weekly seasonality.
- **NAB** — social-media counts with labelled anomalies (volatility
  structure, sparse events).
- **UCR ECG200** — short biomedical waveforms, classification structure.

TSShapeQA provides a 400-question downstream benchmark whose ground
truth is *algorithmically* derived from the series, so scoring is
unambiguous.

## What is deliberately *not* in scope

- Human preference rating — worth doing for a full paper, out of scope
  for a one-semester thesis eval pipeline.
- Language-specific metrics (METEOR / BLEURT) — diminishing returns on
  top of BERTScore in this setting.
- End-to-end model training — this repo evaluates predictions, not
  models. Plug your trained model's outputs in as a prediction JSONL.
