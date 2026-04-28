# ts-caption-eval

A small, pluggable evaluation framework for **time series captioning** models.

Given a set of time series and one or more caption models, `tscapeval`
produces a side-by-side table over three complementary families of metrics:

| Family              | Metrics                                          | What it measures |
|---------------------|--------------------------------------------------|------------------|
| Reference-based     | BLEU-4, ROUGE-L, BERTScore-F1                    | text similarity to a strong LLM-written reference caption |
| LLM-as-judge        | Faithfulness, Completeness (1–5 Likert, averaged)| whether the caption is consistent with the actual series summary, and whether it covers the key features |
| Downstream QA       | Accuracy under 3 conditions (`meta_only`, `caption`, `wrong_caption`) | whether the caption is *useful* to a frozen answerer on a multiple-choice benchmark (TSShapeQA) |

The three evaluators are independent classes and each can be enabled or
disabled via the YAML config. Reference-based and LLM-judge each expose
their own interface — see `tscapeval/metrics/` — and parsers for
different caption formats are plug-ins registered via a decorator, so a
student evaluating a custom model writes a small parser class instead of
editing the evaluator.

---

## Quick start

```bash
git clone https://github.com/<YOUR_GITHUB_USER>/ts-caption-eval.git
cd ts-caption-eval
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

cp .env.example .env
# Edit .env and set OPENAI_API_KEY (and optionally OPENAI_BASE_URL).

# 1. Smoke-test with the bundled dummy predictions.
python -m tscapeval --config configs/thesis.yaml

# Result tables land in results/thesis_caption_eval/:
#   main_table.md, main_table.csv, main_table.json
#   per_sample/<model>__<evaluator>.json
#   per_dataset/<model>__reference_based.json
```

For the smoke test above, dummy predictions in `predictions/` are included
so the pipeline produces a complete table without any model inference.

## Running on your own caption model

1. Run inference with your model on every entry of:
   - `data/captions/*.jsonl` (the 400 caption samples across 4 dataset families)
   - `data/qa/tsshapeqa_400.jsonl` (the 400 QA-series for the downstream task)

   Emit one JSONL row per time series in the following schema:
   ```json
   {"ts_id": "etth1_ot_w000", "dataset": "ett", "pred_caption": "..."}
   ```
   Put the file under `predictions/<your_model_name>.jsonl`.

2. Add it to `configs/thesis.yaml`:
   ```yaml
   models:
     - name: my_student_model
       predictions_path: predictions/my_student_model.jsonl
       parser: default            # or the name of a custom parser
   ```

3. Run `python -m tscapeval --config configs/thesis.yaml`.

### Writing a custom parser

If your model wraps its output in extra markup (reasoning traces, multiple
sections, bilingual dual-language blocks, …) register a parser:

```python
# tscapeval/parsers/my_parser.py
from ..types import Prediction
from . import Parser, register

@register("my_model")
class MyModelParser(Parser):
    def parse_row(self, row: dict) -> Prediction:
        raw = row["pred_caption"]
        caption = raw.split("### Caption ###")[-1].strip()
        return Prediction(ts_id=row["ts_id"], dataset=row.get("dataset", ""), pred_caption=caption)
```

Then reference it in the config: `parser: my_model`.

## Data

- `data/captions/{fred,ett,nab,ucr}.jsonl` — 100 samples per family, derived
  from FRED Blog data, the ETT-h1 OT column, the NAB Twitter-volume
  subset, and the UCR ECG200 test set respectively.
- `data/qa/tsshapeqa_400.jsonl` — 400 multiple-choice questions sampled
  from TSShapeQA v1 (originally built for the OpenTSLM/ChatTS family of
  TS-LLM studies).
- `references/gt_captions.jsonl` — one reference caption per sample,
  generated once with a strong LLM (`gpt-5.5` by default). Ships with the
  repo so students do not need API access to run reference-based metrics.

All series are embedded directly in the JSONL (as Python lists), so no
additional downloads are needed.

## Regenerating references

If you want to rebuild references with a different model:

```bash
python scripts/build_references.py \
    --output references/gt_captions.jsonl \
    --model gpt-5.5
```

The builder is *resumable* — re-running it skips samples that already have
a non-empty reference in the target file.

## Regenerating the data bundle

The bundled JSONLs are the output of `scripts/sample_source_data.py`, which
consumes the four raw dataset roots (FRED JSONL / ETT CSV / NAB numpy /
UCR TSV) plus TSShapeQA v1. Students normally do not need to re-run it;
the script is included only for transparency.

## Environment variables

| Variable              | Purpose |
|-----------------------|---------|
| `OPENAI_API_KEY`      | API key for any OpenAI-compatible endpoint |
| `OPENAI_BASE_URL`     | Endpoint URL (omit to use real OpenAI) |
| `REFERENCE_MODEL`     | Default model for reference caption generation (default `gpt-5.5`) |
| `JUDGE_MODEL`         | Default model for the LLM-as-judge and the QA answerer (default `gpt-5.4-mini`) |

These are read via the standard OpenAI SDK plus a thin wrapper in
`tscapeval/llm.py`. No URLs or keys are hardcoded anywhere.

## Running tests

```bash
pip install -e '.[dev]'
pytest
```

## Cost estimate

With the defaults in `configs/thesis.yaml` (40 judge samples per model,
80 QA samples per model, 4 models):

- Reference-based metrics: free (local).
- LLM judge: 4 × 40 × 2 dimensions = 320 gpt-5.4-mini calls (≈ \$0.10).
- Downstream QA: 4 × 80 × 3 conditions = 960 gpt-5.4-mini calls (≈ \$0.30).
- Reference caption generation (one-time): 400 gpt-5.5 calls (≈ \$2–5).

Scale `max_samples` up once you have confirmed the pipeline works for you.

## License

MIT — see `LICENSE`.
