"""Generate dummy prediction JSONLs for 4 models, for smoke-testing the eval.

Each model produces a caption for every caption sample AND every QA series.
The captions are deliberately simple and differentiated so metric outputs
vary between models — the goal is to verify the pipeline end-to-end, NOT
to produce realistic results. Replace these with your real model outputs
before reporting anything.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tscapeval.io import read_jsonl  # noqa: E402
from tscapeval.stats import summarize_series  # noqa: E402


def _template_caption(series: list[float], style: str, rng: random.Random) -> str:
    s = summarize_series(series)
    trend = s.get("trend", "flat")
    mean = s.get("mean", 0.0)
    amax = s.get("argmax", 0)
    amin = s.get("argmin", 0)
    length = s.get("length", 0)
    first_std = s.get("first_half_std", 0.0)
    second_std = s.get("second_half_std", 0.0)
    vol_region = "first half" if first_std > second_std else "second half"

    if style == "opentslm":
        # Short, factual, occasionally drops details.
        return (
            f"The series of length {length} has {trend} trend with mean {mean:.2f}. "
            f"Maximum at t={amax}, minimum at t={amin}."
        )
    if style == "chatts":
        # ChatTS-style verbose, multi-section. Parser should strip to first block.
        return (
            f"Caption:\n"
            f"Over {length} points the series exhibits a {trend} trend centered "
            f"around {mean:.2f}. The peak occurs near index {amax} and the trough "
            f"near index {amin}. Volatility is concentrated in the {vol_region}.\n\n"
            f"Analysis:\n- Trend: {trend}\n- Argmax: {amax}\n- Argmin: {amin}"
        )
    if style == "pure_prompt":
        # Pure LLM without TS encoder often hallucinates vague language.
        direction = rng.choice(["upward", "downward", "flat", "mixed"])
        return (
            f"The time series appears to show a {direction} pattern with some "
            f"fluctuations throughout the observed period."
        )
    if style == "student":
        # Student model: faithful and covers all four features.
        return (
            f"A {length}-point series with a {trend} trend (mean {mean:.2f}). "
            f"It attains its maximum around t={amax} and minimum around t={amin}. "
            f"Volatility is higher in the {vol_region}, with first-half std "
            f"{first_std:.2f} vs second-half std {second_std:.2f}."
        )
    raise ValueError(style)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--caption-glob", default="data/captions")
    ap.add_argument("--qa-path", default="data/qa/tsshapeqa_400.jsonl")
    ap.add_argument("--out-dir", type=Path, default=Path("predictions"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    caption_rows: list[dict] = []
    for path in sorted(Path(args.caption_glob).glob("*.jsonl")):
        caption_rows.extend(read_jsonl(path))
    qa_rows = read_jsonl(args.qa_path)

    all_items: list[dict] = []
    for r in caption_rows:
        all_items.append({"ts_id": r["ts_id"], "dataset": r["dataset"], "series": r["series"]})
    for r in qa_rows:
        all_items.append({"ts_id": r["ts_id"], "dataset": r["dataset"], "series": r["series"]})

    styles = {
        "student_qwen3_4b_tsenc": "student",
        "opentslm_flamingo_vars1": "opentslm",
        "chatts_14b": "chatts",
        "pure_qwen3_4b_prompt": "pure_prompt",
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, style in styles.items():
        out_path = args.out_dir / f"{name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for item in all_items:
                cap = _template_caption(item["series"], style, rng)
                f.write(
                    json.dumps(
                        {
                            "ts_id": item["ts_id"],
                            "dataset": item["dataset"],
                            "pred_caption": cap,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"  wrote {out_path} ({len(all_items)} rows)")


if __name__ == "__main__":
    main()
