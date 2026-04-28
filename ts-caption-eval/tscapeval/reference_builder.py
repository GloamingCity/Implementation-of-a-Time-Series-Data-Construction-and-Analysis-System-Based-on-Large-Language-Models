"""Build reference captions for each raw sample using a strong LLM."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from .io import index_by, read_jsonl, write_jsonl
from .llm import chat
from .stats import format_stats_for_prompt, summarize_series
from .types import RawSample


_PROMPT = """You are writing a concise, faithful caption for a time series.

[Series summary]
{stats}

[Context]
Dataset: {dataset}
Additional info: {meta}

Write a single caption (3-5 sentences) that describes:
1. The overall trend (upward / downward / flat / non-monotonic).
2. The approximate location of the maximum and minimum.
3. Whether volatility is concentrated in the first or second half.
4. Any notable change points if present.

Do NOT fabricate numbers beyond what is supplied. Keep the tone neutral and
scientific. Return ONLY the caption text, no prefix, no bullet list.
"""


def build_reference_captions(
    raw_samples: Iterable[RawSample] | Iterable[dict],
    output_path: str | Path,
    *,
    model: str | None = None,
    resume: bool = True,
) -> Path:
    """Generate one reference caption per raw sample; write to JSONL.

    Resumes from an existing output file if `resume=True`.
    """
    model = model or os.environ.get("REFERENCE_MODEL", "gpt-5.5")
    output_path = Path(output_path)

    done: dict[str, dict] = {}
    if resume and output_path.exists():
        done = index_by(read_jsonl(output_path), "ts_id")

    samples: list[RawSample] = []
    for r in raw_samples:
        if isinstance(r, dict):
            samples.append(
                RawSample(
                    ts_id=r["ts_id"],
                    dataset=r.get("dataset", ""),
                    series=r.get("series", []),
                    meta=r.get("meta", {}),
                )
            )
        else:
            samples.append(r)

    rows: list[dict] = list(done.values())
    new_ids: list[str] = []

    try:
        for sample in tqdm(samples, desc=f"ref-captions[{model}]"):
            if sample.ts_id in done:
                continue
            stats = summarize_series(sample.series)
            prompt = _PROMPT.format(
                stats=format_stats_for_prompt(stats),
                dataset=sample.dataset,
                meta=str(sample.meta.get("meta_info") or sample.meta)[:400],
            )
            try:
                caption = chat(
                    model,
                    [{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=300,
                ).strip()
            except Exception as e:  # noqa: BLE001
                caption = ""
                print(f"[warn] {sample.ts_id} failed: {e}")
            rows.append(
                {
                    "ts_id": sample.ts_id,
                    "dataset": sample.dataset,
                    "ref_caption": caption,
                    "source": model,
                }
            )
            new_ids.append(sample.ts_id)
            # Periodic flush so a crash doesn't lose progress.
            if len(new_ids) % 25 == 0:
                write_jsonl(output_path, rows)
    finally:
        write_jsonl(output_path, rows)

    return output_path
