"""CLI wrapper around `tscapeval.reference_builder.build_reference_captions`.

Takes every JSONL under `data/captions/*.jsonl`, runs a strong LLM to produce
a single reference caption per sample, and writes them to a single JSONL.

Usage:
    python scripts/build_references.py \
        --caption-glob 'data/captions/*.jsonl' \
        --output references/gt_captions.jsonl \
        --model gpt-5.5
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tscapeval.io import read_jsonl  # noqa: E402
from tscapeval.reference_builder import build_reference_captions  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--caption-glob", default="data/captions/*.jsonl")
    ap.add_argument("--output", default="references/gt_captions.jsonl")
    ap.add_argument("--model", default=None, help="Defaults to REFERENCE_MODEL env var or gpt-5.5")
    ap.add_argument(
        "--no-resume", action="store_true", help="Overwrite existing references"
    )
    ap.add_argument("--limit", type=int, default=None, help="Cap total samples (testing only)")
    args = ap.parse_args()

    samples: list[dict] = []
    for path in sorted(glob.glob(args.caption_glob)):
        samples.extend(read_jsonl(path))
    if args.limit is not None:
        samples = samples[: args.limit]
    print(f"Generating references for {len(samples)} samples")

    out = build_reference_captions(
        samples,
        args.output,
        model=args.model,
        resume=not args.no_resume,
    )
    print(f"Wrote references to {out}")


if __name__ == "__main__":
    main()
