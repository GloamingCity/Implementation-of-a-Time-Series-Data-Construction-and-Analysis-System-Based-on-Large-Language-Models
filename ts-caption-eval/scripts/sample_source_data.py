"""Build the bundled evaluation data from raw source datasets.

This script is provided for provenance only. It turns raw source data (FRED
blog JSONL, ETTh1 CSV, the NAB Twitter-volume subset stored as numpy, and
the UCR 2018 ECG200 .tsv files, plus TSShapeQA v1 JSONL) into the JSONL
files committed under `data/`.

You normally do NOT need to run this — the pre-built JSONLs are shipped with
the repo. It is included so that the data pipeline is fully transparent.

Usage:
    python scripts/sample_source_data.py \
        --fred /path/to/fred_blog.jsonl \
        --ett  /path/to/ETTh1.csv \
        --nab-root /path/to/NAB \
        --ucr-root /path/to/UCRArchive_2018 \
        --tsshapeqa /path/to/tsshapeqa_v1.jsonl \
        --out-dir ./data \
        --n-per-dataset 100 \
        --n-qa 400 \
        --seed 0

The `--nab-root` directory should contain subfolders like `Twitter_volume_AAPL/`
with `test.npy`, `test_label.npy`, `info.json`.
The `--ucr-root` directory is the standard UCR Archive 2018 root with a
subdirectory per dataset containing `{NAME}_TEST.tsv`. This script uses
`ECG200` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np


def sample_fred(src: Path, n: int, rng: random.Random) -> list[dict]:
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng.shuffle(rows)
    out = []
    for r in rows:
        if len(out) >= n:
            break
        # Use the first timeseries; drop short ones.
        series = r.get("timeseries1") or []
        if len(series) < 32:
            continue
        out.append(
            {
                "ts_id": f"fred_{r['index']}_ts1",
                "dataset": "fred",
                "series": series,
                "meta": {
                    "variable": (r.get("cols") or [""])[0],
                    "meta_info": r.get("message", "")[:600],
                    "frequency_hint": r.get("question", ""),
                    "timestamps_head": (r.get("timestamp") or [])[:3],
                },
            }
        )
    return out


def sample_ett(src: Path, n: int, window_len: int = 128, stride: int = 128) -> list[dict]:
    """Sample `n` non-overlapping windows from the OT column of ETTh1."""
    with src.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    ot = [float(r["OT"]) for r in rows]
    timestamps = [r["date"] for r in rows]
    out = []
    i = 0
    while len(out) < n and i + window_len <= len(ot):
        window = ot[i : i + window_len]
        out.append(
            {
                "ts_id": f"etth1_ot_w{len(out):03d}",
                "dataset": "ett",
                "series": window,
                "meta": {
                    "target_col": "OT",
                    "window_len": window_len,
                    "meta_info": (
                        "Electricity Transformer Temperature hourly data, "
                        "target is Oil Temperature (OT)."
                    ),
                    "start_time": timestamps[i],
                    "end_time": timestamps[i + window_len - 1],
                },
            }
        )
        i += stride
    return out


def sample_nab(src: Path, n_total: int, rng: random.Random, window_len: int = 256) -> list[dict]:
    series_dirs = sorted([p for p in src.iterdir() if p.is_dir()])
    if not series_dirs:
        raise RuntimeError(f"No NAB series found under {src}")
    per = max(1, n_total // len(series_dirs))
    out = []
    for sdir in series_dirs:
        arr = np.load(sdir / "test.npy")
        lab = np.load(sdir / "test_label.npy")
        if arr.size < window_len:
            continue
        max_start = arr.size - window_len
        starts = sorted(rng.sample(range(max_start + 1), k=min(per, max_start + 1)))
        for s in starts:
            if len(out) >= n_total:
                break
            window = arr[s : s + window_len]
            lab_win = lab[s : s + window_len]
            out.append(
                {
                    "ts_id": f"nab_{sdir.name}_w{s:05d}",
                    "dataset": "nab",
                    "series": [float(x) for x in window.tolist()],
                    "meta": {
                        "series_name": sdir.name,
                        "meta_info": (
                            f"NAB Twitter volume for {sdir.name.replace('Twitter_volume_', '')}, "
                            f"5-minute cadence."
                        ),
                        "n_anomaly_points": int(lab_win.sum()),
                        "anomaly_ratio": float(lab_win.mean()),
                        "window_start_idx": int(s),
                        "window_len": int(window_len),
                    },
                }
            )
        if len(out) >= n_total:
            break
    return out


def sample_ucr_ecg200(root: Path, n: int) -> list[dict]:
    tsv = root / "ECG200" / "ECG200_TEST.tsv"
    out = []
    with tsv.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.strip().split("\t")
            if not parts:
                continue
            label = int(float(parts[0]))
            series = [float(x) for x in parts[1:]]
            out.append(
                {
                    "ts_id": f"ucr_ecg200_test_{i:03d}",
                    "dataset": "ucr",
                    "series": series,
                    "meta": {
                        "ucr_dataset": "ECG200",
                        "class_label": int(label),
                        "class_name": "Normal" if label == 1 else "Myocardial_Infarction",
                        "meta_info": (
                            "UCR ECG200: single-heartbeat ECG, two classes - "
                            "Normal and Myocardial Infarction."
                        ),
                    },
                }
            )
            if len(out) >= n:
                break
    return out


def sample_tsshapeqa(src: Path, n: int, rng: random.Random) -> list[dict]:
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng.shuffle(rows)
    out = []
    for r in rows[:n]:
        out.append(
            {
                "qa_id": r["id"],
                "ts_id": r["id"],
                "dataset": "tsshapeqa_v1",
                "question": r["question"],
                "options": r["options"],
                "answer": r["answer"],
                "series": r["series"],
                "meta": {
                    "qa_type": r.get("qa_type", ""),
                    "domain": r.get("domain", ""),
                    "target_col": r.get("target_col", ""),
                    "meta_info": r.get("meta_info", ""),
                    "window_len": r.get("window_len", 0),
                },
            }
        )
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fred", required=True, type=Path)
    ap.add_argument("--ett", required=True, type=Path)
    ap.add_argument("--nab-root", required=True, type=Path)
    ap.add_argument("--ucr-root", required=True, type=Path)
    ap.add_argument("--tsshapeqa", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--n-per-dataset", type=int, default=100)
    ap.add_argument("--n-qa", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    write_jsonl(args.out_dir / "captions" / "fred.jsonl", sample_fred(args.fred, args.n_per_dataset, rng))
    write_jsonl(args.out_dir / "captions" / "ett.jsonl", sample_ett(args.ett, args.n_per_dataset))
    write_jsonl(
        args.out_dir / "captions" / "nab.jsonl",
        sample_nab(args.nab_root, args.n_per_dataset, rng),
    )
    write_jsonl(args.out_dir / "captions" / "ucr.jsonl", sample_ucr_ecg200(args.ucr_root, args.n_per_dataset))
    write_jsonl(
        args.out_dir / "qa" / "tsshapeqa_400.jsonl",
        sample_tsshapeqa(args.tsshapeqa, args.n_qa, rng),
    )
    print("Wrote bundles to", args.out_dir)


if __name__ == "__main__":
    main()
