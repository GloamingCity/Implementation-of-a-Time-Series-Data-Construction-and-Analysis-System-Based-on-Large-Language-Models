"""CLI runner: YAML config in, tables out."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from . import parsers
from .io import read_jsonl
from .metrics import (
    DownstreamQAEvaluator,
    LLMJudgeEvaluator,
    ReferenceBasedEvaluator,
)
from .types import QASample, RawSample, Reference


def _load_raw(data_cfg: list[dict], n_per_dataset: int | None) -> list[RawSample]:
    out: list[RawSample] = []
    for d in data_cfg:
        rows = read_jsonl(d["path"])
        if n_per_dataset is not None:
            rows = rows[:n_per_dataset]
        for r in rows:
            out.append(
                RawSample(
                    ts_id=r["ts_id"],
                    dataset=r.get("dataset", d.get("name", "")),
                    series=r.get("series", []),
                    meta=r.get("meta", {}),
                )
            )
    return out


def _load_references(path: str) -> list[Reference]:
    return [
        Reference(
            ts_id=r["ts_id"],
            dataset=r.get("dataset", ""),
            ref_caption=r.get("ref_caption", ""),
            source=r.get("source", ""),
        )
        for r in read_jsonl(path)
    ]


def _load_qa(path: str) -> list[QASample]:
    return [
        QASample(
            qa_id=r["qa_id"],
            ts_id=r["ts_id"],
            dataset=r.get("dataset", ""),
            question=r["question"],
            options=r["options"],
            answer=r["answer"],
            series=r.get("series", []),
            meta=r.get("meta", {}),
        )
        for r in read_jsonl(path)
    ]


def _load_predictions(path: str, parser_name: str):
    parser = parsers.get(parser_name)
    return parser.parse_all(read_jsonl(path))


def _dataset_of(ts_id: str, raw_index: dict[str, RawSample]) -> str:
    r = raw_index.get(ts_id)
    return r.dataset if r else ""


def _write_outputs(result_rows: list[dict], out_dir: Path, formats: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Preserve insertion order per row, but put model/n_* columns first
    # and metric columns after.
    seen: list[str] = []
    for r in result_rows:
        for k in r.keys():
            if k not in seen:
                seen.append(k)
    preamble = [c for c in ("model", "n_caption_preds", "n_total_preds", "n_samples") if c in seen]
    rest = [c for c in seen if c not in preamble]
    columns = preamble + rest

    if "json" in formats:
        (out_dir / "main_table.json").write_text(
            json.dumps(result_rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if "csv" in formats:
        with (out_dir / "main_table.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            for r in result_rows:
                w.writerow({k: r.get(k, "") for k in columns})
    if "md" in formats:
        lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
        for r in result_rows:
            cells = []
            for c in columns:
                v = r.get(c, "")
                if isinstance(v, float):
                    cells.append(f"{v:.4f}")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        (out_dir / "main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: dict) -> list[dict]:
    exp_name = config.get("experiment_name", "tscapeval_run")
    n_per_dataset = int(config.get("n_per_dataset", 100))

    raw = _load_raw(config["datasets"], n_per_dataset)
    raw_index = {r.ts_id: r for r in raw}

    ref_cfg = config.get("reference", {})
    references: list[Reference] = []
    ref_path = ref_cfg.get("path")
    if ref_path and Path(ref_path).exists():
        references = _load_references(ref_path)
    elif config.get("evaluators", {}).get("reference_based", {}).get("enabled"):
        raise FileNotFoundError(
            f"Reference file not found: {ref_path}. "
            f"Run `python -m tscapeval.reference_builder` to build it first."
        )

    qa: list[QASample] = []
    qa_path = config.get("evaluators", {}).get("downstream_qa", {}).get("qa_path")
    if qa_path:
        qa = _load_qa(qa_path)

    out_dir = Path(config.get("output", {}).get("dir", f"results/{exp_name}"))
    formats = config.get("output", {}).get("format", ["md", "csv", "json"])

    evaluators_cfg = config.get("evaluators", {})

    result_rows: list[dict] = []
    for m in config["models"]:
        name = m["name"]
        all_preds = _load_predictions(m["predictions_path"], m.get("parser", "default"))
        # Caption-side predictions: those we have a raw sample for.
        caption_preds = [p for p in all_preds if p.ts_id in raw_index]
        for p in caption_preds:
            if not p.dataset:
                p.dataset = _dataset_of(p.ts_id, raw_index)

        row: dict[str, Any] = {
            "model": name,
            "n_caption_preds": len(caption_preds),
            "n_total_preds": len(all_preds),
        }

        if evaluators_cfg.get("reference_based", {}).get("enabled"):
            cfg = evaluators_cfg["reference_based"]
            ev = ReferenceBasedEvaluator(metrics=cfg.get("metrics"))
            r = ev.evaluate(caption_preds, references=references)
            row.update(r.corpus_scores)
            _dump_per_sample(out_dir, name, r.evaluator, r.per_sample)
            _dump_per_dataset(out_dir, name, r, caption_preds)

        if evaluators_cfg.get("llm_judge", {}).get("enabled"):
            cfg = evaluators_cfg["llm_judge"]
            ev = LLMJudgeEvaluator(
                model=cfg.get("model"),
                dimensions=cfg.get("dimensions"),
                max_samples=cfg.get("max_samples"),
            )
            r = ev.evaluate(caption_preds, raw=raw)
            row.update(r.corpus_scores)
            _dump_per_sample(out_dir, name, r.evaluator, r.per_sample)
            _dump_per_dataset(out_dir, name, r, caption_preds)

        if evaluators_cfg.get("downstream_qa", {}).get("enabled"):
            cfg = evaluators_cfg["downstream_qa"]
            ev = DownstreamQAEvaluator(
                answerer_model=cfg.get("answerer_model"),
                conditions=cfg.get("conditions"),
                max_samples=cfg.get("max_samples"),
            )
            # Use all_preds so QA-side ts_ids (not in raw_index) are visible.
            r = ev.evaluate(all_preds, qa=qa)
            row.update(r.corpus_scores)
            _dump_per_sample(out_dir, name, r.evaluator, r.per_sample)

        result_rows.append(row)

    _write_outputs(result_rows, out_dir, formats)
    return result_rows


def _dump_per_sample(out_dir: Path, model: str, evaluator: str, per_sample: dict) -> None:
    safe = model.replace("/", "_")
    target = out_dir / "per_sample" / f"{safe}__{evaluator}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(per_sample, ensure_ascii=False, indent=2), encoding="utf-8")


def _dump_per_dataset(out_dir: Path, model: str, result, preds) -> None:
    ts_to_ds = {p.ts_id: p.dataset for p in preds}
    per_ds: dict[str, dict[str, list[float]]] = {}
    ids = result.per_sample.get("ts_id", [])
    for metric, vals in result.per_sample.items():
        if metric == "ts_id":
            continue
        for tid, v in zip(ids, vals):
            ds = ts_to_ds.get(tid, "")
            per_ds.setdefault(ds, {}).setdefault(metric, []).append(v)
    summary = {
        ds: {m: (sum(vs) / len(vs) if vs else 0.0) for m, vs in d.items()}
        for ds, d in per_ds.items()
    }
    safe = model.replace("/", "_")
    target = out_dir / "per_dataset" / f"{safe}__{result.evaluator}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser("tscapeval")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    rows = run(config)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
