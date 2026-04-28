"""Smoke test: reference-based metrics only (no network).

Uses the bundled dummy predictions and a tiny synthetic reference file,
so it runs in seconds and requires no API credentials.
"""

from __future__ import annotations

import json
from pathlib import Path

from tscapeval.io import read_jsonl, write_jsonl
from tscapeval.metrics import ReferenceBasedEvaluator
from tscapeval.parsers import get as get_parser
from tscapeval.types import Reference


REPO = Path(__file__).resolve().parents[1]


def test_parser_registry():
    assert "default" in [p for p in __import__("tscapeval.parsers", fromlist=["list_parsers"]).list_parsers()]
    assert "chatts" in __import__("tscapeval.parsers", fromlist=["list_parsers"]).list_parsers()


def test_default_parser_strips_whitespace():
    p = get_parser("default")
    pred = p.parse_row({"ts_id": "a", "dataset": "x", "pred_caption": "  hello  \n"})
    assert pred.pred_caption == "hello"


def test_chatts_parser_strips_caption_prefix():
    p = get_parser("chatts")
    pred = p.parse_row(
        {"ts_id": "a", "dataset": "x", "pred_caption": "Caption:\nHello world.\n\nAnalysis: ..."}
    )
    assert pred.pred_caption == "Hello world."


def test_reference_based_end_to_end(tmp_path: Path):
    # Build a tiny reference set from the first row of each caption file.
    caption_rows = []
    for path in sorted((REPO / "data" / "captions").glob("*.jsonl")):
        caption_rows.extend(read_jsonl(path)[:2])

    ref_rows = [
        {
            "ts_id": r["ts_id"],
            "dataset": r["dataset"],
            "ref_caption": "A short neutral reference caption for testing.",
            "source": "test",
        }
        for r in caption_rows
    ]
    refs = [
        Reference(r["ts_id"], r["dataset"], r["ref_caption"], r["source"]) for r in ref_rows
    ]

    # Pull predictions only for these ids, across all models.
    pred_ids = {r["ts_id"] for r in ref_rows}
    parser = get_parser("default")
    pred_rows = []
    for path in sorted((REPO / "predictions").glob("*.jsonl")):
        for r in read_jsonl(path):
            if r["ts_id"] in pred_ids:
                pred_rows.append(r)
    preds = parser.parse_all(pred_rows)

    ev = ReferenceBasedEvaluator(metrics=["bleu4", "rouge_l"])
    result = ev.evaluate(preds, references=refs)
    assert "bleu4" in result.corpus_scores
    assert "rouge_l" in result.corpus_scores
    assert 0.0 <= result.corpus_scores["bleu4"] <= 1.0
    assert 0.0 <= result.corpus_scores["rouge_l"] <= 1.0
