"""Lightweight time series statistics used as context for the LLM judge.

We feed *summaries* to the judge instead of raw numeric arrays so the judge
prompt stays cheap and stable.
"""

from __future__ import annotations

import numpy as np


def summarize_series(series: list[float], top_k_cps: int = 3) -> dict:
    arr = np.asarray(series, dtype=np.float64)
    if arr.size == 0:
        return {"length": 0}
    diffs = np.diff(arr)
    cp_magnitude = np.abs(diffs)
    top_cp_idx = np.argsort(cp_magnitude)[-top_k_cps:][::-1].tolist() if diffs.size else []
    half = arr.size // 2
    first_std = float(np.std(arr[:half])) if half > 0 else 0.0
    second_std = float(np.std(arr[half:])) if arr.size - half > 0 else 0.0

    # Simple monotonic-trend heuristic: sign of least-squares slope.
    if arr.size >= 2:
        x = np.arange(arr.size, dtype=np.float64)
        slope = float(np.polyfit(x, arr, 1)[0])
    else:
        slope = 0.0
    if abs(slope) < 1e-6:
        trend = "flat"
    elif slope > 0:
        trend = "upward"
    else:
        trend = "downward"

    return {
        "length": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "argmin": int(arr.argmin()),
        "max": float(arr.max()),
        "argmax": int(arr.argmax()),
        "slope": slope,
        "trend": trend,
        "first_half_std": first_std,
        "second_half_std": second_std,
        "top_change_points": [
            {"idx": int(i), "delta": float(diffs[i])} for i in top_cp_idx
        ],
    }


def format_stats_for_prompt(stats: dict) -> str:
    if not stats or stats.get("length", 0) == 0:
        return "(empty series)"
    cps = ", ".join(
        f"t={cp['idx']} (Δ={cp['delta']:+.3f})" for cp in stats.get("top_change_points", [])
    )
    return (
        f"- Length: {stats['length']}\n"
        f"- Mean / Std: {stats['mean']:.3f} / {stats['std']:.3f}\n"
        f"- Min: {stats['min']:.3f} at t={stats['argmin']}\n"
        f"- Max: {stats['max']:.3f} at t={stats['argmax']}\n"
        f"- Slope: {stats['slope']:+.5f} ({stats['trend']})\n"
        f"- First-half std vs second-half std: {stats['first_half_std']:.3f} vs "
        f"{stats['second_half_std']:.3f}\n"
        f"- Top change points: {cps or '(none)'}"
    )
