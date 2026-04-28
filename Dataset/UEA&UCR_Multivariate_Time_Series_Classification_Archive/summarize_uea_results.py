import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    base = Path(__file__).resolve().parent
    cache_files = sorted(base.glob("**/_image_scores_cache.json"))
    analysis_files = sorted(base.glob("**/*_analysis.json"))

    all_scores_100 = []
    all_scores_10 = []
    scores_by_length_100 = defaultdict(list)
    dataset_overall_scores_100 = []
    relation_counts = {"positive": 0, "negative": 0, "flat": 0, "unknown": 0}

    for cache_path in cache_files:
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        for fname, score in data.items():
            if not isinstance(score, int):
                continue
            if score < 0 or score > 100:
                continue

            all_scores_100.append(score)
            all_scores_10.append(score / 10.0)

            m = re.search(r"_L(\d+)", str(fname))
            if m:
                length = int(m.group(1))
                scores_by_length_100[length].append(score)

    # Dataset-level summary from *_analysis.json
    for ap in analysis_files:
        try:
            obj = json.loads(ap.read_text(encoding="utf-8"))
        except Exception:
            continue

        # Current format is a single-item list containing a dict
        row = None
        if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
            row = obj[0]
        elif isinstance(obj, dict):
            row = obj

        if not isinstance(row, dict):
            continue

        overall = row.get("overall_score")
        if isinstance(overall, (int, float)) and 0 <= float(overall) <= 100:
            dataset_overall_scores_100.append(float(overall))

        rel = str(row.get("relation", "unknown")).lower().strip()
        if rel not in relation_counts:
            rel = "unknown"
        relation_counts[rel] += 1

    out_dir = base / "analysis_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not all_scores_100:
        raise RuntimeError("No valid scores found from _image_scores_cache.json files.")

    # Overall stats
    n = len(all_scores_100)
    mean_100 = float(np.mean(all_scores_100))
    median_100 = float(np.median(all_scores_100))
    std_100 = float(np.std(all_scores_100))

    # Buckets on 10-point scale
    arr10 = np.array(all_scores_10)
    fail_cnt = int(np.sum(arr10 < 6.0))
    b6_8_cnt = int(np.sum((arr10 >= 6.0) & (arr10 < 8.0)))
    b8_9_cnt = int(np.sum((arr10 >= 8.0) & (arr10 < 9.0)))
    b9_10_cnt = int(np.sum((arr10 >= 9.0) & (arr10 <= 10.0)))

    # Histogram plot (0-10)
    plt.figure(figsize=(10, 5))
    bins = np.arange(0, 11, 1)
    plt.hist(arr10, bins=bins, edgecolor="black", alpha=0.85)
    plt.title("Score Distribution (0-10 scale)")
    plt.xlabel("Score (0-10)")
    plt.ylabel("Sample Count")
    plt.xticks(np.arange(0, 11, 1))
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    hist_path = out_dir / "score_distribution_histogram.png"
    plt.tight_layout()
    plt.savefig(hist_path, dpi=180)
    plt.close()

    # Mean score by window length
    lengths = sorted(scores_by_length_100.keys())
    mean_by_length_100 = [float(np.mean(scores_by_length_100[l])) for l in lengths]
    mean_by_length_10 = [v / 10.0 for v in mean_by_length_100]

    plt.figure(figsize=(11, 5))
    plt.plot(lengths, mean_by_length_10, marker="o", linewidth=2)
    plt.title("Average Score by Window Length")
    plt.xlabel("Window Length")
    plt.ylabel("Average Score (0-10)")
    plt.grid(True, linestyle="--", alpha=0.4)
    length_plot_path = out_dir / "mean_score_by_window_length.png"
    plt.tight_layout()
    plt.savefig(length_plot_path, dpi=180)
    plt.close()

    # Dataset-level overall score histogram (0-10)
    dataset_hist_path = out_dir / "dataset_overall_score_distribution_histogram.png"
    if dataset_overall_scores_100:
        dataset_arr10 = np.array(dataset_overall_scores_100) / 10.0
        plt.figure(figsize=(10, 5))
        bins = np.arange(0, 11, 1)
        plt.hist(dataset_arr10, bins=bins, edgecolor="black", alpha=0.85, color="#4C72B0")
        plt.title("Dataset Overall Score Distribution (0-10 scale)")
        plt.xlabel("Dataset Overall Score (0-10)")
        plt.ylabel("Dataset Count")
        plt.xticks(np.arange(0, 11, 1))
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(dataset_hist_path, dpi=180)
        plt.close()

    # Relation distribution plot
    relation_plot_path = out_dir / "relation_distribution_histogram.png"
    labels = ["positive", "flat", "negative", "unknown"]
    values = [relation_counts[k] for k in labels]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=["#2E8B57", "#1F77B4", "#D62728", "#7F7F7F"])
    plt.title("Relation Distribution Across Datasets")
    plt.xlabel("Relation")
    plt.ylabel("Dataset Count")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, str(v), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(relation_plot_path, dpi=180)
    plt.close()

    summary = {
        "total_datasets_with_cache": len(cache_files),
        "total_samples": n,
        "score_0_100": {
            "mean": round(mean_100, 3),
            "median": round(median_100, 3),
            "std": round(std_100, 3),
            "min": int(np.min(all_scores_100)),
            "max": int(np.max(all_scores_100)),
        },
        "score_buckets_0_10": {
            "lt_6_fail": fail_cnt,
            "6_to_8": b6_8_cnt,
            "8_to_9": b8_9_cnt,
            "9_to_10": b9_10_cnt,
            "rates": {
                "lt_6_fail": round(fail_cnt / n, 4),
                "6_to_8": round(b6_8_cnt / n, 4),
                "8_to_9": round(b8_9_cnt / n, 4),
                "9_to_10": round(b9_10_cnt / n, 4),
            },
        },
        "window_length_stats": [
            {
                "length": int(l),
                "sample_count": len(scores_by_length_100[l]),
                "mean_score_0_100": round(mean_by_length_100[i], 3),
                "mean_score_0_10": round(mean_by_length_10[i], 3),
            }
            for i, l in enumerate(lengths)
        ],
        "dataset_level": {
            "total_analysis_files": len(analysis_files),
            "overall_score_0_100": {
                "count": len(dataset_overall_scores_100),
                "mean": round(float(np.mean(dataset_overall_scores_100)), 3) if dataset_overall_scores_100 else None,
                "median": round(float(np.median(dataset_overall_scores_100)), 3) if dataset_overall_scores_100 else None,
                "std": round(float(np.std(dataset_overall_scores_100)), 3) if dataset_overall_scores_100 else None,
                "min": round(float(np.min(dataset_overall_scores_100)), 3) if dataset_overall_scores_100 else None,
                "max": round(float(np.max(dataset_overall_scores_100)), 3) if dataset_overall_scores_100 else None,
            },
            "relation_counts": relation_counts,
        },
        "artifacts": {
            "histogram": str(hist_path),
            "mean_by_length_plot": str(length_plot_path),
            "dataset_overall_histogram": str(dataset_hist_path),
            "relation_distribution_plot": str(relation_plot_path),
        },
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Total samples: {n}")
    print(f"Mean score (0-100): {mean_100:.2f}")
    print(f"Buckets (0-10) => <6: {fail_cnt}, 6-8: {b6_8_cnt}, 8-9: {b8_9_cnt}, 9-10: {b9_10_cnt}")
    print(f"Saved: {hist_path}")
    print(f"Saved: {length_plot_path}")
    print(f"Saved: {dataset_hist_path}")
    print(f"Saved: {relation_plot_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
