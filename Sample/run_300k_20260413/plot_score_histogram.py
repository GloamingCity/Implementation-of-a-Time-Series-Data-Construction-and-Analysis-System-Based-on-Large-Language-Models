#!/usr/bin/env python3
# ==============================
# 使用方式
# ==============================
#
# 1. 不带参数（默认模式）：同时生成 score_cache 和所有 jsonl 的直方图
#    python plot_score_histogram.py
#
# 2. 指定 score_cache.json：只画分数缓存分布（横轴从0开始）
#    python plot_score_histogram.py score_cache.json
#
# 3. 只指定 samples_filtered.jsonl（不带上级目录）：合并三个 jsonl 一起统计（横轴从75开始）
#    python plot_score_histogram.py samples_filtered.jsonl
#
# 4. 带上级目录指定 samples_filtered.jsonl：只统计对应子目录（横轴从75开始）
#    python plot_score_histogram.py anomaly_detection/samples_filtered.jsonl
#    python plot_score_histogram.py classification/samples_filtered.jsonl
#    python plot_score_histogram.py prediction/samples_filtered.jsonl
#
# 5. 按窗口长度分组统计分数分布（箱线图）：
#    python plot_score_histogram.py --by-window-length
#    python plot_score_histogram.py --by-window-length anomaly_detection/samples_filtered.jsonl
#

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def load_scores_from_score_cache(path: Path) -> list[int]:
    if not path.exists():
        print(f"[WARN] file not found: {path}")
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[ERROR] failed to parse {path}: {e}")
        return []
    if not isinstance(raw, dict):
        print(f"[ERROR] expected dict in {path}, got {type(raw).__name__}")
        return []
    scores = []
    for k, v in raw.items():
        try:
            sv = int(v)
            if 0 <= sv <= 100:
                scores.append(sv)
        except Exception:
            continue
    return scores


def load_scores_from_jsonl(path: Path) -> list[int]:
    if not path.exists():
        print(f"[WARN] file not found: {path}")
        return []
    scores = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            txt = line.strip()
            if not txt:
                continue
            try:
                obj = json.loads(txt)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            score = obj.get("score")
            try:
                sv = int(score)
                if 0 <= sv <= 100:
                    scores.append(sv)
            except Exception:
                continue
    return scores


def plot_histogram(scores: list[int], title: str, output_path: Path, bins: int = 50, x_min: int = 0) -> None:
    if not scores:
        print(f"[WARN] no scores to plot for: {title}")
        return

    arr = np.array(scores, dtype=np.int32)

    fig, ax = plt.subplots(figsize=(10, 6))
    counts, bin_edges, patches = ax.hist(arr, bins=bins, range=(0, 100), edgecolor="black", alpha=0.85)

    for count_val, patch in zip(counts, patches):
        if count_val > 0:
            ax.annotate(
                str(int(count_val)),
                xy=(patch.get_x() + patch.get_width() / 2, count_val),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xlabel("Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(x_min, 100)

    stats_text = (
        f"N={len(arr)}  "
        f"mean={arr.mean():.2f}  "
        f"median={np.median(arr):.1f}  "
        f"std={arr.std():.2f}\n"
        f"min={arr.min()}  max={arr.max()}  "
        f">=75: {int((arr >= 75).sum())}  "
        f">=88: {int((arr >= 88).sum())}"
    )
    ax.text(
        0.98, 0.97, stats_text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.7),
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"[INFO] saved: {output_path}  (N={len(arr)})")


def load_scores_with_window_length_from_jsonl(path: Path) -> list[tuple[int, int]]:
    if not path.exists():
        print(f"[WARN] file not found: {path}")
        return []
    results = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            txt = line.strip()
            if not txt:
                continue
            try:
                obj = json.loads(txt)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            score = obj.get("score")
            ts = obj.get("time_series")
            try:
                sv = int(score)
            except Exception:
                continue
            if not (0 <= sv <= 100):
                continue
            if isinstance(ts, list):
                wl = len(ts)
            else:
                continue
            results.append((sv, wl))
    return results


def plot_boxplot_by_window_length(data: list[tuple[int, int]], title: str, output_path: Path) -> None:
    if not data:
        print(f"[WARN] no data to plot for: {title}")
        return

    wl_to_scores: dict[int, list[int]] = {}
    for score, wl in data:
        wl_to_scores.setdefault(wl, []).append(score)

    sorted_wls = sorted(wl_to_scores.keys())
    labels = [str(wl) for wl in sorted_wls]
    box_data = [wl_to_scores[wl] for wl in sorted_wls]
    counts = [len(wl_to_scores[wl]) for wl in sorted_wls]

    fig, ax = plt.subplots(figsize=(max(12, len(sorted_wls) * 0.8), 6))

    bp = ax.boxplot(
        box_data,
        tick_labels=labels,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker="o", markersize=3, alpha=0.5),
    )

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(counts), vmax=max(counts))
    for patch, cnt in zip(bp["boxes"], counts):
        patch.set_facecolor(cmap(norm(cnt)))
        patch.set_alpha(0.7)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Sample Count", fontsize=10)

    for i, cnt in enumerate(counts, 1):
        scores_arr = np.array(box_data[i - 1])
        ax.text(
            i, 74.5,
            f"n={cnt}\nμ={scores_arr.mean():.1f}",
            ha="center", va="top", fontsize=7,
        )

    ax.set_xlabel("Window Length", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(73, 101)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    total = sum(counts)
    print(f"[INFO] saved: {output_path}  (N={total}, {len(sorted_wls)} window lengths)")


def find_all_jsonl_under(root: Path) -> list[Path]:
    results = sorted(root.rglob("samples_filtered.jsonl"))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Plot score histograms from score_cache.json or samples_filtered.jsonl"
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help=(
            "Target file to analyze. "
            "If it ends with 'score_cache.json', plot scores from that cache. "
            "If it ends with 'samples_filtered.jsonl', plot scores from that jsonl. "
            "When specifying only 'samples_filtered.jsonl' (without parent dir prefix), "
            "all samples_filtered.jsonl files under the script's directory are combined. "
            "When specifying with parent dir (e.g. 'anomaly_detection/samples_filtered.jsonl'), "
            "only that specific file is used."
        ),
    )
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins (default: 50)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for plots (default: script dir)")
    parser.add_argument("--by-window-length", action="store_true", help="Plot boxplot of score distribution grouped by window length (only for jsonl)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.by_window_length:
        if args.file is not None:
            target = Path(args.file)
            if not target.is_absolute():
                target = SCRIPT_DIR / target
            if target.name.lower() == "samples_filtered.jsonl" and target.exists():
                parent_name = target.parts[-2] if len(target.parts) > 1 else "all"
                wl_data = load_scores_with_window_length_from_jsonl(target)
                plot_boxplot_by_window_length(
                    wl_data,
                    title=f"Score by Window Length ({parent_name})",
                    output_path=output_dir / f"boxplot_{parent_name}_by_window_length.png",
                )
            else:
                print(f"[ERROR] --by-window-length requires a samples_filtered.jsonl file")
                sys.exit(1)
        else:
            jsonl_files = find_all_jsonl_under(SCRIPT_DIR)
            if not jsonl_files:
                print(f"[ERROR] no samples_filtered.jsonl found under {SCRIPT_DIR}")
                sys.exit(1)
            all_wl_data: list[tuple[int, int]] = []
            for jf in jsonl_files:
                all_wl_data.extend(load_scores_with_window_length_from_jsonl(jf))
            plot_boxplot_by_window_length(
                all_wl_data,
                title="Score by Window Length (all accepted)",
                output_path=output_dir / "boxplot_all_by_window_length.png",
            )
            for jf in jsonl_files:
                rel = jf.relative_to(SCRIPT_DIR)
                parent_name = rel.parts[0] if len(rel.parts) > 1 else jf.parent.name
                wl_data = load_scores_with_window_length_from_jsonl(jf)
                plot_boxplot_by_window_length(
                    wl_data,
                    title=f"Score by Window Length ({parent_name})",
                    output_path=output_dir / f"boxplot_{parent_name}_by_window_length.png",
                )
        return

    if args.file is None:
        score_cache_path = SCRIPT_DIR / "score_cache.json"
        if score_cache_path.exists():
            scores = load_scores_from_score_cache(score_cache_path)
            plot_histogram(
                scores,
                title="Score Distribution (score_cache.json)",
                output_path=output_dir / "histogram_score_cache.png",
                bins=args.bins,
            )
        else:
            print(f"[WARN] score_cache.json not found in {SCRIPT_DIR}")

        jsonl_files = find_all_jsonl_under(SCRIPT_DIR)
        if jsonl_files:
            all_scores: list[int] = []
            for jf in jsonl_files:
                all_scores.extend(load_scores_from_jsonl(jf))
            plot_histogram(
                all_scores,
                title="Score Distribution (all samples_filtered.jsonl combined)",
                output_path=output_dir / "histogram_all_accepted.png",
                bins=args.bins,
                x_min=75,
            )

            for jf in jsonl_files:
                rel = jf.relative_to(SCRIPT_DIR)
                parent_name = rel.parts[0] if len(rel.parts) > 1 else jf.parent.name
                scores_single = load_scores_from_jsonl(jf)
                plot_histogram(
                    scores_single,
                    title=f"Score Distribution ({parent_name}/samples_filtered.jsonl)",
                    output_path=output_dir / f"histogram_{parent_name}_accepted.png",
                    bins=args.bins,
                    x_min=75,
                )
        else:
            print(f"[WARN] no samples_filtered.jsonl found under {SCRIPT_DIR}")
        return

    target = Path(args.file)

    if not target.is_absolute():
        target = SCRIPT_DIR / target

    name_lower = target.name.lower()

    if name_lower == "score_cache.json":
        scores = load_scores_from_score_cache(target)
        plot_histogram(
            scores,
            title=f"Score Distribution ({target.name})",
            output_path=output_dir / "histogram_score_cache.png",
            bins=args.bins,
        )
        return

    if name_lower == "samples_filtered.jsonl":
        has_parent_prefix = len(target.parts) > 1 and target.parts[0] != "."

        if has_parent_prefix and target.exists():
            parent_name = target.parts[-2]
            scores = load_scores_from_jsonl(target)
            plot_histogram(
                scores,
                title=f"Score Distribution ({parent_name}/samples_filtered.jsonl)",
                output_path=output_dir / f"histogram_{parent_name}_accepted.png",
                bins=args.bins,
                x_min=75,
            )
        else:
            jsonl_files = find_all_jsonl_under(SCRIPT_DIR)
            if not jsonl_files:
                print(f"[ERROR] no samples_filtered.jsonl found under {SCRIPT_DIR}")
                sys.exit(1)
            all_scores = []
            for jf in jsonl_files:
                all_scores.extend(load_scores_from_jsonl(jf))
            plot_histogram(
                all_scores,
                title="Score Distribution (all samples_filtered.jsonl combined)",
                output_path=output_dir / "histogram_all_accepted.png",
                bins=args.bins,
                x_min=75,
            )
        return

    print(f"[ERROR] unsupported file: {args.file}. Only score_cache.json and samples_filtered.jsonl are supported.")
    sys.exit(1)


if __name__ == "__main__":
    main()
