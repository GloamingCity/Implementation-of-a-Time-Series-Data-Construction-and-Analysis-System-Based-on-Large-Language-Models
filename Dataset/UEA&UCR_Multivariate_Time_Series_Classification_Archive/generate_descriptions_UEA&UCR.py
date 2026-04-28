# Usage
# cd "D:\Users\Win\Desktop\Study\毕业设计-面向大语言模型的时序推理数据构建及分析系统实现\数据收集\数据集\UEA&UCR Multivariate Time Series Classification Archive"
# python "generate_descriptions_UEA&UCR.py" --arff_path CatsDogs\CatsDogs_TRAIN.arff --max_samples 5
# python "generate_descriptions_UEA&UCR.py" --arff_path Multivariate\BasicMotions\BasicMotions_TRAIN.arff --window_lengths 32 --main_channels 2 4 --max_samples 5
# 运行时可通过 --max_samples 指定生成样本数量以便快速调试。
# --window_lengths参数可以指定多个窗口长度，默认是“ --window_lengths 512 1024”
# 多变量场景建议使用 --main_channels（可传一个或多个）；描述使用第一个主通道，其他用于对齐 target_cols 语义。
# --main_channel 仍可用，但仅作为兼容旧命令的别名。

import argparse
import pandas as pd
import numpy as np
import json
import math
from pathlib import Path
import pathlib
import re
import sys

_COMMON_ROOT = Path(__file__).resolve().parents[1]
if str(_COMMON_ROOT) not in sys.path:
    sys.path.append(str(_COMMON_ROOT))
from common_desc_adapt import (
    smooth_signal_for_window,
    robust_classify_global_trend,
    classify_volatility_adaptive,
    detect_change_boundaries,
    segment_features_by_boundaries,
    enforce_summary_consistency,
    detect_local_extreme_events,
    build_joint_linkage_clause,
)


def make_json_safe(obj):
    """递归清洗对象，确保严格 JSON 兼容（NaN/Inf -> None）。"""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [make_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def load_ucr_arff(arff_path: str):
    """
    改进的 UEA/UCR ARFF 解析器，支持单/多变量数据（含 relational 属性）。
    返回 (df_numeric, targets, attrs, is_multivariate, n_channels)
    """
    attrs = []
    relational_attrs = []
    in_relational = False
    has_relational = False
    data_line_idx = None

    with open(arff_path, 'r', encoding='utf-8', errors='ignore') as f:
        all_text = f.read()

    lines = all_text.splitlines()
    for idx, line in enumerate(lines):
        raw = line.strip()
        if not raw or raw.startswith('%'):
            continue
        low = raw.lower()
        if low.startswith('@data'):
            data_line_idx = idx
            break
        if low.startswith('@relation'):
            continue
        if low.startswith('@end'):
            in_relational = False
            continue
        if low.startswith('@attribute') and 'relational' in low:
            in_relational = True
            has_relational = True
            attrs.append(('relationalAtt', 'relational'))
            continue
        if in_relational and low.startswith('@attribute'):
            parts = raw.split()
            if len(parts) >= 3:
                relational_attrs.append((parts[1], ' '.join(parts[2:])))
            continue
        if (not in_relational) and low.startswith('@attribute'):
            parts = raw.split()
            if len(parts) >= 3:
                attrs.append((parts[1], ' '.join(parts[2:])))

    if data_line_idx is None:
        raise ValueError(f"ARFF 文件 {arff_path} 未找到 @data 段")

    data_text = "\n".join(lines[data_line_idx + 1 :]).strip()
    is_multivariate = has_relational and len(relational_attrs) > 0

    numeric_data = []
    targets = []
    mv_series_list = []

    def parse_numeric_token(token: str) -> float:
        txt = token.strip().strip("'").strip('"')
        if txt in ('', '?'):
            return float('nan')
        try:
            return float(txt)
        except ValueError:
            m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', txt)
            if m:
                return float(m.group(0))
            return float('nan')

    if is_multivariate:
        # relational 格式：'ch0_t0,...\nch1_t0,...\n...\nchK_tN',Class
        pat = re.compile(r"'(.*?)'\s*,\s*([^,\r\n]+)", flags=re.DOTALL)
        matches = list(pat.finditer(data_text))
        if not matches:
            raise ValueError(f"ARFF 文件 {arff_path} 的 relational 数据段解析失败")

        for m in matches:
            block = m.group(1)
            label = m.group(2).strip().strip("'").strip('"')
            # UEA 多变量 ARFF 常见两种编码：真实换行 或 字符串转义 "\\n"
            if "\\n" in block and "\n" not in block:
                ch_lines = [ln.strip() for ln in block.split("\\n") if ln.strip()]
            else:
                ch_lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            rows = []
            for ln in ch_lines:
                vals = [parse_numeric_token(tok) for tok in ln.split(',') if tok.strip() != '']
                if vals:
                    rows.append(vals)
            if not rows:
                continue
            min_len = min(len(r) for r in rows)
            if min_len <= 1:
                continue
            arr = np.asarray([r[:min_len] for r in rows], dtype=float).T  # (time, channel)
            mv_series_list.append(arr)
            targets.append(label)

        if not mv_series_list:
            raise ValueError(f"ARFF 文件 {arff_path} 未解析到有效多变量样本")

        n_channels = int(mv_series_list[0].shape[1])
        df_numeric = pd.DataFrame()  # 多变量分支不使用该表
        return df_numeric, targets, attrs, True, n_channels, mv_series_list

    # 单变量格式
    data_rows = []
    for ln in data_text.splitlines():
        raw = ln.strip()
        if not raw or raw.startswith('%'):
            continue
        row = [x.strip() for x in raw.split(',') if x.strip() != '']
        if row:
            data_rows.append(row)

    if not data_rows:
        raise ValueError(f"ARFF 文件 {arff_path} 未包含有效数据行")

    target_idx = None
    for i, (name, typeinfo) in enumerate(attrs):
        if name.lower() == 'target' or ('{' in typeinfo and 'numeric' not in typeinfo.lower()):
            target_idx = i
            break

    numeric_attrs = [name for name, _ in attrs[:target_idx] if target_idx is not None] if target_idx is not None else [name for name, _ in attrs]

    for row in data_rows:
        if target_idx is not None and len(row) > target_idx:
            numeric_part = row[:target_idx]
            targets.append(row[target_idx])
        else:
            numeric_part = row
            targets.append(None)
        numeric_data.append([parse_numeric_token(x) for x in numeric_part])

    n_channels = 1
    df_numeric = pd.DataFrame(numeric_data, columns=numeric_attrs)
    return df_numeric, targets, attrs, False, n_channels, None


def _is_nearly_flat(x: np.ndarray, rel_span_thr: float = 0.08, std_rel_thr: float = 0.03) -> bool:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 8:
        return True
    q05, q95 = np.quantile(arr, [0.05, 0.95])
    span = float(q95 - q05)
    mean_abs = float(np.mean(np.abs(arr)))
    std_rel = float(np.std(arr)) / max(mean_abs, 1.0)
    scale = max(float(np.quantile(np.abs(arr), 0.90)), 1.0)
    return span <= rel_span_thr * scale or std_rel <= std_rel_thr


def _linkage_metrics(main_x: np.ndarray, other_x: np.ndarray) -> tuple[float | None, bool, bool]:
    a = np.asarray(main_x, dtype=float)
    b = np.asarray(other_x, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 8:
        return None, False, True

    if _is_nearly_flat(a) or _is_nearly_flat(b):
        return None, False, True

    rho = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(rho):
        return None, False, False

    stable = True
    if a.size >= 24:
        half = a.size // 2
        if half >= 8 and (a.size - half) >= 8:
            rho_1 = float(np.corrcoef(a[:half], b[:half])[0, 1])
            rho_2 = float(np.corrcoef(a[half:], b[half:])[0, 1])
            if np.isfinite(rho_1) and np.isfinite(rho_2):
                stable = (
                    abs(rho_1) >= 0.12
                    and abs(rho_2) >= 0.12
                    and rho_1 * rho_2 > 0
                    and rho_1 * rho > 0
                    and rho_2 * rho > 0
                )
            else:
                stable = False

    return rho, bool(stable), False


def describe_window_multivariate(win: np.ndarray, dataset_name: str, instance_id: int, class_label: str | None, main_channel: int = 0) -> tuple[str, dict]:
    # win shape: (time, channels)
    n_steps, n_channels = int(win.shape[0]), int(win.shape[1])
    main_channel = int(np.clip(main_channel, 0, max(0, n_channels - 1)))

    # 主通道描述完全复用单变量逻辑，保持原有文本风格
    main_series = win[:, main_channel].astype(float)
    base_desc, base_feat = describe_window_series(main_series, dataset_name, instance_id, class_label)

    channel_features = []
    for ch in range(n_channels):
        x = win[:, ch].astype(float)
        g_min, g_max = float(np.min(x)), float(np.max(x))
        g_mean, g_std = float(np.mean(x)), float(np.std(x))
        vol_level, _, _ = classify_volatility_adaptive(
            x,
            short_window=(n_steps < 200),
            long_window=(n_steps > 300),
        )
        trend_label, _ = classify_global_trend(x)
        best_lag, best_corr = estimate_period(x)
        periodic = bool(best_lag >= 8 and best_corr >= 0.45 and not is_micro_noise_period(x, best_lag, best_corr))
        channel_features.append(
            {
                "channel_idx": int(ch),
                "length": int(n_steps),
                "min": g_min,
                "max": g_max,
                "mean": g_mean,
                "std": g_std,
                "vol_level": vol_level,
                "trend_label": trend_label,
                "best_lag": int(best_lag),
                "best_corr": float(best_corr),
                "periodic": periodic,
            }
        )

    # 仅补充主通道与其他通道联动关系描述
    cross_channel_correlations = []
    for ch in range(n_channels):
        if ch == main_channel:
            continue
        rho, stable, flat = _linkage_metrics(main_series, win[:, ch].astype(float))
        if rho is None:
            continue
        cross_channel_correlations.append(
            {
                "channels": [int(main_channel), int(ch)],
                "correlation": float(rho),
                "stable": bool(stable and (not flat)),
            }
        )
    cross_channel_correlations.sort(key=lambda d: abs(d["correlation"]), reverse=True)

    linkage_lines = []
    corr_map = {}
    corr_stable_map = {}
    for c in cross_channel_correlations:
        _, j = c["channels"]
        key = f"通道 {j}"
        corr_map[key] = float(c["correlation"])
        corr_stable_map[key] = bool(c.get("stable", False))

    joint_clause = build_joint_linkage_clause(
        main_var=f"主通道 {main_channel}",
        corr_map=corr_map,
        stable_map=corr_stable_map,
        strong_thr=0.8,
    )
    if joint_clause:
        linkage_lines.append(joint_clause)

    for c in cross_channel_correlations[:5]:
        i, j = c["channels"]
        r = c["correlation"]
        stable = c.get("stable", False)
        if not stable:
            linkage_lines.append(f"通道 {j} 与主通道 {i} 联动不稳定（r≈{r:.2f}）")
        elif abs(r) >= 0.25:
            rel = "同向" if r >= 0 else "反向"
            linkage_lines.append(f"通道 {j} 与主通道 {i} 呈{rel}关系（r≈{r:.2f}）")
        else:
            linkage_lines.append(f"通道 {j} 与主通道 {i} 联动较弱（r≈{r:.2f}）")

    linkage_text = "联动关系：\n" + ("；\n".join(linkage_lines) + "。" if linkage_lines else "未检测到与主通道的稳定联动关系。")
    marker = "\n\n整体结论："
    if marker in base_desc:
        head, tail = base_desc.rsplit(marker, 1)
        description = head + "\n\n" + linkage_text + marker + tail
    else:
        description = base_desc + "\n\n" + linkage_text

    features = {
        "is_multivariate": True,
        "main_channel": int(main_channel),
        "n_channels": int(n_channels),
        "sequence_length": int(n_steps),
        "main_channel_features": base_feat,
        "channels": channel_features,
        "cross_channel_correlations": cross_channel_correlations,
    }
    return description, features


def sliding_window_indices(n: int, window: int, step: int):
    i = 0
    while i + window <= n:
        yield i
        i += step


def classify_global_trend(x: np.ndarray):
    return robust_classify_global_trend(x)


def classify_volatility(std: float, mean: float, eps: float = 1e-6):
    rel = std / (abs(mean) + eps)
    if rel < 0.05:
        lvl = "low"
    elif rel < 0.15:
        lvl = "medium"
    else:
        lvl = "high"
    return lvl, float(rel)


def estimate_period(x: np.ndarray, max_lag: int | None = None, min_lag: int = 8):
    x = np.asarray(x, dtype=float)
    n = len(x)
    min_lag = max(int(min_lag), min(24, max(8, n // 32)))
    if n < max(20, min_lag * 2):
        return 0, 0.0
    if np.allclose(x, x[0]):
        return 0, 0.0
    # 先去除线性趋势，避免平滑单调变化被误判为周期。
    t = np.arange(n, dtype=float)
    coef = np.polyfit(t, x, 1)
    trend = coef[0] * t + coef[1]
    x = x - trend
    if np.allclose(x, 0.0):
        return 0, 0.0
    x = x - x.mean()
    if max_lag is None:
        max_lag = min(n // 2, 200)
    max_lag = min(max_lag, n // 2)
    if max_lag < min_lag:
        return 0, 0.0
    acf_full = np.correlate(x, x, mode="full")
    acf = acf_full[n - 1 : n + max_lag]
    if acf[0] <= 0:
        return 0, 0.0
    acf = acf / acf[0]
    peaks = []
    for lag in range(min_lag, max_lag + 1):
        left = acf[lag - 1]
        right = acf[lag + 1] if lag < max_lag else -np.inf
        if acf[lag] >= left and acf[lag] >= right:
            peaks.append(lag)
    if peaks:
        best_lag = max(peaks, key=lambda lag: acf[lag])
        best_corr = float(acf[best_lag])
        if best_corr < 0.1:
            best_lag = max(range(min_lag, max_lag + 1), key=lambda lag: acf[lag])
            best_corr = float(acf[best_lag])
    else:
        best_lag = max(range(min_lag, max_lag + 1), key=lambda lag: acf[lag])
        best_corr = float(acf[best_lag])
    if best_corr < 0.1:
        return 0, 0.0
    return int(best_lag), best_corr

def is_micro_noise_period(vals: np.ndarray, lag: int, corr: float) -> bool:
    arr = np.asarray(vals, dtype=float)
    n = int(arr.size)
    if lag < 12 or corr < 0.45 or n < max(3 * lag, 72):
        return True

    span = float(np.max(arr) - np.min(arr))
    if span <= 1e-8:
        return True

    cyc_cnt = n // int(max(lag, 1))
    if cyc_cnt < 3:
        return True

    cut = cyc_cnt * int(lag)
    mat = arr[:cut].reshape(cyc_cnt, int(lag))
    cyc_spans = np.max(mat, axis=1) - np.min(mat, axis=1)
    cyc_span_ratio = float(np.median(cyc_spans)) / max(span, 1e-8)

    d = np.diff(arr)
    if d.size < 4:
        return True
    amp_ratio = float(np.median(np.abs(d))) / max(span, 1e-8)

    norm_cycles = []
    for cyc in mat:
        cyc_centered = cyc - float(np.mean(cyc))
        cyc_std = float(np.std(cyc_centered))
        if cyc_std <= 1e-8:
            continue
        norm_cycles.append(cyc_centered / cyc_std)
    if len(norm_cycles) < 3:
        return True

    proto = np.median(np.vstack(norm_cycles), axis=0)
    proto_std = float(np.std(proto))
    if proto_std <= 1e-8:
        return True

    corr_vals = []
    for cyc in norm_cycles:
        rho = float(np.corrcoef(cyc, proto)[0, 1])
        if np.isfinite(rho):
            corr_vals.append(rho)
    if not corr_vals:
        return True
    cycle_consistency = float(np.median(corr_vals))

    if lag <= 16 and (corr < 0.70 or cyc_span_ratio < 0.30):
        return True
    if lag <= 24 and (corr < 0.62 or cycle_consistency < 0.45):
        return True

    return bool(
        cyc_span_ratio < 0.20
        or amp_ratio < 0.025
        or cycle_consistency < 0.32
        or (corr < 0.52 and cycle_consistency < 0.40)
    )

def period_representative_bounds(vals: np.ndarray, lag: int) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    arr_min = float(np.min(arr))
    arr_max = float(np.max(arr))
    if lag < 2 or arr.size < max(2 * lag, 12):
        return arr_min, arr_max

    cyc_cnt = arr.size // int(lag)
    if cyc_cnt >= 2:
        cut = cyc_cnt * int(lag)
        mat = arr[:cut].reshape(cyc_cnt, int(lag))
        cyc_mins = np.min(mat, axis=1)
        cyc_maxs = np.max(mat, axis=1)
        rep_min = float(np.median(cyc_mins))
        rep_max = float(np.median(cyc_maxs))
    else:
        rep_min, rep_max = arr_min, arr_max

    top_k = int(min(8, max(3, arr.size // 24)))
    sorted_vals = np.sort(arr)
    top_mean = float(np.mean(sorted_vals[-top_k:]))
    bottom_mean = float(np.mean(sorted_vals[:top_k]))

    rep_max = max(rep_max, top_mean)
    rep_min = min(rep_min, bottom_mean)
    rep_max = min(rep_max, arr_max)
    rep_min = max(rep_min, arr_min)
    return rep_min, rep_max
def _lag_corr(y: np.ndarray, lag: int) -> float:
    arr = np.asarray(y, dtype=float)
    n = int(arr.size)
    if lag <= 0 or n <= lag + 2:
        return 0.0
    a = arr[:-lag]
    b = arr[lag:]
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa <= 1e-8 or sb <= 1e-8:
        return 0.0
    rho = float(np.corrcoef(a, b)[0, 1])
    return rho if np.isfinite(rho) else 0.0


def evaluate_global_periodicity(x: np.ndarray, best_lag: int, best_corr: float) -> dict:
    """评估全局周期是否显著，避免只在局部阶段有周期就触发全局周期判定。"""
    arr = np.asarray(x, dtype=float)
    n = int(arr.size)
    out = {
        "is_global_periodic": False,
        "cycle_consistency": 0.0,
        "first_half_corr": 0.0,
        "second_half_corr": 0.0,
        "harmonic_corr": 0.0,
    }
    if best_lag <= 0 or best_corr <= 0.0 or n < max(3 * max(best_lag, 1), 48):
        return out

    t = np.arange(n, dtype=float)
    coef = np.polyfit(t, arr, 1)
    y = arr - (coef[0] * t + coef[1])
    y = y - float(np.mean(y))

    lag2 = int(2 * best_lag)
    harmonic_corr = _lag_corr(y, lag2) if lag2 < n // 2 else 0.0

    n_cycles = n // int(best_lag)
    cycle_consistency = 0.0
    if n_cycles >= 3:
        cut = n_cycles * int(best_lag)
        mat = y[:cut].reshape(n_cycles, int(best_lag))
        norm_cycles = []
        for c in mat:
            s = float(np.std(c))
            if s <= 1e-8:
                continue
            norm_cycles.append((c - float(np.mean(c))) / s)
        if len(norm_cycles) >= 3:
            proto = np.mean(np.vstack(norm_cycles), axis=0)
            ps = float(np.std(proto))
            if ps > 1e-8:
                corr_vals = []
                for c in norm_cycles:
                    rho = float(np.corrcoef(c, proto)[0, 1])
                    if np.isfinite(rho):
                        corr_vals.append(rho)
                if corr_vals:
                    cycle_consistency = float(np.median(corr_vals))

    mid = n // 2
    first_half_corr = _lag_corr(y[:mid], int(best_lag)) if mid > best_lag + 8 else 0.0
    second_half_corr = _lag_corr(y[mid:], int(best_lag)) if (n - mid) > best_lag + 8 else 0.0

    is_global = bool(
        best_corr >= 0.50
        and cycle_consistency >= 0.45
        and min(first_half_corr, second_half_corr) >= 0.25
        and harmonic_corr >= 0.08
    )

    out.update(
        {
            "is_global_periodic": is_global,
            "cycle_consistency": cycle_consistency,
            "first_half_corr": first_half_corr,
            "second_half_corr": second_half_corr,
            "harmonic_corr": harmonic_corr,
        }
    )
    return out


def build_period_phrases(best_lag: int, fallback_unit: str = "个时间点") -> tuple[str, str]:
    fallback_text = f"{best_lag} {fallback_unit}"
    return f"约每隔 {fallback_text}", f"约 {fallback_text}"


def segment_features(x: np.ndarray, num_segments: int = 4, global_std: float = 0.0):
    x = np.asarray(x, dtype=float)
    L = len(x)
    if L < num_segments:
        return []
    seg_len = L // num_segments
    segments = []
    start = 0
    for k in range(num_segments):
        end = (k + 1) * seg_len if k < num_segments - 1 else L
        seg = x[start:end]
        if len(seg) < 2:
            break
        seg_mean = float(seg.mean())
        seg_std = float(seg.std())
        if global_std <= 0:
            vol_level = "low"
        else:
            ratio = seg_std / global_std
            if ratio < 0.7:
                vol_level = "low"
            elif ratio < 1.3:
                vol_level = "medium"
            else:
                vol_level = "high"
        seg_trend_label, _ = classify_global_trend(seg)
        segments.append(
            {
                "idx": k,
                "start": int(start),
                "end": int(end - 1),
                "len": int(end - start),
                "mean": seg_mean,
                "std": seg_std,
                "vol_level": vol_level,
                "trend_label": seg_trend_label,
            }
        )
        start = end
    return segments


def detect_flat_or_square_wave(x: np.ndarray) -> str:
    arr = np.asarray(x, dtype=float)
    if arr.size < 16:
        return "none"
    span = float(np.max(arr) - np.min(arr))
    std = float(np.std(arr))
    if span <= 1e-8 or std <= 1e-8:
        return "flat"
    if (std / max(span, 1e-8)) <= 0.015:
        return "flat"

    lo = float(np.quantile(arr, 0.25))
    hi = float(np.quantile(arr, 0.75))
    if hi - lo <= 1e-8:
        return "none"
    mid = 0.5 * (lo + hi)
    state = np.where(arr >= mid, 1, 0)
    high_ratio = float(np.mean(state == 1))
    low_ratio = 1.0 - high_ratio
    if high_ratio < 0.15 or low_ratio < 0.15:
        return "none"
    trans = int(np.sum(state[1:] != state[:-1]))
    if trans < 2 or trans > max(18, arr.size // 6):
        return "none"
    low_vals = arr[state == 0]
    high_vals = arr[state == 1]
    if low_vals.size < 4 or high_vals.size < 4:
        return "none"
    level_gap = float(abs(np.mean(high_vals) - np.mean(low_vals)))
    if level_gap <= 1e-8:
        return "none"
    noise_ratio = max(float(np.std(low_vals)), float(np.std(high_vals))) / level_gap
    return "square" if noise_ratio <= 0.10 else "none"


def estimate_square_segment_count(x: np.ndarray) -> int:
    arr = np.asarray(x, dtype=float)
    n = int(arr.size)
    if n < 16:
        return 2
    lo = float(np.quantile(arr, 0.25))
    hi = float(np.quantile(arr, 0.75))
    if hi - lo <= 1e-8:
        return 2
    mid = 0.5 * (lo + hi)
    state = np.where(arr >= mid, 1, 0)
    trans = int(np.sum(state[1:] != state[:-1]))
    return int(min(4, max(2, trans + 1)))


def detect_zscore_events(x: np.ndarray, z_thr: float = 2.5):
    return detect_local_extreme_events(x, z_thr=z_thr)


def describe_window_series(series: np.ndarray, dataset_name: str, instance_id: int, class_label: str | None) -> tuple[str, dict]:
    """
    针对单条时间序列窗口生成中文描述及特征。
    """
    x = series.astype(float)
    L = len(x)
    is_long_window = L > 300
    is_short_window = L < 200
    trend_signal = smooth_signal_for_window(x, short_window=is_short_window) if is_short_window else x
    g_min, g_max = float(x.min()), float(x.max())
    g_mean = float(x.mean())
    g_std = float(x.std())
    legacy_vol_level, legacy_vol_ratio = classify_volatility(g_std, g_mean)
    vol_level, vol_ratio, vol_meta = classify_volatility_adaptive(
        x,
        short_window=is_short_window,
        long_window=is_long_window,
    )
    trend_label, slope_norm = classify_global_trend(trend_signal)
    best_lag, best_corr = estimate_period(x)
    period_profile = evaluate_global_periodicity(x, best_lag, best_corr)
    period_every, period_cycle = build_period_phrases(best_lag)
    special_shape = detect_flat_or_square_wave(x)
    if special_shape == "flat":
        segments = []
    elif special_shape == "square":
        seg_n = estimate_square_segment_count(x)
        segments = segment_features(x, num_segments=seg_n, global_std=g_std)
    else:
        boundaries = detect_change_boundaries(
            trend_signal,
            short_window=is_short_window,
            long_window=is_long_window,
        )
        segments = segment_features_by_boundaries(
            x,
            boundaries=boundaries,
            global_std=g_std,
            trend_func=classify_global_trend,
        )
        if not segments:
            segments = segment_features(x, num_segments=4, global_std=g_std)
    events = detect_zscore_events(x, z_thr=2.5)

    def trend_phrase(label: str) -> str:
        mapping = {
            "strong_up": "明显上行",
            "weak_up": "缓慢抬升",
            "flat": "较为平稳",
            "weak_down": "小幅回落",
            "strong_down": "明显下行",
        }
        return mapping.get(label, "较为平稳")

    def vol_phrase(level: str) -> str:
        mapping = {"low": "波动较小", "medium": "波动中等", "high": "波动偏强"}
        return mapping.get(level, "波动中等")

    def segment_range_phrase(seg: dict) -> str:
        if seg.get("vol_level") == "high":
            return "起伏较明显"
        if seg.get("trend_label") == "flat":
            return "变化较小"
        return "有一定起伏"

    def cycle_shape_phrase(cycle_values: np.ndarray) -> str:
        if cycle_values.size < 3:
            return "走势较稳"
        peak_idx = int(np.argmax(cycle_values))
        trough_idx = int(np.argmin(cycle_values))
        if peak_idx == trough_idx:
            return "整体起伏较弱"
        return "先抬升后回落" if peak_idx < trough_idx else "先回落后抬升"

    def cycle_wave_profile(vals: np.ndarray, lag: int) -> tuple[str, float]:
        if lag < 6 or len(vals) < max(2 * lag, 24):
            return "周期起伏型", 0.0
        cycles = [vals[i : i + lag] for i in range(0, len(vals) - lag + 1, lag)]
        if len(cycles) < 2:
            return "周期起伏型", 0.0
        proto_len = int(min(64, max(16, lag)))
        resampled = []
        for c in cycles:
            lo = float(np.min(c))
            hi = float(np.max(c))
            span = hi - lo
            if span <= 1e-8:
                continue
            y = (c - lo) / span
            x_old = np.linspace(0.0, 1.0, num=len(y), endpoint=True)
            x_new = np.linspace(0.0, 1.0, num=proto_len, endpoint=True)
            resampled.append(np.interp(x_new, x_old, y))
        if len(resampled) < 2:
            return "周期起伏型", 0.0
        proto = np.median(np.vstack(resampled), axis=0)
        high_ratio = float(np.mean(proto >= 0.70))
        low_ratio = float(np.mean(proto <= 0.30))
        d = np.diff(proto)
        if d.size == 0:
            return "周期起伏型", high_ratio
        eps = 0.02
        sign = np.zeros_like(d)
        sign[d > eps] = 1
        sign[d < -eps] = -1
        nz = sign[sign != 0]
        sign_changes = int(np.sum(nz[1:] != nz[:-1])) if nz.size >= 2 else 0
        sharp = float(np.quantile(np.abs(d), 0.90))
        rise_ratio = float(np.mean(d > eps))
        peak_mask = (proto[1:-1] > proto[:-2]) & (proto[1:-1] > proto[2:]) & (proto[1:-1] > 0.62)
        peak_cnt = int(np.sum(peak_mask))
        if peak_cnt >= 2:
            wave = "双峰脉冲型"
        elif high_ratio >= 0.42 and low_ratio >= 0.20 and sharp >= 0.10:
            wave = "近似方波型"
        elif sign_changes <= 2 and high_ratio <= 0.34 and low_ratio <= 0.34:
            wave = "近似正弦波型"
        elif sign_changes <= 3 and sharp < 0.10:
            wave = "锯齿波型" if (rise_ratio >= 0.62 or rise_ratio <= 0.38) else "近似三角波型"
        elif high_ratio <= 0.16 and low_ratio >= 0.46 and peak_cnt >= 1:
            wave = "窄峰脉冲型"
        else:
            wave = "周期起伏型"
        return wave, high_ratio

    def shape_phrase() -> str:
        seg_trends = [s.get("trend_label") for s in segments] if segments else []
        up_cnt = sum(t in ("strong_up", "weak_up") for t in seg_trends)
        down_cnt = sum(t in ("strong_down", "weak_down") for t in seg_trends)
        has_period = bool(periodic_cycle_flag)
        seg_means = [float(s.get("mean", 0.0)) for s in segments] if segments else []
        value_span = max(g_max - g_min, 1e-6)
        net_change_ratio = (float(x[-1]) - float(x[0])) / value_span
        move_thr = max(0.06 * value_span, 1e-6)
        mean_diffs = [seg_means[i + 1] - seg_means[i] for i in range(max(0, len(seg_means) - 1))]
        diff_signs = []
        for d in mean_diffs:
            if d > move_thr:
                diff_signs.append("up")
            elif d < -move_thr:
                diff_signs.append("down")

        pos_moves = sum(s == "up" for s in diff_signs)
        neg_moves = sum(s == "down" for s in diff_signs)
        mixed_moves = pos_moves > 0 and neg_moves > 0
        signed_moves = pos_moves + neg_moves
        swing_ratio = (max(seg_means) - min(seg_means)) / value_span if seg_means else 0.0
        event_intense = len(events) > 0 and max(abs(float(ev.get("z_max", 0.0))) for ev in events) >= 2.8
        near_flat_baseline = (
            trend_label == "flat"
            and abs(slope_norm) < 0.07
            and abs(net_change_ratio) < 0.12
            and signed_moves <= 1
        )
        alt_changes = sum(
            diff_signs[i] != diff_signs[i - 1]
            for i in range(1, len(diff_signs))
        ) if len(diff_signs) >= 2 else 0

        if near_flat_baseline and event_intense:
            return "整体较为平稳"

        directions = []
        for trend in seg_trends:
            if trend in ("strong_up", "weak_up"):
                directions.append("up")
            elif trend in ("strong_down", "weak_down"):
                directions.append("down")

        compressed = []
        for direction in directions:
            if not compressed or compressed[-1] != direction:
                compressed.append(direction)

        transition_like = len(compressed) >= 2 and len(compressed) <= 3
        valley_mid = len(seg_means) >= 3 and int(np.argmin(seg_means)) in (1, 2)
        peak_mid = len(seg_means) >= 3 and int(np.argmax(seg_means)) in (1, 2)
        periodic_like = has_period and (mixed_moves or alt_changes >= 1 or len(compressed) >= 3)

        if periodic_like:
            periodic_base = "呈明显周期性起伏" if best_corr >= 0.7 else "呈周期性起伏"
            return periodic_base

        transition_down_up = (
            transition_like
            and compressed[0] == "down"
            and compressed[-1] == "up"
            and up_cnt >= 1
            and down_cnt >= 1
            and valley_mid
            and net_change_ratio >= 0.10
            and swing_ratio >= 0.22
            and signed_moves >= 2
            and alt_changes <= 1
            and not event_intense
        )
        transition_up_down = (
            transition_like
            and compressed[0] == "up"
            and compressed[-1] == "down"
            and up_cnt >= 1
            and down_cnt >= 1
            and peak_mid
            and net_change_ratio <= -0.10
            and swing_ratio >= 0.22
            and signed_moves >= 2
            and alt_changes <= 1
            and not event_intense
        )

        down_dominant = (
            (neg_moves >= 2 and pos_moves == 0)
            or (
                neg_moves >= 2
                and pos_moves == 1
                and trend_label in ("strong_down", "weak_down")
                and (slope_norm <= -0.08 or net_change_ratio <= -0.12)
            )
            or (
                trend_label in ("strong_down", "weak_down")
                and down_cnt >= max(2, up_cnt)
                and net_change_ratio <= -0.08
            )
        )
        up_dominant = (
            (pos_moves >= 2 and neg_moves == 0)
            or (
                pos_moves >= 2
                and neg_moves == 1
                and trend_label in ("strong_up", "weak_up")
                and (slope_norm >= 0.08 or net_change_ratio >= 0.12)
            )
            or (
                trend_label in ("strong_up", "weak_up")
                and up_cnt >= max(2, down_cnt)
                and net_change_ratio >= 0.08
            )
        )

        base = ""
        if np.allclose(x, x[0]):
            base = "整体较为平稳"
        elif seg_trends:
            if transition_down_up and alt_changes <= 1:
                base = "呈先降后升"
            elif transition_up_down and alt_changes <= 1:
                base = "呈先升后降"
            elif mixed_moves or alt_changes >= 1:
                base = "呈多阶段起伏"
            elif down_dominant:
                base = "呈多阶段下降"
            elif up_dominant:
                base = "呈多阶段上升"
            elif mixed_moves or (up_cnt > 0 and down_cnt > 0):
                base = "呈多阶段起伏"
            elif up_cnt >= 2 and down_cnt == 0:
                base = "呈多阶段上升"
            elif down_cnt >= 2 and up_cnt == 0:
                base = "呈多阶段下降"

        if not base:
            fallback = {
                "strong_up": "整体上行明显",
                "weak_up": "整体缓慢抬升",
                "flat": "整体较为平稳",
                "weak_down": "整体小幅回落",
                "strong_down": "整体下行明显",
            }
            base = fallback.get(trend_label, "整体较为平稳")
        return base

    def detect_plateau_with_local_fluctuation() -> tuple[bool, dict]:
        if not segments or len(segments) < 3:
            return False, {}
        seg_stds = np.array([float(s.get("std", 0.0)) for s in segments], dtype=float)
        if seg_stds.size == 0 or not np.all(np.isfinite(seg_stds)):
            return False, {}
        median_std = float(np.median(seg_stds))
        if median_std <= 0:
            return False, {}
        value_span = max(g_max - g_min, 1e-6)
        net_change_ratio = abs(float(x[-1]) - float(x[0])) / value_span
        quiet_thr = max(0.85 * median_std, 0.35 * g_std)
        quiet_count = int(np.sum(seg_stds <= quiet_thr))
        active_idx = int(np.argmax(seg_stds))
        active_seg = segments[active_idx]
        active_share = float(seg_stds[active_idx]) / max(float(np.sum(seg_stds)), 1e-6)
        is_plateau_local = (
            quiet_count >= len(segments) - 1
            and active_share >= 0.34
            and net_change_ratio <= 0.22
            and trend_label in ("flat", "weak_up", "weak_down")
        )
        return bool(is_plateau_local), {"active_seg": active_seg}

    def detect_level_oscillation() -> bool:
        value_span = max(g_max - g_min, 1e-6)
        net_change_ratio = abs(float(x[-1]) - float(x[0])) / value_span
        diff = np.diff(x)
        if diff.size < 6:
            return False
        sign = np.sign(diff)
        sign = sign[sign != 0]
        if sign.size < 6:
            return False
        sign_flip_ratio = float(np.sum(sign[1:] * sign[:-1] < 0)) / max(sign.size - 1, 1)
        turning = (x[1:-1] - x[:-2]) * (x[1:-1] - x[2:])
        turning_ratio = float(np.sum(turning > 0)) / max(len(x) - 2, 1)
        seg_trends = [s.get("trend_label", "") for s in segments]
        trend_switches = sum(1 for i in range(1, len(seg_trends)) if seg_trends[i] != seg_trends[i - 1])
        return bool(
            net_change_ratio <= 0.40
            and sign_flip_ratio >= 0.30
            and turning_ratio >= 0.22
            and trend_switches >= 1
        )

    plateau_local_flag, plateau_local_info = detect_plateau_with_local_fluctuation()
    oscillation_flag = detect_level_oscillation()
    periodic_cycle_flag = bool(period_profile.get("is_global_periodic", False) and not np.allclose(x, x[0]))
    diff_abs = np.abs(np.diff(x))
    jump_thr = max(
        0.30 * max(g_max - g_min, 1e-6),
        (2.5 * float(np.std(np.diff(x)))) if diff_abs.size else 0.0,
    )
    abrupt_switch_flag = bool((int(np.sum(diff_abs >= jump_thr)) if diff_abs.size else 0) >= 2)
    if (not periodic_cycle_flag) and (
        (special_shape == "square" and best_lag >= 6 and best_corr >= 0.22)
        or (abrupt_switch_flag and best_lag >= 8 and best_corr >= 0.30)
    ):
        periodic_cycle_flag = True
    platform_switch_segment_flag = bool(special_shape == "square" or abrupt_switch_flag)
    periodic_score = (0.55 if periodic_cycle_flag else 0.0) + min(max((best_corr - 0.30) / 0.50, 0.0), 0.35)
    segment_score = (0.45 if platform_switch_segment_flag else 0.0) + (0.20 if plateau_local_flag else 0.0) + (0.10 if oscillation_flag else 0.0)
    if periodic_cycle_flag and is_micro_noise_period(x, best_lag, best_corr):
        periodic_cycle_flag = False
    periodic_preferred_flag = bool(periodic_cycle_flag and periodic_score >= segment_score)
    

    def overall_summary_phrase() -> str:
        if periodic_preferred_flag:
            return "呈明显周期性起伏" if best_corr >= 0.7 else "呈周期性起伏"
        summary = shape_phrase()
        if oscillation_flag and not plateau_local_flag:
            return "整体以高频震荡为主，短时起伏较为频繁"
        if plateau_local_flag:
            return "整体以平台期为主，仅在局部时段出现短时波动"
        if summary.startswith("整体较为平稳"):
            if events:
                return "整体较为平稳，但部分时间段存在异常波动"
            late_high = any(s.get("idx", -1) >= max(1, len(segments) - 2) and s.get("vol_level") == "high" for s in segments)
            if late_high:
                return "整体较为平稳，但后段波动有所放大"
            if any(s.get("vol_level") == "high" for s in segments):
                return "整体较为平稳，但局部波动较明显"
        return summary

    lines = []
    var_name = "该实例序列值"
    start_val, end_val = float(x[0]), float(x[-1])
    tag = f"标签为“{class_label}”" if class_label is not None else "标签未知"

    if np.allclose(x, x[0]):
        trait = "整体保持恒定"
    elif periodic_preferred_flag:
        trait = "整体呈明显周期性波动" if best_corr >= 0.7 else "整体呈周期性波动"
    elif oscillation_flag and not plateau_local_flag:
        trait = "整体变化不规则，短时起伏较频繁"
    elif plateau_local_flag:
        trait = "整体以平台波动为主，仅局部有起伏"
    else:
        phrase = shape_phrase()
        trait = phrase[2:] if phrase.startswith("整体") else phrase

    lines.append(
        "窗口概览：\n"
        + f"{dataset_name} 的第 {instance_id} 个实例（{tag}）在当前窗口内共 {L} 个观测点。"
        + f"{var_name} 大致分布在 {g_min:.3g} 到 {g_max:.3g} 之间，平均约 {g_mean:.3g}，起始点为 {start_val:.3g}，结束点为 {end_val:.3g}，{trait}。"
    )

    if segments:
        segment_subject = str(var_name)

        def fmt_num(v: float) -> str:
            av = abs(float(v))
            if av >= 100:
                return f"{float(v):.0f}"
            if av >= 1:
                return f"{float(v):.2f}"
            if av >= 0.01:
                return f"{float(v):.4f}"
            return f"{float(v):.3g}"

        def seg_interval_text(seg: dict) -> str:
            return f"第 {int(seg['start'])} 到 {int(seg['end'])} 点"

        def seg_dir(seg: dict) -> str:
            lbl = str(seg.get("trend_label", "flat"))
            if lbl in ("strong_up", "weak_up"):
                return "up"
            if lbl in ("strong_down", "weak_down"):
                return "down"
            return "flat"

        def seg_amp(seg: dict) -> str:
            lvl = str(seg.get("vol_level", "medium"))
            if lvl == "high":
                return "high"
            if lvl == "low":
                return "low"
            return "mid"

        def group_phrase(direction: str, amp: str, confidence: str = "mid") -> str:
            if direction == "up":
                if confidence == "high":
                    dir_text = "整体抬升"
                elif confidence == "low":
                    dir_text = "略有抬升"
                else:
                    dir_text = "缓慢抬升"
            elif direction == "down":
                if confidence == "high":
                    dir_text = "整体回落"
                elif confidence == "low":
                    dir_text = "略有回落"
                else:
                    dir_text = "缓慢回落"
            else:
                dir_text = "整体较平稳"
            amp_text = {
                "high": "且波动幅度偏大",
                "mid": "且波动幅度中等",
                "low": "且波动幅度较小",
            }.get(amp, "且波动幅度中等")
            return f"{dir_text}，{amp_text}"

        def infer_phase_direction(vals: np.ndarray, fallback: str = "flat") -> tuple[str, str]:
            arr = np.asarray(vals, dtype=float)
            if arr.size < 5:
                return fallback, "low"

            lo = float(np.min(arr))
            hi = float(np.max(arr))
            span = max(hi - lo, 1e-6)

            win = 5 if arr.size >= 24 else 3
            if win % 2 == 0:
                win += 1
            sm = pd.Series(arr).rolling(window=win, center=True, min_periods=1).mean().to_numpy()
            d = np.diff(sm)
            if d.size == 0:
                return fallback, "low"

            eps = 0.015 * span
            pos = int(np.sum(d > eps))
            neg = int(np.sum(d < -eps))
            nz = pos + neg
            if nz == 0:
                return "flat", "low"

            net = float(sm[-1] - sm[0])
            net_ratio = abs(net) / span
            dom_ratio = max(pos, neg) / max(nz, 1)

            if net_ratio < 0.10 or dom_ratio < 0.60:
                return "flat", "low"

            if net > 0 and pos >= int(1.2 * max(neg, 1)):
                conf = "high" if (net_ratio >= 0.28 and dom_ratio >= 0.72) else "mid"
                return "up", conf
            if net < 0 and neg >= int(1.2 * max(pos, 1)):
                conf = "high" if (net_ratio >= 0.28 and dom_ratio >= 0.72) else "mid"
                return "down", conf
            return fallback, "low"

        def has_hump_shape(vals: np.ndarray) -> bool:
            if len(vals) < 9:
                return False
            lo = float(np.min(vals))
            hi = float(np.max(vals))
            span = max(hi - lo, 1e-6)
            p1, p2, p3 = [float(np.mean(x)) for x in np.array_split(vals, 3)]
            hump = p1 < p2 > p3 and (p2 - min(p1, p3)) >= 0.12 * span
            valley = p1 > p2 < p3 and (max(p1, p3) - p2) >= 0.12 * span
            return bool(hump or valley)

        def phase_shape_phrase(vals: np.ndarray) -> str:
            arr = np.asarray(vals, dtype=float)
            if arr.size < 6:
                return "走势较稳"
            lo = float(np.min(arr))
            hi = float(np.max(arr))
            span = max(hi - lo, 1e-6)
            x = (arr - lo) / span
            d = np.diff(x)
            if d.size == 0:
                return "走势较稳"

            eps = 0.03
            sign = np.zeros_like(d)
            sign[d > eps] = 1
            sign[d < -eps] = -1
            nz = sign[sign != 0]
            flip_ratio = float(np.sum(nz[1:] != nz[:-1])) / max(nz.size - 1, 1) if nz.size >= 2 else 0.0

            up_ratio = float(np.mean(d > eps))
            down_ratio = float(np.mean(d < -eps))
            high_ratio = float(np.mean(x >= 0.70))
            low_ratio = float(np.mean(x <= 0.30))
            top_ratio = float(np.mean(x >= 0.88))

            peak_mask = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]) & ((x[1:-1] - np.maximum(x[:-2], x[2:])) >= 0.05)
            valley_mask = (x[1:-1] < x[:-2]) & (x[1:-1] < x[2:]) & ((np.minimum(x[:-2], x[2:]) - x[1:-1]) >= 0.05)
            peak_cnt = int(np.sum(peak_mask)) if x.size >= 5 else 0
            valley_cnt = int(np.sum(valley_mask)) if x.size >= 5 else 0

            p1, p2, p3 = [float(np.mean(xp)) for xp in np.array_split(x, 3)]
            q1, q2, q3, q4 = [float(np.mean(xq)) for xq in np.array_split(x, 4)]
            net = float(x[-1] - x[0])
            amp = float(np.std(x))

            if peak_cnt >= 3 and valley_cnt >= 3 and flip_ratio >= 0.45:
                return "高频密集震荡"
            if peak_cnt >= 2 and valley_cnt >= 2:
                if net >= 0.12:
                    return "锯齿式震荡上行"
                if net <= -0.12:
                    return "锯齿式震荡下行"
                return "多峰往复震荡"
            if peak_cnt >= 2 and valley_cnt <= 1:
                return "双峰起伏"
            if valley_cnt >= 2 and peak_cnt <= 1:
                return "双谷起伏"

            if q1 < q2 < q3 < q4 and (q4 - q1) >= 0.35 and up_ratio >= 0.40:
                return "阶梯式上升"
            if q1 > q2 > q3 > q4 and (q1 - q4) >= 0.35 and down_ratio >= 0.40:
                return "阶梯式下降"

            global_span = max(g_max - g_min, 1e-6)
            phase_span = max(float(np.max(arr)) - float(np.min(arr)), 1e-6)
            span_ratio = phase_span / global_span
            std_ratio = float(np.std(arr)) / max(g_std, 1e-6)
            global_mid_ratio = (float(np.mean(arr)) - g_min) / global_span
            platform_like = (high_ratio >= 0.45 and low_ratio <= 0.15) or (low_ratio >= 0.45 and high_ratio <= 0.15)
            if platform_like:
                if span_ratio >= 0.42 or std_ratio >= 0.95:
                    return "区间震荡"
                if high_ratio >= 0.45 and low_ratio <= 0.15:
                    if global_mid_ratio >= 0.62:
                        return "高位平台震荡"
                    if global_mid_ratio <= 0.38:
                        return "低位平台震荡"
                    return "平台震荡"
                if global_mid_ratio <= 0.38:
                    return "低位平台震荡"
                if global_mid_ratio >= 0.62:
                    return "高位平台震荡"
                return "平台震荡"

            if top_ratio <= 0.10 and peak_cnt >= 1 and low_ratio >= 0.46:
                return "窄峰脉冲型"

            if p1 < p2 > p3 and (p2 - min(p1, p3)) >= 0.10:
                return "先升后降"
            if p1 > p2 < p3 and (max(p1, p3) - p2) >= 0.10:
                return "先降后升"

            if net >= 0.18:
                return "整体抬升"
            if net <= -0.18:
                return "整体回落"
            if amp <= 0.10:
                return "窄幅震荡" if span_ratio <= 0.28 else "区间震荡"
            return "以震荡为主"

        def short_window_curve_detail(vals: np.ndarray) -> str:
            arr = np.asarray(vals, dtype=float)
            if (not is_short_window) or arr.size < 12:
                return ""
            lo = float(np.min(arr))
            hi = float(np.max(arr))
            span = max(hi - lo, 1e-6)

            sm = pd.Series(arr).rolling(window=3, center=True, min_periods=1).mean().to_numpy()
            p1, p2, p3 = [float(np.mean(xp)) for xp in np.array_split(sm, 3)]
            d12 = p2 - p1
            d23 = p3 - p2
            turn_thr = 0.10 * span

            if d12 >= turn_thr and d23 <= -turn_thr:
                return "曲线在中段附近形成局部高点，呈先升后回落"
            if d12 <= -turn_thr and d23 >= turn_thr:
                return "曲线在中段附近形成局部低点，呈先降后回升"

            net = float(sm[-1] - sm[0])
            if net >= 0.16 * span:
                return "前后段对比可见末段整体高于前段"
            if net <= -0.16 * span:
                return "前后段对比可见末段整体低于前段"

            d = np.diff(sm)
            eps = 0.03 * span
            sign = np.zeros_like(d)
            sign[d > eps] = 1
            sign[d < -eps] = -1
            nz = sign[sign != 0]
            flip_cnt = int(np.sum(nz[1:] != nz[:-1])) if nz.size >= 2 else 0
            if flip_cnt >= max(4, int(0.28 * max(arr.size - 1, 1))):
                return "点间方向切换较频繁，呈细碎摆动"

            return "整体以小幅摆动为主"

        segment_lines: list[str] = []
        no_change_local = bool(np.allclose(x, x[0]))
        seg_stds = np.array([float(s.get("std", 0.0)) for s in segments], dtype=float)

        if no_change_local:
            segment_lines = [
                f"各分段数值均保持在 {fmt_num(g_mean)} 附近，窗口内未见可辨别的阶段性变化。",
            ]
        elif periodic_preferred_flag:
            cycle_len = min(max(int(best_lag), 1), L)
            cycle_values = x[:cycle_len]
            half = max(1, len(segments) // 2)
            early_amp = float(np.mean(seg_stds[:half])) if half > 0 else 0.0
            late_amp = float(np.mean(seg_stds[half:])) if len(segments) > half else early_amp
            amp_ratio = late_amp / max(early_amp, 1e-6)
            if amp_ratio >= 1.25:
                amp_note = "后半段周期振幅较前半段更大"
            elif amp_ratio <= 0.80:
                amp_note = "后半段周期振幅较前半段有所收敛"
            else:
                amp_note = "前后时段的周期振幅整体接近"
            wave_label, high_ratio = cycle_wave_profile(x, int(best_lag))
            high_pct = int(round(100.0 * high_ratio))
            periodic_shape_text = (
                f"单周期通常{cycle_shape_phrase(cycle_values)}，波峰形态更接近{wave_label}，"
                f"高位段占比约 {high_pct}%，单周期取值大致在 {fmt_num(float(period_representative_bounds(x, int(best_lag))[0]))} 到 {fmt_num(float(period_representative_bounds(x, int(best_lag))[1]))} 之间"
            )
            segment_lines = [
                f"该窗口内存在稳定重复结构，以周期性变化为主（{period_cycle}，相关系数 corr≈{best_corr:.2f}），{periodic_shape_text}，{amp_note}。",
            ]
        else:
            value_span = max(g_max - g_min, 1e-6)

            def make_phase_item(start_idx: int, end_idx: int) -> dict:
                seg_vals = x[start_idx : end_idx + 1]
                seg_mean = float(np.mean(seg_vals))
                seg_std = float(np.std(seg_vals))
                seg_trend_label, _ = classify_global_trend(seg_vals)
                if g_std <= 0:
                    vol_level_local = "low"
                else:
                    ratio = seg_std / g_std
                    if ratio < 0.7:
                        vol_level_local = "low"
                    elif ratio < 1.3:
                        vol_level_local = "medium"
                    else:
                        vol_level_local = "high"
                return {
                    "idx": -1,
                    "start": int(start_idx),
                    "end": int(end_idx),
                    "len": int(end_idx - start_idx + 1),
                    "mean": seg_mean,
                    "std": seg_std,
                    "vol_level": vol_level_local,
                    "trend_label": seg_trend_label,
                }

            def detect_single_turn_structure(vals: np.ndarray) -> dict | None:
                if periodic_preferred_flag or oscillation_flag:
                    return None
                if len(events) >= 2:
                    return None
                arr = np.asarray(vals, dtype=float)
                n = int(arr.size)
                if n < 18:
                    return None

                lo = float(np.min(arr))
                hi = float(np.max(arr))
                span = max(hi - lo, 1e-6)

                win = 5 if n >= 24 else 3
                if win % 2 == 0:
                    win += 1
                sm = pd.Series(arr).rolling(window=win, center=True, min_periods=1).mean().to_numpy()
                d = np.diff(sm)
                if d.size < 8:
                    return None

                eps = 0.015 * span
                sign = np.zeros_like(d)
                sign[d > eps] = 1
                sign[d < -eps] = -1
                nz = sign[sign != 0]
                if nz.size < max(5, n // 7):
                    return None
                flips = int(np.sum(nz[1:] != nz[:-1])) if nz.size >= 2 else 0
                if flips > 4:
                    return None

                lower = int(0.20 * (n - 1))
                upper = int(0.80 * (n - 1))

                def directional_ratio(part: np.ndarray, positive: bool) -> float:
                    active = part[np.abs(part) > eps]
                    if active.size == 0:
                        return 0.0
                    if positive:
                        return float(np.mean(active > 0))
                    return float(np.mean(active < 0))

                def slope_ratio(part: np.ndarray) -> float:
                    if part.size < 4:
                        return 0.0
                    t = np.arange(part.size, dtype=float)
                    coef = np.polyfit(t, part, 1)
                    return float(coef[0]) / span

                valley_idx = int(np.argmin(sm))
                if lower <= valley_idx <= upper:
                    pre = d[:valley_idx]
                    post = d[valley_idx:]
                    if pre.size >= 4 and post.size >= 4:
                        pre_down = directional_ratio(pre, positive=False)
                        post_up = directional_ratio(post, positive=True)
                        pre_slope = slope_ratio(sm[: valley_idx + 1])
                        post_slope = slope_ratio(sm[valley_idx:])
                        depth = (float(sm[0]) - float(sm[valley_idx])) / span
                        rebound = (float(sm[-1]) - float(sm[valley_idx])) / span
                        if (
                            pre_down >= 0.58
                            and post_up >= 0.58
                            and pre_slope <= -0.006
                            and post_slope >= 0.006
                            and depth >= 0.10
                            and rebound >= 0.10
                        ):
                            return {"type": "valley", "turn_idx": valley_idx}

                peak_idx = int(np.argmax(sm))
                if lower <= peak_idx <= upper:
                    pre = d[:peak_idx]
                    post = d[peak_idx:]
                    if pre.size >= 4 and post.size >= 4:
                        pre_up = directional_ratio(pre, positive=True)
                        post_down = directional_ratio(post, positive=False)
                        pre_slope = slope_ratio(sm[: peak_idx + 1])
                        post_slope = slope_ratio(sm[peak_idx:])
                        rise = (float(sm[peak_idx]) - float(sm[0])) / span
                        retreat = (float(sm[peak_idx]) - float(sm[-1])) / span
                        if (
                            pre_up >= 0.58
                            and post_down >= 0.58
                            and pre_slope >= 0.006
                            and post_slope <= -0.006
                            and rise >= 0.10
                            and retreat >= 0.10
                        ):
                            return {"type": "peak", "turn_idx": peak_idx}

                return None
            vol_rank = {"low": 0, "mid": 1, "high": 2}
            groups: list[dict] = []
            for seg in segments:
                if not groups:
                    groups.append({"items": [seg]})
                    continue
                prev = groups[-1]["items"][-1]
                mean_gap = abs(float(seg.get("mean", 0.0)) - float(prev.get("mean", 0.0))) / value_span
                prev_std = max(float(prev.get("std", 0.0)), 1e-6)
                cur_std = max(float(seg.get("std", 0.0)), 1e-6)
                std_shift = max(cur_std / prev_std, prev_std / cur_std) - 1.0
                dir_shift = seg_dir(seg) != seg_dir(prev)
                amp_shift = abs(vol_rank.get(seg_amp(seg), 1) - vol_rank.get(seg_amp(prev), 1)) >= 1
                split_here = (
                    mean_gap >= 0.16
                    or std_shift >= 0.65
                    or (dir_shift and mean_gap >= 0.09)
                    or (amp_shift and (mean_gap >= 0.10 or std_shift >= 0.45))
                )
                if split_here:
                    groups.append({"items": [seg]})
                else:
                    groups[-1]["items"].append(seg)

            merged: list[dict] = []
            for grp in groups:
                if not merged:
                    merged.append(grp)
                    continue
                left_items = merged[-1]["items"]
                right_items = grp["items"]
                left_mean = float(np.mean([float(xi.get("mean", 0.0)) for xi in left_items]))
                right_mean = float(np.mean([float(xi.get("mean", 0.0)) for xi in right_items]))
                left_std = float(np.mean([float(xi.get("std", 0.0)) for xi in left_items]))
                right_std = float(np.mean([float(xi.get("std", 0.0)) for xi in right_items]))
                mean_gap = abs(right_mean - left_mean) / value_span
                std_shift = max(right_std / max(left_std, 1e-6), left_std / max(right_std, 1e-6)) - 1.0
                left_dir = seg_dir(left_items[-1])
                right_dir = seg_dir(right_items[0])
                close_enough = (
                    (mean_gap < 0.08 and std_shift < 0.30 and left_dir == right_dir)
                    or (mean_gap < 0.04 and std_shift < 0.20)
                )
                if close_enough:
                    merged[-1]["items"].extend(right_items)
                else:
                    merged.append(grp)
            groups = merged

            if not groups:
                groups = [{"items": list(segments)}]

            # 对候选阶段再做一次全局一致性审查：若切分证据偏弱，则回收为单段直述。
            single_turn_info = detect_single_turn_structure(x)
            if len(groups) > 1:
                significant_switches = 0
                for gi in range(1, len(groups)):
                    left_items = groups[gi - 1]["items"]
                    right_items = groups[gi]["items"]
                    left_mean = float(np.mean([float(xi.get("mean", 0.0)) for xi in left_items]))
                    right_mean = float(np.mean([float(xi.get("mean", 0.0)) for xi in right_items]))
                    left_std = float(np.mean([float(xi.get("std", 0.0)) for xi in left_items]))
                    right_std = float(np.mean([float(xi.get("std", 0.0)) for xi in right_items]))
                    mean_gap = abs(right_mean - left_mean) / value_span
                    std_shift = max(right_std / max(left_std, 1e-6), left_std / max(right_std, 1e-6)) - 1.0
                    left_dir = seg_dir(left_items[-1])
                    right_dir = seg_dir(right_items[0])
                    if (
                        mean_gap >= 0.12
                        or std_shift >= 0.45
                        or (left_dir != right_dir and mean_gap >= 0.06)
                    ):
                        significant_switches += 1

                net_change_ratio = abs(float(x[-1]) - float(x[0])) / value_span
                weak_evidence = bool(significant_switches == 0)
                borderline_two_stage = bool(
                    len(groups) == 2
                    and significant_switches <= 1
                    and net_change_ratio < 0.12
                    and not events
                    and not oscillation_flag
                )
                if (weak_evidence or borderline_two_stage) and single_turn_info is None:
                    groups = [{"items": list(segments)}]

            # 若全窗口是平滑单拐点结构，则优先分为两段，拐点处切分。
            if single_turn_info is not None and len(groups) != 2:
                turn_idx = int(single_turn_info.get("turn_idx", -1))
                if 4 <= turn_idx <= (L - 5):
                    left_item = make_phase_item(0, turn_idx)
                    right_item = make_phase_item(turn_idx, L - 1)
                    groups = [
                        {"items": [left_item]},
                        {"items": [right_item]},
                    ]

            # 短窗口若最终仅剩单段，则按前/中/后三段显式切分，避免“说了三段却不见边界”。
            if len(groups) == 1 and is_short_window and (not periodic_preferred_flag) and L >= 18:
                cut1 = max(6, int(round(L / 3.0)))
                cut2 = min(L - 6, int(round(2.0 * L / 3.0)))
                if cut2 - cut1 >= 6:
                    groups = [
                        {"items": [make_phase_item(0, cut1 - 1)]},
                        {"items": [make_phase_item(cut1, cut2 - 1)]},
                        {"items": [make_phase_item(cut2, L - 1)]},
                    ]

            for i, grp in enumerate(groups, start=1):
                first_seg = grp["items"][0]
                last_seg = grp["items"][-1]
                g_mean_val = float(np.mean([float(x.get("mean", 0.0)) for x in grp["items"]]))
                amp_levels = [seg_amp(x) for x in grp["items"]]
                a = max(set(amp_levels), key=amp_levels.count) if amp_levels else "mid"
                phase_start = int(first_seg["start"])
                phase_end = int(last_seg["end"])
                phase_seg = {"start": phase_start, "end": phase_end}
                phase_vals = x[phase_start : phase_end + 1]
                dirs = [seg_dir(x) for x in grp["items"]]
                d_fallback = max(set(dirs), key=dirs.count) if dirs else "flat"
                d, d_conf = infer_phase_direction(phase_vals, fallback=d_fallback)
                phase_phrase = group_phrase(d, a, d_conf)
                if d == "flat" and has_hump_shape(phase_vals):
                    phase_phrase = phase_phrase.replace("整体较平稳", "存在明显起伏变化")
                local_shape = phase_shape_phrase(phase_vals)
                if d == "flat" and d_conf == "low" and local_shape in {
                    "整体回落", "整体抬升", "阶梯式下降", "阶梯式上升", "锯齿式震荡下行", "锯齿式震荡上行"
                }:
                    local_shape = "走势较稳"
                phase_min = float(np.min(phase_vals)) if phase_vals.size else g_min
                phase_max = float(np.max(phase_vals)) if phase_vals.size else g_max
                phase_std = float(np.std(phase_vals)) if phase_vals.size else 0.0
                phase_span = max(phase_max - phase_min, 1e-6)
                std_ratio = phase_std / max(g_std, 1e-6)
                span_ratio = phase_span / max(g_max - g_min, 1e-6)
                if std_ratio >= 1.20 or span_ratio >= 0.55:
                    rel_vol_phrase = "相对全窗口属于较大波动"
                elif std_ratio <= 0.75 and span_ratio <= 0.30:
                    rel_vol_phrase = "相对全窗口属于较小波动"
                else:
                    rel_vol_phrase = "相对全窗口属于中等波动"

                shape_implies_osc = local_shape in {
                    "先升后降", "先降后升", "双峰起伏", "双谷起伏", "多峰往复震荡", "高频密集震荡"
                } or ("震荡" in local_shape)
                if shape_implies_osc:
                    phase_phrase = "存在明显起伏变化"
                    if rel_vol_phrase == "相对全窗口属于较小波动":
                        rel_vol_phrase = "相对全窗口属于中等波动"

                phase_start_val = float(phase_vals[0]) if phase_vals.size else g_mean_val
                phase_end_val = float(phase_vals[-1]) if phase_vals.size else g_mean_val

                local_lag, local_corr = estimate_period(phase_vals)
                local_periodic = bool(local_lag >= 8 and local_corr >= 0.45 and len(phase_vals) >= max(48, 2 * local_lag) and not is_micro_noise_period(phase_vals, local_lag, local_corr))
                periodic_tail = ""
                periodic_cycle_text = ""
                periodic_shape_text = ""
                if local_periodic:
                    _, local_cycle = build_period_phrases(int(local_lag))
                    cyc_len = min(max(int(local_lag), 1), len(phase_vals))
                    cyc_vals = phase_vals[:cyc_len]
                    wave_label, high_ratio = cycle_wave_profile(phase_vals, int(local_lag))
                    high_pct = int(round(100.0 * high_ratio))
                    periodic_cycle_text = local_cycle
                    periodic_shape_text = f"单周期通常{local_shape}，波峰形态更接近{wave_label}，高位段占比约 {high_pct}%，取值大致在 {fmt_num(float(period_representative_bounds(phase_vals, int(local_lag))[0]))} 到 {fmt_num(float(period_representative_bounds(phase_vals, int(local_lag))[1]))} 之间"
                    periodic_tail = (
                        f"；该阶段呈局部周期性起伏（{local_cycle}，相关系数 corr≈{local_corr:.2f}），"
                        f"波形更接近{wave_label}，单周期高位段占比约 {high_pct}%，"
                        f"单周期取值大致在 {fmt_num(float(period_representative_bounds(phase_vals, int(local_lag))[0]))} 到 {fmt_num(float(period_representative_bounds(phase_vals, int(local_lag))[1]))} 之间"
                    )
                shape_clause = (
                    (f"并{local_shape}，" if str(local_shape).startswith("以") else f"并呈{local_shape}，")
                    if not local_periodic
                    else ""
                )
                if (not local_periodic) and local_shape in {"走势较稳", "变化较小", "轻微起伏"} and rel_vol_phrase in {"相对全窗口属于中等波动", "相对全窗口属于较大波动"}:
                    shape_clause = "并以起伏为主，"
                if len(groups) == 1 and local_periodic:
                    segment_lines.append(
                        f"该窗口内存在稳定重复结构，以周期性变化为主（{periodic_cycle_text}，相关系数 corr≈{local_corr:.2f}），{periodic_shape_text}。"
                    )
                elif len(groups) == 1:
                    short_curve_text = short_window_curve_detail(phase_vals)
                    short_curve_clause = f"，{short_curve_text}" if short_curve_text else ""
                    segment_lines.append(
                        f"该窗口内未识别到明确阶段切换，整体变化连续，{phase_phrase}，{shape_clause}{rel_vol_phrase}{short_curve_clause}。"
                    )
                else:
                    segment_lines.append(
                        f"第{i}阶段（{seg_interval_text(phase_seg)}）均值约 {fmt_num(g_mean_val)}，起点约 {fmt_num(phase_start_val)}，终点约 {fmt_num(phase_end_val)}，段内大致在 {fmt_num(phase_min)} 到 {fmt_num(phase_max)} 之间，{phase_phrase}，{shape_clause}{rel_vol_phrase}{periodic_tail}。"
                    )

            if oscillation_flag and not plateau_local_flag:
                segment_lines.append("补充：该窗口阶段间方向切换较频繁，整体以震荡节律为主。")
            elif plateau_local_flag:
                active_seg = plateau_local_info.get("active_seg", max(segments, key=lambda s: float(s.get("std", 0.0))))
                segment_lines.append(
                    f"补充：平台特征较明显，波动相对集中的区间在 {seg_interval_text(active_seg)}。"
                )

        lines.append("分段观察：\n" + "\n".join(segment_lines))

    if events:
        event_desc = []
        for ev in events[:2]:
            s, e = ev["start"], ev["end"]
            v = ev["peak_value"]
            z = ev["z_max"]
            local_base = float(ev.get("local_baseline", np.nan))
            if np.isfinite(local_base):
                baseline = float(local_base)
            else:
                prev_idx = max(0, s - 5)
                baseline = float(x[prev_idx:s].mean()) if s > 0 else g_mean
            signed_delta = ev.get("signed_delta")
            if signed_delta is None:
                signed_delta = float(v) - float(baseline)
            signed_delta = float(signed_delta)
            change_word = "激增" if signed_delta >= 0 else "暴跌"
            compare_word = "偏高" if signed_delta >= 0 else "偏低"
            pos_text = f"第 {s} 点" if int(s) == int(e) else f"第 {s} 到 {e} 点"
            event_desc.append(
                f"{pos_text}，{var_name} 出现一次局部{change_word}（峰值约 {v:.3g}，相较局部基线 {baseline:.3g} 明显{compare_word}，标准化偏离 |z|≈{abs(z):.2f}）"
            )
        lines.append("异常点：\n" + "；\n".join(event_desc) + "。")
    else:
        lines.append(f"异常点：\n本窗口内 {var_name} 未出现特别突出的极端高峰或低谷。")

    if oscillation_flag and not plateau_local_flag:
        lines.append(
            f"整体结论：\n这一窗口中的 {var_name} 以高频震荡为主，短时起伏频繁，未形成持续单向趋势。"
        )
    else:
        summary_phrase = overall_summary_phrase()
        summary_text = summary_phrase[2:] if summary_phrase.startswith("整体") else summary_phrase
        summary_text = enforce_summary_consistency(
            summary_text=summary_text,
            segments=segments,
            target=x,
            slope_norm=slope_norm,
            events=events,
            periodic_preferred_flag=periodic_preferred_flag,
            short_window=is_short_window,
        )
        lines.append(
            f"整体结论：\n这一窗口中的 {var_name} {summary_text}，且{vol_phrase(vol_level)}。"
        )

    description = "\n\n".join(lines)
    features = {
        "global": {"length": L, "min": g_min, "max": g_max, "mean": g_mean, "std": g_std,
               "vol_level": vol_level, "vol_ratio": vol_ratio,
               "vol_level_legacy": legacy_vol_level, "vol_ratio_legacy": legacy_vol_ratio,
               "vol_rel_band": float(vol_meta.get("rel_band", 0.0)),
               "vol_diff_energy": float(vol_meta.get("diff_energy", 0.0)),
               "window_profile": "short" if is_short_window else ("long" if is_long_window else "normal"),
               "trend_label": trend_label, "slope_norm": slope_norm},
        "periodicity": {
            "best_lag": best_lag,
            "best_corr": best_corr,
            "global_periodic": bool(period_profile.get("is_global_periodic", False)),
            "cycle_consistency": float(period_profile.get("cycle_consistency", 0.0)),
            "first_half_corr": float(period_profile.get("first_half_corr", 0.0)),
            "second_half_corr": float(period_profile.get("second_half_corr", 0.0)),
            "harmonic_corr": float(period_profile.get("harmonic_corr", 0.0)),
        },
        "segments": segments, "events": events,
    }
    return description, features


def generate_ucr_jsonl(
    arff_path: str,
    output_path: str,
    window_lengths=(512, 1024),
    step_ratio: float = 0.5,
    max_samples: int | None = None,
    main_channels: list[int] | None = None,
):
    df, targets, attrs, is_multivariate, n_channels, mv_series_list = load_ucr_arff(arff_path)
    n_instances = len(mv_series_list) if is_multivariate else df.shape[0]
    series_len = int(mv_series_list[0].shape[0]) if is_multivariate else int(df.shape[1])
    dataset_name = Path(arff_path).parent.name
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_count = 0
    main_channels = main_channels or [0]
    with out_path.open('w', encoding='utf-8') as f:
        for idx in range(n_instances):
            targ = targets[idx]
            if is_multivariate:
                mat = np.asarray(mv_series_list[idx], dtype=float)  # (time, channel)
                # 按通道分别插值补齐
                for ch in range(mat.shape[1]):
                    s = pd.Series(mat[:, ch], dtype='float64')
                    if s.notna().sum() == 0:
                        continue
                    s = s.interpolate(limit_direction='both').ffill().bfill()
                    mat[:, ch] = s.to_numpy(dtype=float)

                cur_len = int(mat.shape[0])
                for L in window_lengths:
                    if L > cur_len:
                        continue
                    step = max(1, int(L * step_ratio))
                    for start in sliding_window_indices(cur_len, L, step):
                        end = start + L
                        win = mat[start:end, :]
                        selected = [int(c) for c in main_channels if 0 <= int(c) < win.shape[1]]
                        if not selected:
                            selected = [0]
                        main_channel = int(selected[0])
                        desc, feat = describe_window_multivariate(win, dataset_name, idx, targ, main_channel=main_channel)
                        sample = {
                            "id": f"{dataset_name}_{Path(arff_path).stem}_idx{idx}_L{L}_start{start}",
                            "dataset": dataset_name,
                            "task": "classification",
                            "source_file": Path(arff_path).name,
                            "instance_index": int(idx),
                            "class_label": targ,
                            "is_multivariate": True,
                            "n_channels": int(win.shape[1]),
                            "main_channel": int(np.clip(main_channel, 0, max(0, win.shape[1] - 1))),
                            "target_col": f"ch{int(np.clip(main_channel, 0, max(0, win.shape[1] - 1)))}",
                            "target_cols": [f"ch{c}" for c in selected],
                            "window_length": L,
                            "start_index": int(start),
                            "end_index": int(end - 1),
                            "values": win.tolist(),
                            "features": feat,
                            "descriptions": [desc],
                        }
                        safe_sample = make_json_safe(sample)
                        f.write(json.dumps(safe_sample, ensure_ascii=False, allow_nan=False))
                        f.write("\n")
                        sample_count += 1
                        if max_samples is not None and sample_count >= max_samples:
                            print(f"达到 max_samples={max_samples}，提前停止。")
                            return
            else:
                series_raw = df.iloc[idx].to_numpy(dtype=float)
                series_s = pd.Series(series_raw, dtype='float64')
                if series_s.notna().sum() == 0:
                    continue
                series_s = series_s.interpolate(limit_direction='both').ffill().bfill()
                if series_s.notna().sum() == 0:
                    continue
                series = series_s.to_numpy(dtype=float)
                for L in window_lengths:
                    step = max(1, int(L * step_ratio))
                    for start in sliding_window_indices(series_len, L, step):
                        end = start + L
                        win = series[start:end]
                        desc, feat = describe_window_series(win, dataset_name, idx, targ)
                        sample = {
                            "id": f"{dataset_name}_{Path(arff_path).stem}_idx{idx}_L{L}_start{start}",
                            "dataset": dataset_name,
                            "task": "classification",
                            "source_file": Path(arff_path).name,
                            "instance_index": int(idx),
                            "class_label": targ,
                            "is_multivariate": False,
                            "n_channels": 1,
                            "window_length": L,
                            "start_index": int(start),
                            "end_index": int(end-1),
                            "values": win.tolist(),
                            "features": feat,
                            "descriptions": [desc],
                        }
                        safe_sample = make_json_safe(sample)
                        f.write(json.dumps(safe_sample, ensure_ascii=False, allow_nan=False))
                        f.write("\n")
                        sample_count += 1
                        if max_samples is not None and sample_count >= max_samples:
                            print(f"达到 max_samples={max_samples}，提前停止。")
                            return
    print(f"已生成 {sample_count} 个样本，保存在 {out_path}。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="生成UEA/UCR归档中某个ARFF文件的中文文本描述（JSONL格式）。")
    parser.add_argument('--arff_path', type=str, required=True, help='要处理的 .arff 文件路径')
    parser.add_argument('--output_path', type=str, default=None, help='输出jsonl路径（默认：脚本目录下 dataset_descriptions.jsonl）')
    parser.add_argument('--window_lengths', type=int, nargs='+', default=[512, 1024], help='窗口长度列表，例如 512 1024')
    parser.add_argument('--step_ratio', type=float, default=0.5, help='滑动步长相对窗口长度的比例')
    parser.add_argument('--max_samples', type=int, default=None, help='最多生成多少个样本，默认不限制')
    parser.add_argument('--main_channels', type=int, nargs='+', default=[0], help='多变量场景下主通道列表（默认 0），描述使用第一个通道，格式与其他多变量脚本一致')
    parser.add_argument('--main_channel', type=int, default=None, help='兼容旧参数：单个主通道索引')
    args = parser.parse_args()
    if args.output_path is None:
        script_dir = Path(__file__).resolve().parent
        win_tag = "-".join(str(w) for w in args.window_lengths)
        default_name = f"{Path(args.arff_path).parent.name}_{Path(args.arff_path).stem}_L{win_tag}_descriptions.jsonl"
        args.output_path = str(script_dir / default_name)
        print(f"⚙️ 未指定 --output_path，使用默认输出：{args.output_path}")
    generate_ucr_jsonl(
        arff_path=args.arff_path,
        output_path=args.output_path,
        window_lengths=args.window_lengths,
        step_ratio=args.step_ratio,
        max_samples=args.max_samples,
        main_channels=([args.main_channel] if args.main_channel is not None else args.main_channels),
    )











