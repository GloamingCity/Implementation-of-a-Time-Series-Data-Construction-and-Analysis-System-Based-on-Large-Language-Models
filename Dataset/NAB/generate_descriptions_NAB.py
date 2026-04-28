# Usage
# cd D:\Users\Win\Desktop\Study\毕业设计-面向大语言模型的时序推理数据构建及分析系统实现\数据收集\数据集\NAB
# python generate_descriptions_NAB.py --csv_path realTraffic/speed_7578.csv --target_col value --max_samples 5

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_COMMON_ROOT = Path(__file__).resolve().parents[1]
if str(_COMMON_ROOT) not in sys.path:
    sys.path.append(str(_COMMON_ROOT))
from common_desc_adapt import (
    smooth_signal_for_window,
    robust_classify_global_trend,
    classify_volatility_adaptive,
    detect_change_boundaries,
    detect_local_extreme_events,
    segment_features_by_boundaries as common_segment_features_by_boundaries,
    enforce_summary_consistency,
)


def make_json_safe(obj):
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


def load_nab(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")

    time_col = None
    if "timestamp" in df.columns:
        time_col = "timestamp"
    else:
        for c in df.columns:
            low = c.lower()
            if "time" in low or "date" in low:
                time_col = c
                break

    if time_col is not None:
        ts = pd.to_datetime(df[time_col], errors="coerce")
        keep = ~ts.isna()
        df = df.loc[keep].copy()
        ts = ts.loc[keep]
        df = df.drop(columns=[time_col])
        df.index = ts

    return df


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
    # 为避免漏检日周期等短周期，动态下限最高不超过 24。
    min_lag = max(int(min_lag), min(24, max(8, n // 32)))
    if n < max(20, min_lag * 2):
        return 0, 0.0
    if np.allclose(x, x[0]):
        return 0, 0.0

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
    """评估全局周期是否显著，避免局部重复被误判为全窗口周期。"""
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


def infer_time_step(index) -> pd.Timedelta | None:
    if len(index) < 2 or not pd.api.types.is_datetime64_any_dtype(index):
        return None
    deltas = pd.Series(index[1:] - index[:-1])
    deltas = deltas[deltas > pd.Timedelta(0)]
    if deltas.empty:
        return None
    return deltas.mode().iloc[0]


def format_timedelta_cn(delta: pd.Timedelta | None) -> str | None:
    if delta is None:
        return None
    total_seconds = int(delta.total_seconds())
    if total_seconds <= 0:
        return None

    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)

    parts = []
    if days:
        parts.append(f"{days}天")
    if hours:
        parts.append(f"{hours}小时")
    if minutes:
        parts.append(f"{minutes}分钟")
    if seconds:
        parts.append(f"{seconds}秒")
    return "".join(parts) if parts else None


def build_period_phrases(best_lag: int, step_delta: pd.Timedelta | None = None):
    duration = format_timedelta_cn(step_delta * int(best_lag)) if step_delta is not None else None
    if duration:
        return f"约每隔 {duration}", f"约 {duration}"
    return f"约每隔 {best_lag} 个时间点", f"约 {best_lag} 个时间点"


def estimate_adaptive_segment_count(x: np.ndarray) -> int:
    arr = np.asarray(x, dtype=float)
    n = int(arr.size)
    if n < 32:
        return 2
    if np.allclose(arr, arr[0]):
        return 3

    smooth_w = max(5, int(n // 64))
    if smooth_w % 2 == 0:
        smooth_w += 1
    kernel = np.ones(smooth_w, dtype=float) / float(smooth_w)
    arr_sm = np.convolve(arr, kernel, mode="same")

    d = np.diff(arr_sm)
    if d.size == 0:
        return 3
    std_sm = float(np.std(arr_sm))
    mean_abs = float(np.mean(np.abs(arr_sm)))
    rel_vol = std_sm / max(mean_abs, 1.0)

    eps = max(1e-8, 0.10 * std_sm)
    sign = np.zeros_like(d)
    sign[d > eps] = 1
    sign[d < -eps] = -1
    nz = sign[sign != 0]
    turn_cnt = int(np.sum(nz[1:] != nz[:-1])) if nz.size >= 2 else 0

    turn_density = turn_cnt / max(n - 2, 1)
    by_turn = 3 + int(round(min(0.65, turn_density) * 12))
    vol_bonus = 1 if rel_vol >= 0.16 else 0
    by_length = max(4, int(np.ceil(n / 64)) + 2)
    seg_n = min(max(by_turn + vol_bonus, 3), by_length)
    return int(max(3, seg_n))


def detect_flat_or_square_wave(x: np.ndarray) -> str:
    """
    返回:
    - "flat": 近似直线
    - "square": 近似严格方波（双平台切换）
    - "none": 其他
    """
    arr = np.asarray(x, dtype=float)
    if arr.size < 16:
        return "none"

    span = float(np.max(arr) - np.min(arr))
    std = float(np.std(arr))
    if span <= 1e-8 or std <= 1e-8:
        return "flat"

    # 直线/几乎无波动：标准差相对幅度极小
    if (std / max(span, 1e-8)) <= 0.015:
        return "flat"

    # 严格方波判定：两级平台占主导 + 平台内噪声小 + 切换次数有限
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
    low_std = float(np.std(low_vals))
    high_std = float(np.std(high_vals))
    level_gap = float(abs(np.mean(high_vals) - np.mean(low_vals)))
    if level_gap <= 1e-8:
        return "none"

    noise_ratio = max(low_std, high_std) / level_gap
    if noise_ratio <= 0.10:
        return "square"
    return "none"


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
    seg_n = trans + 1
    return int(min(4, max(2, seg_n)))


def segment_features(x: np.ndarray, num_segments: int, global_std: float):
    L = len(x)
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


def segment_features_by_boundaries(x: np.ndarray, boundaries: list[int], global_std: float, trend_func=None):
    trend_fn = trend_func if callable(trend_func) else classify_global_trend
    return common_segment_features_by_boundaries(
        x,
        boundaries=boundaries,
        global_std=global_std,
        trend_func=trend_fn,
    )


def detect_dominant_linear_prefix(x: np.ndarray) -> tuple[bool, int]:
    """检测窗口前段是否存在明显的长线性或准线性漂移段，返回(是否命中, 结束下标)。"""
    arr = np.asarray(x, dtype=float)
    n = int(arr.size)
    if n < 96:
        return False, -1

    span = float(np.max(arr) - np.min(arr))
    if span <= 1e-8:
        return False, -1

    w = max(5, n // 64)
    if w % 2 == 0:
        w += 1
    ker = np.ones(w, dtype=float) / float(w)
    sm = np.convolve(arr, ker, mode="same")

    min_len = max(72, n // 8)
    max_len = min(int(0.58 * n), n - max(64, n // 6))
    if max_len <= min_len:
        return False, -1

    best_end = -1
    best_score = 0.0
    for e in range(min_len, max_len + 1):
        seg = sm[:e]
        d = np.diff(seg)
        if d.size < 8:
            continue
        eps = max(1e-8, 0.03 * np.std(seg))
        pos = float(np.mean(d > eps))
        neg = float(np.mean(d < -eps))
        mono_ratio = max(pos, neg)
        if mono_ratio < 0.42:
            continue

        move_ratio = abs(float(seg[-1] - seg[0])) / span
        if move_ratio < 0.11:
            continue

        t = np.arange(e, dtype=float)
        coef = np.polyfit(t, seg, 1)
        pred = coef[0] * t + coef[1]
        ss_res = float(np.sum((seg - pred) ** 2))
        ss_tot = float(np.sum((seg - np.mean(seg)) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
        if r2 < 0.20:
            continue

        score = move_ratio * mono_ratio * max(r2, 0.0)
        if score > best_score:
            best_score = score
            best_end = e - 1

    return (best_end >= 0 and best_score >= 0.012), int(best_end)


def detect_zscore_events(x: np.ndarray, z_thr: float = 2.5):
    return detect_local_extreme_events(x, z_thr=z_thr)


def describe_window_nab(win_df: pd.DataFrame, target_col: str, window_start: int):
    assert target_col in win_df.columns, f"目标列 {target_col} 不在窗口数据中"
    target = win_df[target_col].to_numpy(dtype=float)
    L = len(target)

    g_min, g_max = float(target.min()), float(target.max())
    g_mean = float(target.mean())
    g_std = float(target.std())
    is_long_window = L > 300
    is_short_window = L < 200
    trend_signal = smooth_signal_for_window(target, short_window=is_short_window) if is_short_window else target
    legacy_vol_level, legacy_vol_ratio = classify_volatility(g_std, g_mean)
    vol_level, vol_ratio, vol_meta = classify_volatility_adaptive(
        target,
        short_window=is_short_window,
        long_window=is_long_window,
    )
    trend_label, slope_norm = classify_global_trend(trend_signal)
    best_lag, best_corr = estimate_period(target)
    period_profile = evaluate_global_periodicity(target, best_lag, best_corr)
    step_delta = infer_time_step(win_df.index)
    _, period_cycle = build_period_phrases(best_lag, step_delta)

    special_shape = detect_flat_or_square_wave(target)
    has_linear_prefix, linear_prefix_end = detect_dominant_linear_prefix(target)
    if special_shape == "flat":
        segments = []
    elif special_shape == "square":
        seg_n = estimate_square_segment_count(target)
        segments = segment_features(target, num_segments=seg_n, global_std=g_std)
    elif has_linear_prefix:
        # 前段线性变化显著：单独切出该段，其余部分再拆成两段
        b1 = int(max(16, min(linear_prefix_end + 1, L - 32)))
        rem = L - b1
        b2 = b1 + max(16, rem // 2)
        boundaries = [0, b1, min(b2, L - 16), L]
        segments = segment_features_by_boundaries(
            target,
            boundaries=boundaries,
            global_std=g_std,
            trend_func=classify_global_trend,
        )
    else:
        boundaries = detect_change_boundaries(
            trend_signal,
            short_window=is_short_window,
            long_window=is_long_window,
        )
        segments = segment_features_by_boundaries(
            target,
            boundaries=boundaries,
            global_std=g_std,
            trend_func=classify_global_trend,
        )
        if not segments:
            seg_n = estimate_adaptive_segment_count(target)
            segments = segment_features(target, num_segments=seg_n, global_std=g_std)
    events = detect_zscore_events(target, z_thr=2.5)

    has_real_time = pd.api.types.is_datetime64_any_dtype(win_df.index)

    def fmt_num(v: float) -> str:
        av = abs(float(v))
        if av >= 100:
            return f"{float(v):.0f}"
        if av >= 1:
            return f"{float(v):.2f}"
        if av >= 0.01:
            return f"{float(v):.4f}"
        return f"{float(v):.3g}"

    def fmt_interval(s: int, e: int) -> str:
        if has_real_time:
            left = pd.Timestamp(win_df.index[int(s)]).strftime("%Y-%m-%d %H:%M")
            right = pd.Timestamp(win_df.index[int(e)]).strftime("%Y-%m-%d %H:%M")
            return f"{left} 到 {right}"
        abs_s = int(window_start) + int(s)
        abs_e = int(window_start) + int(e)
        return f"时间点 {abs_s} 到 {abs_e}"

    diff_abs = np.abs(np.diff(target))
    jump_thr = max(
        0.30 * max(g_max - g_min, 1e-6),
        (2.5 * float(np.std(np.diff(target)))) if diff_abs.size else 0.0,
    )
    abrupt_switch_count = int(np.sum(diff_abs >= jump_thr)) if diff_abs.size else 0
    abrupt_switch_flag = bool(abrupt_switch_count >= 2)
    periodic_cycle_flag = bool(period_profile.get("is_global_periodic", False) and not np.allclose(target, target[0]))
    if (not periodic_cycle_flag) and (special_shape == "square" and best_lag >= 8 and best_corr >= 0.45):
        periodic_cycle_flag = not is_micro_noise_period(target, best_lag, best_corr)
    platform_switch_segment_flag = bool(special_shape == "square" or abrupt_switch_flag)
    periodic_score = (0.55 if periodic_cycle_flag else 0.0) + min(max((best_corr - 0.30) / 0.50, 0.0), 0.35)
    segment_score = (0.45 if platform_switch_segment_flag else 0.0) + (0.15 if abrupt_switch_count >= 4 else 0.0)
    if periodic_cycle_flag and is_micro_noise_period(target, best_lag, best_corr):
        periodic_cycle_flag = False
    periodic_preferred_flag = bool(periodic_cycle_flag and periodic_score >= segment_score)
    
    start_val, end_val = float(target[0]), float(target[-1])

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
        if high_ratio >= 0.42 and low_ratio >= 0.20 and sharp >= 0.10:
            wave = "近似方波型"
        elif sign_changes <= 2 and high_ratio <= 0.34 and low_ratio <= 0.34:
            wave = "近似正弦波型"
        else:
            wave = "周期起伏型"
        return wave, high_ratio

    def extract_plateau_runs(vals: np.ndarray) -> list[tuple[int, int, int]]:
        """基于高低平台状态提取连续区间，返回[(state,start,end), ...]，state:0低位/1高位。"""
        arr = np.asarray(vals, dtype=float)
        if arr.size < 8:
            return []
        lo = float(np.quantile(arr, 0.25))
        hi = float(np.quantile(arr, 0.75))
        if hi - lo <= 1e-8:
            return []
        mid = 0.5 * (lo + hi)
        raw_state = np.where(arr >= mid, 1, 0)

        # 轻度平滑状态，避免短时抖动把平台切碎。
        k = 5 if arr.size >= 32 else 3
        half = k // 2
        sm_state = raw_state.copy()
        for i in range(arr.size):
            l = max(0, i - half)
            r = min(arr.size, i + half + 1)
            sm_state[i] = 1 if np.mean(raw_state[l:r]) >= 0.5 else 0

        runs = []
        s = 0
        cur = int(sm_state[0])
        for i in range(1, int(arr.size)):
            if int(sm_state[i]) != cur:
                runs.append((cur, s, i - 1))
                s = i
                cur = int(sm_state[i])
        runs.append((cur, s, int(arr.size) - 1))

        # 合并过短片段（闪烁），优先并入邻接的长片段。
        min_len = max(3, int(arr.size // 64))
        merged = []
        for st, rs, re in runs:
            if merged and (re - rs + 1) < min_len:
                pst, ps, pe = merged[-1]
                merged[-1] = (pst, ps, re)
            else:
                merged.append((st, rs, re))

        final_runs = []
        for st, rs, re in merged:
            if final_runs and final_runs[-1][0] == st:
                _, fs, fe = final_runs[-1]
                final_runs[-1] = (st, fs, re)
            else:
                final_runs.append((st, rs, re))
        return final_runs

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
        up_ratio = float(np.mean(d > eps))
        down_ratio = float(np.mean(d < -eps))
        p1, p2, p3 = [float(np.mean(xp)) for xp in np.array_split(x, 3)]
        net = float(x[-1] - x[0])
        if p1 < p2 > p3 and (p2 - min(p1, p3)) >= 0.10:
            return "先升后降"
        if p1 > p2 < p3 and (max(p1, p3) - p2) >= 0.10:
            return "先降后升"
        if up_ratio >= 0.45 and down_ratio <= 0.20:
            return "整体抬升"
        if down_ratio >= 0.45 and up_ratio <= 0.20:
            return "整体回落"
        return "以震荡为主"

    def relative_vol_phrase(phase_vals: np.ndarray) -> str:
        phase_std = float(np.std(phase_vals)) if phase_vals.size else 0.0
        phase_span = float(np.max(phase_vals) - np.min(phase_vals)) if phase_vals.size else 0.0
        std_ratio = phase_std / max(g_std, 1e-6)
        span_ratio = phase_span / max(g_max - g_min, 1e-6)
        if std_ratio >= 1.20 or span_ratio >= 0.55:
            return "相对全窗口属于较大波动"
        if std_ratio <= 0.75 and span_ratio <= 0.30:
            return "相对全窗口属于较小波动"
        return "相对全窗口属于中等波动"

    if special_shape == "flat" or np.allclose(target, target[0]):
        trait = "整体保持恒定"
    elif special_shape == "square":
        trait = "整体在高低双平台间规则切换，呈近似方波"
    elif periodic_preferred_flag:
        trait = "整体呈明显周期性波动" if best_corr >= 0.7 else "整体呈周期性波动"
    elif trend_label in ("strong_up", "weak_up"):
        trait = "整体有上行倾向"
    elif trend_label in ("strong_down", "weak_down"):
        trait = "整体有回落倾向"
    else:
        trait = "整体以阶段性起伏为主"

    lines = []
    lines.append(
        "窗口概览：\n"
        + f"这段窗口覆盖范围为 {fmt_interval(0, L - 1)}，共 {L} 个观测点。"
        + f"{target_col} 的数值大致分布在 {fmt_num(g_min)} 到 {fmt_num(g_max)} 之间，平均约 {fmt_num(g_mean)}，起始点为 {fmt_num(start_val)}，结束点为 {fmt_num(end_val)}，{trait}。"
    )

    if segments:
        seg_lines = []
        if periodic_preferred_flag and len(segments) <= 1:
            cyc = target[: min(max(int(best_lag), 1), L)]
            wave_label, high_ratio = cycle_wave_profile(target, int(best_lag))
            high_pct = int(round(100.0 * high_ratio))
            seg_lines.append(
                f"该窗口未出现明确阶段切换，整体以周期性变化为主（{period_cycle}，相关系数 corr≈{best_corr:.2f}），"
                f"单周期取值大致在 {fmt_num(float(period_representative_bounds(target, int(best_lag))[0]))} 到 {fmt_num(float(period_representative_bounds(target, int(best_lag))[1]))} 之间，"
                f"波形更接近{wave_label}，高位段占比约 {high_pct}%。"
            )
        else:
            # 如果只有一个阶段，不用"第一阶段"这种模板
            if len(segments) == 1:
                seg = segments[0]
                tlabel = seg["trend_label"]
                if tlabel in ("strong_up", "weak_up"):
                    tphrase = "以抬升为主"
                elif tlabel in ("strong_down", "weak_down"):
                    tphrase = "以回落为主"
                else:
                    tphrase = "整体较平稳"

                vol = seg["vol_level"]
                if vol == "high":
                    vphrase = "波动偏强"
                elif vol == "low":
                    vphrase = "波动较小"
                else:
                    vphrase = "波动中等"

                s = int(seg["start"])
                e = int(seg["end"])
                phase_vals = target[s : e + 1]
                p_start = float(phase_vals[0]) if phase_vals.size else float(seg["mean"])
                p_end = float(phase_vals[-1]) if phase_vals.size else float(seg["mean"])
                p_min = float(np.min(phase_vals)) if phase_vals.size else float(seg["mean"])
                p_max = float(np.max(phase_vals)) if phase_vals.size else float(seg["mean"])
                local_shape = phase_shape_phrase(phase_vals)
                shape_clause = f"并{local_shape}" if str(local_shape).startswith("以") else f"并呈{local_shape}"

                # 避免逻辑矛盾：如果波动较小，就不说变化平缓，反之亦然
                if vphrase == "波动较小":
                    shape_clause = shape_clause.replace("走势较稳", "").replace("，，", "，").strip("，")
                if "走势较稳" in shape_clause and vphrase != "波动较小":
                    shape_clause = "以起伏为主"
                
                seg_lines.append(
                    f"该窗口未出现明确阶段切换，整体均值约 {fmt_num(seg['mean'])}，起点约 {fmt_num(p_start)}，终点约 {fmt_num(p_end)}，"
                    f"段内大致在 {fmt_num(p_min)} 到 {fmt_num(p_max)} 之间，{tphrase}，且{vphrase}，{shape_clause}。"
                )
            else:
                # 多个阶段的情况
                for i, seg in enumerate(segments, start=1):
                    phase_name = "第一阶段" if i == 1 else ("第二阶段" if i == 2 else ("第三阶段" if i == 3 else f"第{i}阶段"))
                    tlabel = seg["trend_label"]
                    if tlabel in ("strong_up", "weak_up"):
                        tphrase = "以抬升为主"
                    elif tlabel in ("strong_down", "weak_down"):
                        tphrase = "以回落为主"
                    else:
                        tphrase = "整体较平稳"

                    vol = seg["vol_level"]
                    if vol == "high":
                        vphrase = "波动偏强"
                    elif vol == "low":
                        vphrase = "波动较小"
                    else:
                        vphrase = "波动中等"

                    s = int(seg["start"])
                    e = int(seg["end"])
                    phase_vals = target[s : e + 1]
                    p_start = float(phase_vals[0]) if phase_vals.size else float(seg["mean"])
                    p_end = float(phase_vals[-1]) if phase_vals.size else float(seg["mean"])
                    p_min = float(np.min(phase_vals)) if phase_vals.size else float(seg["mean"])
                    p_max = float(np.max(phase_vals)) if phase_vals.size else float(seg["mean"])
                    local_shape = phase_shape_phrase(phase_vals)
                    rel_vol = relative_vol_phrase(phase_vals)
                    shape_clause = f"并{local_shape}" if str(local_shape).startswith("以") else f"并呈{local_shape}"

                    # 避免逻辑矛盾：如果波动较小，就不说变化平缓，反之亦然
                    if vphrase == "波动较小":
                        shape_clause = shape_clause.replace("走势较稳", "").replace("，，", "，").strip("，")
                    if "走势较稳" in shape_clause and rel_vol in {"相对全窗口属于中等波动", "相对全窗口属于较大波动"}:
                        shape_clause = "以起伏为主"
                    
                    seg_lines.append(
                        f"{phase_name}（{fmt_interval(s, e)}）均值约 {fmt_num(seg['mean'])}，起点约 {fmt_num(p_start)}，终点约 {fmt_num(p_end)}，"
                        f"段内大致在 {fmt_num(p_min)} 到 {fmt_num(p_max)} 之间，{tphrase}，且{vphrase}，{shape_clause}，{rel_vol}。"
                    )
            if has_linear_prefix:
                seg_lines.append("补充：窗口前段存在明显长线性变化，已将其单独作为一个阶段描述。")
            # 方向切换频繁时补一句，避免漏掉多阶段起伏特征
            diff = np.diff(target)
            sign = np.sign(diff)
            sign = sign[sign != 0]
            if sign.size >= 6:
                flip_ratio = float(np.sum(sign[1:] * sign[:-1] < 0)) / max(sign.size - 1, 1)
                if flip_ratio >= 0.32:
                    seg_lines.append("补充：该窗口阶段间方向切换较频繁，整体以震荡节律为主。")
        if seg_lines:
            lines.append("分段观察：\n" + "\n".join(seg_lines))

    if special_shape == "flat":
        lines.append("分段观察：\n该窗口近似水平直线，未检测到可辨别的阶段切换。")

    if events:
        ev_lines = []
        for ev in events[:2]:
            s = int(ev["start"])
            e = int(ev["end"])
            v = float(ev["peak_value"])
            z = abs(float(ev["z_max"]))
            local_base = float(ev.get("local_baseline", np.nan))
            if np.isfinite(local_base):
                baseline = float(local_base)
            else:
                prev_idx = max(0, s - 5)
                baseline = float(target[prev_idx:s].mean()) if s > 0 else g_mean
            signed_delta = ev.get("signed_delta")
            if signed_delta is None:
                signed_delta = float(v) - float(baseline)
            signed_delta = float(signed_delta)
            change_word = "激增" if signed_delta >= 0 else "暴跌"
            compare_word = "偏高" if signed_delta >= 0 else "偏低"
            ev_lines.append(
                f"{fmt_interval(s, e)} 出现一次局部{change_word}（峰值约 {fmt_num(v)}，相较局部基线 {fmt_num(baseline)} 明显{compare_word}，标准化偏离 |z|≈{z:.1f}）"
            )
        lines.append("异常点：\n" + "；\n".join(ev_lines) + "。")
    else:
        lines.append(f"异常点：\n本窗口内 {target_col} 未出现特别突出的尖峰或急跌。")

    if periodic_preferred_flag:
        overall = "呈明显周期性起伏" if best_corr >= 0.7 else "呈周期性起伏"
        overall = enforce_summary_consistency(
            summary_text=overall,
            segments=segments,
            target=target,
            slope_norm=slope_norm,
            events=events,
            periodic_preferred_flag=periodic_preferred_flag,
            short_window=is_short_window,
        )
        lines.append(f"整体结论：\n这一窗口中的 {target_col} {overall}，且波动{ {'low':'较小','medium':'中等','high':'偏强'}[vol_level] }。")
    else:
        overall = enforce_summary_consistency(
            summary_text="呈阶段性起伏",
            segments=segments,
            target=target,
            slope_norm=slope_norm,
            events=events,
            periodic_preferred_flag=periodic_preferred_flag,
            short_window=is_short_window,
        )
        lines.append(f"整体结论：\n这一窗口中的 {target_col} {overall}，且波动{ {'low':'较小','medium':'中等','high':'偏强'}[vol_level] }。")

    features = {
        "global": {
            "length": L,
            "min": g_min,
            "max": g_max,
            "mean": g_mean,
            "std": g_std,
            "vol_level": vol_level,
            "vol_ratio": vol_ratio,
            "vol_level_legacy": legacy_vol_level,
            "vol_ratio_legacy": legacy_vol_ratio,
            "vol_rel_band": float(vol_meta.get("rel_band", 0.0)),
            "vol_diff_energy": float(vol_meta.get("diff_energy", 0.0)),
            "window_profile": "short" if is_short_window else ("long" if is_long_window else "normal"),
            "trend_label": trend_label,
            "slope_norm": slope_norm,
        },
        "periodicity": {
            "best_lag": best_lag,
            "best_corr": best_corr,
            "global_periodic": bool(period_profile.get("is_global_periodic", False)),
            "cycle_consistency": float(period_profile.get("cycle_consistency", 0.0)),
            "first_half_corr": float(period_profile.get("first_half_corr", 0.0)),
            "second_half_corr": float(period_profile.get("second_half_corr", 0.0)),
            "harmonic_corr": float(period_profile.get("harmonic_corr", 0.0)),
        },
        "segments": segments,
        "events": events,
    }
    return "\n\n".join(lines), features


def generate_nab_jsonl(
    csv_path: str,
    output_path: str,
    window_lengths=(512, 1024),
    step_ratio: float = 0.5,
    max_samples: int | None = None,
    target_col: str = "value",
):
    df = load_nab(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"目标列 {target_col} 不在数据列中：{df.columns.tolist()}")

    n = len(df)
    feature_names = list(df.columns)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sample_count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for L in window_lengths:
            step = max(1, int(L * step_ratio))
            for start in sliding_window_indices(n, L, step):
                end = start + L
                win_df = df.iloc[start:end]
                desc, feat = describe_window_nab(win_df, target_col, window_start=int(start))

                dataset_name = Path(csv_path).stem
                sample = {
                    "id": f"{dataset_name}_{target_col}_L{L}_start{start}",
                    "dataset": dataset_name,
                    "task": "forecasting",
                    "window_length": int(L),
                    "start_index": int(start),
                    "end_index": int(end - 1),
                    "time": [str(t) for t in win_df.index],
                    "feature_names": feature_names,
                    "values": win_df.to_numpy().tolist(),
                    "features": feat,
                    "descriptions": [desc],
                    "target_col": target_col,
                }

                safe_sample = make_json_safe(sample)
                f.write(json.dumps(safe_sample, ensure_ascii=False, allow_nan=False))
                f.write("\n")
                sample_count += 1

                if max_samples is not None and sample_count >= max_samples:
                    print(f"达到 max_samples={max_samples}，提前停止。")
                    return

    print(f"已生成 {sample_count} 个样本，保存在 {out_path}。")


def main():
    parser = argparse.ArgumentParser(description="为 NAB 单个 CSV 生成中文描述 JSONL")
    parser.add_argument("--csv_path", type=str, required=True, help="要处理的 CSV 文件路径")
    parser.add_argument("--output_path", type=str, default=None, help="输出 jsonl 路径")
    parser.add_argument("--window_lengths", type=int, nargs="+", default=[512, 1024], help="窗口长度列表")
    parser.add_argument("--step_ratio", type=float, default=0.5, help="步长比例")
    parser.add_argument("--max_samples", type=int, default=None, help="最大生成样本数")
    parser.add_argument("--target_col", type=str, default="value", help="目标列名，默认 value")
    args = parser.parse_args()

    if args.output_path is None:
        csv_path_obj = Path(args.csv_path)
        parent_folder = csv_path_obj.parent.name
        win_tag = "-".join(str(w) for w in args.window_lengths)
        output_path = Path(__file__).parent / f"{parent_folder}_{csv_path_obj.stem}_{args.target_col}_L{win_tag}_descriptions.jsonl"
    else:
        output_path = Path(args.output_path)

    generate_nab_jsonl(
        csv_path=args.csv_path,
        output_path=str(output_path),
        window_lengths=args.window_lengths,
        step_ratio=args.step_ratio,
        max_samples=args.max_samples,
        target_col=args.target_col,
    )


if __name__ == "__main__":
    main()






