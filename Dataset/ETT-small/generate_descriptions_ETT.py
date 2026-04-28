# Usage
# cd D:\Users\Win\Desktop\Study\毕业设计-面向大语言模型的时序推理数据构建及分析系统实现\数据收集\数据集\ETT-small
# python generate_descriptions_ETT.py --csv_path Dataset\ETTh1.csv --target_col OT --max_samples 5
# python generate_descriptions_ETT.py --csv_path Dataset\ETTh1.csv --target_cols OT HUFL --max_samples 5
# --target_col 只允许指定一列（默认 OT）：会自动检测并加入强相关列
# --target_cols 支持指定一列或多列：按指定列原样写入 jsonl（不自动补充其他列）
# 该脚本生成的数据量很大，可以添加“ --max_samples x”生成少量数据，x表示生成x行jsonl数据
# --window_lengths参数可以指定多个窗口长度，默认是“ --window_lengths 512 1024”

import argparse
import pandas as pd
import numpy as np
import json
import math
from pathlib import Path
import sys

_COMMON_ROOT = Path(__file__).resolve().parents[1]
if str(_COMMON_ROOT) not in sys.path:
    sys.path.append(str(_COMMON_ROOT))
from common_desc_adapt import (
    smooth_signal_for_window,
    robust_classify_global_trend,
    classify_volatility_adaptive,
    detect_change_boundaries,
    detect_local_extreme_events,
    segment_features_by_boundaries,
    enforce_summary_consistency,
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

def load_etth1(csv_path: str) -> pd.DataFrame:
    """
    读取 ETTh1 数据集，返回以 datetime 为索引、按标准列顺序排列的 DataFrame。
    期望列名：date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
    """
    df = pd.read_csv(csv_path, encoding="utf-8")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    feature_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    # 有些文件可能已经去掉 date 这一列，这里做个兼容：
    available_cols = [c for c in feature_cols if c in df.columns]
    df = df[available_cols]
    return df


def sliding_window_indices(n: int, window: int, step: int):
    """生成滑动窗口的起始下标。"""
    i = 0
    while i + window <= n:
        yield i
        i += step


def classify_global_trend(x: np.ndarray):
    return robust_classify_global_trend(x)


def classify_volatility(std: float, mean: float, eps: float = 1e-6):
    """
    根据 相对波动率 = std / (|mean| + eps) 粗略划分波动等级。
    返回 (level, rel_value)，level ∈ {'low','medium','high'}
    """
    rel = std / (abs(mean) + eps)
    if rel < 0.05:
        lvl = "low"
    elif rel < 0.15:
        lvl = "medium"
    else:
        lvl = "high"
    return lvl, float(rel)


def estimate_period(x: np.ndarray, max_lag: int | None = None, min_lag: int = 8):
    """利用自相关峰值估计主周期，避免把 lag=1 的平滑相关误判为周期。"""
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
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
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


def build_period_phrases(best_lag: int, step_delta: pd.Timedelta | None = None, fallback_unit: str = "个时间点") -> tuple[str, str]:
    def _space_num_unit(text: str) -> str:
        out = []
        for i, ch in enumerate(text):
            if i > 0:
                prev = text[i - 1]
                if (prev.isdigit() and not ch.isdigit()) or (not prev.isdigit() and ch.isdigit()):
                    out.append(" ")
            out.append(ch)
        return "".join(out)

    duration_text = format_timedelta_cn(step_delta * int(best_lag)) if step_delta is not None else None
    if duration_text:
        duration_text = _space_num_unit(duration_text)
        return f"约每隔 {duration_text}", f"约 {duration_text}"
    fallback_text = _space_num_unit(f"{best_lag}{fallback_unit}")
    return f"约每隔 {fallback_text}", f"约 {fallback_text}"


def segment_features(x: np.ndarray, num_segments: int, global_std: float):
    """
    把序列均匀切成 num_segments 段，计算每段的均值/标准差/趋势等。
    返回一个 list，每个元素是字典：
    {
      'idx': k,
      'start': start,
      'end': end,   # inclusive
      'mean': ...,
      'std': ...,
      'vol_level': 'low'/'medium'/'high'（相对全局 std）,
      'trend_label': ...
    }
    """
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
        # 相对全局 std
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


def estimate_adaptive_segment_count(x: np.ndarray) -> int:
    """
    根据序列长度与方向切换频率自适应估计基础切分段数。
    不使用固定小上限，让可输出段数随数据复杂度增长。
    """
    arr = np.asarray(x, dtype=float)
    n = int(arr.size)
    if n < 32:
        return 2
    if np.allclose(arr, arr[0]):
        return 3

    smooth_w = max(5, int(n // 64))
    if smooth_w % 2 == 0:
        smooth_w += 1
    if smooth_w > 1:
        kernel = np.ones(smooth_w, dtype=float) / float(smooth_w)
        arr_sm = np.convolve(arr, kernel, mode="same")
    else:
        arr_sm = arr

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


def describe_window_etth1(
    win_df: pd.DataFrame,
    main_target_col: str = "OT",
    target_cols: list[str] | None = None,
) -> tuple[str, dict]:
    """
    对 ETTh1 的一个窗口（多变量，含 OT）生成中文描述和结构化特征。
    win_df: index 为时间戳，columns 至少包含 'OT' 及若干负载变量。
    返回: (description_text, features_dict)
    """
    assert main_target_col in win_df.columns, f"ETTh1 窗口必须包含 {main_target_col} 列"
    target_cols = target_cols or [main_target_col]
    target = win_df[main_target_col].to_numpy(dtype=float)
    L = len(target)
    t0 = win_df.index[0]
    t1 = win_df.index[-1]
    # 检查索引是否为 datetime 类型（判断是否有时间戳）
    has_timestamp = pd.api.types.is_datetime64_any_dtype(win_df.index)
    step_delta = infer_time_step(win_df.index) if has_timestamp else None

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
    period_every, period_cycle = build_period_phrases(best_lag, step_delta)

    periodic_cycle_flag = bool(period_profile.get("is_global_periodic", False) and not np.allclose(target, target[0]))
    strong_periodic_flag = bool(best_corr >= 0.75 and float(period_profile.get("cycle_consistency", 0.0)) >= 0.55)

    special_shape = detect_flat_or_square_wave(target)
    diff_abs = np.abs(np.diff(target))
    jump_thr = max(
        0.30 * max(g_max - g_min, 1e-6),
        (2.5 * float(np.std(np.diff(target)))) if diff_abs.size else 0.0,
    )
    abrupt_switch_flag = bool((int(np.sum(diff_abs >= jump_thr)) if diff_abs.size else 0) >= 2)
    if (not periodic_cycle_flag) and (special_shape == "square" and best_lag >= 6 and best_corr >= 0.30):
        periodic_cycle_flag = True
    platform_switch_segment_flag = bool(special_shape == "square" or abrupt_switch_flag)
    periodic_score = (0.55 if periodic_cycle_flag else 0.0) + min(max((best_corr - 0.30) / 0.50, 0.0), 0.35)
    segment_score = (0.45 if platform_switch_segment_flag else 0.0)
    if periodic_cycle_flag and is_micro_noise_period(target, best_lag, best_corr):
        periodic_cycle_flag = False
    periodic_preferred_flag = bool(periodic_cycle_flag and periodic_score >= segment_score)
    
    if special_shape == "flat":
        segments = []
    elif special_shape == "square":
        base_seg_n = estimate_square_segment_count(target)
        segments = segment_features(target, num_segments=base_seg_n, global_std=g_std)
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
            base_seg_n = estimate_adaptive_segment_count(target)
            segments = segment_features(target, num_segments=base_seg_n, global_std=g_std)
    events = detect_zscore_events(target, z_thr=2.5)

    # 多变量相关性：主变量 vs 其他
    corr_info = []
    corr_map = {}
    corr_stable_map = {}
    for var in [c for c in win_df.columns if c != main_target_col]:
        rho_val, stable, flat = _linkage_metrics(target, win_df[var].to_numpy(dtype=float))
        corr_map[var] = rho_val
        corr_stable_map[var] = stable and (not flat)
        if rho_val is not None:
            corr_info.append({"var": var, "rho": float(rho_val), "stable": bool(corr_stable_map[var])})
    corr_info.sort(key=lambda d: abs(d["rho"]), reverse=True)

    # ---------- 下面开始拼接中文描述（增强自然语言连贯性） ----------

    def fmt_time_range(s: int, e: int) -> str:
        if has_timestamp:
            left = win_df.index[s].strftime("%Y-%m-%d %H:%M")
            right = win_df.index[e].strftime("%Y-%m-%d %H:%M")
            return f"{left} 至 {right}"
        return f"第 {s} 到 {e} 个时间点"

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
        mapping = {
            "low": "波动较小",
            "medium": "波动中等",
            "high": "波动偏强",
        }
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
        has_period = periodic_cycle_flag
        seg_means = [float(s.get("mean", 0.0)) for s in segments] if segments else []
        value_span = max(g_max - g_min, 1e-6)
        net_change_ratio = (float(target[-1]) - float(target[0])) / value_span
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
            periodic_base = "呈明显周期性起伏" if strong_periodic_flag else "呈周期性起伏"
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
        if np.allclose(target, target[0]):
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
        net_change_ratio = abs(float(target[-1]) - float(target[0])) / value_span
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
        net_change_ratio = abs(float(target[-1]) - float(target[0])) / value_span
        diff = np.diff(target)
        if diff.size < 6:
            return False
        sign = np.sign(diff)
        sign = sign[sign != 0]
        if sign.size < 6:
            return False
        sign_flip_ratio = float(np.sum(sign[1:] * sign[:-1] < 0)) / max(sign.size - 1, 1)
        turning = (target[1:-1] - target[:-2]) * (target[1:-1] - target[2:])
        turning_ratio = float(np.sum(turning > 0)) / max(len(target) - 2, 1)
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
    segment_score = (0.45 if platform_switch_segment_flag else 0.0) + (0.20 if plateau_local_flag else 0.0) + (0.10 if oscillation_flag else 0.0)
    if periodic_cycle_flag and is_micro_noise_period(target, best_lag, best_corr):
        periodic_cycle_flag = False
    periodic_preferred_flag = bool(periodic_cycle_flag and periodic_score >= segment_score)
    
    def overall_summary_phrase() -> str:
        if periodic_preferred_flag:
            return "呈明显周期性起伏" if strong_periodic_flag else "呈周期性起伏"
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
    start_val, end_val = float(target[0]), float(target[-1])

    # 1) 开场总述
    if has_timestamp:
        intro_head = f"这段窗口覆盖时间为 {t0} 到 {t1}，共 {L} 个观测点。"
    else:
        intro_head = f"这段窗口共包含 {L} 个观测点。"
    if np.allclose(target, target[0]):
        trait = "整体保持恒定"
    elif periodic_preferred_flag:
        trait = "整体呈明显周期性波动" if strong_periodic_flag else "整体呈周期性波动"
    elif oscillation_flag and not plateau_local_flag:
        trait = "整体变化不规则，短时起伏较频繁"
    elif plateau_local_flag:
        trait = "整体以平台波动为主，仅局部有起伏"
    else:
        phrase = shape_phrase()
        trait = phrase[2:] if phrase.startswith("整体") else phrase
    intro_body = (
        f"{main_target_col} 的取值大致落在 {g_min:.1f} 到 {g_max:.1f} 之间，"
        f"平均约 {g_mean:.1f}，起始点为 {start_val:.1f}，结束点为 {end_val:.1f}，{trait}。"
    )
    lines.append("窗口概览：\n" + intro_head + intro_body)

    # 4) 分段观察（按窗口特征自适应组织分段关系）
    if segments:
        segment_subject = str(main_target_col)

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
            return fmt_time_range(int(seg["start"]), int(seg["end"]))

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

        def group_phrase(direction: str, amp: str) -> str:
            dir_text = {
                "up": "以抬升为主",
                "down": "以回落为主",
                "flat": "整体较平稳",
            }.get(direction, "整体较平稳")
            amp_text = {
                "high": "且波动幅度偏大",
                "mid": "且波动幅度中等",
                "low": "且波动幅度较小",
            }.get(amp, "且波动幅度中等")
            return f"{dir_text}，{amp_text}"

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

        segment_lines: list[str] = []
        no_change_local = bool(np.allclose(target, target[0]))
        seg_stds = np.array([float(s.get("std", 0.0)) for s in segments], dtype=float)
        seg_means = np.array([float(s.get("mean", 0.0)) for s in segments], dtype=float)

        if no_change_local:
            segment_lines = [
                f"各分段数值均保持在 {fmt_num(g_mean)} 附近，窗口内未见可辨别的阶段性变化。",
            ]
        elif periodic_preferred_flag:
            cycle_len = min(max(int(best_lag), 1), L)
            cycle_values = target[:cycle_len]
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
            wave_label, high_ratio = cycle_wave_profile(target, int(best_lag))
            high_pct = int(round(100.0 * high_ratio))
            value_span = max(g_max - g_min, 1e-6)
            split_idx = L // 2
            front_vals = target[:split_idx]
            back_vals = target[split_idx:]
            level_shift = bool(
                split_idx >= 24
                and (L - split_idx) >= 24
                and abs(float(np.mean(back_vals)) - float(np.mean(front_vals))) / value_span >= 0.12
            )
            periodic_word = "明显周期性起伏" if strong_periodic_flag else "周期性起伏"

            if level_shift:
                front_mean = float(np.mean(front_vals))
                back_mean = float(np.mean(back_vals))
                front_min = float(np.min(front_vals))
                front_max = float(np.max(front_vals))
                back_min = float(np.min(back_vals))
                back_max = float(np.max(back_vals))
                front_start = float(front_vals[0])
                front_end = float(front_vals[-1])
                back_start = float(back_vals[0])
                back_end = float(back_vals[-1])
                level_word = "低于" if back_mean < front_mean else "高于"
                segment_lines = [
                    f"第1阶段（{fmt_time_range(0, split_idx - 1)}）均值约 {fmt_num(front_mean)}，起点约 {fmt_num(front_start)}，终点约 {fmt_num(front_end)}，段内大致在 {fmt_num(front_min)} 到 {fmt_num(front_max)} 之间，以{periodic_word}为主。",
                    f"第2阶段（{fmt_time_range(split_idx, L - 1)}）均值约 {fmt_num(back_mean)}，起点约 {fmt_num(back_start)}，终点约 {fmt_num(back_end)}，段内大致在 {fmt_num(back_min)} 到 {fmt_num(back_max)} 之间，以{periodic_word}为主，且均值整体{level_word}第1阶段。",
                ]
            else:
                segment_lines = [
                    f"以单个周期看（{period_cycle}），{segment_subject} 在一个周期内通常{cycle_shape_phrase(cycle_values)}。",
                    f"该周期取值大致在 {fmt_num(float(period_representative_bounds(target, int(best_lag))[0]))} 到 {fmt_num(float(period_representative_bounds(target, int(best_lag))[1]))} 之间，{amp_note}；波形更接近{wave_label}，单周期高位段占比约 {high_pct}%。",
                ]
        else:
            value_span = max(g_max - g_min, 1e-6)
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
                    mean_gap >= 0.12
                    or std_shift >= 0.45
                    or (dir_shift and mean_gap >= 0.05)
                    or (amp_shift and (mean_gap >= 0.06 or std_shift >= 0.25))
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
                left_mean = float(np.mean([float(x.get("mean", 0.0)) for x in left_items]))
                right_mean = float(np.mean([float(x.get("mean", 0.0)) for x in right_items]))
                left_std = float(np.mean([float(x.get("std", 0.0)) for x in left_items]))
                right_std = float(np.mean([float(x.get("std", 0.0)) for x in right_items]))
                mean_gap = abs(right_mean - left_mean) / value_span
                std_shift = max(right_std / max(left_std, 1e-6), left_std / max(right_std, 1e-6)) - 1.0
                left_dir = seg_dir(left_items[-1])
                right_dir = seg_dir(right_items[0])
                close_enough = mean_gap < 0.05 and std_shift < 0.20 and left_dir == right_dir
                if close_enough:
                    merged[-1]["items"].extend(right_items)
                else:
                    merged.append(grp)
            groups = merged

            # 减少过度分段：将很短的小碎段并入相邻更接近的阶段。
            tiny_len_thr = max(int(0.08 * L), int(max(8, L // max(6, len(segments) + 1))))
            changed = True
            while changed and len(groups) > 1:
                changed = False
                for gi in range(len(groups)):
                    items = groups[gi]["items"]
                    g_start = int(items[0]["start"])
                    g_end = int(items[-1]["end"])
                    g_len = g_end - g_start + 1
                    if g_len >= tiny_len_thr:
                        continue

                    cur_mean = float(np.mean([float(x.get("mean", 0.0)) for x in items]))
                    left_gap = np.inf
                    right_gap = np.inf
                    if gi > 0:
                        left_items = groups[gi - 1]["items"]
                        left_mean = float(np.mean([float(x.get("mean", 0.0)) for x in left_items]))
                        left_gap = abs(cur_mean - left_mean)
                    if gi < len(groups) - 1:
                        right_items = groups[gi + 1]["items"]
                        right_mean = float(np.mean([float(x.get("mean", 0.0)) for x in right_items]))
                        right_gap = abs(cur_mean - right_mean)

                    if left_gap <= right_gap and gi > 0:
                        groups[gi - 1]["items"].extend(items)
                        groups.pop(gi)
                    elif gi < len(groups) - 1:
                        groups[gi + 1]["items"] = items + groups[gi + 1]["items"]
                        groups.pop(gi)
                    else:
                        continue
                    changed = True
                    break

            # 弥补欠分段：若整体仅一段但内部存在高分边界，按边界再拆分。
            if len(groups) == 1 and len(segments) >= 5:
                boundary_scores: list[tuple[float, int]] = []
                for si in range(len(segments) - 1):
                    a = segments[si]
                    b = segments[si + 1]
                    mean_gap = abs(float(b.get("mean", 0.0)) - float(a.get("mean", 0.0))) / value_span
                    a_std = max(float(a.get("std", 0.0)), 1e-6)
                    b_std = max(float(b.get("std", 0.0)), 1e-6)
                    std_shift = max(b_std / a_std, a_std / b_std) - 1.0
                    dir_shift = 1.0 if seg_dir(a) != seg_dir(b) else 0.0
                    score = mean_gap + 0.55 * std_shift + 0.18 * dir_shift
                    boundary_scores.append((float(score), si))

                boundary_scores.sort(reverse=True, key=lambda x: x[0])
                selected: list[int] = []
                for score, si in boundary_scores:
                    if score < 0.34:
                        continue
                    if si < 1 or si > len(segments) - 3:
                        continue
                    if any(abs(si - s) < 2 for s in selected):
                        continue
                    selected.append(si)
                    if len(selected) >= 2:
                        break

                if selected:
                    selected = sorted(selected)
                    rebuilt = []
                    start_idx = 0
                    for cut in selected:
                        rebuilt.append({"items": segments[start_idx : cut + 1]})
                        start_idx = cut + 1
                    rebuilt.append({"items": segments[start_idx:]})
                    groups = rebuilt

            # 无论窗口属于何种形态，均逐段输出，避免时间覆盖不完整。
            if not groups:
                groups = [{"items": list(segments)}]

            # 若强异常附近缺少边界，尝试在异常点附近补一刀，提升分段可解释性。
            if events and len(segments) >= 4:
                anchor = max(events, key=lambda ev: abs(float(ev.get("z_max", 0.0))))
                anchor_z = abs(float(anchor.get("z_max", 0.0)))
                anchor_idx = int(anchor.get("peak_idx", anchor.get("start", 0)))
                if anchor_z >= 2.8:
                    boundary_pos = [int(grp["items"][-1]["end"]) for grp in groups[:-1]]
                    near_tol = max(6, int(0.03 * L))
                    has_near_boundary = any(abs(pos - anchor_idx) <= near_tol for pos in boundary_pos)
                    if not has_near_boundary:
                        min_side = max(24, int(0.10 * L))
                        event_cands: list[tuple[float, int]] = []
                        for si in range(len(segments) - 1):
                            a = segments[si]
                            b = segments[si + 1]
                            left_end = int(a["end"])
                            right_start = int(b["start"])
                            if left_end < min_side or (L - right_start) < min_side:
                                continue
                            mean_gap = abs(float(b.get("mean", 0.0)) - float(a.get("mean", 0.0))) / value_span
                            a_std = max(float(a.get("std", 0.0)), 1e-6)
                            b_std = max(float(b.get("std", 0.0)), 1e-6)
                            std_shift = max(b_std / a_std, a_std / b_std) - 1.0
                            dir_shift = 1.0 if seg_dir(a) != seg_dir(b) else 0.0
                            prox = 1.0 - min(abs(left_end - anchor_idx), abs(right_start - anchor_idx)) / max(1.0, 0.45 * L)
                            score = mean_gap + 0.50 * std_shift + 0.16 * dir_shift + 0.70 * max(0.0, prox)
                            if score >= 0.30:
                                event_cands.append((float(score), si))

                        if event_cands:
                            event_cands.sort(reverse=True, key=lambda x: x[0])
                            best_si = int(event_cands[0][1])

                            # 若该边界尚未被 groups 表示，则在其所在组内拆分。
                            seg_pos: dict[int, int] = {id(seg): idx for idx, seg in enumerate(segments)}
                            inserted = False
                            for gi, grp in enumerate(groups):
                                idxs = [seg_pos.get(id(seg), -1) for seg in grp["items"]]
                                if not idxs:
                                    continue
                                g_lo, g_hi = min(idxs), max(idxs)
                                if g_lo <= best_si < g_hi:
                                    left_items = [seg for seg in grp["items"] if seg_pos.get(id(seg), -1) <= best_si]
                                    right_items = [seg for seg in grp["items"] if seg_pos.get(id(seg), -1) > best_si]
                                    if left_items and right_items:
                                        groups = groups[:gi] + [{"items": left_items}, {"items": right_items}] + groups[gi + 1 :]
                                        inserted = True
                                    break
                            if inserted:
                                pass

            def make_phase_item(start_idx_rel: int, end_idx_rel: int) -> dict:
                phase_vals = target[start_idx_rel : end_idx_rel + 1]
                phase_mean = float(np.mean(phase_vals))
                phase_std = float(np.std(phase_vals))
                phase_trend, _ = classify_global_trend(phase_vals)
                if g_std <= 0:
                    phase_vol = "low"
                else:
                    ratio = phase_std / g_std
                    if ratio < 0.7:
                        phase_vol = "low"
                    elif ratio < 1.3:
                        phase_vol = "medium"
                    else:
                        phase_vol = "high"
                return {
                    "idx": -1,
                    "start": int(start_idx_rel),
                    "end": int(end_idx_rel),
                    "len": int(end_idx_rel - start_idx_rel + 1),
                    "mean": phase_mean,
                    "std": phase_std,
                    "vol_level": phase_vol,
                    "trend_label": phase_trend,
                }

            if len(groups) == 1 and L >= 64:
                half = L // 2
                if half >= 24 and (L - half) >= 24:
                    front_vals = target[:half]
                    back_vals = target[half:]
                    value_span = max(g_max - g_min, 1e-6)
                    mean_gap_ratio = abs(float(np.mean(back_vals)) - float(np.mean(front_vals))) / value_span
                    if mean_gap_ratio >= 0.12:
                        groups = [
                            {"items": [make_phase_item(0, half - 1)]},
                            {"items": [make_phase_item(half, L - 1)]},
                        ]

            for i, grp in enumerate(groups, start=1):
                first_seg = grp["items"][0]
                last_seg = grp["items"][-1]
                g_mean_val = float(np.mean([float(x.get("mean", 0.0)) for x in grp["items"]]))
                dirs = [seg_dir(x) for x in grp["items"]]
                d = max(set(dirs), key=dirs.count) if dirs else "flat"
                amp_levels = [seg_amp(x) for x in grp["items"]]
                a = max(set(amp_levels), key=amp_levels.count) if amp_levels else "mid"
                phase_start = int(first_seg["start"])
                phase_end = int(last_seg["end"])
                phase_seg = {"start": phase_start, "end": phase_end}
                phase_phrase = group_phrase(d, a)
                phase_vals = target[phase_start : phase_end + 1]
                if d == "flat" and has_hump_shape(phase_vals):
                    phase_phrase = phase_phrase.replace("整体较平稳", "存在明显起伏变化")

                local_shape = phase_shape_phrase(phase_vals)
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
                local_profile = evaluate_global_periodicity(phase_vals, local_lag, local_corr)
                local_periodic = bool(
                    local_profile.get("is_global_periodic", False)
                    and len(phase_vals) >= max(48, 2 * max(local_lag, 1))
                    and not is_micro_noise_period(phase_vals, local_lag, local_corr)
                )
                periodic_tail = ""
                periodic_cycle_text = ""
                periodic_shape_text = ""
                if local_periodic:
                    _, local_cycle = build_period_phrases(int(local_lag), step_delta)
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
                    segment_lines.append(
                        f"该窗口内未识别到明确阶段切换，整体变化连续，{phase_phrase}，{shape_clause}{rel_vol_phrase}。"
                    )
                else:
                    segment_lines.append(
                        f"第{i}阶段（{seg_interval_text(phase_seg)}）均值约 {fmt_num(g_mean_val)}，起点约 {fmt_num(phase_start_val)}，终点约 {fmt_num(phase_end_val)}，段内大致在 {fmt_num(phase_min)} 到 {fmt_num(phase_max)} 之间，{phase_phrase}，{shape_clause}{rel_vol_phrase}{periodic_tail}。"
                    )

        lines.append("分段观察：\n" + "\n".join(segment_lines))

    # 5) 关键事件
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
                baseline = float(target[prev_idx:s].mean()) if s > 0 else g_mean
            signed_delta = ev.get("signed_delta")
            if signed_delta is None:
                signed_delta = float(v) - float(baseline)
            signed_delta = float(signed_delta)
            change_word = "激增" if signed_delta >= 0 else "暴跌"
            compare_word = "偏高" if signed_delta >= 0 else "偏低"
            if int(s) == int(e):
                if has_timestamp:
                    ev_pos = win_df.index[s].strftime("%Y-%m-%d %H:%M")
                else:
                    ev_pos = f"第 {s} 个时间点"
            else:
                ev_pos = fmt_time_range(s, e)
            event_desc.append(
                f"{ev_pos} 出现一次局部{change_word}（峰值约 {v:.1f}，相较局部基线 {baseline:.1f} 明显{compare_word}，标准化偏离 |z|≈{abs(z):.1f}）"
            )
        lines.append("异常点：\n" + "；\n".join(event_desc) + "。")
    else:
        lines.append("异常点：\n本窗口未出现特别突出的尖峰或急跌。")

    # 6) 多变量关系
    if len(target_cols) > 1:
        extra_targets = [c for c in target_cols if c != main_target_col]
        if extra_targets:
            parts = []
            joint_clause = build_joint_linkage_clause(
                main_var=main_target_col,
                corr_map=corr_map,
                stable_map=corr_stable_map,
                strong_thr=0.8,
            )
            if joint_clause:
                parts.append(joint_clause)
            for var in extra_targets:
                rho = corr_map.get(var)
                stable = corr_stable_map.get(var, False)
                if rho is None or not stable:
                    parts.append(f"{var} 与 {main_target_col} 无明显联动关系")
                elif abs(rho) >= 0.25:
                    relation = "同向" if rho >= 0 else "反向"
                    parts.append(f"{var} 与 {main_target_col} 呈{relation}关系（相关系数 rho≈{rho:.2f}）")
                else:
                    parts.append(f"{var} 与 {main_target_col} 联动关系弱（相关系数 rho≈{rho:.2f}）")
            lines.append("联动关系：\n" + "；\n".join(parts) + "。")

    # 7) 一句话收束，读起来更像完整段落
    if oscillation_flag and not plateau_local_flag:
        lines.append(
            f"整体结论：\n这一窗口中的 {main_target_col} 以高频震荡为主，短时起伏频繁，未形成持续单向趋势。"
        )
    else:
        summary_phrase = overall_summary_phrase()
        summary_text = summary_phrase[2:] if summary_phrase.startswith("整体") else summary_phrase
        summary_text = enforce_summary_consistency(
            summary_text=summary_text,
            segments=segments,
            target=target,
            slope_norm=slope_norm,
            events=events,
            periodic_preferred_flag=periodic_preferred_flag,
            short_window=is_short_window,
        )
        lines.append(
            f"整体结论：\n这一窗口中的 {main_target_col} {summary_text}，且{vol_phrase(vol_level)}。"
        )

    description = "\n\n".join(lines)

    # 结构化特征，方便后续做 QA / LLM 改写
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
        "correlations": corr_info,
    }

    return description, features


def _parse_target_cols(target_cols_input: list[str] | str | None, default_col: str = "OT") -> list[str]:
    if target_cols_input is None:
        return [default_col]
    if isinstance(target_cols_input, str):
        raw = [target_cols_input]
    else:
        raw = target_cols_input
    cols = []
    for item in raw:
        for token in str(item).split(","):
            col = token.strip()
            if col and col not in cols:
                cols.append(col)
    return cols or [default_col]


def generate_etth1_jsonl(
    csv_path: str,
    output_path: str,
    window_lengths=(512, 1024),
    step_ratio: float = 0.5,
    max_samples: int | None = None,
    target_col: str = "OT",
    target_cols: list[str] | str | None = None,
):
    """
    主函数：读取 etth1.csv，按给定窗口长度滑动切片，对每个窗口生成描述，
    并将样本以一行一个 JSON 的形式写入 output_path（JSONL 格式）。

    参数语义：
    - target_col: 单列主变量；若未给 target_cols，会自动补充强相关列。
    - target_cols: 显式列集合（1列或多列），按指定列原样写入，不自动补充其他列。
    """
    df = load_etth1(csv_path)
    n = len(df)
    main_target_col = str(target_col).strip()
    if not main_target_col:
        raise ValueError("target_col 不能为空")

    explicit_targets = _parse_target_cols(target_cols, default_col=main_target_col) if target_cols is not None else None
    if explicit_targets is not None:
        main_target_col = explicit_targets[0]

    required_cols = explicit_targets if explicit_targets is not None else [main_target_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"目标列 {col} 不存在，可用列：{list(df.columns)}")

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    corr_candidates = [c for c in numeric_cols if c != main_target_col]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sample_count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for L in window_lengths:
            step = max(1, int(L * step_ratio))
            for start in sliding_window_indices(n, L, step):
                end = start + L
                win_df = df.iloc[start:end]
                if explicit_targets is not None:
                    sample_keep_cols = list(explicit_targets)
                else:
                    sample_keep_cols = [main_target_col]
                    for col in corr_candidates:
                        rho_val, stable, flat = _linkage_metrics(
                            win_df[main_target_col].to_numpy(dtype=float),
                            win_df[col].to_numpy(dtype=float),
                        )
                        if stable and (not flat) and rho_val is not None and abs(rho_val) >= 0.25 and col not in sample_keep_cols:
                            sample_keep_cols.append(col)

                sample_strong_corr_cols = [c for c in sample_keep_cols if c != main_target_col]
                desc, feat = describe_window_etth1(
                    win_df,
                    main_target_col=main_target_col,
                    target_cols=sample_keep_cols,
                )

                # dataset name derived from csv filename stem (e.g., ETTh1 or ETTh2)
                dataset_name = Path(csv_path).stem
                sample = {
                    "id": f"{dataset_name}_{main_target_col}_L{L}_start{start}",
                    "dataset": dataset_name,
                    "task": "forecasting",
                    "window_length": L,
                    "start_index": int(start),
                    "end_index": int(end - 1),
                    "start_time": str(win_df.index[0]),
                    "end_time": str(win_df.index[-1]),
                    "time": [str(t) for t in win_df.index],
                    "feature_names": sample_keep_cols,
                    "values": win_df[sample_keep_cols].to_numpy().tolist(),
                    "features": feat,                      # 结构化特征
                    "descriptions": [desc],                # 目前每个窗口 1 条描述，你可以后续扩展多条
                    "target_col": main_target_col,
                    "target_cols": sample_keep_cols,
                    "strong_corr_cols": sample_strong_corr_cols,
                }

                safe_sample = make_json_safe(sample)
                f.write(json.dumps(safe_sample, ensure_ascii=False, allow_nan=False))
                f.write("\n")

                sample_count += 1
                if max_samples is not None and sample_count >= max_samples:
                    print(f"达到 max_samples={max_samples}，提前停止。")
                    return

    print(f"已生成 {sample_count} 个样本，保存在 {out_path}。")


if __name__ == "__main__":
    # 示例用法：
    #   python generate_etth1_descriptions.py --csv_path ETTh1.csv --output_path etth1_desc.jsonl
    import argparse

    parser = argparse.ArgumentParser(description="生成 ETTh1 长时间序列的中文文本描述（JSONL 格式）。")
    parser.add_argument("--csv_path", type=str, required=True, help="ETTh1.csv 的路径")
    parser.add_argument("--output_path", type=str, default=None, help="输出 jsonl 路径（可选，若不指定则自动生成）")
    parser.add_argument(
        "--window_lengths",
        type=int,
        nargs="+",
        default=[512, 1024],
        help="窗口长度列表，例如 512 1024",
    )
    parser.add_argument(
        "--step_ratio",
        type=float,
        default=0.5,
        help="滑动步长相对窗口长度的比例，默认 0.5 表示步长 = window_length*0.5",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最多生成多少个样本，默认不限制",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="OT",
        help="主目标列（仅单列，默认 OT）。不指定 --target_cols 时，会自动补充强相关列",
    )
    parser.add_argument(
        "--target_cols",
        type=str,
        nargs="+",
        default=None,
        help="显式指定要写入 jsonl 的列（支持一列或多列，空格或逗号分隔）",
    )
    args = parser.parse_args()

    if args.target_cols is not None:
        explicit_targets = _parse_target_cols(args.target_cols)
        main_target_col = explicit_targets[0]
    else:
        explicit_targets = None
        main_target_col = str(args.target_col).strip()
        if not main_target_col:
            raise ValueError("--target_col 不能为空")

    # 如果没有传入 --output_path，则在脚本所在目录生成默认文件名：{csv_stem}_{target_col}_descriptions.jsonl
    if args.output_path is None:
        win_tag = "-".join(str(w) for w in args.window_lengths)
        default_name = f"{Path(args.csv_path).stem}_{main_target_col}_L{win_tag}_descriptions.jsonl"
        output_path = Path(__file__).parent / default_name
    else:
        output_path = Path(args.output_path)

    generate_etth1_jsonl(
        csv_path=args.csv_path,
        output_path=str(output_path),
        window_lengths=args.window_lengths,
        step_ratio=args.step_ratio,
        max_samples=args.max_samples,
        target_col=main_target_col,
        target_cols=explicit_targets,
    )











