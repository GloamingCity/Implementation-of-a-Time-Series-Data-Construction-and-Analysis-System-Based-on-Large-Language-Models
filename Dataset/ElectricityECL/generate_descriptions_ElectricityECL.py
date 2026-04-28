# Usage
# cd D:\Users\Win\Desktop\Study\毕业设计-面向大语言模型的时序推理数据构建及分析系统实现\数据收集\数据集\ElectricityECL
# python generate_descriptions_ElectricityECL.py --csv_path Dataset\LD2011_2014.txt --target_col MT_200 --max_samples 5
# 运行前请在末尾添加“ --target_col”来指定目标电表列（默认MT_001）
# 该脚本生成的数据量很大，可以添加“ --max_samples x”生成少量数据，x表示生成x行jsonl数据
# --window_lengths参数可以指定多个窗口长度，默认是“ --window_lengths 512 1024”

import argparse
import pandas as pd
import numpy as np
import json
import math
from pathlib import Path
import pathlib
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

def convert_to_csv(input_path: str) -> str:
    """
    仅处理 .txt 文件并将其转换为标准 CSV（逗号分隔、小数点为点）。
    如果输入已经是 .csv，则直接返回原路径。
    :param input_path: 输入文件路径（.txt 或 .csv）
    :return: 转换后的CSV文件路径（或原 csv 路径）
    """
    input_path = pathlib.Path(input_path)
    
    # 检查文件是否存在
    if not input_path.exists():
        raise FileNotFoundError(f"文件不存在：{input_path}")
    
    # 获取文件基本信息
    file_dir = input_path.parent  # 原文件目录
    file_name = input_path.stem   # 原文件名（无后缀）
    file_suffix = input_path.suffix.lower()  # 后缀（小写）
    # 转换后的CSV路径
    csv_output_path = file_dir / f"{file_name}.csv"

    # 如果已是 .csv 则直接返回
    if file_suffix == ".csv":
        return str(input_path)

    if file_suffix != ".txt":
        raise ValueError(f"仅支持 .txt 文件进行转换；传入文件后缀为：{file_suffix}")

    # 如果同目录下已存在转换后的CSV，直接使用它，跳过转换
    if csv_output_path.exists():
        print(f"ℹ️  发现已存在的CSV文件，跳过转换：{csv_output_path}")
        return str(csv_output_path)

    # 处理 .txt 文件（ElectricityECL 常见为分号分隔且小数使用逗号）
    try:
        df = pd.read_csv(
            input_path,
            sep=";",
            decimal=",",
            encoding="utf-8",
            low_memory=False,
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            input_path,
            sep=";",
            decimal=",",
            encoding="latin1",
            low_memory=False,
        )

    df.to_csv(csv_output_path, index=False, encoding="utf-8")
    print(f"已将 {input_path.name} 转换为 {csv_output_path.name}")
    return str(csv_output_path)


def sliding_window_indices(n: int, window: int, step: int):
    i = 0
    while i + window <= n:
        yield i
        i += step


def load_electricity_ecl(csv_path: str, target_col: str = "MT_001") -> pd.DataFrame:
    """加载 ElectricityECL 并返回按时间索引的数值表。"""
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"输入文件为空：{csv_path}")

    # ElectricityECL 常见首列名为 Unnamed: 0，代表时间戳。
    time_col = "Unnamed: 0" if "Unnamed: 0" in df.columns else str(df.columns[0])
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).set_index(time_col)

    # 仅保留电表列并清洗为数值。
    meter_cols = [c for c in df.columns if str(c).startswith("MT_")]
    if not meter_cols:
        raise ValueError("未检测到 MT_ 开头的电表列，请检查输入文件格式。")
    df = df[meter_cols].apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both").ffill().bfill()

    if target_col not in df.columns:
        raise ValueError(f"目标列 {target_col} 不存在。可选示例：{', '.join(df.columns[:6])} ...")
    return df


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


def segment_features(x: np.ndarray, num_segments: int, global_std: float, period_hint: int = 0):
    """
    自适应分段：基于变化强度局部峰值选择分界点，而非机械等分。
    同时强制分段覆盖 [0, L-1] 全窗口。
    """
    L = len(x)
    if L < 4:
        return []

    max_segments = max(2, min(int(num_segments) + 1, 8))
    min_seg_len = max(24, L // 10)

    dx = np.abs(np.diff(np.asarray(x, dtype=float)))
    if dx.size == 0:
        boundaries = [0, L]
    else:
        smooth_w = max(3, min(21, L // 24 if L // 24 > 0 else 3))
        kernel = np.ones(smooth_w, dtype=float) / float(smooth_w)
        score = np.convolve(dx, kernel, mode="same")

        q75 = float(np.quantile(score, 0.75)) if score.size >= 4 else float(np.max(score))
        med = float(np.median(score))
        mad = float(np.median(np.abs(score - med)))
        robust_thr = med + 1.5 * mad
        thr = max(q75, robust_thr)

        candidates: list[tuple[float, int]] = []
        for i in range(1, len(score) - 1):
            if score[i] >= score[i - 1] and score[i] >= score[i + 1] and score[i] >= thr:
                candidates.append((float(score[i]), int(i + 1)))

        if int(period_hint) >= 12:
            lag = int(period_hint)
            for b in range(lag, L, lag):
                if min_seg_len <= b <= L - min_seg_len:
                    candidates.append((0.25, int(b)))

        candidates.sort(key=lambda t: t[0], reverse=True)
        selected: list[int] = []
        anchors = [0, L]
        for _, b in candidates:
            if len(selected) >= max_segments - 1:
                break
            if min(abs(b - a) for a in anchors + selected) >= min_seg_len:
                selected.append(b)

        if not selected and L >= 2 * min_seg_len and score.size >= 1:
            fallback_b = int(np.argmax(score) + 1)
            fallback_b = min(max(fallback_b, min_seg_len), L - min_seg_len)
            if min_seg_len <= fallback_b <= L - min_seg_len:
                selected.append(fallback_b)

        boundaries = [0] + sorted(set(selected)) + [L]

    segments = []
    for k, (start, end_exclusive) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        start = int(start)
        end_exclusive = int(end_exclusive)
        if end_exclusive - start < 2:
            continue
        seg = x[start:end_exclusive]
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
                "end": int(end_exclusive - 1),
                "len": int(end_exclusive - start),
                "mean": seg_mean,
                "std": seg_std,
                "vol_level": vol_level,
                "trend_label": seg_trend_label,
            }
        )

    if not segments:
        seg = np.asarray(x, dtype=float)
        seg_mean = float(seg.mean())
        seg_std = float(seg.std())
        seg_trend_label, _ = classify_global_trend(seg)
        vol_level = "low"
        if global_std > 0:
            ratio = seg_std / global_std
            if ratio >= 1.3:
                vol_level = "high"
            elif ratio >= 0.7:
                vol_level = "medium"
        segments = [{"idx": 0, "start": 0, "end": L - 1, "len": L, "mean": seg_mean, "std": seg_std, "vol_level": vol_level, "trend_label": seg_trend_label}]

    segments[0]["start"] = 0
    segments[-1]["end"] = L - 1
    segments[-1]["len"] = int(segments[-1]["end"] - segments[-1]["start"] + 1)
    return segments


def detect_zscore_events(x: np.ndarray, z_thr: float = 2.5):
    return detect_local_extreme_events(x, z_thr=z_thr)


def describe_window_electricity_ecl(win_df: pd.DataFrame, target_col: str) -> tuple[str, dict]:
    """
    对ElectricityECL的一个窗口生成中文描述和结构化特征
    :param win_df: 窗口DataFrame（含MT列）
    :param target_col: 目标电表列（如MT_001）
    :return: (描述文本, 结构化特征)
    """
    # 转换电表名称为中文友好名称（如MT_200 -> 电表200）
    meter_number = target_col.split('_')[1]
    chinese_name = f"电表{meter_number}"
    
    target = win_df[target_col].to_numpy(dtype=float)
    L = len(target)
    t0 = win_df.index[0].strftime('%Y-%m-%d %H:%M')
    t1 = win_df.index[-1].strftime('%Y-%m-%d %H:%M')
    g_min, g_max = float(target.min()), float(target.max())
    g_mean = float(target.mean())
    g_std = float(target.std())
    no_change_flag = bool(np.allclose(target, target[0]))
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
    period_every, period_cycle = build_period_phrases(best_lag, step_delta)
    special_shape = detect_flat_or_square_wave(target)
    if special_shape == "flat":
        segments = []
    elif special_shape == "square":
        seg_target = estimate_square_segment_count(target)
        segments = segment_features(target, num_segments=seg_target, global_std=g_std, period_hint=best_lag)
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
            seg_target = 3 if is_short_window else (5 if is_long_window else 4)
            segments = segment_features(target, num_segments=seg_target, global_std=g_std, period_hint=best_lag)
    events = detect_zscore_events(target, z_thr=2.5)
    
    # 非关联性数据集：不输出联动关系段
    corr_info = []
    
    # 中文描述拼接（与 ETT 脚本风格保持一致）
    def fmt_time_range(s: int, e: int) -> str:
        left = win_df.index[s].strftime("%Y-%m-%d %H:%M")
        right = win_df.index[e].strftime("%Y-%m-%d %H:%M")
        return f"{left} 至 {right}"

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
        has_period = bool(periodic_cycle_flag)
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
        # ElectricityECL 中存在明显周期但窗口首尾处于不同相位的情形：
        # 若分段均值中枢变化很小，却出现较大的首尾净变化，优先按周期起伏描述。
        phase_shift_periodic = (
            has_period
            and swing_ratio <= 0.18
            and abs(net_change_ratio) >= 0.30
            and (up_cnt >= 2 or down_cnt >= 2)
        )
        periodic_flat_like = (
            has_period
            and trend_label == "flat"
            and swing_ratio <= 0.25
            and (up_cnt >= 2 or down_cnt >= 2)
        )
        periodic_like = has_period and (mixed_moves or alt_changes >= 1 or len(compressed) >= 3 or periodic_flat_like)

        if phase_shift_periodic:
            periodic_base = "呈明显周期性起伏" if best_corr >= 0.7 else "呈周期性起伏"
            return periodic_base

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
        # 若已检出较强稳定周期，优先归为“周期性起伏”，避免与“高频震荡”表述冲突。
        if bool(period_profile.get("is_global_periodic", False)) and not is_micro_noise_period(target, best_lag, best_corr):
            return False
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
    periodic_cycle_flag = bool(period_profile.get("is_global_periodic", False) and not no_change_flag)
    diff_abs = np.abs(np.diff(target))
    jump_thr = max(
        0.30 * max(g_max - g_min, 1e-6),
        (2.5 * float(np.std(np.diff(target)))) if diff_abs.size else 0.0,
    )
    abrupt_switch_flag = bool((int(np.sum(diff_abs >= jump_thr)) if diff_abs.size else 0) >= 2)
    if (not periodic_cycle_flag) and (special_shape == "square" and best_lag >= 8 and best_corr >= 0.45):
        periodic_cycle_flag = not is_micro_noise_period(target, best_lag, best_corr)
    platform_switch_segment_flag = bool(special_shape == "square" or abrupt_switch_flag)
    periodic_score = (0.55 if periodic_cycle_flag else 0.0) + min(max((best_corr - 0.30) / 0.50, 0.0), 0.35)
    segment_score = (0.45 if platform_switch_segment_flag else 0.0) + (0.20 if plateau_local_flag else 0.0) + (0.10 if oscillation_flag else 0.0)
    if periodic_cycle_flag and is_micro_noise_period(target, best_lag, best_corr):
        periodic_cycle_flag = False
    periodic_grouping_flag = bool(periodic_cycle_flag and periodic_score >= segment_score)
    

    def overall_summary_phrase() -> str:
        if periodic_grouping_flag:
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
    start_val, end_val = float(target[0]), float(target[-1])

    intro_head = f"这段窗口覆盖时间为 {t0} 到 {t1}，共 {L} 个观测点。"
    if no_change_flag:
        trait = "整体保持恒定"
        intro_body = (
            f"目标{chinese_name}在该窗口内取值恒为 {g_mean:.1f}，"
            f"起始点为 {start_val:.1f}，结束点为 {end_val:.1f}，{trait}。"
        )
    else:
        if periodic_grouping_flag:
            trait = "整体呈明显周期性波动" if best_corr >= 0.7 else "整体呈周期性波动"
        elif oscillation_flag and not plateau_local_flag:
            trait = "整体变化不规则，短时起伏较频繁"
        elif plateau_local_flag:
            trait = "整体以平台波动为主，仅局部有起伏"
        else:
            phrase = shape_phrase()
            trait = phrase[2:] if phrase.startswith("整体") else phrase
        intro_body = (
            f"目标{chinese_name}的取值大致落在 {g_min:.1f} 到 {g_max:.1f} 之间，平均约 {g_mean:.1f}，"
            f"起始点为 {start_val:.1f}，结束点为 {end_val:.1f}，{trait}。"
        )
    lines.append("窗口概览：\n" + intro_head + intro_body)

    if segments:
        if no_change_flag:
            segment_lines = [
                f"各分段数值均保持在 {g_mean:.1f}，未见可辨别的起伏。",
                "窗口内各时段形态一致，整体为恒定序列。",
            ]
            lines.append("分段观察：\n" + "\n".join(segment_lines))
        else:
            value_span = max(g_max - g_min, 1e-6)
            vol_rank = {"low": 0, "medium": 1, "high": 2}

            base_parts: list[dict] = []
            for seg in segments:
                s_idx = int(seg["start"])
                e_idx = int(seg["end"])
                seg_vals = target[s_idx : e_idx + 1]
                local_lag, local_corr = estimate_period(
                    seg_vals,
                    min_lag=4,
                    max_lag=min(96, max(8, len(seg_vals) // 2)),
                )
                local_periodic = bool(
                    len(seg_vals) >= 48
                    and local_lag >= 8
                    and local_corr >= 0.42
                    and len(seg_vals) >= int(1.6 * local_lag)
                )
                local_wave_type = "nonperiodic"
                local_high_ratio = 0.0
                if local_periodic:
                    local_wave_type, local_high_ratio = cycle_wave_profile(seg_vals, int(local_lag))
                base_parts.append(
                    {
                        "seg": seg,
                        "start": s_idx,
                        "end": e_idx,
                        "mean": float(seg.get("mean", 0.0)),
                        "std": float(seg.get("std", 0.0)),
                        "trend": str(seg.get("trend_label", "flat")),
                        "vol": str(seg.get("vol_level", "medium")),
                        "periodic": local_periodic,
                        "lag": int(local_lag),
                        "corr": float(local_corr),
                        "wave_type": local_wave_type,
                        "high_ratio": float(local_high_ratio),
                        "vals": seg_vals,
                    }
                )

            groups: list[dict] = []
            for part in base_parts:
                if not groups:
                    groups.append({"parts": [part]})
                    continue
                prev = groups[-1]["parts"][-1]
                mean_gap = abs(part["mean"] - prev["mean"]) / value_span
                prev_std = max(prev["std"], 1e-6)
                cur_std = max(part["std"], 1e-6)
                std_shift = max(cur_std / prev_std, prev_std / cur_std) - 1.0
                trend_shift = part["trend"] != prev["trend"]
                amp_shift = abs(vol_rank.get(part["vol"], 1) - vol_rank.get(prev["vol"], 1)) >= 1
                regime_shift = part["periodic"] != prev["periodic"]
                wave_shift = (
                    part["periodic"]
                    and prev["periodic"]
                    and part.get("wave_type", "") != prev.get("wave_type", "")
                )
                high_ratio_gap = abs(float(part.get("high_ratio", 0.0)) - float(prev.get("high_ratio", 0.0)))
                mean_thr = 0.16 if periodic_grouping_flag else 0.12
                std_thr = 0.55 if periodic_grouping_flag else 0.45

                split_here = (
                    ((not periodic_grouping_flag) and regime_shift)
                    or mean_gap >= mean_thr
                    or std_shift >= std_thr
                    or (trend_shift and mean_gap >= 0.05)
                    or (amp_shift and (mean_gap >= 0.06 or std_shift >= 0.25))
                    or (wave_shift and high_ratio_gap >= 0.08)
                    or (part["periodic"] and prev["periodic"] and high_ratio_gap >= 0.16)
                )
                if split_here:
                    groups.append({"parts": [part]})
                else:
                    groups[-1]["parts"].append(part)

            merged: list[dict] = []
            for grp in groups:
                if not merged:
                    merged.append(grp)
                    continue
                left = merged[-1]["parts"]
                right = grp["parts"]
                left_periodic = sum(1 for p in left if p["periodic"]) >= max(1, len(left) // 2)
                right_periodic = sum(1 for p in right if p["periodic"]) >= max(1, len(right) // 2)
                left_mean = float(np.mean([p["mean"] for p in left]))
                right_mean = float(np.mean([p["mean"] for p in right]))
                left_std = float(np.mean([p["std"] for p in left]))
                right_std = float(np.mean([p["std"] for p in right]))
                mean_gap = abs(right_mean - left_mean) / value_span
                std_shift = max(right_std / max(left_std, 1e-6), left_std / max(right_std, 1e-6)) - 1.0
                left_dir = left[-1]["trend"]
                right_dir = right[0]["trend"]
                left_wave = max(left, key=lambda p: p["corr"]).get("wave_type", "")
                right_wave = max(right, key=lambda p: p["corr"]).get("wave_type", "")
                left_hr = float(np.mean([float(p.get("high_ratio", 0.0)) for p in left]))
                right_hr = float(np.mean([float(p.get("high_ratio", 0.0)) for p in right]))
                wave_consistent = (left_wave == right_wave) and (abs(left_hr - right_hr) < 0.12)
                close_enough = (
                    left_periodic == right_periodic
                    and mean_gap < 0.05
                    and std_shift < 0.20
                    and left_dir == right_dir
                    and ((not left_periodic) or wave_consistent)
                )
                if close_enough:
                    merged[-1]["parts"].extend(right)
                else:
                    merged.append(grp)
            groups = merged

            # 二次并段：若相邻分组都判为周期段，则放宽中枢阈值，优先保持完整周期阶段。
            def classify_group_periodic(parts: list[dict]) -> bool:
                if not parts:
                    return False
                s = int(parts[0]["start"])
                e = int(parts[-1]["end"])
                vals = target[s : e + 1]
                lag, corr = estimate_period(
                    vals,
                    min_lag=8,
                    max_lag=min(128, max(16, len(vals) // 2)),
                )
                lag_close_global = abs(lag - best_lag) <= max(10, int(0.35 * max(best_lag, 1)))
                strong_rule = (
                    lag >= 8
                    and corr >= 0.45
                    and len(vals) >= int(1.6 * lag)
                    and ((not periodic_grouping_flag) or lag_close_global or corr >= 0.70)
                )
                weak_global_rule = (
                    periodic_grouping_flag
                    and lag >= 8
                    and corr >= 0.28
                    and len(vals) >= lag
                    and abs(lag - best_lag) <= max(8, int(0.30 * max(best_lag, 1)))
                )
                global_lag_rule = False
                if periodic_grouping_flag and best_lag >= 8 and len(vals) >= max(best_lag + 32, int(1.4 * best_lag)):
                    a = vals[:-best_lag]
                    b = vals[best_lag:]
                    if len(a) >= 16 and float(np.std(a)) > 1e-8 and float(np.std(b)) > 1e-8:
                        corr_global = float(np.corrcoef(a, b)[0, 1])
                        global_lag_rule = bool(np.isfinite(corr_global) and corr_global >= 0.30)
                return bool(strong_rule or weak_global_rule or global_lag_rule)

            periodic_merged: list[dict] = []
            for grp in groups:
                if not periodic_merged:
                    periodic_merged.append(grp)
                    continue
                left = periodic_merged[-1]["parts"]
                right = grp["parts"]
                left_periodic = classify_group_periodic(left)
                right_periodic = classify_group_periodic(right)
                left_mean = float(np.mean([p["mean"] for p in left]))
                right_mean = float(np.mean([p["mean"] for p in right]))
                left_std = float(np.mean([p["std"] for p in left]))
                right_std = float(np.mean([p["std"] for p in right]))
                mean_gap = abs(right_mean - left_mean) / value_span
                std_shift = max(right_std / max(left_std, 1e-6), left_std / max(right_std, 1e-6)) - 1.0
                left_wave = max(left, key=lambda p: p["corr"]).get("wave_type", "")
                right_wave = max(right, key=lambda p: p["corr"]).get("wave_type", "")
                left_hr = float(np.mean([float(p.get("high_ratio", 0.0)) for p in left]))
                right_hr = float(np.mean([float(p.get("high_ratio", 0.0)) for p in right]))
                wave_similar = (left_wave == right_wave) and (abs(left_hr - right_hr) < 0.14)
                if left_periodic and right_periodic and mean_gap < 0.22 and std_shift < 0.45 and wave_similar:
                    periodic_merged[-1]["parts"].extend(right)
                else:
                    periodic_merged.append(grp)
            groups = periodic_merged

            group_periodic_flags = [classify_group_periodic(grp["parts"]) for grp in groups]

            if len(groups) >= 2 and periodic_grouping_flag:
                last_idx = len(groups) - 1
                prev_idx = last_idx - 1
                if (not group_periodic_flags[last_idx]) and group_periodic_flags[prev_idx]:
                    last_parts = groups[last_idx]["parts"]
                    prev_parts = groups[prev_idx]["parts"]
                    last_len = int(last_parts[-1]["end"] - last_parts[0]["start"] + 1)
                    prev_mean = float(np.mean([p["mean"] for p in prev_parts]))
                    last_mean = float(np.mean([p["mean"] for p in last_parts]))
                    mean_gap = abs(last_mean - prev_mean) / value_span
                    if last_len <= max(128, int(1.4 * max(best_lag, 1))) and mean_gap < 0.22:
                        groups[prev_idx]["parts"].extend(last_parts)
                        groups.pop()
                        group_periodic_flags = [classify_group_periodic(grp["parts"]) for grp in groups]

            # 纯周期双段纠偏：若首段占比偏小且两段单周期形态相反，可能是窗口起点落在周期尾段导致过早切分。
            if periodic_grouping_flag and len(groups) == 2 and all(group_periodic_flags):
                left_parts = groups[0]["parts"]
                right_parts = groups[1]["parts"]
                left_len = int(left_parts[-1]["end"] - left_parts[0]["start"] + 1)
                right_len = int(right_parts[-1]["end"] - right_parts[0]["start"] + 1)
                total_len = max(left_len + right_len, 1)
                left_share = left_len / total_len
                left_mean_now = float(np.mean([p["mean"] for p in left_parts]))
                right_mean_now = float(np.mean([p["mean"] for p in right_parts]))
                upward_level_shift = (right_mean_now - left_mean_now) / max(value_span, 1e-6)

                if left_share <= 0.52 and upward_level_shift >= 0.05:
                    parts_stream: list[dict] = []
                    for grp in groups:
                        parts_stream.extend(grp["parts"])

                    best_split_idx: int | None = None
                    if len(parts_stream) >= 3:
                        part_lens = [int(p["end"] - p["start"] + 1) for p in parts_stream]
                        best_score = -1e9

                        for split_idx in range(1, len(parts_stream)):
                            l_len = sum(part_lens[:split_idx])
                            r_len = sum(part_lens[split_idx:])
                            if l_len <= 0 or r_len <= 0:
                                continue
                            l_share = l_len / total_len
                            if l_share < 0.55 or l_share > 0.82:
                                continue

                            cand_left = parts_stream[:split_idx]
                            cand_right = parts_stream[split_idx:]
                            if not (classify_group_periodic(cand_left) and classify_group_periodic(cand_right)):
                                continue

                            l_mean = float(np.mean([p["mean"] for p in cand_left]))
                            r_mean = float(np.mean([p["mean"] for p in cand_right]))
                            l_std = float(np.mean([p["std"] for p in cand_left]))
                            r_std = float(np.mean([p["std"] for p in cand_right]))
                            mean_gap = abs(r_mean - l_mean) / value_span
                            std_shift = max(r_std / max(l_std, 1e-6), l_std / max(r_std, 1e-6)) - 1.0
                            late_pref = 1.0 - abs(l_share - 0.68)
                            score = 1.45 * mean_gap + 0.55 * std_shift + 0.22 * late_pref

                            if score > best_score:
                                best_score = score
                                best_split_idx = split_idx

                    if best_split_idx is not None:
                        groups = [
                            {"parts": parts_stream[:best_split_idx]},
                            {"parts": parts_stream[best_split_idx:]},
                        ]
                        group_periodic_flags = [classify_group_periodic(grp["parts"]) for grp in groups]
                    else:
                        # 兜底：并段后可选边界过少时，直接在窗口索引上重选切点。
                        min_side = max(64, int(1.2 * max(best_lag, 1)))
                        cut_start = max(int(0.55 * total_len), min_side)
                        cut_end = min(int(0.82 * total_len), total_len - min_side)
                        step = max(8, int(max(best_lag, 8) // 4))
                        best_cut: int | None = None
                        best_cut_score = -1e9

                        for cut in range(cut_start, cut_end + 1, step):
                            left_vals = target[:cut]
                            right_vals = target[cut:]
                            if len(left_vals) < min_side or len(right_vals) < min_side:
                                continue

                            l_lag, l_corr = estimate_period(
                                left_vals,
                                min_lag=8,
                                max_lag=min(128, max(16, len(left_vals) // 2)),
                            )
                            r_lag, r_corr = estimate_period(
                                right_vals,
                                min_lag=8,
                                max_lag=min(128, max(16, len(right_vals) // 2)),
                            )
                            l_periodic = l_lag >= 8 and l_corr >= 0.28 and len(left_vals) >= max(l_lag, int(1.5 * l_lag))
                            r_periodic = r_lag >= 8 and r_corr >= 0.28 and len(right_vals) >= max(r_lag, int(1.5 * r_lag))
                            if not (l_periodic and r_periodic):
                                continue

                            l_mean = float(np.mean(left_vals))
                            r_mean = float(np.mean(right_vals))
                            l_std = float(np.std(left_vals))
                            r_std = float(np.std(right_vals))
                            mean_gap = abs(r_mean - l_mean) / value_span
                            std_shift = max(r_std / max(l_std, 1e-6), l_std / max(r_std, 1e-6)) - 1.0
                            l_share = cut / total_len
                            late_pref = 1.0 - abs(l_share - 0.68)
                            score = 1.45 * mean_gap + 0.55 * std_shift + 0.22 * late_pref

                            if score > best_cut_score:
                                best_cut_score = score
                                best_cut = cut

                        if best_cut is not None:
                            def build_period_part(s: int, e: int) -> dict:
                                vals = target[s : e + 1]
                                lag, corr = estimate_period(
                                    vals,
                                    min_lag=8,
                                    max_lag=min(128, max(16, len(vals) // 2)),
                                )
                                std_v = float(np.std(vals))
                                span_v = max(float(np.max(vals) - np.min(vals)), 1e-6)
                                drift_v = float(vals[-1] - vals[0]) / span_v
                                if drift_v >= 0.22:
                                    trend_v = "weak_up"
                                elif drift_v <= -0.22:
                                    trend_v = "weak_down"
                                else:
                                    trend_v = "flat"
                                if std_v >= 1.35 * max(g_std, 1e-6):
                                    vol_v = "high"
                                elif std_v <= 0.75 * max(g_std, 1e-6):
                                    vol_v = "low"
                                else:
                                    vol_v = "medium"
                                return {
                                    "seg": {"start": s, "end": e, "mean": float(np.mean(vals)), "std": std_v, "trend_label": trend_v, "vol_level": vol_v},
                                    "start": s,
                                    "end": e,
                                    "mean": float(np.mean(vals)),
                                    "std": std_v,
                                    "trend": trend_v,
                                    "vol": vol_v,
                                    "periodic": True,
                                    "lag": int(lag),
                                    "corr": float(corr),
                                    "vals": vals,
                                }

                            left_part = build_period_part(0, best_cut - 1)
                            right_part = build_period_part(best_cut, total_len - 1)
                            groups = [{"parts": [left_part]}, {"parts": [right_part]}]
                            group_periodic_flags = [True, True]

            # 周期窗口默认不细分；但若存在明显阶段切换（如前高后低、前段与后段形态差异大），保留两段。
            if periodic_grouping_flag and len(groups) > 1:
                def group_len(grp: dict) -> int:
                    parts = grp["parts"]
                    return int(parts[-1]["end"] - parts[0]["start"] + 1)

                def detect_cycle_regime_shift(vals: np.ndarray, lag: int) -> bool:
                    if lag < 8 or len(vals) < max(4 * lag, 96):
                        return False
                    cycles = [vals[i : i + lag] for i in range(0, len(vals) - lag + 1, lag)]
                    if len(cycles) < 4:
                        return False
                    peaks = np.array([float(np.max(c)) for c in cycles], dtype=float)
                    troughs = np.array([float(np.min(c)) for c in cycles], dtype=float)
                    means = np.array([float(np.mean(c)) for c in cycles], dtype=float)
                    amps = peaks - troughs
                    h = len(cycles) // 2
                    p1, p2 = float(np.median(peaks[:h])), float(np.median(peaks[h:]))
                    a1, a2 = float(np.median(amps[:h])), float(np.median(amps[h:]))
                    m1, m2 = float(np.median(means[:h])), float(np.median(means[h:]))
                    peak_ratio = p2 / max(abs(p1), 1e-6)
                    amp_ratio = a2 / max(abs(a1), 1e-6)
                    mean_gap_ratio = abs(m2 - m1) / max(value_span, 1e-6)
                    return bool(
                        peak_ratio <= 0.80
                        or peak_ratio >= 1.25
                        or amp_ratio <= 0.75
                        or amp_ratio >= 1.35
                        or mean_gap_ratio >= 0.18
                    )

                mixed_regime = any(group_periodic_flags) and any(not f for f in group_periodic_flags)
                cycle_regime_shift = detect_cycle_regime_shift(target, int(best_lag))

                if mixed_regime or cycle_regime_shift:
                    # 保留两段：混合场景优先按“非周期前段 + 周期后段”切；纯周期切换按时间中点切。
                    if mixed_regime:
                        parts_stream: list[dict] = []
                        for grp in groups:
                            parts_stream.extend(grp["parts"])

                        if len(parts_stream) >= 2:
                            part_lens = [int(p["end"] - p["start"] + 1) for p in parts_stream]
                            part_flags = [bool(p["periodic"]) for p in parts_stream]
                            total_len = max(sum(part_lens), 1)
                            min_side_len = max(48, int(0.75 * max(best_lag, 1)))

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

                            best_score = -1e9
                            best_split_idx: int | None = None
                            for split_idx in range(1, len(parts_stream)):
                                left_lens = part_lens[:split_idx]
                                right_lens = part_lens[split_idx:]
                                left_len = sum(left_lens)
                                right_len = sum(right_lens)
                                if left_len < min_side_len or right_len < min_side_len:
                                    continue

                                left_periodic_len = sum(l for l, f in zip(left_lens, part_flags[:split_idx]) if f)
                                right_periodic_len = sum(l for l, f in zip(right_lens, part_flags[split_idx:]) if f)
                                left_ratio = left_periodic_len / max(left_len, 1)
                                right_ratio = right_periodic_len / max(right_len, 1)

                                left_start = int(parts_stream[0]["start"])
                                left_end = int(parts_stream[split_idx - 1]["end"])
                                left_vals = target[left_start : left_end + 1]
                                left_share = left_len / total_len

                                # 右侧越周期、左侧越非周期越好；同时避免极端短段，并轻微偏向稍晚切分。
                                contrast = (right_ratio - left_ratio)
                                balance = min(left_len, right_len) / total_len
                                late_bias = left_share
                                score = 2.2 * contrast + 0.8 * balance + 0.15 * late_bias

                                # 左段若已明显周期化，则抑制继续向后切分，避免“过晚切段”。
                                if left_ratio > 0.55:
                                    score -= 1.35 * (left_ratio - 0.55)

                                # 右段周期占比不足时降权，避免过早进入“周期段”。
                                if right_ratio < 0.60:
                                    score -= 0.35

                                # 若左段已呈明显“先升后降/先降后升”，适度鼓励把左段保留得更完整。
                                if has_hump_shape(left_vals) and left_share < 0.56 and left_ratio <= 0.45:
                                    score += 0.22

                                if score > best_score:
                                    best_score = score
                                    best_split_idx = split_idx

                            if best_split_idx is not None:
                                left_parts = parts_stream[:best_split_idx]
                                right_parts = parts_stream[best_split_idx:]
                                if left_parts and right_parts:
                                    groups = [{"parts": left_parts}, {"parts": right_parts}]
                    if len(groups) > 2:
                        total_len = sum(group_len(g) for g in groups)
                        min_side_len = max(48, int(0.14 * max(total_len, 1)))
                        best_score = -1e9
                        best_split_idx: int | None = None
                        acc = 0

                        for gi in range(1, len(groups)):
                            acc += group_len(groups[gi - 1])
                            left_len = acc
                            right_len = total_len - acc
                            if left_len < min_side_len or right_len < min_side_len:
                                continue

                            left_grp = groups[gi - 1]["parts"]
                            right_grp = groups[gi]["parts"]
                            left_mean = float(np.mean([p["mean"] for p in left_grp]))
                            right_mean = float(np.mean([p["mean"] for p in right_grp]))
                            left_std = float(np.mean([p["std"] for p in left_grp]))
                            right_std = float(np.mean([p["std"] for p in right_grp]))
                            mean_gap = abs(right_mean - left_mean) / value_span
                            std_shift = max(right_std / max(left_std, 1e-6), left_std / max(right_std, 1e-6)) - 1.0

                            left_periodic = bool(group_periodic_flags[gi - 1])
                            right_periodic = bool(group_periodic_flags[gi])
                            regime_shift = left_periodic != right_periodic
                            left_wave = max(left_grp, key=lambda p: p["corr"]).get("wave_type", "") if left_grp else ""
                            right_wave = max(right_grp, key=lambda p: p["corr"]).get("wave_type", "") if right_grp else ""
                            left_hr = float(np.mean([float(p.get("high_ratio", 0.0)) for p in left_grp])) if left_grp else 0.0
                            right_hr = float(np.mean([float(p.get("high_ratio", 0.0)) for p in right_grp])) if right_grp else 0.0
                            high_gap = abs(right_hr - left_hr)
                            wave_shift = left_periodic and right_periodic and (left_wave != right_wave)

                            score = 1.85 * mean_gap + 0.85 * std_shift + 0.95 * high_gap
                            if regime_shift:
                                score += 0.85
                            if wave_shift:
                                score += 0.65

                            if mixed_regime:
                                left_periodic_len = sum(
                                    group_len(groups[k]) for k in range(gi) if group_periodic_flags[k]
                                )
                                right_periodic_len = sum(
                                    group_len(groups[k]) for k in range(gi, len(groups)) if group_periodic_flags[k]
                                )
                                left_ratio = left_periodic_len / max(left_len, 1)
                                right_ratio = right_periodic_len / max(right_len, 1)
                                score += 1.20 * max(0.0, right_ratio - left_ratio)

                            # 轻微抑制“总在中间切”，鼓励按真实边界信号切分。
                            share = left_len / max(total_len, 1)
                            mid_bias_penalty = max(0.0, 1.0 - 2.0 * abs(share - 0.5))
                            score -= 0.10 * mid_bias_penalty

                            if score > best_score:
                                best_score = score
                                best_split_idx = gi

                        if best_split_idx is not None:
                            # 非混合场景下，若边界证据过弱则不强制分段。
                            if (not mixed_regime) and best_score < 0.34:
                                merged_parts: list[dict] = []
                                for grp in groups:
                                    merged_parts.extend(grp["parts"])
                                groups = [{"parts": merged_parts}]
                            else:
                                left_parts: list[dict] = []
                                right_parts: list[dict] = []
                                for gi, grp in enumerate(groups):
                                    if gi < best_split_idx:
                                        left_parts.extend(grp["parts"])
                                    else:
                                        right_parts.extend(grp["parts"])
                                if left_parts and right_parts:
                                    groups = [{"parts": left_parts}, {"parts": right_parts}]
                    group_periodic_flags = [classify_group_periodic(grp["parts"]) for grp in groups]
                else:
                    merged_parts: list[dict] = []
                    for grp in groups:
                        merged_parts.extend(grp["parts"])
                    groups = [{"parts": merged_parts}]
                    group_periodic_flags = [True]

            # 后置纠偏：对最终两段结果再检查一次，修正“纯周期且首段偏短”的过早切段。
            if periodic_grouping_flag and len(groups) == 2 and all(group_periodic_flags):
                left_parts = groups[0]["parts"]
                right_parts = groups[1]["parts"]
                left_len = int(left_parts[-1]["end"] - left_parts[0]["start"] + 1)
                total_len = int(right_parts[-1]["end"] - left_parts[0]["start"] + 1)
                total_len = max(total_len, 1)
                left_share = left_len / total_len
                left_mean = float(np.mean([p["mean"] for p in left_parts]))
                right_mean = float(np.mean([p["mean"] for p in right_parts]))
                upward_level_shift = (right_mean - left_mean) / max(value_span, 1e-6)

                if left_share <= 0.52 and upward_level_shift >= 0.05:
                    min_side = max(64, int(1.15 * max(best_lag, 1)))
                    cut_start = max(int(0.55 * total_len), min_side)
                    cut_end = min(int(0.80 * total_len), total_len - min_side)
                    step = max(8, int(max(best_lag, 8) // 4))
                    best_cut: int | None = None
                    best_score = -1e9

                    for cut in range(cut_start, cut_end + 1, step):
                        left_vals = target[:cut]
                        right_vals = target[cut:]
                        if len(left_vals) < min_side or len(right_vals) < min_side:
                            continue

                        l_lag, l_corr = estimate_period(
                            left_vals,
                            min_lag=8,
                            max_lag=min(128, max(16, len(left_vals) // 2)),
                        )
                        r_lag, r_corr = estimate_period(
                            right_vals,
                            min_lag=8,
                            max_lag=min(128, max(16, len(right_vals) // 2)),
                        )
                        l_periodic = l_lag >= 8 and l_corr >= 0.24 and len(left_vals) >= max(l_lag, int(1.4 * l_lag))
                        r_periodic = r_lag >= 8 and r_corr >= 0.24 and len(right_vals) >= max(r_lag, int(1.4 * r_lag))
                        if not (l_periodic and r_periodic):
                            continue

                        l_mean = float(np.mean(left_vals))
                        r_mean = float(np.mean(right_vals))
                        l_std = float(np.std(left_vals))
                        r_std = float(np.std(right_vals))
                        mean_gap = abs(r_mean - l_mean) / value_span
                        std_shift = max(r_std / max(l_std, 1e-6), l_std / max(r_std, 1e-6)) - 1.0
                        share = cut / total_len
                        late_pref = 1.0 - abs(share - 0.66)
                        score = 1.35 * mean_gap + 0.50 * std_shift + 0.30 * late_pref

                        if score > best_score:
                            best_score = score
                            best_cut = cut

                    if best_cut is not None:
                        def build_period_part(s: int, e: int) -> dict:
                            vals = target[s : e + 1]
                            lag, corr = estimate_period(
                                vals,
                                min_lag=8,
                                max_lag=min(128, max(16, len(vals) // 2)),
                            )
                            std_v = float(np.std(vals))
                            span_v = max(float(np.max(vals) - np.min(vals)), 1e-6)
                            drift_v = float(vals[-1] - vals[0]) / span_v
                            if drift_v >= 0.22:
                                trend_v = "weak_up"
                            elif drift_v <= -0.22:
                                trend_v = "weak_down"
                            else:
                                trend_v = "flat"
                            if std_v >= 1.35 * max(g_std, 1e-6):
                                vol_v = "high"
                            elif std_v <= 0.75 * max(g_std, 1e-6):
                                vol_v = "low"
                            else:
                                vol_v = "medium"
                            return {
                                "seg": {"start": s, "end": e, "mean": float(np.mean(vals)), "std": std_v, "trend_label": trend_v, "vol_level": vol_v},
                                "start": s,
                                "end": e,
                                "mean": float(np.mean(vals)),
                                "std": std_v,
                                "trend": trend_v,
                                "vol": vol_v,
                                "periodic": True,
                                "lag": int(lag),
                                "corr": float(corr),
                                "vals": vals,
                            }

                        groups = [
                            {"parts": [build_period_part(0, best_cut - 1)]},
                            {"parts": [build_period_part(best_cut, total_len - 1)]},
                        ]
                        group_periodic_flags = [True, True]

            # 起始异常前缀纠偏：若窗口开头1-2个周期与后续主周期明显不一致，优先在前缀结束处切分。
            if periodic_grouping_flag and len(groups) == 2 and all(group_periodic_flags) and best_lag >= 8:
                left_len = int(groups[0]["parts"][-1]["end"] - groups[0]["parts"][0]["start"] + 1)
                total_len = int(groups[1]["parts"][-1]["end"] - groups[0]["parts"][0]["start"] + 1)
                total_len = max(total_len, 1)
                left_share = left_len / total_len

                # 仅在当前切点偏中后位置时尝试前移，避免干扰已合理的早切样本。
                if 0.48 <= left_share <= 0.80:
                    lag = int(best_lag)
                    n_cycles = len(target) // lag
                    if n_cycles >= 4:
                        cycles = [target[i * lag : (i + 1) * lag] for i in range(n_cycles)]

                        def _norm_cycle(c: np.ndarray) -> np.ndarray:
                            c = np.asarray(c, dtype=float)
                            return (c - float(np.mean(c))) / max(float(np.std(c)), 1e-8)

                        norm_cycles = [_norm_cycle(c) for c in cycles]
                        tail = np.vstack(norm_cycles[max(1, n_cycles // 2) :])
                        ref = np.median(tail, axis=0)

                        corr_seq: list[float] = []
                        for c in norm_cycles:
                            v = float(np.corrcoef(c, ref)[0, 1])
                            corr_seq.append(v if np.isfinite(v) else 0.0)

                        prefix_cycles = 0
                        for i, v in enumerate(corr_seq[:2]):
                            if v < 0.35:
                                prefix_cycles = i + 1
                            else:
                                break

                        good_follow = False
                        if prefix_cycles >= 1 and prefix_cycles + 1 < len(corr_seq):
                            follow = corr_seq[prefix_cycles : min(prefix_cycles + 2, len(corr_seq))]
                            good_follow = bool(len(follow) >= 1 and float(np.mean(follow)) >= 0.52)

                        if prefix_cycles >= 1 and good_follow:
                            cut = prefix_cycles * lag
                            min_side = max(48, int(0.90 * lag))
                            if min_side <= cut <= (len(target) - min_side):
                                def _build_period_part(s: int, e: int) -> dict:
                                    vals = target[s : e + 1]
                                    lag_v, corr_v = estimate_period(
                                        vals,
                                        min_lag=8,
                                        max_lag=min(128, max(16, len(vals) // 2)),
                                    )
                                    std_v = float(np.std(vals))
                                    span_v = max(float(np.max(vals) - np.min(vals)), 1e-6)
                                    drift_v = float(vals[-1] - vals[0]) / span_v
                                    if drift_v >= 0.22:
                                        trend_v = "weak_up"
                                    elif drift_v <= -0.22:
                                        trend_v = "weak_down"
                                    else:
                                        trend_v = "flat"
                                    if std_v >= 1.35 * max(g_std, 1e-6):
                                        vol_v = "high"
                                    elif std_v <= 0.75 * max(g_std, 1e-6):
                                        vol_v = "low"
                                    else:
                                        vol_v = "medium"
                                    wave_v, high_v = cycle_wave_profile(vals, max(int(lag_v), 1))
                                    return {
                                        "seg": {"start": s, "end": e, "mean": float(np.mean(vals)), "std": std_v, "trend_label": trend_v, "vol_level": vol_v},
                                        "start": s,
                                        "end": e,
                                        "mean": float(np.mean(vals)),
                                        "std": std_v,
                                        "trend": trend_v,
                                        "vol": vol_v,
                                        "periodic": True,
                                        "lag": int(lag_v),
                                        "corr": float(corr_v),
                                        "wave_type": wave_v,
                                        "high_ratio": float(high_v),
                                        "vals": vals,
                                    }

                                groups = [
                                    {"parts": [_build_period_part(0, cut - 1)]},
                                    {"parts": [_build_period_part(cut, len(target) - 1)]},
                                ]
                                group_periodic_flags = [True, True]

            # 兜底约束：确保最终分段连续且覆盖完整窗口 [0, L-1]。
            if groups:
                flat_parts = [p for grp in groups for p in grp.get("parts", [])]
                if flat_parts:
                    flat_parts.sort(key=lambda p: int(p.get("start", 0)))

                    def _refresh_part(part: dict, s: int, e: int) -> dict:
                        s = int(max(0, min(s, L - 1)))
                        e = int(max(0, min(e, L - 1)))
                        if e <= s:
                            if s < L - 1:
                                e = s + 1
                            else:
                                s = max(0, s - 1)
                                e = s + 1
                        vals = target[s : e + 1]
                        trend_v, _ = classify_global_trend(vals)
                        std_v = float(np.std(vals))
                        if std_v >= 1.35 * max(g_std, 1e-6):
                            vol_v = "high"
                        elif std_v <= 0.75 * max(g_std, 1e-6):
                            vol_v = "low"
                        else:
                            vol_v = "medium"
                        part["start"] = int(s)
                        part["end"] = int(e)
                        part["vals"] = vals
                        part["mean"] = float(np.mean(vals))
                        part["std"] = std_v
                        part["trend"] = trend_v
                        part["vol"] = vol_v
                        if isinstance(part.get("seg"), dict):
                            part["seg"]["start"] = int(s)
                            part["seg"]["end"] = int(e)
                            part["seg"]["mean"] = float(np.mean(vals))
                            part["seg"]["std"] = std_v
                            part["seg"]["trend_label"] = trend_v
                            part["seg"]["vol_level"] = vol_v
                        return part

                    flat_parts[0] = _refresh_part(flat_parts[0], 0, int(flat_parts[0]["end"]))
                    cursor = int(flat_parts[0]["end"]) + 1
                    for j in range(1, len(flat_parts)):
                        cur = flat_parts[j]
                        cur_end = int(cur["end"])
                        cur_start = max(int(cur.get("start", cursor)), cursor)
                        if cur_start > cur_end:
                            cur_start = cur_end
                        flat_parts[j] = _refresh_part(cur, cur_start, cur_end)
                        cursor = int(flat_parts[j]["end"]) + 1

                    if int(flat_parts[-1]["end"]) < L - 1:
                        flat_parts[-1] = _refresh_part(flat_parts[-1], int(flat_parts[-1]["start"]), L - 1)

                    # 按原分组大小回填，保持阶段数基本不变。
                    group_sizes = [len(grp.get("parts", [])) for grp in groups]
                    rebuilt_groups: list[dict] = []
                    pos = 0
                    for gi, sz in enumerate(group_sizes):
                        if sz <= 0:
                            continue
                        take = flat_parts[pos : pos + sz]
                        pos += sz
                        if gi == len(group_sizes) - 1 and pos < len(flat_parts):
                            take = take + flat_parts[pos:]
                            pos = len(flat_parts)
                        if take:
                            rebuilt_groups.append({"parts": take})
                    groups = rebuilt_groups if rebuilt_groups else [{"parts": flat_parts}]
                    group_periodic_flags = [classify_group_periodic(grp["parts"]) for grp in groups]

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

            def nonperiodic_shape_and_range(vals: np.ndarray) -> tuple[str, float, float]:
                lo = float(np.min(vals))
                hi = float(np.max(vals))
                return f"段内呈{phase_shape_phrase(vals)}", lo, hi

            segment_lines = []
            if len(groups) == 1:
                parts = groups[0]["parts"]
                stage_start = int(parts[0]["start"])
                stage_end = int(parts[-1]["end"])
                stage_mean = float(np.mean([p["mean"] for p in parts]))
                stage_std = float(np.mean([p["std"] for p in parts]))
                stage_periodic = bool((group_periodic_flags and group_periodic_flags[0]) or periodic_cycle_flag)
                if stage_periodic:
                    rep = max(parts, key=lambda p: p["corr"])
                    rep_lag = int(rep["lag"]) if int(rep["lag"]) > 0 else int(best_lag)
                    rep_lag = max(rep_lag, 1)
                    stage_vals = target[stage_start : stage_end + 1]
                    stage_lag, stage_corr = estimate_period(
                        stage_vals,
                        min_lag=8,
                        max_lag=min(128, max(16, len(stage_vals) // 2)),
                    )
                    if stage_lag < 8:
                        stage_lag = rep_lag
                    if stage_corr <= 0.05:
                        stage_corr = float(rep.get("corr", 0.0))
                    rep_every, _ = build_period_phrases(int(stage_lag), step_delta)
                    cycle_len = min(rep_lag, len(target))
                    cycle_vals = target[:cycle_len]
                    stage_shape = cycle_shape_phrase(cycle_vals)
                    stage_lo = float(np.min(stage_vals)) if len(stage_vals) else float(np.min(cycle_vals))
                    stage_hi = float(np.max(stage_vals)) if len(stage_vals) else float(np.max(cycle_vals))
                    stage_std_real = float(np.std(stage_vals)) if len(stage_vals) else 0.0
                    stage_span_real = max(float(np.max(stage_vals) - np.min(stage_vals)), 1e-6) if len(stage_vals) else 0.0
                    std_ratio = stage_std_real / max(g_std, 1e-6)
                    span_ratio = stage_span_real / max(g_max - g_min, 1e-6)
                    if std_ratio >= 1.20 or span_ratio >= 0.55:
                        rel_vol_phrase = "相对全窗口属于较大波动"
                    elif std_ratio <= 0.75 and span_ratio <= 0.30:
                        rel_vol_phrase = "相对全窗口属于较小波动"
                    else:
                        rel_vol_phrase = "相对全窗口属于中等波动"
                    wave_label, high_ratio = cycle_wave_profile(stage_vals, rep_lag)
                    high_pct = int(round(100.0 * high_ratio))
                    segment_lines.append(
                        f"该窗口内存在稳定重复结构，以周期性变化为主（{rep_every}，相关系数 corr≈{stage_corr:.2f}），单周期通常{stage_shape}，波峰形态更接近{wave_label}，高位段占比约 {high_pct}%，取值大致在 {stage_lo:.1f} 到 {stage_hi:.1f} 之间。"
                    )
                else:
                    stage_vol = "high" if stage_std >= 1.35 * max(g_std, 1e-6) else "low" if stage_std <= 0.75 * max(g_std, 1e-6) else "medium"
                    stage_vals = target[stage_start : stage_end + 1]
                    trend_seq = [p["trend"] for p in parts]
                    up_cnt = sum(t in ("strong_up", "weak_up") for t in trend_seq)
                    down_cnt = sum(t in ("strong_down", "weak_down") for t in trend_seq)
                    stage_span = max(float(np.max(stage_vals) - np.min(stage_vals)), 1e-6)
                    stage_drift = float(stage_vals[-1] - stage_vals[0]) / stage_span
                    if up_cnt > 0 and down_cnt > 0:
                        stage_form = "不规则多阶段起伏"
                    elif up_cnt > 0:
                        stage_form = "以阶段性上行为主" if stage_drift >= 0.25 else "不规则波动"
                    elif down_cnt > 0:
                        stage_form = "以阶段性回落为主" if stage_drift <= -0.25 else "不规则波动"
                    else:
                        stage_form = "整体较平稳"
                    stage_shape_extra, stage_lo, stage_hi = nonperiodic_shape_and_range(stage_vals)
                    stage_start_val = float(stage_vals[0]) if len(stage_vals) else stage_mean
                    stage_end_val = float(stage_vals[-1]) if len(stage_vals) else stage_mean
                    stage_std_real = float(np.std(stage_vals)) if len(stage_vals) else 0.0
                    stage_span_real = max(float(np.max(stage_vals) - np.min(stage_vals)), 1e-6) if len(stage_vals) else 0.0
                    std_ratio = stage_std_real / max(g_std, 1e-6)
                    span_ratio = stage_span_real / max(g_max - g_min, 1e-6)
                    if std_ratio >= 1.20 or span_ratio >= 0.55:
                        rel_vol_phrase = "相对全窗口属于较大波动"
                    elif std_ratio <= 0.75 and span_ratio <= 0.30:
                        rel_vol_phrase = "相对全窗口属于较小波动"
                    else:
                        rel_vol_phrase = "相对全窗口属于中等波动"

                    if ("先抬升后回落" in stage_shape_extra or "先回落后抬升" in stage_shape_extra or "震荡" in stage_shape_extra) and rel_vol_phrase == "相对全窗口属于较小波动":
                        rel_vol_phrase = "相对全窗口属于中等波动"

                    local_lag, local_corr = estimate_period(stage_vals)
                    local_periodic = bool(local_lag >= 8 and local_corr >= 0.45 and len(stage_vals) >= max(48, 2 * local_lag) and not is_micro_noise_period(stage_vals, local_lag, local_corr))
                    periodic_tail = ""
                    if local_periodic:
                        _, local_cycle = build_period_phrases(int(local_lag), step_delta)
                        cyc_len = min(max(int(local_lag), 1), len(stage_vals))
                        cyc_vals = stage_vals[:cyc_len]
                        wave_label, high_ratio = cycle_wave_profile(stage_vals, int(local_lag))
                        high_pct = int(round(100.0 * high_ratio))
                        periodic_tail = (
                            f"；该段呈局部周期性起伏（{local_cycle}，相关系数 corr≈{local_corr:.2f}），"
                            f"波形更接近{wave_label}，高位段占比约 {high_pct}%，"
                            f"取值大致在 {float(period_representative_bounds(stage_vals, int(local_lag))[0]):.1f} 到 {float(period_representative_bounds(stage_vals, int(local_lag))[1]):.1f} 之间"
                        )
                        stage_shape_extra = "段内以周期性起伏为主"
                        stage_form = "以周期性变化为主"
                    if "先抬升后回落" in stage_shape_extra or "先回落后抬升" in stage_shape_extra or "震荡" in stage_shape_extra:
                        stage_form = "存在明显起伏变化"
                    segment_lines.append(
                        f"该窗口内存在稳定重复结构，{stage_form}，且{vol_phrase(stage_vol)}；{stage_shape_extra}，取值大致在 {stage_lo:.1f} 到 {stage_hi:.1f} 之间，{rel_vol_phrase}{periodic_tail}。"
                    )
            else:
                for i, grp in enumerate(groups, start=1):
                    parts = grp["parts"]
                    stage_start = int(parts[0]["start"])
                    stage_end = int(parts[-1]["end"])
                    stage_mean = float(np.mean([p["mean"] for p in parts]))
                    stage_std = float(np.mean([p["std"] for p in parts]))
                    stage_periodic = bool(group_periodic_flags[i - 1])
                    stage_seg = {"start": stage_start, "end": stage_end}

                    if stage_periodic:
                        rep = max(parts, key=lambda p: p["corr"])
                        rep_lag = int(rep["lag"]) if int(rep["lag"]) > 0 else int(best_lag)
                        rep_lag = max(rep_lag, 1)
                        stage_vals = target[stage_start : stage_end + 1]
                        stage_lag, stage_corr = estimate_period(
                            stage_vals,
                            min_lag=8,
                            max_lag=min(128, max(16, len(stage_vals) // 2)),
                        )
                        if stage_lag < 8:
                            stage_lag = rep_lag
                        if stage_corr <= 0.05:
                            stage_corr = float(rep.get("corr", 0.0))
                        rep_every, _ = build_period_phrases(int(stage_lag), step_delta)
                        cycle_len = min(rep_lag, len(rep["vals"]))
                        cycle_vals = rep["vals"][:cycle_len]
                        stage_shape = cycle_shape_phrase(cycle_vals)
                        stage_start_val = float(stage_vals[0]) if len(stage_vals) else stage_mean
                        stage_end_val = float(stage_vals[-1]) if len(stage_vals) else stage_mean
                        stage_lo = float(np.min(stage_vals)) if len(stage_vals) else float(np.min(cycle_vals))
                        stage_hi = float(np.max(stage_vals)) if len(stage_vals) else float(np.max(cycle_vals))
                        stage_std_real = float(np.std(stage_vals)) if len(stage_vals) else 0.0
                        stage_span_real = max(float(np.max(stage_vals) - np.min(stage_vals)), 1e-6) if len(stage_vals) else 0.0
                        std_ratio = stage_std_real / max(g_std, 1e-6)
                        span_ratio = stage_span_real / max(g_max - g_min, 1e-6)
                        if std_ratio >= 1.20 or span_ratio >= 0.55:
                            rel_vol_phrase = "相对全窗口属于较大波动"
                        elif std_ratio <= 0.75 and span_ratio <= 0.30:
                            rel_vol_phrase = "相对全窗口属于较小波动"
                        else:
                            rel_vol_phrase = "相对全窗口属于中等波动"
                        wave_label, high_ratio = cycle_wave_profile(target[stage_start : stage_end + 1], rep_lag)
                        high_pct = int(round(100.0 * high_ratio))
                        segment_lines.append(
                            f"第{i}阶段（{fmt_time_range(stage_seg['start'], stage_seg['end'])}）均值约 {stage_mean:.1f}，起点约 {stage_start_val:.1f}，终点约 {stage_end_val:.1f}，段内大致在 {stage_lo:.1f} 到 {stage_hi:.1f} 之间，以周期性变化为主，{rel_vol_phrase}；该段呈局部周期性起伏（{rep_every}，相关系数 corr≈{stage_corr:.2f}），单周期通常{stage_shape}，波形更接近{wave_label}，高位段占比约 {high_pct}%，取值大致在 {float(period_representative_bounds(stage_vals, int(stage_lag))[0]):.1f} 到 {float(period_representative_bounds(stage_vals, int(stage_lag))[1]):.1f} 之间。"
                        )
                    else:
                        trend_seq = [p["trend"] for p in parts]
                        up_cnt = sum(t in ("strong_up", "weak_up") for t in trend_seq)
                        down_cnt = sum(t in ("strong_down", "weak_down") for t in trend_seq)
                        stage_vals = target[stage_start : stage_end + 1]
                        stage_span = max(float(np.max(stage_vals) - np.min(stage_vals)), 1e-6)
                        stage_drift = float(stage_vals[-1] - stage_vals[0]) / stage_span
                        if up_cnt > 0 and down_cnt > 0:
                            stage_form = "不规则多阶段起伏"
                        elif up_cnt > 0:
                            stage_form = "以阶段性上行为主" if stage_drift >= 0.25 else "不规则波动"
                        elif down_cnt > 0:
                            stage_form = "以阶段性回落为主" if stage_drift <= -0.25 else "不规则波动"
                        else:
                            stage_form = "整体较平稳"
                        stage_vol = "high" if stage_std >= 1.35 * max(g_std, 1e-6) else "low" if stage_std <= 0.75 * max(g_std, 1e-6) else "medium"
                        stage_shape_extra, stage_lo, stage_hi = nonperiodic_shape_and_range(stage_vals)
                        stage_start_val = float(stage_vals[0]) if len(stage_vals) else stage_mean
                        stage_end_val = float(stage_vals[-1]) if len(stage_vals) else stage_mean
                        stage_std_real = float(np.std(stage_vals)) if len(stage_vals) else 0.0
                        stage_span_real = max(float(np.max(stage_vals) - np.min(stage_vals)), 1e-6) if len(stage_vals) else 0.0
                        std_ratio = stage_std_real / max(g_std, 1e-6)
                        span_ratio = stage_span_real / max(g_max - g_min, 1e-6)
                        if std_ratio >= 1.20 or span_ratio >= 0.55:
                            rel_vol_phrase = "相对全窗口属于较大波动"
                        elif std_ratio <= 0.75 and span_ratio <= 0.30:
                            rel_vol_phrase = "相对全窗口属于较小波动"
                        else:
                            rel_vol_phrase = "相对全窗口属于中等波动"

                        if ("先抬升后回落" in stage_shape_extra or "先回落后抬升" in stage_shape_extra or "震荡" in stage_shape_extra) and rel_vol_phrase == "相对全窗口属于较小波动":
                            rel_vol_phrase = "相对全窗口属于中等波动"

                        local_lag, local_corr = estimate_period(stage_vals)
                        local_periodic = bool(local_lag >= 8 and local_corr >= 0.45 and len(stage_vals) >= max(48, 2 * local_lag) and not is_micro_noise_period(stage_vals, local_lag, local_corr))
                        periodic_tail = ""
                        if local_periodic:
                            _, local_cycle = build_period_phrases(int(local_lag), step_delta)
                            cyc_len = min(max(int(local_lag), 1), len(stage_vals))
                            cyc_vals = stage_vals[:cyc_len]
                            wave_label, high_ratio = cycle_wave_profile(stage_vals, int(local_lag))
                            high_pct = int(round(100.0 * high_ratio))
                            periodic_tail = (
                                f"；该段呈局部周期性起伏（{local_cycle}，相关系数 corr≈{local_corr:.2f}），"
                                f"波形更接近{wave_label}，高位段占比约 {high_pct}%，"
                                f"取值大致在 {float(period_representative_bounds(stage_vals, int(local_lag))[0]):.1f} 到 {float(period_representative_bounds(stage_vals, int(local_lag))[1]):.1f} 之间"
                            )
                            stage_shape_extra = "段内以周期性起伏为主"
                            stage_form = "以周期性变化为主"
                        if "走势较稳" in stage_shape_extra and rel_vol_phrase in {"相对全窗口属于中等波动", "相对全窗口属于较大波动"}:
                            stage_shape_extra = "段内以起伏为主"
                        if "走势较稳" in stage_shape_extra and rel_vol_phrase in {"相对全窗口属于中等波动", "相对全窗口属于较大波动"}:
                            stage_shape_extra = "段内以起伏为主"
                        if "先抬升后回落" in stage_shape_extra or "先回落后抬升" in stage_shape_extra or "震荡" in stage_shape_extra:
                            stage_form = "存在明显起伏变化"
                        segment_lines.append(
                            f"第{i}阶段（{fmt_time_range(stage_seg['start'], stage_seg['end'])}）{stage_form}，均值约 {stage_mean:.1f}，起点约 {stage_start_val:.1f}，终点约 {stage_end_val:.1f}，且{vol_phrase(stage_vol)}；{stage_shape_extra}，取值大致在 {stage_lo:.1f} 到 {stage_hi:.1f} 之间，{rel_vol_phrase}{periodic_tail}。"
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
                baseline = float(target[prev_idx:s].mean()) if s > 0 else g_mean
            signed_delta = ev.get("signed_delta")
            if signed_delta is None:
                signed_delta = float(v) - float(baseline)
            signed_delta = float(signed_delta)
            change_word = "激增" if signed_delta >= 0 else "暴跌"
            compare_word = "偏高" if signed_delta >= 0 else "偏低"
            if int(s) == int(e):
                ev_pos = win_df.index[s].strftime("%Y-%m-%d %H:%M")
            else:
                ev_pos = fmt_time_range(s, e)
            event_desc.append(
                f"{ev_pos} 出现一次局部{change_word}（峰值约 {v:.1f}，相较局部基线 {baseline:.1f} 明显{compare_word}，标准化偏离 |z|≈{abs(z):.1f}）"
            )
        lines.append("异常点：\n" + "；\n".join(event_desc) + "。")
    elif no_change_flag:
        lines.append("异常点：\n本窗口序列保持恒定，未出现异常变化。")
    else:
        lines.append("异常点：\n本窗口未出现特别突出的尖峰或急跌。")

    if no_change_flag:
        lines.append(
            f"整体结论：\n这一窗口中的 {chinese_name} 保持不变，整体基本无波动。"
        )
    elif oscillation_flag and not plateau_local_flag:
        lines.append(
            f"整体结论：\n这一窗口中的 {chinese_name} 以高频震荡为主，短时起伏频繁，未形成持续单向趋势。"
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
            periodic_preferred_flag=periodic_grouping_flag,
            short_window=is_short_window,
        )
        lines.append(
            f"整体结论：\n这一窗口中的 {chinese_name} {summary_text}，且{vol_phrase(vol_level)}。"
        )

    description = "\n\n".join(lines)
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


def generate_electricity_ecl_jsonl(
    csv_path: str,
    output_path: str,
    target_col: str,
    window_lengths=(512, 1024),
    step_ratio: float = 0.5,
    max_samples: int | None = None,
):
    """
    主函数：生成ElectricityECL数据集的时序描述（JSONL格式）
    """
    df = load_electricity_ecl(csv_path, target_col)
    n = len(df)
    feature_names = list(df.columns)
    # 转换特征名称为中文
    feature_names_chinese = [f"电表{col.split('_')[1]}" for col in feature_names]
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_count = 0
    
    with out_path.open("w", encoding="utf-8") as f:
        for L in window_lengths:
            step = max(1, int(L * step_ratio))
            for start in sliding_window_indices(n, L, step):
                end = start + L
                win_df = df.iloc[start:end]
                desc, feat = describe_window_electricity_ecl(win_df, target_col)
                sample = {
                    "id": f"ElectricityECL_{target_col}_L{L}_start{start}",
                    "dataset": "ElectricityECL",
                    "task": "forecasting",
                    "window_length": L,
                    "start_index": int(start),
                    "end_index": int(end - 1),
                    "start_time": str(win_df.index[0]),
                    "end_time": str(win_df.index[-1]),
                    "time": [str(t) for t in win_df.index],
                    "feature_names": feature_names,
                    "feature_names_chinese": feature_names_chinese,  # 添加中文特征名
                    "target_col": target_col,
                    "target_col_chinese": f"电表{target_col.split('_')[1]}",  # 添加中文目标列名
                    "values": win_df.to_numpy().tolist(),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成ElectricityECL数据集的中文文本描述（JSONL格式）。")
    parser.add_argument("--csv_path", type=str, required=True, help="输入文件路径（支持.csv/.xlsx/.xls/.txt/.npy）")
    parser.add_argument("--output_path", type=str, default=None, help="输出jsonl路径（默认：脚本目录下 ElectricityECL_{target_col}_descriptions.jsonl）")
    parser.add_argument(
        "--target_col",
        type=str,
        default="MT_001",
        help="目标电表列（默认MT_001，可选MT_001~MT_370）"
    )
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
    args = parser.parse_args()
    
    try:
        # 自动转换为CSV（保留原有逻辑）
        converted_csv_path = convert_to_csv(args.csv_path)
        args.csv_path = converted_csv_path
    
    except Exception as e:
        print(f"❌ 文件转换失败：{e}")
        exit(1)
    # 如果未指定 --output_path，则默认放在脚本同目录，文件名根据目标列自动命名
    if args.output_path is None:
        script_dir = Path(__file__).resolve().parent
        win_tag = "-".join(str(w) for w in args.window_lengths)
        default_name = f"ElectricityECL_{args.target_col}_L{win_tag}_descriptions.jsonl"
        args.output_path = str(script_dir / default_name)
        print(f"⚙️ 未指定 --output_path，使用默认输出：{args.output_path}")
    
    # 生成ElectricityECL描述
    generate_electricity_ecl_jsonl(
        csv_path=args.csv_path,
        output_path=args.output_path,
        target_col=args.target_col,
        window_lengths=args.window_lengths,
        step_ratio=args.step_ratio,
        max_samples=args.max_samples
    )










