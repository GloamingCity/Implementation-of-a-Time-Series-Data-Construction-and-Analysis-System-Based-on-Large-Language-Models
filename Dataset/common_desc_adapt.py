import numpy as np


def smooth_signal_for_window(x: np.ndarray, short_window: bool = False) -> np.ndarray:
    """Apply light smoothing for short windows to reduce local-noise sensitivity."""
    arr = np.asarray(x, dtype=float)
    n = int(arr.size)
    if n < 5:
        return arr.copy()
    if short_window:
        win = max(3, min(9, n // 18))
    else:
        win = max(3, min(7, n // 36))
    if win % 2 == 0:
        win += 1
    if win <= 1:
        return arr.copy()
    kernel = np.ones(win, dtype=float) / float(win)
    pad = win // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _robust_linear_fit(arr: np.ndarray) -> tuple[float, float]:
    values = np.asarray(arr, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n < 2:
        return 0.0, float(values[0]) if n == 1 else 0.0

    t = np.arange(n, dtype=float)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    sigma = max(1.4826 * mad, 1e-6)

    inlier = np.abs(values - med) <= (3.5 * sigma)
    min_inliers = max(6, int(0.45 * n))
    if int(np.sum(inlier)) < min_inliers:
        inlier = np.ones(n, dtype=bool)

    t_fit = t[inlier]
    y_fit = values[inlier]
    if t_fit.size < 2:
        return 0.0, float(np.median(values))

    try:
        slope, intercept = np.polyfit(t_fit, y_fit, 1)
    except Exception:
        slope = 0.0
        intercept = float(np.median(values))
    return float(slope), float(intercept)


def _interp_nan_1d(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=float).copy()
    n = int(out.size)
    if n <= 1:
        return out
    mask = np.isfinite(out)
    if int(np.sum(mask)) == 0:
        return np.zeros_like(out)
    if int(np.sum(mask)) == n:
        return out
    x = np.arange(n, dtype=float)
    out[~mask] = np.interp(x[~mask], x[mask], out[mask])
    return out


def _pulse_mask_from_events(n: int, events: list[dict], pad: int = 1) -> np.ndarray:
    mask = np.zeros(int(max(0, n)), dtype=bool)
    for ev in events or []:
        s = int(ev.get("start", 0))
        e = int(ev.get("end", s))
        p = int(ev.get("peak_idx", s))
        s = max(0, s - int(pad))
        e = min(int(n) - 1, e + int(pad))
        if s <= e:
            mask[s : e + 1] = True
        if 0 <= p < int(n):
            lo = max(0, p - int(pad))
            hi = min(int(n) - 1, p + int(pad))
            mask[lo : hi + 1] = True
    return mask


def _periodicity_signals(arr: np.ndarray) -> tuple[bool, bool, float, float]:
    vals = np.asarray(arr, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = int(vals.size)
    if n < 24:
        return False, False, 0.0, 0.0
    if float(np.std(vals)) < 1e-8:
        return False, False, 0.0, 0.0

    centered = vals - float(np.mean(vals))

    max_lag = min(max(8, n // 4), 96)
    acf_full = np.correlate(centered, centered, mode="full")
    acf = acf_full[n - 1 : n + max_lag]
    best_acf_abs = 0.0
    acf1_abs = 0.0
    if acf.size > 2 and acf[0] > 0:
        acf = acf / acf[0]
        acf1_abs = abs(float(acf[1])) if acf.size > 1 else 0.0
        best_acf_abs = max((abs(float(acf[i])) for i in range(2, acf.size)), default=0.0)

    diff = np.diff(vals)
    if diff.size:
        sign = np.sign(diff)
        nz = sign[sign != 0]
        switch_ratio = (float(np.sum(nz[1:] != nz[:-1])) / max(1, nz.size - 1)) if nz.size > 1 else 0.0
    else:
        switch_ratio = 0.0
    net_change = abs(float(vals[-1] - vals[0])) / max(float(np.ptp(vals)), 1e-6)

    win = np.hanning(n) if n >= 16 else np.ones(n, dtype=float)
    spec = np.fft.rfft(centered * win)
    power = np.abs(spec) ** 2
    fft_peak_ratio = 0.0
    if power.size > 1:
        power = power.astype(float)
        power[0] = 0.0
        total = float(np.sum(power))
        if total > 0:
            peak_idx = int(np.argmax(power))
            fft_peak_ratio = float(power[peak_idx] / total)

    strong_periodic = bool((best_acf_abs >= 0.78 and fft_peak_ratio >= 0.18) or fft_peak_ratio >= 0.34)
    stationary_osc = bool(best_acf_abs >= 0.48 and acf1_abs <= 0.55 and switch_ratio >= 0.42 and net_change <= 0.18)
    return strong_periodic, stationary_osc, float(best_acf_abs), float(fft_peak_ratio)


def _detect_major_turn_shape(arr: np.ndarray) -> str | None:
    vals = np.asarray(arr, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = int(vals.size)
    if n < 12:
        return None

    span = max(float(np.ptp(vals)), 1e-6)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sigma = max(1.4826 * mad, 1e-6)
    mid_lo = int(0.12 * n)
    mid_hi = int(0.88 * n)

    v_idx = int(np.argmin(vals))
    if mid_lo <= v_idx <= mid_hi:
        left = vals[: v_idx + 1]
        right = vals[v_idx:]
        drop = max(float(left.max() - vals[v_idx]), float(vals[0] - vals[v_idx]))
        rebound = max(float(vals[-1] - vals[v_idx]), float(right.max() - vals[v_idx]))
        left_slope = float(np.median(np.diff(left))) if left.size >= 4 else 0.0
        right_slope = float(np.median(np.diff(right))) if right.size >= 4 else 0.0
        if (
            drop >= max(2.2 * sigma, 0.22 * span)
            and rebound >= max(2.2 * sigma, 0.22 * span)
            and left_slope < 0
            and right_slope > 0
        ):
            return "v_shape"

    iv_idx = int(np.argmax(vals))
    if mid_lo <= iv_idx <= mid_hi:
        left = vals[: iv_idx + 1]
        right = vals[iv_idx:]
        rise = max(float(vals[iv_idx] - left.min()), float(vals[iv_idx] - vals[0]))
        fallback = max(float(vals[iv_idx] - vals[-1]), float(vals[iv_idx] - right.min()))
        left_slope = float(np.median(np.diff(left))) if left.size >= 4 else 0.0
        right_slope = float(np.median(np.diff(right))) if right.size >= 4 else 0.0
        if (
            rise >= max(2.2 * sigma, 0.22 * span)
            and fallback >= max(2.2 * sigma, 0.22 * span)
            and left_slope > 0
            and right_slope < 0
        ):
            return "inv_v_shape"

    return None


def robust_classify_global_trend(x: np.ndarray) -> tuple[str, float]:
    """Classify trend with outlier-robust linear fit to avoid endpoint pulse hijacking."""
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n < 3:
        return "flat", 0.0
    if np.allclose(arr, arr[0]):
        return "flat", 0.0

    short_window = n < 200

    # Remove isolated pulse outliers before global trend estimation.
    pulse_events = detect_local_extreme_events(arr, z_thr=3.0)
    pulse_mask = _pulse_mask_from_events(n=n, events=pulse_events, pad=1)
    core = arr.copy()
    if np.any(pulse_mask) and int(np.sum(~pulse_mask)) >= 3:
        core[pulse_mask] = np.nan
        core = _interp_nan_1d(core)

    smoothed = smooth_signal_for_window(core, short_window=short_window)

    # Avoid end-point bias in deep V / inverted-V windows.
    turn_shape = _detect_major_turn_shape(smoothed)
    if turn_shape in {"v_shape", "inv_v_shape"}:
        return "flat", 0.0

    slope, _ = _robust_linear_fit(smoothed)

    q10, q90 = np.quantile(core, [0.10, 0.90])
    denom = max(float(q90 - q10), float(np.quantile(np.abs(arr), 0.90)), 1.0)
    slope_norm = float((slope * n) / max(denom, 1e-6))

    if slope_norm >= 0.28:
        label = "strong_up"
    elif slope_norm >= 0.09:
        label = "weak_up"
    elif slope_norm <= -0.28:
        label = "strong_down"
    elif slope_norm <= -0.09:
        label = "weak_down"
    else:
        label = "flat"
    return label, float(slope_norm)


def classify_volatility_adaptive(
    x: np.ndarray,
    short_window: bool = False,
    long_window: bool = False,
    eps: float = 1e-6,
) -> tuple[str, float, dict]:
    """Estimate volatility from detrended residuals and first-difference shocks."""
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 6:
        return "low", 0.0, {"rel_mad": 0.0, "rel_iqr": 0.0, "diff_energy": 0.0, "shock_ratio": 0.0}

    n = int(arr.size)
    t = np.arange(n, dtype=float)
    slope, intercept = _robust_linear_fit(arr)
    trend = slope * t + intercept

    resid = arr - trend
    resid_centered = resid - float(np.median(resid))

    mad = float(np.median(np.abs(resid_centered)))
    robust_sigma = max(1.4826 * mad, eps)

    base_scale = max(abs(float(np.median(arr))), float(np.quantile(np.abs(arr), 0.90)), 1.0)
    rel_mad = robust_sigma / base_scale

    q15, q85 = np.quantile(resid, [0.15, 0.85])
    rel_iqr = float(q85 - q15) / max(base_scale, eps)

    diff = np.diff(resid, prepend=resid[0])
    if diff.size:
        diff_centered = diff - float(np.median(diff))
        diff_mad = float(np.median(np.abs(diff_centered)))
        diff_energy = (1.4826 * diff_mad) / max(base_scale, eps)
        shock_ratio = float(np.quantile(np.abs(diff_centered), 0.90)) / max(robust_sigma, eps)
    else:
        diff_energy = 0.0
        shock_ratio = 0.0

    # Residual and first-difference features are used to decouple trend amplitude and true volatility.
    score = 0.40 * rel_mad + 0.25 * rel_iqr + 0.25 * diff_energy + 0.10 * shock_ratio

    if short_window:
        low_thr, high_thr = 0.045, 0.125
    elif long_window:
        low_thr, high_thr = 0.035, 0.095
    else:
        low_thr, high_thr = 0.040, 0.108

    if score < low_thr:
        level = "low"
    elif score < high_thr:
        level = "medium"
    else:
        level = "high"

    return level, float(score), {
        "rel_mad": float(rel_mad),
        "rel_iqr": float(rel_iqr),
        "diff_energy": float(diff_energy),
        "shock_ratio": float(shock_ratio),
        "trend_slope": float(slope),
    }


def detect_change_boundaries(
    x: np.ndarray,
    short_window: bool = False,
    long_window: bool = False,
) -> list[int]:
    """Detect boundaries via penalized change-point search on piecewise linear fit errors."""
    arr = np.asarray(x, dtype=float)
    n = int(arr.size)
    if n < 24:
        return [0, n]

    # 先剔除孤立脉冲，再评估是否需要分段，避免“脉冲绑架阶段切分”。
    pulse_events = detect_local_extreme_events(arr, z_thr=3.0)
    pulse_mask = _pulse_mask_from_events(n=n, events=pulse_events, pad=1)
    core = arr.copy()
    if np.any(pulse_mask) and int(np.sum(~pulse_mask)) >= 3:
        core[pulse_mask] = np.nan
        core = _interp_nan_1d(core)

    core_centered = core - float(np.median(core))
    core_scale = max(float(np.quantile(np.abs(core_centered), 0.90)), 1e-6)
    core_std_ratio = float(np.std(core_centered)) / core_scale

    # 基线恒定+孤立脉冲：强制单段输出。
    if pulse_events and core_std_ratio <= 0.02:
        return [0, n]

    # 强周期或全局平稳震荡：强制单段输出，禁止碎片化切分。
    strong_periodic, stationary_osc, _, _ = _periodicity_signals(core)
    if strong_periodic or stationary_osc:
        return [0, n]

    # 短序列防过拟合：L<100时优先单段，除非存在极强阶段证据。
    if n < 100:
        diff = np.diff(core)
        if diff.size:
            diff_center = diff - float(np.median(diff))
            diff_mad = max(1.4826 * float(np.median(np.abs(diff_center))), 1e-6)
            jump_ratio = float(np.max(np.abs(diff_center))) / diff_mad
        else:
            jump_ratio = 0.0
        if jump_ratio < 7.0:
            return [0, n]

    smooth = smooth_signal_for_window(core, short_window=short_window)
    if np.allclose(smooth, smooth[0]):
        return [0, n]

    if n < 100:
        min_seg = max(32, n // 2)
        max_breaks = 1
        lambda_coef = 3.20
    elif short_window:
        min_seg = max(18, n // 6)
        max_breaks = 2
        lambda_coef = 2.20
    elif long_window:
        min_seg = max(24, n // 12)
        max_breaks = 6
        lambda_coef = 1.40
    else:
        min_seg = max(20, n // 9)
        max_breaks = 4
        lambda_coef = 1.80

    t = np.arange(n, dtype=float)
    y = np.asarray(smooth, dtype=float)
    prefix_t = np.concatenate([[0.0], np.cumsum(t)])
    prefix_t2 = np.concatenate([[0.0], np.cumsum(t * t)])
    prefix_y = np.concatenate([[0.0], np.cumsum(y)])
    prefix_y2 = np.concatenate([[0.0], np.cumsum(y * y)])
    prefix_ty = np.concatenate([[0.0], np.cumsum(t * y)])

    def _segment_sse(start: int, end: int) -> float:
        # Segment is [start, end), 1-based prefix arrays.
        m = int(end - start)
        if m <= 2:
            return float("inf")

        sum_t = float(prefix_t[end] - prefix_t[start])
        sum_t2 = float(prefix_t2[end] - prefix_t2[start])
        sum_y = float(prefix_y[end] - prefix_y[start])
        sum_y2 = float(prefix_y2[end] - prefix_y2[start])
        sum_ty = float(prefix_ty[end] - prefix_ty[start])

        denom = m * sum_t2 - sum_t * sum_t
        if abs(denom) <= 1e-12:
            slope = 0.0
            intercept = sum_y / max(m, 1)
        else:
            slope = (m * sum_ty - sum_t * sum_y) / denom
            intercept = (sum_y - slope * sum_t) / m

        sse = (
            sum_y2
            - 2.0 * slope * sum_ty
            - 2.0 * intercept * sum_y
            + slope * slope * sum_t2
            + 2.0 * slope * intercept * sum_t
            + m * intercept * intercept
        )
        return float(max(sse, 0.0))

    signal_var = max(float(np.var(y)), 1e-8)
    penalty = float(lambda_coef * signal_var * np.log(max(8, n)))

    # Penalized dynamic programming (change-point style), practical for n<=~1k.
    dp = [float("inf")] * (n + 1)
    prev = [-1] * (n + 1)
    dp[0] = -penalty

    for end in range(min_seg, n + 1):
        best_cost = float("inf")
        best_start = -1
        max_start = end - min_seg
        for start in range(0, max_start + 1):
            if start != 0 and prev[start] < 0:
                continue
            if end - start < min_seg:
                continue
            seg_cost = _segment_sse(start, end)
            if not np.isfinite(seg_cost):
                continue
            total = dp[start] + seg_cost + penalty
            if total < best_cost:
                best_cost = total
                best_start = start

        dp[end] = best_cost
        prev[end] = best_start

    if prev[n] < 0:
        return [0, n]

    cp: list[int] = []
    cur = n
    while cur > 0:
        start = prev[cur]
        if start <= 0:
            break
        cp.append(int(start))
        cur = int(start)

    cp = sorted(set(cp))
    if not cp:
        return [0, n]

    if len(cp) > max_breaks:
        marks = [0] + cp + [n]
        gains: list[tuple[int, float]] = []
        for i, split in enumerate(cp, start=1):
            left = marks[i - 1]
            right = marks[i + 1]
            gain = _segment_sse(left, right) - (_segment_sse(left, split) + _segment_sse(split, right))
            gains.append((int(split), float(gain)))
        gains.sort(key=lambda item: item[1], reverse=True)
        cp = sorted(split for split, _ in gains[:max_breaks])

    return [0] + cp + [n]


def detect_local_extreme_events(
    x: np.ndarray,
    z_thr: float = 2.5,
    local_window: int | None = None,
) -> list[dict]:
    """Detect spike/crash events with local baseline and slope-jump gating."""
    arr = np.asarray(x, dtype=float)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    n = int(arr.size)
    if n < 8:
        return []

    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return []
    fill_value = float(np.nanmedian(arr[finite_mask]))
    arr = np.where(finite_mask, arr, fill_value)

    if local_window is None:
        local_window = max(9, min(41, (n // 12) * 2 + 1))
    local_window = int(max(5, local_window))
    if local_window % 2 == 0:
        local_window += 1

    kernel = np.ones(local_window, dtype=float) / float(local_window)
    pad = local_window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    local_mean = np.convolve(padded, kernel, mode="valid")

    residual = arr - local_mean
    resid_centered = residual - float(np.median(residual))
    mad = float(np.median(np.abs(resid_centered)))
    robust_sigma = max(1.4826 * mad, 1e-6)

    # 局部波动越大，阈值越高，避免把连续波浪峰谷误判为异常。
    abs_resid = np.abs(resid_centered)
    local_abs = np.convolve(np.pad(abs_resid, (pad, pad), mode="edge"), kernel, mode="valid")
    local_sigma = np.maximum(1.2533 * local_abs, robust_sigma * 0.65)
    local_sigma = np.maximum(local_sigma, 1e-6)

    local_vol_ratio = local_sigma / max(float(np.median(local_sigma)), 1e-6)
    dynamic_thr = float(z_thr) * np.clip(1.00 + 0.95 * np.maximum(0.0, local_vol_ratio - 1.0), 1.00, 3.20)

    strong_periodic, stationary_osc, _, fft_peak_ratio = _periodicity_signals(arr)
    if strong_periodic or stationary_osc:
        boost = 1.12 + 0.35 * max(0.0, fft_peak_ratio - 0.18)
        dynamic_thr = dynamic_thr * min(1.85, boost)
    z = residual / local_sigma

    diff = np.diff(arr, prepend=arr[0])
    local_diff_mean = np.convolve(np.pad(diff, (pad, pad), mode="edge"), kernel, mode="valid")
    slope_shock = diff - local_diff_mean
    slope_centered = slope_shock - float(np.median(slope_shock))
    slope_mad = float(np.median(np.abs(slope_centered)))
    slope_sigma = max(1.4826 * slope_mad, 1e-6)

    curvature = np.diff(diff, prepend=diff[0])
    curv_centered = curvature - float(np.median(curvature))
    curv_mad = float(np.median(np.abs(curv_centered)))
    curv_sigma = max(1.4826 * curv_mad, 1e-6)

    idxs = np.where(np.abs(z) >= dynamic_thr)[0]
    if idxs.size == 0:
        return []

    groups: list[tuple[int, int]] = []
    start = int(idxs[0])
    prev_idx = int(idxs[0])
    for idx in idxs[1:]:
        idx = int(idx)
        if idx == prev_idx + 1:
            prev_idx = idx
            continue
        groups.append((start, prev_idx))
        start = idx
        prev_idx = idx
    groups.append((start, prev_idx))

    events: list[dict] = []
    for s, e in groups:
        seg_z = z[s : e + 1]
        if seg_z.size <= 0:
            continue
        peak_rel = int(np.argmax(np.abs(seg_z)))
        peak_idx = int(s + peak_rel)
        peak_abs_z = abs(float(z[peak_idx]))
        slope_z = abs(float(slope_shock[peak_idx])) / slope_sigma
        curv_slice = curvature[max(0, peak_idx - 1) : min(n, peak_idx + 2)]
        curvature_z = (float(np.max(np.abs(curv_slice))) / curv_sigma) if curv_slice.size else 0.0

        # Reject smooth accelerated tails: need clear local deviation plus slope/curvature shock.
        local_thr = float(dynamic_thr[peak_idx]) if peak_idx < dynamic_thr.size else float(z_thr)
        if peak_abs_z < (local_thr + 0.45) and slope_z < 1.25 and curvature_z < 1.45:
            continue
        if (int(e) - int(s) + 1) >= max(5, local_window // 2) and slope_z < 1.60 and curvature_z < 1.80:
            continue

        # In strongly periodic windows, keep only sharp impulses, not regular wave peaks.
        if (strong_periodic or stationary_osc) and (slope_z < 1.85 or curvature_z < 2.10):
            continue

        signed_delta = float(arr[peak_idx] - local_mean[peak_idx])
        events.append(
            {
                "start": int(s),
                "end": int(e),
                "peak_idx": int(peak_idx),
                "peak_value": float(arr[peak_idx]),
                "z_max": float(z[peak_idx]),
                "local_baseline": float(local_mean[peak_idx]),
                "signed_delta": float(signed_delta),
                "slope_z": float(slope_z),
                "curvature_z": float(curvature_z),
                "dynamic_threshold": float(local_thr),
                "direction": "spike" if signed_delta >= 0 else "crash",
            }
        )

    events.sort(
        key=lambda d: abs(float(d.get("z_max", 0.0))) + 0.25 * float(d.get("curvature_z", 0.0)),
        reverse=True,
    )
    return events


def segment_features_by_boundaries(
    x: np.ndarray,
    boundaries: list[int],
    global_std: float,
    trend_func,
) -> list[dict]:
    """Build segment features from boundaries and merge weakly-differentiated adjacent phases."""
    arr = np.asarray(x, dtype=float)
    n = int(arr.size)
    if n < 2:
        return []
    if not boundaries or boundaries[0] != 0 or boundaries[-1] != n:
        boundaries = [0, n]

    def _lag1_autocorr(vals: np.ndarray) -> float:
        v = np.asarray(vals, dtype=float)
        if v.size < 4:
            return 0.0
        v = v - float(np.mean(v))
        if float(np.std(v)) < 1e-8:
            return 0.0
        rho = float(np.corrcoef(v[:-1], v[1:])[0, 1])
        return rho if np.isfinite(rho) else 0.0

    def _make_segment(start: int, end_exclusive: int, idx: int) -> dict | None:
        if end_exclusive - start < 2:
            return None
        seg = arr[start:end_exclusive]
        seg_mean = float(np.mean(seg))
        seg_std = float(np.std(seg))
        if global_std <= 0:
            vol_level = "low"
        else:
            ratio = seg_std / max(global_std, 1e-8)
            if ratio < 0.7:
                vol_level = "low"
            elif ratio < 1.3:
                vol_level = "medium"
            else:
                vol_level = "high"
        seg_trend_label, _ = trend_func(seg)
        return {
            "idx": int(idx),
            "start": int(start),
            "end": int(end_exclusive - 1),
            "len": int(end_exclusive - start),
            "mean": seg_mean,
            "std": seg_std,
            "vol_level": vol_level,
            "trend_label": seg_trend_label,
        }

    def _segment_similarity(left: dict, right: dict, scale: float) -> bool:
        mean_delta = abs(float(left["mean"]) - float(right["mean"])) / max(scale, 1e-8)
        l_std = max(float(left["std"]), 1e-8)
        r_std = max(float(right["std"]), 1e-8)
        std_ratio = max(l_std, r_std) / max(min(l_std, r_std), 1e-8)

        left_vals = arr[int(left["start"]) : int(left["end"]) + 1]
        right_vals = arr[int(right["start"]) : int(right["end"]) + 1]
        acf_delta = abs(_lag1_autocorr(left_vals) - _lag1_autocorr(right_vals))

        return bool(mean_delta <= 0.32 and std_ratio <= 1.35 and acf_delta <= 0.22)

    segments = []
    for k in range(len(boundaries) - 1):
        start = int(boundaries[k])
        end_exclusive = int(boundaries[k + 1])
        built = _make_segment(start, end_exclusive, len(segments))
        if built is not None:
            segments.append(built)

    if len(segments) <= 1:
        return segments

    # Global periodic/noise consistency guard: avoid unnecessary phase fragmentation.
    centered = arr - float(np.mean(arr))
    periodic_like = False
    noise_like = False
    acf1_abs = 0.0
    best_acf_abs = 0.0
    if n >= 32 and float(np.std(centered)) > 1e-8:
        max_lag = min(max(8, n // 4), 64)
        acf_full = np.correlate(centered, centered, mode="full")
        acf = acf_full[n - 1 : n + max_lag]
        if acf.size > 2 and acf[0] > 0:
            acf = acf / acf[0]
            acf1_abs = abs(float(acf[1])) if acf.size > 1 else 0.0
            best_acf_abs = max((abs(float(acf[i])) for i in range(2, acf.size)), default=0.0)
            periodic_like = best_acf_abs >= 0.72
            noise_like = best_acf_abs <= 0.26 and acf1_abs <= 0.15

    scale = max(float(global_std), float(np.std(arr)), 1e-8)
    mean_jumps = [
        abs(float(segments[i + 1]["mean"]) - float(segments[i]["mean"])) / max(scale, 1e-8)
        for i in range(len(segments) - 1)
    ]
    std_ratios = []
    for i in range(len(segments) - 1):
        a = max(float(segments[i]["std"]), 1e-8)
        b = max(float(segments[i + 1]["std"]), 1e-8)
        std_ratios.append(max(a, b) / max(min(a, b), 1e-8))

    if periodic_like and np.median(mean_jumps) <= 0.24 and np.median(std_ratios) <= 1.45:
        merged = _make_segment(0, n, 0)
        return [merged] if merged is not None else []
    if noise_like and np.median(mean_jumps) <= 0.20 and np.median(std_ratios) <= 1.30:
        merged = _make_segment(0, n, 0)
        return [merged] if merged is not None else []

    # Iterative adjacent merge with implicit penalty on statistically weak split evidence.
    for _ in range(4):
        if len(segments) <= 1:
            break
        changed = False
        merged_segments: list[dict] = []
        i = 0
        while i < len(segments):
            if i < len(segments) - 1 and _segment_similarity(segments[i], segments[i + 1], scale=scale):
                start = int(segments[i]["start"])
                end_exclusive = int(segments[i + 1]["end"]) + 1
                built = _make_segment(start, end_exclusive, len(merged_segments))
                if built is not None:
                    merged_segments.append(built)
                changed = True
                i += 2
                continue

            keep = dict(segments[i])
            keep["idx"] = int(len(merged_segments))
            merged_segments.append(keep)
            i += 1

        segments = merged_segments
        if not changed:
            break

    for i, seg in enumerate(segments):
        seg["idx"] = int(i)
    return segments


def enforce_summary_consistency(
    summary_text: str,
    segments: list[dict],
    target: np.ndarray,
    slope_norm: float,
    events: list[dict],
    periodic_preferred_flag: bool = False,
    short_window: bool = False,
) -> str:
    """Downshift over-strong summary wording when local evidence is weak/conflicting."""
    text = str(summary_text)
    if not segments:
        return text

    dirs = [s.get("trend_label", "flat") for s in segments]
    up_cnt = sum(d in ("strong_up", "weak_up") for d in dirs)
    down_cnt = sum(d in ("strong_down", "weak_down") for d in dirs)
    high_vol_cnt = sum(s.get("vol_level") == "high" for s in segments)
    max_event_z = max((abs(float(ev.get("z_max", 0.0))) for ev in events), default=0.0)

    arr = np.asarray(target, dtype=float)
    span = max(float(np.max(arr) - np.min(arr)), 1e-8)
    net_change = (float(arr[-1]) - float(arr[0])) / span if arr.size >= 2 else 0.0

    # 专属模板：基线恒定 + 孤立脉冲。
    pulse_events = [ev for ev in events if abs(float(ev.get("z_max", 0.0))) >= 2.6]
    if arr.size >= 8 and pulse_events:
        pulse_mask = _pulse_mask_from_events(n=int(arr.size), events=pulse_events, pad=1)
        core = arr.copy()
        if np.any(pulse_mask) and int(np.sum(~pulse_mask)) >= 3:
            core[pulse_mask] = np.nan
            core = _interp_nan_1d(core)
        core_centered = core - float(np.median(core))
        core_scale = max(float(np.quantile(np.abs(core_centered), 0.90)), 1e-6)
        core_std_ratio = float(np.std(core_centered)) / core_scale
        if core_std_ratio <= 0.02:
            return "基线整体恒定，仅出现孤立脉冲"

    # 明显深V/倒V优先输出拐点模板，避免端点偏差误判为单调趋势。
    turn_shape = _detect_major_turn_shape(arr)
    if turn_shape == "v_shape":
        return "先降后升（V型）"
    if turn_shape == "inv_v_shape":
        return "先升后降（倒V型）"

    if ("上升" in text or "抬升" in text) and (up_cnt <= down_cnt or net_change < 0.08):
        text = "呈阶段性起伏"
    if ("下降" in text or "回落" in text) and (down_cnt <= up_cnt or net_change > -0.08):
        text = "呈阶段性起伏"
    if "平稳" in text and (high_vol_cnt >= 2 or len(events) >= 1):
        text = "呈阶段性起伏"
    if "周期" in text and (not periodic_preferred_flag):
        text = "呈阶段性起伏"
    if periodic_preferred_flag and ("周期" not in text):
        text = "呈周期性起伏"
    if max_event_z >= 3.0 and ("平稳" in text or "阶段性起伏" in text):
        text = "呈脉冲式波动"
    if short_window and "明显" in text and abs(float(slope_norm)) < 0.12:
        text = text.replace("明显", "轻微")
    return text


def build_joint_linkage_clause(
    main_var: str,
    corr_map: dict[str, float | None],
    stable_map: dict[str, bool] | None = None,
    strong_thr: float = 0.8,
    display_name_map: dict[str, str] | None = None,
    max_vars: int = 2,
    stage_label: str = "主要阶段",
) -> str | None:
    """Build a multivariate joint-stage clause when strong stable linkage exists."""
    if not isinstance(corr_map, dict) or not corr_map:
        return None

    name_map = display_name_map if isinstance(display_name_map, dict) else {}
    strong_items: list[tuple[str, float]] = []
    for var, rho in corr_map.items():
        try:
            val = float(rho)
        except Exception:
            continue
        if not np.isfinite(val):
            continue
        if stable_map is not None and not bool(stable_map.get(var, False)):
            continue
        if abs(val) >= float(strong_thr):
            strong_items.append((str(var), float(val)))

    if not strong_items:
        return None

    strong_items.sort(key=lambda item: abs(item[1]), reverse=True)
    picked = strong_items[: max(1, int(max_vars))]
    parts = []
    for var, rho in picked:
        vname = str(name_map.get(var, var))
        rel = "同向" if rho >= 0 else "反向"
        parts.append(f"{vname}与{main_var}呈高度{rel}协同变化（rho≈{rho:.2f}）")

    return f"多变量协同变化：在{stage_label}，" + "；".join(parts) + "。"
