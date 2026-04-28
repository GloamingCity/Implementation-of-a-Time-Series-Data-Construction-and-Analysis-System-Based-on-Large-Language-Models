import torch


DETAIL_LEVELS = ("brief", "standard", "detailed")


def compute_series_complexity(ts_tensor: torch.Tensor):
    ts = ts_tensor.detach().float().flatten()
    n = int(ts.numel())
    if n <= 0:
        return {
            "detail_level": "standard",
            "complexity_score": 0.5,
            "seq_len": 0,
            "min_value": 0.0,
            "max_value": 0.0,
            "mean_abs_diff": 0.0,
            "turning_rate": 0.0,
            "spike_ratio": 0.0,
            "recommended_segments": "2-3",
        }

    min_value = float(torch.min(ts).item())
    max_value = float(torch.max(ts).item())

    if n == 1:
        mean_abs_diff = 0.0
        turning_rate = 0.0
        spike_ratio = 0.0
    else:
        dx = ts[1:] - ts[:-1]
        abs_dx = torch.abs(dx)
        mean_abs_diff = float(abs_dx.mean().item()) if abs_dx.numel() > 0 else 0.0

        if dx.numel() >= 2:
            turning = (dx[1:] * dx[:-1] < 0).float()
            turning_rate = float(turning.mean().item())
        else:
            turning_rate = 0.0

        local_std = float(abs_dx.std(unbiased=False).item()) if abs_dx.numel() > 1 else 0.0
        spike_threshold = mean_abs_diff + 2.5 * local_std + 1e-8
        spike_ratio = (
            float((abs_dx > spike_threshold).float().mean().item())
            if abs_dx.numel() > 0
            else 0.0
        )

    std = float(ts.std(unbiased=False).item()) if n > 1 else 0.0

    norm_turning = min(1.0, turning_rate * 3.0)
    norm_spike = min(1.0, spike_ratio * 12.0)
    norm_volatility = min(1.0, mean_abs_diff / (std + 1e-6) / 2.0)

    complexity_score = 0.5 * norm_turning + 0.3 * norm_spike + 0.2 * norm_volatility
    if n < 64:
        complexity_score *= 0.85

    if complexity_score < 0.33:
        detail_level = "brief"
        recommended_segments = "1-2"
    elif complexity_score < 0.66:
        detail_level = "standard"
        recommended_segments = "2-4"
    else:
        detail_level = "detailed"
        recommended_segments = "4-6"

    if n >= 192 and detail_level == "brief":
        detail_level = "standard"
        recommended_segments = "2-4"

    if n >= 320 and detail_level == "standard" and complexity_score >= 0.45:
        detail_level = "detailed"
        recommended_segments = "4-6"

    return {
        "detail_level": detail_level,
        "complexity_score": float(complexity_score),
        "seq_len": n,
        "min_value": min_value,
        "max_value": max_value,
        "mean_abs_diff": mean_abs_diff,
        "turning_rate": turning_rate,
        "spike_ratio": spike_ratio,
        "recommended_segments": recommended_segments,
    }


def recommended_min_new_tokens(detail_level: str):
    level = detail_level if detail_level in DETAIL_LEVELS else "standard"
    if level == "brief":
        return 48
    if level == "detailed":
        return 160
    return 96


def _detail_level_text(detail_level: str):
    if detail_level == "brief":
        return "简洁"
    if detail_level == "detailed":
        return "详细"
    return "标准"


def build_adaptive_prompt_messages(ts_tensor: torch.Tensor, force_detail_level: str | None = None):
    info = compute_series_complexity(ts_tensor)

    if force_detail_level in DETAIL_LEVELS:
        info["detail_level"] = force_detail_level
        if force_detail_level == "brief":
            info["recommended_segments"] = "1-2"
        elif force_detail_level == "detailed":
            info["recommended_segments"] = "4-6"
        else:
            info["recommended_segments"] = "2-4"

    detail_text = _detail_level_text(info["detail_level"])

    system_prompt = "你是一个严谨的时间序列分析助手，擅长按序列复杂度自适应输出长度。"
    user_prompt = (
        "请生成与描述脚本风格一致的中文时间序列说明，输出应按复杂度自动控制详略。\n"
        "请尽量使用以下结构化小标题：\n"
        "窗口概览：\n分段观察：\n异常点：\n联动关系：\n整体结论：\n\n"
        "要求：\n"
        "1. 复杂度低时可简短，复杂度高时需展开分段细节。\n"
        "2. 若无明显异常点，请明确写"无明显异常点"。\n"
        "3. 若无明显联动关系，请明确写"无明显联动关系"。\n"
        "4. 描述尽量客观、定量、自然。\n\n"
        f"当前复杂度等级：{detail_text}。\n"
        f"建议分段数：{info['recommended_segments']}。\n"
        f"辅助统计：序列长度约 {info['seq_len']}，取值范围约 [{info['min_value']:.4g}, {info['max_value']:.4g}]，"
        f"平均波动强度约 {info['mean_abs_diff']:.4g}，拐点率约 {info['turning_rate']:.2%}。"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages, info
