# 推理启动命令（项目根目录执行，Linux 服务器可直接运行）
#
# 1) 自动选空闲卡并前台推理（默认仅输出多模态结果；prompt 不使用 caption）
#    IDLE_GPU=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', *' '$2<1000 && $3<10 {print $1; exit}'); [ -z "$IDLE_GPU" ] && echo "无空闲GPU" && exit 1; CUDA_VISIBLE_DEVICES=$IDLE_GPU python -u Train/infer.py --checkpoint Train/Checkpoints/Frozen_LLM_Stage/cnn_epoch_3/custom_ts_weights.pth --input_jsonl Sample/run_300k_20260413/combined_jsonl.jsonl --sample_index 123 --encoder_type cnn --model_path Models/Qwen3-4B-Instruct-2507 --precision bf16 --max_new_tokens 320 --min_new_tokens 96 --force_detail_level standard --show_base_text false --repetition_penalty 1.08 --no_repeat_ngram_size 6 --retry_on_incomplete true --retry_max_new_tokens 896 > Train/logs/infer_cnn_e3_simple.log 2>&1
#
# 2) 若希望同时打印纯文本基座输出（用于诊断），加上 --show_base_text true
#
# 3) 若你只知道 source_id，不知道 sample_index，可先用下面命令查索引
#    python - <<'PY'
#    import json
#    target_source_id = "在这里填 source_id"
#    p = "Sample/run_300k_20260413/combined_jsonl.jsonl"
#    with open(p, "r", encoding="utf-8-sig") as f:
#        for i, line in enumerate(f):
#            line = line.strip()
#            if not line:
#                continue
#            obj = json.loads(line)
#            if str(obj.get("source_id", "")) == target_source_id or str(obj.get("id", "")) == target_source_id:
#                print(i)
#                break
#        else:
#            print("NOT_FOUND")
#    PY

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer

from Encoders.cnn import CNN1DEncoder
from Encoders.mlp import MLPEncoder
from Encoders.patchtst import PatchTSTEncoder
from Models.multimodal_qwen import MultimodalQwen
from prompting import build_adaptive_prompt_messages, recommended_min_new_tokens


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "on"):
        return True
    if s in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {v}")


def resolve_precision(device, precision_arg):
    if device.type != "cuda":
        return "fp32", torch.float32, False

    bf16_supported = torch.cuda.is_bf16_supported()
    if precision_arg == "auto":
        mode = "bf16" if bf16_supported else "fp16"
    elif precision_arg == "bf16":
        mode = "bf16" if bf16_supported else "fp16"
    elif precision_arg in ("fp16", "fp32"):
        mode = precision_arg
    else:
        raise ValueError(f"未知精度模式: {precision_arg}")

    if mode == "bf16":
        return mode, torch.bfloat16, True
    if mode == "fp16":
        return mode, torch.float16, True
    return mode, torch.float32, False


def build_encoder(encoder_type: str, ts_seq_len: int):
    if encoder_type == "cnn":
        return CNN1DEncoder()
    if encoder_type == "mlp":
        return MLPEncoder(seq_len=ts_seq_len)
    if encoder_type == "patchtst":
        return PatchTSTEncoder(seq_len=ts_seq_len)
    raise ValueError(f"不支持的 encoder_type: {encoder_type}")


def load_tokenizer(model_path: str):
    """
    某些 tokenizer 配置会提示 regex 兼容问题，优先尝试修复参数；
    若当前 transformers 版本不支持该参数则自动回退。
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
            fix_mistral_regex=True,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_jsonl(path: Path):
    samples = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                samples.append(obj)
    return samples


def extract_series(sample: dict, main_channel_override: int | None = None):
    ts = sample.get("time_series")
    if isinstance(ts, list) and ts:
        return [float(x) for x in ts]

    values = sample.get("values")
    if isinstance(values, list) and values:
        first = values[0]
        if isinstance(first, list):
            if main_channel_override is not None:
                ch = int(main_channel_override)
            else:
                features = sample.get("features")
                if isinstance(features, dict):
                    ch = sample.get("main_channel", features.get("main_channel", 0))
                else:
                    ch = sample.get("main_channel", 0)
                ch = int(ch) if ch is not None else 0
            out = []
            for row in values:
                if not isinstance(row, list) or ch >= len(row):
                    raise ValueError("values 结构不合法或主变量索引越界")
                out.append(float(row[ch]))
            return out

        return [float(x) for x in values]

    raise ValueError("样本缺少 time_series 或 values 字段")


def normalize_ts(ts_values, ts_seq_len: int):
    ts = torch.tensor(ts_values, dtype=torch.float32).flatten()
    cur_len = ts.numel()
    if cur_len == ts_seq_len:
        return ts
    if cur_len > ts_seq_len:
        return ts[:ts_seq_len]
    pad_len = ts_seq_len - cur_len
    return torch.nn.functional.pad(ts, (0, pad_len), value=0.0)


def _looks_like_datetime_text(value: str) -> bool:
    s = str(value or "").strip()
    if not s:
        return False
    has_date_sep = ("-" in s) or ("/" in s) or ("." in s)
    has_time_sep = (":" in s)
    has_t = ("T" in s) or (" " in s)
    return has_date_sep and (has_time_sep or has_t)


def has_explicit_time_metadata(sample: dict) -> bool:
    if not isinstance(sample, dict):
        return False

    pair_candidates = [
        ("start_time", "end_time"),
        ("window_start_time", "window_end_time"),
        ("start_timestamp", "end_timestamp"),
    ]
    for k_start, k_end in pair_candidates:
        s = str(sample.get(k_start, "")).strip()
        e = str(sample.get(k_end, "")).strip()
        if _looks_like_datetime_text(s) and _looks_like_datetime_text(e):
            return True

    for key in ("timestamps", "time_index", "datetime", "time"):
        arr = sample.get(key)
        if isinstance(arr, list) and arr:
            first = str(arr[0]).strip()
            last = str(arr[-1]).strip()
            if _looks_like_datetime_text(first) and _looks_like_datetime_text(last):
                return True

    return False


def normalize_series_name(raw_name: str) -> str:
    name = str(raw_name or "").strip()
    if not name:
        return ""

    name = name.replace("-", "_")
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_\u4e00-\u9fff]", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        return ""

    parts = [p for p in name.split("_") if p]
    if len(name) <= 24 and len(parts) <= 3:
        return name

    # 长名称更容易在生成时出现拼写漂移，优先降级为稳定短名。
    if any(p.lower() == "value" for p in parts):
        return "value"

    tail = parts[-1] if parts else ""
    if tail.isdigit() and len(parts) >= 2:
        short = f"{parts[-2]}_{tail}"
        return short if len(short) <= 16 else "value"

    if re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,15}", tail or ""):
        return tail

    return "value"


def infer_series_name_from_source_id(source_id: str) -> str:
    sid = str(source_id or "").strip()
    if not sid:
        return ""

    m = re.search(r"([A-Za-z0-9_-]+)_L\d+", sid)
    if m is None:
        return ""

    base = str(m.group(1)).strip()
    if not base:
        return ""

    parts = [p for p in re.split(r"[_-]+", base) if p]
    if not parts:
        return ""

    if len(parts) >= 2 and parts[-1].isdigit():
        cand = f"{parts[-2]}_{parts[-1]}"
    else:
        cand = parts[-1]

    name = normalize_series_name(cand)
    if name:
        return name
    return normalize_series_name(base)


def infer_series_name(sample: dict, main_channel_override: int | None = None) -> str:
    if not isinstance(sample, dict):
        return "value"

    # Common direct name fields.
    for key in ("target_name", "target", "var_name", "feature_name", "column_name", "series_name"):
        v = normalize_series_name(sample.get(key, ""))
        if v:
            return v

    # Try features dictionary.
    features = sample.get("features")
    if isinstance(features, dict):
        for key in ("target_name", "target", "var_name", "feature_name", "column_name", "series_name"):
            v = normalize_series_name(features.get(key, ""))
            if v:
                return v

        feature_names = features.get("feature_names")
        if isinstance(feature_names, list) and feature_names:
            if main_channel_override is not None:
                idx = int(main_channel_override)
            else:
                idx = sample.get("main_channel", features.get("main_channel", 0))
                idx = int(idx) if idx is not None else 0
            if 0 <= idx < len(feature_names):
                name = normalize_series_name(feature_names[idx])
                if name:
                    return name

    sid = str(sample.get("source_id") or sample.get("id") or "").strip()
    sid_name = infer_series_name_from_source_id(sid)
    if sid_name:
        return sid_name

    return "value"


def build_prompt(
    tokenizer,
    ts_tensor,
    sample=None,
    force_detail_level=None,
    series_name="value",
    enforce_relative_time_no_meta=True,
):
    """基础复杂度提示 + 轻量无caption约束（时间锚点与变量名）。"""
    prompt_messages, prompt_meta = build_adaptive_prompt_messages(
        ts_tensor=ts_tensor,
        force_detail_level=force_detail_level,
    )

    sample_obj = sample if isinstance(sample, dict) else {}
    has_time_meta = has_explicit_time_metadata(sample_obj)

    if has_time_meta:
        time_rule = (
            "时间约束：样本包含时间锚点，可使用样本内可验证的时间表达；"
            "不要虚构新的时间范围。"
        )
    else:
        if bool(enforce_relative_time_no_meta):
            time_rule = (
                "时间约束：当前样本不含可验证绝对时间锚点。"
                "禁止生成具体日期或时刻；统一使用“第k个时间点/第a-b个时间点”表达。"
            )
        else:
            time_rule = "时间约束：若缺少可验证时间锚点，请谨慎使用时间表达，避免编造。"

    series_rule = (
        f"变量名约束：全文主变量名必须严格写成“{series_name}”，"
        "不要使用近似拼写、错别字、缩写或连字符变体。"
    )

    prompt_messages = list(prompt_messages) + [
        {"role": "user", "content": f"{time_rule}\n{series_rule}"}
    ]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_info = {
        "has_time_metadata": bool(has_time_meta),
        "series_name": str(series_name),
    }
    return prompt_text, prompt_meta, prompt_info


def decode_generated_text(tokenizer, generated_ids, prompt_token_len, extra_prefix_tokens=0):
    token_ids = generated_ids[0] if generated_ids.ndim == 2 else generated_ids
    cut_idx = prompt_token_len + max(0, int(extra_prefix_tokens))

    if token_ids.numel() > cut_idx:
        new_token_ids = token_ids[cut_idx:]
        text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        if text:
            return text

    if token_ids.numel() > prompt_token_len:
        new_token_ids = token_ids[prompt_token_len:]
        text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        if text:
            return text

    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def looks_incomplete_output(text: str) -> bool:
    t = str(text or "").strip()
    if not t:
        return True

    if "整体结论" not in t:
        return True

    if t.count("（") > t.count("）") or t.count("(") > t.count(")"):
        return True

    if t.endswith(("至", "到", "约", "为", "在", "从", "（", "(", "：", ":", "，", "、", "；", "-")):
        return True

    # 结构化结尾太短时，通常意味着被 token 上限截断。
    if re.search(r"整体结论\s*[：:]\s*[^。！？!?]{0,10}$", t):
        return True

    return False


def estimate_repetition_penalty(text: str) -> int:
    t = str(text or "")
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return 0

    signatures = []
    for ln in lines:
        sig = re.sub(r"第\s*\d+\s*阶段", "第N阶段", ln)
        sig = re.sub(r"\d+(?:\.\d+)?", "<n>", sig)
        sig = re.sub(r"\s+", "", sig)
        signatures.append(sig)

    dup_lines = len(signatures) - len(set(signatures))
    penalty = dup_lines * 55

    stage_lines = [s for s in signatures if "第N阶段" in s]
    if len(stage_lines) >= 4:
        dup_stages = len(stage_lines) - len(set(stage_lines))
        penalty += dup_stages * 75

    return int(penalty)


def estimate_prompt_artifact_penalty(text: str) -> int:
    t = str(text or "")
    if not t:
        return 0

    patterns = (
        r"请仅输出",
        r"时间约束",
        r"变量名约束",
        r"不要使用近似拼写",
        r"禁止生成具体日期",
        r"你是一位",
        r"自适应动态扩展",
    )
    hits = sum(1 for p in patterns if re.search(p, t))
    return int(hits * 220)


def score_decoded_text_quality(text: str, allow_absolute_time: bool = True) -> int:
    t = str(text or "")
    if not t.strip():
        return -10**9

    score = min(len(t), 260)
    if len(t) < 60:
        score -= 160
    if len(t) > 2200:
        score -= 120

    score -= estimate_prompt_artifact_penalty(t)

    for h in ("窗口概览", "分段观察", "异常点", "联动关系", "整体结论"):
        cnt = len(re.findall(rf"{re.escape(h)}\s*[：:]", t))
        if cnt > 1:
            score -= (cnt - 1) * 95

    markers = ("窗口概览", "分段观察", "异常点", "联动关系", "整体结论")
    marker_pos = {m: t.find(m) for m in markers}

    for m, pos in marker_pos.items():
        if pos >= 0:
            score += 80

    if t.startswith("窗口概览") or t.startswith("**窗口概览"):
        score += 140

    # Reward correct marker order; penalize out-of-order sections.
    order_ref = ["窗口概览", "分段观察", "异常点", "联动关系", "整体结论"]
    prev_pos = -1
    ordered_cnt = 0
    for m in order_ref:
        pos = marker_pos[m]
        if pos >= 0:
            if pos >= prev_pos:
                ordered_cnt += 1
                prev_pos = pos
            else:
                score -= 60
    score += ordered_cnt * 40

    if "（本段为自适应动态扩展" in t:
        score -= 220

    if looks_incomplete_output(t):
        score -= 260

    score -= estimate_repetition_penalty(t)

    if t and t[0] in "，。；：:、!！?？)）0123456789-":
        score -= 80

    # If there is no explicit time metadata, absolute timestamps are likely hallucinations.
    if not bool(allow_absolute_time):
        if re.search(r"\d{4}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{1,2}", t):
            score -= 420
        if re.search(r"\d{1,2}\s*:\s*\d{2}", t):
            score -= 180
        if re.search(r"第\s*\d+\s*(?:-|到|至)\s*\d+\s*个时间点", t):
            score += 40
        if re.search(r"第\s*\d+\s*个时间点", t):
            score += 20

    return int(score)


def decode_multimodal_with_fallback(
    tokenizer,
    mm_ids,
    prompt_token_len: int,
    soft_prompt_len: int,
    allow_absolute_time: bool = True,
):
    mm_text_soft_cut = decode_generated_text(
        tokenizer,
        mm_ids,
        prompt_token_len=prompt_token_len,
        extra_prefix_tokens=soft_prompt_len,
    )
    mm_text_prompt_cut = decode_generated_text(
        tokenizer,
        mm_ids,
        prompt_token_len=prompt_token_len,
        extra_prefix_tokens=0,
    )
    mm_text_no_cut = decode_generated_text(
        tokenizer,
        mm_ids,
        prompt_token_len=0,
        extra_prefix_tokens=0,
    )

    candidate_scores = {
        "soft_cut": score_decoded_text_quality(mm_text_soft_cut, allow_absolute_time=allow_absolute_time),
        "prompt_cut": score_decoded_text_quality(mm_text_prompt_cut, allow_absolute_time=allow_absolute_time),
        "no_cut": score_decoded_text_quality(mm_text_no_cut, allow_absolute_time=allow_absolute_time),
    }

    # 默认优先 soft/prompt 两种切分，避免 no_cut 把提示词回显误判成高质量输出。
    if candidate_scores["prompt_cut"] >= candidate_scores["soft_cut"]:
        best_name = "prompt_cut"
        best_text = mm_text_prompt_cut
    else:
        best_name = "soft_cut"
        best_text = mm_text_soft_cut

    no_cut_penalty = estimate_prompt_artifact_penalty(mm_text_no_cut)
    if no_cut_penalty == 0 and candidate_scores["no_cut"] >= candidate_scores[best_name] + 120:
        best_name = "no_cut"
        best_text = mm_text_no_cut

    return best_text, best_name, candidate_scores


def main():
    parser = argparse.ArgumentParser(description="加载 checkpoint 做独立推理验证（简化版）")
    train_dir = Path(__file__).resolve().parent
    project_root = train_dir.parent
    preferred_model_path = project_root / "Models" / "Qwen3-0.6B-Instruct-2512"
    fallback_model_path = train_dir / "Models" / "Qwen3-0.6B-Instruct-2512"
    default_model_path = preferred_model_path if preferred_model_path.exists() else fallback_model_path

    parser.add_argument("--checkpoint", type=str, required=True, help="custom_ts_weights.pth 路径")
    parser.add_argument("--input_jsonl", type=str, required=True, help="用于推理的 JSONL 文件")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--encoder_type", type=str, default="patchtst", choices=["cnn", "mlp", "patchtst"])
    parser.add_argument("--model_path", type=str, default=str(default_model_path))
    parser.add_argument("--ts_seq_len", type=int, default=512)
    parser.add_argument("--soft_prompt_len", type=int, default=None)
    parser.add_argument("--use_gating", type=str, choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--use_bridge_norm", type=str, choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--main_channel", type=int, default=None, help="多变量时可手动指定主变量索引")
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--min_new_tokens", type=int, default=-1, help="-1 表示按复杂度自动设置")
    parser.add_argument("--do_sample", type=str2bool, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.08, help="重复惩罚，>1 可减少模板化重复")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=6, help="禁止重复的 ngram 长度；0 表示关闭")
    parser.add_argument(
        "--retry_on_incomplete",
        type=str2bool,
        default=True,
        help="检测到输出未完整收尾时，自动补全重试一次",
    )
    parser.add_argument(
        "--retry_max_new_tokens",
        type=int,
        default=896,
        help="补全重试时使用的 max_new_tokens（需大于当前 --max_new_tokens 才生效）",
    )
    parser.add_argument(
        "--force_detail_level",
        type=str,
        choices=["auto", "brief", "standard", "detailed"],
        default="auto",
        help="强制输出详细度；auto 表示由序列复杂度自动判定",
    )
    parser.add_argument(
        "--show_reference_caption",
        type=str2bool,
        default=True,
        help="是否打印样本原始 caption（仅用于人工对比，不参与生成）",
    )
    parser.add_argument(
        "--show_base_text",
        type=str2bool,
        default=False,
        help="是否同时打印纯文本基座输出（用于诊断）",
    )
    parser.add_argument(
        "--series_name",
        type=str,
        default="auto",
        help="主变量名；auto 表示从样本字段自动推断（推断失败则使用“目标序列”）",
    )
    parser.add_argument(
        "--enforce_relative_time_no_meta",
        type=str2bool,
        default=True,
        help="当样本无可验证时间锚点时，是否强制使用相对时间点表达并禁止具体日期",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    input_jsonl = Path(args.input_jsonl).resolve()
    if not input_jsonl.exists():
        raise FileNotFoundError(f"input_jsonl 不存在: {input_jsonl}")

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    ckpt_encoder_type = str(ckpt_args.get("encoder_type", "")).strip()
    ckpt_model_path = str(ckpt_args.get("model_path", "")).strip()
    ckpt_ts_seq_len = ckpt_args.get("ts_seq_len", None)
    ckpt_soft_prompt_len = ckpt_args.get("soft_prompt_len", None)

    print("[INFER] ===== checkpoint args 摘要 =====")
    print(
        "[INFER] ckpt_args: "
        f"encoder_type={ckpt_encoder_type or 'N/A'} "
        f"model_path={ckpt_model_path or 'N/A'} "
        f"ts_seq_len={ckpt_ts_seq_len if ckpt_ts_seq_len is not None else 'N/A'} "
        f"soft_prompt_len={ckpt_soft_prompt_len if ckpt_soft_prompt_len is not None else 'N/A'}"
    )

    if ckpt_encoder_type and ckpt_encoder_type != str(args.encoder_type):
        print(
            "[WARN] encoder_type 与 checkpoint 记录不一致: "
            f"cli={args.encoder_type} ckpt={ckpt_encoder_type}"
        )
    if ckpt_ts_seq_len is not None and int(ckpt_ts_seq_len) != int(args.ts_seq_len):
        print(
            "[WARN] ts_seq_len 与 checkpoint 记录不一致: "
            f"cli={args.ts_seq_len} ckpt={int(ckpt_ts_seq_len)}"
        )
    if ckpt_model_path:
        cli_model_name = Path(str(args.model_path)).name
        ckpt_model_name = Path(ckpt_model_path).name
        if cli_model_name and ckpt_model_name and cli_model_name != ckpt_model_name:
            print(
                "[WARN] model_path 与 checkpoint 记录可能不一致: "
                f"cli={cli_model_name} ckpt={ckpt_model_name}"
            )

    soft_prompt_len = args.soft_prompt_len
    if soft_prompt_len is None:
        soft_prompt_len = int(ckpt_args.get("soft_prompt_len", 4))

    if args.use_gating == "auto":
        use_gating = bool(ckpt_args.get("use_gating", True))
    else:
        use_gating = args.use_gating == "true"

    if args.use_bridge_norm == "auto":
        use_bridge_norm = bool(ckpt_args.get("use_bridge_norm", True))
    else:
        use_bridge_norm = args.use_bridge_norm == "true"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision_mode, compute_dtype, use_amp = resolve_precision(device, args.precision)

    tokenizer = load_tokenizer(args.model_path)

    encoder = build_encoder(args.encoder_type, ts_seq_len=args.ts_seq_len)
    model = MultimodalQwen(
        encoder=encoder,
        qwen_model_path=args.model_path,
        llm_dtype=compute_dtype,
        soft_prompt_len=soft_prompt_len,
        use_gating=use_gating,
        use_bridge_norm=use_bridge_norm,
    )

    if isinstance(ckpt, dict) and "trainable_state_dict" in ckpt:
        load_result = model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
        missing_keys = list(getattr(load_result, "missing_keys", []) or [])
        unexpected_keys = list(getattr(load_result, "unexpected_keys", []) or [])
        print(
            "[INFER] load_state_dict(trainable_state_dict): "
            f"missing={len(missing_keys)} unexpected={len(unexpected_keys)}"
        )
        if missing_keys:
            print(f"[INFER][WARN] missing_keys sample: {missing_keys[:8]}")
        if unexpected_keys:
            print(f"[INFER][WARN] unexpected_keys sample: {unexpected_keys[:8]}")
    elif isinstance(ckpt, dict) and "encoder_state_dict" in ckpt and "projection_state_dict" in ckpt:
        enc_result = model.encoder.load_state_dict(ckpt["encoder_state_dict"], strict=False)
        proj_result = model.projection.load_state_dict(ckpt["projection_state_dict"], strict=False)
        enc_missing = list(getattr(enc_result, "missing_keys", []) or [])
        enc_unexpected = list(getattr(enc_result, "unexpected_keys", []) or [])
        proj_missing = list(getattr(proj_result, "missing_keys", []) or [])
        proj_unexpected = list(getattr(proj_result, "unexpected_keys", []) or [])
        print(
            "[INFER] load_state_dict(split): "
            f"encoder_missing={len(enc_missing)} encoder_unexpected={len(enc_unexpected)} "
            f"projection_missing={len(proj_missing)} projection_unexpected={len(proj_unexpected)}"
        )
        if enc_missing or enc_unexpected or proj_missing or proj_unexpected:
            print(
                "[INFER][WARN] split load samples: "
                f"enc_missing={enc_missing[:8]} enc_unexpected={enc_unexpected[:8]} "
                f"proj_missing={proj_missing[:8]} proj_unexpected={proj_unexpected[:8]}"
            )
    else:
        raise ValueError("checkpoint 格式不受支持，未找到可加载的权重字段")

    model.to(device)
    model.eval()

    samples = load_jsonl(input_jsonl)
    if not samples:
        raise ValueError(f"输入 JSONL 为空: {input_jsonl}")

    idx = max(0, min(int(args.sample_index), len(samples) - 1))
    sample = samples[idx]
    ts = extract_series(sample, main_channel_override=args.main_channel)

    prompt_ts_tensor = torch.tensor(ts, dtype=torch.float32).flatten()
    if prompt_ts_tensor.numel() > args.ts_seq_len:
        prompt_ts_tensor = prompt_ts_tensor[: args.ts_seq_len]
    ts_tensor = normalize_ts(ts, ts_seq_len=args.ts_seq_len).unsqueeze(0).to(device)

    force_detail_level = None if args.force_detail_level == "auto" else args.force_detail_level
    if str(args.series_name).strip().lower() == "auto":
        series_name = infer_series_name(sample, main_channel_override=args.main_channel)
    else:
        series_name = str(args.series_name).strip() or "目标序列"

    prompt_text, prompt_meta, prompt_info = build_prompt(
        tokenizer=tokenizer,
        ts_tensor=prompt_ts_tensor,
        sample=sample,
        force_detail_level=force_detail_level,
        series_name=series_name,
        enforce_relative_time_no_meta=bool(args.enforce_relative_time_no_meta),
    )
    prompt_enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = prompt_enc["input_ids"].to(device)
    attention_mask = prompt_enc["attention_mask"].to(device)

    has_time_metadata = bool((prompt_info or {}).get("has_time_metadata", False))
    allow_absolute_time = has_time_metadata or (not bool(args.enforce_relative_time_no_meta))

    auto_min_tokens = recommended_min_new_tokens(prompt_meta["detail_level"])
    if int(args.min_new_tokens) >= 0:
        min_new_tokens = int(args.min_new_tokens)
    else:
        min_new_tokens = auto_min_tokens
    min_new_tokens = min(min_new_tokens, int(args.max_new_tokens))

    gen_kwargs = {
        "max_new_tokens": int(args.max_new_tokens),
        "min_new_tokens": int(min_new_tokens),
        "do_sample": bool(args.do_sample),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if float(args.repetition_penalty) > 1.0:
        gen_kwargs["repetition_penalty"] = float(args.repetition_penalty)
    if int(args.no_repeat_ngram_size) > 0:
        gen_kwargs["no_repeat_ngram_size"] = int(args.no_repeat_ngram_size)
    if args.do_sample:
        gen_kwargs["temperature"] = float(args.temperature)
        gen_kwargs["top_p"] = float(args.top_p)
    else:
        # 关闭采样时去掉温度相关噪声告警。
        try:
            model.llm.generation_config.do_sample = False
            if hasattr(model.llm.generation_config, "temperature"):
                model.llm.generation_config.temperature = None
            if hasattr(model.llm.generation_config, "top_p"):
                model.llm.generation_config.top_p = None
            if hasattr(model.llm.generation_config, "top_k"):
                model.llm.generation_config.top_k = None
        except Exception:
            pass

    if use_amp and device.type == "cuda":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=True)
    else:
        autocast_ctx = torch.autocast(device_type=device.type, enabled=False)

    with torch.no_grad(), autocast_ctx:
        if bool(args.show_base_text):
            base_ids = model.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
        else:
            base_ids = None

        mm_ids = model.generate(
            ts_data=ts_tensor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    base_text = ""
    if bool(args.show_base_text) and base_ids is not None:
        base_text = decode_generated_text(
            tokenizer,
            base_ids,
            prompt_token_len=input_ids.size(1),
            extra_prefix_tokens=0,
        )

    mm_text, best_name, candidate_scores = decode_multimodal_with_fallback(
        tokenizer,
        mm_ids,
        prompt_token_len=input_ids.size(1),
        soft_prompt_len=soft_prompt_len,
        allow_absolute_time=allow_absolute_time,
    )

    if bool(args.retry_on_incomplete):
        retry_target = int(args.retry_max_new_tokens)
        if looks_incomplete_output(mm_text) and retry_target > int(args.max_new_tokens):
            print(
                "[INFER][INFO] 检测到输出可能不完整，执行一次补全重试："
                f"max_new_tokens {args.max_new_tokens} -> {retry_target}"
            )
            retry_kwargs = dict(gen_kwargs)
            retry_kwargs["max_new_tokens"] = retry_target
            retry_kwargs["min_new_tokens"] = min(int(retry_kwargs["min_new_tokens"]), int(retry_target // 2))

            with torch.no_grad(), autocast_ctx:
                mm_ids_retry = model.generate(
                    ts_data=ts_tensor,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **retry_kwargs,
                )

            mm_text_retry, best_name_retry, candidate_scores_retry = decode_multimodal_with_fallback(
                tokenizer,
                mm_ids_retry,
                prompt_token_len=input_ids.size(1),
                soft_prompt_len=soft_prompt_len,
                allow_absolute_time=allow_absolute_time,
            )

            if score_decoded_text_quality(mm_text_retry, allow_absolute_time=allow_absolute_time) > score_decoded_text_quality(
                mm_text,
                allow_absolute_time=allow_absolute_time,
            ):
                mm_text = mm_text_retry
                best_name = best_name_retry
                candidate_scores = candidate_scores_retry

    if best_name != "soft_cut":
        print(f"[INFER][INFO] multimodal decode fallback selected: {best_name}")
    print(
        "[INFER][INFO] multimodal decode scores: "
        f"soft_cut={candidate_scores['soft_cut']} "
        f"prompt_cut={candidate_scores['prompt_cut']} "
        f"no_cut={candidate_scores['no_cut']}"
    )

    print("[INFER] ===== 独立推理验证（简化版） =====")
    print(f"[INFER] checkpoint: {ckpt_path}")
    print(f"[INFER] encoder_type: {args.encoder_type}")
    print(f"[INFER] device: {device}, precision: {precision_mode}")
    print(f"[INFER] sample_index: {idx}")
    print("[INFER] caption_in_prompt: False（仅使用时序数据构建提示词）")
    print(
        "[INFER] prompt_constraints: "
        f"series_name={series_name} "
        f"has_time_metadata={has_time_metadata} "
        f"enforce_relative_time_no_meta={args.enforce_relative_time_no_meta}"
    )
    print(
        "[INFER] detail_level: "
        f"{prompt_meta['detail_level']} "
        f"(complexity={prompt_meta['complexity_score']:.3f}, min_new_tokens={min_new_tokens}, max_new_tokens={args.max_new_tokens})"
    )
    if "id" in sample:
        print(f"[INFER] sample_id: {sample.get('id')}")
    if bool(args.show_reference_caption) and ("caption" in sample):
        print(f"[INFER] reference_caption: {sample.get('caption')}")
    if bool(args.show_base_text):
        print(f"[INFER] base_text: {base_text if base_text else '[空输出]'}")
    print(f"[INFER] multimodal_text: {mm_text if mm_text else '[空输出]'}")


if __name__ == "__main__":
    main()
