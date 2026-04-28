import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers import AutoTokenizer

from Encoders.cnn import CNN1DEncoder
from Encoders.mlp import MLPEncoder
from Encoders.patchtst import PatchTSTEncoder
from Models.multimodal_qwen import MultimodalQwen
from prompting import build_adaptive_prompt_messages, recommended_min_new_tokens


def get_free_gpu():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split("\n")
        gpu_list = []
        for line in lines:
            parts = line.split(",")
            if len(parts) >= 2:
                idx = int(parts[0].strip())
                free_mem = int(parts[1].strip())
                gpu_list.append((idx, free_mem))
        if gpu_list:
            gpu_list.sort(key=lambda x: x[1], reverse=True)
            free_idx = gpu_list[0][0]
            print(f"[AUTO_GPU] 选择 GPU {free_idx}，空闲内存: {gpu_list[0][1]} MiB")
            return free_idx
    except Exception as e:
        print(f"[AUTO_GPU] 自动选择GPU失败: {e}")
    return 0


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
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
            fix_mistral_regex=True,
            local_files_only=True,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_jsonl(path):
    samples = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                samples.append(obj)
    return samples


def extract_series_from_qa(sample):
    ts = sample.get("series")
    if isinstance(ts, list) and ts:
        return [float(x) for x in ts]
    raise ValueError("样本缺少 series 字段")


def normalize_ts(ts_values, ts_seq_len):
    ts = torch.tensor(ts_values, dtype=torch.float32).flatten()
    cur_len = ts.numel()
    if cur_len == ts_seq_len:
        return ts
    if cur_len > ts_seq_len:
        return ts[:ts_seq_len]
    pad_len = ts_seq_len - cur_len
    return torch.nn.functional.pad(ts, (0, pad_len), value=0.0)


def build_prompt(tokenizer, ts_tensor, series_name="value", enforce_relative_time_no_meta=True, encoder_type="cnn"):
    prompt_messages, prompt_meta = build_adaptive_prompt_messages(
        ts_tensor=ts_tensor,
        force_detail_level=None,
    )

    time_rule = (
        "时间约束：当前样本不含可验证绝对时间锚点。"
        "禁止生成具体日期或时刻；统一使用\"第k个时间点/第a-b个时间点\"表达。"
        if bool(enforce_relative_time_no_meta) else
        "时间约束：若缺少可验证时间锚点，请谨慎使用时间表达，避免编造。"
    )

    series_rule = (
        f"变量名约束：全文主变量名必须严格写成\"{series_name}\"，"
        "不要使用近似拼写、错别字、缩写或连字符变体。"
    )

    # 根据编码器类型使用不同的结构约束
    if encoder_type == "patchtst":
        structure_rule = "输出格式约束：\n必须以\"窗口概览：\"开头，严格按顺序生成以下四个部分，每部分必须有实质内容：\n窗口概览：（描述序列整体特征）\n分段观察：（描述各阶段变化）\n异常点：（识别异常情况，无异常写\"无明显异常点\"）\n整体结论：（总结整体趋势）\n\n重要：\n1. 开头必须是\"窗口概览：\"，不要有任何其他文字\n2. 必须包含上述四个部分，缺一不可\n3. 生成完整内容后再停止，不要中途截断\n4. 不要在输出中重复这段约束文字"
    else:
        structure_rule = "结构约束：请严格按照以下结构生成描述：\n1. 窗口概览：\n2. 分段观察：\n3. 异常点：\n4. 整体结论：\n\n重要要求：\n- 必须以\"窗口概览：\"作为开头\n- 每个部分都必须有详细内容，不能跳过任何部分\n- 确保输出完整的描述，不要中途停止\n- 不要重复输出提示词或结构模板\n- 直接开始生成内容，不要有任何引言或开场白"

    prompt_messages = list(prompt_messages) + [
        {"role": "user", "content": f"{time_rule}\n{series_rule}\n{structure_rule}"}
    ]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt_text, prompt_meta


def decode_generated_text(tokenizer, generated_ids, prompt_token_len, extra_prefix_tokens=0):
    token_ids = generated_ids[0] if generated_ids.ndim == 2 else generated_ids
    cut_idx = prompt_token_len + max(0, int(extra_prefix_tokens))
    if token_ids.numel() > cut_idx:
        new_token_ids = token_ids[cut_idx:]
        text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        if text:
            # 确保只返回生成的内容，移除可能的提示词重复
            if "窗口概览：" in text:
                # 找到窗口概览的位置，从这里开始返回
                start_idx = text.find("窗口概览：")
                if start_idx != -1:
                    return text[start_idx:]
            return text
    if token_ids.numel() > prompt_token_len:
        new_token_ids = token_ids[prompt_token_len:]
        text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        if text:
            # 确保只返回生成的内容，移除可能的提示词重复
            if "窗口概览：" in text:
                # 找到窗口概览的位置，从这里开始返回
                start_idx = text.find("窗口概览：")
                if start_idx != -1:
                    return text[start_idx:]
            return text
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="QA数据集批量推理脚本")
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint 路径")
    parser.add_argument("--input_file", type=str, required=True, help="QA数据文件路径")
    parser.add_argument("--output_file", type=str, required=True, help="输出预测文件路径")
    parser.add_argument("--encoder_type", type=str, default="patchtst", choices=["cnn", "mlp", "patchtst"])
    parser.add_argument("--model_path", type=str, default="Models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--ts_seq_len", type=int, default=512)
    parser.add_argument("--precision", type=str, default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--max_new_tokens", type=int, default=0, help="0表示根据编码器类型自动设置")
    parser.add_argument("--min_new_tokens", type=int, default=-1, help="-1表示根据编码器类型和序列复杂度自动设置")
    parser.add_argument("--temperature", type=float, default=0.0, help="0表示根据编码器类型自动设置")
    parser.add_argument("--repetition_penalty", type=float, default=0.0, help="0表示根据编码器类型自动设置")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=6)
    parser.add_argument("--soft_prompt_len", type=int, default=4)
    parser.add_argument("--enforce_relative_time_no_meta", type=str2bool, default=True)
    parser.add_argument("--log_interval", type=int, default=10, help="日志输出间隔")
    parser.add_argument("--sample_size", type=int, default=0, help="测试模式：仅处理前N个样本，0表示处理所有样本")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    ckpt_path = project_root / args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    input_file = Path(args.input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    # 检查CUDA_VISIBLE_DEVICES是否已设置
    import os
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible:
        # 如果已设置CUDA_VISIBLE_DEVICES，使用默认设备0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[AUTO_GPU] CUDA_VISIBLE_DEVICES已设置为 {cuda_visible}，使用默认设备 cuda:0")
    else:
        # 自动选择空闲GPU
        free_gpu_id = get_free_gpu()
        torch.cuda.set_device(free_gpu_id)
        device = torch.device(f"cuda:{free_gpu_id}" if torch.cuda.is_available() else "cpu")
    precision_mode, compute_dtype, use_amp = resolve_precision(device, args.precision)

    print(f"[QA_INFER] device={device}, precision={precision_mode}")
    print(f"[QA_INFER] 加载 tokenizer: {args.model_path}")
    tokenizer = load_tokenizer(args.model_path)

    print(f"[QA_INFER] 构建 encoder: {args.encoder_type}")
    encoder = build_encoder(args.encoder_type, ts_seq_len=args.ts_seq_len)

    print(f"[QA_INFER] 加载模型...")
    model = MultimodalQwen(
        encoder=encoder,
        qwen_model_path=args.model_path,
        llm_dtype=compute_dtype,
        soft_prompt_len=args.soft_prompt_len,
        use_gating=True,
        use_bridge_norm=True,
    )

    print(f"[QA_INFER] 加载权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    if isinstance(ckpt, dict) and "trainable_state_dict" in ckpt:
        load_result = model.load_state_dict(ckpt["trainable_state_dict"], strict=False)
    elif isinstance(ckpt, dict) and "encoder_state_dict" in ckpt and "projection_state_dict" in ckpt:
        model.encoder.load_state_dict(ckpt["encoder_state_dict"], strict=False)
        model.projection.load_state_dict(ckpt["projection_state_dict"], strict=False)
    else:
        raise ValueError("checkpoint 格式不受支持")

    model.to(device)
    model.eval()

    # 根据编码器类型设置推理参数
    encoder_params = {
        "cnn": {
            "max_new_tokens": 500,
            "min_new_tokens_base": 150,
            "temperature": 0.7,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 6
        },
        "mlp": {
            "max_new_tokens": 600,
            "min_new_tokens_base": 200,
            "temperature": 0.1,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 10
        },
        "patchtst": {
            "max_new_tokens": 700,
            "min_new_tokens_base": 200,
            "temperature": 0.2,
            "repetition_penalty": 1.25,
            "no_repeat_ngram_size": 10
        }
    }

    # 使用用户指定的参数或默认参数
    params = encoder_params.get(args.encoder_type, encoder_params["cnn"])
    max_new_tokens = args.max_new_tokens if args.max_new_tokens > 0 else params["max_new_tokens"]
    min_new_tokens_base = params["min_new_tokens_base"]
    temperature = args.temperature if args.temperature > 0 else params["temperature"]
    repetition_penalty = args.repetition_penalty if args.repetition_penalty > 0 else params["repetition_penalty"]
    no_repeat_ngram_size = params.get("no_repeat_ngram_size", args.no_repeat_ngram_size)

    print(f"[QA_INFER] 推理参数: max_new_tokens={max_new_tokens}, temperature={temperature}, repetition_penalty={repetition_penalty}, no_repeat_ngram_size={no_repeat_ngram_size}")

    # 加载QA数据集
    print(f"[QA_INFER] 处理数据集: {input_file.name}")
    samples = load_jsonl(input_file)

    # 测试模式：仅处理前N个样本
    if args.sample_size > 0:
        samples = samples[:args.sample_size]
        print(f"[QA_INFER] 测试模式：仅处理前 {args.sample_size} 个样本")

    all_predictions = []
    total_samples = 0

    for idx, sample in enumerate(samples):
        ts_id = sample.get("ts_id", f"unknown_{idx}")
        dataset = sample.get("dataset", "tsshapeqa_v1")
        series_name = "value"  # QA数据集使用value作为变量名

        ts_values = extract_series_from_qa(sample)
        ts_tensor = normalize_ts(ts_values, ts_seq_len=args.ts_seq_len).unsqueeze(0).to(device)

        prompt_text, prompt_meta = build_prompt(
            tokenizer=tokenizer,
            ts_tensor=torch.tensor(ts_values, dtype=torch.float32).flatten()[:args.ts_seq_len],
            series_name=series_name,
            enforce_relative_time_no_meta=bool(args.enforce_relative_time_no_meta),
            encoder_type=args.encoder_type,
        )

        prompt_enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids = prompt_enc["input_ids"].to(device)
        attention_mask = prompt_enc["attention_mask"].to(device)

        auto_min_tokens = recommended_min_new_tokens(prompt_meta["detail_level"])
        # 结合编码器基础值和自动计算值，取较大者
        min_new_tokens = int(args.min_new_tokens) if int(args.min_new_tokens) >= 0 else max(auto_min_tokens, min_new_tokens_base)
        min_new_tokens = min(min_new_tokens, max_new_tokens)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
        }

        if use_amp and device.type == "cuda":
            autocast_ctx = torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=True)
        else:
            autocast_ctx = torch.autocast(device_type=device.type, enabled=False)

        with torch.no_grad(), autocast_ctx:
            mm_ids = model.generate(
                ts_data=ts_tensor,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        pred_caption = decode_generated_text(
            tokenizer,
            mm_ids,
            prompt_token_len=input_ids.size(1),
            extra_prefix_tokens=args.soft_prompt_len,
        )

        all_predictions.append({
            "ts_id": ts_id,
            "dataset": dataset,
            "pred_caption": pred_caption,
        })

        total_samples += 1
        if (idx + 1) % args.log_interval == 0 or (idx + 1) == len(samples):
            print(f"[QA_INFER] {input_file.name}: {idx + 1}/{len(samples)} 样本已处理")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for pred in all_predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    print(f"[QA_INFER] 完成！共处理 {total_samples} 个样本")
    print(f"[QA_INFER] 预测文件已保存到: {output_path}")


if __name__ == "__main__":
    main()
