# 训练启动命令（项目根目录执行，全部相对路径）
#
# 说明：若命令中未显式传入 --max_text_length / --ts_seq_len，
# 会自动使用脚本默认值 512 / 512。
#
# 一、训练集输入策略（决定"是否自动合并 JSONL"）
# 1) 使用默认样本目录（会自动合并 Sample/run_300k_20260413/*/samples_filtered.jsonl）
#    python -u Train/train.py --training_stage frozen
# 2) 指定样本目录与合并文件名（仍会自动合并）
#    python -u Train/train.py --sample_run_dir Sample/run_300k_20260413 --combined_jsonl_name combined_jsonl.jsonl --training_stage frozen
# 3) 直接指定已存在训练集 jsonl（跳过合并，服务器复现更推荐）
#    python -u Train/train.py --train_jsonl Sample/run_300k_20260413/combined_jsonl.jsonl --training_stage frozen
#
# 二、Linux 服务器执行方式（与上面的"输入策略"正交，可组合使用）
# 4) 服务器单卡 4B（建议先前台跑通）
#    CUDA_VISIBLE_DEVICES=0 python -u Train/train.py --training_stage frozen --model_path Models/Qwen3-4B-Instruct-2507 --batch_size 1 --num_workers 4 --precision bf16 --train_jsonl Sample/run_300k_20260413/combined_jsonl.jsonl --preview_max_new_tokens 192
# 5) 服务器后台运行（断开 SSH 后继续训练）
#    nohup bash -lc 'CUDA_VISIBLE_DEVICES=0 python -u Train/train.py --training_stage frozen --model_path Models/Qwen3-4B-Instruct-2507 --batch_size 1 --num_workers 4 --precision bf16 --train_jsonl Sample/run_300k_20260413/combined_jsonl.jsonl --preview_max_new_tokens 192' > Train/logs/train_4b_frozen_gpu0.log 2>&1 &
# 6) 三卡并行（非 DDP，自动从全机 8 卡里选任意 3 张空闲卡；每张卡跑一个独立实验）
#    IDLE_GPUS=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', *' '$2<1000 && $3<10 {print $1}' | head -n 3 | xargs); [ "$(echo "$IDLE_GPUS" | awk '{print NF}')" -lt 3 ] && echo "空闲GPU不足3张" && exit 1; set -- $IDLE_GPUS; G0=$1; G1=$2; G2=$3; nohup bash -lc "CUDA_VISIBLE_DEVICES=$G0 python -u Train/train.py --training_stage frozen --model_path Models/Qwen3-4B-Instruct-2507 --batch_size 1 --num_workers 4 --precision bf16 --encoder_type cnn --train_jsonl Sample/run_300k_20260413/combined_jsonl.jsonl --preview_max_new_tokens 192" > Train/logs/train_4b_frozen_gpu${G0}_cnn.log 2>&1 & nohup bash -lc "CUDA_VISIBLE_DEVICES=$G1 python -u Train/train.py --training_stage frozen --model_path Models/Qwen3-4B-Instruct-2507 --batch_size 1 --num_workers 4 --precision bf16 --encoder_type mlp --train_jsonl Sample/run_300k_20260413/combined_jsonl.jsonl --preview_max_new_tokens 192" > Train/logs/train_4b_frozen_gpu${G1}_mlp.log 2>&1 & nohup bash -lc "CUDA_VISIBLE_DEVICES=$G2 python -u Train/train.py --training_stage frozen --model_path Models/Qwen3-4B-Instruct-2507 --batch_size 1 --num_workers 4 --precision bf16 --encoder_type patchtst --train_jsonl Sample/run_300k_20260413/combined_jsonl.jsonl --preview_max_new_tokens 192" > Train/logs/train_4b_frozen_gpu${G2}_patchtst.log 2>&1 & echo "已启动GPU: $G0 $G1 $G2"
# 7) 自动选任意 1 张空闲卡并前台启动（显存<1000MB 且利用率<10%）
#    IDLE_GPU=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', *' '$2<1000 && $3<10 {print $1; exit}'); [ -z "$IDLE_GPU" ] && echo "无空闲GPU" && exit 1; CUDA_VISIBLE_DEVICES=$IDLE_GPU python -u Train/train.py --training_stage frozen --model_path Models/Qwen3-4B-Instruct-2507 --batch_size 1 --num_workers 4 --precision bf16 --train_jsonl Sample/run_300k_20260413/combined_jsonl.jsonl --preview_max_new_tokens 192
#    IDLE_GPU=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', *' '$2<1000 && $3<10 {print $1; exit}'); [ -z "$IDLE_GPU" ] && echo "无空闲GPU" && exit 1; CUDA_VISIBLE_DEVICES=$IDLE_GPU python -u Train/train.py --training_stage frozen --model_path Models/Qwen3-4B-Instruct-2507 --batch_size 1 --num_workers 4 --precision bf16 --preview_max_new_tokens 192
# 8) 自动选任意 1 张空闲卡并后台启动（推荐）
#    IDLE_GPU=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', *' '$2<1000 && $3<10 {print $1; exit}'); [ -z "$IDLE_GPU" ] && echo "无空闲GPU" && exit 1; nohup bash -lc "CUDA_VISIBLE_DEVICES=$IDLE_GPU python -u Train/train.py --training_stage frozen --model_path Models/Qwen3-4B-Instruct-2507 --batch_size 1 --num_workers 4 --precision bf16 --train_jsonl Sample/run_300k_20260413/combined_jsonl.jsonl --preview_max_new_tokens 192" > Train/logs/train_4b_frozen_gpu${IDLE_GPU}.log 2>&1 &

import argparse
import contextlib
import json
import shutil
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from Encoders.cnn import CNN1DEncoder
from Encoders.mlp import MLPEncoder
from Encoders.patchtst import PatchTSTEncoder
from Models.multimodal_qwen import MultimodalQwen
from dataset import JsonlTimeSeriesDataset, build_collate_fn
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


def resolve_relative_path(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def merge_filtered_samples_jsonl(sample_run_dir: Path, combined_jsonl_path: Path) -> dict:
    sample_run_dir = Path(sample_run_dir).resolve()
    combined_jsonl_path = Path(combined_jsonl_path).resolve()

    if not sample_run_dir.exists():
        raise FileNotFoundError(f"样本目录不存在: {sample_run_dir}")

    source_files = sorted(
        [p for p in sample_run_dir.glob("*/samples_filtered.jsonl") if p.is_file()],
        key=lambda p: str(p).lower(),
    )
    if not source_files:
        raise FileNotFoundError(
            "未找到可合并的 samples_filtered.jsonl，期望路径形如: "
            f"{sample_run_dir}/*/samples_filtered.jsonl"
        )

    combined_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = combined_jsonl_path.with_name(combined_jsonl_path.name + ".tmp")

    written_lines = 0
    invalid_lines = 0
    changed_files: list[str] = []
    per_file_written: dict[str, int] = {}

    try:
        with tmp_path.open("w", encoding="utf-8", newline="\n") as f_out:
            for src in source_files:
                pre_stat = src.stat()
                kept = 0

                with src.open("r", encoding="utf-8", errors="ignore") as f_in:
                    for raw in f_in:
                        line = raw.strip()
                        if not line:
                            continue
                        try:
                            json.loads(line)
                        except Exception:
                            invalid_lines += 1
                            continue
                        f_out.write(line + "\n")
                        kept += 1
                        written_lines += 1

                per_file_written[str(src)] = kept
                post_stat = src.stat()
                if (
                    pre_stat.st_size != post_stat.st_size
                    or pre_stat.st_mtime_ns != post_stat.st_mtime_ns
                ):
                    changed_files.append(str(src))

        tmp_path.replace(combined_jsonl_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return {
        "source_file_count": len(source_files),
        "written_lines": written_lines,
        "invalid_lines": invalid_lines,
        "changed_files": changed_files,
        "per_file_written": per_file_written,
    }


def resolve_precision(device, precision_arg):
    if device.type != "cuda":
        return "fp32", torch.float32, False, False

    bf16_supported = torch.cuda.is_bf16_supported()

    if precision_arg == "auto":
        precision_mode = "bf16" if bf16_supported else "fp16"
    elif precision_arg == "bf16":
        if bf16_supported:
            precision_mode = "bf16"
        else:
            print("[WARN] 当前 GPU 不支持 BF16，自动回退到 FP16。")
            precision_mode = "fp16"
    elif precision_arg in ("fp16", "fp32"):
        precision_mode = precision_arg
    else:
        raise ValueError(f"未知精度模式: {precision_arg}")

    if precision_mode == "bf16":
        return precision_mode, torch.bfloat16, True, False
    if precision_mode == "fp16":
        return precision_mode, torch.float16, True, True
    return precision_mode, torch.float32, False, False


def build_checkpoint_stage_dir(train_dir, training_stage):
    root = train_dir / "Checkpoints"
    if training_stage == "frozen":
        stage_dir = root / "Frozen_LLM_Stage"
        legacy_dir = root / "1_Frozen_LLM_Stage"
    else:
        stage_dir = root / "Unfrozen_LoRA_Stage"
        legacy_dir = root / "2_Unfrozen_LoRA_Stage"
    stage_dir.mkdir(parents=True, exist_ok=True)

    if legacy_dir.exists() and legacy_dir != stage_dir:
        for child in legacy_dir.iterdir():
            dst = stage_dir / child.name
            if dst.exists():
                continue
            shutil.move(str(child), str(dst))
        try:
            legacy_dir.rmdir()
        except OSError:
            pass

    return stage_dir


def get_epoch_folder_name(encoder_type, training_stage, epoch):
    if training_stage == "frozen":
        return f"{encoder_type}_epoch_{epoch}"
    return f"{encoder_type}_lora_epoch_{epoch}"


def extract_trainable_state_dict(model):
    trainable = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable[name] = param.detach().cpu()
    return trainable


def maybe_save_lora_adapter(model, output_dir):
    has_peft = hasattr(model.llm, "peft_config") and bool(model.llm.peft_config)
    can_save = hasattr(model.llm, "save_pretrained")

    if has_peft and can_save:
        model.llm.save_pretrained(output_dir)
        print("[INFO] 已导出 LoRA adapter 文件。")
    else:
        print("[INFO] 当前不是 PEFT/LoRA 模型，跳过 adapter 文件导出。")


def load_tokenizer(model_path):
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


def build_preview_prompt(tokenizer, ts_tensor, force_detail_level=None):
    prompt_messages, prompt_meta = build_adaptive_prompt_messages(
        ts_tensor=ts_tensor,
        force_detail_level=force_detail_level,
    )
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
            return text

    if token_ids.numel() > prompt_token_len:
        new_token_ids = token_ids[prompt_token_len:]
        text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        if text:
            return text

    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def run_inference_preview(model, tokenizer, dataset, device, args, compute_dtype, use_amp):
    if len(dataset.samples) == 0:
        print("[WARN] 训练集为空，跳过推理预览。")
        return

    preview_index = max(0, min(args.preview_index, len(dataset.samples) - 1))
    sample = dataset.samples[preview_index]
    prompt_ts_tensor = torch.tensor(sample["time_series"], dtype=torch.float32).flatten()
    if prompt_ts_tensor.numel() > args.ts_seq_len:
        prompt_ts_tensor = prompt_ts_tensor[: args.ts_seq_len]
    ts_tensor = dataset._normalize_ts(sample["time_series"]).unsqueeze(0).to(device)

    force_detail_level = None if args.preview_force_detail_level == "auto" else args.preview_force_detail_level
    prompt_text, prompt_meta = build_preview_prompt(
        tokenizer=tokenizer,
        ts_tensor=prompt_ts_tensor,
        force_detail_level=force_detail_level,
    )
    prompt_enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = prompt_enc["input_ids"].to(device)
    attention_mask = prompt_enc["attention_mask"].to(device)

    auto_min_tokens = recommended_min_new_tokens(prompt_meta["detail_level"])
    if int(args.preview_min_new_tokens) >= 0:
        min_new_tokens = int(args.preview_min_new_tokens)
    else:
        min_new_tokens = auto_min_tokens
    min_new_tokens = min(min_new_tokens, int(args.preview_max_new_tokens))

    if use_amp and device.type == "cuda":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=True)
    else:
        autocast_ctx = contextlib.nullcontext()

    gen_kwargs = {
        "min_new_tokens": min_new_tokens,
        "max_new_tokens": args.preview_max_new_tokens,
        "do_sample": args.preview_do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.preview_do_sample:
        gen_kwargs["temperature"] = args.preview_temperature
        gen_kwargs["top_p"] = args.preview_top_p

    model.eval()
    with torch.no_grad(), autocast_ctx:
        text_only_ids = model.llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

        generated_ids = model.generate(
            ts_data=ts_tensor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

    text_only_preview = decode_generated_text(
        tokenizer=tokenizer,
        generated_ids=text_only_ids,
        prompt_token_len=input_ids.size(1),
        extra_prefix_tokens=0,
    )

    preview_text = decode_generated_text(
        tokenizer=tokenizer,
        generated_ids=generated_ids,
        prompt_token_len=input_ids.size(1),
        extra_prefix_tokens=args.soft_prompt_len,
    )

    print("[INFER] ===== 训练后推理预览 =====")
    print(f"[INFER] sample_index: {preview_index}")
    print(
        "[INFER] detail_level: "
        f"{prompt_meta['detail_level']} "
        f"(complexity={prompt_meta['complexity_score']:.3f}, min_new_tokens={min_new_tokens}, max_new_tokens={args.preview_max_new_tokens})"
    )
    print(f"[INFER] 原始标注: {str(sample['caption'])}")
    print(f"[INFER] 纯文本基座输出: {text_only_preview if text_only_preview else '[空输出]'}")
    print(f"[INFER] 模型输出: {preview_text if preview_text else '[空输出]'}")

def main():
    parser = argparse.ArgumentParser(description="多模态时序大模型微调主控台")
    parser.add_argument("--encoder_type", type=str, default="cnn", choices=["cnn", "mlp", "patchtst"])
    train_dir = Path(__file__).resolve().parent
    project_root = train_dir.parent
    parser.add_argument("--model_path", type=str, default=str(Path("Models") / "Qwen3-0.6B-Instruct-2512"))
    parser.add_argument(
        "--train_jsonl",
        type=str,
        default=None,
        help="训练集 JSONL 文件路径（相对项目根目录）。不指定时将从 sample_run_dir 自动合并生成。",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder_lr", type=float, default=None, help="编码器学习率，默认跟随 --lr")
    parser.add_argument("--projection_lr", type=float, default=None, help="投影与桥接模块学习率，默认=3*--lr")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_text_length", type=int, default=512)
    parser.add_argument("--ts_seq_len", type=int, default=512)
    parser.add_argument("--soft_prompt_len", type=int, default=4)
    parser.add_argument("--use_gating", type=str2bool, default=True)
    parser.add_argument("--use_bridge_norm", type=str2bool, default=True)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--preview_after_train", type=str2bool, default=True, help="训练结束后打印一条模型生成结果")
    parser.add_argument("--preview_index", type=int, default=0, help="推理预览样本索引")
    parser.add_argument("--preview_max_new_tokens", type=int, default=96)
    parser.add_argument("--preview_min_new_tokens", type=int, default=-1, help="-1 表示按复杂度自动设置")
    parser.add_argument("--preview_do_sample", type=str2bool, default=False)
    parser.add_argument("--preview_temperature", type=float, default=0.7)
    parser.add_argument("--preview_top_p", type=float, default=0.9)
    parser.add_argument(
        "--preview_force_detail_level",
        type=str,
        choices=["auto", "brief", "standard", "detailed"],
        default="auto",
        help="训练后预览时强制详细度；auto 表示由序列复杂度自动判定",
    )
    parser.add_argument(
        "--training_stage",
        type=str,
        default="frozen",
        choices=["frozen", "lora"],
        help="训练阶段：frozen=冻结LLM阶段，lora=解冻LoRA阶段",
    )
    parser.add_argument(
        "--sample_run_dir",
        type=str,
        default=str(Path("Sample") / "run_300k_20260413"),
        help="样本运行目录（包含 anomaly_detection/classification/prediction 子目录）",
    )
    parser.add_argument(
        "--combined_jsonl_name",
        type=str,
        default="combined_jsonl.jsonl",
        help="自动合并后的 JSONL 文件名",
    )
    args = parser.parse_args()

    project_root = Path(project_root).resolve()
    train_dir = Path(train_dir).resolve()

    model_path = resolve_relative_path(args.model_path, project_root)
    print(f"[INFO] 模型路径: {model_path}")
    print(f"[INFO] 编码器类型: {args.encoder_type}")
    print(f"[INFO] 训练阶段: {args.training_stage}")
    print(f"[INFO] 精度模式: {args.precision}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 训练设备: {device}")

    precision_mode, compute_dtype, use_amp, use_grad_scaler = resolve_precision(device, args.precision)
    print(f"[INFO] 精度策略: {precision_mode} ({compute_dtype}), AMP={use_amp}, GradScaler={use_grad_scaler}")

    if args.train_jsonl:
        train_jsonl_path = resolve_relative_path(args.train_jsonl, project_root)
        merge_info = None
    else:
        sample_run_dir = resolve_relative_path(args.sample_run_dir, project_root)
        combined_jsonl_name = args.combined_jsonl_name
        train_jsonl_path = sample_run_dir / combined_jsonl_name
        print(f"[INFO] 未指定训练集，自动从样本目录合并: {sample_run_dir}")
        merge_info = merge_filtered_samples_jsonl(sample_run_dir, train_jsonl_path)
        print(f"[INFO] 合并完成: {merge_info['written_lines']} 条有效记录，{merge_info['invalid_lines']} 条无效（已跳过）")

    print(f"[INFO] 加载分词器: {model_path}")
    tokenizer = load_tokenizer(model_path)

    print(f"[INFO] 加载数据集: {train_jsonl_path}")
    dataset = JsonlTimeSeriesDataset(
        jsonl_path=str(train_jsonl_path),
        tokenizer=tokenizer,
        max_text_length=args.max_text_length,
        ts_seq_len=args.ts_seq_len,
        random_crop=True,
    )
    print(f"[INFO] 训练样本数: {len(dataset)}")

    collate_fn = build_collate_fn(pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    print(f"[INFO] 初始化编码器: {args.encoder_type}")
    if args.encoder_type == "cnn":
        encoder = CNN1DEncoder()
    elif args.encoder_type == "mlp":
        encoder = MLPEncoder(seq_len=args.ts_seq_len)
    elif args.encoder_type == "patchtst":
        encoder = PatchTSTEncoder(seq_len=args.ts_seq_len)
    else:
        raise ValueError(f"未知编码器类型: {args.encoder_type}")

    model = MultimodalQwen(
        encoder=encoder,
        qwen_model_path=str(model_path),
        soft_prompt_len=args.soft_prompt_len,
        use_gating=args.use_gating,
        use_bridge_norm=args.use_bridge_norm,
        llm_dtype=compute_dtype,
    )
    model = model.to(device)

    encoder_lr = args.encoder_lr or args.lr
    projection_lr = args.projection_lr or (args.lr * 3)

    encoder_params = list(model.encoder.parameters())
    projection_params = list(model.projection.parameters()) + list(model.pre_norm.parameters()) + list(model.post_norm.parameters())
    if model.gate_proj is not None:
        projection_params += list(model.gate_proj.parameters())

    optimizer = optim.AdamW(
        [
            {"params": encoder_params, "lr": encoder_lr, "weight_decay": args.weight_decay},
            {"params": projection_params, "lr": projection_lr, "weight_decay": args.weight_decay},
        ],
    )

    total_steps = len(dataloader) * args.epochs
    print(f"[INFO] 总训练步数: {total_steps} (epochs={args.epochs}, batches_per_epoch={len(dataloader)})")

    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    checkpoint_stage_dir = build_checkpoint_stage_dir(train_dir, args.training_stage)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        epoch_pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")

        for step, batch in enumerate(epoch_pbar, start=1):
            ts_data = batch["ts_data"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            if use_amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=compute_dtype):
                    outputs = model(
                        ts_data=ts_data,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(
                    ts_data=ts_data,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / step

            epoch_pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg": f"{avg_loss:.4f}",
                "step": f"{step}/{len(dataloader)}",
            })

        epoch_loss = total_loss / len(dataloader)
        print(f"[EPOCH {epoch}] 平均损失: {epoch_loss:.4f}")

        epoch_folder = get_epoch_folder_name(args.encoder_type, args.training_stage, epoch)
        epoch_dir = checkpoint_stage_dir / epoch_folder
        epoch_dir.mkdir(parents=True, exist_ok=True)

        save_path = epoch_dir / "custom_ts_weights.pth"
        torch.save(extract_trainable_state_dict(model), save_path)
        print(f"[CHECKPOINT] 已保存: {save_path}")

        if args.preview_after_train and epoch == args.epochs:
            print("[INFER] 开始训练后推理预览...")
            run_inference_preview(model, tokenizer, dataset, device, args, compute_dtype, use_amp)

    print("[DONE] 训练流程全部完成。")


if __name__ == "__main__":
    main()
