# ==============================
# 服务器续跑命令（30w 配置）速查
# ==============================
#
# 统一说明:
# 1) 推荐把 PY 替换成你服务器上的 Python 命令（如 python3 或虚拟环境 python）。
# 2) apply_300k_profile=1 会自动启用 auto_resume/auto_regen_on_exhausted 等大规模预设。
# 3) 若继续同一批次，output_base_dir/report_path/score_cache_path 必须指向同一目录。
#
# 场景A: 常规断点续跑（已有输出与描述池，最快进入评审）
# python -u workspace/TS_LLM_Project/Train/generate_filtered_samples.py --apply_300k_profile 1 --target_qualified 300000 --output_base_dir workspace/TS_LLM_Project/Sample/run_300k_20260413 --report_path workspace/TS_LLM_Project/Sample/run_300k_20260413/generation_report.json --score_cache_path workspace/TS_LLM_Project/Sample/run_300k_20260413/score_cache.json --skip_description_generation --auto_resume 1 --clean_output 0 --max_regen_rounds 12
# python -u workspace/TS_LLM_Project/Train/generate_filtered_samples.py --apply_300k_profile 1 --target_qualified 300000 --output_base_dir workspace/TS_LLM_Project/Sample/run_300k_20260413 --report_path workspace/TS_LLM_Project/Sample/run_300k_20260413/generation_report.json --score_cache_path workspace/TS_LLM_Project/Sample/run_300k_20260413/score_cache.json --auto_resume 1 --clean_output 0 --enable_window_resample 0 --auto_regen_on_exhausted 1 --max_regen_rounds 30 --samples_per_dataset 600 --description_runs_per_dataset 8 --seed $(( (RANDOM<<15) | RANDOM ))
# python -u workspace/TS_LLM_Project/Train/generate_filtered_samples.py --apply_300k_profile 1 --target_qualified 300000 --output_base_dir workspace/TS_LLM_Project/Sample/run_300k_20260413 --report_path workspace/TS_LLM_Project/Sample/run_300k_20260413/generation_report.json --score_cache_path workspace/TS_LLM_Project/Sample/run_300k_20260413/score_cache.json --skip_description_generation --auto_resume 1 --clean_output 0 --enable_window_resample 0 --auto_regen_on_exhausted 1 --max_regen_rounds 30 --samples_per_dataset 600 --description_runs_per_dataset 8 --seed $(( (RANDOM<<15) | RANDOM ))
#
# 场景B: 描述池缺失/被清空（例如 *_descriptions.jsonl 不在了）
#   续跑命令与场景A基本相同，但去掉 --skip_description_generation，先补描述池再评审:
#   PY -u Train/generate_filtered_samples.py \
#     --apply_300k_profile 1 \
#     --target_qualified 300000 \
#     --output_base_dir Sample/run_300k_20260413 \
#     --report_path Sample/run_300k_20260413/generation_report.json \
#     --score_cache_path Sample/run_300k_20260413/score_cache.json \
#     --auto_resume 1 \
#     --clean_output 0 \
#     --max_regen_rounds 12
#
# 其他常见情况:
# 1) 想重新开新实验（不接历史）: 改新的 output_base_dir，并使用 --clean_output 1。
# 2) 从本机迁移到服务器: 需同步 output_base_dir 下历史结果与 score_cache.json，否则无法真正续跑。

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import inspect
import json
import math
import random
import re
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


TRAIN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAIN_DIR.parent
DEFAULT_DATASETS_ROOT = PROJECT_ROOT / "Dataset"
DEFAULT_OUTPUT_BASE_DIR = PROJECT_ROOT / "Sample"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "Sample" / "generation_report.json"
DEFAULT_SCORE_CACHE_PATH = PROJECT_ROOT / "Sample" / "score_cache.json"
DEFAULT_WINDOW_LENGTH_CANDIDATES = [24, 32, 48, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512]
DEFAULT_DESCRIPTION_RUNS_PER_DATASET = 3
DEFAULT_MAX_REGEN_ROUNDS = 8

DESCRIPTION_PICK_COUNTS: dict[str, dict[str, int]] = {}
DESCRIPTION_NUMERIC_CSV_COLS_CACHE: dict[str, list[str]] = {}
DESCRIPTION_NUMERIC_TXT_COLS_CACHE: dict[str, list[int]] = {}
DESCRIPTION_UEA_MAIN_CHANNELS_CACHE: dict[str, list[int]] = {}
DESCRIPTION_SCRIPT_MODULE_CACHE: dict[str, Any | None] = {}

HARDCODED_MULTIVARIATE_GROUPS = {
    "ETT-small",
    "Traffic",
    "Weather",
}
HARDCODED_SINGLE_TARGET_MULTIDIM_GROUPS = {
    "ElectricityECL",
    "Exchange_Rate",
}
UEA_UCR_GROUP_NAME = "UEA&UCR_Multivariate_Time_Series_Classification_Archive"

# Import scoring function from existing review script.
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))
import run_analysis  # noqa: E402
import dataset_type_config  # noqa: E402


MODEL_SCORE_THRESHOLD_OVERRIDES = {
    "doubao::doubao-seed-2-0-lite-260215::doubao_seed_20_lite_260215": 75,
    "doubao::doubao-seed-2-0-pro-260215::doubao_seed_20_pro_260215": 75,
    "doubao::doubao-seed-2-0-mini-260215::doubao_seed_20_mini_260215": 75,
    "doubao_seed_20_lite_260215": 75,
    "doubao_seed_20_pro_260215": 75,
    "doubao_seed_20_mini_260215": 75,
    "huggingface::Qwen/Qwen3-VL-30B-A3B-Thinking::hf_qwen3_vl_30b_thinking": 72,
    "huggingface::Qwen/Qwen3-VL-8B-Instruct::hf_qwen3_vl_8b": 72,
}


def _resolve_score_threshold_for_model(default_threshold: int, model_meta: dict | None) -> int:
    base = int(default_threshold)
    if not isinstance(model_meta, dict):
        return base
    model_id = str(model_meta.get("model_id") or "").strip()
    if not model_id:
        return base
    override = MODEL_SCORE_THRESHOLD_OVERRIDES.get(model_id)
    if override is None:
        return base
    return int(override)


def _find_description_script(dataset_dir: Path) -> Path | None:
    for script in dataset_dir.glob("generate_descriptions_*.py"):
        return script
    return None


def _find_data_file(dataset_dir: Path) -> Path | None:
    """返回第一个找到的数据文件"""
    for pattern in ["*.csv", "*.tsf", "*.arff", "*.txt"]:
        files = list(dataset_dir.glob(pattern))
        if files:
            return files[0]
        # 搜索子目录
        files = list(dataset_dir.rglob(pattern))
        if files:
            return files[0]
    return None


def _find_all_data_files(dataset_dir: Path) -> list[Path]:
    """返回所有找到的数据文件"""
    def _is_valid_data_file(path_obj: Path) -> bool:
        ptxt = str(path_obj).replace("\\", "/")
        name = path_obj.name
        if "/__MACOSX/" in ptxt:
            return False
        if name.startswith("._"):
            return False
        try:
            if path_obj.is_file() and path_obj.stat().st_size <= 0:
                return False
        except Exception:
            return False
        return True

    all_files = []
    for pattern in ["*.csv", "*.tsf", "*.arff", "*.txt"]:
        files = list(dataset_dir.glob(pattern))
        all_files.extend([f for f in files if _is_valid_data_file(f)])
        # 搜索子目录
        files = list(dataset_dir.rglob(pattern))
        all_files.extend([f for f in files if _is_valid_data_file(f)])
    # 去重
    unique_files = []
    seen = set()
    for f in all_files:
        if str(f) not in seen:
            seen.add(str(f))
            unique_files.append(f)
    return unique_files


def _filter_data_files_by_dataset(dataset_name: str, files: list[Path]) -> list[Path]:
    """Keep only file types that match each dataset family's generation script."""
    name = str(dataset_name).lower()

    if "uea&ucr" in name:
        arff_files = [p for p in files if p.suffix.lower() == ".arff"]
        mv_arff_files = [
            p
            for p in arff_files
            if "/multivariate/" in _normalize_path_key(p)
        ]
        if mv_arff_files:
            arff_files = mv_arff_files
        if not arff_files:
            return list(files)

        train_arff_files = [p for p in arff_files if re.search(r"_train\.arff$", p.name, flags=re.IGNORECASE)]
        if not train_arff_files:
            train_arff_files = list(arff_files)

        # Prefer merged TRAIN files and avoid per-dimension files like Dimension1_TRAIN.arff.
        merged_train_files = [
            p
            for p in train_arff_files
            if not re.search(r"dimension\d+_train\.arff$", p.name, flags=re.IGNORECASE)
        ]
        return merged_train_files if merged_train_files else train_arff_files

    allowed_suffixes: set[str] | None = None
    if "monash" in name:
        allowed_suffixes = {".tsf"}
    elif "exchange" in name:
        allowed_suffixes = {".txt"}
    elif any(k in name for k in ["ett", "electricity", "traffic", "weather", "nab"]):
        allowed_suffixes = {".csv", ".txt"}

    if not allowed_suffixes:
        return list(files)

    picked = [p for p in files if p.suffix.lower() in allowed_suffixes]
    return picked if picked else list(files)


def _normalize_path_key(path_obj: Path) -> str:
    try:
        return str(path_obj.resolve()).replace("\\", "/").lower()
    except Exception:
        return str(path_obj).replace("\\", "/").lower()


def _balanced_pick_from_candidates(counter_key: str, candidates: list[str], fallback: str) -> str:
    cleaned = [str(c).strip() for c in candidates if str(c).strip()]
    if not cleaned:
        return str(fallback)

    counter = DESCRIPTION_PICK_COUNTS.setdefault(str(counter_key), {})
    min_count = min(int(counter.get(c, 0)) for c in cleaned)
    bucket = [c for c in cleaned if int(counter.get(c, 0)) == min_count]
    picked = random.choice(bucket)
    counter[picked] = int(counter.get(picked, 0)) + 1
    return picked


def _detect_numeric_csv_columns(csv_path: Path, max_rows: int = 400) -> list[str]:
    cache_key = _normalize_path_key(csv_path)
    cached = DESCRIPTION_NUMERIC_CSV_COLS_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)

    numeric_cols: list[str] = []
    try:
        with csv_path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
            head_sample = f.read(4096)
            f.seek(0)
            delimiter_counts = {
                ",": head_sample.count(","),
                "\t": head_sample.count("\t"),
                ";": head_sample.count(";"),
            }
            if any(delimiter_counts.values()):
                delimiter = max(delimiter_counts.items(), key=lambda x: x[1])[0]
            else:
                delimiter = ","

            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader, None)
            if not isinstance(header, list) or not header:
                DESCRIPTION_NUMERIC_CSV_COLS_CACHE[cache_key] = []
                return []

            n_cols = len(header)
            non_empty = [0] * n_cols
            numeric = [0] * n_cols

            for row_idx, row in enumerate(reader):
                if row_idx >= int(max_rows):
                    break
                if not isinstance(row, list):
                    continue

                if len(row) < n_cols:
                    row = row + [""] * (n_cols - len(row))
                elif len(row) > n_cols:
                    row = row[:n_cols]

                for col_idx, raw_val in enumerate(row):
                    txt = str(raw_val).strip()
                    if not txt or txt.lower() in {"nan", "na", "null", "none"}:
                        continue
                    non_empty[col_idx] += 1
                    if _safe_float(txt) is not None:
                        numeric[col_idx] += 1

            for col_idx, raw_name in enumerate(header):
                col_name = str(raw_name).strip()
                if not col_name:
                    continue
                seen = int(non_empty[col_idx])
                if seen <= 0:
                    continue
                if int(numeric[col_idx]) >= max(1, int(math.ceil(seen * 0.8))):
                    numeric_cols.append(col_name)
    except Exception:
        numeric_cols = []

    DESCRIPTION_NUMERIC_CSV_COLS_CACHE[cache_key] = list(numeric_cols)
    return list(numeric_cols)


def _detect_numeric_txt_column_indices(txt_path: Path, max_rows: int = 400) -> list[int]:
    cache_key = _normalize_path_key(txt_path)
    cached = DESCRIPTION_NUMERIC_TXT_COLS_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)

    n_cols = 0
    non_empty: list[int] = []
    numeric: list[int] = []
    rows_seen = 0
    try:
        with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                row = str(line).strip()
                if not row:
                    continue
                parts = row.split(",") if "," in row else re.split(r"\s+", row)
                parts = [str(p).strip() for p in parts if str(p).strip()]
                if not parts:
                    continue

                if n_cols <= 0:
                    n_cols = len(parts)
                    non_empty = [0] * n_cols
                    numeric = [0] * n_cols

                if len(parts) != n_cols:
                    continue

                for idx, token in enumerate(parts):
                    if token.lower() in {"nan", "na", "null", "none"}:
                        continue
                    non_empty[idx] += 1
                    if _safe_float(token) is not None:
                        numeric[idx] += 1

                rows_seen += 1
                if rows_seen >= int(max_rows):
                    break
    except Exception:
        DESCRIPTION_NUMERIC_TXT_COLS_CACHE[cache_key] = []
        return []

    candidates: list[int] = []
    for idx in range(n_cols):
        seen = int(non_empty[idx]) if idx < len(non_empty) else 0
        if seen <= 0:
            continue
        num = int(numeric[idx]) if idx < len(numeric) else 0
        if num >= max(1, int(math.ceil(seen * 0.8))):
            candidates.append(int(idx))

    DESCRIPTION_NUMERIC_TXT_COLS_CACHE[cache_key] = list(candidates)
    return list(candidates)


def _load_script_module(script_path: Path) -> Any | None:
    cache_key = _normalize_path_key(script_path)
    if cache_key in DESCRIPTION_SCRIPT_MODULE_CACHE:
        return DESCRIPTION_SCRIPT_MODULE_CACHE[cache_key]

    try:
        spec = importlib.util.spec_from_file_location(
            f"_desc_gen_{abs(hash(cache_key))}",
            str(script_path),
        )
        if spec is None or spec.loader is None:
            DESCRIPTION_SCRIPT_MODULE_CACHE[cache_key] = None
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        DESCRIPTION_SCRIPT_MODULE_CACHE[cache_key] = module
        return module
    except Exception:
        DESCRIPTION_SCRIPT_MODULE_CACHE[cache_key] = None
        return None


def _infer_uea_channels_from_relational_arff(data_file: Path, max_chars: int = 400000) -> list[int] | None:
    """Fallback parser for UEA relational ARFF files with quoted multi-line channel blocks."""
    try:
        text = data_file.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    if len(text) > int(max_chars):
        text = text[: int(max_chars)]

    lower = text.lower()
    data_pos = lower.find("@data")
    if data_pos < 0:
        return None

    payload = text[data_pos + len("@data") :]
    first_quote_pos = -1
    quote_char = ""
    for qc in ("'", '"'):
        qpos = payload.find(qc)
        if qpos >= 0 and (first_quote_pos < 0 or qpos < first_quote_pos):
            first_quote_pos = qpos
            quote_char = qc

    if first_quote_pos < 0 or not quote_char:
        return None

    end_pos = payload.find(f"{quote_char},", first_quote_pos + 1)
    if end_pos < 0:
        end_pos = payload.find(quote_char, first_quote_pos + 1)
    if end_pos < 0:
        return None

    block = payload[first_quote_pos + 1 : end_pos]
    lines = [ln for ln in block.splitlines() if str(ln).strip()]
    n_channels = len(lines)
    if n_channels <= 1:
        return None

    return list(range(int(n_channels)))


def _infer_uea_main_channel_candidates(script_path: Path, data_file: Path) -> list[int]:
    cache_key = _normalize_path_key(data_file)
    cached = DESCRIPTION_UEA_MAIN_CHANNELS_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)

    candidates = [0]
    module = _load_script_module(script_path)
    load_ucr_arff = getattr(module, "load_ucr_arff", None) if module is not None else None
    if callable(load_ucr_arff):
        try:
            loaded = load_ucr_arff(str(data_file))
            if isinstance(loaded, tuple) and len(loaded) >= 5:
                is_multivariate = bool(loaded[3])
                n_channels = max(1, int(loaded[4]))
                if is_multivariate and n_channels > 1:
                    candidates = list(range(n_channels))
                else:
                    candidates = [0]
        except Exception:
            candidates = [0]

    if len(candidates) <= 1:
        fallback_candidates = _infer_uea_channels_from_relational_arff(data_file)
        if fallback_candidates:
            candidates = list(fallback_candidates)

    cleaned = sorted({int(c) for c in candidates if int(c) >= 0})
    if not cleaned:
        cleaned = [0]

    DESCRIPTION_UEA_MAIN_CHANNELS_CACHE[cache_key] = list(cleaned)
    return list(cleaned)


def _pick_uea_main_channel(script_path: Path, data_file: Path) -> int:
    candidates = _infer_uea_main_channel_candidates(script_path=script_path, data_file=data_file)
    picked = _balanced_pick_from_candidates(
        counter_key=f"uea::{_normalize_path_key(data_file)}::main_channel",
        candidates=[str(int(c)) for c in candidates],
        fallback="0",
    )
    try:
        return int(picked)
    except Exception:
        return 0


def _pick_numeric_target_col_from_csv(csv_path: Path, dataset_name: str, fallback: str) -> str:
    numeric_cols = _detect_numeric_csv_columns(csv_path)
    ds_name = str(dataset_name).lower()

    if "electricity" in ds_name:
        meter_cols = [c for c in numeric_cols if re.fullmatch(r"MT_\d+", str(c))]
        if meter_cols:
            numeric_cols = meter_cols

    if not numeric_cols:
        return str(fallback)

    return _balanced_pick_from_candidates(
        counter_key=f"csv::{ds_name}::{_normalize_path_key(csv_path)}::target_col",
        candidates=[str(c) for c in numeric_cols],
        fallback=str(fallback),
    )


def _pick_exchange_target_col(txt_path: Path, fallback: str = "c0") -> str:
    """Choose a balanced random Exchange_Rate target column like c0/c1/..."""
    numeric_indices = _detect_numeric_txt_column_indices(txt_path)

    if not numeric_indices:
        try:
            with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    row = str(line).strip()
                    if not row:
                        continue
                    parts = row.split(",") if "," in row else re.split(r"\s+", row)
                    parts = [str(p).strip() for p in parts if str(p).strip()]
                    if parts:
                        numeric_indices = list(range(len(parts)))
                    break
        except Exception:
            numeric_indices = []

    if not numeric_indices:
        return str(fallback)

    return _balanced_pick_from_candidates(
        counter_key=f"exchange::{_normalize_path_key(txt_path)}::target_col",
        candidates=[f"c{int(i)}" for i in numeric_indices],
        fallback=str(fallback),
    )


def _run_description_generation(
    script_path: Path,
    data_file: Path,
    max_samples: int = 100,
    window_lengths: list[int] | None = None,
) -> Path | None:
    if window_lengths is None:
        window_lengths = [256, 512]
    
    import subprocess
    import sys
    import os
    import random
    
    cmd = [
        sys.executable,
        str(script_path),
    ]
    
    dataset_name = script_path.parent.name
    
    # 根据不同数据集添加参数
    if "UEA&UCR" in dataset_name:
        cmd.extend(["--arff_path", str(data_file)])
        main_channel = _pick_uea_main_channel(script_path=script_path, data_file=Path(data_file))
        cmd.extend(["--main_channels", str(main_channel)])
    elif "Monash" in dataset_name:
        cmd.extend(["--tsf_path", str(data_file)])
    elif "Exchange" in dataset_name or "Exchange_Rate" in dataset_name:
        cmd.extend(["--txt_path", str(data_file)])
        target_col = _pick_exchange_target_col(Path(data_file), fallback="c0")
        cmd.extend(["--target_col", str(target_col)])
    else:
        # 其他数据集用 csv_path
        cmd.extend(["--csv_path", str(data_file)])
        if "ETT" in dataset_name:
            target_col = _pick_numeric_target_col_from_csv(Path(data_file), dataset_name, fallback="OT")
            cmd.extend(["--target_col", str(target_col)])
        elif "Weather" in dataset_name:
            target_col = _pick_numeric_target_col_from_csv(Path(data_file), dataset_name, fallback="T (degC)")
            cmd.extend(["--target_col", str(target_col)])
        elif "Electricity" in dataset_name:
            target_col = _pick_numeric_target_col_from_csv(Path(data_file), dataset_name, fallback="MT_001")
            cmd.extend(["--target_col", str(target_col)])
        elif "Traffic" in dataset_name:
            target_col = _pick_numeric_target_col_from_csv(Path(data_file), dataset_name, fallback="traffic_volume")
            cmd.extend(["--target_col", str(target_col)])
        else:
            target_col = _pick_numeric_target_col_from_csv(Path(data_file), dataset_name, fallback="value")
            cmd.extend(["--target_col", str(target_col)])
    
    # 随机选择窗口长度
    if window_lengths:
        win_len = random.choice(window_lengths)
        cmd.extend(["--window_lengths", str(win_len)])
    else:
        cmd.extend(["--window_lengths", "256", "512"])
    
    cmd.extend(["--max_samples", str(max_samples)])
    cmd.extend(["--step_ratio", "0.3"])
    
    print(f"[描述生成] 运行: {' '.join(cmd)}")
    
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            cmd,
            cwd=str(script_path.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=600,
            env=env,
        )
        
        if result.returncode != 0:
            print(f"[描述生成] 错误: {result.stderr}")
            return None
        
        print(f"[描述生成] 成功: {result.stdout}")
        
        output_dir = script_path.parent
        jsonl_files = list(output_dir.rglob("*_descriptions.jsonl"))
        if jsonl_files:
            latest = max(jsonl_files, key=lambda p: p.stat().st_mtime)
            return latest
        
        return None
        
    except Exception as e:
        print(f"[描述生成] 异常: {e}")
        return None


def _generate_descriptions_round(
    datasets_root: Path,
    samples_per_dataset: int,
    window_lengths: list[int],
    runs_per_dataset: int,
    round_idx: int,
) -> dict[str, int]:
    """Generate one outer-round description batch across all dataset folders."""
    stats = {
        "round": int(round_idx),
        "dataset_dirs": 0,
        "runs": 0,
        "successes": 0,
        "failures": 0,
        "skipped": 0,
    }
    dataset_dirs = [
        d
        for d in datasets_root.iterdir()
        if d.is_dir() and (not d.name.startswith(".")) and (not d.name.startswith("__"))
    ]
    random.shuffle(dataset_dirs)
    stats["dataset_dirs"] = len(dataset_dirs)

    print(
        f"[描述生成] 外层轮次 round={round_idx} start "
        f"datasets={len(dataset_dirs)} runs_per_dataset={max(1, int(runs_per_dataset))}"
    )

    for dataset_dir in dataset_dirs:
        print(f"[描述生成] 检查目录: {dataset_dir.name}")
        desc_script = _find_description_script(dataset_dir)
        if not desc_script:
            stats["skipped"] += 1
            print(f"[描述生成] 跳过 {dataset_dir.name}（未找到描述生成脚本）")
            continue

        all_data_files = _find_all_data_files(dataset_dir)
        all_data_files = _filter_data_files_by_dataset(dataset_dir.name, all_data_files)
        if not all_data_files:
            stats["skipped"] += 1
            print(f"[描述生成] 跳过 {dataset_dir.name}（未找到数据文件）")
            continue

        for i in range(max(1, int(runs_per_dataset))):
            stats["runs"] += 1
            data_file = random.choice(all_data_files)
            print(f"[描述生成] 为 {dataset_dir.name} 生成描述 (轮次 {round_idx}, 第 {i+1}/{runs_per_dataset} 次)...")
            jsonl_path = _run_description_generation(
                script_path=desc_script,
                data_file=data_file,
                max_samples=samples_per_dataset,
                window_lengths=window_lengths,
            )
            if jsonl_path:
                stats["successes"] += 1
                print(f"[描述生成] 成功: {jsonl_path}")
            else:
                stats["failures"] += 1
                print(f"[描述生成] 失败: {dataset_dir.name}")

    print(
        "[描述生成] 外层轮次完成 "
        f"round={round_idx} runs={stats['runs']} successes={stats['successes']} "
        f"failures={stats['failures']} skipped={stats['skipped']}"
    )
    return stats


def _infer_dimension_mode_with_hardcode(
    dataset_group: str,
    source_path: Path,
    sample_obj: dict[str, Any] | None,
) -> tuple[int, str]:
    """Apply user-defined hardcoded dimensionality rules with safe fallback."""
    inferred_channels, _ = _infer_sample_dimension_mode(sample_obj or {})
    group = str(dataset_group)
    path_lower = str(source_path).replace("\\", "/").lower()

    if group in HARDCODED_SINGLE_TARGET_MULTIDIM_GROUPS:
        return max(2, int(inferred_channels)), "single_target_multidim"

    if group in HARDCODED_MULTIVARIATE_GROUPS:
        return max(2, int(inferred_channels)), "multivariate"

    if group == UEA_UCR_GROUP_NAME:
        if "/multivariate/" in path_lower:
            return max(2, int(inferred_channels)), "multivariate"
        if int(inferred_channels) > 1:
            return max(2, int(inferred_channels)), "multivariate"
        return 1, "univariate_1d"

    return 1, "univariate_1d"


def _sanitize_filename(text: str, max_len: int = 120) -> str:
    s = str(text)
    s = re.sub(r"[^0-9A-Za-z_.()-]+", "_", s)
    s = s.strip("._")
    if not s:
        s = "unknown"
    if len(s) > max_len:
        s = s[:max_len]
    return s


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _count_nonempty_lines(path: Path) -> int:
    cnt = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cnt += 1
    return cnt


def _is_augmented_source_id(source_id: Any) -> bool:
    sid = str(source_id or "").strip()
    return sid.startswith("augSRC")


def _load_records_from_jsonl(path: Path, skip_augmented_source_ids: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            txt = line.strip()
            if not txt:
                continue
            try:
                obj = json.loads(txt)
            except Exception:
                continue
            if isinstance(obj, dict):
                if skip_augmented_source_ids:
                    rec_id = obj.get("id")
                    if rec_id is None:
                        rec_id = obj.get("source_id")
                    if _is_augmented_source_id(rec_id):
                        continue
                records.append(obj)
    return records


def _infer_sample_dimension_mode(sample: dict[str, Any]) -> tuple[int, str]:
    """Infer channel count and dimension mode from one sample record."""
    matrix = _extract_values_matrix(sample)
    if not matrix:
        return 1, "univariate_1d"
    n_channels = len(matrix[0]) if matrix and matrix[0] else 1
    if n_channels <= 1:
        return 1, "univariate_1d"

    target_cols = sample.get("target_cols")
    if isinstance(target_cols, list) and len(target_cols) > 1:
        return int(n_channels), "multivariate"
    return int(n_channels), "single_target_multidim"


def _parse_window_length_candidates(raw_value: str | list[int] | None) -> list[int]:
    if isinstance(raw_value, list):
        out = [int(v) for v in raw_value if int(v) > 0]
    else:
        txt = str(raw_value).strip() if raw_value is not None else ""
        if not txt:
            out = list(DEFAULT_WINDOW_LENGTH_CANDIDATES)
        else:
            out = []
            for part in re.split(r"[\s,;]+", txt):
                if not part:
                    continue
                try:
                    v = int(part)
                except Exception:
                    continue
                if v > 0:
                    out.append(v)

    dedup = sorted(set(out))
    return dedup if dedup else list(DEFAULT_WINDOW_LENGTH_CANDIDATES)


def _extract_values_matrix(sample: dict[str, Any]) -> list[list[float]] | None:
    values = sample.get("values")
    if isinstance(values, list) and values:
        first = values[0]
        if isinstance(first, list):
            n_cols = len(first)
            if n_cols <= 0:
                return None
            matrix: list[list[float]] = []
            for row in values:
                if not isinstance(row, list) or len(row) != n_cols:
                    return None
                parsed_row: list[float] = []
                for v in row:
                    fv = _safe_float(v)
                    if fv is None:
                        return None
                    parsed_row.append(fv)
                matrix.append(parsed_row)
            return matrix if matrix else None

        series_1d: list[float] = []
        for v in values:
            fv = _safe_float(v)
            if fv is None:
                return None
            series_1d.append(fv)
        return [[v] for v in series_1d] if series_1d else None

    ts = sample.get("time_series")
    if isinstance(ts, list) and ts:
        series_1d: list[float] = []
        for v in ts:
            fv = _safe_float(v)
            if fv is None:
                return None
            series_1d.append(fv)
        return [[v] for v in series_1d] if series_1d else None

    return None


def _effective_window_lengths(
    seq_len: int,
    min_window_len: int,
    max_window_len: int,
    window_length_candidates: list[int],
) -> list[int]:
    lo = max(2, int(min_window_len))
    hi = max(lo, int(max_window_len))
    if seq_len < lo:
        return []

    lengths = [w for w in window_length_candidates if lo <= int(w) <= hi and int(w) <= seq_len]
    lengths = sorted(set(int(w) for w in lengths))
    if lengths:
        return lengths

    fallback = min(seq_len, hi)
    return [fallback] if fallback >= lo else []


def _choose_main_channel(source: "SourceInfo", n_channels: int, balance_main_channel: bool) -> int:
    if n_channels <= 1:
        source.main_channel_attempts[0] = source.main_channel_attempts.get(0, 0) + 1
        return 0

    if not balance_main_channel:
        picked = random.randint(0, n_channels - 1)
        source.main_channel_attempts[picked] = source.main_channel_attempts.get(picked, 0) + 1
        return picked

    counts = {i: int(source.main_channel_attempts.get(i, 0)) for i in range(n_channels)}
    min_count = min(counts.values())
    candidates = [i for i, c in counts.items() if c == min_count]
    picked = random.choice(candidates)
    source.main_channel_attempts[picked] = source.main_channel_attempts.get(picked, 0) + 1
    return picked


def _compute_main_channel_correlations(matrix: list[list[float]], main_idx: int) -> list[dict[str, Any]]:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] <= 1:
        return []

    n_cols = int(arr.shape[1])
    main = arr[:, main_idx]
    corr_items: list[dict[str, Any]] = []
    for j in range(n_cols):
        if j == main_idx:
            continue
        other = arr[:, j]
        if float(np.std(main)) < 1e-8 or float(np.std(other)) < 1e-8:
            rho = 0.0
        else:
            rho = float(np.corrcoef(main, other)[0, 1])
            if not math.isfinite(rho):
                rho = 0.0
        corr_items.append(
            {
                "channels": [int(main_idx), int(j)],
                "correlation": float(rho),
                "stable": bool(abs(rho) >= 0.12),
            }
        )
    return corr_items


def _build_simple_caption(main_series: list[float], main_channel: int, n_channels: int) -> str:
    arr = np.asarray(main_series, dtype=np.float32)
    if arr.size <= 1:
        trend = "平稳"
        vol = "较小"
    else:
        start = float(arr[0])
        end = float(arr[-1])
        std = float(np.std(arr))
        value_range = float(np.max(arr) - np.min(arr))

        trend_score = (end - start) / max(std, 1e-6)
        if trend_score > 0.6:
            trend = "上升"
        elif trend_score < -0.6:
            trend = "下降"
        else:
            trend = "平稳"

        vol_ratio = std / max(value_range, 1e-6)
        if vol_ratio < 0.18:
            vol = "较小"
        elif vol_ratio < 0.32:
            vol = "中等"
        else:
            vol = "较大"

    if n_channels > 1:
        return f"该窗口以主变量 ch{main_channel} 为主，整体{trend}，波动{vol}。"
    return f"该序列整体{trend}，波动{vol}。"


def _build_augmented_sample(
    source: "SourceInfo",
    base_sample: dict[str, Any],
    synthetic_idx: int,
    min_window_len: int,
    max_window_len: int,
    window_length_candidates: list[int],
    balance_main_channel: bool,
) -> tuple[dict[str, Any], int] | None:
    matrix = _extract_values_matrix(base_sample)
    if not matrix:
        return None

    seq_len = len(matrix)
    n_channels = len(matrix[0]) if matrix and matrix[0] else 0
    if n_channels <= 0:
        return None

    valid_lengths = _effective_window_lengths(
        seq_len=seq_len,
        min_window_len=min_window_len,
        max_window_len=max_window_len,
        window_length_candidates=window_length_candidates,
    )
    if not valid_lengths:
        return None

    picked_len = random.choice(valid_lengths)
    max_start = max(0, seq_len - picked_len)
    start_offset = random.randint(0, max_start) if max_start > 0 else 0

    window = matrix[start_offset : start_offset + picked_len]
    main_idx = _choose_main_channel(source, n_channels=n_channels, balance_main_channel=balance_main_channel)
    main_series = [row[main_idx] for row in window]
    caption = _build_simple_caption(main_series, main_channel=main_idx, n_channels=n_channels)
    corr_items = _compute_main_channel_correlations(window, main_idx=main_idx)

    sample = copy.deepcopy(base_sample)
    # Keep augmented id free of any extra "_L<length>" tokens.
    # Some visualization scripts infer length from id tokens and may read the last match.
    source_tag = abs(hash(str(source.source_path))) % 1000000
    sample["id"] = (
        f"augSRC{source_tag}_N{synthetic_idx:08d}_L{picked_len}_S{start_offset}_M{main_idx}"
    )

    # Keep time-like sequences aligned with the augmented window to avoid index/value mismatch.
    for time_key in ("time", "timestamps", "datetime", "date"):
        seq_obj = sample.get(time_key)
        if not isinstance(seq_obj, list):
            continue
        if len(seq_obj) >= (start_offset + picked_len):
            sample[time_key] = list(seq_obj[start_offset : start_offset + picked_len])
        else:
            sample.pop(time_key, None)

    if n_channels == 1:
        sample["values"] = [float(row[0]) for row in window]
        sample["time_series"] = [float(row[0]) for row in window]
        sample["main_channel"] = 0
        if not isinstance(sample.get("feature_names"), list) or len(sample.get("feature_names", [])) != 1:
            sample["feature_names"] = ["ch0"]
        sample["target_col"] = str(sample["feature_names"][0])
        sample["target_cols"] = [str(sample["feature_names"][0])]
    else:
        sample["values"] = [[float(v) for v in row] for row in window]
        sample["main_channel"] = int(main_idx)

        feature_names = sample.get("feature_names")
        if not isinstance(feature_names, list) or len(feature_names) != n_channels:
            feature_names = [f"ch{i}" for i in range(n_channels)]
        else:
            feature_names = [str(x) for x in feature_names]
        sample["feature_names"] = feature_names

        feature_names_chinese = sample.get("feature_names_chinese")
        if not isinstance(feature_names_chinese, list) or len(feature_names_chinese) != n_channels:
            feature_names_chinese = [str(x) for x in feature_names]
        else:
            feature_names_chinese = [str(x) for x in feature_names_chinese]
        sample["feature_names_chinese"] = feature_names_chinese

        # Augmented windows should truly switch the main variable.
        main_feature_name = str(feature_names[main_idx])
        sample["target_col"] = main_feature_name
        sample["target_cols"] = [main_feature_name]
        if 0 <= int(main_idx) < len(feature_names_chinese):
            sample["target_col_chinese"] = str(feature_names_chinese[int(main_idx)])

    features_obj = sample.get("features")
    if not isinstance(features_obj, dict):
        features_obj = {}
    features_obj["main_channel"] = int(main_idx)
    features_obj["cross_channel_correlations"] = corr_items
    sample["features"] = features_obj

    base_start = 0
    for k in ("start_index", "start"):
        v = sample.get(k)
        try:
            base_start = int(v)
            break
        except Exception:
            continue

    sample["window_length"] = int(picked_len)
    sample["start_index"] = int(base_start + start_offset)
    sample["end_index"] = int(base_start + start_offset + picked_len - 1)
    sample["start"] = int(sample["start_index"])
    sample["end"] = int(sample["end_index"])
    sample["descriptions"] = [caption]
    sample["caption"] = caption

    source.augmented_count += 1
    synthetic_line_idx = source.line_count + source.augmented_count
    return sample, synthetic_line_idx


def _build_augmented_sample_from_source(
    source: "SourceInfo",
    synthetic_idx: int,
    min_window_len: int,
    max_window_len: int,
    window_length_candidates: list[int],
    balance_main_channel: bool,
) -> tuple[dict[str, Any], int] | None:
    if not source.records:
        return None

    indices = list(range(len(source.records)))
    random.shuffle(indices)

    for idx in indices:
        out = _build_augmented_sample(
            source=source,
            base_sample=source.records[idx],
            synthetic_idx=synthetic_idx,
            min_window_len=min_window_len,
            max_window_len=max_window_len,
            window_length_candidates=window_length_candidates,
            balance_main_channel=balance_main_channel,
        )
        if out is not None:
            return out

    source.augmentable = False
    return None


def _estimate_theoretical_capacity(
    sources: list["SourceInfo"],
    min_window_len: int,
    max_window_len: int,
    window_length_candidates: list[int],
) -> dict[str, Any]:
    by_group: dict[str, int] = {}
    by_source: dict[str, int] = {}
    total_capacity = 0

    for src in sources:
        src_capacity = 0
        for sample in src.records:
            matrix = _extract_values_matrix(sample)
            if not matrix:
                continue
            seq_len = len(matrix)
            n_channels = len(matrix[0]) if matrix and matrix[0] else 0
            if n_channels <= 0:
                continue

            valid_lengths = _effective_window_lengths(
                seq_len=seq_len,
                min_window_len=min_window_len,
                max_window_len=max_window_len,
                window_length_candidates=window_length_candidates,
            )
            if not valid_lengths:
                continue

            base_windows = sum(max(1, seq_len - w + 1) for w in valid_lengths)
            src_capacity += int(base_windows * max(1, n_channels))

        by_source[str(src.source_path)] = int(src_capacity)
        by_group[src.dataset_group] = by_group.get(src.dataset_group, 0) + int(src_capacity)
        total_capacity += int(src_capacity)

    return {
        "total_theoretical_windows": int(total_capacity),
        "group_theoretical_windows": by_group,
        "source_theoretical_windows": by_source,
    }


def _load_score_cache(cache_path: Path) -> dict[str, int]:
    if not cache_path.exists():
        return {}
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}

    out: dict[str, int] = {}
    for k, v in raw.items():
        key = str(k).strip()
        if not key:
            continue
        if isinstance(v, bool):
            continue
        try:
            sv = int(v)
        except Exception:
            continue
        if 0 <= sv <= 100:
            out[key] = sv
    return out


def _save_score_cache(cache_path: Path, score_cache: dict[str, int]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(k): int(v) for k, v in score_cache.items() if isinstance(v, int) and 0 <= int(v) <= 100}
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_score_cache_key(source: "SourceInfo", sample_obj: dict[str, Any], line_idx: int) -> str:
    """Build a stable score-cache key for both original and augmented samples."""
    sid = str(sample_obj.get("id") or f"{source.source_path.stem}_line{int(line_idx)}")
    base_sid = sid.split("_aug_", 1)[0] if "_aug_" in sid else sid

    def _to_int(v: Any, default: int = -1) -> int:
        try:
            return int(v)
        except Exception:
            return int(default)

    if "_aug_" in sid:
        start_idx = _to_int(sample_obj.get("start_index", sample_obj.get("start", -1)))
        end_idx = _to_int(sample_obj.get("end_index", sample_obj.get("end", -1)))
        win_len = _to_int(sample_obj.get("window_length", -1))
        main_ch = _to_int(sample_obj.get("main_channel", 0), default=0)
        if start_idx >= 0 and end_idx >= 0 and win_len > 0:
            return f"{source.source_path}#aug|{base_sid}|S{start_idx}|E{end_idx}|L{win_len}|M{main_ch}"

    return f"{source.source_path}#line={int(line_idx)}|id={base_sid}"


def _build_resume_dedup_key(source_jsonl: str, source_id: str) -> str:
    """Recover a stable dedup key from persisted output records."""
    sid = str(source_id).strip()
    src = str(source_jsonl).strip()
    m = re.match(r"^(?P<base>.+?)_aug_\d+_L(?P<L>\d+)_S(?P<S>-?\d+)_M(?P<M>-?\d+)$", sid)
    if m is not None:
        base = str(m.group("base"))
        L = int(m.group("L"))
        S = int(m.group("S"))
        M = int(m.group("M"))
        if L > 0 and S >= 0:
            E = S + L - 1
            return f"{src}|{base}|S{S}|E{E}|L{L}|M{M}"
    return f"{src}|id={sid}"


def _load_existing_accept_state(
    type_output_jsonl: dict[str, Path],
    type_targets: dict[str, int],
    skip_augmented_source_ids: bool = False,
) -> dict[str, Any]:
    """Recover accepted counts and dedup keys from existing outputs for auto-resume."""
    accepted_total = 0
    type_accepted: dict[str, int] = {t: 0 for t in type_targets}
    type_accepted_count: dict[str, int] = {t: 0 for t in type_targets}
    resume_accept_keys: set[str] = set()
    resume_dedup_keys: set[str] = set()
    source_accept_counts: dict[str, int] = {}

    for type_name, jsonl_path in type_output_jsonl.items():
        if not jsonl_path.exists() or not jsonl_path.is_file():
            continue

        local_count = 0
        try:
            with jsonl_path.open("r", encoding="utf-8") as f:
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

                    source_id = str(obj.get("source_id") or "").strip()
                    source_jsonl = str(obj.get("source_jsonl") or "").strip()
                    if skip_augmented_source_ids and _is_augmented_source_id(source_id):
                        continue
                    if not source_id or not source_jsonl:
                        continue

                    resume_key = _build_resume_accept_key(source_jsonl=source_jsonl, source_id=source_id)
                    dedup_key = str(obj.get("dedup_key") or "").strip()
                    if not dedup_key:
                        dedup_key = _build_resume_dedup_key(source_jsonl=source_jsonl, source_id=source_id)

                    if dedup_key in resume_dedup_keys:
                        continue
                    if resume_key in resume_accept_keys:
                        continue

                    resume_accept_keys.add(resume_key)
                    resume_dedup_keys.add(dedup_key)
                    local_count += 1
                    source_accept_counts[source_jsonl] = source_accept_counts.get(source_jsonl, 0) + 1
        except Exception:
            continue

        type_accepted[type_name] = int(local_count)
        type_accepted_count[type_name] = int(local_count)
        accepted_total += int(local_count)

    return {
        "accepted_total": int(accepted_total),
        "type_accepted": type_accepted,
        "type_accepted_count": type_accepted_count,
        "resume_accept_keys": resume_accept_keys,
        "resume_dedup_keys": resume_dedup_keys,
        "source_accept_counts": source_accept_counts,
    }


def _allocate_targets_by_weight(weight_map: dict[str, int], total_target: int) -> dict[str, int]:
    total_target = max(0, int(total_target))
    if total_target == 0:
        return {k: 0 for k in weight_map}

    positive = {k: int(v) for k, v in weight_map.items() if int(v) > 0}
    if not positive:
        return {k: 0 for k in weight_map}

    num_types = len(positive)
    min_per_type = max(1, total_target // num_types)

    out = {k: 0 for k in weight_map}
    remaining = total_target

    for type_name in positive:
        out[type_name] = min_per_type
        remaining -= min_per_type

    if remaining > 0:
        ranked = sorted(
            positive.keys(),
            key=lambda k: (positive[k], k),
            reverse=True,
        )
        for i in range(remaining):
            out[ranked[i % len(ranked)]] += 1

    return out


def _weighted_choice(items: list[Any], weight_fn: Callable[[Any], float]) -> Any | None:
    if not items:
        return None

    weights = []
    total = 0.0
    for it in items:
        w = max(0.0, float(weight_fn(it)))
        weights.append(w)
        total += w

    if total <= 0:
        return random.choice(items)

    hit = random.uniform(0.0, total)
    acc = 0.0
    for it, w in zip(items, weights):
        acc += w
        if hit <= acc:
            return it
    return items[-1]


@dataclass
class SourceInfo:
    source_path: Path
    dataset_group: str
    subdataset_name: str
    viz_script: Path
    line_count: int
    dataset_type: str = field(default="")
    records: list[dict[str, Any]] = field(default_factory=list, repr=False)
    accepted_target: int = 0
    accepted_count: int = 0
    attempted_count: int = 0
    exhausted: bool = False
    augmentable: bool = True
    augmented_count: int = 0
    main_channel_attempts: dict[int, int] = field(default_factory=dict)
    channel_count_estimate: int = 1
    dimension_mode: str = "univariate_1d"
    file_mtime: float = 0.0

    _line_idx: int = field(default=-1, init=False, repr=False)

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def next_record(self) -> tuple[dict[str, Any], int] | None:
        if self.exhausted:
            return None

        next_idx = self._line_idx + 1
        if next_idx >= len(self.records):
            self.exhausted = True
            return None

        self._line_idx = next_idx
        return self.records[self._line_idx], self._line_idx

    @property
    def can_augment(self) -> bool:
        return bool(self.augmentable and self.records)

    @property
    def remaining_estimate(self) -> int:
        return max(0, int(self.line_count) - max(0, self._line_idx + 1))


@dataclass
class CandidateItem:
    source: SourceInfo
    sample_obj: dict[str, Any]
    line_idx: int
    sample_id: str
    caption: str
    ts: list[float]
    tmp_image_path: Path
    dataset_name: str
    score_cache_key: str
    cached_score: int | None = None


def _build_sample_dedup_key(
    source_path: Path,
    sample_obj: dict[str, Any],
    sample_id: str,
    line_idx: int,
) -> str:
    """Build a stable dedup key directly from source/sample fields."""
    sid = str(sample_obj.get("id") or sample_id or f"{source_path.stem}_line{int(line_idx)}")
    base_sid = sid.split("_aug_", 1)[0] if "_aug_" in sid else sid

    def _to_int(v: Any, default: int = -1) -> int:
        try:
            return int(v)
        except Exception:
            return int(default)

    start_idx = _to_int(sample_obj.get("start_index", sample_obj.get("start", -1)))
    end_idx = _to_int(sample_obj.get("end_index", sample_obj.get("end", -1)))
    win_len = _to_int(sample_obj.get("window_length", -1))
    main_ch = _to_int(sample_obj.get("main_channel", 0), default=0)

    if start_idx >= 0 and end_idx >= 0 and win_len > 0:
        return f"{source_path}|{base_sid}|S{start_idx}|E{end_idx}|L{win_len}|M{main_ch}"
    return f"{source_path}|id={base_sid}"


def _build_candidate_dedup_key(cand: CandidateItem) -> str:
    """Stable uniqueness key for accepted samples (original and augmented)."""
    return _build_sample_dedup_key(
        source_path=cand.source.source_path,
        sample_obj=cand.sample_obj,
        sample_id=cand.sample_id,
        line_idx=int(cand.line_idx),
    )


def _build_resume_accept_key(source_jsonl: str, source_id: str) -> str:
    return f"{str(source_jsonl).strip()}|{str(source_id).strip()}"


def _discover_sources(
    datasets_root: Path,
    skip_augmented_source_ids: bool = False,
) -> tuple[list[SourceInfo], dict[str, int]]:
    if not datasets_root.exists() or not datasets_root.is_dir():
        raise FileNotFoundError(f"datasets_root does not exist: {datasets_root}")

    import random
    sources: list[SourceInfo] = []
    type_line_counts: dict[str, int] = {}

    top_level_dirs = sorted(
        [p for p in datasets_root.iterdir() if p.is_dir() and not p.name.startswith(".")],
        key=lambda p: p.name.lower(),
    )

    for group_dir in top_level_dirs:
        viz_scripts = sorted(group_dir.glob("viz_*.py"), key=lambda p: p.name.lower())
        if not viz_scripts:
            continue
        viz_script = viz_scripts[0]

        # 获取数据集类型
        dataset_type = dataset_type_config.get_dataset_type(group_dir.name)
        
        # 如果返回 None，跳过这个文件夹
        if dataset_type is None:
            continue

        jsonl_files = sorted(group_dir.rglob("*_descriptions*.jsonl"), key=lambda p: str(p).lower())
        total_lines = 0
        for jf in jsonl_files:
            try:
                records = _load_records_from_jsonl(
                    jf,
                    skip_augmented_source_ids=skip_augmented_source_ids,
                )
            except Exception:
                continue
            line_count = len(records)
            if line_count <= 0:
                continue
            total_lines += line_count

            subdataset_name = jf.stem
            if "_descriptions" in subdataset_name:
                subdataset_name = subdataset_name.split("_descriptions", 1)[0]

            # 随机打乱每个描述文件的记录
            random.shuffle(records)
            ch_est, dim_mode = _infer_dimension_mode_with_hardcode(
                dataset_group=group_dir.name,
                source_path=jf,
                sample_obj=records[0] if records else None,
            )
            try:
                file_mtime = float(jf.stat().st_mtime)
            except Exception:
                file_mtime = 0.0

            sources.append(
                SourceInfo(
                    source_path=jf,
                    dataset_group=group_dir.name,
                    subdataset_name=subdataset_name,
                    viz_script=viz_script,
                    line_count=line_count,
                    dataset_type=dataset_type,
                    records=records,
                    channel_count_estimate=int(ch_est),
                    dimension_mode=str(dim_mode),
                    file_mtime=float(file_mtime),
                )
            )

        if total_lines > 0:
            type_line_counts[dataset_type] = type_line_counts.get(dataset_type, 0) + total_lines

    if not sources:
        raise RuntimeError(f"No usable *_descriptions*.jsonl found under {datasets_root}")

    # 随机打乱描述文件的顺序
    random.shuffle(sources)

    return sources, type_line_counts


def _merge_discovered_sources(existing_sources: list[SourceInfo], discovered_sources: list[SourceInfo]) -> dict[str, int]:
    """Merge newly discovered sources into the in-memory pool for outer refill rounds."""
    stats = {"added": 0, "refreshed": 0, "reopened": 0}
    by_path = {str(s.source_path): s for s in existing_sources}

    for fresh in discovered_sources:
        key = str(fresh.source_path)
        old = by_path.get(key)
        if old is None:
            existing_sources.append(fresh)
            by_path[key] = fresh
            stats["added"] += 1
            continue

        changed = bool(
            float(fresh.file_mtime) > float(old.file_mtime) + 1e-6
            or int(fresh.line_count) != int(old.line_count)
        )
        if changed:
            old.records = list(fresh.records)
            old.line_count = int(fresh.line_count)
            old.dataset_group = str(fresh.dataset_group)
            old.subdataset_name = str(fresh.subdataset_name)
            old.viz_script = fresh.viz_script
            old.dataset_type = str(fresh.dataset_type)
            old.channel_count_estimate = int(fresh.channel_count_estimate)
            old.dimension_mode = str(fresh.dimension_mode)
            old.file_mtime = float(fresh.file_mtime)
            old.augmentable = True
            old.augmented_count = 0
            old.exhausted = False
            old._line_idx = -1
            stats["refreshed"] += 1
            continue

        if old.exhausted and old.records:
            random.shuffle(old.records)
            old.exhausted = False
            old._line_idx = -1
            stats["reopened"] += 1

    return stats


def _compute_type_line_counts_from_sources(sources: list[SourceInfo]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in sources:
        out[s.dataset_type] = out.get(s.dataset_type, 0) + max(0, int(s.line_count))
    return out


def _assign_source_targets(sources: list[SourceInfo], type_targets: dict[str, int]) -> None:
    type_to_sources: dict[str, list[SourceInfo]] = {}
    for s in sources:
        type_to_sources.setdefault(s.dataset_type, []).append(s)

    for type_name, src_list in type_to_sources.items():
        src_weight_map = {str(s.source_path): s.line_count for s in src_list}
        src_targets = _allocate_targets_by_weight(src_weight_map, type_targets.get(type_name, 0))
        for s in src_list:
            s.accepted_target = src_targets.get(str(s.source_path), 0)


def _extract_caption(sample: dict[str, Any]) -> str | None:
    descs = sample.get("descriptions")
    if isinstance(descs, list):
        for it in descs:
            if isinstance(it, str) and it.strip():
                return it.strip()
    if isinstance(descs, str) and descs.strip():
        return descs.strip()

    caption = sample.get("caption")
    if isinstance(caption, str) and caption.strip():
        return caption.strip()

    description = sample.get("description")
    if isinstance(description, str) and description.strip():
        return description.strip()

    return None


def _infer_target_index(sample: dict[str, Any], n_cols: int) -> int:
    if n_cols <= 1:
        return 0

    feature_names = sample.get("feature_names")
    if isinstance(feature_names, list) and feature_names:
        target_col = sample.get("target_col")
        if isinstance(target_col, str) and target_col in feature_names:
            return int(feature_names.index(target_col))

        target_cols = sample.get("target_cols")
        if isinstance(target_cols, list):
            for col in target_cols:
                if isinstance(col, str) and col in feature_names:
                    return int(feature_names.index(col))

    features_obj = sample.get("features")
    if isinstance(features_obj, dict):
        main_channel = features_obj.get("main_channel")
        if main_channel is not None:
            try:
                idx = int(main_channel)
                if 0 <= idx < n_cols:
                    return idx
            except Exception:
                pass

    main_channel = sample.get("main_channel")
    if main_channel is not None:
        try:
            idx = int(main_channel)
            if 0 <= idx < n_cols:
                return idx
        except Exception:
            pass

    return 0


def _extract_time_series(sample: dict[str, Any]) -> list[float] | None:
    # Already-normalized format.
    ts = sample.get("time_series")
    if isinstance(ts, list) and ts:
        out = []
        for v in ts:
            fv = _safe_float(v)
            if fv is None:
                return None
            out.append(fv)
        return out if out else None

    values = sample.get("values")
    if not isinstance(values, list) or not values:
        return None

    first = values[0]
    if isinstance(first, list):
        n_cols = len(first)
        if n_cols <= 0:
            return None
        col_idx = _infer_target_index(sample, n_cols=n_cols)
        out = []
        for row in values:
            if not isinstance(row, list) or col_idx >= len(row):
                return None
            fv = _safe_float(row[col_idx])
            if fv is None:
                return None
            out.append(fv)
        return out if out else None

    out = []
    for v in values:
        fv = _safe_float(v)
        if fv is None:
            return None
        out.append(fv)
    return out if out else None


def _load_viz_module(viz_script: Path, module_cache: dict[Path, Any]) -> Any:
    cached = module_cache.get(viz_script)
    if cached is not None:
        return cached

    module_name = f"_viz_{_sanitize_filename(viz_script.stem, 32)}_{abs(hash(str(viz_script))) % 10000000}"
    spec = importlib.util.spec_from_file_location(module_name, str(viz_script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load visualization script: {viz_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module_cache[viz_script] = module
    return module


def _render_sample_image(
    source: SourceInfo,
    sample: dict[str, Any],
    image_path: Path,
    dpi: int,
    module_cache: dict[Path, Any],
) -> None:
    module = _load_viz_module(source.viz_script, module_cache)
    plot_fn = getattr(module, "plot_sample_with_description", None)
    if plot_fn is None or not callable(plot_fn):
        raise RuntimeError(f"plot_sample_with_description not found in {source.viz_script}")

    sig = inspect.signature(plot_fn)
    params = sig.parameters

    kwargs: dict[str, Any] = {}
    if "output_path" in params:
        kwargs["output_path"] = str(image_path)
    if "dpi" in params:
        kwargs["dpi"] = int(dpi)
    if "show_events" in params:
        kwargs["show_events"] = True

    # Electricity visualization requires an explicit feature argument when using plot function directly.
    infer_feature_fn = getattr(module, "infer_feature_from_sample", None)
    if "feature" in params and callable(infer_feature_fn):
        try:
            kwargs["feature"] = infer_feature_fn(sample, str(source.source_path))
        except Exception:
            pass

    plot_fn(sample, **kwargs)

    if not image_path.exists():
        raise RuntimeError(f"Visualization did not create image: {image_path}")


def _choose_next_source(
    sources: list[SourceInfo],
    type_targets: dict[str, int],
    type_accepted: dict[str, int],
    enable_window_resample: bool,
) -> SourceInfo | None:
    active = [
        s
        for s in sources
        if (not s.exhausted) or (bool(enable_window_resample) and s.can_augment)
    ]
    if not active:
        return None

    type_to_sources: dict[str, list[SourceInfo]] = {}
    for s in active:
        type_to_sources.setdefault(s.dataset_type, []).append(s)

    needing_types = []
    for type_name, tgt in type_targets.items():
        if type_name not in type_to_sources:
            continue
        deficit = int(tgt) - int(type_accepted.get(type_name, 0))
        if deficit > 0:
            needing_types.append(type_name)

    def _type_quality(type_name: str) -> float:
        srcs = type_to_sources.get(type_name, [])
        attempts = sum(max(0, int(s.attempted_count)) for s in srcs)
        accepts = sum(max(0, int(s.accepted_count)) for s in srcs)
        # Smoothed pass-rate estimate in [0,1].
        return float((accepts + 1.0) / (attempts + 2.0))

    def _source_quality(src: SourceInfo) -> float:
        return float((max(0, int(src.accepted_count)) + 1.0) / (max(0, int(src.attempted_count)) + 2.0))

    if needing_types:
        chosen_type = _weighted_choice(
            needing_types,
            weight_fn=lambda t: float(max(1, type_targets[t] - type_accepted.get(t, 0)))
            * max(0.35, _type_quality(t)),
        )
    else:
        fallback_types = list(type_to_sources.keys())
        chosen_type = _weighted_choice(
            fallback_types,
            weight_fn=lambda t: float(sum(max(1, s.remaining_estimate) for s in type_to_sources[t]))
            * max(0.35, _type_quality(t)),
        )

    if chosen_type is None:
        return None

    return _weighted_choice(
        type_to_sources[chosen_type],
        weight_fn=lambda s: float(max(1, s.remaining_estimate)) * max(0.35, _source_quality(s)),
    )


def _cleanup_tmp_images(candidates: list[CandidateItem]) -> None:
    for cand in candidates:
        p = cand.tmp_image_path
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass


def _build_candidate_batch(
    sources: list[SourceInfo],
    type_targets: dict[str, int],
    type_accepted: dict[str, int],
    batch_size: int,
    attempted_total: int,
    max_attempts: int,
    dpi: int,
    module_cache: dict[Path, Any],
    tmp_dir: Path,
    score_cache: dict[str, int] | None,
    score_threshold: int,
    use_score_cache: bool,
    enable_window_resample: bool,
    min_window_len: int,
    max_window_len: int,
    window_length_candidates: list[int],
    balance_main_channel: bool,
    resume_accept_keys: set[str],
    accepted_dedup_keys: set[str],
) -> tuple[list[CandidateItem], int, int, int]:
    candidates: list[CandidateItem] = []
    attempted_delta = 0
    cache_hit_count = 0
    cache_low_skip_count = 0

    while len(candidates) < int(batch_size) and (attempted_total + attempted_delta) < int(max_attempts):
        source = _choose_next_source(
            sources=sources,
            type_targets=type_targets,
            type_accepted=type_accepted,
            enable_window_resample=enable_window_resample,
        )
        if source is None:
            break

        item = source.next_record()
        if item is None:
            if enable_window_resample and source.can_augment:
                synthetic = _build_augmented_sample_from_source(
                    source=source,
                    synthetic_idx=(attempted_total + attempted_delta + 1),
                    min_window_len=min_window_len,
                    max_window_len=max_window_len,
                    window_length_candidates=window_length_candidates,
                    balance_main_channel=balance_main_channel,
                )
                if synthetic is None:
                    continue
                sample_obj, line_idx = synthetic
            else:
                continue
        else:
            sample_obj, line_idx = item

        caption = _extract_caption(sample_obj)
        ts = _extract_time_series(sample_obj)
        if not caption or not ts:
            continue

        sample_id = sample_obj.get("id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            sample_id = f"{source.source_path.stem}_line{line_idx}"
        sample_id = sample_id.strip()

        if (not enable_window_resample) and _is_augmented_source_id(sample_id):
            continue

        # Auto-resume prefilter: skip already accepted samples before rendering/scoring.
        resume_key = _build_resume_accept_key(
            source_jsonl=str(source.source_path),
            source_id=str(sample_id),
        )
        if resume_key in resume_accept_keys:
            continue

        pre_dedup_key = _build_sample_dedup_key(
            source_path=source.source_path,
            sample_obj=sample_obj,
            sample_id=sample_id,
            line_idx=int(line_idx),
        )
        if pre_dedup_key in accepted_dedup_keys:
            continue

        attempted_delta += 1
        source.attempted_count += 1

        score_cache_key = _build_score_cache_key(source=source, sample_obj=sample_obj, line_idx=int(line_idx))
        cached_score = None
        if use_score_cache and score_cache is not None:
            cached_score = score_cache.get(score_cache_key)
            if isinstance(cached_score, int):
                cache_hit_count += 1
                if int(cached_score) < int(score_threshold):
                    cache_low_skip_count += 1
                    continue

        tmp_image_path = tmp_dir / f"tmp_{attempted_total + attempted_delta:08d}.png"
        if tmp_image_path.exists():
            tmp_image_path.unlink()

        try:
            _render_sample_image(
                source=source,
                sample=sample_obj,
                image_path=tmp_image_path,
                dpi=dpi,
                module_cache=module_cache,
            )
        except Exception as exc:
            print(f"[WARN] sample failed before scoring: id={sample_id} reason={exc}")
            if tmp_image_path.exists():
                tmp_image_path.unlink()
            continue

        dataset_name = sample_obj.get("dataset")
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            dataset_name = source.dataset_group

        candidates.append(
            CandidateItem(
                source=source,
                sample_obj=sample_obj,
                line_idx=line_idx,
                sample_id=sample_id,
                caption=caption,
                ts=ts,
                tmp_image_path=tmp_image_path,
                dataset_name=dataset_name,
                score_cache_key=score_cache_key,
                cached_score=(int(cached_score) if isinstance(cached_score, int) else None),
            )
        )

    return candidates, attempted_delta, cache_hit_count, cache_low_skip_count


def _score_candidate_batch(
    candidates: list[CandidateItem],
    use_batch_review: bool,
    batch_query_size: int,
) -> dict[str, Any]:
    scored: list[tuple[CandidateItem, int | None, dict | None]] = []
    batch_calls = 0
    batch_hits = 0
    single_fallback_calls = 0
    cache_used = 0

    if not candidates:
        return {
            "scored": scored,
            "batch_calls": batch_calls,
            "batch_hits": batch_hits,
            "single_fallback_calls": single_fallback_calls,
            "cache_used": cache_used,
        }

    pending_candidates = []
    for cand in candidates:
        if isinstance(cand.cached_score, int):
            cache_used += 1
            scored.append((cand, int(cand.cached_score), None))
        else:
            pending_candidates.append(cand)

    if not pending_candidates:
        return {
            "scored": scored,
            "batch_calls": batch_calls,
            "batch_hits": batch_hits,
            "single_fallback_calls": single_fallback_calls,
            "cache_used": cache_used,
        }

    if use_batch_review and len(pending_candidates) >= 2:
        chunk_size = max(1, int(batch_query_size))

        # Group by dataset type so each batch can attach type-matched reference hints.
        type_to_candidates: dict[str, list[CandidateItem]] = {}
        for cand in pending_candidates:
            type_to_candidates.setdefault(cand.source.dataset_type, []).append(cand)

        for type_name, type_candidates in type_to_candidates.items():
            for i in range(0, len(type_candidates), chunk_size):
                chunk = type_candidates[i : i + chunk_size]
                image_paths = [c.tmp_image_path for c in chunk]
                chunk_scores: dict[str, int] = {}
                chunk_model_meta: dict | None = None
                try:
                    try:
                        batch_result = run_analysis.get_batch_image_scores(
                            image_paths,
                            sample_type=type_name,
                            return_meta=True,
                        )
                    except TypeError:
                        batch_result = run_analysis.get_batch_image_scores(
                            image_paths,
                            sample_type=type_name,
                        )

                    if isinstance(batch_result, tuple) and len(batch_result) == 2:
                        chunk_scores, chunk_model_meta = batch_result
                    else:
                        chunk_scores = batch_result

                    batch_calls += 1
                except Exception as exc:
                    print(f"[WARN] batch scoring failed, will fallback to single scoring: {exc}")

                for cand in chunk:
                    score = None
                    score_model_meta: dict | None = None
                    got = chunk_scores.get(cand.tmp_image_path.name)
                    if isinstance(got, int):
                        score = int(got)
                        score_model_meta = chunk_model_meta
                        batch_hits += 1
                    else:
                        single_fallback_calls += 1
                        try:
                            single_result = run_analysis.get_image_score(
                                cand.tmp_image_path,
                                sample_type=type_name,
                                return_meta=True,
                            )
                        except TypeError:
                            single_result = run_analysis.get_image_score(
                                cand.tmp_image_path,
                                sample_type=type_name,
                            )

                        if isinstance(single_result, tuple) and len(single_result) == 2:
                            score, score_model_meta = single_result
                        else:
                            score = single_result

                    scored.append((cand, score, score_model_meta))
    else:
        for cand in pending_candidates:
            single_fallback_calls += 1
            try:
                single_result = run_analysis.get_image_score(
                    cand.tmp_image_path,
                    sample_type=cand.source.dataset_type,
                    return_meta=True,
                )
            except TypeError:
                single_result = run_analysis.get_image_score(
                    cand.tmp_image_path,
                    sample_type=cand.source.dataset_type,
                )

            if isinstance(single_result, tuple) and len(single_result) == 2:
                score, score_model_meta = single_result
            else:
                score = single_result
                score_model_meta = None
            scored.append((cand, score, score_model_meta))

    return {
        "scored": scored,
        "batch_calls": batch_calls,
        "batch_hits": batch_hits,
        "single_fallback_calls": single_fallback_calls,
        "cache_used": cache_used,
    }


def generate_filtered_samples(
    datasets_root: Path,
    target_qualified: int,
    score_threshold: int,
    output_base_dir: Path,
    report_path: Path = DEFAULT_REPORT_PATH,
    max_attempt_multiplier: float = 8.0,
    random_seed: int = 42,
    dpi: int = 300,
    review_batch_size: int = 5,
    batch_query_size: int = 5,
    use_batch_review: bool = True,
    score_workers: int = 1,
    prefetch_batches: int = 2,
    score_cache_path: Path | None = DEFAULT_SCORE_CACHE_PATH,
    use_score_cache: bool = True,
    prefer_primary_api_only: bool = True,
    clean_output: bool = True,
    enable_window_resample: bool = True,
    min_window_len: int = 24,
    max_window_len: int = 512,
    window_length_candidates: str | list[int] | None = None,
    balance_main_channel: bool = True,
    capacity_check_only: bool = False,
    skip_description_generation: bool = False,
    samples_per_dataset: int = 100,
    description_runs_per_dataset: int = DEFAULT_DESCRIPTION_RUNS_PER_DATASET,
    auto_regen_on_exhausted: bool = True,
    max_regen_rounds: int = DEFAULT_MAX_REGEN_ROUNDS,
    auto_resume: bool = True,
    delete_descriptions: bool = False,
    enable_online_reference_update: bool = True,
    reference_update_min_score: int = 85,
) -> dict[str, Any]:
    import random
    random.seed(int(random_seed))
    target_qualified = max(1, int(target_qualified))
    score_threshold = int(score_threshold)
    review_batch_size = max(1, int(review_batch_size))
    batch_query_size = max(1, min(5, int(batch_query_size)))
    use_batch_review = bool(use_batch_review)
    score_workers = max(1, int(score_workers))
    prefetch_batches = max(1, int(prefetch_batches))
    use_score_cache = bool(use_score_cache)
    prefer_primary_api_only = bool(prefer_primary_api_only)
    enable_online_reference_update = bool(enable_online_reference_update)
    reference_update_min_score = max(0, int(reference_update_min_score))
    enable_window_resample = bool(enable_window_resample)
    balance_main_channel = bool(balance_main_channel)
    min_window_len = max(2, int(min_window_len))
    max_window_len = max(min_window_len, int(max_window_len))
    description_runs_per_dataset = max(1, int(description_runs_per_dataset))
    auto_regen_on_exhausted = bool(auto_regen_on_exhausted)
    max_regen_rounds = max(0, int(max_regen_rounds))
    auto_resume = bool(auto_resume)
    parsed_window_lengths = _parse_window_length_candidates(window_length_candidates)

    if prefer_primary_api_only and getattr(run_analysis, "MODEL_FALLBACKS", None):
        primary_cfg = run_analysis.MODEL_FALLBACKS[0]
        run_analysis.MODEL_FALLBACKS = [primary_cfg]
        print(
            "[INFO] prefer_primary_api_only enabled: "
            f"provider={primary_cfg.get('provider')} model={primary_cfg.get('name')}"
        )

    description_round_stats: list[dict[str, int]] = []
    regen_rounds_used = 0
    if not skip_description_generation:
        description_round_stats.append(
            _generate_descriptions_round(
                datasets_root=datasets_root,
                samples_per_dataset=int(samples_per_dataset),
                window_lengths=list(DEFAULT_WINDOW_LENGTH_CANDIDATES),
                runs_per_dataset=description_runs_per_dataset,
                round_idx=1,
            )
        )
        print("[描述生成] 初始轮次完成")
    else:
        print("[描述生成] skip_description_generation=True，跳过初始轮次，直接使用已有描述池")

    skip_augmented_source_ids = not enable_window_resample
    if skip_augmented_source_ids:
        print("[INFO] enable_window_resample=False: skip all samples whose source_id startswith 'augSRC'")

    sources, type_line_counts = _discover_sources(
        datasets_root,
        skip_augmented_source_ids=skip_augmented_source_ids,
    )
    type_line_counts = _compute_type_line_counts_from_sources(sources)

    capacity_report = _estimate_theoretical_capacity(
        sources=sources,
        min_window_len=min_window_len,
        max_window_len=max_window_len,
        window_length_candidates=parsed_window_lengths,
    )
    print(
        "[INFO] capacity_check "
        f"theoretical_windows={capacity_report['total_theoretical_windows']} "
        f"target={target_qualified} reachable={capacity_report['total_theoretical_windows'] >= target_qualified}"
    )

    if capacity_check_only:
        report = {
            "target_qualified": int(target_qualified),
            "capacity_check_only": True,
            "enable_window_resample": enable_window_resample,
            "min_window_len": min_window_len,
            "max_window_len": max_window_len,
            "window_length_candidates": parsed_window_lengths,
            "balance_main_channel": balance_main_channel,
            "type_line_counts": type_line_counts,
            "capacity": capacity_report,
            "target_reachable_in_theory": bool(capacity_report["total_theoretical_windows"] >= int(target_qualified)),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] capacity check report written: {report_path}")
        return report

    # 按类型分配目标样本数
    type_targets = _allocate_targets_by_weight(type_line_counts, total_target=target_qualified)
    for type_name, tgt in sorted(type_targets.items(), key=lambda x: x[0].lower()):
        type_display_name = dataset_type_config.TYPE_NAMES.get(type_name, type_name)
        print(
            f"[INFO] type={type_display_name} ({type_name}) total_lines={type_line_counts[type_name]} "
            f"accepted_target={tgt}"
        )

    _assign_source_targets(sources=sources, type_targets=type_targets)

    for s in sorted(sources, key=lambda x: (x.dataset_type.lower(), x.dataset_group.lower(), x.subdataset_name.lower())):
        type_display_name = dataset_type_config.TYPE_NAMES.get(s.dataset_type, s.dataset_type)
        print(
            f"[INFO] subdataset={type_display_name}/{s.dataset_group}/{s.subdataset_name} "
            f"line_count={s.line_count} accepted_target={s.accepted_target}"
        )

    # 为每个类型创建输出目录和文件
    output_base_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    type_output_jsonl: dict[str, Path] = {}
    type_output_image_dir: dict[str, Path] = {}
    type_fouts: dict[str, Any] = {}
    
    existing_resume_artifacts = False
    for type_name in type_targets:
        type_display_name = dataset_type_config.TYPE_NAMES.get(type_name, type_name)
        type_dir = output_base_dir / type_display_name
        type_image_dir = type_dir / "image"
        type_jsonl = type_dir / "samples_filtered.jsonl"
        
        type_dir.mkdir(parents=True, exist_ok=True)
        type_image_dir.mkdir(parents=True, exist_ok=True)
        
        type_output_jsonl[type_name] = type_jsonl
        type_output_image_dir[type_name] = type_image_dir

        try:
            if type_jsonl.exists() and type_jsonl.stat().st_size > 0:
                existing_resume_artifacts = True
        except Exception:
            pass
        if not existing_resume_artifacts:
            try:
                if any(type_image_dir.glob("*.png")):
                    existing_resume_artifacts = True
            except Exception:
                pass

    if clean_output and auto_resume and existing_resume_artifacts:
        print("[INFO] auto_resume detected existing outputs, switch clean_output=False")
        clean_output = False

    for type_name in type_targets:
        type_jsonl = type_output_jsonl[type_name]
        type_image_dir = type_output_image_dir[type_name]
        if clean_output:
            if type_jsonl.exists():
                type_jsonl.unlink()
            for child in type_image_dir.iterdir():
                if child.is_file():
                    child.unlink()
                elif child.is_dir() and child.name == "_tmp":
                    shutil.rmtree(child, ignore_errors=True)

    # 创建全局临时目录
    tmp_dir = output_base_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    max_attempts = max(int(target_qualified), int(math.ceil(target_qualified * float(max_attempt_multiplier))))
    print(
        f"[INFO] target_qualified={target_qualified} score_threshold>={score_threshold} "
        f"max_attempts={max_attempts} review_batch_size={review_batch_size} "
        f"batch_query_size={batch_query_size} use_batch_review={use_batch_review} "
        f"score_workers={score_workers} prefetch_batches={prefetch_batches} "
        f"use_score_cache={use_score_cache} prefer_primary_api_only={prefer_primary_api_only} "
        f"enable_online_reference_update={enable_online_reference_update} "
        f"reference_update_min_score={reference_update_min_score} "
        f"enable_window_resample={enable_window_resample} min_window_len={min_window_len} "
        f"max_window_len={max_window_len} balance_main_channel={balance_main_channel} "
        f"skip_description_generation={skip_description_generation} "
        f"auto_resume={auto_resume} auto_regen_on_exhausted={auto_regen_on_exhausted} "
        f"max_regen_rounds={max_regen_rounds} description_runs_per_dataset={description_runs_per_dataset}"
    )

    accepted_total = 0
    attempted_total = 0
    type_accepted: dict[str, int] = {t: 0 for t in type_targets}
    type_accepted_count: dict[str, int] = {t: 0 for t in type_targets}
    resume_accept_keys: set[str] = set()
    resume_dedup_keys: set[str] = set()
    resume_dedup_base_count = 0
    resume_loaded_samples = 0

    if auto_resume and not clean_output:
        resume_state = _load_existing_accept_state(
            type_output_jsonl=type_output_jsonl,
            type_targets=type_targets,
            skip_augmented_source_ids=skip_augmented_source_ids,
        )
        accepted_total = int(resume_state.get("accepted_total", 0))
        type_accepted = dict(resume_state.get("type_accepted", type_accepted))
        type_accepted_count = dict(resume_state.get("type_accepted_count", type_accepted_count))
        resume_accept_keys = set(resume_state.get("resume_accept_keys", set()))
        resume_dedup_keys = set(resume_state.get("resume_dedup_keys", set()))
        resume_dedup_base_count = len(resume_dedup_keys)
        resume_loaded_samples = int(accepted_total)

        source_accept_counts = dict(resume_state.get("source_accept_counts", {}))
        for s in sources:
            s.accepted_count = int(source_accept_counts.get(str(s.source_path), 0))

        if resume_loaded_samples > 0:
            print(
                "[INFO] auto_resume restored "
                f"accepted={resume_loaded_samples} unique_ids={len(resume_accept_keys)} "
                f"unique_dedup={len(resume_dedup_keys)}"
            )

    module_cache: dict[Path, Any] = {}
    batch_calls_total = 0
    batch_hits_total = 0
    single_fallback_total = 0
    score_cache_hits = 0
    score_cache_used = 0
    score_cache_low_skips = 0
    score_cache_updates = 0
    online_reference_updates = 0
    duplicate_skips = 0
    model_threshold_override_hits = 0
    model_threshold_override_counts: dict[str, int] = {}
    batches_submitted = 0
    loop_start_ts = time.time()
    accepted_dedup_keys: set[str] = set(resume_dedup_keys)

    score_cache: dict[str, int] = {}
    if use_score_cache and score_cache_path is not None:
        score_cache = _load_score_cache(Path(score_cache_path))
        print(f"[INFO] score_cache loaded entries={len(score_cache)} path={score_cache_path}")

    # 打开每个类型的输出文件
    try:
        for type_name in type_output_jsonl:
            type_fouts[type_name] = type_output_jsonl[type_name].open("a", encoding="utf-8", buffering=1)
        with ThreadPoolExecutor(max_workers=score_workers, thread_name_prefix="score_batch") as score_executor:
            inflight: dict[Any, list[CandidateItem]] = {}
            no_more_to_submit = False

            while True:
                # Fill the score queue with prefetched batches.
                while (
                    (not no_more_to_submit)
                    and accepted_total < target_qualified
                    and attempted_total < max_attempts
                    and len(inflight) < prefetch_batches
                ):
                    next_batch, next_attempted, batch_cache_hits, batch_cache_low_skips = _build_candidate_batch(
                        sources=sources,
                        type_targets=type_targets,
                        type_accepted=type_accepted,
                        batch_size=review_batch_size,
                        attempted_total=attempted_total,
                        max_attempts=max_attempts,
                        dpi=dpi,
                        module_cache=module_cache,
                        tmp_dir=tmp_dir,
                        score_cache=score_cache,
                        score_threshold=score_threshold,
                        use_score_cache=use_score_cache,
                        enable_window_resample=enable_window_resample,
                        min_window_len=min_window_len,
                        max_window_len=max_window_len,
                        window_length_candidates=parsed_window_lengths,
                        balance_main_channel=balance_main_channel,
                        resume_accept_keys=resume_accept_keys,
                        accepted_dedup_keys=accepted_dedup_keys,
                    )
                    attempted_total += next_attempted
                    score_cache_hits += int(batch_cache_hits)
                    score_cache_low_skips += int(batch_cache_low_skips)

                    if not next_batch:
                        no_more_to_submit = True
                        break

                    fut = score_executor.submit(
                        _score_candidate_batch,
                        next_batch,
                        use_batch_review,
                        batch_query_size,
                    )
                    inflight[fut] = next_batch
                    batches_submitted += 1

                if not inflight:
                    if attempted_total >= max_attempts and accepted_total < target_qualified:
                        print("[WARN] reached max_attempts, stop generation")
                        break
                    if no_more_to_submit and accepted_total < target_qualified:
                        can_refill = bool(
                            auto_regen_on_exhausted
                            and regen_rounds_used < max_regen_rounds
                        )
                        if can_refill:
                            regen_rounds_used += 1
                            refill_round = 1 + regen_rounds_used
                            print(
                                "[INFO] source pool exhausted, trigger outer refill "
                                f"round={refill_round}/{1 + max_regen_rounds}"
                            )
                            description_round_stats.append(
                                _generate_descriptions_round(
                                    datasets_root=datasets_root,
                                    samples_per_dataset=int(samples_per_dataset),
                                    window_lengths=list(DEFAULT_WINDOW_LENGTH_CANDIDATES),
                                    runs_per_dataset=description_runs_per_dataset,
                                    round_idx=refill_round,
                                )
                            )
                            try:
                                discovered_sources, _ = _discover_sources(
                                    datasets_root,
                                    skip_augmented_source_ids=skip_augmented_source_ids,
                                )
                            except Exception as exc:
                                print(f"[WARN] refill discovery failed: {exc}")
                                print("[WARN] no renderable source remains, stop generation")
                                break
                            merge_stats = _merge_discovered_sources(
                                existing_sources=sources,
                                discovered_sources=discovered_sources,
                            )
                            type_line_counts = _compute_type_line_counts_from_sources(sources)
                            type_targets = _allocate_targets_by_weight(type_line_counts, total_target=target_qualified)
                            _assign_source_targets(sources=sources, type_targets=type_targets)
                            for t in type_targets:
                                if t not in type_output_jsonl:
                                    type_display_name = dataset_type_config.TYPE_NAMES.get(t, t)
                                    type_dir = output_base_dir / type_display_name
                                    type_image_dir = type_dir / "image"
                                    type_jsonl = type_dir / "samples_filtered.jsonl"
                                    type_dir.mkdir(parents=True, exist_ok=True)
                                    type_image_dir.mkdir(parents=True, exist_ok=True)
                                    type_output_jsonl[t] = type_jsonl
                                    type_output_image_dir[t] = type_image_dir
                                    if t not in type_fouts:
                                        type_fouts[t] = type_jsonl.open("a", encoding="utf-8", buffering=1)
                                if t not in type_accepted:
                                    type_accepted[t] = 0
                                if t not in type_accepted_count:
                                    type_accepted_count[t] = 0
                            print(
                                "[INFO] refill merged "
                                f"added={merge_stats['added']} refreshed={merge_stats['refreshed']} reopened={merge_stats['reopened']}"
                            )
                            no_more_to_submit = False
                            continue
                        print("[WARN] no renderable source remains, stop generation")
                    break

                done_set, _ = wait(set(inflight.keys()), return_when=FIRST_COMPLETED)
                for done_fut in done_set:
                    done_batch = inflight.pop(done_fut, None)

                    try:
                        batch_result = done_fut.result()
                    except Exception as exc:
                        print(f"[WARN] score batch failed: {exc}")
                        if done_batch:
                            _cleanup_tmp_images(done_batch)
                        continue

                    scored_pairs = batch_result.get("scored", [])
                    batch_calls_total += int(batch_result.get("batch_calls", 0))
                    batch_hits_total += int(batch_result.get("batch_hits", 0))
                    single_fallback_total += int(batch_result.get("single_fallback_calls", 0))
                    score_cache_used += int(batch_result.get("cache_used", 0))

                    last_type = ""
                    for cand, score, score_model_meta in scored_pairs:
                        last_type = cand.source.dataset_type

                        if use_score_cache and (score is not None) and (cand.cached_score is None):
                            try:
                                score_int = int(score)
                            except Exception:
                                score_int = None
                            if isinstance(score_int, int) and 0 <= score_int <= 100:
                                score_cache[cand.score_cache_key] = score_int
                                score_cache_updates += 1
                                if (
                                    score_cache_path is not None
                                    and score_cache_updates > 0
                                    and score_cache_updates % 10 == 0
                                ):
                                    _save_score_cache(Path(score_cache_path), score_cache)

                        # Target already reached for this type: skip
                        type_name = cand.source.dataset_type
                        if type_accepted_count.get(type_name, 0) >= type_targets.get(type_name, 0):
                            if cand.tmp_image_path.exists():
                                cand.tmp_image_path.unlink()
                            continue

                        # Target already reached globally: keep draining inflight work but discard outputs.
                        if accepted_total >= target_qualified:
                            if cand.tmp_image_path.exists():
                                cand.tmp_image_path.unlink()
                            continue

                        effective_score_threshold = _resolve_score_threshold_for_model(
                            score_threshold,
                            score_model_meta,
                        )
                        if effective_score_threshold != int(score_threshold):
                            model_threshold_override_hits += 1
                            model_id_for_cnt = str((score_model_meta or {}).get("model_id") or "unknown")
                            model_threshold_override_counts[model_id_for_cnt] = (
                                int(model_threshold_override_counts.get(model_id_for_cnt, 0)) + 1
                            )

                        if score is None or int(score) < int(effective_score_threshold):
                            if cand.tmp_image_path.exists():
                                cand.tmp_image_path.unlink()
                            continue

                        dedup_key = _build_candidate_dedup_key(cand)
                        resume_key = _build_resume_accept_key(
                            source_jsonl=str(cand.source.source_path),
                            source_id=str(cand.sample_id),
                        )
                        if dedup_key in accepted_dedup_keys or resume_key in resume_accept_keys:
                            duplicate_skips += 1
                            if cand.tmp_image_path.exists():
                                cand.tmp_image_path.unlink()
                            continue
                        accepted_dedup_keys.add(dedup_key)
                        resume_accept_keys.add(resume_key)

                        accepted_total += 1
                        type_accepted_count[type_name] = type_accepted_count.get(type_name, 0) + 1
                        cand.source.accepted_count += 1
                        type_accepted[type_name] = type_accepted.get(type_name, 0) + 1

                        type_display_name = dataset_type_config.TYPE_NAMES.get(type_name, type_name)
                        image_name = (
                            f"{type_accepted_count[type_name]:06d}_"
                            f"{_sanitize_filename(cand.dataset_name, 40)}_"
                            f"{_sanitize_filename(cand.sample_id, 100)}.png"
                        )
                        final_image_path = type_output_image_dir[type_name] / image_name
                        if final_image_path.exists():
                            final_image_path.unlink()
                        cand.tmp_image_path.replace(final_image_path)

                        train_record = {
                            "time_series": cand.ts,
                            "caption": cand.caption,
                            "dataset": cand.dataset_name,
                            "source_id": cand.sample_id,
                            "source_jsonl": str(cand.source.source_path),
                            "dedup_key": dedup_key,
                            "score": int(score),
                            "image": str(Path("Sample") / type_display_name / "image" / image_name),
                        }
                        type_fouts[type_name].write(json.dumps(train_record, ensure_ascii=False) + "\n")
                        type_fouts[type_name].flush()

                        if enable_online_reference_update:
                            try:
                                updated = run_analysis.register_high_score_reference(
                                    sample_type=type_name,
                                    image_path=final_image_path,
                                    score=int(score),
                                    dataset=str(cand.dataset_name),
                                    source_id=str(cand.sample_id),
                                    min_score=reference_update_min_score,
                                )
                                if bool(updated):
                                    online_reference_updates += 1
                            except Exception as exc:
                                print(
                                    "[WARN] online reference update failed: "
                                    f"type={type_name} sample_id={cand.sample_id} reason={exc}"
                                )

                    # Ensure tmp files are cleaned for this completed batch.
                    for cand, _, _ in scored_pairs:
                        if cand.tmp_image_path.exists():
                            cand.tmp_image_path.unlink()

                    if scored_pairs:
                        elapsed = time.time() - loop_start_ts
                        last_type_display = dataset_type_config.TYPE_NAMES.get(last_type, last_type) if last_type else 'n/a'
                        print(
                            f"[INFO] batch_done attempted={attempted_total} accepted={accepted_total}/{target_qualified} "
                            f"last_type={last_type_display} inflight={len(inflight)} elapsed={elapsed:.1f}s"
                        )

                if accepted_total >= target_qualified and not inflight:
                    break

            if accepted_total < target_qualified and attempted_total >= max_attempts:
                print("[WARN] qualified target was not met before attempt budget exhausted")
    finally:
        # 关闭所有类型的输出文件
        for fout in type_fouts.values():
            fout.close()
        for src in sources:
            src.close()

    shutil.rmtree(tmp_dir, ignore_errors=True)

    if use_score_cache and score_cache_path is not None:
        _save_score_cache(Path(score_cache_path), score_cache)

    capacity_report = _estimate_theoretical_capacity(
        sources=sources,
        min_window_len=min_window_len,
        max_window_len=max_window_len,
        window_length_candidates=parsed_window_lengths,
    )

    source_report = []
    dimension_mode_counts: dict[str, int] = {}
    for s in sorted(sources, key=lambda x: (x.dataset_type.lower(), x.dataset_group.lower(), str(x.source_path).lower())):
        dimension_mode_counts[s.dimension_mode] = dimension_mode_counts.get(s.dimension_mode, 0) + 1
        source_report.append(
            {
                "dataset_type": s.dataset_type,
                "dataset_group": s.dataset_group,
                "subdataset_name": s.subdataset_name,
                "source_jsonl": str(s.source_path),
                "line_count": s.line_count,
                "theoretical_windows": int(capacity_report.get("source_theoretical_windows", {}).get(str(s.source_path), 0)),
                "accepted_target": s.accepted_target,
                "attempted_count": s.attempted_count,
                "accepted_count": s.accepted_count,
                "exhausted": s.exhausted,
                "channel_count_estimate": int(s.channel_count_estimate),
                "dimension_mode": str(s.dimension_mode),
            }
        )

    total_elapsed_sec = max(1e-6, time.time() - loop_start_ts)
    attempted_per_min = float(attempted_total) / (total_elapsed_sec / 60.0)
    accepted_per_min = float(accepted_total) / (total_elapsed_sec / 60.0)

    report = {
        "target_qualified": target_qualified,
        "score_threshold_inclusive": score_threshold,
        "model_score_threshold_overrides": MODEL_SCORE_THRESHOLD_OVERRIDES,
        "model_threshold_override_hits": model_threshold_override_hits,
        "model_threshold_override_counts": model_threshold_override_counts,
        "max_attempts": max_attempts,
        "attempted_samples": attempted_total,
        "accepted_samples": accepted_total,
        "target_met": bool(accepted_total >= target_qualified),
        "elapsed_seconds": round(total_elapsed_sec, 3),
        "attempted_per_min": round(attempted_per_min, 3),
        "accepted_per_min": round(accepted_per_min, 3),
        "pipeline_mode": "staged_render_and_batch_review",
        "review_batch_size": review_batch_size,
        "batch_query_size": batch_query_size,
        "use_batch_review": use_batch_review,
        "score_workers": score_workers,
        "prefetch_batches": prefetch_batches,
        "use_score_cache": use_score_cache,
        "prefer_primary_api_only": prefer_primary_api_only,
        "enable_online_reference_update": enable_online_reference_update,
        "reference_update_min_score": reference_update_min_score,
        "online_reference_updates": online_reference_updates,
        "score_cache_path": (str(score_cache_path) if score_cache_path is not None else None),
        "score_cache_entries": len(score_cache),
        "score_cache_hits": score_cache_hits,
        "score_cache_used": score_cache_used,
        "score_cache_low_skips": score_cache_low_skips,
        "score_cache_updates": score_cache_updates,
        "duplicate_skips": duplicate_skips,
        "accepted_unique_count": len(accepted_dedup_keys),
        "accepted_unique_count_this_run": max(0, len(accepted_dedup_keys) - int(resume_dedup_base_count)),
        "resume_loaded_samples": resume_loaded_samples,
        "auto_resume": auto_resume,
        "batches_submitted": batches_submitted,
        "batch_calls": batch_calls_total,
        "batch_hits": batch_hits_total,
        "single_fallback_calls": single_fallback_total,
        "datasets_root": str(datasets_root),
        "output_base_dir": str(output_base_dir),
        "type_line_counts": type_line_counts,
        "type_targets": type_targets,
        "type_accepted": type_accepted,
        "dimension_mode_counts": dimension_mode_counts,
        "enable_window_resample": enable_window_resample,
        "min_window_len": min_window_len,
        "max_window_len": max_window_len,
        "window_length_candidates": parsed_window_lengths,
        "balance_main_channel": balance_main_channel,
        "description_runs_per_dataset": description_runs_per_dataset,
        "skip_description_generation": bool(skip_description_generation),
        "auto_regen_on_exhausted": auto_regen_on_exhausted,
        "max_regen_rounds": max_regen_rounds,
        "regen_rounds_used": regen_rounds_used,
        "description_round_stats": description_round_stats,
        "capacity": capacity_report,
        "sources": source_report,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[INFO] finished accepted={accepted_total}/{target_qualified} "
        f"report={report_path}"
    )
    
    # 删除原始描述文件（仅在达标后执行，避免影响续跑）
    if delete_descriptions and bool(accepted_total >= target_qualified):
        print("[INFO] 删除各个数据集目录下的原始 *_descriptions.jsonl 文件...")
        deleted_count = 0
        for jsonl_path in datasets_root.rglob("*_descriptions.jsonl"):
            try:
                jsonl_path.unlink()
                deleted_count += 1
                print(f"[INFO] 已删除: {jsonl_path}")
            except Exception as e:
                print(f"[WARN] 删除失败 {jsonl_path}: {e}")
        print(f"[INFO] 共删除 {deleted_count} 个描述文件")
    elif delete_descriptions:
        print("[INFO] target未达成，保留描述文件以便后续续跑")
    
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate training samples from dataset description JSONL files, "
            "render each sample with existing visualization scripts, review by run_analysis score, "
            "and keep only high-score samples into one unified JSONL."
        )
    )
    parser.add_argument("--datasets_root", type=str, default=str(DEFAULT_DATASETS_ROOT))
    parser.add_argument("--target_qualified", type=int, default=10000)
    parser.add_argument(
        "--score_threshold",
        type=int,
        default=75,
        help="样本最低保留分数阈值（含边界，例如75表示保留score>=75）",
    )
    parser.add_argument("--output_base_dir", type=str, default=str(DEFAULT_OUTPUT_BASE_DIR))
    parser.add_argument("--report_path", type=str, default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--max_attempt_multiplier", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--review_batch_size", type=int, default=5, help="Render-and-review batch size")
    parser.add_argument(
        "--batch_query_size",
        type=int,
        default=5,
        help="Max images per batch review API request (current API upper bound is 5)",
    )
    parser.add_argument(
        "--disable_batch_review",
        action="store_true",
        help="Disable batch review and fallback to single-image review only",
    )
    parser.add_argument(
        "--score_workers",
        type=int,
        default=1,
        help="Number of concurrent scoring workers (higher uses more CPU/network resources)",
    )
    parser.add_argument(
        "--prefetch_batches",
        type=int,
        default=2,
        help="How many rendered batches can be queued for scoring concurrently",
    )
    parser.add_argument(
        "--score_cache_path",
        type=str,
        default=str(DEFAULT_SCORE_CACHE_PATH),
        help="Persistent score cache json path",
    )
    parser.add_argument(
        "--disable_score_cache",
        action="store_true",
        help="Disable persistent score cache",
    )
    parser.add_argument(
        "--prefer_primary_api_only",
        type=int,
        choices=[0, 1],
        default=1,
        help="1=only use primary review API and keep retrying it; 0=allow fallback chain",
    )
    parser.add_argument("--clean_output", type=int, choices=[0, 1], default=1)
    parser.add_argument(
        "--enable_window_resample",
        type=int,
        choices=[0, 1],
        default=1,
        help="1=允许窗口重采样增广；0=禁用增广，并跳过 source_id 以 augSRC 开头的记录",
    )
    parser.add_argument("--min_window_len", type=int, default=24)
    parser.add_argument("--max_window_len", type=int, default=512)
    parser.add_argument(
        "--window_length_candidates",
        type=str,
        default=",".join(str(x) for x in DEFAULT_WINDOW_LENGTH_CANDIDATES),
        help="逗号分隔窗口长度候选，例如 24,32,64,128,256,512",
    )
    parser.add_argument("--balance_main_channel", type=int, choices=[0, 1], default=1)
    parser.add_argument(
        "--capacity_check_only",
        action="store_true",
        help="仅做理论容量检测并输出报告，不执行筛选与训练样本生成",
    )
    parser.add_argument(
        "--skip_description_generation",
        action="store_true",
        help="只跳过初始描述生成；源池耗尽时仍可按 auto_regen_on_exhausted 触发补池轮次",
    )
    parser.add_argument(
        "--samples_per_dataset",
        type=int,
        default=100,
        help="每个数据集生成的描述文本样本数（默认100）",
    )
    parser.add_argument(
        "--description_runs_per_dataset",
        type=int,
        default=DEFAULT_DESCRIPTION_RUNS_PER_DATASET,
        help="每个数据集在每轮描述生成中的随机采样运行次数",
    )
    parser.add_argument(
        "--auto_regen_on_exhausted",
        type=int,
        choices=[0, 1],
        default=1,
        help="1=当源池耗尽且未达标时，自动触发外层新一轮描述生成并继续筛选",
    )
    parser.add_argument(
        "--max_regen_rounds",
        type=int,
        default=DEFAULT_MAX_REGEN_ROUNDS,
        help="外层自动补料的最大追加轮数（不含初始轮）",
    )
    parser.add_argument(
        "--auto_resume",
        type=int,
        choices=[0, 1],
        default=1,
        help="1=检测到已有输出时自动续跑（不清空历史结果）",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="迭代轮数，例如 1 表示保存到 iteration_1 目录下",
    )
    parser.add_argument(
        "--delete_descriptions",
        action="store_true",
        help="生成完样本后，删除各个数据集目录下的原始 *_descriptions.jsonl 文件（默认已启用）",
    )
    parser.add_argument(
        "--keep_descriptions",
        action="store_true",
        help="保留各数据集原始 *_descriptions.jsonl 文件（用于调试/复现）",
    )
    parser.add_argument(
        "--enable_online_reference_update",
        type=int,
        choices=[0, 1],
        default=1,
        help="1=边评审边更新提示样本池；0=禁用在线更新",
    )
    parser.add_argument(
        "--reference_update_min_score",
        type=int,
        default=85,
        help="在线更新提示样本的最低分数阈值（含边界）",
    )
    parser.add_argument(
        "--apply_300k_profile",
        type=int,
        choices=[0, 1],
        default=0,
        help="1=应用30万样本批量生成预设（目标>=300000、score>=75、入池>=85、增强采样与缓存）",
    )
    args = parser.parse_args()
    # Respect explicit CLI override for window resample when 300k profile is enabled.
    args._user_specified_enable_window_resample = "--enable_window_resample" in sys.argv
    return args


def _apply_300k_profile(args: argparse.Namespace) -> None:
    """Apply a safe large-scale preset for 300k-sample generation."""
    if not bool(getattr(args, "apply_300k_profile", 0)):
        return

    args.target_qualified = max(int(args.target_qualified), 300000)
    args.score_threshold = max(int(args.score_threshold), 75)
    args.reference_update_min_score = max(int(args.reference_update_min_score), 85)
    args.max_attempt_multiplier = max(float(args.max_attempt_multiplier), 8.0)

    args.enable_online_reference_update = 1
    # 300k大规模模式默认启用回退链，提高可用吞吐并避免单API阻塞。
    args.prefer_primary_api_only = 0
    if not bool(getattr(args, "_user_specified_enable_window_resample", False)):
        args.enable_window_resample = 1
    args.balance_main_channel = 1
    args.disable_batch_review = False
    args.disable_score_cache = False
    args.auto_regen_on_exhausted = 1
    args.auto_resume = 1
    args.max_regen_rounds = max(int(args.max_regen_rounds), DEFAULT_MAX_REGEN_ROUNDS)
    args.description_runs_per_dataset = max(int(args.description_runs_per_dataset), DEFAULT_DESCRIPTION_RUNS_PER_DATASET)

    # Force 5-image concurrent review for throughput.
    args.review_batch_size = 5
    args.batch_query_size = 5
    # Keep upstream throughput at least 7 workers for this 300k profile.
    desired_workers = max(7, int(getattr(run_analysis, "MAX_CONCURRENCY", 2)))
    args.score_workers = max(desired_workers, int(args.score_workers))
    args.prefetch_batches = max(int(args.score_workers), int(args.prefetch_batches), 2)
    args.samples_per_dataset = max(300, int(args.samples_per_dataset))

    print(
        "[INFO] apply_300k_profile enabled: "
        f"target_qualified={args.target_qualified} score_threshold>={args.score_threshold} "
        f"reference_update_min_score={args.reference_update_min_score} "
        f"samples_per_dataset={args.samples_per_dataset} use_score_cache=True "
        f"enable_window_resample={args.enable_window_resample} "
        f"auto_resume={args.auto_resume} auto_regen_on_exhausted={args.auto_regen_on_exhausted} "
        f"max_regen_rounds={args.max_regen_rounds}"
    )


def main() -> None:
    args = parse_args()
    _apply_300k_profile(args)
    
    # 处理迭代轮数
    output_base_dir = Path(args.output_base_dir)
    report_path = Path(args.report_path)
    score_cache_path = Path(args.score_cache_path) if args.score_cache_path else None
    
    if args.iteration is not None:
        iteration_dir_name = f"iteration_{args.iteration}"
        output_base_dir = output_base_dir / iteration_dir_name
        report_path = output_base_dir / "generation_report.json"
        if score_cache_path is not None:
            score_cache_path = output_base_dir / "score_cache.json"

    delete_descriptions = True
    if bool(args.keep_descriptions):
        delete_descriptions = False
    elif bool(args.delete_descriptions):
        delete_descriptions = True
    
    generate_filtered_samples(
        datasets_root=Path(args.datasets_root),
        target_qualified=args.target_qualified,
        score_threshold=args.score_threshold,
        output_base_dir=output_base_dir,
        report_path=report_path,
        max_attempt_multiplier=args.max_attempt_multiplier,
        random_seed=args.seed,
        dpi=args.dpi,
        review_batch_size=args.review_batch_size,
        batch_query_size=args.batch_query_size,
        use_batch_review=(not args.disable_batch_review),
        score_workers=args.score_workers,
        prefetch_batches=args.prefetch_batches,
        score_cache_path=score_cache_path,
        use_score_cache=(not args.disable_score_cache),
        prefer_primary_api_only=bool(args.prefer_primary_api_only),
        clean_output=bool(args.clean_output),
        enable_window_resample=bool(args.enable_window_resample),
        min_window_len=args.min_window_len,
        max_window_len=args.max_window_len,
        window_length_candidates=args.window_length_candidates,
        balance_main_channel=bool(args.balance_main_channel),
        capacity_check_only=bool(args.capacity_check_only),
        skip_description_generation=bool(args.skip_description_generation),
        samples_per_dataset=int(args.samples_per_dataset),
        description_runs_per_dataset=int(args.description_runs_per_dataset),
        auto_regen_on_exhausted=bool(args.auto_regen_on_exhausted),
        max_regen_rounds=int(args.max_regen_rounds),
        auto_resume=bool(args.auto_resume),
        delete_descriptions=bool(delete_descriptions),
        enable_online_reference_update=bool(args.enable_online_reference_update),
        reference_update_min_score=int(args.reference_update_min_score),
    )


if __name__ == "__main__":
    main()
