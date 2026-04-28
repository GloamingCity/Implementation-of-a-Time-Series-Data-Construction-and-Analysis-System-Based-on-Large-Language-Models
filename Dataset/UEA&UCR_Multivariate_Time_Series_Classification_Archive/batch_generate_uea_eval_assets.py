import json
import random
import shutil
import argparse
from dataclasses import dataclass
from pathlib import Path
import importlib.util

import numpy as np


ROOT = Path(__file__).resolve().parent
MULTI_DIR = ROOT / "Multivariate"
UNI_DIR = ROOT / "Univariate"
SEED = 20260325
random.seed(SEED)
np.random.seed(SEED)


@dataclass
class DatasetTask:
    split: str
    dataset_name: str
    dataset_dir: Path
    arff_path: Path
    is_multivariate: bool


def _load_module(file_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


gen_mod = _load_module(ROOT / "generate_descriptions_UEA&UCR.py", "uea_gen")
viz_mod = _load_module(ROOT / "viz_UEA&UCR_samples.py", "uea_viz")


def list_dataset_dirs(base_dir: Path):
    if not base_dir.exists():
        return []
    return sorted([p for p in base_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower())


def choose_arff_file(dataset_dir: Path):
    arffs = sorted(dataset_dir.glob("*.arff"), key=lambda p: p.name.lower())
    if not arffs:
        return None
    plain = [p for p in arffs if "train" not in p.stem.lower() and "test" not in p.stem.lower()]
    if plain:
        return sorted(plain, key=lambda p: (len(p.name), p.name.lower()))[0]
    trains = [p for p in arffs if "train" in p.stem.lower()]
    if trains:
        return sorted(trains, key=lambda p: (len(p.name), p.name.lower()))[0]
    return arffs[0]


def list_candidate_arffs(dataset_dir: Path):
    arffs = sorted(dataset_dir.glob("*.arff"), key=lambda p: p.name.lower())
    if not arffs:
        return []
    first = choose_arff_file(dataset_dir)
    out = [first] if first is not None else []
    for a in arffs:
        if first is None or a != first:
            out.append(a)
    return out


def sanitize_name(s: str):
    bad = '\\/:*?"<>| '
    out = s
    for ch in bad:
        out = out.replace(ch, "_")
    return out


def adaptive_five_lengths(series_len: int):
    base = [96, 192, 288, 384, 512]
    if series_len >= 512:
        return base
    low = max(24, min(96, max(8, series_len // 6)))
    arr = np.linspace(low, series_len, num=5)
    vals = sorted({int(max(8, min(series_len, round(v)))) for v in arr})
    while len(vals) < 5:
        cand = max(8, vals[-1] - 1 if vals else low)
        if cand not in vals:
            vals.append(cand)
        vals = sorted(set(vals))
    if len(vals) > 5:
        vals = vals[-5:]
    return vals


def build_default_jsonl_name(arff_path: Path, lengths: list[int]):
    win_tag = "-".join(str(x) for x in lengths)
    return f"{arff_path.parent.name}_{arff_path.stem}_L{win_tag}_descriptions.jsonl"


def tag_scores_from_features(features: dict):
    g = features.get("global", {})
    p = features.get("periodicity", {})
    events = features.get("events", []) or []
    segments = features.get("segments", []) or []

    trend = str(g.get("trend_label", "flat"))
    vol = str(g.get("vol_level", "medium"))
    periodic = bool(p.get("global_periodic", False))
    corr = float(p.get("best_corr", 0.0) or 0.0)

    dirs = []
    for seg in segments:
        lbl = str(seg.get("trend_label", ""))
        if lbl in ("strong_up", "weak_up"):
            dirs.append("up")
        elif lbl in ("strong_down", "weak_down"):
            dirs.append("down")
    switch = 0
    for i in range(1, len(dirs)):
        if dirs[i] != dirs[i - 1]:
            switch += 1

    return {
        "periodic": (1.0 if periodic else 0.0) + corr,
        "volatile_event": (1.0 if vol == "high" else 0.0) + (1.0 if len(events) > 0 else 0.0),
        "transition": float(switch),
        "upward": 1.0 if trend in ("strong_up", "weak_up") else 0.0,
        "downward": 1.0 if trend in ("strong_down", "weak_down") else 0.0,
        "stable": (1.0 if trend == "flat" else 0.0) + (1.0 if vol == "low" else 0.0),
    }


def pick_five_groups(candidates: list[dict]):
    if not candidates:
        return []
    tags = ["periodic", "volatile_event", "transition", "upward", "downward"]
    used = set()
    picked = []

    for tag in tags:
        best = None
        best_score = -1e9
        for c in candidates:
            key = (c["instance_index"], c["start_index"])
            if key in used:
                continue
            score = float(c["scores"].get(tag, 0.0))
            for p in picked:
                if p["instance_index"] == c["instance_index"] and abs(p["start_index"] - c["start_index"]) < 16:
                    score -= 0.1
            if score > best_score:
                best_score = score
                best = c
        if best is not None:
            picked.append(best)
            used.add((best["instance_index"], best["start_index"]))

    if len(picked) < 5:
        leftovers = [c for c in candidates if (c["instance_index"], c["start_index"]) not in used]
        leftovers.sort(key=lambda x: sum(float(v) for v in x["scores"].values()), reverse=True)
        for c in leftovers:
            picked.append(c)
            used.add((c["instance_index"], c["start_index"]))
            if len(picked) >= 5:
                break

    return picked[:5]


def build_candidates(arff_path: Path, is_multivariate: bool, main_channel: int, lengths: list[int], dataset_name: str):
    df, targets, attrs, is_multi, n_channels, mv_series_list = gen_mod.load_ucr_arff(str(arff_path))
    max_len = max(lengths)
    candidates = []

    if is_multi:
        n_instances = len(mv_series_list)
        inst_indices = list(range(n_instances))
        random.shuffle(inst_indices)
        inst_indices = inst_indices[: min(40, len(inst_indices))]

        for idx in inst_indices:
            mat = np.asarray(mv_series_list[idx], dtype=float)
            if mat.shape[0] < max_len:
                continue
            ch = int(np.clip(main_channel, 0, mat.shape[1] - 1))
            ser = mat[:, ch].astype(float)
            s = gen_mod.pd.Series(ser, dtype="float64").interpolate(limit_direction="both").ffill().bfill()
            ser = s.to_numpy(dtype=float)
            max_start = ser.shape[0] - max_len
            starts = sorted(set(np.linspace(0, max_start, num=min(20, max_start + 1), dtype=int).tolist()))
            random.shuffle(starts)
            starts = starts[:20]

            for st in starts:
                win = ser[st : st + max_len]
                desc, feat = gen_mod.describe_window_series(win, dataset_name, idx, targets[idx])
                candidates.append(
                    {
                        "instance_index": int(idx),
                        "start_index": int(st),
                        "class_label": targets[idx],
                        "scores": tag_scores_from_features(feat),
                    }
                )
    else:
        n_instances = int(df.shape[0])
        inst_indices = list(range(n_instances))
        random.shuffle(inst_indices)
        inst_indices = inst_indices[: min(40, len(inst_indices))]

        for idx in inst_indices:
            ser = df.iloc[idx].to_numpy(dtype=float)
            s = gen_mod.pd.Series(ser, dtype="float64").interpolate(limit_direction="both").ffill().bfill()
            ser = s.to_numpy(dtype=float)
            if ser.shape[0] < max_len:
                continue
            max_start = ser.shape[0] - max_len
            starts = sorted(set(np.linspace(0, max_start, num=min(20, max_start + 1), dtype=int).tolist()))
            random.shuffle(starts)
            starts = starts[:20]

            for st in starts:
                win = ser[st : st + max_len]
                desc, feat = gen_mod.describe_window_series(win, dataset_name, idx, targets[idx])
                candidates.append(
                    {
                        "instance_index": int(idx),
                        "start_index": int(st),
                        "class_label": targets[idx],
                        "scores": tag_scores_from_features(feat),
                    }
                )

    return candidates


def process_dataset(task: DatasetTask):
    df, targets, attrs, is_multi, n_channels, mv_series_list = gen_mod.load_ucr_arff(str(task.arff_path))
    effective_multi = bool(is_multi)
    if is_multi:
        if not mv_series_list or len(mv_series_list) == 0:
            return {"dataset": task.dataset_name, "status": "skipped_empty"}
        series_len = int(mv_series_list[0].shape[0])
    else:
        if int(df.shape[0]) == 0:
            return {"dataset": task.dataset_name, "status": "skipped_empty"}
        series_len = int(df.shape[1])

    if series_len < 16:
        return {"dataset": task.dataset_name, "status": "skipped_too_short", "series_len": series_len}

    lengths = adaptive_five_lengths(series_len)
    max_len = max(lengths)
    main_channel = int(np.random.randint(0, max(1, n_channels))) if effective_multi else 0

    candidates = build_candidates(task.arff_path, effective_multi, main_channel, lengths, task.dataset_name)
    groups = pick_five_groups(candidates)
    if len(groups) < 5:
        return {
            "dataset": task.dataset_name,
            "status": "skipped_insufficient_groups",
            "series_len": series_len,
            "lengths": lengths,
            "candidate_count": len(candidates),
        }

    jsonl_dir = task.dataset_dir / "jsonl"
    images_dir = task.dataset_dir / "images"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = jsonl_dir / build_default_jsonl_name(task.arff_path, lengths)

    out_rows = []
    plan = []

    if effective_multi:
        for gi, grp in enumerate(groups, start=1):
            idx = int(grp["instance_index"])
            st = int(grp["start_index"])
            targ = targets[idx]
            mat = np.asarray(mv_series_list[idx], dtype=float)
            for ch in range(mat.shape[1]):
                s = gen_mod.pd.Series(mat[:, ch], dtype="float64")
                if s.notna().sum() == 0:
                    continue
                s = s.interpolate(limit_direction="both").ffill().bfill()
                mat[:, ch] = s.to_numpy(dtype=float)

            plan_item = {"group": gi, "instance_index": idx, "start_index": st, "scores": grp["scores"], "samples": {}}
            for L in lengths:
                ed = st + L
                if ed > mat.shape[0]:
                    continue
                win = mat[st:ed, :]
                desc, feat = gen_mod.describe_window_multivariate(win, task.dataset_name, idx, targ, main_channel=main_channel)
                sid = f"{task.dataset_name}_{task.arff_path.stem}_idx{idx}_L{L}_start{st}"
                sample = {
                    "id": sid,
                    "dataset": task.dataset_name,
                    "task": "classification",
                    "source_file": task.arff_path.name,
                    "instance_index": int(idx),
                    "class_label": targ,
                    "is_multivariate": True,
                    "n_channels": int(win.shape[1]),
                    "main_channel": int(main_channel),
                    "target_col": f"ch{int(main_channel)}",
                    "target_cols": [f"ch{int(main_channel)}"],
                    "window_length": int(L),
                    "start_index": int(st),
                    "end_index": int(ed - 1),
                    "values": win.tolist(),
                    "features": feat,
                    "descriptions": [desc],
                }
                out_rows.append(sample)
                plan_item["samples"][str(L)] = sid
            plan.append(plan_item)
    else:
        for gi, grp in enumerate(groups, start=1):
            idx = int(grp["instance_index"])
            st = int(grp["start_index"])
            targ = targets[idx]
            ser = df.iloc[idx].to_numpy(dtype=float)
            s = gen_mod.pd.Series(ser, dtype="float64")
            s = s.interpolate(limit_direction="both").ffill().bfill()
            ser = s.to_numpy(dtype=float)

            plan_item = {"group": gi, "instance_index": idx, "start_index": st, "scores": grp["scores"], "samples": {}}
            for L in lengths:
                ed = st + L
                if ed > ser.shape[0]:
                    continue
                win = ser[st:ed]
                desc, feat = gen_mod.describe_window_series(win, task.dataset_name, idx, targ)
                sid = f"{task.dataset_name}_{task.arff_path.stem}_idx{idx}_L{L}_start{st}"
                sample = {
                    "id": sid,
                    "dataset": task.dataset_name,
                    "task": "classification",
                    "source_file": task.arff_path.name,
                    "instance_index": int(idx),
                    "class_label": targ,
                    "is_multivariate": False,
                    "n_channels": 1,
                    "window_length": int(L),
                    "start_index": int(st),
                    "end_index": int(ed - 1),
                    "values": win.tolist(),
                    "features": feat,
                    "descriptions": [desc],
                }
                out_rows.append(sample)
                plan_item["samples"][str(L)] = sid
            plan.append(plan_item)

    if len(out_rows) < 25:
        return {
            "dataset": task.dataset_name,
            "status": "skipped_insufficient_samples",
            "series_len": series_len,
            "lengths": lengths,
            "sample_count": len(out_rows),
        }

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            safe_row = gen_mod.make_json_safe(row)
            f.write(json.dumps(safe_row, ensure_ascii=False, allow_nan=False))
            f.write("\n")

    (jsonl_dir / "group_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    sample_map = {r["id"]: r for r in out_rows}
    for item in plan:
        g = int(item["group"])
        for L in lengths:
            sid = item["samples"].get(str(L))
            if not sid:
                continue
            sample = sample_map[sid]
            out_png = images_dir / f"group{g:02d}_L{int(L):03d}_{sanitize_name(sid)}.png"
            viz_mod.plot_sample_with_description(
                sample=sample,
                feature=None,
                main_channels=(str(main_channel) if effective_multi else None),
                wrap_width=32,
                show_events=True,
                output_path=str(out_png),
                dpi=300,
            )

    return {
        "dataset": task.dataset_name,
        "status": "ok",
        "split": task.split,
        "is_multivariate": effective_multi,
        "selected_arff": task.arff_path.name,
        "series_len": series_len,
        "lengths": lengths,
        "main_channel": int(main_channel),
        "jsonl": str(jsonl_path),
        "images_dir": str(images_dir),
        "group_count": 5,
        "image_count": 25,
    }


def main():
    parser = argparse.ArgumentParser(description="UEA分批生成评估用jsonl与图片")
    parser.add_argument("--batch_size", type=int, default=10, help="每批处理的数据集数量，默认10")
    parser.add_argument("--batch_index", type=int, default=0, help="从0开始的批次索引")
    parser.add_argument("--clean_empty_only", action="store_true", help="仅清理无ARFF目录并输出统计，不生成文件")
    args = parser.parse_args()

    all_dirs = []
    for base in [MULTI_DIR, UNI_DIR]:
        all_dirs.extend(list_dataset_dirs(base))
    initial_total = len(all_dirs)

    tasks = []
    deleted_no_arff = []

    for split_name, base, is_multi in [("Multivariate", MULTI_DIR, True), ("Univariate", UNI_DIR, False)]:
        for ds_dir in list_dataset_dirs(base):
            arff = choose_arff_file(ds_dir)
            if arff is None:
                shutil.rmtree(ds_dir)
                deleted_no_arff.append(str(ds_dir.relative_to(ROOT)))
                continue
            tasks.append(
                DatasetTask(
                    split=split_name,
                    dataset_name=ds_dir.name,
                    dataset_dir=ds_dir,
                    arff_path=arff,
                    is_multivariate=is_multi,
                )
            )

    tasks = sorted(tasks, key=lambda t: (t.split, t.dataset_name.lower()))
    total_tasks = len(tasks)
    bsz = max(1, int(args.batch_size))
    bidx = max(0, int(args.batch_index))
    bstart = bidx * bsz
    bend = min(total_tasks, bstart + bsz)
    batch_tasks = tasks[bstart:bend]

    results = []
    if not args.clean_empty_only:
        for i, t in enumerate(batch_tasks, start=1):
            print(f"[batch {bidx} | {i}/{len(batch_tasks)} | global {bstart + i}/{total_tasks}] {t.split}/{t.dataset_name}")
            res = None
            last_err = None
            for cand_arff in list_candidate_arffs(t.dataset_dir):
                t2 = DatasetTask(
                    split=t.split,
                    dataset_name=t.dataset_name,
                    dataset_dir=t.dataset_dir,
                    arff_path=cand_arff,
                    is_multivariate=t.is_multivariate,
                )
                try:
                    res = process_dataset(t2)
                    # 若成功或是可解释跳过状态，直接采用
                    if res.get("status") != "error":
                        break
                except Exception as e:
                    last_err = str(e)
                    continue

            if res is None:
                res = {
                    "dataset": t.dataset_name,
                    "status": "error",
                    "selected_arff": t.arff_path.name,
                    "error": last_err or "unknown error",
                }
            results.append(res)

    ok_count = sum(1 for r in results if r.get("status") == "ok")

    summary = {
        "seed": SEED,
        "initial_total": int(initial_total),
        "deleted_no_arff": deleted_no_arff,
        "deleted_count": int(len(deleted_no_arff)),
        "remaining_count": int(initial_total - len(deleted_no_arff)),
        "total_tasks": int(total_tasks),
        "batch_size": int(bsz),
        "batch_index": int(bidx),
        "batch_start": int(bstart),
        "batch_end": int(bend),
        "batch_task_count": int(len(batch_tasks)),
        "ok_count": int(ok_count),
        "image_total": int(ok_count * 25),
        "results": results,
    }

    out = ROOT / f"uea_batch_generation_summary_batch{bidx:03d}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
