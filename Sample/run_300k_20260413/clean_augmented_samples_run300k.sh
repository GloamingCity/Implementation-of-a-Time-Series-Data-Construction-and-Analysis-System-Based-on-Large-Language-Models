#!/usr/bin/env bash
# Run examples:
#   chmod +x clean_augmented_samples_run300k.sh
#   ./clean_augmented_samples_run300k.sh --dry-run
#   ./clean_augmented_samples_run300k.sh
#
# Default behavior:
#   1) Clean "*/samples_filtered.jsonl" under the target dir by removing rows
#      whose source_id starts with prefix (default: augSRC).
#   2) Clean "score_cache.json" under the target dir by removing keys that
#      contain the prefix (default: augSRC) or "#aug|".
#   3) Remove matched augmented sample images under "*/image" based on removed rows.
#      If rows were already removed, fallback-scan "*/image" by filename prefix.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$SCRIPT_DIR"
PREFIX="augSRC"
DRY_RUN=0
NO_BACKUP=0
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'EOF'
Usage:
  ./clean_augmented_samples_run300k.sh [options]

Default behavior:
  - Target directory is the script directory itself.
  - Clean all files matching */samples_filtered.jsonl under target directory.
  - Remove rows whose source_id starts with "augSRC".
    - Clean target_dir/score_cache.json by removing keys containing "augSRC" or "#aug|".
    - Remove matched augmented sample images under */image based on removed rows.
    - If no rows are removed in this run, fallback-scan */image by filename prefix and clean residual images.

Options:
  --dry-run             Analyze only; do not modify files.
  --no-backup           Do not create backup files before overwrite.
  --prefix <value>      Augmented source_id prefix (default: augSRC).
  --target-dir <dir>    Directory containing type subdirectories.
  --python <bin>        Python executable (default: python3, fallback: python).
  -h, --help            Show help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --no-backup)
      NO_BACKUP=1
      ;;
    --prefix)
      shift
      PREFIX="${1:-}"
      if [[ -z "$PREFIX" ]]; then
        echo "[ERROR] --prefix requires a value" >&2
        exit 2
      fi
      ;;
    --target-dir)
      shift
      TARGET_DIR="${1:-}"
      if [[ -z "$TARGET_DIR" ]]; then
        echo "[ERROR] --target-dir requires a value" >&2
        exit 2
      fi
      ;;
    --python)
      shift
      PYTHON_BIN="${1:-}"
      if [[ -z "$PYTHON_BIN" ]]; then
        echo "[ERROR] --python requires a value" >&2
        exit 2
      fi
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
  shift
done

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "[ERROR] target directory not found: $TARGET_DIR" >&2
  exit 2
fi
TARGET_DIR="$(cd "$TARGET_DIR" && pwd)"
SCORE_CACHE_PATH="$TARGET_DIR/score_cache.json"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "[ERROR] Python executable not found. Set --python or PYTHON_BIN." >&2
    exit 2
  fi
fi

echo "[INFO] target_dir=$TARGET_DIR prefix=$PREFIX dry_run=$DRY_RUN no_backup=$NO_BACKUP python=$PYTHON_BIN"
echo "[INFO] score_cache_path=$SCORE_CACHE_PATH"

"$PYTHON_BIN" - "$TARGET_DIR" "$PREFIX" "$DRY_RUN" "$NO_BACKUP" "$SCORE_CACHE_PATH" <<'PY'
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


def should_remove(line: str, prefix: str):
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return False, False, None
    sid = obj.get("source_id")
    if isinstance(sid, str) and sid.strip().startswith(prefix):
        return True, True, obj
    return False, True, obj


def _extract_image_name(obj):
    if not isinstance(obj, dict):
        return None
    raw = obj.get("image")
    if not isinstance(raw, str):
        return None
    txt = raw.strip()
    if not txt:
        return None
    txt = txt.replace("\\", "/")
    name = Path(txt).name
    return name if name else None


def _collect_fallback_aug_image_paths(jsonl_path: Path, prefix: str, kept_image_names: set[str]):
    token = str(prefix or "").strip().lower()
    if not token:
        return []

    image_dir = jsonl_path.parent / "image"
    if (not image_dir.exists()) or (not image_dir.is_dir()):
        return []

    out = []
    seen = set()
    for p in image_dir.glob("*"):
        if not p.is_file():
            continue
        name = p.name
        if token not in name.lower():
            continue
        if name in kept_image_names:
            continue
        key = str(p.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)

    return sorted(out, key=lambda x: x.name.lower())


def clean_file(path: Path, prefix: str, dry_run: bool, backup: bool):
    stats = {
        "total": 0,
        "kept": 0,
        "removed": 0,
        "invalid_json": 0,
        "changed": False,
        "backup_path": None,
        "image_targets_from_rows": 0,
        "image_targets_from_scan": 0,
        "image_targets": 0,
        "image_existing": 0,
        "image_missing": 0,
        "image_deleted": 0,
        "image_delete_errors": 0,
    }

    tmp_path = path.with_name(path.name + ".tmp.clean")
    fout = None
    removed_image_names = []
    kept_image_names = set()
    image_existing_paths = []
    image_targets_map = {}

    try:
        if not dry_run:
            fout = tmp_path.open("w", encoding="utf-8", newline="")

        with path.open("r", encoding="utf-8", newline="") as fin:
            for line in fin:
                stats["total"] += 1
                if not line.strip():
                    stats["kept"] += 1
                    if fout is not None:
                        fout.write(line)
                    continue

                remove, is_json, obj = should_remove(line, prefix)
                if not is_json:
                    stats["invalid_json"] += 1
                else:
                    image_name = _extract_image_name(obj)
                    if image_name:
                        if remove:
                            removed_image_names.append(image_name)
                        else:
                            kept_image_names.add(image_name)

                if remove:
                    stats["removed"] += 1
                    continue

                stats["kept"] += 1
                if fout is not None:
                    fout.write(line)

        if removed_image_names:
            # Avoid deleting an image that is still referenced by a kept row.
            target_names = sorted({
                name for name in removed_image_names if name and name not in kept_image_names
            })
            stats["image_targets_from_rows"] = int(len(target_names))
            for name in target_names:
                ip = path.parent / "image" / name
                image_targets_map[str(ip).lower()] = ip

        # Fallback for already-cleaned JSONL: scan image dir by filename prefix.
        fallback_paths = _collect_fallback_aug_image_paths(
            jsonl_path=path,
            prefix=prefix,
            kept_image_names=kept_image_names,
        )
        stats["image_targets_from_scan"] = int(len(fallback_paths))
        for ip in fallback_paths:
            image_targets_map[str(ip).lower()] = ip

        image_targets = sorted(image_targets_map.values(), key=lambda x: x.name.lower())
        stats["image_targets"] = int(len(image_targets))
        image_existing_paths = [p for p in image_targets if p.exists()]
        stats["image_existing"] = int(len(image_existing_paths))
        stats["image_missing"] = int(stats["image_targets"] - stats["image_existing"])

        if fout is not None:
            fout.close()
            fout = None

        if dry_run:
            if tmp_path.exists():
                tmp_path.unlink()
            return stats

        if stats["removed"] <= 0 and stats["image_existing"] <= 0:
            if tmp_path.exists():
                tmp_path.unlink()
            return stats

        if stats["removed"] > 0:
            if backup:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = path.with_name(f"{path.name}.bak_{stamp}")
                shutil.copy2(path, backup_path)
                stats["backup_path"] = backup_path

            tmp_path.replace(path)
            stats["changed"] = True
        else:
            if tmp_path.exists():
                tmp_path.unlink()

        for image_path in image_existing_paths:
            try:
                image_path.unlink()
                stats["image_deleted"] += 1
            except Exception as exc:
                stats["image_delete_errors"] += 1
                print(f"[WARN] failed to delete image: {image_path} | {exc}")

        if stats["image_deleted"] > 0:
            stats["changed"] = True

        return stats
    except Exception:
        if fout is not None:
            try:
                fout.close()
            except Exception:
                pass
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise


def _is_aug_cache_key(key: str, prefix: str) -> bool:
    k = str(key)
    if prefix and prefix in k:
        return True
    return "#aug|" in k


def clean_score_cache(cache_path: Path, prefix: str, dry_run: bool, backup: bool):
    if not cache_path.exists():
        return {
            "exists": False,
            "error": None,
            "total": 0,
            "removed": 0,
            "kept": 0,
            "changed": False,
            "backup_path": None,
            "would_change": False,
        }

    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "exists": True,
            "error": f"invalid_json: {exc}",
            "total": 0,
            "removed": 0,
            "kept": 0,
            "changed": False,
            "backup_path": None,
            "would_change": False,
        }

    if not isinstance(raw, dict):
        return {
            "exists": True,
            "error": "json_root_not_object",
            "total": 0,
            "removed": 0,
            "kept": 0,
            "changed": False,
            "backup_path": None,
            "would_change": False,
        }

    total = int(len(raw))
    kept_payload = {}
    removed = 0
    for k, v in raw.items():
        if _is_aug_cache_key(str(k), prefix):
            removed += 1
        else:
            kept_payload[k] = v

    kept = int(len(kept_payload))
    would_change = bool(removed > 0)
    backup_path = None

    if dry_run:
        return {
            "exists": True,
            "error": None,
            "total": total,
            "removed": int(removed),
            "kept": kept,
            "changed": False,
            "backup_path": None,
            "would_change": would_change,
        }

    changed = False
    if removed > 0:
        if backup:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = cache_path.with_name(f"{cache_path.name}.bak_{stamp}")
            shutil.copy2(cache_path, backup_path)

        cache_path.write_text(
            json.dumps(kept_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        changed = True

    return {
        "exists": True,
        "error": None,
        "total": total,
        "removed": int(removed),
        "kept": kept,
        "changed": changed,
        "backup_path": backup_path,
        "would_change": would_change,
    }


def main() -> int:
    target_dir = Path(sys.argv[1]).resolve()
    prefix = str(sys.argv[2])
    dry_run = bool(int(sys.argv[3]))
    no_backup = bool(int(sys.argv[4]))
    score_cache_path = Path(sys.argv[5]).resolve()

    targets = sorted(p.resolve() for p in target_dir.glob("*/samples_filtered.jsonl") if p.is_file())
    if not targets:
        print(f"[ERROR] no target files found under: {target_dir}")
        return 2

    print(f"[INFO] files={len(targets)}")
    for p in targets:
        print(f"  - {p}")

    total_removed = 0
    total_kept = 0
    total_invalid = 0
    changed_files = 0
    total_image_targets = 0
    total_image_targets_from_rows = 0
    total_image_targets_from_scan = 0
    total_image_existing = 0
    total_image_missing = 0
    total_image_deleted = 0
    total_image_delete_errors = 0

    for path in targets:
        file_stats = clean_file(
            path=path,
            prefix=prefix,
            dry_run=dry_run,
            backup=(not no_backup),
        )
        total_removed += int(file_stats.get("removed", 0))
        total_kept += int(file_stats.get("kept", 0))
        total_invalid += int(file_stats.get("invalid_json", 0))
        total_image_targets += int(file_stats.get("image_targets", 0))
        total_image_targets_from_rows += int(file_stats.get("image_targets_from_rows", 0))
        total_image_targets_from_scan += int(file_stats.get("image_targets_from_scan", 0))
        total_image_existing += int(file_stats.get("image_existing", 0))
        total_image_missing += int(file_stats.get("image_missing", 0))
        total_image_deleted += int(file_stats.get("image_deleted", 0))
        total_image_delete_errors += int(file_stats.get("image_delete_errors", 0))

        if bool(file_stats.get("changed", False)):
            changed_files += 1

        backup_path = file_stats.get("backup_path")
        backup_txt = str(backup_path) if backup_path is not None else "none"
        image_effective_label = "image_would_delete" if dry_run else "image_deleted"
        image_effective_value = (
            int(file_stats.get("image_existing", 0))
            if dry_run
            else int(file_stats.get("image_deleted", 0))
        )
        print(
            "[OK] "
            f"file={path} total={file_stats.get('total')} removed={file_stats.get('removed')} "
            f"kept={file_stats.get('kept')} invalid_json={file_stats.get('invalid_json')} "
            f"changed={file_stats.get('changed')} backup={backup_txt} "
            f"image_targets_from_rows={file_stats.get('image_targets_from_rows')} "
            f"image_targets_from_scan={file_stats.get('image_targets_from_scan')} "
            f"image_targets={file_stats.get('image_targets')} "
            f"image_existing={file_stats.get('image_existing')} "
            f"{image_effective_label}={image_effective_value} "
            f"image_missing={file_stats.get('image_missing')} "
            f"image_delete_errors={file_stats.get('image_delete_errors')}"
        )

    cache_stats = clean_score_cache(
        cache_path=score_cache_path,
        prefix=prefix,
        dry_run=dry_run,
        backup=(not no_backup),
    )

    if not bool(cache_stats.get("exists", False)):
        print(f"[WARN] score_cache not found: {score_cache_path}")
    elif cache_stats.get("error"):
        print(
            "[WARN] "
            f"score_cache={score_cache_path} skipped reason={cache_stats.get('error')}"
        )
    else:
        cache_backup_txt = (
            str(cache_stats.get("backup_path"))
            if cache_stats.get("backup_path") is not None
            else "none"
        )
        print(
            "[OK] "
            f"score_cache={score_cache_path} total={cache_stats.get('total')} "
            f"removed={cache_stats.get('removed')} kept={cache_stats.get('kept')} "
            f"changed={cache_stats.get('changed')} would_change={cache_stats.get('would_change')} "
            f"backup={cache_backup_txt}"
        )

    image_total_effective_label = "image_would_delete_total" if dry_run else "image_deleted_total"
    image_total_effective_value = total_image_existing if dry_run else total_image_deleted

    print(
        "[DONE] "
        f"files={len(targets)} changed_files={changed_files} removed_total={total_removed} "
        f"kept_total={total_kept} invalid_json_total={total_invalid} "
        f"image_targets_total={total_image_targets} "
        f"image_targets_from_rows_total={total_image_targets_from_rows} "
        f"image_targets_from_scan_total={total_image_targets_from_scan} "
        f"image_existing_total={total_image_existing} "
        f"{image_total_effective_label}={image_total_effective_value} "
        f"image_missing_total={total_image_missing} image_delete_errors_total={total_image_delete_errors} "
        f"score_cache_removed={cache_stats.get('removed', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY