#!/usr/bin/env bash
set -euo pipefail

# clean_files.sh
# Linux/bash 版本：复制自 Dataset\clean_files.bat
# 功能：删除当前目录下的 PNG/JSONL 文件，以及每个同级子目录根目录下的 PNG/JSONL 文件

echo "删除当前目录中的所有PNG文件..."
find . -maxdepth 1 -type f -name '*.png' -print -delete || true

echo "删除当前目录中的所有JSONL文件..."
find . -maxdepth 1 -type f -name '*.jsonl' -print -delete || true

echo "删除同级目录中的文件夹根目录下的PNG文件..."
for d in */; do
  [ -d "$d" ] || continue
  find "$d" -maxdepth 1 -type f -name '*.png' -print -delete || true
done

echo "删除同级目录中的文件夹根目录下的JSONL文件..."
for d in */; do
  [ -d "$d" ] || continue
  find "$d" -maxdepth 1 -type f -name '*.jsonl' -print -delete || true
done

echo "清理完成！"
