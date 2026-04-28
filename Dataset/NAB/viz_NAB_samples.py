# Usage
# cd D:\Users\Win\Desktop\Study\毕业设计-面向大语言模型的时序推理数据构建及分析系统实现\数据收集\数据集\NAB
# python viz_NAB_samples.py --jsonl_path realTraffic_speed_7578_value_descriptions.jsonl --sample_index 0
# 可选参数：
#   --sample_id <id>   指定样本 id（二选一 sample_index）
#   --target_col <col> 指定要绘制的列名（默认首列）
#   --no_events        不高亮 z-score 检测到的异常事件
#   --dpi <数字>       输出图片分辨率，默认 300
# 输出文件名格式为 0000_<dataset>_<sample_id>.png，方便排序并包含子数据集信息。

import json
from pathlib import Path
import argparse
import re
import pandas as pd
import numpy as np
import matplotlib
# 使用无显示后端以便在没有显示服务器的环境中也能保存 PNG
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path as _Path

# 字体设置与其他 viz 脚本一致
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
]

FONT_PATH = None
for _p in FONT_CANDIDATES:
    if _Path(_p).exists():
        FONT_PATH = _p
        break

cn_font = None
if FONT_PATH:
    try:
        cn_font = font_manager.FontProperties(fname=FONT_PATH)
        matplotlib.rcParams["font.sans-serif"] = [cn_font.get_name()]
    except Exception:
        cn_font = None

# 解决负号显示问题
matplotlib.rcParams["axes.unicode_minus"] = False


def _token_visual_width(token: str) -> float:
    """粗略估计 token 的显示宽度：中文字符按 1.0，ASCII 按 0.56。"""
    w = 0.0
    for ch in token:
        if "\u4e00" <= ch <= "\u9fff":
            w += 1.0
        elif "\uff00" <= ch <= "\uffef":
            w += 1.0
        else:
            w += 0.56
    return w


def _wrap_single_line_mixed_text(line: str, max_width_units: float) -> list[str]:
    """
    对单行文本做中英混排折行：
    - 中文可按字切分
    - 英文/数字/路径等连续片段尽量不拆
    - 尽量避免标点落在新行行首
    """
    if not line:
        return [""]

    token_pattern = re.compile(r"[A-Za-z0-9]+(?:[._:/\\-][A-Za-z0-9]+)*|[\u4e00-\u9fff]|\s+|.")
    tokens = token_pattern.findall(line)

    lines: list[str] = []
    cur_tokens: list[str] = []
    cur_w = 0.0

    def flush_current():
        nonlocal cur_tokens, cur_w
        lines.append("".join(cur_tokens).rstrip())
        cur_tokens = []
        cur_w = 0.0

    for tok in tokens:
        if not cur_tokens and tok.isspace():
            continue

        tok_w = _token_visual_width(tok)
        if cur_tokens and (cur_w + tok_w > max_width_units):
            if re.fullmatch(r"[，。；：、！？）】》,.;:!?)]", tok):
                cur_tokens.append(tok)
                cur_w += tok_w
                continue
            flush_current()
            if tok.isspace():
                continue

        cur_tokens.append(tok)
        cur_w += tok_w

    if cur_tokens or not lines:
        flush_current()

    return lines


def format_description_for_plot(description: str, wrap_width: int) -> str:
    """
    保留描述文本中的显式换行，再对每一行做智能折行。
    """
    logical_lines = description.split("\n")
    wrapped_lines: list[str] = []
    for ln in logical_lines:
        if ln.strip() == "":
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(_wrap_single_line_mixed_text(ln, max_width_units=float(wrap_width)))
    return "\n".join(wrapped_lines)


def _draw_fitted_description(ax_txt, description: str, wrap_width: int, cn_font=None):
    """
    在右侧文本面板内自适应折行和字号，尽量保证整段文本完整显示。
    """
    fig = ax_txt.figure
    base_kwargs = dict(
        ha="left",
        va="top",
        linespacing=1.3,
        transform=ax_txt.transAxes,
        clip_on=False,
    )
    if cn_font:
        base_kwargs["fontproperties"] = cn_font

    width_candidates = [max(16, wrap_width + d) for d in (0, -2, -4, -6)]
    fontsize_candidates = [10.0, 9.5, 9.0, 8.5, 8.0]

    best = None
    best_ratio = float("inf")

    for w in width_candidates:
        wrapped = format_description_for_plot(description, wrap_width=w)
        for fs in fontsize_candidates:
            txt = ax_txt.text(0.0, 0.99, wrapped, fontsize=fs, **base_kwargs)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            txt_h = txt.get_window_extent(renderer=renderer).height
            ax_h = ax_txt.get_window_extent(renderer=renderer).height
            ratio = txt_h / max(ax_h, 1.0)
            if ratio <= 0.98:
                return txt
            if ratio < best_ratio:
                best_ratio = ratio
                best = (wrapped, fs)
            txt.remove()

    wrapped, fs = best if best is not None else (format_description_for_plot(description, wrap_width=max(16, wrap_width - 6)), 8.0)
    return ax_txt.text(0.0, 0.99, wrapped, fontsize=fs, **base_kwargs)


def _maybe_rotate_xticks(ax, fig, angle: int = 30) -> None:
    """
    当时间刻度文本在当前宽度下可能重叠时，自动倾斜显示。
    """
    try:
        fig.canvas.draw()
        labels = [lbl for lbl in ax.get_xticklabels() if lbl.get_visible() and lbl.get_text()]
        if not labels:
            return

        renderer = fig.canvas.get_renderer()
        axis_width_px = ax.get_window_extent(renderer=renderer).width
        total_label_width = sum(lbl.get_window_extent(renderer=renderer).width for lbl in labels)

        # 经验阈值：若标签总宽明显超出可用宽度，则改为斜着显示。
        if total_label_width > axis_width_px * 1.15:
            for lbl in labels:
                lbl.set_rotation(angle)
                lbl.set_horizontalalignment("right")
                lbl.set_rotation_mode("anchor")
    except Exception:
        # 仅影响观感，不应中断主流程
        return


def load_sample_from_jsonl(jsonl_path: str, sample_index: int | None = None, sample_id: str | None = None):
    """
    从 JSONL 文件里读取一个样本，返回 (样本对象, 索引值)。
    与其他 viz 脚本相同。
    """
    jsonl_path = Path(jsonl_path)
    assert jsonl_path.exists(), f"{jsonl_path} 不存在"

    if sample_id is not None:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                obj = json.loads(line)
                if obj.get("id") == sample_id:
                    return obj, idx
        raise ValueError(f"在 {jsonl_path} 中未找到 id={sample_id} 的样本")

    if sample_index is None:
        sample_index = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == sample_index:
                return json.loads(line), i

    raise IndexError(f"sample_index={sample_index} 超出范围")


def sample_to_dataframe(sample: dict) -> pd.DataFrame:
    """
    把 NAB 样本的 time/values 转成 DataFrame。
    时间列可能不等距，直接用 pandas 解析；若解析失败则回退到整数索引。
    """
    times = sample.get("time", [])
    start_index = int(sample.get("start_index", 0))
    idx = None
    if times:
        try:
            ts = pd.to_datetime(times, errors="coerce")
            if not ts.isnull().any() and len(ts) == len(times):
                idx = ts
        except Exception:
            idx = None
    if idx is None:
        idx = pd.RangeIndex(start=start_index, stop=start_index + len(sample.get("values", [])), step=1)

    vals = sample.get("values", [])
    arr = np.array(vals)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    cols = sample.get("feature_names", ["value"])
    df = pd.DataFrame(arr, index=idx, columns=cols)
    return df


def plot_sample_with_description(
    sample: dict,
    wrap_width: int = 28,
    show_events: bool = True,
    output_path: str | None = None,
    dpi: int = 300,
    target_col: str | None = None,
):
    """
    绘图：左图为时间序列，右图显示描述文本。
    与前几个视觉脚本几乎相同。
    """
    df = sample_to_dataframe(sample)
    # 选择列：优先使用传入的 target_col
    series_name = None
    if target_col:
        if target_col in df.columns:
            series_name = target_col
        else:
            raise ValueError(f"指定的 target_col '{target_col}' 不在样本列 {df.columns.tolist()} 中")
    if series_name is None:
        series_name = df.columns[0] if len(df.columns) else "value"

    desc_list = sample.get("descriptions", [])
    description = desc_list[0] if desc_list else "(样本中没有 descriptions 字段)"
    features_struct = sample.get("features", {})

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 2.2, 1.6])

    ax_ts = fig.add_subplot(gs[:, :2])
    ax_ts.plot(df.index, df[series_name], linewidth=1.2)
    ax_ts.set_title(f"{sample.get('id', '')}  |  {series_name}", fontsize=12, fontweight='bold')
    if pd.api.types.is_datetime64_any_dtype(df.index):
        ax_ts.set_xlabel("时间")
    else:
        ax_ts.set_xlabel("时间点")
    ax_ts.set_ylabel(series_name)
    ax_ts.grid(True, alpha=0.3)

    try:
        from matplotlib.dates import DateFormatter
        if pd.api.types.is_datetime64_any_dtype(df.index):
            ax_ts.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
            _maybe_rotate_xticks(ax_ts, fig, angle=30)
    except Exception:
        pass

    if show_events and "events" in features_struct:
        events = features_struct["events"]
        if events:
            times = df.index.to_list()
            for ev in events[:5]:
                s = ev["start"]
                e = ev["end"]
                s = max(0, min(s, len(times) - 1))
                e = max(0, min(e, len(times) - 1))
                ax_ts.axvspan(times[s], times[e], alpha=0.15, color='red')

    ax_txt = fig.add_subplot(gs[:, 2])
    ax_txt.axis("off")
    _draw_fitted_description(ax_txt, description, wrap_width=wrap_width, cn_font=cn_font)
    ax_txt.set_title("自动生成的中文描述", loc="left", fontsize=11, fontweight='bold')

    plt.tight_layout()
    out = Path(output_path) if output_path is not None else Path("output_viz_NAB.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="可视化 NAB 数据集描述样本：画出时间序列 + 中文描述。"
    )
    parser.add_argument("--jsonl_path", type=str, required=True, help="生成的 JSONL 样本文件路径")
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="要可视化的样本行号（从 0 开始）。与 --sample_id 二选一。",
    )
    parser.add_argument(
        "--sample_id",
        type=str,
        default=None,
        help="要可视化的样本 id，例如 speed_7578_value_L512_start0",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default=None,
        help="要绘制的目标列名，如果 jsonl 样本含有多个，会从中选择；默认使用第一个列",
    )
    parser.add_argument(
        "--no_events",
        action="store_true",
        help="不高亮 z-score 检测到的异常事件",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="输出图片的分辨率（dpi），默认300",
    )

    args = parser.parse_args()

    sample, sample_idx = load_sample_from_jsonl(
        jsonl_path=args.jsonl_path,
        sample_index=None if args.sample_id is not None else args.sample_index,
        sample_id=args.sample_id,
    )
    # 提取数据集名称并显示，用于日志/标题
    dataset_name = sample.get("dataset", "unknown_dataset")
    print(f"[info] 样本属于数据集：{dataset_name}")

    sample_id = sample.get("id", f"NAB_{sample_idx}")
    invalid_chars = set('\\/:*?"<>|')
    def sanitize(s: str) -> str:
        return ''.join('_' if c in invalid_chars else c for c in str(s)).strip()
    
    # 从 JSONL 文件名提取子数据集名称（如 realTraffic）
    jsonl_stem = Path(args.jsonl_path).stem  # 去掉扩展名
    # jsonl_stem 格式通常为：realTraffic_speed_7578_value_descriptions
    # 提取第一个下划线之前的部分作为文件夹名
    folder_name = jsonl_stem.split('_')[0] if '_' in jsonl_stem else 'NAB'
    
    # 输出文件名格式：编号_文件夹名_sample_id.png
    output_filename = f"{sample_idx:04d}_{sanitize(folder_name)}_{sanitize(sample_id)}.png"
    output_path = Path(args.jsonl_path).parent / output_filename

    plot_sample_with_description(
        sample,
        show_events=not args.no_events,
        output_path=str(output_path),
        dpi=args.dpi,
        target_col=args.target_col,
    )

    print(f"✅ 已生成可视化图片：{output_path}")


if __name__ == "__main__":
    main()
