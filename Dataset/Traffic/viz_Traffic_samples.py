# Usage
# cd "D:\Users\Win\Desktop\Study\毕业设计-面向大语言模型的时序推理数据构建及分析系统实现\数据收集\数据集\Traffic"
# python viz_Traffic_samples.py --jsonl_path Traffic_descriptions.jsonl --sample_index 0
# python viz_Traffic_samples.py --jsonl_path Traffic_descriptions.jsonl --sample_index 0 --features traffic_volume,temp
# 默认会优先绘制样本中的 target_cols；如果样本包含联动列，会自动画出多条曲线
# 如需强制指定绘制列，请使用 --features（支持一个或多个，空格或逗号分隔）
# 可通过 "--sample_index x" 指定要可视化的样本行号（从0开始），也可以用 "--sample_id id" 按 id 精确查找，两者二选一

import json
from pathlib import Path
import argparse
import re
import textwrap
import pandas as pd
import matplotlib
# 使用无显示后端以便在没有显示服务器的环境中也能保存 PNG
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path as _Path

# 跨平台查找常见的中文/默认字体，优先使用本机可用字体
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


def _compute_title_wrap_width(fig, ax, title_fontsize: int) -> int:
    """根据左侧图表可用宽度估算标题换行阈值，避免过早换行。"""
    fig.canvas.draw()
    ax_width_in = fig.get_figwidth() * ax.get_position().width
    avg_char_in = max((title_fontsize / 72.0) * 0.62, 1e-6)
    est_chars = int(ax_width_in / avg_char_in)
    return max(52, min(140, est_chars))


def load_sample_from_jsonl(jsonl_path: str, sample_index: int | None = None, sample_id: str | None = None):
    """
    从 JSONL 文件里读取一个样本，返回 (样本对象, 索引值)。
    - 如果给出 sample_id，则按 id 精确匹配并返回其所在行索引。
    - 否则用 sample_index（第几行，从 0 开始）。
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
    将一个 sample 转成 pandas DataFrame，index 为时间戳，columns 为 feature_names。
    """
    times = pd.to_datetime(sample.get("time", []))
    feature_names = sample.get("feature_names", [])
    values = sample.get("values", [])

    df = pd.DataFrame(values, columns=feature_names, index=times)
    return df


def plot_sample_with_description(
    sample: dict,
    features: list[str] | None = None,
    wrap_width: int = 40,
    show_events: bool = True,
    output_path: str | None = None,
    dpi: int = 300,
):
    """
    在同一张图中：
    - 左侧子图：画出指定 features 的时间序列
    - 右侧子图：展示该窗口的中文描述（descriptions[0]）
    可选：在曲线上标出 z-score 检测到的异常事件区域。
    """
    df = sample_to_dataframe(sample)

    if features is None:
        features = sample.get("target_cols") or [sample.get("target_col") or (df.columns[0] if len(df.columns) > 0 else None)]
    if isinstance(features, str):
        features = [features]
    plot_cols = [c for c in features if c in df.columns]
    if not plot_cols:
        raise ValueError(f"features {features} 均不在 sample['feature_names'] 中：{df.columns.tolist()}")

    desc_list = sample.get("descriptions", [])
    description = desc_list[0] if desc_list else "(样本中没有 descriptions 字段)"
    features_struct = sample.get("features", {})

    fig = plt.figure(figsize=(15.5, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.3, 2.3, 1.9])

    ax_ts = fig.add_subplot(gs[:, :2])
    main_col = sample.get("target_col") or plot_cols[0]

    # 自动多纵轴：仅当量纲差异明显时才启用额外纵轴
    def _series_scale(col_name: str) -> float:
        s = df[col_name].astype(float)
        q95 = float(s.quantile(0.95))
        q05 = float(s.quantile(0.05))
        span = abs(q95 - q05)
        return max(span, 1e-6)

    main_scale = _series_scale(main_col)
    groups = {0: [main_col], 1: [], 2: []}
    same_axis_thr = 6.0
    third_axis_thr = 40.0
    for col in plot_cols:
        if col == main_col:
            continue
        ratio = main_scale / _series_scale(col)
        inv_ratio = 1.0 / max(ratio, 1e-6)
        mag_gap = max(ratio, inv_ratio)
        if mag_gap < same_axis_thr:
            groups[0].append(col)
        elif mag_gap < third_axis_thr:
            groups[1].append(col)
        else:
            groups[2].append(col)

    # 只有两级量纲差时，不额外启用第三轴
    if groups[2] and not groups[1]:
        groups[1] = groups[2]
        groups[2] = []

    ax_list = [ax_ts]
    if groups[1]:
        ax1 = ax_ts.twinx()
        ax1.spines["right"].set_position(("axes", 1.02))
        ax1.set_ylabel(" / ".join(groups[1]), fontsize=9)
        ax1.tick_params(axis="y", labelsize=8)
        ax_list.append(ax1)
    if groups[2]:
        ax2 = ax_ts.twinx()
        ax2.spines["right"].set_position(("axes", 1.08))
        ax2.set_ylabel(" / ".join(groups[2]), fontsize=9)
        ax2.tick_params(axis="y", labelsize=8)
        ax_list.append(ax2)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    handles, labels = [], []
    for tier, cols in groups.items():
        if not cols:
            continue
        axis = ax_list[min(tier, len(ax_list) - 1)]
        axis_tag = "左轴" if tier == 0 else ("右轴1" if tier == 1 else "右轴2")
        for idx, col in enumerate(cols):
            lw = 2.0 if col == main_col else 1.1
            alpha = 0.95 if col == main_col else 0.85
            color = color_cycle[(len(handles) + idx) % len(color_cycle)] if color_cycle else None
            legend_label = f"{col} [{axis_tag}]"
            line, = axis.plot(df.index, df[col], linewidth=lw, alpha=alpha, label=legend_label, color=color)
            handles.append(line)
            labels.append(legend_label)
    title_fontsize = 11
    raw_title = f"{sample.get('id','')}  |  {', '.join(plot_cols)}"
    title_wrap_width = _compute_title_wrap_width(fig, ax_ts, title_fontsize=title_fontsize)
    title_text = "\n".join(
        textwrap.wrap(raw_title, width=title_wrap_width, break_long_words=False, break_on_hyphens=False)
    )
    title_artist = ax_ts.set_title(title_text, fontsize=title_fontsize, fontweight='bold', loc='center', pad=10)
    title_artist.set_multialignment("center")
    ax_ts.set_xlabel("时间")
    ax_ts.set_ylabel(main_col)
    ax_ts.grid(True, alpha=0.3)
    if len(plot_cols) > 1:
        ax_ts.legend(handles, labels, loc="upper left", fontsize=8, ncol=2, framealpha=0.9)

    # 设置时间轴格式
    from matplotlib.dates import DateFormatter
    ax_ts.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))

    if show_events and "events" in features_struct:
        events = features_struct.get("events", [])
        if events:
            times = df.index.to_list()
            for ev in events[:5]:
                s = ev.get("start", 0)
                e = ev.get("end", 0)
                s = max(0, min(s, len(times) - 1))
                e = max(0, min(e, len(times) - 1))
                ax_ts.axvspan(times[s], times[e], alpha=0.15, color='red')

    ax_txt = fig.add_subplot(gs[:, 2])
    ax_txt.axis("off")

    wrapped_text = format_description_for_plot(description, wrap_width=wrap_width)
    text_kwargs = dict(
        ha="left",
        va="top",
        fontsize=10,
        wrap=False,
    )
    if cn_font:
        text_kwargs["fontproperties"] = cn_font

    ax_txt.text(0.0, 1.0, wrapped_text, **text_kwargs)
    ax_txt.set_title("自动生成的中文描述", loc="left", fontsize=11, fontweight='bold', pad=12)

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.94])
    out = Path(output_path) if output_path is not None else Path("output_viz_Traffic.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="可视化 Traffic 样本：画出时间序列 + 中文描述，方便人工检查描述是否准确。"
    )
    parser.add_argument("--jsonl_path", type=str, required=True, help="生成的 JSONL 样本文件路径")
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="要可视化的样本行号（从 0 开始）。与 --sample_id 二选一，如果两者都给则优先 sample_id。",
    )
    parser.add_argument(
        "--sample_id",
        type=str,
        default=None,
        help="要可视化的样本 id，例如 Traffic_L512_start0",
    )
    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        default=None,
        help="要绘制的特征列，支持一个或多个，空格或逗号分隔；默认绘制样本中的 target_cols",
    )
    parser.add_argument(
        "--no_events",
        action="store_true",
        help="不在图中高亮 z-score 检测到的异常事件区域",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="输出图片的分辨率（dpi），默认300，值越高越清晰但文件越大",
    )

    args = parser.parse_args()

    sample, sample_idx = load_sample_from_jsonl(
        jsonl_path=args.jsonl_path,
        sample_index=None if args.sample_id is not None else args.sample_index,
        sample_id=args.sample_id,
    )

    # 构造输出文件名
    dataset = sample.get("dataset", "unknown_dataset")
    raw_features = args.features
    features = None
    if raw_features is not None:
        features = []
        for item in raw_features:
            for token in str(item).split(","):
                col = token.strip()
                if col and col not in features:
                    features.append(col)
    chosen_cols = features or sample.get("target_cols") or [sample.get("target_col", "")]
    feature_str = chosen_cols[0]
    window_length = sample.get("window_length", "L?")
    start_idx = sample.get("start_index", sample.get("start", "0"))

    invalid_chars = set('\\/:*?"<>|')
    def sanitize(s: str) -> str:
        return ''.join('_' if c in invalid_chars else c for c in str(s)).strip()

    safe_dataset = sanitize(dataset)
    safe_feature = sanitize(feature_str)
    safe_window = sanitize(window_length)
    safe_start = sanitize(start_idx)

    output_filename = f"{sample_idx:04d}_{safe_dataset}_{safe_feature}_L{safe_window}_start{safe_start}.png"
    output_path = Path(args.jsonl_path).parent / output_filename

    plot_sample_with_description(
        sample,
        features=chosen_cols,
        show_events=not args.no_events,
        output_path=str(output_path),
        dpi=args.dpi,
    )

    print(f"已生成可视化图片：{output_path}")


if __name__ == "__main__":
    main()
