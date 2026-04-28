# Usage
# cd "D:\Users\Win\Desktop\Study\毕业设计-面向大语言模型的时序推理数据构建及分析系统实现\数据收集\数据集\Exchange Rate"
# python viz_ER_samples.py --jsonl_path ExchangeRate_c0_descriptions.jsonl --feature c0 --sample_index 0
# 运行时必须通过“ --feature”来指定目标列，需要跟运行 generate_descriptions_ExchangeRate.py 时指定的 target_col 列保持一致
# 可通过“ --sample_index x”来指定生成 jsonl 的第 x+1 行进行可视化
# --sample_index 和 --sample_id 二选一，如果两者都给则优先 sample_id

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

# 字体选择和中文支持与其他脚本保持一致
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
    """保留占位函数，兼容旧调用。"""
    return float(len(token))


def _measure_text_width_px(fig, text: str, fontsize: int, font_props: font_manager.FontProperties | None) -> float:
    """使用 Matplotlib 渲染器测量文本像素宽度，避免按字符估算导致提前换行。"""
    renderer = fig.canvas.get_renderer()
    if font_props is not None:
        fp = font_props.copy()
        fp.set_size(fontsize)
    else:
        fp = font_manager.FontProperties(size=fontsize)
    width_px, _, _ = renderer.get_text_width_height_descent(text, fp, ismath=False)
    return float(width_px)


def _wrap_single_line_mixed_text(
    line: str,
    fig,
    max_width_px: float,
    fontsize: int,
    font_props: font_manager.FontProperties | None,
) -> list[str]:
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

    def flush_current():
        nonlocal cur_tokens
        lines.append("".join(cur_tokens).rstrip())
        cur_tokens = []

    overflow_tolerance_px = max(24.0, 0.18 * max_width_px)
    for tok in tokens:
        if not cur_tokens and tok.isspace():
            continue

        if cur_tokens:
            candidate = "".join(cur_tokens) + tok
            cand_w = _measure_text_width_px(fig, candidate, fontsize=fontsize, font_props=font_props)
        else:
            cand_w = _measure_text_width_px(fig, tok, fontsize=fontsize, font_props=font_props)

        if cur_tokens and (cand_w > max_width_px + overflow_tolerance_px):
            if re.fullmatch(r"[，。；：、！？）】》,.;:!?)]", tok):
                cur_tokens.append(tok)
                continue
            flush_current()
            if tok.isspace():
                continue

        cur_tokens.append(tok)

    if cur_tokens or not lines:
        flush_current()

    return lines


def format_description_for_plot(
    description: str,
    fig,
    ax,
    fontsize: int,
    font_props: font_manager.FontProperties | None,
) -> str:
    """
    保留描述文本中的显式换行，再对每一行做智能折行。
    """
    fig.canvas.draw()
    max_width_px = max(80.0, float(ax.get_window_extent().width) * 1.05)

    logical_lines = description.split("\n")
    wrapped_lines: list[str] = []
    for ln in logical_lines:
        if ln.strip() == "":
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(
            _wrap_single_line_mixed_text(
                ln,
                fig=fig,
                max_width_px=max_width_px,
                fontsize=fontsize,
                font_props=font_props,
            )
        )
    return "\n".join(wrapped_lines)


def _compute_title_wrap_width(fig, ax, title_fontsize: int) -> int:
    """根据左侧图表可用宽度估算标题换行阈值，避免标题过早换行。"""
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
    将一个 sample 转成 pandas DataFrame，index 为时间（或索引），columns 为 feature_names。
    对于 Exchange Rate 数据集，time 是整数索引，因此先尝试转 datetime，失败时保持原样。
    """
    times = sample.get("time", [])
    # 如果是字符串，则转为 datetime，否则保留原始序列
    if len(times) > 0 and isinstance(times[0], str):
        times = pd.to_datetime(times)
    else:
        # numeric index,直接用RangeIndex或Int64Index
        times = pd.Index(times)

    feature_names = sample.get("feature_names", [])
    values = sample.get("values", [])

    df = pd.DataFrame(values, columns=feature_names, index=times)
    return df


def plot_sample_with_description(
    sample: dict,
    feature: str | None = None,
    show_events: bool = True,
    output_path: str | None = None,
    dpi: int = 300,
):
    """
    在一张图中：左侧画时间序列，右侧显示中文描述。
    如果提供 feature，将画该列；否则使用 sample 中的 target_col 或第一列。
    可选在异常事件上绘制高亮。
    """
    df = sample_to_dataframe(sample)

    if feature is None:
        feature = sample.get("target_col") or (df.columns[0] if len(df.columns) else None)

    if feature not in df.columns:
        raise ValueError(f"feature '{feature}' 不在 sample['feature_names'] 中：{df.columns.tolist()}")

    desc_list = sample.get("descriptions", [])
    description = desc_list[0] if desc_list else "(样本中没有 descriptions 字段)"
    features_struct = sample.get("features", {})

    fig = plt.figure(figsize=(15.5, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.3, 2.3, 1.9])

    ax_ts = fig.add_subplot(gs[:, :2])
    ax_ts.plot(df.index, df[feature], linewidth=1.25)
    title_fontsize = 11
    raw_title = f"{sample.get('id', '')}  |  {feature}"
    title_wrap_width = _compute_title_wrap_width(fig, ax_ts, title_fontsize=title_fontsize)
    title_text = "\n".join(
        textwrap.wrap(raw_title, width=title_wrap_width, break_long_words=False, break_on_hyphens=False)
    )
    title_artist = ax_ts.set_title(title_text, fontsize=title_fontsize, fontweight='bold', loc='center', pad=10)
    title_artist.set_multialignment("center")
    ax_ts.set_xlabel("时间")
    ax_ts.set_ylabel(feature)
    ax_ts.grid(True, alpha=0.3)

    # 如果时间为 datetime，格式化刻度
    try:
        from matplotlib.dates import DateFormatter

        if pd.api.types.is_datetime64_any_dtype(df.index):
            ax_ts.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
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

    wrapped_text = format_description_for_plot(
        description,
        fig=fig,
        ax=ax_txt,
        fontsize=10,
        font_props=cn_font,
    )
    text_kwargs = dict(
        ha="left",
        va="top",
        fontsize=10,
        wrap=False,
    )
    if cn_font:
        text_kwargs["fontproperties"] = cn_font

    ax_txt.text(0.0, 1.0, wrapped_text, **text_kwargs)
    ax_txt.set_title("自动生成的中文描述", loc="left", fontsize=11, pad=12)

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.94])
    out = Path(output_path) if output_path is not None else Path("output_viz_ER.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="可视化 Exchange Rate 描述样本：画出时间序列 + 中文描述，方便核对生成文本。"
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
        help="要可视化的样本 id，例如 exchange_rate_c0_L512_start0",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default=None,
        help="要绘制的特征名，例如 c0，默认取 sample['target_col'] 或第一列",
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

    dataset = sample.get("dataset", "unknown")
    feature = args.feature or sample.get("target_col", "")
    window_length = sample.get("window_length", "?")
    start_idx = sample.get("start_index", sample.get("start", 0))

    invalid_chars = set('\\/:*?"<>|')
    def sanitize(s: str) -> str:
        return ''.join('_' if c in invalid_chars else c for c in str(s)).strip()

    safe_dataset = sanitize(dataset)
    safe_feature = sanitize(feature)
    safe_window = sanitize(window_length)
    safe_start = sanitize(start_idx)

    output_filename = f"{sample_idx:04d}_{safe_dataset}_{safe_feature}_L{safe_window}_start{safe_start}.png"
    output_path = Path(args.jsonl_path).parent / output_filename

    plot_sample_with_description(
        sample,
        feature=args.feature,
        show_events=not args.no_events,
        output_path=str(output_path),
        dpi=args.dpi,
    )

    print(f"已生成可视化图片：{output_path}")


if __name__ == "__main__":
    main()
