# Usage
# cd "D:\Users\Win\Desktop\Study\毕业设计-面向大语言模型的时序推理数据构建及分析系统实现\数据收集\数据集\Monash Time Series Forecasting Archive"
# python viz_Monash_samples.py --jsonl_path Monash_electricity_hourly_dataset_descriptions.jsonl --sample_index 0
# 运行时无需指定 feature，因为 Monash 样本本质上是单变量序列；
# id 或 sample_index 选其一（若两者都给则优先 sample_id）。
# 注意 Monash 数据集的时间单位各不相同，脚本会优先读取样本内的 frequency 字段；
# 若缺失再尝试根据样本 id 或 dataset 名称推断频率（如 "hour"→H、"sec"→S），
# 也可通过 --freq 参数显式指定 pandas 频率字符串（H/T/S/...）。
# 如果无法推断，索引将使用简单的整数序号

import json
from pathlib import Path
import argparse
import re
import textwrap
import pandas as pd
from pandas.tseries.frequencies import to_offset
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


def infer_freq_from_name(name: str) -> str | None:
    """根据字符串中某些关键词简单猜测频率。"""
    key = name.lower()
    if "hour" in key:
        return "H"
    if "min" in key or "minute" in key:
        return "T"
    if "sec" in key or "second" in key:
        return "S"
    if "day" in key:
        return "D"
    return None


def infer_freq_from_monash_frequency(monash_freq: str | None) -> str | None:
    if not monash_freq:
        return None
    key = str(monash_freq).strip().lower().replace('-', '_').replace(' ', '_')
    mapping = {
        "yearly": "YS",
        "quarterly": "QS",
        "monthly": "MS",
        "weekly": "W",
        "daily": "D",
        "hourly": "h",
        "half_hourly": "30min",
        "quarter_hourly": "15min",
        "minutely": "min",
        "secondly": "s",
    }
    if key in mapping:
        return mapping[key]
    m = re.match(r'^(\d+)_?(second|minute|hour|day|week|month|quarter|year)s?$', key)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit == "second":
            return f"{n}s"
        if unit == "minute":
            return f"{n}min"
        if unit == "hour":
            return f"{n}h"
        if unit == "day":
            return f"{n}D"
        if unit == "week":
            return f"{n}W"
        if unit == "month":
            return f"{n}MS"
        if unit == "quarter":
            return f"{n}QS"
        if unit == "year":
            return f"{n}YS"
    return None


def normalize_pandas_freq(freq: str | None) -> str | None:
    if not freq:
        return None
    key = str(freq).strip()
    simple = {
        "H": "h",
        "h": "h",
        "T": "min",
        "min": "min",
        "S": "s",
        "s": "s",
    }
    if key in simple:
        return simple[key]
    return key


def sample_to_dataframe(sample: dict, freq: str | None = None) -> pd.DataFrame:
    """
    将 Monash 样本转换成 pandas DataFrame。
    样本中只有一列 "values"；索引用 start_timestamp 开始并根据 freq 生成，
    频率仅使用命令行参数或样本内 frequency 字段；若未知则用时间点索引。
    返回的 DataFrame 列名为 series_name 或 "value"。
    """
    vals = sample.get("values", [])
    series_name = sample.get("series_name") or "value"
    start_index = int(sample.get("start_index", 0))

    # 尝试推断频率
    original_freq = freq
    freq_source = None
    freq = normalize_pandas_freq(freq)
    if freq is None:
        # 优先读取样本内记录的 Monash 原始频率
        freq = infer_freq_from_monash_frequency(sample.get("frequency"))
        if freq is not None:
            freq_source = "样本 frequency 字段"
    # 不再默认设置小时
    if original_freq is None:
        if freq is not None:
            source = freq_source or "自动推断"
            print(f"[info] 使用频率 '{freq}' 来构造索引（{source}）。")
        else:
            print("[info] 缺少可用频率，索引将使用时间点。")
    else:
        print(f"[info] 使用命令行指定的频率 '{freq}' 来构造索引。")

    idx = None
    start_ts = sample.get("start_timestamp")
    if start_ts:
        try:
            ts0 = pd.to_datetime(start_ts)
            if freq is not None:
                # 左图应显示窗口真实时间：序列起点 + start_index 偏移
                offset = to_offset(freq)
                ts_start = ts0 + start_index * offset
                idx = pd.date_range(start=ts_start, periods=len(vals), freq=offset)
            else:
                idx = pd.RangeIndex(start=start_index, stop=start_index + len(vals), step=1)
        except Exception:
            idx = pd.RangeIndex(start=start_index, stop=start_index + len(vals), step=1)
    else:
        idx = pd.RangeIndex(start=start_index, stop=start_index + len(vals), step=1)

    df = pd.DataFrame({series_name: vals}, index=idx)
    return df


def plot_sample_with_description(
    sample: dict,
    show_events: bool = True,
    output_path: str | None = None,
    dpi: int = 300,
    freq: str | None = None,
):
    """
    绘图：左图为时间序列，右图显示描述文本。
    """
    df = sample_to_dataframe(sample, freq=freq)
    series_name = df.columns[0] if len(df.columns) else "value"

    desc_list = sample.get("descriptions", [])
    description = desc_list[0] if desc_list else "(样本中没有 descriptions 字段)"
    features_struct = sample.get("features", {})

    fig = plt.figure(figsize=(15.5, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.3, 2.3, 1.9])

    ax_ts = fig.add_subplot(gs[:, :2])
    ax_ts.plot(df.index, df[series_name], linewidth=1.25)
    title_fontsize = 11
    raw_title = f"{sample.get('id', '')}  |  {series_name}"
    title_wrap_width = _compute_title_wrap_width(fig, ax_ts, title_fontsize=title_fontsize)
    title_text = "\n".join(
        textwrap.wrap(raw_title, width=title_wrap_width, break_long_words=False, break_on_hyphens=False)
    )
    title_artist = ax_ts.set_title(title_text, fontsize=title_fontsize, fontweight='bold', loc='center', pad=10)
    title_artist.set_multialignment("center")
    if pd.api.types.is_datetime64_any_dtype(df.index):
        ax_ts.set_xlabel("时间")
    else:
        ax_ts.set_xlabel("时间点")
    ax_ts.set_ylabel(series_name)
    ax_ts.grid(True, alpha=0.3)

    try:
        from matplotlib.dates import DateFormatter
        if pd.api.types.is_datetime64_any_dtype(df.index):
            ax_ts.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H'))
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
    out = Path(output_path) if output_path is not None else Path("output_viz_Monash.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="可视化 Monash 数据集描述样本：画出时间序列 + 中文描述。"
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
        help="要可视化的样本 id，例如 Monash_electricity_hourly_dataset_T2_L512_start0",
    )
    parser.add_argument(
        "--no_events",
        action="store_true",
        help="不高亮 z-score 检测到的异常事件",
    )
    parser.add_argument("--freq",
        type=str,
        default=None,
        help="时间序列频率字符串，如 H/T/S 等；缺省时从 id/dataset 名称猜测，若无法猜测则默认按小时。",
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

    # 输出文件名使用行号前缀 + sample.id 保证可排序且与内容一致。
    sample_id = sample.get("id", f"Monash_{sample_idx}")
    invalid_chars = set('\\/:*?"<>|')
    def sanitize(s: str) -> str:
        return ''.join('_' if c in invalid_chars else c for c in str(s)).strip()
    output_filename = f"{sample_idx:04d}_{sanitize(sample_id)}.png"
    output_path = Path(args.jsonl_path).parent / output_filename

    plot_sample_with_description(
        sample,
        show_events=not args.no_events,
        output_path=str(output_path),
        dpi=args.dpi,
        freq=args.freq,
    )

    print(f"已生成可视化图片：{output_path}")


if __name__ == "__main__":
    main()
