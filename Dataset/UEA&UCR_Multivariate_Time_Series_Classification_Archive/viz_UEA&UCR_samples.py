# Usage
# cd "D:\Users\Win\Desktop\Study\毕业设计-面向大语言模型的时序推理数据构建及分析系统实现\数据收集\数据集\UEA&UCR Multivariate Time Series Classification Archive"
# python "viz_UEA&UCR_samples.py" --jsonl_path BinaryHeartbeat_BinaryHeartbeat_TRAIN_descriptions.jsonl --sample_index 0
# python "viz_UEA&UCR_samples.py" --jsonl_path BasicMotions_BasicMotions_TRAIN_L32_descriptions.jsonl --sample_index 0 --main_channels 0,2,5
# 参数说明：
#   --jsonl_path   : 生成的 JSONL 样本文件路径（必须）
#   --sample_index : 样本行号（从0开始），与 --sample_id 二选一，如果两者都给则优先 sample_id。
#   --sample_id    : 样本 id，例如 BinaryHeartbeat_BinaryHeartbeat_TRAIN_idx0_L512_start0
#   --main_channels: 指定一个或多个通道（如 0,2,5 或 ch0,ch2），语义对齐 target_cols
#   --no_events    : 不在图中高亮 z-score 检测到的异常事件区域
#   --dpi          : 输出图片的分辨率（dpi），默认300
# 兼容旧参数：可继续使用 --feature（单通道）或 --main-channels（连字符写法）。

import json
from pathlib import Path
import argparse
import re
import pandas as pd
import matplotlib
# 使用无显示后端以便在没有显示服务器的环境中也能保存 PNG
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path as _Path

# 字体配置与其它可视化脚本保持一致
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


def load_sample_from_jsonl(jsonl_path: str, sample_index: int | None = None, sample_id: str | None = None):
    """
    从 JSONL 文件里读取一个样本，返回 (样本对象, 行索引)。
    如果给出 sample_id，则按 id 精确匹配并返回其所在行编号。
    否则使用 sample_index（默认0）。
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
    将样本的 values 列表转换为 DataFrame，索引为整数序列。
    列名使用 sample 中的 feature 姓或默认 "value"。
    """
    vals = sample.get("values", [])
    if not vals:
        return pd.DataFrame()

    # 多变量: values 形如 [[t0_ch0, t0_ch1, ...], [t1_ch0, t1_ch1, ...], ...]
    if isinstance(vals[0], list):
        n_cols = len(vals[0])
        colnames = sample.get("target_cols") or [f"ch{i}" for i in range(n_cols)]
        if len(colnames) != n_cols:
            colnames = [f"ch{i}" for i in range(n_cols)]
        return pd.DataFrame(vals, columns=colnames)

    # 单变量
    colname = sample.get("feature_names") or sample.get("target_col") or "value"
    return pd.DataFrame(vals, columns=[colname])


def _parse_main_channels_arg(main_channels: str | None, df: pd.DataFrame) -> list[str] | None:
    if not main_channels:
        return None
    raw = [x for x in re.split(r"[\s,]+", main_channels.strip()) if x]
    if not raw:
        return None

    picked = []
    for tok in raw:
        # 支持数字索引（如 0,1）和列名（如 ch0,ch2）
        if tok.isdigit():
            idx = int(tok)
            if 0 <= idx < len(df.columns):
                picked.append(df.columns[idx])
            else:
                raise ValueError(f"main-channels 索引越界: {idx}，可用范围 0~{len(df.columns)-1}")
        else:
            if tok not in df.columns:
                raise ValueError(f"main-channels 指定列不存在: {tok}，可用列: {df.columns.tolist()}")
            picked.append(tok)

    # 去重且保持顺序
    dedup = []
    seen = set()
    for c in picked:
        if c not in seen:
            dedup.append(c)
            seen.add(c)
    return dedup if dedup else None


def _get_main_channel_idx(sample: dict) -> int:
    v = sample.get("main_channel")
    if v is None:
        v = sample.get("features", {}).get("main_channel", 0)
    try:
        return int(v)
    except Exception:
        return 0


def _get_strong_linked_channel_indices(sample: dict, main_idx: int, thr: float = 0.25) -> list[int]:
    corr_list = sample.get("features", {}).get("cross_channel_correlations", []) or []
    out: list[int] = []
    for item in corr_list:
        chs = item.get("channels", [])
        if not isinstance(chs, list) or len(chs) != 2:
            continue
        a, b = int(chs[0]), int(chs[1])
        rho = float(item.get("correlation", 0.0))
        stable = bool(item.get("stable", True))
        other = b if a == main_idx else (a if b == main_idx else None)
        if other is None:
            continue
        if stable and abs(rho) >= thr:
            out.append(other)
    # 去重保序
    dedup = []
    seen = set()
    for x in out:
        if x not in seen:
            dedup.append(x)
            seen.add(x)
    return dedup


def _map_idx_to_col(df: pd.DataFrame, idx: int) -> str | None:
    c = f"ch{idx}"
    if c in df.columns:
        return c
    if 0 <= idx < len(df.columns):
        return str(df.columns[idx])
    return None


def _col_to_idx(df: pd.DataFrame, col: str) -> int | None:
    m = re.fullmatch(r"ch(\d+)", str(col))
    if m:
        return int(m.group(1))
    try:
        return int(df.columns.get_loc(col))
    except Exception:
        return None


def _build_linkage_text(sample: dict, df: pd.DataFrame, selected_cols: list[str]) -> str:
    main_idx = _get_main_channel_idx(sample)
    corr_list = sample.get("features", {}).get("cross_channel_correlations", []) or []

    # 建立 main -> other 映射
    rel_map: dict[int, tuple[float, bool]] = {}
    for item in corr_list:
        chs = item.get("channels", [])
        if not isinstance(chs, list) or len(chs) != 2:
            continue
        a, b = int(chs[0]), int(chs[1])
        rho = float(item.get("correlation", 0.0))
        stable = bool(item.get("stable", True))
        if a == main_idx:
            rel_map[b] = (rho, stable)
        elif b == main_idx:
            rel_map[a] = (rho, stable)

    selected_non_main_idx = []
    for c in selected_cols:
        idx = _col_to_idx(df, c)
        if idx is None or idx == main_idx:
            continue
        selected_non_main_idx.append(idx)

    # 去重保序
    dedup = []
    seen = set()
    for x in selected_non_main_idx:
        if x not in seen:
            dedup.append(x)
            seen.add(x)
    selected_non_main_idx = dedup

    if not selected_non_main_idx:
        return "联动关系：\n当前图中未包含非主通道。"

    parts = []
    for ch in selected_non_main_idx:
        if ch not in rel_map:
            parts.append(f"通道 {ch} 与主通道 {main_idx} 联动信息不足")
            continue
        rho, stable = rel_map[ch]
        if (not stable):
            parts.append(f"通道 {ch} 与主通道 {main_idx} 联动不稳定（r≈{rho:.2f}）")
        elif abs(rho) >= 0.25:
            rel = "同向" if rho >= 0 else "反向"
            parts.append(f"通道 {ch} 与主通道 {main_idx} 呈{rel}关系（r≈{rho:.2f}）")
        else:
            parts.append(f"通道 {ch} 与主通道 {main_idx} 联动较弱（r≈{rho:.2f}）")

    return "联动关系：\n" + "；\n".join(parts) + "。"


def _replace_linkage_section(description: str, linkage_text: str) -> str:
    text = description or ""
    marker = "\n\n联动关系："
    overall = "\n\n整体结论："

    if marker in text:
        left = text.split(marker)[0]
        right = text.split(marker, 1)[1]
        # 删除原联动段到整体结论前
        if overall in right:
            right_tail = overall + right.split(overall, 1)[1]
        else:
            right_tail = ""
        return left.rstrip() + "\n\n" + linkage_text + right_tail

    if overall in text:
        head = text.split(overall)[0]
        tail = overall + text.split(overall, 1)[1]
        return head.rstrip() + "\n\n" + linkage_text + tail

    return text.rstrip() + "\n\n" + linkage_text


def plot_sample_with_description(
    sample: dict,
    feature: str | None = None,
    main_channels: str | None = None,
    wrap_width: int = 32,
    show_events: bool = True,
    output_path: str | None = None,
    dpi: int = 300,
):
    """
    画出指定 feature 的时间序列以及对应的中文描述。
    """
    df = sample_to_dataframe(sample)
    if len(df.columns) == 0:
        raise ValueError("样本 values 为空，无法绘图")

    forced_cols = _parse_main_channels_arg(main_channels, df)
    main_idx = _get_main_channel_idx(sample)

    selected_cols = forced_cols
    if selected_cols is None:
        if feature is not None:
            if feature not in df.columns:
                raise ValueError(f"feature '{feature}' 不在数据列中：{df.columns.tolist()}")
            selected_cols = [feature]
        elif len(df.columns) == 1:
            selected_cols = [df.columns[0]]
        else:
            # 默认：主通道 + 与主通道联动强的通道
            main_col = _map_idx_to_col(df, main_idx)
            selected_cols = [main_col] if main_col is not None else [df.columns[0]]
            strong_idx = _get_strong_linked_channel_indices(sample, main_idx, thr=0.25)
            for idx in strong_idx:
                c = _map_idx_to_col(df, idx)
                if c is not None and c not in selected_cols:
                    selected_cols.append(c)

    multi_plot = len(selected_cols) > 1

    desc_list = sample.get("descriptions", [])
    description = desc_list[0] if desc_list else "(样本中没有 descriptions 字段)"
    features_struct = sample.get("features", {})

    # 右侧联动关系描述严格跟随当前绘图通道
    if len(df.columns) > 1:
        linkage_text = _build_linkage_text(sample, df, selected_cols)
        description = _replace_linkage_section(description, linkage_text)

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 2.2, 1.6])

    # 左侧时间序列
    ax_ts = fig.add_subplot(gs[:, :2])
    if multi_plot:
        for c in selected_cols:
            ax_ts.plot(df.index, df[c], linewidth=1.0, alpha=0.9, label=str(c))
        ax_ts.legend(loc="upper right", fontsize=8, ncol=2)
        title_feat = ",".join(map(str, selected_cols))
    else:
        one = selected_cols[0]
        ax_ts.plot(df.index, df[one], linewidth=1.2)
        title_feat = str(one)
    ax_ts.set_title(f"{sample.get('id','')}  |  {title_feat}", fontsize=12, fontweight='bold')
    ax_ts.set_xlabel("时间点")
    ax_ts.set_ylabel(selected_cols[0] if (not multi_plot) else "value")
    ax_ts.grid(True, alpha=0.3)

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

    # 右侧文本描述
    ax_txt = fig.add_subplot(gs[:, 2])
    ax_txt.axis("off")

    wrapped_text = format_description_for_plot(description, wrap_width=wrap_width)

    text_kwargs = dict(ha="left", va="top", fontsize=10, wrap=True)
    if cn_font:
        text_kwargs["fontproperties"] = cn_font
    ax_txt.text(0.0, 1.0, wrapped_text, **text_kwargs)
    ax_txt.set_title("自动生成的中文描述", loc="left", fontsize=11, fontweight='bold')

    plt.tight_layout()
    out = Path(output_path) if output_path is not None else Path("output_viz_UEA&UCR.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="可视化 UEA&UCR 描述样本：画出时间序列 + 中文描述，方便肉眼检查描述是否准确。"
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
        help="要可视化的样本 id，例如 BinaryHeartbeat_BinaryHeartbeat_TRAIN_idx0_L512_start0",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default=None,
        help="兼容旧参数：要绘制的单个特征名",
    )
    parser.add_argument(
        "--main-channels",
        "--main_channels",
        type=str,
        default=None,
        help="指定一个或多个通道（如 0 或 0,2,5 或 ch0,ch2）；语义与 target_cols 类似",
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
    inst = sample.get("instance_index", "?")
    label = sample.get("class_label", "")
    window_length = sample.get("window_length", "L?")
    start_idx = sample.get("start_index", sample.get("start", "0"))

    invalid_chars = set('\\/:*?"<>|')
    def sanitize(s: str) -> str:
        return ''.join('_' if c in invalid_chars else c for c in str(s)).strip()

    safe_dataset = sanitize(dataset)
    safe_label = sanitize(label)
    safe_inst = sanitize(inst)
    safe_window = sanitize(window_length)
    safe_start = sanitize(start_idx)

    ch_tag = ""
    if args.main_channels:
        ch_tag = "_chs" + sanitize(args.main_channels.replace(",", "-"))
    elif args.feature:
        ch_tag = "_ch" + sanitize(args.feature)

    output_filename = f"{sample_idx:04d}_{safe_dataset}_idx{safe_inst}_cls{safe_label}_L{safe_window}_start{safe_start}{ch_tag}.png"
    output_path = Path(args.jsonl_path).parent / output_filename

    plot_sample_with_description(
        sample,
        feature=args.feature,
        main_channels=args.main_channels,
        show_events=not args.no_events,
        output_path=str(output_path),
        dpi=args.dpi,
    )

    print(f"✅ 已生成可视化图片：{output_path}")


if __name__ == "__main__":
    main()
