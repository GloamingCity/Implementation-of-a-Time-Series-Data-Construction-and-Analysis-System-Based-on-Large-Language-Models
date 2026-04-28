# Usage
# cd D:\Users\Win\Desktop\Study\毕业设计-面向大语言模型的时序推理数据构建及分析系统实现\数据收集\数据集\ElectricityECL
# python viz_ElectricityECL_samples.py --jsonl_path ElectricityECL_MT_200_L512-1024_descriptions.jsonl --sample_index 0
# 不传 --feature 时会自动识别目标电表列（优先 target_col，其次 sample id，再次 jsonl 文件名）
# 可通过“ --sample_index x”来指定生成jsonl第x+1行的可视化图像

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
                    return obj, idx  # 返回样本和索引
        raise ValueError(f"在 {jsonl_path} 中未找到 id={sample_id} 的样本")

    if sample_index is None:
        sample_index = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == sample_index:
                return json.loads(line), i  # 返回样本和索引

    raise IndexError(f"sample_index={sample_index} 超出范围")


def sample_to_dataframe(sample: dict) -> pd.DataFrame:
    """
    将一个 sample 转成 pandas DataFrame，index 为时间戳，columns 为 feature_names。
    ElectricityECL 数据集的 feature_names 包含 MT_001 到 MT_370 等电表列。
    """
    times = pd.to_datetime(sample["time"])
    feature_names = sample["feature_names"]
    values = sample["values"]

    df = pd.DataFrame(values, columns=feature_names, index=times)
    return df


def infer_feature_from_sample(sample: dict, jsonl_path: str) -> str:
    """
    自动识别要绘制的电表列。
    优先级：sample['target_col'] > sample['id'] > jsonl 文件名。
    若无法识别则抛错，避免静默回退到 MT_001 造成误导。
    """
    feature_names = sample.get("feature_names", [])

    target_col = sample.get("target_col")
    if isinstance(target_col, str) and target_col in feature_names:
        return target_col

    sample_id = str(sample.get("id", ""))
    m_id = re.search(r"ElectricityECL_(MT_\d+)_L\d+_start\d+", sample_id)
    if m_id and m_id.group(1) in feature_names:
        return m_id.group(1)

    jsonl_name = Path(jsonl_path).name
    m_file = re.search(r"ElectricityECL_(MT_\d+)_descriptions\.jsonl", jsonl_name)
    if m_file and m_file.group(1) in feature_names:
        return m_file.group(1)

    raise ValueError(
        "无法自动识别电表列，请显式传入 --feature（例如 MT_200）。"
    )


def plot_sample_with_description(
    sample: dict,
    feature: str = None,
    wrap_width: int = 32,
    show_events: bool = True,
    output_path: str | None = None,
    dpi: int = 300,
):
    """
    在同一张图中：
    - 左侧子图：画出指定 feature 的时间序列（电表读数）
    - 右侧子图：展示该窗口的中文描述（descriptions[0]）
    可选：在曲线上标出 z-score 检测到的异常事件区域。
    """
    df = sample_to_dataframe(sample)

    # 如果没有指定 feature，使用第一个 MT_xxx 列
    # feature 由 main() 统一决定，plot 函数内不再做静默回退。
    if feature is None:
        raise ValueError("未指定 feature，且未完成自动识别。")

    if feature not in df.columns:
        raise ValueError(f"feature '{feature}' 不在 sample['feature_names'] 中：{df.columns.tolist()}")

    desc_list = sample.get("descriptions", [])
    description = desc_list[0] if desc_list else "(样本中没有 descriptions 字段)"
    features_struct = sample.get("features", {})

    # 准备画布：左 2/3 画曲线，右 1/3 显示文本
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 2.2, 1.6])

    # ----- 左侧：时间序列 -----
    ax_ts = fig.add_subplot(gs[:, :2])
    ax_ts.plot(df.index, df[feature], linewidth=1.2)
    ax_ts.set_title(f"{sample.get('id', '')}  |  {feature}", fontsize=12, fontweight='bold')
    ax_ts.set_xlabel("时间")
    ax_ts.set_ylabel("电表读数")
    ax_ts.grid(True, alpha=0.3)
    
    # 设置时间轴格式
    from matplotlib.dates import DateFormatter
    ax_ts.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))

    # 可选：标出异常事件（z-score）
    if show_events and "events" in features_struct:
        events = features_struct["events"]
        if events:
            times = df.index.to_list()
            for ev in events[:5]:  # 最多标前 5 个
                s = ev["start"]
                e = ev["end"]
                s = max(0, min(s, len(times) - 1))
                e = max(0, min(e, len(times) - 1))
                ax_ts.axvspan(times[s], times[e], alpha=0.15, color='red')

    # ----- 右侧：文本描述 -----
    ax_txt = fig.add_subplot(gs[:, 2])
    ax_txt.axis("off")

    wrapped_text = format_description_for_plot(description, wrap_width=wrap_width)
    text_kwargs = dict(
        ha="left",
        va="top",
        fontsize=10,
        wrap=True,
    )
    if cn_font:
        text_kwargs["fontproperties"] = cn_font

    ax_txt.text(0.0, 1.0, wrapped_text, **text_kwargs)
    ax_txt.set_title("自动生成的中文描述", loc="left", fontsize=11, fontweight='bold')

    plt.tight_layout()
    out = Path(output_path) if output_path is not None else Path("output_viz_ElectricityECL.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="可视化 ElectricityECL 描述样本：画出时间序列 + 中文描述，方便肉眼检查描述是否准确。"
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
        help="要可视化的样本 id，例如 ElectricityECL_MT_001_L512_start0",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default=None,
        help="要绘制的电表列名（如 MT_001），默认使用第一个 MT_xxx 列",
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

    # 加载样本并获取索引
    sample, sample_idx = load_sample_from_jsonl(
        jsonl_path=args.jsonl_path,
        sample_index=None if args.sample_id is not None else args.sample_index,
        sample_id=args.sample_id,
    )

    # 生成目标文件名：000X_样本id.png（X为索引，3位数字补零）
    sample_id = sample.get("id", "unknown_id")
    output_filename = f"{sample_idx:04d}_{sample_id}.png"
    output_path = Path(args.jsonl_path).parent / output_filename

    # 未显式指定 --feature 时，严格自动识别，避免误画 MT_001。
    feature = args.feature if args.feature is not None else infer_feature_from_sample(sample, args.jsonl_path)

    plot_sample_with_description(
        sample,
        feature=feature,
        show_events=not args.no_events,
        output_path=str(output_path),
        dpi=args.dpi,
    )

    print(f"✅ 已生成可视化图片：{output_path}")


if __name__ == "__main__":
    main()
