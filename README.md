# TS-LLM: 面向大语言模型的时序推理数据构建及分析系统

> 本科毕业设计项目 | Time Series Large Language Model Project

本项目实现了一个完整的"时序数据 → 文本描述 → 大语言模型推理" pipeline，包含多源时序数据集收集与描述生成、时序编码器训练、以及基于 [ts-caption-eval](https://github.com/Ringhu/ts-caption-eval) 的三维度评测体系。

## 项目结构

```
TS_LLM_Project/
├── Dataset/                  # 多源时序数据集、描述生成脚本与可视化脚本
│   ├── ElectricityECL/       #   电力消耗数据集与描述生成脚本
│   ├── ETT-small/            #   ETT 数据集（电力变压器温度）
│   ├── Exchange_Rate/        #   汇率数据集
│   ├── Monash_Time_Series_Forecasting_Archive/  #   Monash 时序预测归档数据集
│   ├── NAB/                  #   NAB 异常检测数据集
│   ├── Traffic/              #   交通流量数据集
│   ├── UEA&UCR_Multivariate_Time_Series_Classification_Archive/  #   UEA/UCR 多变量分类数据；其中 FruitFlies.arff 与 UrbanSound.arff 以分片形式保存
│   ├── Weather/              #   气象数据集
│   ├── clean_files.bat/sh    #   临时文件清理脚本
│   └── common_desc_adapt.py  #   通用描述适配脚本
├── Models/                   # 预训练大语言模型权重与 tokenizer 文件
│   ├── Qwen2.5-3B-Instruct/  #   3B 指令模型；两个主 safetensors 文件以分片形式保存
│   ├── Qwen3-0.6B-Instruct-2512/  #   0.6B 指令模型
│   └── Qwen3-4B-Instruct-2507/    #   4B 指令模型；两个主 safetensors 文件以分片形式保存
├── Sample/                   # 筛选后的训练/分析样本
│   ├── iteration_1/          #   第 1 轮迭代样本
│   ├── iteration_2/          #   第 2 轮迭代样本
│   ├── iteration_3/          #   第 3 轮迭代样本
│   ├── iteration_4/          #   第 4 轮迭代样本
│   └── run_300k_20260413/    #   大规模运行结果，含 JSONL 与可视化图片
├── Train/                    # 模型训练、推理与样本分析核心代码
│   ├── API_Test/             #   多路 API 探测与评估日志
│   ├── Checkpoints/          #   训练检查点
│   ├── Encoders/             #   时序编码器实现（CNN / MLP / PatchTST）
│   ├── Models/               #   多模态 Qwen 模型封装
│   ├── logs/                 #   训练与 smoke test 日志
│   ├── dataset.py            #   数据集加载器
│   ├── dataset_type_config.py #   数据集类型与字段配置
│   ├── generate_filtered_samples.py  #   样本筛选与生成主脚本
│   ├── infer.py              #   通用推理脚本
│   ├── infer_for_qa.py       #   QA 任务推理输出
│   ├── infer_for_tscapeval.py #   ts-caption-eval 格式推理输出
│   ├── prompting.py          #   自适应提示词构建
│   ├── run_analysis.py       #   样本分析脚本
│   └── train.py              #   训练启动脚本
├── ts-caption-eval/          # 评测框架（forked & adapted）
│   ├── configs/              #   评测配置（smoke / thesis / thesis_qa）
│   ├── data/                 #   caption 与 QA 评测数据
│   ├── docs/                 #   设计说明文档
│   ├── logs/                 #   评测日志
│   ├── predictions/          #   各模型预测文件
│   ├── references/           #   参考答案文件
│   ├── results/              #   评测结果输出
│   ├── scripts/              #   数据准备与辅助脚本
│   ├── tests/                #   smoke test
│   ├── tscapeval/            #   评测核心代码
│   └── tscapeval.egg-info/   #   打包元数据
├── SplitLargeFiles/          # GitHub LFS 单对象 2 GiB 限制下的仓内分片存储
│   ├── Dataset/              #   Dataset 下 2 个超大 .arff 文件的分片
│   ├── Models/               #   Models 下 4 个超大 .safetensors 文件的分片
│   ├── manifest.json         #   原始路径、大小、SHA-256 与分片映射
│   ├── build_parts.ps1       #   分片生成脚本（维护用）
│   ├── reassemble.ps1        #   分片重组与校验脚本
│   └── README.md             #   分片目录说明
├── UPLOAD_LIMITATIONS.md     # 超大文件分片与受影响原路径说明
└── README.md                 # 项目说明
```

说明：仓库当前未纳入运行时缓存文件（如 `__pycache__/`、`.pyc`）以及本地敏感配置文件 `ts-caption-eval/.env`。

## 超大文件分片保存与重组

GitHub LFS 对单个对象存在 2 GiB 上限，因此以下 6 个原始大文件没有以“单一原文件”的形式直接存放在对应目录中，而是按字节切分后保存在 `SplitLargeFiles/` 下：

- `Dataset/UEA&UCR_Multivariate_Time_Series_Classification_Archive/Multivariate/FruitFlies/FruitFlies.arff`
- `Dataset/UEA&UCR_Multivariate_Time_Series_Classification_Archive/Multivariate/UrbanSound/UrbanSound.arff`
- `Models/Qwen2.5-3B-Instruct/model-00001-of-00002.safetensors`
- `Models/Qwen2.5-3B-Instruct/model-00002-of-00002.safetensors`
- `Models/Qwen3-4B-Instruct-2507/model-00001-of-00003.safetensors`
- `Models/Qwen3-4B-Instruct-2507/model-00002-of-00003.safetensors`

对应分片信息记录在 `SplitLargeFiles/manifest.json` 中。该文件保存了：

- 原始相对路径
- 原始文件大小
- 原始文件 SHA-256
- 每个分片的路径与大小

### 如何重组分片文件

在仓库根目录执行以下命令即可将所有分片恢复到原始相对路径，并自动校验文件大小与 SHA-256：

```powershell
powershell -ExecutionPolicy Bypass -File .\SplitLargeFiles\reassemble.ps1 -DestinationRoot .
```

如果只想重组某一个文件，可以指定 `-Target`：

```powershell
powershell -ExecutionPolicy Bypass -File .\SplitLargeFiles\reassemble.ps1 -DestinationRoot . -Target 'Models/Qwen2.5-3B-Instruct/model-00002-of-00002.safetensors'
```

如果你不希望把重组后的大文件直接写回当前仓库目录，也可以指定一个单独的输出目录：

```powershell
powershell -ExecutionPolicy Bypass -File .\SplitLargeFiles\reassemble.ps1 -DestinationRoot D:\restored_ts_llm_project
```

重组脚本会逐个拼接分片，并在完成后验证目标文件的大小和 SHA-256；如果校验不通过，脚本会直接报错，避免得到静默损坏的结果。

## 快速开始

### 环境准备

```bash
# 任选其一创建并激活 Python 虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# 或
source .venv/bin/activate        # Linux / macOS

# 安装依赖
pip install torch transformers accelerate openai pyyaml sacrebleu rouge_score bert_score
```

### 1. 生成时序描述

各数据集目录下均配有 `generate_descriptions_*.py` 脚本和 `viz_*_samples.py` 可视化脚本：

```bash
cd Dataset/ETT-small/
python generate_descriptions_ETT.py
python viz_ETT_samples_v2.py
```

### 2. 生成筛选样本

```bash
# 运行样本生成（自动从数据集中筛选合格样本、生成描述与可视化图片）
python Train/generate_filtered_samples.py \
  --target_qualified 10000 \
  --output_base_dir Sample \
  --seed 42
```

### 3. 训练模型

```bash
# 单卡训练（CNN 编码器示例）
CUDA_VISIBLE_DEVICES=0 python -u Train/train.py \
  --training_stage frozen \
  --model_path Models/Qwen3-4B-Instruct-2507 \
  --batch_size 1 \
  --num_workers 4 \
  --precision bf16 \
  --encoder_type cnn \
  --train_jsonl Sample/run_300k_20260413/combined_jsonl.jsonl \
  --preview_max_new_tokens 192

# 三卡并行（CNN / MLP / PatchTST 同时训练）
IDLE_GPUS=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits | awk -F', *' '$2<1000 && $3<10 {print $1}' | head -n 3 | xargs)
set -- $IDLE_GPUS
CUDA_VISIBLE_DEVICES=$1 python -u Train/train.py --encoder_type cnn ... > cnn.log 2>&1 &
CUDA_VISIBLE_DEVICES=$2 python -u Train/train.py --encoder_type mlp ... > mlp.log 2>&1 &
CUDA_VISIBLE_DEVICES=$3 python -u Train/train.py --encoder_type patchtst ... > patchtst.log 2>&1 &
```

### 4. 生成推理文件

```bash
# 生成 ts-caption-eval 格式 Reference-based + LLM-as-judge 评测的预测文件
python Train/infer_for_tscapeval.py

# 生成 ts-caption-eval 格式 QA 任务的预测文件
python Train/infer_for_qa.py
```

### 5. 运行评测

```bash
cd ts-caption-eval

# Reference-based + LLM-as-judge 评测
python -m tscapeval --config configs/thesis.yaml

# Downstream QA 评测
python -m tscapeval --config configs/thesis_qa.yaml
```

评测结果将输出至 `ts-caption-eval/results/` 目录下，包含：
- `main_table.json/csv/md` — 汇总指标表
- `per_sample/` — 逐样本评测结果
- `per_dataset/` — 分数据集统计

## 评测体系

本项目采用三维度互补评测框架：

| 评测维度 | 指标 | 说明 |
|---------|------|------|
| **Reference-based** | BLEU-4, ROUGE-L, BERTScore-F1 | 与强 LLM 生成的参考 caption 的文本相似度 |
| **LLM-as-judge** | Faithfulness, Completeness (1-5 Likert) | 由 LLM 评判 caption 的忠实度和完整性 |
| **Downstream QA** | Accuracy (meta_only / caption / wrong_caption) | caption 对下游多选题任务的实用性 |

## 技术栈

- **大语言模型**: Qwen2.5-3B, Qwen3-0.6B, Qwen3-4B
- **时序编码器**: CNN 1D, MLP, PatchTST
- **深度学习框架**: PyTorch, Transformers, Accelerate
- **API 服务**: 智谱 GLM、硅基流动、豆包（多路并发）
- **评测框架**: ts-caption-eval（BLEU, ROUGE, BERTScore, LLM-judge）
- **数据处理**: NumPy, SciPy, Pandas

## 数据集来源

| 数据集 | 领域 | 来源 |
|-------|------|------|
| ETT-small | 电力变压器温度 | [GitHub](https://github.com/zhouhaoyi/ETDataset) |
| ElectricityECL | 电力消耗 | UCI ML Repository |
| Exchange_Rate | 汇率 | [GitHub](https://github.com/laiguokun/multivariate-time-series-data) |
| Monash | 多领域时序预测 | [Monash Forecasting Repository](https://forecastingdata.org/) |
| NAB | 异常检测 | [Numenta Anomaly Benchmark](https://github.com/numenta/NAB) |
| Traffic | 交通流量 | UCI ML Repository |
| Weather | 气象数据 | Max Planck Institute |

## 许可证

本项目仅供学术研究使用。

## 引用

如果本项目对您的研究有帮助，欢迎引用：

```bibtex
@misc{ts-llm-2026,
  title={TS-LLM: Time Series Reasoning Data Construction and Analysis System for Large Language Models},
  author={GloamingCity},
  year={2026},
  howpublished={GitHub},
  url={https://github.com/GloamingCity/Implementation-of-a-Time-Series-Data-Construction-and-Analysis-System-Based-on-Large-Language-Models}
}
```
