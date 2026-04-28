# TS-LLM: 面向大语言模型的时序推理数据构建及分析系统

> 本科毕业设计项目 | Time Series Large Language Model Project

本项目实现了一个完整的"时序数据 → 文本描述 → 大语言模型推理" pipeline，包含多源时序数据集收集与描述生成、时序编码器训练、以及基于 [ts-caption-eval](https://github.com/Ringhu/ts-caption-eval) 的三维度评测体系。

## 项目结构

```
TS_LLM_Project/
├── Dataset/                  # 多源时序数据集 & 描述生成脚本
│   ├── ETT-small/            #   ETT 数据集（电力变压器温度）
│   ├── ElectricityECL/       #   电力消耗数据集
│   ├── Exchange_Rate/        #   汇率数据集
│   ├── Monash_Time_Series/   #   Monash 时序预测归档数据集
│   ├── NAB/                  #   NAB 异常检测数据集
│   ├── Traffic/              #   交通流量数据集
│   ├── Weather/              #   气象数据集
│   ├── Data_Collection.xlsx  #   数据集汇总清单
│   ├── common_desc_adapt.py  #   通用描述适配脚本
│   └── clean_files.bat/sh    #   临时文件清理脚本
├── Models/                   # 预训练大语言模型权重
│   ├── Qwen2.5-3B-Instruct/
│   ├── Qwen3-0.6B-Instruct-2512/
│   └── Qwen3-4B-Instruct-2507/
├── Sample/                   # 筛选后的训练样本
│   ├── iteration_1~4/        #   多轮迭代筛选结果
│   └── run_300k_20260413/    #   最终运行样本（含可视化图片）
├── Train/                    # 模型训练与推理核心代码
│   ├── Encoders/             #   时序编码器实现（CNN / MLP / PatchTST）
│   ├── Models/               #   多模态 Qwen 模型封装
│   ├── Checkpoints/          #   训练检查点
│   ├── API_Test/             #   多路 API 探测与评估日志
│   ├── logs/                 #   训练日志
│   ├── train.py              #   训练启动脚本（支持多卡并行）
│   ├── infer.py              #   通用推理脚本
│   ├── infer_for_tscapeval.py#   ts-caption-eval 格式推理输出
│   ├── infer_for_qa.py       #   QA 任务推理输出
│   ├── dataset.py            #   数据集加载器
│   ├── prompting.py          #   自适应提示词构建
│   ├── run_analysis.py       #   样本筛选与分析主脚本
│   └── generate_filtered_samples.py  # 过滤样本生成
├── ts-caption-eval/          # 评测框架（forked & adapted）
│   ├── configs/              #   评测配置（thesis.yaml / thesis_qa.yaml）
│   ├── predictions/          #   模型推理预测文件
│   ├── results/              #   评测结果输出
│   ├── tscapeval/            #   评测核心代码
│   └── data/                 #   评测数据集
└── python/                   # Python 虚拟环境
```

## 快速开始

### 环境准备

```bash
# 激活虚拟环境
python/myenv_bs/Scripts/Activate.ps1   # Windows PowerShell
# 或
source python/myenv_bs/bin/activate    # Linux

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
