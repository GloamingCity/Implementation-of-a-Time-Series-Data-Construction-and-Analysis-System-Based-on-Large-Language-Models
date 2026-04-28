import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from prompting import build_adaptive_prompt_messages


class JsonlTimeSeriesDataset(Dataset):
    """
    读取离线 JSONL 训练集。

    每行样本格式示例：
    {
      "time_series": [0.1, 0.3, ...],
      "caption": "这段序列整体先升后降..."
    }
    """

    def __init__(
        self,
        jsonl_path,
        tokenizer,
        max_text_length=512,
        ts_seq_len=512,
        random_crop=True,
    ):
        self.jsonl_path = Path(jsonl_path)
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"找不到训练文件: {self.jsonl_path}")

        self.tokenizer = tokenizer
        self.max_text_length = int(max_text_length)
        self.ts_seq_len = int(ts_seq_len)
        self.random_crop = bool(random_crop)

        self.samples = self._load_jsonl(self.jsonl_path)
        if len(self.samples) == 0:
            raise ValueError(f"训练文件为空: {self.jsonl_path}")

    @staticmethod
    def _load_jsonl(path):
        samples = []
        with path.open("r", encoding="utf-8-sig") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL 第 {line_idx} 行格式错误: {exc}") from exc

                if "time_series" not in obj or "caption" not in obj:
                    raise ValueError(
                        f"JSONL 第 {line_idx} 行缺少 time_series 或 caption 字段"
                    )
                samples.append(obj)
        return samples

    def __len__(self):
        return len(self.samples)

    def _normalize_ts(self, ts_values):
        """
        将变长时序统一到固定长度：
        - 长于 ts_seq_len：随机裁剪（训练增强）或前截断
        - 短于 ts_seq_len：右侧补零
        """
        ts = torch.tensor(ts_values, dtype=torch.float32).flatten()
        cur_len = ts.numel()

        if cur_len == self.ts_seq_len:
            return ts

        if cur_len > self.ts_seq_len:
            if self.random_crop:
                start = random.randint(0, cur_len - self.ts_seq_len)
            else:
                start = 0
            return ts[start : start + self.ts_seq_len]

        pad_len = self.ts_seq_len - cur_len
        return torch.nn.functional.pad(ts, (0, pad_len), value=0.0)

    def _build_chat_text(self, caption, ts_tensor):
        """
        用 Qwen-Instruct chat 模板组织 prompt + answer：
        - prompt 用于条件输入
        - answer 为监督目标
        """
        prompt_messages, _ = build_adaptive_prompt_messages(ts_tensor)
        full_messages = prompt_messages + [
            {"role": "assistant", "content": str(caption)}
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return prompt_text, full_text

    def __getitem__(self, idx):
        obj = self.samples[idx]
        raw_ts = torch.tensor(obj["time_series"], dtype=torch.float32).flatten()
        cur_len = raw_ts.numel()
        if cur_len >= self.ts_seq_len:
            if cur_len == self.ts_seq_len:
                prompt_ts = raw_ts
            else:
                if self.random_crop:
                    start = random.randint(0, cur_len - self.ts_seq_len)
                else:
                    start = 0
                prompt_ts = raw_ts[start : start + self.ts_seq_len]
            ts_tensor = prompt_ts
        else:
            prompt_ts = raw_ts
            pad_len = self.ts_seq_len - cur_len
            ts_tensor = torch.nn.functional.pad(raw_ts, (0, pad_len), value=0.0)

        caption = obj["caption"]

        prompt_text, full_text = self._build_chat_text(caption, prompt_ts)

        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_text_length,
            add_special_tokens=False,
            return_attention_mask=True,
        )
        prompt_enc = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_text_length,
            add_special_tokens=False,
            return_attention_mask=False,
        )

        input_ids = torch.tensor(full_enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(full_enc["attention_mask"], dtype=torch.long)
        labels = input_ids.clone()

        prompt_len = min(len(prompt_enc["input_ids"]), labels.numel())
        labels[:prompt_len] = -100

        if torch.all(labels == -100):
            labels[-1] = input_ids[-1]

        return {
            "ts_data": ts_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def build_collate_fn(pad_token_id):
    """
    动态 padding：
    - 文本按 batch 内最长序列补齐
    - labels 以 -100 补齐，确保 padding 不参与损失
    """

    def collate_fn(batch):
        ts_data = torch.stack([x["ts_data"] for x in batch], dim=0)

        input_ids_list = [x["input_ids"] for x in batch]
        attention_mask_list = [x["attention_mask"] for x in batch]
        labels_list = [x["labels"] for x in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask_list,
            batch_first=True,
            padding_value=0,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=-100,
        )

        return {
            "ts_data": ts_data,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate_fn
