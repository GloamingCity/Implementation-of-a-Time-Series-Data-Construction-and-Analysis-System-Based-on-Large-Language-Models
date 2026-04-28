"""Example custom parser for a ChatTS-style verbose output.

ChatTS tends to emit multi-section outputs such as:

    Caption:
    The series exhibits ...

    Analysis:
    It starts with a downward trend ...

We strip the leading "Caption:" tag if present and keep the first paragraph,
which is treated as the primary descriptive caption for evaluation.
"""

from __future__ import annotations

from ..types import Prediction
from . import Parser, register


@register("chatts")
class ChatTSParser(Parser):
    def parse_row(self, row: dict) -> Prediction:
        raw = str(row.get("pred_caption", "")).strip()
        lowered = raw.lower()
        if lowered.startswith("caption:"):
            raw = raw.split(":", 1)[1].strip()
        # Keep up to the first blank-line separator (first paragraph).
        first_block = raw.split("\n\n", 1)[0].strip()
        return Prediction(
            ts_id=row["ts_id"],
            dataset=row.get("dataset", ""),
            pred_caption=first_block,
        )
