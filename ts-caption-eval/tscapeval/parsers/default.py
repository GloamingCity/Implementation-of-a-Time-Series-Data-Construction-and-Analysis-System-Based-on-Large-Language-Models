"""Default parser: treat the `pred_caption` field as-is, trimmed."""

from __future__ import annotations

from . import Parser, register


@register("default")
class DefaultParser(Parser):
    pass
