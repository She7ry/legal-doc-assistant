"""评测用常量：拒答措辞列表等，与 answer_guard 的 REFUSAL_TERMS 对齐。"""

from __future__ import annotations

DEFAULT_REFUSAL_TERMS: tuple[str, ...] = (
    "not found",
    "not provided",
    "cannot determine",
    "not enough information",
    "relevant text was not found",
    "did not find enough relevant text",
    "do not contain",
    "does not contain",
    "do not specify",
    "does not specify",
    "do not mention",
    "does not mention",
)
