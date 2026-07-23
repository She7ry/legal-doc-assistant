"""RAG 评测用常量：拒答措辞列表等，与 answer_guard 的 REFUSAL_TERMS 对齐。"""

from __future__ import annotations

DEFAULT_REFUSAL_TERMS: tuple[str, ...] = (
    "未找到",
    "未检索到",
    "未提供",
    "无法确定",
    "无法判断",
    "无法完整回答",
    "没有足够信息",
    "信息不足",
    "文档未说明",
    "文档中未说明",
)
