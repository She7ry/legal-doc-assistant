"""跨模块共享的引用与问答结果数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from doc_assistant.memory.schemas import MemoryUsage


@dataclass(frozen=True)
class Citation:
    """一条可引用的文档证据。

    用途：RAG 检索到的片段会转成 Citation；LLM 在答案里用 ``[S1]``、``[S2]`` 标注，
    answer_guard 再校验这些 ID 是否真实存在、是否支撑对应句子。

    关键字段：
    - source_id: 引用编号，如 S1、D2（document）、W3（web）
    - preview / exact_quote: 展示给用户的摘录
    - page / chunk_id / section_heading: 定位到原文位置
    """

    source_id: str
    file_name: str
    preview: str
    page: int | None = None
    chunk_id: int | None = None
    source_type: str = "document"
    file_id: str | None = None
    document_key: str | None = None
    document_version: int | None = None
    page_label: str | None = None
    section_heading: str | None = None
    exact_quote: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    retrieval_score: float | None = None
    retrieval_relevance: float | None = None

    def location_label(self) -> str:
        parts = []
        if self.page is not None:
            parts.append(self.page_label or f"page {self.page + 1}")
        if self.chunk_id is not None:
            parts.append(f"chunk {self.chunk_id}")
        if self.section_heading:
            parts.append(str(self.section_heading))
        return f" ({', '.join(parts)})" if parts else ""


@dataclass(frozen=True)
class QAAnswer:
    """``DocumentQAService.ask()`` 的完整返回值。

    用途：API 层把 content 返回前端，citations 渲染为引用列表，
    guard_warnings 提示用户哪些表述证据不足，metadata.evidence 供审计。
    """

    content: str
    citations: list[Citation] = field(default_factory=list)
    memories_used: list[MemoryUsage] = field(default_factory=list)
    confidence: str | None = None
    guard_warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestResult:
    """``DocumentVectorStore.ingest_file()`` 的入库结果摘要。

    用途：告诉 API/前端本次上传解析了多少页/块、是否因内容相同而跳过、
    有无 OCR 警告等；file_id 是内容 SHA256，用于去重与版本追踪。
    """

    file_id: str
    file_name: str
    document_count: int
    chunk_count: int
    document_key: str = ""
    document_version: int = 1
    file_extension: str = ""
    page_count: int | None = None
    skipped: bool = False
    warnings: list[str] = field(default_factory=list)
