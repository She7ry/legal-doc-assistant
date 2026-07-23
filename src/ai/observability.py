"""本地操作日志与 LangSmith Agent 追踪边界。"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from typing import Any

from langsmith import Client, tracing_context
from langsmith.anonymizer import create_secret_anonymizer

from ai.config.settings import settings

logger = logging.getLogger("doc_assistant")

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_enabled(name: str) -> bool:
    raw_value = os.getenv(name)
    if raw_value is not None:
        return raw_value.strip().lower() in _TRUE_VALUES
    return settings.langsmith_tracing


def hash_trace_identifier(value: str | None) -> str | None:
    """在 trace metadata 中保留可关联性，但不上传原始用户标识。"""
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


@cache
def get_langsmith_client() -> Client:
    pii_rules = [
        {
            "pattern": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
            "replace": "[EMAIL_REDACTED]",
        },
        {
            "pattern": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
            "replace": "[PHONE_REDACTED]",
        },
        {
            "pattern": re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
            "replace": "[ID_REDACTED]",
        },
    ]
    return Client(
        api_url=settings.langsmith_endpoint,
        api_key=settings.langsmith_api_key or None,
        tracing_sampling_rate=settings.langsmith_tracing_sampling_rate,
        anonymizer=create_secret_anonymizer(extra_rules=pii_rules),
    )


@contextmanager
def langsmith_agent_context(
    *,
    task_id: str,
    user_id: str | None,
    conversation_id: str | None,
) -> Iterator[None]:
    """为一次后台 Agent 任务设置 LangSmith 项目、标签和脱敏客户端。"""
    if not _env_enabled("LANGSMITH_TRACING"):
        yield
        return

    metadata = {
        "task_id": task_id,
        "user_id_hash": hash_trace_identifier(user_id),
        "conversation_id_hash": hash_trace_identifier(conversation_id),
        "observability_schema_version": "1",
    }
    with tracing_context(
        enabled=True,
        client=get_langsmith_client(),
        project_name=os.getenv("LANGSMITH_PROJECT", settings.langsmith_project),
        tags=["legal-agent", "production"],
        metadata={key: value for key, value in metadata.items() if value is not None},
    ):
        yield


def agent_trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """只上传评估所需输入；移除服务对象、回调和原始身份标识。"""
    return {
        "objective": inputs.get("objective"),
        "focus_areas": inputs.get("focus_areas") or [],
        "user_role": inputs.get("user_role"),
        "max_steps": inputs.get("max_steps"),
        "task_id": inputs.get("task_id"),
        "user_id_hash": hash_trace_identifier(inputs.get("user_id")),
        "conversation_id_hash": hash_trace_identifier(inputs.get("conversation_id")),
    }


def agent_trace_outputs(output: Any) -> dict[str, Any]:
    """将 Agent 结果压缩成可供 Judge 使用、且不含原始工具返回值的结构。"""
    if output is None:
        return {"status": "interrupted"}
    if hasattr(output, "questions"):
        return {
            "task_id": output.task_id,
            "status": "needs_input",
            "questions": list(output.questions),
        }
    citations = [
        {
            "source_id": citation.source_id,
            "file_name": citation.file_name,
            "page": citation.page,
            "section_heading": citation.section_heading,
            "exact_quote": citation.exact_quote,
            "preview": citation.preview,
        }
        for citation in output.citations
    ]
    steps = [
        {
            "step_id": step.step_id,
            "title": step.title,
            "tool": step.tool,
            "status": step.status,
            "summary": step.summary,
            "guard_warnings": step.guard_warnings,
        }
        for step in output.steps
    ]
    metadata = output.metadata if isinstance(output.metadata, dict) else {}
    tool_calls = metadata.get("tool_calls")
    return {
        "task_id": output.task_id,
        "status": output.status,
        "objective": output.objective,
        "report": output.report,
        "confidence": output.confidence,
        "human_review_required": output.human_review_required,
        "guard_warnings": output.guard_warnings,
        "steps": steps,
        "citations": citations,
        "step_count": len(steps),
        "citation_count": len(citations),
        "tool_call_count": len(tool_calls) if isinstance(tool_calls, list) else 0,
        "runtime": metadata.get("runtime"),
        "planning_mode": metadata.get("planning_mode"),
    }


@contextmanager
def traced_operation(operation: str, **context: Any) -> Iterator[None]:
    """上下文管理器：记录 operation 名称、耗时(ms) 与可选 context 字段到日志。"""
    start = time.perf_counter()
    try:
        yield
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.error(
            "Operation failed",
            extra={
                "operation": operation,
                "duration_ms": duration_ms,
                "error": str(exc),
                **context,
            },
            exc_info=True,
        )
        raise
    else:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "Operation completed",
            extra={"operation": operation, "duration_ms": duration_ms, **context},
        )
