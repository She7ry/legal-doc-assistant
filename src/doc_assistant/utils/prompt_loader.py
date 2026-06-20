"""从 ``prompts/`` 目录加载 LLM 提示词模板。"""

from __future__ import annotations

from doc_assistant.config.settings import settings


def load_prompt(name: str) -> str:
    """读取指定文件名（如 ``document_qa.txt``）的 UTF-8 文本。"""
    prompt_path = settings.project_root / "src" / "doc_assistant" / "prompts" / name
    return prompt_path.read_text(encoding="utf-8")


def load_base_legal_prompt() -> str:
    """所有法律场景共用的基础系统提示（角色、合规边界、引用规范）。"""
    return load_prompt("base_legal_assistant.txt")
