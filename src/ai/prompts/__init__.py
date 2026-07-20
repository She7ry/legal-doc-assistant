"""从 ``prompts/`` 目录加载 LLM 提示词模板。"""

from __future__ import annotations

from pathlib import Path

_PROMPT_DIR = Path(__file__).resolve().parent


def load_prompt(name: str) -> str:
    """读取指定文件名（如 ``document_qa.txt``）的 UTF-8 文本。"""
    return (_PROMPT_DIR / name).read_text(encoding="utf-8")


def load_base_legal_prompt() -> str:
    """所有法律场景共用的基础系统提示（角色、合规边界、引用规范）。"""
    return load_prompt("base_legal_assistant.txt")
