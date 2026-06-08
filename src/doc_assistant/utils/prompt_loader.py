from __future__ import annotations

from doc_assistant.config.settings import settings


def load_prompt(name: str) -> str:
    prompt_path = settings.project_root / "src" / "doc_assistant" / "prompts" / name
    return prompt_path.read_text(encoding="utf-8")


def load_base_legal_prompt() -> str:
    return load_prompt("base_legal_assistant.txt")
