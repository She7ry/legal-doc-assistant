"""Metadata-only runtime skill selection."""

from __future__ import annotations

import re

from doc_assistant.skills.models import SkillMetadata, SkillSelection, SkillValidationError

_WORD_PATTERN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+", re.IGNORECASE)
_STOP_WORDS = {
    "a", "an", "and", "for", "from", "in", "of", "on", "or", "the", "to", "use", "when",
    "with", "this", "that", "answer", "task", "skill",
}


def _tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in _WORD_PATTERN.findall(text.casefold()):
        if match in _STOP_WORDS:
            continue
        tokens.add(match)
        if re.fullmatch(r"[\u4e00-\u9fff]+", match) and len(match) > 2:
            tokens.update(match[index : index + 2] for index in range(len(match) - 1))
    return tokens


class SkillSelector:
    """Rank skills using only their names and descriptions."""

    def __init__(self, skills: tuple[SkillMetadata, ...]) -> None:
        self.skills = skills
        self._by_name = {skill.name: skill for skill in skills}

    def select(
        self,
        task: str,
        *,
        max_skills: int = 4,
        required_names: tuple[str, ...] = (),
    ) -> SkillSelection:
        unknown = set(required_names) - set(self._by_name)
        if unknown:
            raise SkillValidationError("Unknown required skills: " + ", ".join(sorted(unknown)))
        task_tokens = _tokens(task)
        ranked: list[tuple[float, SkillMetadata]] = []
        scores: dict[str, float] = {}
        for skill in self.skills:
            name_tokens = _tokens(skill.name.replace("-", " "))
            description_tokens = _tokens(skill.description)
            name_overlap = len(task_tokens & name_tokens)
            description_overlap = len(task_tokens & description_tokens)
            score = float(name_overlap * 3 + description_overlap)
            scores[skill.name] = score
            if score > 0 or skill.name in required_names:
                ranked.append((score, skill))

        ranked.sort(key=lambda item: (-item[0], item[1].name))
        chosen = [skill for _, skill in ranked[: max(0, max_skills)]]
        for name in required_names:
            skill = self._by_name[name]
            if skill not in chosen:
                if len(chosen) >= max_skills and chosen:
                    chosen.pop()
                chosen.append(skill)
        chosen.sort(key=lambda skill: skill.name)
        selected_scores = {skill.name: scores[skill.name] for skill in chosen}
        if chosen:
            details = ", ".join(f"{name}={score:g}" for name, score in selected_scores.items())
            reason = f"Metadata-only lexical selection from name and description ({details})."
        else:
            reason = "No skill name or description matched the task metadata."
        return SkillSelection(skills=tuple(chosen), reason=reason, scores=selected_scores)

