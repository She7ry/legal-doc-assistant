"""Multilingual tokenization used to construct Qdrant sparse vectors."""

from __future__ import annotations

import re
_SEARCH_TOKEN_PATTERN = re.compile(
    r"[A-Za-z]+(?:[-_][A-Za-z0-9]+)*|\d+(?:\.\d+)*%?|[一-鿿]"
)
def _tokenize_for_search(text: str) -> list[str]:
    return [token.casefold() for token in _SEARCH_TOKEN_PATTERN.findall(text or "")]
