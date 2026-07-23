"""用于构造 Qdrant 稀疏向量的中文分词。"""

from __future__ import annotations

import re

_SEARCH_TOKEN_PATTERN = re.compile(r"\d+(?:\.\d+)*%?|[一-鿿]+")


def _tokenize_for_search(text: str) -> list[str]:
    tokens = []
    for token in _SEARCH_TOKEN_PATTERN.findall(text or ""):
        if re.fullmatch(r"[一-鿿]+", token) and len(token) > 1:
            tokens.extend(token[index : index + 2] for index in range(len(token) - 1))
        else:
            tokens.append(token)
    return tokens
