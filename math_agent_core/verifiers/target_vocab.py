from __future__ import annotations

import re


CANONICAL_TARGET_ALIASES = {
    "determinant": ("determinant", "det", "行列式"),
    "rank": ("rank", "秩"),
}


def target_present(text: str, target: str) -> bool:
    value = str(text or "").lower()
    aliases = CANONICAL_TARGET_ALIASES.get(target, (target,))
    return any(
        re.search(rf"\b{re.escape(alias)}\b", value, flags=re.IGNORECASE)
        if alias.isascii() else alias in value
        for alias in aliases
    )


def target_aliases(target: str) -> tuple[str, ...]:
    return CANONICAL_TARGET_ALIASES.get(target, (target,))
