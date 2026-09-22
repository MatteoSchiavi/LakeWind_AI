"""SQL identifier safety helper (Data & Prediction audit, Minor hygiene).

A handful of query builders interpolate table/column names into SQL strings.
Today every value comes from a fixed internal set (config enums, literals), so
the pattern is not exploitable — but three separate call sites had grown, and
a future refactor that passes user-adjacent input would silently create an
injection path. `safe_identifier()` is the one shared gate: it admits ONLY
plain lowercase snake_case identifiers and, when an allowlist is provided,
rejects anything outside it. Use it wherever a name is interpolated.
"""
from __future__ import annotations

import re

_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]*$")

__all__ = ["safe_identifier"]


def safe_identifier(name: str, *, allowlist: set[str] | None = None) -> str:
    """Return `name` if it is a safe SQL identifier, else raise ValueError.

    Rules:
      - non-empty, ASCII lowercase letter/underscore first, then
        letters/digits/underscores (no quotes, dots, spaces, dashes, etc.)
      - when `allowlist` is given, the name must also be a member.
    """
    if not isinstance(name, str) or not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    if allowlist is not None and name not in allowlist:
        raise ValueError(
            f"SQL identifier {name!r} not in allowlist: {sorted(allowlist)}"
        )
    return name
