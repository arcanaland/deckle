"""A small deterministic TOML writer."""

from __future__ import annotations

import re
from typing import Any

_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+\Z")

_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


class TomlError(TypeError):
    """A value that cannot be written as TOML."""


def format_key(key: str) -> str:
    if _BARE_KEY.fullmatch(key):
        return key
    return format_string(key)


def format_string(value: str) -> str:
    out = ['"']
    for ch in value:
        if ch in _ESCAPES:
            out.append(_ESCAPES[ch])
        elif ch < " " or ch == "\x7f":
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def format_value(value: Any) -> str:
    # bool before int
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return format_string(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise TomlError(f"cannot write non-finite float {value!r}")
        text = repr(value)
        return text if ("." in text or "e" in text) else text + ".0"
    if isinstance(value, dict):
        inner = ", ".join(f"{format_key(k)} = {format_value(v)}" for k, v in value.items())
        return "{ " + inner + " }" if inner else "{}"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(format_value(v) for v in value) + "]"
    raise TomlError(f"cannot write {type(value).__name__} as TOML: {value!r}")


def _is_table(value: Any) -> bool:
    return isinstance(value, dict)


def _emit(doc: dict[str, Any], prefix: list[str], lines: list[str]) -> None:
    scalars = {k: v for k, v in doc.items() if not _is_table(v)}
    tables = {k: v for k, v in doc.items() if _is_table(v)}

    if prefix and (scalars or not tables):
        if lines:
            lines.append("")
        lines.append("[" + ".".join(format_key(p) for p in prefix) + "]")
    for key, value in scalars.items():
        lines.append(f"{format_key(key)} = {format_value(value)}")
    for key, value in tables.items():
        _emit(value, [*prefix, key], lines)


def dumps(doc: dict[str, Any]) -> str:
    """Serialise preserving key order."""
    lines: list[str] = []
    _emit(doc, [], lines)
    return "\n".join(lines).strip("\n") + "\n"
