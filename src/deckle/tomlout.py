"""A small, deterministic TOML writer.

The standard library reads TOML (`tomllib`) and does not write it, and deckle writes two
kinds of TOML: its own `deckle.toml`, which must round-trip without reordering or losing
keys it does not understand, and the `deck.toml` and name files `emit` generates, which
must come out byte-identical on every run so that a deck directory is genuinely
disposable ([[ADR-003]]).

Both of those are properties of the *writer*, so it is worth the sixty lines to own one:
insertion order in, same order out, no timestamps, no hash-seed-dependent iteration, no
third-party version to drift underneath a byte-comparison test.

Scope is what deckle emits and no more — scalars, homogeneous arrays, inline tables inside
arrays (which is how `[deck].links` is written in the spec's own examples) and nested
tables. Datetimes are deliberately unsupported: §4.1 requires `created_date` and
`updated_date` to be *strings*, and a writer that cannot produce a TOML date cannot
accidentally produce one.
"""

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
    """A value deckle cannot write as TOML."""


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
    # bool before int: bool is a subclass of int and `True` must not become `1`.
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
    """Serialise `doc`, preserving its key order. Always ends in exactly one newline."""
    lines: list[str] = []
    _emit(doc, [], lines)
    return "\n".join(lines).strip("\n") + "\n"
