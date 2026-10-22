"""The staging index: what rectify measured, kept until assign consumes it."""

from __future__ import annotations

import os
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import build_id, tomlout
from .project import Project

if TYPE_CHECKING:
    from .detect import Card

STAGING_INDEX = "index.toml"

#: Measurements carried from the index into provenance when present.
MEASURED_KEYS = ("width_mm", "height_mm", "aspect", "skew_deg", "edge_strategy", "dpi")


def utc_now() -> str:
    """An ISO-8601 UTC timestamp to the second."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_staging_index(project: Project) -> dict[str, Any]:
    path = project.staging_dir / STAGING_INDEX
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except (tomllib.TOMLDecodeError, OSError):
        return {}


def write_staging_index(project: Project, index: dict[str, Any]) -> None:
    project.staging_dir.mkdir(parents=True, exist_ok=True)
    path = project.staging_dir / STAGING_INDEX
    tmp = path.with_name(f".{STAGING_INDEX}.{os.getpid()}.tmp")
    try:
        tmp.write_text(tomlout.dumps(dict(sorted(index.items()))), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def record_rectified(
    project: Project,
    scan: Path,
    cards: list[Card],
    written: list[Path],
    *,
    strategy: str,
    dpi: float,
) -> None:
    """Index the masters rectify wrote into this project's staging/.

    Masters written anywhere else, or into a degraded project, are not recorded.
    """
    if project.degraded:
        return
    staging = project.staging_dir.resolve()
    rows = {
        path.name: {
            "source_scan": scan.name,
            "slot": f"r{card.window.row}c{card.window.col}",
            "width_mm": round(card.width_mm, 4),
            "height_mm": round(card.height_mm, 4),
            "aspect": round(card.aspect, 5),
            "skew_deg": round(card.skew_deg, 4),
            "edge_strategy": str(strategy),
            "dpi": dpi,
        }
        for path, card in zip(written, cards, strict=True)
        if path.parent.resolve() == staging
    }
    if not rows:
        return
    index = read_staging_index(project)
    index.update(rows)
    write_staging_index(project, index)


def provenance_for(filename: str, index: dict[str, Any], *, rotate_180: bool) -> dict[str, Any]:
    """A provenance row for a staged file.

    Without an index row, the scan and slot are recovered from rectify's
    `<scan-stem>_r<row>c<col>.png` naming and the measurements are omitted.
    """
    row = dict(index.get(filename, {}))
    stem = filename[: -len(".png")] if filename.endswith(".png") else filename
    scan, _, slot = stem.rpartition("_")
    prov: dict[str, Any] = {}
    prov["source_scan"] = row.get("source_scan") or (f"{scan}.jpg" if scan else filename)
    if row.get("slot") or slot:
        prov["slot"] = row.get("slot") or slot
    for key in MEASURED_KEYS:
        if key in row:
            prov[key] = row[key]
    prov["rotate_180"] = rotate_180
    prov["deckle_version"] = build_id()
    prov["assigned_at"] = utc_now()
    return prov
