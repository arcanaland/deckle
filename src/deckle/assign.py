"""Assign a staged master to a canonical ID"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

import cv2

from .ids import CardId, check_custom_name, resolve_ref
from .labels import ROTATE_MARK, Label, LabelError, format_label, parse_label, write_label
from .project import CARD_BACKS_DIR, PROJECT_FILE, Project
from .staging import MEASURED_KEYS, provenance_for, read_staging_index, write_staging_index

_SLOT = re.compile(r"r[0-9]+c[0-9]+")


class AssignError(RuntimeError):
    """A busted assignment."""


@dataclass(frozen=True)
class Assignment:
    ref: str
    source: Path
    dest: Path
    replaced: Path | None
    rotated: bool


def _move(src: Path, dest: Path, *, rotate_180: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not rotate_180:
        try:
            os.replace(src, dest)
        except OSError:  # different filesystem
            shutil.move(str(src), str(dest))
        return
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise AssignError(f"cannot read {src}")
    if not cv2.imwrite(
        str(dest), cv2.rotate(img, cv2.ROTATE_180), [cv2.IMWRITE_PNG_COMPRESSION, 6]
    ):
        raise AssignError(f"failed to write {dest}")
    src.unlink()


def assign(
    project: Project,
    source: Path,
    ref: str | None = None,
    *,
    card_back: str | None = None,
    rotate_180: bool = False,
) -> Assignment:
    """Assign a staged master to a card ID"""
    if project.degraded:
        raise AssignError(
            f"cannot assign: {project.degraded_reason}. Fix or remove the state file first."
        )
    if (ref is None) == (card_back is None):
        raise AssignError("give exactly one of a canonical ID or --card-back <design>")
    if not source.is_file():
        raise AssignError(f"no such file: {source}")

    card: CardId | None = None
    if card_back is not None:
        key = check_custom_name(card_back, "card back design key")
        ref_out = f"{CARD_BACKS_DIR}.{key}"
        dest = project.back_path(key)
    else:
        card, variant = resolve_ref(ref)  # type: ignore[arg-type]
        ref_out = ref  # type: ignore[assignment]
        dest = project.master_path(card, variant)

    if source.resolve() == dest.resolve():
        raise AssignError(f"{source} is already the master for {ref_out}")

    replaced: Path | None = None
    previous = project.cards.get(ref_out, {}).get("master")
    if previous:
        old = project.root / previous
        if old.exists() and old.resolve() != dest.resolve():
            old.unlink()
            replaced = old

    _move(source, dest, rotate_180=rotate_180)

    index = read_staging_index(project)
    prov = provenance_for(source.name, index, rotate_180=rotate_180)
    prov["master"] = dest.relative_to(project.root).as_posix()
    project.record_card(ref_out, prov)
    project.save()

    if index.pop(source.name, None) is not None:
        write_staging_index(project, index)

    return Assignment(ref=ref_out, source=source, dest=dest, replaced=replaced, rotated=rotate_180)


@dataclass(frozen=True)
class Unassignment:
    ref: str
    master: Path
    restaged: Path
    label: str
    labelled: bool  # False when the filesystem refused the label xattr


@dataclass(frozen=True)
class _Undo:
    label: Label
    master: Path
    restaged: Path
    row: dict[str, Any]


def _restaged_name(label: Label, record: dict[str, Any]) -> str:
    """The name rectify gave the file, so a later assign finds its index row again."""
    scan, slot = record.get("source_scan"), record.get("slot")
    if scan and slot and _SLOT.fullmatch(slot):
        return f"{PurePosixPath(scan).stem}_{slot}.png"
    return f"{label.target.replace(':', '.')}.png"


def unassign(project: Project, refs: list[str]) -> list[Unassignment]:
    """Put masters back in staging/ as they were before assign: name, index row, label."""
    if project.degraded:
        raise AssignError(
            f"cannot unassign: {project.degraded_reason}. Fix or remove the state file first."
        )

    back_default = project.doc.get("project", {}).get("card_back_default")
    undo: list[_Undo] = []
    problems: list[str] = []
    for text in refs:
        try:
            label = parse_label(text, card_back_default=back_default)
        except LabelError as exc:
            problems.append(str(exc))
            continue
        if label.rotate_180:
            problems.append(f"{text!r}: {ROTATE_MARK} has no meaning here; unassign restores it")
            continue
        record = project.cards.get(label.target)
        if record is None:
            problems.append(f"{label.target} has no record in {PROJECT_FILE}")
            continue
        master = project.root / record["master"] if record.get("master") else None
        if master is None or not master.is_file():
            problems.append(f"{label.target}: its master {record.get('master')!r} is missing")
            continue
        rotated = bool(record.get("rotate_180"))
        restaged = project.staging_dir / _restaged_name(label, record)
        undo.append(_Undo(replace(label, rotate_180=rotated), master, restaged, record))

    seen: dict[Path, str] = {}
    for u in undo:
        if u.restaged.exists():
            problems.append(f"{u.label.target}: {u.restaged.name} is already in staging/")
        elif u.restaged in seen:
            problems.append(
                f"{u.label.target} and {seen[u.restaged]} both restage as {u.restaged.name}"
            )
        seen[u.restaged] = u.label.target
    if problems:
        raise AssignError("nothing was unassigned:\n  " + "\n  ".join(problems))

    index = read_staging_index(project)
    out: list[Unassignment] = []
    for u in undo:
        _move(u.master, u.restaged, rotate_180=u.label.rotate_180)
        text = format_label(u.label)
        try:
            write_label(u.restaged, text)
            labelled = True
        except OSError:
            labelled = False
        if _SLOT.fullmatch(u.row.get("slot") or ""):
            index[u.restaged.name] = {
                k: u.row[k] for k in ("source_scan", "slot", *MEASURED_KEYS) if k in u.row
            }
            write_staging_index(project, index)
        project.forget_card(u.label.target)
        project.save()
        out.append(Unassignment(u.label.target, u.master, u.restaged, text, labelled))
    return out
