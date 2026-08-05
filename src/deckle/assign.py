"""Move a rectified master out of `staging/` and into `masters/` under a canonical ID.

This is the plumbing half of `assign`. The review UX — contact sheet, single-keystroke
confirmation, proposed-ID ordering, rotate-180 preview — is deliberately a later task; what
is here is the operation that UX will eventually drive, and it is what makes a project
populatable at all.

[[ADR-003]] requires provenance to be recorded by whatever puts a card into `masters/`, so
the move and the record are one operation. They are still not atomic together: the file
lands first and `deckle.toml` is saved after. That order is the safe one — a master with no
provenance row is recoverable by re-assigning it, whereas a provenance row with no master
is a claim about a file that is not there.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2

from .ids import CardId, check_custom_name
from .project import (
    CARD_BACKS_DIR,
    Project,
    provenance_for,
    read_staging_index,
    resolve_ref,
    write_staging_index,
)


class AssignError(RuntimeError):
    """An assignment that cannot be carried out."""


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
    # A 180° rotation is a relabelling of pixels, not a resample: every source pixel
    # appears exactly once in the output, so re-encoding it as PNG loses nothing.
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
    """Assign one staged master to one canonical ID (or one card back design).

    Re-assigning is allowed and rewrites the provenance row. The old master is removed
    rather than left behind, because a stale file under `masters/` would still be counted
    by the filesystem index and would quietly make `status` wrong.
    """
    if project.degraded:
        raise AssignError(
            f"cannot assign: {project.degraded_reason}. Fix or remove the state file first "
            "— assigning now would write a fresh one over whatever is there."
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
