"""Card identity read from a staged file's comment xattr."""

from __future__ import annotations

import errno
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .ids import (
    MAJOR,
    MINOR,
    CardId,
    IdError,
    check_custom_name,
    parse_card_id,
)
from .project import CARD_BACKS_DIR, Project, ProjectError

COMMENT_XATTR = "user.xdg.comment"

BACK = "back"
ROTATE_MARK = "~"

_SUITS = {"w": "wands", "c": "cups", "s": "swords", "p": "pentacles"}
_RANKS = {
    "1": "ace",
    "a": "ace",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
    "t": "ten",
    "p": "page",
    "n": "knight",
    "q": "queen",
    "k": "king",
}
_MAJOR_SHORT = re.compile(r"m([0-9]{1,2})\Z")
_MINOR_SHORT = re.compile(r"([wcsp])(10|[1-9]|[atpnqk])\Z")
_RECTIFIED_STEM = re.compile(r"_r[0-9]+c[0-9]+\Z")

_EXPECTED = (
    "expected m<number>, <suit><rank> (suit w c s p; rank 1-10, a, t, p, n, q, k), "
    f"a canonical ID, or {BACK}; then an optional :<key> and a trailing {ROTATE_MARK}"
)

# Linux reports a missing attribute as ENODATA, the BSDs as ENOATTR.
_NO_LABEL = {getattr(errno, "ENOATTR", errno.ENODATA), errno.ENODATA, errno.ENOTSUP}


class LabelError(ValueError):
    """A comment that is not a label."""


@dataclass(frozen=True)
class Label:
    """Either a card reference or a card back design, never both."""

    card: CardId | None
    variant: str | None
    card_back: str | None
    rotate_180: bool

    @property
    def ref(self) -> str | None:
        """The canonical `<id>[:<variant>]`, or None for a card back."""
        if self.card is None:
            return None
        return str(self.card) if self.variant is None else f"{self.card}:{self.variant}"

    @property
    def target(self) -> str:
        return self.ref or f"{CARD_BACKS_DIR}.{self.card_back}"


def _card(text: str) -> CardId:
    if text.startswith((f"{MAJOR}.", f"{MINOR}.")):
        try:
            return parse_card_id(text)
        except IdError as exc:
            raise LabelError(str(exc)) from None
    if m := _MAJOR_SHORT.fullmatch(text):
        return CardId("major", key=f"{int(m[1]):02d}")
    if m := _MINOR_SHORT.fullmatch(text):
        return CardId("minor", suit=_SUITS[m[1]], rank=_RANKS[m[2]])
    raise LabelError(f"{text!r} is not a card: {_EXPECTED}")


def _key(text: str, what: str) -> str:
    try:
        return check_custom_name(text, what)
    except IdError as exc:
        raise LabelError(str(exc)) from None


def parse_label(text: str, *, card_back_default: str | None) -> Label:
    """Parse `card[:variant][~]` or `back[:design][~]`."""
    body = text.strip()
    if not body:
        raise LabelError("empty label")

    rotate = body.endswith(ROTATE_MARK)
    if rotate:
        body = body[: -len(ROTATE_MARK)]
    if ROTATE_MARK in body:
        raise LabelError(f"{text.strip()!r}: {ROTATE_MARK} goes at the end, and only there")

    head, sep, key = body.partition(":")
    if sep and not key:
        raise LabelError(f"{text.strip()!r}: empty key after ':'")

    if head == BACK:
        if sep:
            design = _key(key, "card back design key")
        elif card_back_default:
            design = _key(card_back_default, "card_back_default")
        else:
            raise LabelError(
                f"{text.strip()!r}: no card_back_default is set, so give a design: {BACK}:<key>"
            )
        return Label(card=None, variant=None, card_back=design, rotate_180=rotate)

    card = _card(head)
    variant = _key(key, "variant key") if sep else None
    return Label(card=card, variant=variant, card_back=None, rotate_180=rotate)


def read_label(path: Path) -> str | None:
    """The comment xattr, stripped, or None when there is none."""
    try:
        raw = os.getxattr(path, COMMENT_XATTR)
    except OSError as exc:
        if exc.errno in _NO_LABEL:
            return None
        raise
    text = raw.decode("utf-8", errors="replace").strip()
    return text or None


@dataclass(frozen=True)
class PlannedFile:
    source: Path
    label: Label
    dest: Path
    replaces: Path | None

    def describe(self, root: Path) -> str:
        line = f"{self.source.name} -> {self.label.target}"
        if self.label.rotate_180:
            line += " (rotated 180)"
        if self.replaces is not None:
            line += f" (replaces {self.replaces.relative_to(root)})"
        return line


@dataclass
class StagingPlan:
    files: int = 0
    unlabelled: int = 0
    rows: list[PlannedFile] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def labelled(self) -> int:
        return self.files - self.unlabelled

    def summary(self) -> str:
        return (
            f"{self.files} files, {self.labelled} labelled, {self.unlabelled} unlabelled, "
            f"{len(self.problems)} problem(s)"
        )


def staged_files(project: Project) -> list[Path]:
    staging = project.staging_dir
    if not staging.is_dir():
        return []
    return sorted(p for p in staging.glob("*.png") if p.is_file())


def plan_staging(project: Project, *, force: bool) -> StagingPlan:
    """Check every labelled file in staging/ without moving anything."""
    if project.degraded:
        raise ProjectError(
            f"cannot assign: {project.degraded_reason}. Fix or remove the state file first."
        )

    # Read the raw key: a bad edge_strategy elsewhere in [project] must not hide staging.
    back_default = project.doc.get("project", {}).get("card_back_default")
    roster = {str(c) for c in project.roster.cards()}
    plan = StagingPlan()

    for path in staged_files(project):
        plan.files += 1
        try:
            text = read_label(path)
        except OSError as exc:
            plan.problems.append(f"{path.name}: cannot read its label: {exc.strerror or exc}")
            continue
        if text is None:
            plan.unlabelled += 1
            continue
        try:
            label = parse_label(text, card_back_default=back_default)
        except LabelError as exc:
            plan.problems.append(f"{path.name}: {exc}")
            continue

        if label.card is not None:
            if str(label.card) not in roster:
                plan.problems.append(f"{path.name}: {label.card} is not in the roster")
            dest = project.master_path(label.card, label.variant)
        else:
            dest = project.back_path(label.card_back)  # type: ignore[arg-type]

        # assign() only checks the record, and its move overwrites an unrecorded file.
        replaces: Path | None = None
        if dest.exists():
            replaces = dest
        elif label.target in project.cards:
            previous = project.cards[label.target].get("master")
            replaces = project.root / previous if previous else dest
        if replaces is not None and not force:
            plan.problems.append(
                f"{path.name}: {label.target} already has a master "
                f"({replaces.relative_to(project.root)}); pass --force to replace it"
            )

        if not _RECTIFIED_STEM.search(path.stem):
            plan.warnings.append(
                f"{path.name}: not a rectified slot name, so its provenance will be thin"
            )
        plan.rows.append(PlannedFile(path, label, dest, replaces))

    _check_collisions(plan)
    _check_defaults(project, plan)
    return plan


def _check_collisions(plan: StagingPlan) -> None:
    claims: dict[str, list[PlannedFile]] = defaultdict(list)
    for row in plan.rows:
        claims[row.label.target].append(row)
    for target, rows in claims.items():
        if len(rows) < 2:
            continue
        names = ", ".join(r.source.name for r in rows)
        msg = f"{target} is claimed by {len(rows)} files: {names}"
        if rows[0].label.card is not None and rows[0].label.variant is None:
            msg += "; give one of these a variant key"
        plan.problems.append(msg)


def _check_defaults(project: Project, plan: StagingPlan) -> None:
    """A card the plan gives a variant must end up with a bare master too."""
    bare = {str(m.card) for m in project.masters() if m.card is not None and m.variant is None}
    bare |= {str(r.label.card) for r in plan.rows if r.label.card and r.label.variant is None}
    variants: dict[str, list[str]] = defaultdict(list)
    for r in plan.rows:
        if r.label.card is not None and r.label.variant is not None:
            variants[str(r.label.card)].append(r.source.name)
    for card, names in variants.items():
        if card not in bare:
            plan.problems.append(
                f"{card} would have a variant ({', '.join(names)}) but no default master; "
                f"label its default artwork with the bare card"
            )
