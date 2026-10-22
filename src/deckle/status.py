"""What a project holds against what its roster asks for."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .project import Master, Project, ProjectError


@dataclass(frozen=True)
class Status:
    """`total` is None when the roster is unavailable."""

    root: Path
    degraded_reason: str | None
    edge_strategy: str | None
    config_problem: str | None
    present: list[Master]
    total: int | None
    missing: list[str]
    unexpected: list[str]
    backs: list[str]
    strays: list[Path]


def status(project: Project) -> Status:
    """Never raises on a bad deckle.toml; a broken config is reported."""
    masters = project.masters()

    strategy: str | None = None
    config_problem: str | None = None
    if not project.degraded:
        try:
            strategy = project.config.edge_strategy
        except ProjectError as exc:
            config_problem = str(exc)

    backs = [m.ref for m in masters if m.is_back]
    fronts = [m for m in masters if not m.is_back]
    # A variant is another artwork for a card, not another card.
    present_ids = {str(m.card) for m in fronts}

    if project.degraded:
        return Status(
            root=project.root,
            degraded_reason=project.degraded_reason,
            edge_strategy=None,
            config_problem=None,
            present=fronts,
            total=None,
            missing=[],
            unexpected=[],
            backs=backs,
            strays=project.strays(),
        )

    wanted = [str(c) for c in project.roster.cards()]
    return Status(
        root=project.root,
        degraded_reason=None,
        edge_strategy=strategy,
        config_problem=config_problem,
        present=fronts,
        total=len(wanted),
        missing=[c for c in wanted if c not in present_ids],
        unexpected=sorted(present_ids - set(wanted)),
        backs=backs,
        strays=project.strays(),
    )
