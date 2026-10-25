"""Status of scanned vs staged."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .labels import StagingPlan, plan_staging, staged_files
from .project import Master, Project, ProjectError


@dataclass(frozen=True)
class Status:
    root: Path
    degraded_reason: str | None
    edge_strategy: str | None
    config_problem: str | None
    present: list[Master]  # bare masters only: a variant alone does not make a card present
    total: int | None
    missing: list[str]
    unexpected: list[str]
    backs: list[str]
    strays: list[Path]
    variants: dict[str, list[str]] = field(default_factory=dict)  # card ID -> variant refs
    staging: StagingPlan | None = None

    @property
    def no_default(self) -> list[str]:
        """Cards with variant masters and no bare one."""
        bare = {str(m.card) for m in self.present}
        return [c for c in self.variants if c not in bare]


def status(project: Project) -> Status:
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
    bare = [m for m in fronts if m.variant is None]
    present_ids = {str(m.card) for m in bare}
    variants: dict[str, list[str]] = defaultdict(list)
    for m in fronts:
        if m.variant is not None:
            variants[str(m.card)].append(m.ref)

    if project.degraded:
        return Status(
            root=project.root,
            degraded_reason=project.degraded_reason,
            edge_strategy=None,
            config_problem=None,
            present=bare,
            total=None,
            missing=[],
            unexpected=[],
            backs=backs,
            strays=project.strays(),
            variants=dict(variants),
        )

    wanted = [str(c) for c in project.roster.cards()]
    return Status(
        root=project.root,
        degraded_reason=None,
        edge_strategy=strategy,
        config_problem=config_problem,
        present=bare,
        total=len(wanted),
        missing=[c for c in wanted if c not in present_ids],
        unexpected=sorted({str(m.card) for m in fronts} - set(wanted)),
        backs=backs,
        strays=project.strays(),
        variants=dict(variants),
        staging=plan_staging(project, force=False) if staged_files(project) else None,
    )
