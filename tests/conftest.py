from __future__ import annotations

import os
from pathlib import Path

import pytest


def scan_library() -> Path | None:
    """The library passed in DECKLE_SCANS, or None when it is unset."""
    raw = os.environ.get("DECKLE_SCANS", "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_dir() else None


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if scan_library() is not None:
        return
    needing = [item.nodeid for item in items if item.get_closest_marker("scans") is not None]
    if not needing:
        return
    raise pytest.UsageError(f"{len(needing)} scan test(s) need a library (set DECKLE_SCANS)")


@pytest.fixture(scope="session")
def scans() -> Path:
    """The scan library"""
    library = scan_library()
    assert library is not None
    return library
