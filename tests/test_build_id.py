"""`build_id()`: which deckle build is running.

Every case below is a `direct_url.json` shape that `uv tool install` writes.
"""

from __future__ import annotations

import json

import pytest

import deckle
from deckle import __version__, build_id


@pytest.fixture(autouse=True)
def _uncached():
    """`build_id` is cached for the life of the process."""
    build_id.cache_clear()
    yield
    build_id.cache_clear()


class _Dist:
    def __init__(self, payload):
        self._payload = payload

    def read_text(self, name):
        assert name == "direct_url.json"
        return self._payload


def _install(monkeypatch, payload):
    monkeypatch.setattr(deckle.Distribution, "from_name", staticmethod(lambda name: _Dist(payload)))


def test_a_tag_install_names_the_tag_and_the_commit(monkeypatch):
    """`uv tool install git+file://...@release-1.2`"""
    _install(
        monkeypatch,
        json.dumps(
            {
                "url": "file:///src/deckle",
                "vcs_info": {
                    "vcs": "git",
                    "commit_id": "8e82be9111865a380db409a6edf5333c7c1f5e6c",
                    "requested_revision": "release-1.2",
                },
            }
        ),
    )
    assert build_id() == f"{__version__}+release.1.2.g8e82be9"


def test_the_commit_survives_the_branch_moving(monkeypatch):
    """A branch install records the commit it resolved, not the branch's tip today."""
    _install(
        monkeypatch,
        json.dumps({"vcs_info": {"vcs": "git", "commit_id": "0387fd91a485a67bd1238945"}}),
    )
    assert build_id() == f"{__version__}+g0387fd9"


def test_an_editable_install_says_so(monkeypatch):
    """No commit describes an editable install; a bare version would imply a release."""
    _install(monkeypatch, json.dumps({"url": "file:///x", "dir_info": {"editable": True}}))
    assert build_id() == f"{__version__}+editable"


@pytest.mark.parametrize("payload", [None, "", "{not json", json.dumps({})])
def test_unreadable_metadata_falls_back_rather_than_raising(monkeypatch, payload):
    _install(monkeypatch, payload)
    assert build_id() == __version__


def test_a_missing_distribution_falls_back(monkeypatch):
    def boom(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(deckle.Distribution, "from_name", staticmethod(boom))
    assert build_id() == __version__


def test_the_real_build_id_starts_with_the_package_version():
    """Whatever this checkout is installed as, it is the version plus a local segment."""
    got = build_id()
    assert got == __version__ or got.startswith(f"{__version__}+")
