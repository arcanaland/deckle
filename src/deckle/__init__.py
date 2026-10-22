"""Package version, and the build identity that goes into provenance.

`__version__` is the number in `pyproject.toml` and does not move between builds, so two
builds that behave differently stamp the same string. `build_id()` adds which build this
is: a VCS install (`uv tool install git+...@rev`) records the resolved commit in the
distribution's `direct_url.json`, so the commit is knowable at runtime with nothing to bump
by hand.
"""

from __future__ import annotations

import json
import re
from functools import cache
from importlib.metadata import Distribution

__version__ = "0.1.0"

# PEP 440 local-version segments allow only alphanumerics and periods.
_SAFE = re.compile(r"[^A-Za-z0-9.]+")


@cache
def build_id() -> str:
    """`__version__`, plus which build this is when that is knowable.

    * `0.1.0+v1.2.g8e82be9` — installed from a tag or branch, with the commit the
      installer resolved, so it stays right after the branch moves on.
    * `0.1.0+editable` — an editable install, which no commit describes.
    * `0.1.0` — a plain install, or metadata that is absent or unreadable.

    Never raises: it is called while writing provenance, and must not fail a run.
    """
    try:
        raw = Distribution.from_name("deckle").read_text("direct_url.json")
        info = json.loads(raw) if raw else {}
    except Exception:  # see the docstring: this must never raise
        return __version__

    commit = str((info.get("vcs_info") or {}).get("commit_id") or "")[:7]
    if commit:
        rev = str((info.get("vcs_info") or {}).get("requested_revision") or "")
        rev = _SAFE.sub(".", rev).strip(".")
        return f"{__version__}+{rev}.g{commit}" if rev else f"{__version__}+g{commit}"
    if (info.get("dir_info") or {}).get("editable"):
        return f"{__version__}+editable"
    return __version__
