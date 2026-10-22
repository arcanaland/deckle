import json
import re
from functools import cache
from importlib.metadata import Distribution

__version__ = "0.1.0"

# PEP 440 local-version segments allow only alphanumerics and periods.
_SAFE = re.compile(r"[^A-Za-z0-9.]+")


@cache
def build_id() -> str:
    try:
        raw = Distribution.from_name("deckle").read_text("direct_url.json")
        info = json.loads(raw) if raw else {}
    except Exception:
        return __version__

    commit = str((info.get("vcs_info") or {}).get("commit_id") or "")[:7]

    if commit:
        rev = str((info.get("vcs_info") or {}).get("requested_revision") or "")
        rev = _SAFE.sub(".", rev).strip(".")
        return f"{__version__}+{rev}.g{commit}" if rev else f"{__version__}+g{commit}"

    if (info.get("dir_info") or {}).get("editable"):
        return f"{__version__}+editable"

    return __version__
