"""Single source of truth for 'is the Happy daemon injection patch applied?'.

Imported by spawn_session.py (pre-spawn guard) and autonomous_session_watch.py
(proactive escalate-only pass). patch_happy_daemon.py also imports SENTINEL /
DAEMON_FILE from here so there is exactly ONE definition of the sentinel string
and the daemon file path.

Cheap by construction: a single file read + substring test (single-digit ms),
no subprocess, no root. Safe when the daemon file is absent entirely (returns
the ``missing`` state — the spawn guard then disambiguates 'Happy never
installed' from 'patch file hash-renamed away by npm update' via a second,
independent check on the daemon RPC's daemon.state.json — see spawn_session.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DAEMON_DIR = Path("/usr/lib/node_modules/happy/dist")

# The daemon bundle is emitted under a CONTENT-HASHED filename
# (``index-<hash>.mjs``) that changes on every ``npm update happy``. Pinning the
# hash meant an update silently pointed every consumer at a nonexistent path
# (#2054, 2026-08-04: the pin sat at ``index-q9G4ktSK.mjs`` across an upgrade).
# Resolve by CONTENT instead: the spawn control-server log line below is stable
# across versions and unique to the daemon bundle.
DAEMON_MARKER = "[CONTROL SERVER] Spawn session request"
_LEGACY_DAEMON_FILE = DAEMON_DIR / "index-q9G4ktSK.mjs"


def resolve_daemon_file(dist_dir: Path | None = None) -> Path:
    """Locate the Happy daemon bundle by content marker, not by hashed filename.

    Scans ``dist_dir`` for ``index-*.mjs`` and returns the first one containing
    :data:`DAEMON_MARKER`. Falls back to the legacy hashed path when the
    directory is absent or no candidate matches, so the ``missing`` state still
    surfaces (with a real path in the message) rather than raising.
    """
    d = dist_dir or DAEMON_DIR
    if d.is_dir():
        for cand in sorted(d.glob("index-*.mjs")):
            try:
                if DAEMON_MARKER in cand.read_text(encoding="utf-8", errors="replace"):
                    return cand
            except OSError:
                continue
    return _LEGACY_DAEMON_FILE


DAEMON_FILE = resolve_daemon_file()
SENTINEL = "// EPS-PATCH: initial-prompt-seed + claudeArgs-spread + no-takeover-downgrade v5"

REAPPLY_CMD = "sudo uv run python scripts/patch_happy_daemon.py apply"
RESTART_CMD = "happy daemon stop && happy daemon start"


@dataclass(frozen=True)
class PatchStatus:
    """Result of :func:`classify_patch`.

    ``state`` ∈ {``"patched"``, ``"reverted"``, ``"drifted"``, ``"missing"``};
    ``detail`` is a human-readable one-liner for the log / fail-loud message.
    """

    state: str
    detail: str


def _file_text(f: Path) -> str:
    # Pin utf-8 for locale determinism (the .mjs is ascii/utf-8 source).
    return f.read_text(encoding="utf-8")


def classify_patch(daemon_file: Path | None = None) -> PatchStatus:
    """Classify whether the Happy daemon injection patch is applied.

    Returns a :class:`PatchStatus` whose ``state`` is one of:

    - ``"patched"`` — the sentinel comment is present (the daemon honors
      ``claudeArgs`` / ``HAPPY_INITIAL_PROMPT``).
    - ``"reverted"`` — the file exists, the sentinel is gone, but every
      ``PATCHES`` search-string still matches, so a plain ``apply`` would work.
    - ``"drifted"`` — the file exists, the sentinel is gone, AND at least one
      ``PATCHES`` search-string no longer matches (the daemon shape changed —
      a blind ``apply`` cannot fix it; manual reconciliation is required).
    - ``"missing"`` — the daemon file is absent at ``daemon_file``. AMBIGUOUS by
      itself (Happy not installed vs. the hashed bundle renamed away by
      ``npm update happy``); the caller disambiguates (see spawn_session.py's
      two-step probe).

    ``daemon_file`` defaults to :data:`DAEMON_FILE`; tests pass a synthetic path.
    """
    f = daemon_file or DAEMON_FILE
    if not f.is_file():
        return PatchStatus("missing", f"Happy daemon file not found at {f}")
    text = _file_text(f)
    if SENTINEL in text:
        return PatchStatus("patched", "sentinel present")
    # Reverted. Distinguish a clean revert (search-strings still match -> a
    # plain `apply` will work) from a drifted shape (a search-string no longer
    # matches -> manual reconciliation). Reuse patch_happy_daemon's PATCHES so
    # there is one shape definition. LAZY import to avoid any cycle.
    from importlib import import_module

    patch_mod = import_module("patch_happy_daemon")  # scripts/ on sys.path
    missing = [name for name, search, _ in patch_mod.PATCHES if search not in text]
    if missing:
        return PatchStatus(
            "drifted",
            "daemon upgraded; PATCHES no longer match: " + ", ".join(missing),
        )
    return PatchStatus("reverted", "sentinel absent; search-strings still match")


if __name__ == "__main__":
    import sys

    st = classify_patch()
    print(st.state, "—", st.detail)
    sys.exit({"patched": 0, "reverted": 1, "drifted": 2, "missing": 3}[st.state])
