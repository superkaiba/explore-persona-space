"""Pin the Step 10d lint-gate-tree archive pathspec to the manifests
``check_prod_import_lockfile`` reads (#2253).

The #1212 pre-push lint gate builds an ephemeral tree via
``git -C "$WT" archive origin/main -- <pathspec...>``
(.claude/skills/issue/steps/18-step-10d.md) and runs workflow_lint.py inside
it. ``--check-prod-import-lockfile`` (bundled into the no-flags run, #2253)
reads ``uv.lock`` + ``pyproject.toml`` at the tree ROOT and fails loud when
either is missing — contracted behaviour (#2253 acceptance A3: never a
silent skip). A pathspec omitting either manifest therefore makes EVERY
Step 10d lint gate fleet-wide fail on a synthetic-tree artifact (the #931
breakage shape) the moment the check reaches main — exactly the false block
#2253 round 4 fixed by adding ``uv.lock`` to the pathspec. These tests catch
that omission at authoring time instead of at the merge gate.

NOTE for future step-doc editors: the parse below anchors on the literal
``archive origin/main -- \\`` invocation and its ``| tar -x`` pipe. A
legitimate reshaping of that invocation must update the anchors here IN THE
SAME COMMIT, or the suite goes red.
"""

from __future__ import annotations

import re

from tests.issue_skill_source import issue_skill_text

#: Manifests check_prod_import_lockfile reads at the gate-tree root
#: (scripts/workflow_lint.py::check_prod_import_lockfile defaults:
#: root/"uv.lock" and root/"pyproject.toml").
LOCKFILE_CHECK_MANIFESTS = ("uv.lock", "pyproject.toml")

_ARCHIVE_ANCHOR = "archive origin/main -- \\"


def _archive_pathspec_tokens() -> list[str]:
    """Pathspec tokens of the gate-tree ``git archive origin/main --`` call."""
    text = issue_skill_text()
    start = text.index(_ARCHIVE_ANCHOR) + len(_ARCHIVE_ANCHOR)
    end = text.index("| tar -x", start)
    tokens = [t for t in re.split(r"[\s\\]+", text[start:end]) if t]
    assert tokens, "gate-tree archive pathspec parsed empty — anchors moved?"
    return tokens


def test_gate_tree_archive_ships_lockfile_check_manifests():
    tokens = _archive_pathspec_tokens()
    for manifest in LOCKFILE_CHECK_MANIFESTS:
        assert manifest in tokens, (
            f"Step 10d gate-tree archive pathspec omits {manifest!r}: "
            "check_prod_import_lockfile fails loud on a tree missing it, "
            "blocking EVERY Step 10d lint gate fleet-wide (#2253)."
        )


def test_gate_tree_archive_covers_lint_scan_roots():
    # Sanity that the parse found the REAL pathspec (the known scan roots),
    # so the manifest assertions above cannot pass on a mis-anchored slice.
    tokens = _archive_pathspec_tokens()
    for root in (".claude", "CLAUDE.md", "scripts", "src", "tests", "docs"):
        assert root in tokens, f"gate-tree archive pathspec missing scan root {root!r}"
