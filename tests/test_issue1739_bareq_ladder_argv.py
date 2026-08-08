"""#1739 bare-query L-ladder box — argv + upload-glob regressions (run-1 crash).

Run 1 of ``scripts/issue1739_bareq_ladder.sh`` lost all 9 budget rungs in ~1 s
each to the scorer's rail-3 out-root guard, and then lost its own rc-accounting
JSON to a ``**/``-only upload allow-list. Both are argv/glob defects in the BOX
script, not in the reviewed scorer — these tests pin the composed shapes so the
class cannot recur silently.

The out-root test drives the scorer's REAL guard (``_assert_outputs_safe``), not
a re-implementation of it, so it stays honest if the guard's predicate changes.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LADDER_SH = REPO_ROOT / "scripts" / "issue1739_bareq_ladder.sh"


def _ladder_text() -> str:
    return LADDER_SH.read_text(encoding="utf-8")


def _composed_out_roots() -> list[str]:
    """Render every ``--out-root`` the ladder script composes, shell vars resolved."""
    text = _ladder_text()
    base = re.search(r'^OUT_BASE="([^"]+)"', text, re.M)
    assert base, "OUT_BASE assignment not found in the ladder script"
    out_base = base.group(1)
    raw = re.findall(r'--out-root "([^"]+)"', text)
    assert raw, "no --out-root argument found in the ladder script"
    return [r.replace("$OUT_BASE", out_base).replace("$L", "250") for r in raw]


def test_composed_out_root_passes_the_scorers_real_rail3_guard(tmp_path):
    """Every composed --out-root must satisfy the scorer's OWN out-root guard.

    Pre-fix the script composed ``<base>/L<budget>`` and this raises SystemExit
    ("--out-root must be a 'bareq_map' subtree"), which is exactly how run 1
    died on all 9 rungs.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import issue1739_bareq_score as bq

    composed = _composed_out_roots()
    assert composed, "expected at least one composed --out-root"
    for rel in composed:
        # Resolve under a tmp root: the guard reads only the leaf NAME, and this
        # keeps the probe from depending on the repo's working tree.
        out_root = tmp_path / rel
        out_root.mkdir(parents=True, exist_ok=True)
        bq._assert_outputs_safe([], out_root=out_root, allow=False)


def test_composed_out_root_leaf_is_the_scorers_out_root_name():
    """Belt-and-braces on the same invariant, expressed against the constant."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import issue1739_bareq_score as bq

    for rel in _composed_out_roots():
        assert Path(rel).name == bq.OUT_ROOT_NAME, (
            f"composed --out-root {rel!r} has leaf {Path(rel).name!r}, "
            f"but the scorer requires {bq.OUT_ROOT_NAME!r}"
        )


def test_upload_allow_patterns_cover_root_level_files():
    """`**/*.ext` alone drops ROOT-level files — the round's own accounting JSON.

    huggingface_hub fnmatches allow_patterns against the folder-RELATIVE path,
    so ``**/*.json`` matches only files at least one directory deep. Run 1's
    upload carried the 9 ``logs/*.log`` files and silently dropped
    ``bareq_ladder_invocations.json`` at the root.
    """
    text = _ladder_text()
    block = re.search(r"allow_patterns=\[(.*?)\]", text, re.S)
    assert block, "no allow_patterns list found in the ladder script's upload leg"
    patterns = set(re.findall(r'"([^"]+)"', block.group(1)))
    for ext in ("json", "jsonl", "log"):
        assert f"*.{ext}" in patterns, (
            f"allow_patterns lacks the root-level form '*.{ext}' — a file at the "
            f"upload root would be silently dropped (only '**/*.{ext}' present)"
        )


@pytest.mark.parametrize("ext", ["json", "jsonl", "log"])
def test_root_level_glob_semantics_hold_for_huggingface_hub(ext):
    """Pin the upstream semantics this fix rests on, so a hub change surfaces here."""
    from huggingface_hub.utils import filter_repo_objects

    root_file = f"bareq_ladder_invocations.{ext}"
    nested = f"logs/evil_L250.{ext}"
    only_nested = list(filter_repo_objects([root_file, nested], allow_patterns=[f"**/*.{ext}"]))
    assert nested in only_nested
    assert root_file not in only_nested, (
        "upstream now matches root-level files with '**/' — the belt-and-braces "
        "root patterns stay harmless, but this test's premise has changed"
    )
    both = list(
        filter_repo_objects([root_file, nested], allow_patterns=[f"*.{ext}", f"**/*.{ext}"])
    )
    assert set(both) == {root_file, nested}
