"""Prose pin for the #1964 dispatch-preflight duties (staging / env-pin /
per-leg / relaunch-flag probes).

Pins (a) the Step 6b dispatch-input/env/flag preflight block in the /issue
SKILL.md (anchor phrase + the four duties' key command literals — the
lane-aware staged-input probes, the env-read enumeration grep, the per-LEG
carry-over gate, and the handle-sidecar flag-fidelity clause), (b) its
placement — AFTER the hand-composed argv dry-run block, inside Step 6b
(before Step 6c), (c) the crash-fix-rounds § Changed-argv relaunch mirror
clause (relaunch-flag fidelity + machine caps), and (d) this file's own
registration in the Step-9c selector's WORKFLOW_INVARIANT set (SKILL.md
diffs select only that set — no discovery arm reaches a .md pin file, so
an unregistered pin never runs on the diffs it guards).

Family precedent: tests/test_issue_skill_trigger_dense_tag_adoption.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKILL_MD = REPO / ".claude" / "skills" / "issue" / "SKILL.md"
CRASH_FIX_MD = REPO / ".claude" / "rules" / "crash-fix-rounds.md"
SELECTOR_PY = REPO / "scripts" / "select_step9c_tests.py"

PIN_FILE_RELPATH = "tests/test_issue_skill_dispatch_preflight_pin.py"

ANCHOR = "Dispatch-input/env/flag preflight"


def test_dispatch_preflight_block_present():
    text = SKILL_MD.read_text(encoding="utf-8")
    idx = text.index(ANCHOR)  # ValueError = hard fail
    # Window 5000: the landed block is ~4.4k chars anchor-to-tail; 5000
    # leaves headroom for wording tweaks without letting the pin drift
    # file-wide.
    window = text[idx : idx + 5000]
    # Duty (a): lane-aware staged-input existence probes.
    assert "cat-file -e" in window  # git-clone-lane pushed-ref probe
    assert "list_repo_tree" in window  # scoped HF probe (never full-tree)
    assert "--lane rsync" in window  # rsync-lane sync-set coverage
    # The argv dry-run's judged-pass row is PARSE-only — does not satisfy (a).
    assert "does NOT satisfy this probe" in window
    # Scope split: extension of the Step 6a.5 gate, not a re-run of it.
    assert "does NOT re-run the 6a.5" in window
    # Duty (b): env-read enumeration checked against the composed launch env.
    assert r"os\.environ|os\.getenv" in window
    # Duty (c): per-LEG carry-over gate.
    assert "verify_carryover_inputs.py" in window
    # Duty (d): handle-sidecar flag fidelity + target-machine cap re-derive.
    assert "handle.json" in window
    assert "--rss-cap-gb" in window


def test_block_placement_inside_step_6b():
    text = SKILL_MD.read_text(encoding="utf-8")
    idx = text.index(ANCHOR)
    # Sits after the argv dry-run block's anchor, inside Step 6b, before 6c.
    assert text.index("Hand-composed phase argv dry-run") < idx
    assert text.index("#### Step 6b") < idx
    assert idx < text.index("#### Step 6c")


def test_crash_fix_rounds_mirror_clause_present():
    text = CRASH_FIX_MD.read_text(encoding="utf-8")
    i0 = text.index("Relaunch-flag fidelity + machine caps (#1964)")
    window = text[i0 : i0 + 1200]
    assert "issue-<N>-handle.json" in window  # verbatim-from-sidecar source
    assert "--rss-cap-gb" in window  # target-machine cap re-derivation
    # Points back at the canonical SKILL.md block for probes (a)-(c).
    assert "dispatch-preflight" in window
    # Sits inside § Changed-argv relaunch, before the symbol-rename section.
    assert text.index("Changed-argv relaunch") < i0
    assert i0 < text.index("symbol-rename whole-tree grep duty")


def test_registered_in_step9c_workflow_invariant():
    # Import by path, matching tests/test_select_step9c_tests.py (the
    # selector lives under scripts/, not an importable package).
    spec = importlib.util.spec_from_file_location("select_step9c_tests_1964", SELECTOR_PY)
    assert spec is not None and spec.loader is not None
    sel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sel)
    assert PIN_FILE_RELPATH in sel.WORKFLOW_INVARIANT
