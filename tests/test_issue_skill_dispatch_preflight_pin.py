"""Prose pin for the #1964 dispatch-preflight duties (staging / env-pin /
per-leg / relaunch-flag / out-scratch-isolation probes).

Pins (a) the Step 6b dispatch-input/env/flag preflight block in the /issue
SKILL.md (anchor phrase + the five duties' key command literals — the
lane-aware staged-input probes, the env-read enumeration grep, the per-LEG
carry-over gate, the handle-sidecar flag-fidelity clause, and the per-leg
out/scratch isolation item), (b) its
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

from tests.issue_skill_source import issue_skill_text

REPO = Path(__file__).resolve().parent.parent
SKILL_MD = REPO / ".claude" / "skills" / "issue" / "SKILL.md"
CRASH_FIX_MD = REPO / ".claude" / "rules" / "crash-fix-rounds.md"
EXPERIMENTER_MD = REPO / ".claude" / "agents" / "experimenter.md"
SELECTOR_PY = REPO / "scripts" / "select_step9c_tests.py"

PIN_FILE_RELPATH = "tests/test_issue_skill_dispatch_preflight_pin.py"

ANCHOR = "Dispatch-input/env/flag preflight"

# Anchor for the #2330 item (e); its OWN anchor deliberately — the (e) block
# lands past the existing ANCHOR + 5000 window (the (a)-(d) block is ~4.4k
# chars), so the legacy window must not be relied on for it.
PER_LEG_ISOLATION_ANCHOR = "(e) Per-leg out/scratch isolation"


def test_dispatch_preflight_block_present():
    text = issue_skill_text()
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
    text = issue_skill_text()
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


def test_dispatch_preflight_has_per_leg_out_scratch_isolation_item():
    """Pins the preflight item (e) — per-leg out/scratch isolation (#2330 fu1).

    Reads the COMPOSED orchestrator spec (`issue_skill_text()`) so the pin
    binds wherever the Step 6b body physically lives after the #2155 split
    (currently `.claude/skills/issue/steps/10-step-6.md`).
    """
    text = issue_skill_text()
    anchor_idx = text.index(ANCHOR)
    idx = text.index(PER_LEG_ISOLATION_ANCHOR)
    # Placement: inside the preflight list — after item (d), before the
    # handle-sidecar paragraph that closes the block.
    assert anchor_idx < idx
    assert text.index("(d) Relaunch flags verbatim") < idx
    assert idx < text.index("The handle the dispatch helper returns")
    window = text[idx : idx + 2000]
    # Trigger names already-live siblings, not only batch-composed legs.
    assert "already-live sibling" in window
    # Collision test: same/layout-sharing driver + split + shard indices.
    assert "layout-sharing" in window
    assert "overlapping shard indices" in window
    # ACTION mandate: a possible collision derives a per-leg out/scratch
    # root pre-launch. Two asserts so the line-break/indent between the
    # phrases cannot break the pin.
    assert "derives a PER-LEG" in window
    assert "root BEFORE launch" in window
    # Non-collision escape = basename disjointness; bare driver difference
    # is explicitly NOT blessed.
    assert "BASENAME" in window and "DISJOINTNESS" in window
    assert "driver difference alone is NOT sufficient" in window
    # Disposition idiom + the canonical experimenter.md recipe pointer.
    assert "disposition in the dispatch note" in window
    assert "experimenter.md" in window
    assert "step 1c" in window
    assert "#2330" in window


def test_crash_fix_rounds_per_leg_out_roots_covers_concurrent_legs():
    """Pins the crash-fix-rounds § Per-leg out-roots widening (#2330)."""
    text = CRASH_FIX_MD.read_text(encoding="utf-8")
    i0 = text.index("Per-leg out-roots for regime-keyed drivers")
    # Heading trigger widened to concurrent same-driver legs.
    assert "AND concurrent same-driver" in text[i0 : i0 + 120]
    window = text[i0 : i0 + 2600]
    # Second trigger sentence: #2330 alongside the existing #1333 case.
    assert "SECOND TRIGGER" in window
    assert "#2330" in window
    assert "#1333" in window  # the original regime-keyed case stays
    assert "FileNotFoundError" in window  # the sibling flush death
    # Reap scope-guard: the LATER-leg reap covers chained/dead roots only.
    assert "CHAINED / DEAD sibling roots only" in window
    assert "NEVER reap a root whose" in window
    assert "owning leg is still live" in window


def test_experimenter_launcher_per_leg_isolation_step():
    """Pins experimenter.md "During Execution" step 1c (#2330 fu1)."""
    text = EXPERIMENTER_MD.read_text(encoding="utf-8")
    idx = text.index("1c. **Per-leg out/scratch isolation")
    # Placement: after step 1b's closing generic-contract parenthetical,
    # before step 2's disconnect-survival probe.
    assert text.index("Pid-file launch contract") < idx
    assert idx < text.index("Confirm the launch survived disconnect")
    window = text[idx : idx + 2600]
    # Trigger = RUNTIME CONCURRENCY incl. already-live siblings (not only
    # batch-composed legs), probed via the step-1b liveness probe.
    assert "RUNTIME CONCURRENCY" in window
    assert "already-live sibling" in window
    assert "pgrep -af" in window
    # Collision test + per-leg fix + breadcrumb duty.
    assert "overlapping" in window and "shard indices" in window
    assert "EPM_I<N>_OUT_DIR" in window
    assert "epm:run-launched" in window
    # Non-collision escape: basename disjointness; driver difference alone
    # is explicitly NOT blessed.
    assert "BASENAME DISJOINTNESS" in window
    assert "DRIVER DIFFERENCE ALONE IS NOT SUFFICIENT" in window
    # Sibling cross-refs: #1315 download-staging, #1333 regime-keyed, #2330.
    assert "#1315" in window
    assert "#1333" in window
    assert "#2330" in window


def test_registered_in_step9c_workflow_invariant():
    # Import by path, matching tests/test_select_step9c_tests.py (the
    # selector lives under scripts/, not an importable package).
    spec = importlib.util.spec_from_file_location("select_step9c_tests_1964", SELECTOR_PY)
    assert spec is not None and spec.loader is not None
    sel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sel)
    assert PIN_FILE_RELPATH in sel.WORKFLOW_INVARIANT
