"""Token-presence pins for the #1305 gate-scope verification duty (#1288).

The duty lives in workflow PROSE — `.claude/agents/implementer.md`
(§ After Implementation item 1 + the `(c) How to verify` report template),
`.claude/skills/issue/SKILL.md` (the Step 4b implementer-brief bullet), and
`.claude/agents/experiment-implementer.md` (the mapped-leg checklist item) —
so, mirroring `tests/test_step10d_guard3.py`, these tests pin load-bearing
TOKENS inside region-scoped slices of those files (rewording survives; a
silent drop of the duty does not).

What is pinned (plan #1305 §4.4 + the binding review concerns):

(i)   The SKILL.md Step 4b brief region carries `select_step9c_tests.py`,
      and the duty bullet does NOT pair it with a CONTIGUOUS
      `--base main` invocation. The bullet's prose legitimately contains
      the literal `--base main` inside the "never `--base main`" warning —
      the negative assertion therefore discriminates the contiguous command
      pairing (`select_step9c_tests.py [flags...] --base main`) from that
      warning phrase, and a companion positive assertion keeps the warning
      itself from being deleted to satisfy the pin.
(ii)  implementer.md § After Implementation carries `select_step9c_tests.py`
      plus a pin-sweep keyword (same no-contiguous-`--base main` check).
(iii) experiment-implementer.md § After implementation carries `--map-files`
      (the Step 10d mapped-leg variant — experiment kinds skip Step 9c).
(iv)  implementer.md's `### (c) How to verify` report template carries the
      `Gate-scope check` line.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SKILL = _REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"
_IMPLEMENTER = _REPO_ROOT / ".claude" / "agents" / "implementer.md"
_EXP_IMPLEMENTER = _REPO_ROOT / ".claude" / "agents" / "experiment-implementer.md"

# A CONTIGUOUS invocation pairing: `select_step9c_tests.py` followed only by
# flag-like tokens ending in `--base main`. Prose like "never `--base main`"
# (non-flag words / punctuation between) deliberately does NOT match.
_CONTIGUOUS_BASE_MAIN = re.compile(r"select_step9c_tests\.py(?:\s+--?[\w=/.<>-]+)*\s+--base\s+main")


def _region(text: str, start_marker: str, end_marker: str, *, label: str) -> str:
    """Slice ``text`` between two unique-enough anchors, asserting both exist
    and are ordered — the `test_step10d_guard3.py` region-scoping pattern."""
    start = text.find(start_marker)
    end = text.find(end_marker, start + len(start_marker) if start != -1 else 0)
    assert start != -1, f"{label}: start marker not found: {start_marker!r}"
    assert end != -1, f"{label}: end marker not found: {end_marker!r}"
    assert start < end, f"{label}: start marker must precede end marker"
    return text[start:end]


def _has_contiguous_base_main_invocation(region: str) -> bool:
    """True iff the region pairs the selector with `--base main` as one command."""
    collapsed = re.sub(r"\s+", " ", region)
    return bool(_CONTIGUOUS_BASE_MAIN.search(collapsed))


# --------------------------------------------------------------------------
# Pin (i) — SKILL.md Step 4b brief bullet
# --------------------------------------------------------------------------


def _step4b_brief_region() -> str:
    return _region(
        _SKILL.read_text(encoding="utf-8"),
        "Brief passed to the implementer:",
        "**TDD mode (opt-in).**",
        label="SKILL.md Step 4b brief region",
    )


def _step4b_duty_bullet() -> str:
    return _region(
        _step4b_brief_region(),
        "gate-scope verification duty",
        "**Marker-version discipline",
        label="SKILL.md Step 4b gate-scope duty bullet",
    )


def test_step4b_brief_carries_gate_scope_duty():
    region = _step4b_brief_region()
    assert "select_step9c_tests.py" in region, (
        "Step 4b implementer-brief region must carry the gate-scope duty "
        "(selector enumeration via select_step9c_tests.py; #1288/#1305)"
    )
    duty = _step4b_duty_bullet()
    assert "select_step9c_tests.py" in duty
    assert "pin-sweep" in duty.lower(), (
        "the Step 4b duty bullet must name the changed-literal pin-sweep"
    )


def test_step4b_duty_bullet_never_pairs_selector_with_base_main():
    duty = _step4b_duty_bullet()
    # The warning phrase must stay (never resolved by deleting it)...
    assert "--base main" in duty, (
        "the duty bullet must keep the 'never `--base main`' warning phrase — "
        "do not delete the warning to satisfy the contiguous-pairing pin"
    )
    # ...but the selector must never be INVOKED with --base main in the duty.
    assert not _has_contiguous_base_main_invocation(duty), (
        "the Step 4b duty bullet pairs select_step9c_tests.py with a contiguous "
        "`--base main` invocation — the duty must use the DEFAULT (fetched "
        "origin/main) base per #1289"
    )


# --------------------------------------------------------------------------
# Pin (ii) — implementer.md § After Implementation
# --------------------------------------------------------------------------


def _implementer_after_impl_region() -> str:
    return _region(
        _IMPLEMENTER.read_text(encoding="utf-8"),
        "### After Implementation",
        "### Local runs are same-turn, synchronous work",
        label="implementer.md After Implementation section",
    )


def test_implementer_spec_carries_gate_scope_duty():
    region = _implementer_after_impl_region()
    assert "select_step9c_tests.py" in region, (
        "implementer.md § After Implementation must enumerate the Step 9c "
        "selection via select_step9c_tests.py (#1288/#1305)"
    )
    assert "pin-sweep" in region.lower(), (
        "implementer.md § After Implementation must name the changed-literal "
        "pin-sweep over the enumerated selection"
    )
    assert not _has_contiguous_base_main_invocation(region), (
        "implementer.md's duty pairs select_step9c_tests.py with a contiguous "
        "`--base main` invocation — must use the DEFAULT (fetched origin/main) base"
    )


# --------------------------------------------------------------------------
# Pin (iii) — experiment-implementer.md mapped-leg duty
# --------------------------------------------------------------------------


def test_experiment_implementer_carries_mapped_leg_duty():
    region = _region(
        _EXP_IMPLEMENTER.read_text(encoding="utf-8"),
        "### After implementation (mandatory checklist)",
        "### Smoke runs are same-turn, synchronous work",
        label="experiment-implementer.md After implementation checklist",
    )
    assert "--map-files" in region, (
        "experiment-implementer.md's after-implementation checklist must carry "
        "the Step 10d mapped-scan enumeration (`select_step9c_tests.py "
        "--map-files ...`); experiment kinds skip Step 9c, so the mapped "
        "invariant-test leg is their merge-time test surface (#1144/#1288)"
    )
    assert "pin-sweep" in region.lower(), (
        "experiment-implementer.md's checklist must name the changed-literal pin-sweep over tests/"
    )


# --------------------------------------------------------------------------
# Pin (iv) — implementer.md (c) How to verify template
# --------------------------------------------------------------------------


def test_implementer_report_template_carries_gate_scope_check():
    region = _region(
        _IMPLEMENTER.read_text(encoding="utf-8"),
        "### (c) How to verify",
        "### (d) Needs human eyeball",
        label="implementer.md (c) How to verify template",
    )
    assert "Gate-scope check" in region, (
        "implementer.md's `(c) How to verify` report template must carry the "
        "`Gate-scope check` bullet (selector n_tests + base, locally-run subset, "
        "pin-sweep summary, deferred invariant-only count; #1305)"
    )
