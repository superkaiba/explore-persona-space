"""Durability pins for the #1289 fetched-`origin/main` diff-base default.

The shared repo-root `main` can lag `origin/main` (2026-07-12: three
concurrent sessions got foreign-file pollution — #1280's 202,578-byte
`main...HEAD` diff vs 11,637 against `origin/main`; #1281's 41-test-file
gate). #1289 defaulted the Step 9c selector's diff base to FETCHED
`origin/main` and aligned the reviewer diff-scoping recipes. These pins read
the LIVE tree (the #1242/#1268 SKILL.md-content-pin pattern) so a future
SKILL.md-only or agent-prose-only edit reverting a recipe to local `main`
fails loudly. This file is a `WORKFLOW_INVARIANT` member — SKILL.md diffs
gate ONLY via that tuple.

Stale-literal checks are EXACT literals (e.g. ``git diff main...HEAD | wc -c``)
— never a loose regex, which would false-fail on the fixed
``origin/main...HEAD`` form (whose substring contains ``main...HEAD``).
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SKILL = _ROOT / ".claude" / "skills" / "issue" / "SKILL.md"
_CODE_REVIEWER = _ROOT / ".claude" / "agents" / "code-reviewer.md"
_DIFF_BUDGET = _ROOT / ".claude" / "rules" / "diff-size-budget.md"
_SELECTOR = _ROOT / "scripts" / "select_step9c_tests.py"


def _load_selector():
    """Import the selector by path (it lives under scripts/, not a package)."""
    spec = importlib.util.spec_from_file_location("select_step9c_tests_pin", _SELECTOR)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_step9c_selector_invocation_has_no_local_main_base():
    """No SKILL.md selector invocation pins `--base main` (the pre-#1289 recipe).

    The Step 9c 1a recipe must let the in-script fetched-origin/main default
    apply; an explicit `--base main` there reintroduces the lagging-root
    foreign-file pollution (#1281).
    """
    text = _SKILL.read_text()
    lines = [ln for ln in text.splitlines() if re.search(r"select_step9c_tests\.py", ln)]
    assert lines, f"no select_step9c_tests.py invocation found in {_SKILL}"
    offenders = [ln for ln in lines if "--base main" in ln]
    assert offenders == [], (
        f"SKILL.md selector invocation(s) pin '--base main' (pre-#1289 recipe): {offenders}"
    )


def test_selector_default_base_is_fetched_origin_main():
    """The selector's in-script default is fetched origin/main (#1289)."""
    sel = _load_selector()
    assert sel.DEFAULT_BASE == "origin/main"
    assert sel.FETCH_TIMEOUT_S == 120  # the SKILL.md Step 10d bounded-fetch precedent
    # Parser wiring pin: the argparse default IS the module constant (a literal
    # "main" default silently reverting the wiring would keep DEFAULT_BASE green).
    src = _SELECTOR.read_text()
    assert "default=DEFAULT_BASE" in src, (
        "select_step9c_tests.py --base is no longer wired to DEFAULT_BASE"
    )


def test_reviewer_scoping_recipes_use_origin_main():
    """code-reviewer.md + diff-size-budget.md scope diffs against origin/main.

    Exact-literal checks: the fixed `origin/main...HEAD` form CONTAINS the
    substring `main...HEAD`, so the stale check pins the full command literal
    (`git diff main...HEAD | wc -c`), never a bare `main...HEAD` regex.
    """
    reviewer = _CODE_REVIEWER.read_text()
    budget = _DIFF_BUDGET.read_text()
    # Fixed forms present:
    assert "git diff origin/main...HEAD | wc -c" in reviewer
    assert "git diff origin/main...HEAD | wc -c" in budget
    assert "git diff --name-only origin/main...HEAD" in reviewer  # Step 0 classify
    # Stale local-main sizing literal absent from both:
    assert "git diff main...HEAD | wc -c" not in reviewer
    assert "git diff main...HEAD | wc -c" not in budget
