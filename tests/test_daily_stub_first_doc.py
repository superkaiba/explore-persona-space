"""Spec-doc regression guard for the /daily stub-first headless rule (#1189).

`/daily` is an LLM-driven SKILL.md, not a runtime surface, so the mechanical
guard against a future cosmetic edit silently dropping the stub-first rule (or
its load-bearing invariants) is a pure-text assertion that the prose contract
survives — the same pattern as ``test_daily_three_route_classifier_doc.py``
(#706). The runtime half (the healthcheck husk arm) is behavior-pinned by
``tests/test_cron_daily_healthcheck.py``.

Minimal + durable by design — substring assertions, NOT structure parsing.
Pinned here:

* The mechanical headless rule 0 exists (``Stub-first (mechanical``), with its
  before-any-mining ordering phrase and the skeleton commit-message string.
* The skeleton's ``## Applied workflow improvements`` section stays EMPTY —
  the invariant whose silent drop would blind the healthcheck husk detector
  AND break the § Output refuse-rule interplay (a placeholder-bearing husk
  would refuse its own backfill recovery).
* The five-vs-six H2 count inconsistency stays fixed (both mentions say six).
* Cross-file agreement: the literal ``## Applied workflow improvements`` H2
  appears in BOTH the skill and the healthcheck script — a rename of the
  section in one file fails this test instead of causing daily false alarms.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DAILY_SKILL = REPO_ROOT / ".claude" / "skills" / "daily" / "SKILL.md"
HEALTHCHECK = REPO_ROOT / "scripts" / "cron_daily_healthcheck.sh"

APPLIED_H2 = "## Applied workflow improvements"


@pytest.fixture(scope="module")
def daily_skill_text() -> str:
    assert DAILY_SKILL.is_file(), f"daily SKILL.md not found at {DAILY_SKILL}"
    return DAILY_SKILL.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def healthcheck_text() -> str:
    assert HEALTHCHECK.is_file(), f"healthcheck script not found at {HEALTHCHECK}"
    return HEALTHCHECK.read_text(encoding="utf-8")


def test_headless_stub_first_rule_pinned(daily_skill_text: str):
    """Headless rule 0: skeleton written + committed BEFORE any mining work."""
    assert "Stub-first (mechanical" in daily_skill_text
    # The ordering phrase: the skeleton is the run's first action, before the
    # slow work the harness can kill through.
    assert "before any" in daily_skill_text
    assert "transcript mining" in daily_skill_text
    # The skeleton commit message (§ Commit recipe, run at rule-0 time).
    assert "logs: daily stub for" in daily_skill_text


def test_skeleton_applied_section_stays_empty(daily_skill_text: str):
    """The skeleton's Applied section is EMPTY — load-bearing for the husk
    detector (an empty section is what the healthcheck arm reads as
    "never enriched") and for the § Output refuse-rule interplay (a
    placeholder-bearing husk would refuse its own backfill recovery)."""
    assert "stays empty until" in daily_skill_text
    assert "bodies EMPTY" in daily_skill_text


def test_h2_section_count_consistent(daily_skill_text: str):
    """The five-vs-six H2 count inconsistency (pre-#1189 line 53) stays fixed."""
    assert "five H2" not in daily_skill_text
    assert daily_skill_text.count("six H2 sections") >= 2


def test_applied_h2_pinned_in_skill_and_healthcheck(daily_skill_text: str, healthcheck_text: str):
    """The skill's done-predicate H2 and the healthcheck detector's awk pattern
    name the SAME literal section — a rename in either file fails here instead
    of producing daily false alarms (or silent non-detection) at 06:00."""
    assert APPLIED_H2 in daily_skill_text
    assert APPLIED_H2 in healthcheck_text
