"""Spec-doc regression guard for the /daily three-route classifier (#706).

`/daily` is an LLM-driven SKILL.md, not a runtime surface, so the only
mechanical guard against a future cosmetic edit collapsing the three-route
classifier back to a binary apply-vs-hold split is a pure-text assertion that
the prose contract survives. The watcher invariant test
(`test_autonomous_session_watch.py::test_sweep_candidate_query_skips_needs_human`)
pins the runtime half; this file pins the prose half.

Minimal + durable by design — substring assertions, NOT structure parsing.
Each assertion is a string the round-2 implementation MUST keep in
`.claude/skills/daily/SKILL.md`:

* The THREE route labels exist (trivial mechanical / behavior-or-logic / a
  genuine judgment call) — proving the binary classifier was replaced.
* Route 2 wires to `file_infra_task.py` with `--tag daily-auto-filed`.
* Route 3 wires to `file_infra_task.py --no-dispatch` with both
  `--tag needs-human` and `--tag daily-held`.
* The 5-item judgment-call carve-out list survives verbatim (it is REUSED as
  the route-3 trigger, not re-authored).

At the TDD propose stage every assertion FAILs (the SKILL.md still carries
the binary two-bucket classifier); the round-2 implementation makes them pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DAILY_SKILL = REPO_ROOT / ".claude" / "skills" / "daily" / "SKILL.md"


@pytest.fixture(scope="module")
def daily_skill_text() -> str:
    assert DAILY_SKILL.is_file(), f"daily SKILL.md not found at {DAILY_SKILL}"
    return DAILY_SKILL.read_text(encoding="utf-8")


# ── the three route labels replace the binary classifier ──────────────────────


@pytest.mark.parametrize(
    "label",
    [
        "Trivial mechanical",  # route 1 — self-apply, no behavior change
        "behavior/logic change",  # route 2 — file for independent review
        "judgment call",  # route 3 — tracked needs-human task
    ],
)
def test_three_route_labels_present(daily_skill_text: str, label: str):
    assert label in daily_skill_text, (
        f"route label {label!r} missing from daily/SKILL.md — the three-route "
        "classifier may have regressed to a binary apply-vs-hold split"
    )


# ── route 2 files for review via file_infra_task.py + daily-auto-filed tag ─────


def test_route2_files_for_review(daily_skill_text: str):
    assert "file_infra_task.py" in daily_skill_text, (
        "route 2 must wire to scripts/file_infra_task.py (files + auto-dispatches "
        "behavior/logic changes to /issue --auto for independent review)"
    )
    assert "daily-auto-filed" in daily_skill_text, (
        "route 2 must tag filings with daily-auto-filed (distinguishes /daily "
        "review filings from manual workflow-fix-on-bug filings; feeds the PM digest count)"
    )


# ── route 3 files a tracked needs-human task (no dispatch) ─────────────────────


@pytest.mark.parametrize(
    "token",
    [
        "--no-dispatch",  # files at proposed WITHOUT spawning a session
        "needs-human",  # the PM-surfaced, auto-dispatch-excluded tag
        "daily-held",  # marks the held item as /daily-originated
    ],
)
def test_route3_files_tracked_needs_human_task(daily_skill_text: str, token: str):
    assert token in daily_skill_text, (
        f"route 3 must reference {token!r} — a /daily-held judgment call becomes a "
        "TRACKED proposed task (file_infra_task.py --no-dispatch --tag needs-human "
        "--tag daily-held), no longer a dead-end log note"
    )


# ── the 5-item carve-out list survives (reused verbatim as the route-3 trigger) ─


@pytest.mark.parametrize(
    "carve_out_anchor",
    [
        "Scientific-meaning changes",
        "Destructive / irreversible actions",
        "Spends money or launches compute",
        "External side-effects",
        "Genuinely ambiguous intent",
    ],
)
def test_carve_out_list_survives(daily_skill_text: str, carve_out_anchor: str):
    assert carve_out_anchor in daily_skill_text, (
        f"judgment-call carve-out item {carve_out_anchor!r} dropped — the 5-item "
        "list is REUSED verbatim as the route-3 trigger and must be preserved"
    )
