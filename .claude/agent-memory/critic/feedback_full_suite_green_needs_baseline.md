---
name: Full-suite-green criteria need a pre-existing-failure baseline
description: Full-pytest-suite-green (and repo-wide ruff check) acceptance criteria are false-FAIL-by-construction — main routinely carries pre-existing failures; require a recorded baseline + no-NEW-failures wording (#584, #588, #554)
type: feedback
---

A criterion of the form `uv run pytest tests/ -q  # full suite green` is unsatisfiable whenever main HEAD already carries failures — a guaranteed false FAIL, with no recorded way to distinguish an introduced failure from a pre-existing one.

**Why:** 2026-06-10 (#584): main HEAD had 11 deterministic pre-existing workflow-pin failures (drift from the fresh /campaign skill). Recurred 2026-06-11 (#588): same 11 still live; the plan said "Full test suite green" verbatim → REVISE'd. Recurred 2026-06-12 (#554): the 11 were fixed but the suite STILL wasn't green (1 different failure) — the specific failure set rotates while the class persists. Same class, new axis: repo-wide `uv run ruff check .` was also false-FAIL-by-construction (1761 pre-existing errors at HEAD; the workflow-improver spec already scopes ruff to touched files).

**How to apply:** when a plan's §6 includes a full-suite run, REVISE unless it (a) records a pre-change full-suite baseline in its tests-first step and (b) states the criterion as "no NEW failures vs the baseline" (or scopes to touched test files). Same rule for repo-wide `ruff check .` — scope to touched files or baseline it. Verify by actually running the suite (or the workflow-pin files) + `ruff check . | tail -1` at review time — the failures are deterministic, not flakes.

**Counter-point (#626, 2026-06-12):** the class is intermittent on the pytest axis (suite was GREEN at review: 3291 passed) but near-permanent on ruff (~1761 pre-existing). Verdict hinges on the LIVE run: green suite -> Concern (recommend baseline phrasing anyway); red suite -> REVISE.
