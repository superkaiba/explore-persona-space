---
name: reconstructed-marker-c-roster-audit
description: Orchestrator-reconstructed implementer markers omit the two conditional (c) roster fields (ruff-policy pin, Gate-scope check); run both instruments yourself and attach measured evidence so the marker-shape blockers cure by re-post
metadata:
  type: feedback
---

When the round's implementation marker is an ORCHESTRATOR RECONSTRUCTION
(watcher-killed implementer, `pre_split_review_guard` → orchestrator-owned
post), audit the `(c)` roster against the two CONDITIONAL fields the
reconstruction predictably omits — a brief line saying the marker "conforms"
covers the four-H3 shape, not these:

1. **Ruff-policy pin field (#1716)** — required whenever the diff touches a
   `tests/test_ruff_policy.py::LIVE_WORKFLOW_HELPERS` path (workflow_lint.py
   and select_step9c_tests.py are members, so most workflow-surface rounds
   trigger it). Absence ⇒ marker-shape Critical.
2. **`Gate-scope check (#1288):` line (Step 4.6)** — required on every
   `type:infra` `epm:results` with ts ≥ 2026-07-15. Absence ⇒ a SECOND,
   separately-keyed marker-shape Critical whose body must name the literal
   `Gate-scope check` (5c-bis strips per blocker on that name).

Then RUN both instruments yourself and put the measured results in the
verdict: `uv run pytest tests/test_ruff_policy.py -q` and
`select_step9c_tests.py --map-files <file-list-file>` (the flag takes ONE
file-of-paths argument, not positional paths) + the key mapped tests. With
evidence attached, the FAIL is mechanical-contract-only and the cure is a
v3 re-post transcribing your numbers — zero re-work, no substance blocked.

Sibling check the same rounds keep hitting: a NEW `scripts/` helper invoked
from a skill-doc arm must JOIN `LIVE_WORKFLOW_HELPERS` (the roster's own
add-here comment; `step5a_sibling_probe.py` precedent) — and run the full
ruleset on it first (`ruff check --config 'lint.per-file-ignores = {}'`):
fresh helpers routinely carry SIM105/C901 that the project config hides, so
enrollment is code touches + list append, not a one-liner.

**Why:** #2327 R1 g1 — reconstructed `epm:results` v2 omitted BOTH fields
while the diff touched two roster paths; the new helper was unenrolled and
full-ruleset-red (SIM105 L183, C901 L319). Running the pin (2 passed) and
the selector (141 pairs / 76 tests) myself turned two would-be bounce
rounds into a transcription fix.

**How to apply:** any CONTRACT-BEARING review where the marker is
orchestrator-posted / reconstructed, and any diff adding a `scripts/`
workflow helper.
