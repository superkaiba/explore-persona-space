---
title: 'daily-fix: code-reviewer has no full-ruleset ruff policy-pin'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ad4c77fa3602
- daily-auto-filed
created_at: '2026-07-27T07:14:19Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-26 problem sweep (route 2): code-reviewer.md prescribes
  only a bare ruff check, which per-file-ignores make blind on scripts/*, so round-1
  reviews certify ruff-clean on diffs that then FAIL the Step 9c full-ruleset policy
  pin'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-26 problem sweep (route 2). Surfaced by 4 independent
miner group(s) over the 2026-07-26 session transcripts.

## Goal

Add the full-ruleset ruff-policy-pin leg to `.claude/agents/code-reviewer.md`'s mechanical
pre-pass, mirroring the leg `implementer.md` already carries, so a bare `ruff check` can no
longer be written up as "ruff clean" on a diff that fails the Step 9c policy pin.

## Workflow gap

- **Bug observed:** `code-reviewer.md` prescribes only a bare `uv run ruff check
  path/to/changed/files`, which `pyproject.toml`'s per-file-ignores make blind on
  `scripts/*`, so round-1 reviews returned PASS on diffs that the Step 9c gate then FAILed
  on `tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset`.
- **Why it is a workflow gap:** #1699 added the policy-pin duty to `implementer.md` and
  `experiment-implementer.md` only; the code-reviewer — the second net, and the one whose
  verdict gates the round — was never updated, so the 0.3 s check is first executed by a
  ~25-minute pytest gate.
- **Confidence (emitter):** high
- verified-at-filing: absence-of-guard claim, per-target —
  `grep -nc 'test_ruff_policy\|LIVE_WORKFLOW_HELPERS\|per-file-ignores' .claude/agents/code-reviewer.md`
  → **0 hits**; semantic probe (fragments, not the verbatim literal)
  `grep -nic 'full-ruleset\|full ruleset\|ruff_policy\|policy pin\|policy-pin' .claude/agents/code-reviewer.md`
  → **0 hits**. The only ruff mentions in the target are `grep -n 'ruff' .claude/agents/code-reviewer.md`
  → **5 hits**: L99 (`Check style — ruff compliance…`), L1514 (`uv run ruff check
  path/to/changed/files`), L1515 (`uv run ruff format --check …`), L1533, L1800 — all bare
  forms. Partner presence confirmed:
  `grep -n 'test_ruff_policy' .claude/agents/implementer.md` → **1 hit at L176**, carrying
  the pin leg verbatim (`… AND uv run pytest tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset -x
  (the policy pin the gate enforces on live workflow helpers, measured 0.30 s total /
  0.03 s test call on 2026-07-26)`); `grep -nc 'test_ruff_policy' .claude/agents/experiment-implementer.md`
  → **2 hits**. Target test exists: `grep -n 'LIVE_WORKFLOW_HELPERS\|def test_live_workflow_helpers_clean_under_full_ruleset' tests/test_ruff_policy.py`
  → L44 (`LIVE_WORKFLOW_HELPERS = [`), L101 (the test def), L111, L122.
  Landed-fix check: `git log --oneline --since='7 days ago' -- .claude/agents/code-reviewer.md`
  → 3 commits (`ad3549bc2a`, `164b46cd5c`, `abd1f5c737`), none touching ruff. (2026-07-26)

## Evidence

- Session `564d9a53`, 2026-07-26: the round-1 implementer ran only `uv run ruff check
  <changed files>` (twice, 07:46:52Z / 07:47:06Z) and the code-reviewer ran only
  `uv run ruff check <files>` + `ruff format --check <files>`, then PASSed at 08:03:45Z
  with the verdict line `"**Tests:** PASS (5/5 new tick_triage + 26/26 selected watcher +
  63/63 regression on \`boot_death\` + 1/1 SKILL contract pin, ruff clean)"`. 24 min later
  the Step 9c gate FAILed: `"PYTEST_RC=1 … 1 failure
  (test_live_workflow_helpers_clean_under_full_ruleset) — ruff C901 complexity + SIM108 in
  the rewired _process_boot_death"` (`_process_boot_death` cyclomatic complexity crossed
  15→19). Full recovery round: test-verdict FAIL marker → implementer round 2 →
  code-review round 2 → a second 24 min 43 s gate re-run. Measured cost ≈ 42 min wall
  (round-2 implementer 10 min + review 3 min + 25-min gate re-run + orchestrator turns)
  plus one wasted 22-min gate. The IMPLEMENTER side of this gap was closed by #1699, which
  merged at 13:01:52Z — ~5.5 h AFTER this round-1 spawn (07:33Z), so it was not live; the
  REVIEWER side remains uncovered.
- Session `891b2cc6`, 2026-07-26T14:20:20Z: the Step 5 code-reviewer returned
  `"**Verdict:** PASS (round 1 of task #1704 code review)"` on the `partial_bundle_pass`
  diff; ~30 min later the Step 9c gate FAILed on that same diff —
  `"scripts/autonomous_session_watch.py:7588:67 — RUF003 ambiguous × (MULTIPLICATION SIGN)
  in comment … scripts/autonomous_session_watch.py:7864:5 — C901 partial_bundle_pass
  complexity 16 > 15 pin."` The round-1 PASS was worthless as a gate: a 26-min pytest run
  was the first thing to catch a 0.3 s check.
- Session `a2c4bae3`, 2026-07-26T16:21:21Z: the round-1 implementer ran bare ruff only and
  shipped 8 full-ruleset errors — `"scripts/autonomous_session_watch.py:6528:37: RUF003
  Comment contains ambiguous \`∪\` (UNION). … :6637:15: UP037 [*] Remove quotes from type
  annotation"` — despite `implementer.md:176` having landed the pin duty ~2 h 20 m earlier
  (commit `d79fa07b0e`, 13:01Z; this implementer ran ~15:20Z). Two points follow: the
  reviewer had no pin duty to catch it, and the implementer-side duty is prose rather than
  a reportable, mechanically-checkable field.
- Session `06447a89`, 2026-07-26T08:28:32Z → 09:04:06Z: the Step 9c gate (24 min) failed on
  exactly one test — `PYTEST_RC=1 with ONE failure:
  test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset — a B007 lint
  under full ruleset` — for an unused loop variable `tid_raw` at
  `scripts/autonomous_session_watch.py:7662`. The round-2 fix was renaming `tid_raw` →
  `_tid_raw` (`"1-char change, zero behavioral risk"`), and the round-2 implementer's own
  verification ran the exact discriminating command
  `ruff check <file> --config 'lint.per-file-ignores = {}'`. Cost ≈ 35 min of avoidable
  wall-clock (gate round-1 24 min + compare 1 min + implementer round 2 6 min + gate
  round-2 27 min) and one extra commit.
- Four sessions, one root cause: `pyproject.toml`'s per-file-ignores relax rules on
  `scripts/*`, and the only surface that runs the unrelaxed ruleset is the ~25-minute Step
  9c gate. The discriminating check costs 0.30 s (measured, quoted in `implementer.md:176`).

## Proposed change

- In `.claude/agents/code-reviewer.md`, next to the ruff block at ~L1514-1515, add the
  policy-pin leg mirroring `implementer.md:176`: when the diff touches any path listed in
  `tests/test_ruff_policy.py`'s `LIVE_WORKFLOW_HELPERS`, the reviewer MUST also run
  `uv run pytest tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset -x`
  and treat a failure as a blocker (tag `substantive`).
- Add the explicit prohibition at the same site: the reviewer may NOT write "ruff clean" —
  in the verdict line or anywhere in the report — from a bare `ruff check` alone on a diff
  touching `LIVE_WORKFLOW_HELPERS`. Bare `ruff check` is not a style verdict on `scripts/*`.
- Mirror the one-line rationale into the reviewer's L99 style bullet ("ruff compliance")
  so a reader who never reaches L1514 still sees the per-file-ignores caveat.
- Make the implementer side mechanically checkable: in `.claude/agents/implementer.md`
  § final report, promote the pin from prose into the MANDATORY `(c)` checklist — require
  the marker body to carry the literal command plus its exit code — and add the
  corresponding marker-shape check to `code-reviewer.md` Step 0.5 so a missing pin line
  FAILs with the `marker-shape` tag rather than passing silently.
- The equivalent discriminating one-liner
  (`uv run ruff check <touched files> --config 'lint.per-file-ignores = {}'`) may be
  documented as the fast local probe, but the PIN TEST is the authoritative form — it is
  what the gate runs, and it is the one whose node id the FAIL will name.

## Scope / surfaces

- Primary target: `.claude/agents/code-reviewer.md`
- `.claude/agents/implementer.md` (promote the L176 pin from prose into the mandatory `(c)`
  report checklist with command + exit code)
- `.claude/agents/codex-code-reviewer.md` (the Codex twin composes the same rubric — the
  pin leg must ride the composed prompt, or the twin reviews a weaker rubric than its
  Claude counterpart)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `uv run python scripts/workflow_lint.py` passes (no-flags); ruff clean on touched files —
  and this task's own diff must pass the pin it is adding.
- Do not change `pyproject.toml`'s per-file-ignores or `tests/test_ruff_policy.py`'s
  `LIVE_WORKFLOW_HELPERS` roster; the relaxation is deliberate and the pin is the
  compensating control.
- Agent-spec byte budgets are enforced by `workflow_lint.py`; keep the added text tight —
  `code-reviewer.md` is already a large spec.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route
  its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: ad4c77fa3602

- workflow_fix_target: .claude/agents/code-reviewer.md
- fingerprint: PENDING

/daily 2026-07-26 route-2 filing. Miner refs: F-P2, I-P3, B-P6, E-P4.
