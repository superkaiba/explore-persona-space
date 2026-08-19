---
title: 'workflow-fix: raise the per-reviewer review-round cap from 5 to 10 across
  all three review loops'
kind: infra
tags:
- wf-fix
created_at: '2026-08-19T19:05:03Z'
has_clean_result: false
origin_prompt: 2147 -> approve and change workflow so we can have up to 10 reviewer
  rounds (user, 2026-08-19 interactive chat; scope confirmed to all three loops via
  follow-up)
workflow: v1
---
# workflow-fix: raise the per-reviewer review-round cap from 5 to 10 across all three review loops

## Goal

Raise the per-reviewer round cap from **5 to 10** on every iterating review loop in the
workflow, so a review site with a genuine substantive residual at round 5 can keep
iterating instead of parking the task at `blocked`.

Direct motivation: **#2147** hit the cap at 5/5 with a CONFIRMED substantive residual
(Claude PASS vs Codex FAIL on a real bug in a deletion path — a text-mode git stdout CR
translation making an authoritative worktree-registration probe return a
successful-but-incomplete set). The cap-hit terminal is working as designed — it
SURFACED rather than shipping past — but 5 rounds proved too few for a site that was
still making real progress. The fix is more headroom, not a weaker terminal.

User decision (2026-08-19, interactive chat): "change workflow so we can have up to 10
reviewer rounds", scoped by follow-up to **all three loops** (ensemble review sites,
v2 report loops, adversarial-planner-v2 lens panel).

## Scope — three loops, all in scope

**A. Ensemble review sites** (the 4 doubled iterating Claude+Codex sites: `critic`,
`code-reviewer`, `interpretation-critic`, `clean-result-critic`; `follow-up-critic` is a
single-pass redundancy screen and is NOT affected).

**B. Workflow-v2 report loops** — `methodology-critic` (issue-v2 Step 7d) and
`report-verifier` (Step 7e).

**C. adversarial-planner-v2 lens panel** — the per-lens critic cap in Phase 2
(`statistics-critic` / `methodology-baselines-critic` / `efficiency-critic`).

## Surfaces (measured 2026-08-19; 36 literal cap-5 hits across 12 files + tests)

Authoritative knobs:

- `.claude/workflow.yaml` — `ensemble_review.round_cap_per_reviewer: 5` (~L1215)
- `.claude/workflow.yaml` — `reviewer_pairs.max_rounds: 5` (~L1451)

The Pydantic schema (`src/explore_persona_space/workflow.py:121`) is
`round_cap_per_reviewer: int = Field(ge=1)` — no hardcoded default, so it reads the YAML.
Confirm no other code path hardcodes 5.

Per-file literal counts from the grep sweep:

```
12  .claude/workflow.yaml            5  .claude/skills/issue-v2/SKILL.md
 4  .claude/rules/agents-vs-skills.md 3  .claude/agents/codex-clean-result-critic.md
 2  .claude/skills/issue/SKILL.md     2  .claude/skills/adversarial-planner-v2/SKILL.md
 2  .claude/agents/report-verifier.md 2  .claude/agents/methodology-critic.md
 1  CLAUDE.md                         1  .claude/rules/codex-ensemble-review.md
 1  .claude/agents/interpretation-critic.md
 1  .claude/agents/clean-result-critic.md
```

Also carrying cap-coupled text (verify during planning, counts not in the grep above):
`.claude/agents/codex-interpretation-critic.md`, `.claude/agents/reconciler.md`
(the "reconcile rounds do NOT count toward the cap" invariant — MUST be preserved),
`.claude/skills/adversarial-planner/SKILL.md`.

Non-obvious coupled surfaces that a naive find-and-replace will miss:

1. **`pivot_criteria` trigger NAMES embed the number** —
   `code_review_ensemble_cap_5_surface`, `interpretation_critic_cap_5_surface`,
   `clean_result_critic_cap_5_surface`. Renaming them is a public-ish contract change
   (the #784 precedent renamed `..._cap_3` -> `..._cap_5`). Decide rename-vs-keep in
   planning and state the reason; if renamed, sweep every reference.
2. **`.claude/skills/issue/SKILL.md` flow-diagram branches + exit-kind table** carry
   numeric comparisons, not prose: `FAIL + count<5 --> running`,
   `FAIL + count>=5 --> ... Step 5d cap-hit rule`, and the exit-kind row
   `Step 5b code-review FAIL revision_round>=5`.
3. **`tests/test_ensemble_review_cap.py`** pins ALL of the above and additionally runs a
   WIDENED stale-number scan with an out-of-scope context allowlist. Its whole
   "stale-`3`" apparatus becomes a "stale-`5`" apparatus. Read its module docstring
   before editing — it enumerates the five out-of-scope look-alikes that must NOT be
   swept (Step 9c test-verdict gate, cheap-band follow-up cap, crash-fix K=4 circuit
   breaker, uploader 3-round loop, generic/infra-respawn/plan-contradiction pivots).
4. `tests/test_issue_tick_skill.py` also references the cap.

## Out of scope — do NOT sweep these look-alikes

Every other "5"/"3" in the workflow surface stays untouched, specifically: the Step 9c
test-verdict gate, the cheap-band follow-up round cap (C2 = 2), the crash-fix K=4
circuit breaker, the uploader 3-round loop, the infra-respawn cap-3 pivot, the
plan-contradiction pivot, and `follow-up-critic`'s single-pass screen.

## Invariants that MUST survive

- **Reconciler invocations still do not count toward the cap.**
- **The cap-hit terminal is unchanged in KIND** — at the (new) cap with a substantive
  residual the orchestrator still applies the mechanical-contract-only strip once more
  and then either continues (all residual stripped -> PASS) or SURFACES; it never ships
  past. Autonomous: `epm:failure v1` + `status:blocked` + notify + CRON-TEARDOWN.
  Interactive: surface to the user. This task raises the NUMBER, not the policy.
- The mechanical-contract-only strip tags (`marker-shape`, `smoke-run-missing`,
  `git-provenance`) and their evidence rules are untouched.
- `follow-up-critic` / `codex-follow-up-critic` stay single-pass.

## Acceptance criteria

1. `load_workflow_yaml().ensemble_review.round_cap_per_reviewer == 10` and raw
   `reviewer_pairs.max_rounds == 10`.
2. All three loops (A/B/C) document a cap of 10; zero stale "cap 5" / "cap (5)" /
   "Round cap 5" / `count<5` / `count>=5` / `revision_round>=5` strings remain on an
   in-scope surface.
3. Every out-of-scope look-alike above is demonstrably untouched.
4. `tests/test_ensemble_review_cap.py` updated to pin 10 and to scan for stale `5`
   with the same allowlist discipline; full suite green.
5. `uv run python scripts/workflow_lint.py` clean.

## Note on cost

A higher cap raises the WORST-CASE review spend per site (up to 10 Claude + 10 Codex
rounds). It does not change the typical case — most sites PASS in 1-2 rounds — and the
alternative it replaces is a blocked task needing manual user release, which is more
expensive in wall-clock. No dollar caps (`tests/test_no_dollar_budget_caps.py`).

## Provenance

- workflow_fix_target: `.claude/workflow.yaml` (`ensemble_review.round_cap_per_reviewer`,
  `reviewer_pairs.max_rounds`) + the coupled cap-5 surfaces in `.claude/skills/issue/SKILL.md`,
  `.claude/skills/issue-v2/SKILL.md`, `.claude/skills/adversarial-planner-v2/SKILL.md`,
  `.claude/rules/codex-ensemble-review.md`, `.claude/rules/agents-vs-skills.md`,
  the reviewer agent specs, `CLAUDE.md`, and `tests/test_ensemble_review_cap.py`

- User instruction, 2026-08-19 interactive chat: "2147 -> approve and change workflow so
  we can have up to 10 reviewer rounds"; scope confirmed to all three loops via
  follow-up question the same session.
- Precedent: #784 raised this same cap 3 -> 5 and is the template for the coordinated
  multi-surface sweep (including the `pivot_criteria` trigger rename and the widened
  stale-number test scan).
- Related: #2147 (the cap-hit that motivated this; unblocked separately by explicit user
  grant and does NOT depend on this task landing).
