---
title: 'daily-held: decide ruff-debt burn-down (2149 errors)'
kind: infra
tags:
- daily-held
created_at: '2026-07-04T21:37:07Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-02 backfill route-3: bulk lint-debt scope/priority
  decision for Thomas'
workflow: v1
---
# daily-held: decide ruff-debt burn-down (2149 errors)

## Overview / Motivation

Filed by the /daily 2026-07-02 backfill problem sweep (route 3 — genuine
judgment call, needs Thomas). Held under the carve-out: bulk scope/priority
decision with large blast radius (NOT auto-dispatchable).

## The decision needed

`main` carries ~2,149 pre-existing repo-wide ruff errors (2,072 on 07-02,
growing). On 2026-07-02 alone, at least 7 sessions each burned a
verification round proving their diff didn't cause the red lint. Options:

1. **One-time burn-down task** — `ruff check --fix` + manual triage of the
   rest, as a dedicated `kind: infra` task. Cost: a huge diff that will
   conflict with ~15 live worktree branches; needs a quiet window.
2. **Baseline-and-freeze** — accept the debt, rely on the Step 9c
   known-red baseline ledger (filed separately) so sessions stop paying
   the re-derivation tax; ratchet: no NEW errors allowed.
3. **Do nothing** — status quo; every code session keeps paying the
   pre-existing-ness triage round.

Suggested: option 2 now (cheap, no conflicts), option 1 in a quiet window
after the current experiment wave lands.

## Decision (2026-07-17, task #1023 — autonomous under the 2026-07-17 PM greenlight)

- **Option 2 (baseline-and-freeze): RATIFIED as standing policy.** Mechanism =
  #1022's `scripts/step9c_baseline.py` ledger + live ratchet (landed 2026-07-17,
  before this task ran; the "filed separately" task in option 2's text). The
  Step 9c lint compare computes root-vs-worktree counts LIVE, so the ratchet
  survives baseline drops; only an INCREASE fails.
- **Option 1 (bulk burn-down): DEFERRED + DESCOPED.** Only 73/2,226 errors were
  safe-auto-fixable; 86% of the debt sat in frozen per-issue experiment scripts.
  After the config scoping below, the burn-down target is ~33 errors + 17
  format-dirty files (~19 safe auto-fixes). Filed as **#1486** (`on_hold`;
  revive trigger: <3 live issue sessions per `spawn_session.py list`, or
  2026-08-01, whichever first).
- **Option 3 (do nothing): REJECTED.**
- **Executed now:** pyproject ruff scoping (config-only; branch commit
  `dff56be0bc` on issue-1023, squash-merged to main at Step 10d — see the
  `epm:merged` marker for the main SHA): `extend-exclude` gains `eval_results` +
  `figures`; `[tool.ruff.lint.per-file-ignores]` turns off style/noise rules on
  `scripts/*`, `experiments/*`, `eps/experiments/*`. Repo-wide count
  **2,226 → 33** (measured pre-merge in the worktree AND by CLI simulation at
  the repo root; format count unchanged at 17). Real-bug rules stay ON
  everywhere (F undefined-names/imports, E9 syntax, core bugbear B); `src/` +
  `tests/` keep the full ruleset; 35 live workflow helpers pinned full-ruleset-
  clean by `tests/test_ruff_policy.py`.
- **E7 substitution (narrower than planned):** the wholesale `E7` ignore was
  replaced by specific `E731` + `E741` (the only E7xx codes in the frozen-path
  histogram) — **E722 bare-except stays enforced** at zero residual cost. Cost
  of the narrowing: future frozen scripts' other-E7xx hits become visible count
  growth (acceptable; they are real style regressions worth seeing).
- **Accepted real-bug-adjacent losses on scoped paths (named deliberately):**
  B905 (zip-strict), B023 (loop-var binding), F841 (unused variable) are off on
  `scripts/*` / `experiments/*` / `eps/experiments/*`. The code-review ensemble
  remains the gate on every new diff there.
- **Honest justification for the scoping** (not a "write-once" claim — 389/668
  frozen-pattern scripts were re-touched in ≥2 commits over 30 days): (a) style
  enforcement was vacuous on the debt-accretion channel — experiment-branch
  merges never run ruff, so the rules never gated where the debt grew; (b) the
  Step 9c touched-file absolute-clean rule made touching one line of a frozen
  script demand fixing all its style errors — friction with no bug-finding value.
- **Bulk `--add-noqa`: REJECTED** (~2k-line diff across hundreds of files; noqa
  debt compounds — 776 files already carry noqa whose RUF100 interaction added
  +3,278 phantom errors in simulation; RUF100 is kept in every ignore list to
  guard exactly that explosion).
- **Growth-channel merge-gate (ruff at experiment merges): NOT filed.** The
  scoping removes new frozen scripts' style noise from the count. Re-check
  trigger: if the visible count grows >+5/week sustained over 2 weeks (style
  noise via src/ counts too, not just real bugs), file the workflow-fix
  candidate for a merge-time ruff ratchet then. Escalation observer: the
  watcher's gate-push Telegram channel + /daily problem sweep.
- **Step 9c transition note (R1):** a worktree cut BEFORE this change lints
  with its own old pyproject, so its Step 9c lint compare reads
  `wt_ruff_count >> base_ruff_count` with touched files clean — the
  main-side-cleanup false-FAIL `step9c_baseline.py::lint_verdict` pre-registers
  as one-glance diagnosable (~8 active infra-kind tasks in scope at decision
  time; experiments never run ruff). Recovery: rebase the branch onto
  origin/main and re-run. Revert recipe if the wave misbehaves (>48h multiple
  sessions blocked): `git revert <merge-sha>` — one hunk, direction-safe (a
  post-change worktree at 33 still passes against a reverted 2,226 root).

## Provenance

- source: /daily 2026-07-02 backfill problem sweep (route 3, needs-human)
- carve-out: scope/priority judgment with large blast radius
- greenlight: PM directive 2026-07-17 (epm:progress v1) — autonomous background
  execution; decision executed via plan v2 (planner → fact-checker →
  3-lens critic ensemble APPROVE ×3 + consistency PASS; Codex twins
  sentinel-skipped #1204)
