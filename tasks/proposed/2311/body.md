---
title: 'Step 5a sibling arm: synced sibling tests run against branch-era src/; #2208
  collection probe cannot see the runtime skew'
kind: infra
tags: []
created_at: '2026-08-15T02:29:25Z'
has_clean_result: false
origin_prompt: 'Surfaced by the #2303 orchestrator: the Step 5a sibling arm synced
  tests/test_issue2225_steer_hook.py from current main into a worktree whose src/
  was fork-era, producing 2 NEW Step 9c failures (ValueError: unknown mask mode ''context_end'')
  that the #2208 import-satisfiability probe could not catch because the test collects
  cleanly and fails at runtime.'
workflow: v1
---
# Step 5a sibling arm: synced sibling TESTS run against branch-era `src/`, and the #2208 probe cannot see it (runtime failure, not a collection error)

`kind: infra`. Workflow-surface gap in `.claude/skills/issue/SKILL.md` § Step 5a, the #1972 sibling-issue file freshness arm. Reproduced live on #2303, 2026-08-14, where it cost a full ~1h26m Step 9c gate run plus a re-gate.

Distinct from #2308 (Step 9c 1b newline splice) and from #2302 (the sibling arm's #2024 `ordering_suspect` carve-out). Same arm as #2302, different failure mode.

## The defect

The sibling arm syncs a sibling issue's `scripts/issue<M>_*` and `tests/test_issue<M>_*` from `origin/main` into the current worktree, as a PAIR, so a gated test does not read a fork-era sibling script. `src/` is deliberately **not** in `SPECS` — but a sibling's tests routinely import from `src/explore_persona_space/experiments/issue<M>/`.

So the sync can pull a sibling's **current-main test** into a worktree whose `src/` is still at the branch's fork point. The test then exercises an API the worktree's module does not have yet, and fails — through no fault of the branch under review.

Concrete instance on #2303 (worktree forked ~2 h before the sync; the sibling's `context_end` work landed on main in between):

```
tests/test_issue2225_steer_hook.py:268
    mask = masks_for_mode("context_end", attention_mask=am, labels=labels)
src/explore_persona_space/experiments/issue2225/steer_train.py:81
    raise ValueError(f"unknown mask mode {mode!r}; expected one of {MASK_MODES}")
E   ValueError: unknown mask mode 'context_end'; expected one of ('all', 'context', 'response', 'prefix')
```

Two nodes failed this way. The Step 9c compare correctly classified both as **NEW**, so the gate blocked — on breakage the branch neither introduced nor touched.

## Why #2208 does not cover it

The #2208 import-satisfiability probe reverts a synced issue-`<M>` pair when the synced test **fails pytest collection** in this worktree (the #2206 branch-era-src-import-skew shape: `ImportError` at collection time). That is the right guard for a *missing symbol at import*.

This failure is one layer later: the module imports fine, the test collects fine, and the mismatch only surfaces when a collected test **calls** an API whose behavior changed. `masks_for_mode` exists in both revisions; only its accepted argument set differs. A collection-time probe is structurally incapable of catching an argument/behavior-level skew, so the pair is never reverted and the gate reds.

## Options (implementer picks; 1 and 2 are the cheap ones)

1. **Extend the probe from collect-only to a bounded run.** After the existing collection probe passes, run the synced sibling test file (short per-file `timeout`, `-q --tb=no`); on failure apply the SAME existing revert arm (restore branch-era content, or `git rm` a main-NEW file). Reuses #2208's machinery — only the probe's verdict source changes. Cost: the sibling test files' own runtime, on sync rather than at the gate. Guard against a sibling suite that is slow or needs GPU/network — cap it and treat a timeout as "leave synced, flag", never as a silent revert.
2. **Sync the sibling's `src/` subtree with the pair.** Add `src/explore_persona_space/experiments/issue<M>/` to what the sibling arm moves, making it a triple. Closer to the arm's own pair-atomic doctrine, but widens the blast radius into `src/` — which `SPECS` has deliberately excluded, so this needs an explicit argument before adoption.
3. **Classify it, don't prevent it.** Have the Step 9c compare recognize a failure whose test file arrived via a sibling-sync commit AND whose traceback terminates in `src/` as a distinct `sibling-sync-src-skew` class — reported, not counted NEW. Weakest option: the gate wall is still spent and a real regression could hide behind the label.

Whatever is chosen, the fallback the #2303 session used by hand should be documented in the arm's prose: **bring the branch current** (merge `origin/main` into the issue branch — not a rebase, which would need a force-push) so `src/` and the synced sibling tests agree. That is the root-cause fix; reverting the pair only hides it.

## Acceptance criteria

1. A worktree forked before a sibling's `src/` change, whose sibling tests are then synced from current main, does NOT produce NEW gate failures attributable solely to that skew.
2. Whatever guard is added fails loud or reverts explicitly — never silently drops a sibling pair, and never silently strips a genuine failure.
3. A pin test reproduces the shape: a synced sibling test that COLLECTS cleanly but fails at runtime against branch-era `src/`. `tests/test_issue_skill_lint_family_sync.py` already hosts the sibling-arm `_run_git` harness and the #2206 repro, and is the natural home.
4. No change to the family arm (#2303's surface), the on-main skip, the #1972 selection logic, or the explicit-pathspec commit discipline.

## Provenance

Surfaced by the #2303 orchestrator, 2026-08-14. Gate round 1: `3 failed, 8112 passed, 12 skipped` in 5130.41 s; compare `2 NEW, 1 stripped`. Resolved for that branch by merging `origin/main` (commit `642a792d39`), after which the file passed 14/14 — but nothing in the repo was changed, so the arm still carries this hole.
