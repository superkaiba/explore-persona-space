---
title: step5a_coupling_check._git_show reports clean over an undecidable state; non-repo
  test presumes TMPDIR outside a git repo
kind: infra
tags:
- workflow-fix
created_at: '2026-08-24T09:49:08Z'
has_clean_result: false
parent_id: 2327
origin_prompt: 'Two ledger NITs left open when #2327 advanced at round 3: git-show-rc-conflation-2327
  (codex-code-reviewer — _git_show maps every nonzero rc to absence, so a blob-read
  failure prints a false ''absent'' notice, skips Arm A, and ends in ''coupling check:
  clean'', violating the principle #2327 established for the agent-caps regime) and
  non-repo-test-tmpdir-enclosing-repo (code-reviewer — test presumes mkdtemp root
  is outside any git repo)'
workflow: v1
---
---
kind: infra
tags:
  - workflow-fix
parent_id: 2327
---

# `step5a_coupling_check._git_show` reports `clean` over an undecidable state, and the non-repo regression test presumes its TMPDIR is outside a git repo

## Goal

Close the two NITs left open when #2327 advanced at round 3, both in `scripts/step5a_coupling_check.py` and its test file. The load-bearing one restores a principle #2327 itself established: **the coupling detector must never print `clean` over a state it could not decide.** #2327 enforced that for the agent-caps regime; the lint-source regime still violates it.

## Item 1 — `_git_show` rc conflation ends in `clean` (the load-bearing one)

Raised by `codex-code-reviewer` as `git-show-rc-conflation-2327` (#2327 round 3, NIT, non-blocking). Traced end to end by the #2327 orchestrator:

1. `_git_show:148-153` returns `None` on **any** nonzero rc, while its docstring at `:149` says "or None when the path is absent at the ref". The docstring overclaims — it describes absence, the code catches absence *and* every failure.
2. `_load_side_main:303-305` maps that `None` to `_notice("<lint> absent at origin/main — cap-coherence arm skipped")` — a factually **false** absence claim on a read failure — and returns `None`.
3. A `_notice` is not a warn, so nothing is appended to the warn list.
4. `:655` sees zero warns and prints `[step5a] coupling check: clean`, return 0.

So a blob-read / object-store failure on `workflow_lint.py` at `origin/main` produces `clean` plus a wrong "absent" notice. This is the SAME defect class #2327 round 2 was convened over — a detector reporting `clean` over an undecidable state — and it leaves the two cap regimes internally inconsistent: the agent-caps regime now emits a non-clean `cap-source-invalid` WARN for exactly this shape, while the lint-source regime silently degrades to `clean`.

**Reachability (measured, not assumed).** `main():633` runs `git diff --name-only origin/main` before any `_git_show`, and `_git:137` raises on nonzero rc, so the repo and the ref are known-good by the time `_git_show` runs. Reaching this cell requires a corrupted object store mid-run on that specific blob. The consequence is fail-open — the operator loses the pre-diagnosis and hits the deterministic red they would have hit anyway — which is why #2327 advanced rather than opening a round 4. It is not a new harm, but it IS a silent-failure channel in a file whose entire value is being trustworthy about what it detects.

**Fix (Codex's sketch, and it is a good one):** use the `rev-parse --verify --quiet` existence grammar that `_sibling_vintage` now establishes in this same file (#2327 round 3) to settle presence first, then require `git show` itself to SUCCEED once presence is known — so a read failure on a present blob raises rather than masquerading as absence. Also correct the `:149` docstring. The `RuntimeError` → `main()` → rc 1 → advisory-unavailable-line path already exists from round 3; route into it.

**Why the round-3 reviewer's counter-argument does not cover this.** The Claude `code-reviewer` examined the same function and ruled the `_sibling_vintage`/`_git_show` asymmetry "forced and defensible", correctly probing that `git show` returns rc 128 for missing-path, missing-ref, and not-a-repo alike — no rc discrimination available — and characterizing `_git_show`'s `None` as degrading to "a skipped-arm notice or WARN, never a fabricated `fresh`". That is right about Arm B (no false `fresh`) and under-weights the Arm A path, where the outcome is `clean`. The fix above sidesteps the rc-discrimination problem entirely by asking a DIFFERENT question (existence) with a call that CAN answer it, then treating any `show` failure on a known-present path as a real failure. So "rc 128 is ambiguous" is true and no longer load-bearing.

Regression: probe or mock an rc-128 `git show` after a successful existence check; assert `main()` returns 1 and that `clean` is NOT printed.

## Item 2 — non-repo regression test presumes its TMPDIR is outside a git repo

Raised by `code-reviewer` as `non-repo-test-tmpdir-enclosing-repo` (#2327 round 3, NIT). `tests/test_step5a_coupling_check.py:599` presumes `mkdtemp()`'s root sits outside any git repository. With an in-repo `TMPDIR`, `git -C <tmpdir>` discovers the ENCLOSING repo, so the call returns rc 1 silent instead of rc 128 — the test's expected `RuntimeError` never fires and it fails as DID-NOT-RAISE. A spurious, environment-dependent failure whose cause is non-obvious from the failure text, which is the expensive part.

Fix: set `GIT_CEILING_DIRECTORIES` in the test's `_ENV`, or add an in-test not-a-repo premise assert so a wrong-environment run fails with a legible message instead of a confusing one.

## Non-goals

Do not weaken the round-3 rc narrowing in `_sibling_vintage` — it removed a fabrication channel (pre-fix, `rc ∉ {0,1}` was misclassified as absence and could fabricate a false `fresh`). Do not move the cap-source engagement short-circuit at `:366`: a caps file identically invalid on BOTH sides is main-side red rather than a half-sync, and warning there would fire on every healthy branch whenever main is red — that scoping was decided in #2327 round 3 and its rationale is recorded in the `check_cap_coherence` docstring. Do not change the advisory / fail-open posture at the Step 5a call site.

## Standing note (separate, deliberately not folded in)

`scripts/sync_repo_root.py:1068` uses the same unfiltered `git hash-object` form that #2327 round 3 fixed in the coupling checker. It is pre-existing trunk code with a DIFFERENT consumer where check-in (filtered) semantics may well be intended, so it needs its own judgement rather than a mechanical sweep. Recorded here only so the observation is not lost; it is not in this task's scope.

## Provenance

Both items are ledger concerns from #2327 round 3, left open when that task advanced to Step 9c with both review legs recommending merge and zero blockers. Full reasoning, including why advancing was correct there and what would make it wrong (this follow-up evaporating), is in #2327's round-3 `epm:code-review` marker.
