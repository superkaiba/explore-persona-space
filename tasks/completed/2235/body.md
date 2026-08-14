---
title: Payload-scope the inline lint gate (9m08s measured -> target <60s for analysis-script
  payloads)
kind: infra
tags:
- inline-gate-scoping
created_at: '2026-08-11T14:41:21Z'
has_clean_result: false
origin_prompt: Do we have a lot of gates slowing down the workflow? Especially for
  hte back and forth plotting and analysis needed to write up results ... what is
  the point of these gates? ... but i think im okay with the plotting code being messy
  ... file it and make sure it starts
workflow: v1
---
# Payload-scope the inline lint gate: stop re-scanning 842K lines of per-issue analysis code to certify a 400-line plot script

## Goal

Make the Step 9a-ter inline payload lint gate answer the question it exists to
answer — *does THIS payload add new red?* — instead of re-deriving whole-repo
absolute state. Target: a figure/analysis-script payload certifies in **< 60 s**
instead of the currently-measured **9m08s**, with the #1388 fleet-integrity
property preserved exactly (no session may land new red that reddens another
session's Step 9c gate).

## Problem

All figures measured on `main` 2026-08-11 on the shared VM (32 cores).

**The gate cost:**

| Run | Wall | User |
|---|---|---|
| `workflow_lint.py` no-flags (the gate's heavy leg) | **9m07.9s** | 9m04.3s |
| 12 correctness-class checks only | 6m55.3s | 6m48.8s |
| 1 check (`--check-judge-model-pins`) | 20.1s | 19.0s |

`user ≈ real` ⇒ genuinely CPU-bound, not contention-starved.

**Check-selection is a dead lever.** 12 of ~40 checks cost 76% of the full
runtime, because the expensive checks are exactly the ones regex-scanning
`scripts/`. Dropping every hygiene check still leaves a ~7 min gate.

**What is being scanned:**

```
scripts/*.py    1,833 files   1,171,715 lines
  per-issue     1,378 (75%)     842,056 lines   <- write-once-per-round
  durable         455 (25%)     329,659 lines
```

So the gate re-proves 842K lines of prior rounds' analysis exhaust clean in
order to certify the ~400 lines the current round wrote. Cost scales with the
*history* of past inline rounds, not with the change — a ratchet. Commits
adding `scripts/issue*.py` ran 12 -> 187 -> 395/month May->Jul, 193 in the
first 11 days of Aug. This is also why `SKILL.md:7977`'s documented
"~2.5-6 min" is now stale by ~2x, which produces rc=124 INCONCLUSIVE verdicts
on self-set bounds and costs a full re-run each time.

**Fleet cost:** from the cert ledger `/tmp/eps-inline-lint-cert-v1.txt` —
500 cert rows / **281 distinct gate runs** over 10.3 days (1.8 paths per run),
peak 33 runs/day. At the measured 9.13 min that is **~42.8 CPU-hours, ~4.2
CPU-h/day**, inline gate alone. Four concurrent no-flags lints were observed
running simultaneously (load 16.87/32) computing the same global answer over
the same tree.

**The redundancy:** `inline_lint_gate.py` contains **zero references to the
baseline ledger** — it invokes `["uv","run","python","scripts/workflow_lint.py"]`
bare (`inline_lint_gate.py:580`), then re-implements payload attribution
(new-this-round / added-lines / conservative-block) in code and prose to
recover the answer a baseline diff gives for free. Step 9c, running the *same
instrument*, already does it correctly via `step9c_baseline.py compare`.

**The standard is already looser than the gate enforces.** Live
`.claude/cache/step9c-baseline.json` (refreshed 2026-08-10):
`dirty_code_paths: true` with 15 named `scripts/issue*` analysis/figure
scripts, `ruff_count: 46`, `ruff_format_files: 25`, `failing_tests: []`
(5,431 tests green). Of the 46 ruff violations, **2 are in a `scripts/issue*`
file** — the rest are in `src/`, `eps/experiments/`, `tests/`, and committed
task artifacts. The corpus being scanned is the cost, not the hazard.

## Scope decision (user, 2026-08-11)

Messiness in per-issue plotting/analysis scripts is **explicitly acceptable**
("i think im okay with the plotting code being messy"). It is already the de
facto fleet standard per the baseline above. This task does NOT need to keep
style/structure checks running over that corpus.

**But messy != wrong.** The correctness-class checks bite specifically on
analysis scripts and MUST still fire on the payload: `--check-judge-model-pins`
(a scoring script pinned to the wrong judge produces wrong numbers silently),
the `--check-upload-*` family (`or-true` / `return-discard` / `file-in-loop` /
`prefix-clobber` — swallowed upload failures = lost artifacts),
`--check-dotenv-before-hf-import`, `--check-jsonl-splitlines`,
`--check-batch-judge-client`.

## Proposed change

1. **Baseline-diff the inline gate.** `inline_lint_gate.py` consumes
   `.claude/cache/step9c-baseline.json` the way `step9c_baseline.py compare`
   does, so pre-existing red is subtracted rather than re-derived. This is the
   small, high-leverage change and can ship alone.
2. **Payload-scoping mode for `workflow_lint.py`.** Add a changed-paths mode
   (e.g. `--files <path-list>`); none of the current 86 flags scope input.
   Split the check registry **path-local vs global**; run the global set only
   when the payload touches `.claude/`, `workflow.yaml`, or `workflow_lint.py`
   itself.
3. **Reachable-import closure.** Per-issue scripts are NOT leaf code: **725**
   of them import another per-issue script by bare module name (e.g.
   `scripts/issue1005_f2f3.py:57` -> `from issue658_fit_predictors import ...`,
   plus `issue928_common` / `issue928_fit_decomposition` /
   `issue928_null_bootstrap`), and **1,250** import `explore_persona_space`.
   Scoping must include the payload's reachable imports, not just its own path.
4. **Correct or moot the stale timing figure** at `SKILL.md:7977`
   ("~2.5-6 min" -> measured), so self-set bounds stop returning rc=124.
5. Optional, if cheap: cache the global-check result on a repo-tree hash so
   concurrent sessions read one answer instead of recomputing it N times.

## Acceptance criteria

- [ ] A payload of only `scripts/issue<N>_*_fig*.py`-class files certifies in
      **< 60 s** measured on `main` (report the measured number).
- [ ] The #1388 property holds: a payload introducing NEW red still BLOCKS.
      Add a regression test with a deliberately-red payload.
- [ ] Pre-existing red anywhere outside the payload never blocks (the #1092 /
      #1388 attribution semantics are preserved or simplified, not weakened).
- [ ] Correctness checks listed under "Scope decision" still fire on payload
      paths; prove with a test payload that trips `--check-judge-model-pins`.
- [ ] Reachable per-issue imports are in scope (test with an `issue928_common`
      -style importer).
- [ ] Cert format + `guard_root_code_commit.sh` contract UNCHANGED — the hook
      must still validate certs written by the new path; no hand-written certs
      (#1082 parity).
- [ ] `EPM_INLINE_GATE_LINT_CMD` / `EPM_INLINE_GATE_MAP_CMD` override seams
      still work.
- [ ] Verdict semantics preserved: exit 0 PASS / 1 BLOCK / 3 INCONCLUSIVE,
      with the instrument-ran completeness evidence requirement intact (a dead
      leg is never push-clean).

## Non-goals

- Relocating per-issue scripts out of `scripts/` — they are a dependency web
  (725 cross-imports, 1,250 src imports) with a load-bearing reuse-discovery
  contract (#1739), not detachable artifacts.
- Cleaning up the 46 existing ruff violations / 25 unformatted files.
- Changing Step 9c or `step9c_baseline.py` semantics (read from it; do not
  alter it).
- Removing the gate. The #1388 interlock is sound: inline rounds land code at
  the shared root bypassing the `/issue` review pipeline, and the no-flags lint
  is also every session's Step 9c instrument, so one lint-red inline script
  reddens the whole fleet. Commit-time rather than push-time is likewise
  correct (#1460) because `auto_push_main.sh` publishes local main every ~2 min.

## Provenance

Diagnosed in an interactive chat session 2026-08-11 after a plotting round's
self-set 420 s bound on the gate returned rc=124 INCONCLUSIVE. All numbers in
this body were measured in that session on `main`, not estimated.
