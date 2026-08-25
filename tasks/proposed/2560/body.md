---
title: 'Step 10d lint gate normalization keeps workflow_lint note: lines whose scan
  COUNT changes with the payload, so any branch adding a scanned file gets a spurious
  block'
kind: infra
tags: []
created_at: '2026-08-24T22:07:32Z'
has_clean_result: false
parent_id: 2327
workflow: v1
---
---
kind: infra
tags:
  - workflow-fix
---

# Step 10d lint gate: normalization keeps `workflow_lint: note:` lines whose embedded scan COUNT changes with the payload, so any branch that adds a scanned file gets a spurious `block`

## Goal

Make the Step 10d pre-push lint gate's baseline-vs-gated normalization drop `workflow_lint: note:` informational lines — or strip their volatile counters — so a payload that adds files to a check's scan set cannot manufacture a NEW-set entry and a false `block`. The gate's own stated contract is that `NEW = gated − baseline` is payload-CAUSED by construction; a line reporting *how many files were scanned* carries no finding and violates that contract by construction.

## The defect

`.claude/skills/issue/steps/18-step-10d.md` normalizes each leg's lint output with:

```bash
grep -h '^workflow_lint: ' "…-$leg.txt" \
  | grep -vE '^workflow_lint: (PASS$|FAIL \()' \
  | sed -E 's/:[0-9]+:/::/g' | sort -u
```

The exclusion covers `PASS` and `FAIL (N error(s))`. The recipe's own comment says why: *"their COUNT changes even when the failure identities match — a payload that fixes one pre-existing error must not false-block on a differing summary."* That reasoning applies verbatim to `note:` lines, which also embed counts, but they are not excluded — they start with `workflow_lint: ` and are neither `PASS` nor `FAIL (`, so they survive into the failure-line set and into `comm -23` NEW.

The gate ALREADY treats `note:` as non-attributable in the sibling arm (same file, own-diff attribution): *"A line whose leading token is not a path (a check name, a `note:`) never attributes here."* So `note:` lines are excluded from the attribution arm but retained in the NEW-set arm — and the NEW-set arm is the one that blocks. That asymmetry is the bug.

Block requires `GATED_RC != 0 AND (owndiff non-empty OR NEW non-empty)`. Pre-existing main red satisfies the first conjunct on essentially every branch, so a single spurious NEW line is sufficient to block.

## Observed instance (#2327 gate run 4, 2026-08-24, tip `f361c9724c8d`)

The gate ran in its fully clean configuration — `lint-vintage 3-way merge: yes`, `landing-union overlay: merged=0 fallback=0`, correct tree, 15-file own-diff — and still returned `block`. The COMPLETE normalized diff between the two legs:

```
4c4
< workflow_lint: note: --check-no-unannotated-gcp-pin-guidance scanned 1276 file(s), 0 WARN(s)
---
> workflow_lint: note: --check-no-unannotated-gcp-pin-guidance scanned 1278 file(s), 0 WARN(s)
```

`0 WARN(s)` on BOTH sides. Supporting evidence, all from the gate's own artifacts:

| Signal | Value |
|---|---|
| normalized lines, baseline / gated | 5 / 5 |
| `lint-owndiff.txt` (payload-attributed failures) | EMPTY |
| `tg-new.txt` / `tg-new-nodes.txt` (new test failures) | EMPTY / EMPTY |
| `FAIL` summary, both legs | `FAIL (1 error(s))` — identical |
| the one real error, both legs | `scripts/issue823_ladder_ext_gen.py:: process-shared atomic-write temp name (#2336)` — not in the payload |

The count moved 1276 → 1278 because the payload adds two `.py` files (`scripts/step5a_coupling_check.py`, `tests/test_step5a_coupling_check.py`).

**Why this has not blocked the entire fleet:** the trip requires the payload to add a file inside that specific check's scan set. The sibling note on the same run — `note: --check-lane-order-adjective head='runpod', scanned 147 file(s), 0 finding(s)` — did NOT trip, because 147 was unchanged on both legs. `--check-no-unannotated-gcp-pin-guidance` scans ~1,278 files, so its blast radius is much wider: any branch adding a `.py` file in its scan set is exposed.

## Scope to investigate

1. **Drop `note:` lines from the normalization**, extending the existing exclusion: `grep -vE '^workflow_lint: (PASS$|FAIL \(|note:)'`. The recipe already declares them non-attributable, so this aligns the two arms. Confirm no check reports a genuine FINDING via a `note:` prefix — if one does, strip the counters instead (`s/scanned [0-9]+ file\(s\)/scanned N file(s)/`) rather than dropping the line.
2. **Audit for other volatile-counter lines** that survive normalization: any `scanned N`, `N finding(s)`, `N WARN(s)`, elapsed times, or absolute paths not covered by the existing `:[0-9]+:` blanking.
3. **Consider a floor invariant**: a NEW set containing no line whose leading token is a path cannot be payload-attributed, so it should not block on its own. That is a structural guard covering this whole class rather than the one line.
4. Check whether the sibling Step 9c gate and the Step 9a-ter inline payload lint gate share the same normalization and therefore the same defect.

## Non-goals

Do not fix this by relaxing the `GATED_RC != 0` conjunct or by treating pre-existing red as clean — that conjunct is load-bearing. Do not suppress the `note:` output itself in `scripts/workflow_lint.py`; the scan-count telemetry is useful, and the defect is in the gate's comparison, not in the linter. Do not add a per-branch allowlist for specific NEW lines — that converts a general normalization bug into per-branch maintenance and would let a real finding be allowlisted by accident.

## Provenance

Diagnosed by the #2327 orchestrator after its Step 10d gate returned `block` on a payload the same run proved introduces zero new lint errors and zero new test failures. Full evidence in that task's `epm:failure` marker and the gate artifacts under `/tmp/issue-2327-lint-*`. Confidence: high — the complete normalized diff is one line and both sides read `0 WARN(s)`. Dedup target: `.claude/skills/issue/steps/18-step-10d.md` normalization block (the `for leg in baseline gated` loop), distinct from #2539 (outer-fence derivation) and from the `WT`-guard gap filed alongside this.
