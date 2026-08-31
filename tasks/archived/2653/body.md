---
title: 'workflow-fix: Step 10d lint gate blocks on informational note-line scan counts
  (false block on clean payload)'
kind: infra
tags:
- wf-fix
created_at: '2026-08-31T01:06:33Z'
has_clean_result: false
parent_id: 2384
origin_prompt: 'Emitted as a workflow-fix-candidate v1 by the Step 10d subagent during
  /issue 2384: the pre-push workflow-lint gate normalizer admits count-bearing ''workflow_lint:
  note:'' status lines, so a payload that adds a file to a note-emitting check''s
  scan cone produces a false NEW line and, on a red baseline, a mechanical ''block''
  verdict. #2384''s PASS+PASS payload was parked by exactly this.'
workflow: v1
---
## Goal

Stop the `/issue` Step 10d pre-push workflow-lint gate from writing `block` on a payload that introduces no lint errors and no test failures, when the only NEW normalized line is an informational `workflow_lint: note:` status line whose embedded file count moved because the payload legitimately added a file to a check's scan cone.

## The defect

The gate's "Normalize failure lines" step — in BOTH executable blocks (the form (i)/(ii) workload and the form (iii) surgical block) of `.claude/skills/issue/steps/18-step-10d.md` — keeps every `^workflow_lint: ` line except those matching `PASS$` or `FAIL (`. That admits count-bearing informational status lines, for example:

```
workflow_lint: note: --check-no-unannotated-gcp-pin-guidance scanned 1402 file(s), 0 WARN(s)
```

Those counts are payload-sensitive by construction: any payload that adds a file to a note-emitting check's scan cone shifts the number, so the baseline-vs-gated diff shows a NEW line that describes scan coverage rather than any defect. On a tree whose baseline is already lint-red for unrelated reasons (`GATED_RC != 0`), the verdict then mechanically writes `block`.

This is the same count-instability class the normalizer's own FAIL-summary drop rationale already identifies and handles for `FAIL (N)` lines. The `note:` lines were missed.

At least two checks emit such lines: `--check-no-unannotated-gcp-pin-guidance` and `--check-lane-order-adjective` (`scanned N file(s)`).

## Incident of record

Task #2384 Step 10d, 2026-08-31. A fully-reviewed payload — Claude PASS + Codex PASS at round 5, concerns ledger empty, Step 9c selected set 9292 passed / 12 skipped / 0 failed — was gate-blocked solely by the gcp-pin note's count moving 1401 to 1402, because #2384 adds `scripts/check_cited_body_currency.py` to that check's cone.

Attribution from the gate's own artifacts:

- `lint-owndiff` EMPTY — no lint error is attributable to the payload.
- `tg-new*` EMPTY — no new test failure.
- Baseline and gated BOTH `FAIL (23)`, with 21 identical pre-existing error identities (bare-Hub-call / import-guard / splitlines offenders in `scripts/issue2643_*` and `scripts/issue779_ctxansviz_*`).
- The 3 failed test nodes are byte-identical to the merge-base baseline.
- Diff of the two normalized files is exactly the one `note:` line.

So every substantive signal the gate exists to read was clean, and the merge parked at `epm:merge-failed v1` regardless.

## Fix sketch

Either extend the drop regex in BOTH executable blocks:

```
grep -vE '^workflow_lint: (PASS$|FAIL \(|note: )'
```

or blank the `scanned [0-9]+ file\(s\)` count tokens the same way `:<line>:` numbers are already blanked. Blanking is the more conservative option: it keeps a genuinely new `note:` line visible if one appears for a reason other than a count shift.

Add a pinned fixture reproducing the false block: a red baseline plus a payload whose only normalized delta is a note-line count, asserting the verdict is not `block`.

## Scope note

This task fixes the NORMALIZER only. The 21 pre-existing lint errors on trunk are a separate condition and belong to their own owners; do not fold them in. They are relevant here only because a red baseline is what converts this false NEW line into a `block` rather than a harmless diff.

## Acceptance

1. A payload whose only normalized delta is a note-line count does not produce `block`, demonstrated by a committed fixture that fails before the change and passes after.
2. A genuinely new `note:` line that is not a pure count shift still surfaces (do not blanket-suppress the channel).
3. Both executable blocks changed — a fix to one leaves the other reachable.
4. #2384's merge re-runs and lands: after this fix is on `main`, re-invoke `/issue 2384`, whose Step 10d re-enters idempotently with branch and PR already prepared.
