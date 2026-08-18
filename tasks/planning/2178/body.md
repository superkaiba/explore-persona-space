---
title: 'verify_plan.py: validate a plan''s stated smoke-fixture size against realized
  fixtures, and flag a smoke claim whose producing script is absent from the modified-file
  list'
kind: infra
tags: []
created_at: '2026-08-07T15:48:33Z'
has_clean_result: false
origin_prompt: 'Surfaced by all five #1336 plan v16 reviewers: the plan claimed >=40
  rows per smoke corpus; realized fixtures are 8 (six corpora) and 32 (one), SMOKE_SAMPLE_N=8,
  and the producing script appears zero times in the plan. The newly-binding smoke
  gate is unsatisfiable at that size. Complements the #2165 smoke blind-spot enumeration
  rule by verifying the numeric premises enumerations rest on.'
workflow: v1
---
## Goal

Add a mechanical check to `scripts/verify_plan.py` that validates a plan's
stated SMOKE-FIXTURE SIZE claim against the realized fixtures on disk at the
pinned tip, and flags a smoke-scale claim whose producing script is absent from
the plan's own modified-file list.

## Why

`.claude/rules/smoke-blind-spots.md` (created out of #1336 via workflow-fix
#2165) made plans ENUMERATE what a smoke PASS does not certify. #1336 plan v16
complied — it carried a correct `Smoke blind-spot enumeration:` block. But the
enumeration rested on a FALSE PREMISE that no mechanical gate checked:

> "the smoke fixture slice is sized >= 40 rows per smoke corpus so the
>  whole-group packer has resolution"  (v16.md:127)

Ground truth at the pinned tip `8c7b7b2406`, counted directly:

    gsm8k_test1319.jsonl     8
    gsm8k_train_full.jsonl   8
    if11k.jsonl              8
    lmsys23k.jsonl          32
    math7500.jsonl           8
    sft11k.jsonl             8
    uf11k.jsonl              8
    SMOKE_SAMPLE_N = 8   at scripts/issue1336_stage_corpora.py:142

The plan claimed >= 40; realized was 8 for six of seven corpora — a 5x
overstatement on the exact quantity its newly-BINDING smoke gate depended on.
At n=8 the plan's acceptance window [0.15, 0.28] is knife-edge (2 groups = 25%
passes; 1 group = 12.5% fails the floor); at n=7 it is arithmetically
UNSATISFIABLE (1/7 = 14.3% < 15; 2/7 = 28.6% > 28). So the first dispatch could
HALT on a gate the plan believed it had sized for.

Compounding it: `issue1336_stage_corpora` — the script that would have to change
to make the claim true — appears **zero** times in v16, while the plan's §4 "New
vs reused code" says "Nothing else changes". The plan was therefore either wrong
about its fixtures or wrong about its modified-file list, and no gate could tell.

This is the natural mechanical complement to the #2165 enumeration rule: #2165
made plans DISCLOSE smoke divergences; this check VERIFIES the numeric premises
those disclosures rest on. An enumeration built on an unchecked number inherits
the number's error.

## Proposed check

Add to `scripts/verify_plan.py`:

**Arm A — claimed vs realized fixture size.** When a plan states a per-corpus /
per-slice smoke fixture row floor (`>= N rows per smoke corpus`, `smoke slice of
N rows`, `SMOKE_SAMPLE_N = N`), resolve the fixture glob the plan names and
compare N against the realized `wc -l` of each committed fixture, and against
any `SMOKE_SAMPLE_N`-style constant found in the repo. FAIL when the claimed
floor exceeds the realized minimum.

**Arm B — modified-file-list coverage.** When Arm A's claim can only be
satisfied by changing a producing script (the file defining the sample-size
constant or generating the fixtures), and that file is absent from the plan's
declared new/modified-code list, WARN — the plan is asserting a state it has not
budgeted to create.

Arm A is FAIL-grade (it is a checkable factual claim about files in the repo);
Arm B is WARN-grade (modified-file lists are prose and the resolution is
heuristic).

## Acceptance criteria

1. Both arms registered with stable ids; standard `{id, name, status, detail}`
   JSON shape; Arm A FAIL, Arm B WARN.
2. Fixture reproducing the #1336 v16 shape (claim >= 40, realized 8/32,
   producing script absent from the modified list) trips BOTH arms.
3. A plan claiming a floor at or below the realized minimum passes Arm A.
4. A plan that DOES list the producing script in its modified-file list passes
   Arm B even when Arm A trips (it has budgeted the change).
5. SKIP cleanly — never FAIL — when the plan names no smoke fixture size, when
   the fixture path does not resolve, or when the plan declares no smoke run.
6. Regression sweep across `tasks/**/plans/*.md`; report the hit list so a noisy
   matcher is caught before landing.
7. `.claude/rules/smoke-blind-spots.md` gains a pointer to the new check ids,
   noting that the enumeration duty and this premise-verification are
   complementary.

## Provenance

Surfaced as a prose follow-up by the Statistics critic lens during the #1336
plan v16 Phase 2 review, 2026-08-07 ("mechanizable: yes — assert
`min(wc -l corpora_v2_smoke/*.jsonl) >= <plan-claimed floor>` at the pinned tip
whenever a plan claims a smoke fixture size; flag a smoke-scale claim whose
producing script is absent from the modified-file list"), and independently by
the Methodology and Alternatives lenses plus the Phase 1.5 fact-checker
(WRONG-1) and the consistency-checker (WARN-1) — five of five reviewers. Filed
by the #1336 orchestrator per `.claude/rules/workflow-fix-on-bug.md`.

Sibling: the §9 compute-row basis-vs-booked arithmetic check filed in the same
batch — a DISTINCT bug on the same target file (`scripts/verify_plan.py`), per
the `(target_file, candidate-fingerprint)` dedup rule.
