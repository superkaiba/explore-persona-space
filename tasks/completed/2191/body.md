---
title: 'verify_report.py: assert committed-under paths resolve at the pin, and assert
  code-SHA rows match per-phase reproducibility cards'
kind: infra
tags:
- verify-report-mechanical-asserts
created_at: '2026-08-08T02:19:29Z'
has_clean_result: false
origin_prompt: 'Surfaced by the methodology-critic during #2162''s round-1 report
  accuracy review (the first workflow v2 report in the repo). Both checks fired as
  MUST-FIXes: a draft claimed judge scores/items were committed under a path holding
  zero such blobs at the pin, and the code-SHA table pinned the stage-1 grid to the
  margin leg''s commit rather than the grid phase''s own reproducibility card. Both
  are pure asserts.'
workflow: v1
---
# `verify_report.py`: mechanize two report-accuracy checks that both fired as MUST-FIXes on the first v2 report

## Goal

Add two pure asserts to `scripts/verify_report.py`. Both caught real defects in
#2162's round-1 methodology-accuracy review, both are mechanical rather than
judgement calls, and both are the kind of error a human reviewer catches only by
manually running a git command per claim.

## Check (a) — every "committed under `<path>`" claim resolves at the pinned tree

A report that says an artifact set is committed under some path should be
falsifiable by `git ls-tree -r <pin> -- <path>` returning at least one blob.

**What it caught on #2162.** The draft asserted that judge scores and items were
committed under `eval_results/issue_2162/judge/`. At the pinned tree that
directory held gates (7), raw (70), audits (2), items **1 of 168**,
`judge_summary.json` and `pools.json` — and **zero** files under
`judge/scores/`, while 336 score files existed on disk untracked. The corpus
genuinely lives on the HF data repo under `raw_completions/judge_raw` per the
wave-output convention, so the claim named a git home the artifacts do not have.

Why it matters beyond tidiness: a reader who trusts that sentence will later
clone at the pin and find nothing, and will not know whether the data was lost
or merely lives elsewhere. The convention that wave outputs stay out of git is
correct — the report just has to say so.

## Check (b) — every Code-SHA row matches the per-phase reproducibility card

A multi-phase run has one reproducibility card per phase, each carrying its own
`git_commit`. A report's code-SHA table should agree with them phase by phase.

**What it caught on #2162.** The table pinned the stage-1 grid to the MARGIN
leg's commit (`ba3485b619…`) when the grid/anchors phase's own card records
`b4ab6ed5f9…`, six commits earlier. The reviewer then diffed the full constant
block of the run script between the two commits and found them byte-identical,
so no hyperparameter value was wrong — but the provenance pairing was, and only
an explicit comparison surfaced it. A future run where the constants DID change
between phases would turn the same mistake into wrong reported hyperparameters.

Sources to compare against: the per-phase `upload_done.json` reproducibility
cards, plus gate reports that carry a `repro` block.

## Suggested shape

- Both as asserts in `scripts/verify_report.py`, so they run in the existing
  `report-verifier` gate rather than as a separate tool.
- Check (a): parse the report for "committed under `<path>`"-shaped claims and
  their pin, then `git ls-tree -r <pin> -- <path>`. Fail loud naming the claim
  and the empty path. Be careful to accept a correctly-worded HF-home claim —
  the goal is catching a false GIT-home claim, not discouraging the convention.
- Check (b): collect `git_commit` from every reproducibility card under the
  issue's `eval_results/<issue>/**`, then assert each code-SHA row in the report
  matches the card for the phase it names. When a run legitimately spans
  multiple commits, the expected output is a per-phase split, not a single SHA —
  so the failure message should suggest that rather than implying one is wrong.

## Scope notes

- These are ADDITIVE asserts; do not weaken any existing `verify_report.py`
  check.
- #2162 is the FIRST `workflow: v2` task in the repo, so this is the first
  report the gate has ever run against — expect more mechanizable patterns to
  surface as later v2 reports land, and prefer a shape that is easy to extend.
- The reviewer that surfaced these also verified a large amount of the draft as
  clean, so the intent is not to distrust the methodology-writer — it is to move
  two specific classes of error from human re-derivation into a machine check.
- Confidence is high on the value of both checks (each has a concrete, confirmed
  instance) and moderate on the exact parsing shape, since report phrasing is
  free-form. The implementing session should look at #2162's actual sections at
  `tasks/*/2162/artifacts/issue-2162-report-sections.md` for realistic input
  before settling on a matcher, and should prefer a conservative matcher that
  under-fires over one that produces false FAILs on well-formed reports.
