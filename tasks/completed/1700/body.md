---
title: 'daily-fix: verify_plan exempt-kind + commit-SHA check'
kind: infra
tags:
- wf-fix
- wf-fix-fp:8e634db1b54f
- daily-auto-filed
created_at: '2026-07-26T07:06:31Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-25 problem sweep (route 2): Five of six sessions in
  one wave had to hand-disposition an experiment-shaped plan-verifier WARN on a kind
  infra plan with one check re-firing across three plan versions, the #1689 plan carried
  four spurious or parser-limited WARNs needing prose justification, and two separate
  plans cited a commit SHA that did not resolve.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the `/daily` 2026-07-25 problem sweep. `verify_plan.py` produced a WARN
needing hand-disposition in 5 of 6 sessions in one wave, plus four more on #1689 — all
checker limitations rather than plan defects.

## Goal

Let the exempt-kind branch of `c8_success_kill_criteria` accept an acceptance-criteria
block instead of emitting a WARN, add an N/A escape to `c4_contrastive_negatives` for
measurement plans, teach `c20_verdict_lattice_coherence` to parse unicode inequality
atoms, fix `c18`'s N/A-vs-present-contrasts collision, and add a check that every hex
token the plan cites as a commit resolves.

## Workflow gap

1. **Experiment-shaped WARNs on `kind: infra` plans — 5 of 6 sessions.** Every plan
   version had to carry the WARN into the fact-checker and critic briefs and be
   explicitly dispositioned in the `epm:plan-verify` marker: #1667 (c11), #1668 (c8,
   re-firing on **v1, v2 and v3**), #1669 (c38), #1670 (c11), #1672 (c38 + c37). Only
   #1671 ran WARN-free. #1668's marker, posted identically three times: *"verdict:
   PASS; n_fail=0; n_warn=1 (c8_success_kill_criteria — infra-plan heading nit,
   acceptance criteria in §1)"*.
   **Premise correction (compose-time read).** `c8` is ALREADY kind-aware — its
   docstring reads *"`kind: experiment` FAILs on both-absent; exempt kinds WARN, and
   the exempt-kind missing-kill WARN detail carries the standard §0.0 remedy
   sentence."* So the WARN is by design, not a kind-blindness bug. The defect is
   narrower and real: for an exempt kind whose plan DOES carry acceptance criteria
   (just not under the experiment-style heading), the check has no way to be satisfied
   — it can only WARN, forever, on every version. The fix is to let an
   `## Acceptance criteria` / §1 acceptance block SATISFY the exempt-kind branch, not
   to make the check kind-aware (it already is).
2. **Four spurious/parser-limited WARNs on one #1689 plan.** From its `epm:plan-verify`
   v3 marker: `c4_contrastive_negatives` lacks an "N/A — not a behavior-implantation"
   escape for measurement experiments; `c18_paired_contrast_source_coverage` flagged
   an N/A that *"collides with actual paired contrasts — the N/A is spurious"*;
   `c8_success_kill_criteria` *"spurious"*; and
   `c20_verdict_lattice_coherence` — *"all 5 hypotheses now have explicit DISJOINT and
   exhaustive ⇔-partition form, but the parser doesn't recognize the unicode ≥/≤
   inequality atoms — parser limitation, not a plan defect."*
3. **No commit-SHA resolution check on plans.** #1683's plan v1 cited `7c7095f40e`;
   the real SHA was `7c8095f40e`, caught by the fact-checker and patched into v2 as
   "patch 1: SHA typo". #1689's plan carried analogous fact-checker corrections.
   `.claude/rules/workflow-fix-on-bug.md` clause (d) already mandates
   `git rev-parse --verify` for every hex-token-cited-as-commit **in a filing body**;
   there is no equivalent gate on **plans**, so the fact-checker burns a round instead.
- **Cost shape:** each WARN is cosmetic alone, but it rides into every downstream
  brief and must be re-dispositioned per plan version — #1668 paid it three times for
  one check. The SHA item costs a fact-checker round outright.
- **Confidence (emitter):** high on (1) and (3); medium on (2), where the four WARNs
  are quoted from the session's own disposition prose rather than independently
  re-derived.
- verified-at-filing: per-target read of `scripts/verify_plan.py` —
  `grep -n 'c8_success_kill_criteria'` → line **1026**, and the surrounding docstring
  (lines ~1020–1026) read in context per clause (c), which is what refuted the
  "not kind-aware" premise and produced the corrected framing above.
  `grep -c 'kind.*infra' scripts/verify_plan.py` → **33** (kind-awareness is pervasive,
  confirming the check family already branches on kind). SHA claim verified per clause
  (d): `git rev-parse --verify --quiet '7c8095f40e^{commit}'` resolves,
  `'7c7095f40e^{commit}'` does NOT — exactly the typo the fact-checker caught.
  Landed-fix history check `git log --oneline --since='7 days ago' --
  scripts/verify_plan.py` → no commits in the window. (2026-07-25)

## Proposed change (refine in planning)

```
+ c8 (exempt-kind branch): accept an `## Acceptance criteria` / §1 acceptance block
+   as SATISFYING the check for kind: infra|batch|survey — stop WARNing when the
+   criteria are present under a non-experiment heading.
+ c4_contrastive_negatives: add an "N/A — not a behavior-implantation experiment"
+   escape (measurement/geometry plans have no positives to contrast).
+ c18_paired_contrast_source_coverage: do not flag a declared N/A when paired
+   contrasts are in fact present — resolve the collision.
+ c20_verdict_lattice_coherence: parse unicode >= / <= inequality atoms.
+ new check: every 8-40-char hex token the plan cites AS A COMMIT resolves under
+   `git rev-parse --verify --quiet '<sha>^{commit}'`; FAIL on a non-resolving one
+   (mirrors workflow-fix-on-bug.md clause (d) on the filing side).
```

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`.
- `tests/test_verify_plan.py` — each changed check needs a both-directions pin (the
  satisfying shape passes; the genuinely-missing shape still WARNs/FAILs).
- The new SHA check must only fire on tokens the plan presents AS commits — a bare hex
  string (an HF revision, a fingerprint, a transcript basename) must not FAIL. Reuse
  the clause-(d) framing: cite-as-commit context, not "looks like hex".

## Constraints / invariants

- Do NOT reduce a FAIL to a WARN anywhere: (1) narrows a WARN's firing condition,
  (2) and (3) fix parser correctness and add one new check. Loosening a real gate is
  out of scope.
- The `kind: experiment` behaviour of c8 is unchanged (still FAILs on both-absent).
- `scripts/workflow_lint.py --check-references` / `--check-asks` pass; ruff passes;
  `tests/test_verify_plan.py` green.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- sha-verify (filing-time, #1467): `7c7095f40e` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: 8e634db1b54f
- Source: `/daily` 2026-07-25 transcript sweep, sessions `203baf55` (#1667),
  `a05fcbcf` (#1668), `7457e1a3` (#1669), `ad35514c` (#1670), `188282d2` (#1672),
  `ea7470c1` (#1683), `5c5a89e8` (#1689).
