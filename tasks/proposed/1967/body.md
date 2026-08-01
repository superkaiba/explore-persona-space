---
title: 'daily-fix: per-tier pilot walls + c12 screen vocabulary'
kind: infra
tags:
- wf-fix
- wf-fix-fp:8bc22471216c
- daily-auto-filed
created_at: '2026-08-01T07:06:01Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-31 problem sweep (route 2): Measured per-group walls
  transferred across a 2x token-budget tier (4/6 #1739 lanes rc=7 self-halted) with
  no tier-transfer clause; a near-dupe screen escaped verify_plan c12 (serial ~3.3
  h vs 1.0 h budget, #1901) because the trigger vocabulary covers draw batteries only
  (pilot-at-production-n + RAM keying halves already landed in #1798, scoped out)'
workflow: v1
---
# daily-fix: per-tier pilot walls + c12 screen vocabulary

## Overview / Motivation

Auto-filed by the /daily 2026-07-31 problem sweep (CONSOLIDATED H5; miner-3:P8, miner-1:P4/P3, miner-7:P1). Source sessions: 55419495 (#1739 — per-group fence walls measured on the evil behavior at its 8k token budget and proxied to 16k-budget behaviors: sycophancy pilot 302 s vs the assumed 140 s, so 4 of 6 lanes rc=7 self-halted and were relaunched with measured fences; round ~43-45 GPU-h vs ~30 named), 74476b0d (#1901 — a near-dupe screen shipped serial at 12,339 µs/row = ~3.3 h vs the 1.0 h sub-budget; died fail-loud and needed a mid-round 175× vectorize fix round that plan critics had passed), 3318f0b2 + 3a60e6ee (#1902 — pilot timed at n=354 vs production n=13,674, ~12× §9 deviation; adherence/vintage context below).

## Goal

Add a heterogeneous-tier wall-time re-pilot clause to plan-compute-sizing and extend verify_plan's c12 trigger vocabulary to dedupe/near-dupe screens over a large fixed pool.

## Workflow gap

- **Bug observed:** (1) A measured per-cell/per-group WALL transferred across cells whose token-budget tier differed 2× (#1739) — the fence sized off the wrong tier self-halted 4/6 healthy lanes; no clause forbids the transfer or requires a re-pilot per distinct tier. (2) A near-dupe SCREEN phase escaped verify_plan's mechanical c12 battery check entirely (#1901) — the trigger regex covers permutation/bootstrap/null-draw/resample vocabulary only, while the plan-compute-sizing RULE's pool-scale clause explicitly names "pairwise similarity / near-dupe screens ~ pool²" as the covered kernel class: the rule and the verifier vocabulary have diverged.
- **Why it is a workflow gap:** The 2026-07-29 #1798 landing (commit 6c75fcce7b) added production-shape/pool-scale pilot clauses and RAM largest-cell keying (the latter already citing the #1739 RAM incident) — so the CONSOLIDATED entry's pilot-at-production-n and RAM-row halves are ALREADY LANDED and are SCOPED OUT of this filing (the #1902 wrong-n pilot is plan-vintage/adherence against the landed clause). The two residuals above are verified absent: no tier-transfer clause anywhere in the rule, and no screen vocabulary in the c12 trigger.
- **Confidence (emitter):** medium-high (residuals grep-verified; the #1739 tier-transfer attribution is the session's own compute-deviation marker diagnosis)
- verified-at-filing: `grep -n -iE 'does not transfer|re-pilot|per (distinct )?tier|budget tier|group size|pilot-shape' .claude/rules/plan-compute-sizing.md` → 0 hits (absence = evidence for residual 1); `grep -n -iE 'screen|dedup|near-dup|pairwise|per-row' scripts/verify_plan.py` → hits only in unrelated contexts (L3034 precedence-phrase screen, L3040 quantifier screen, L1719 display-dedupe); `_BATTERY_TRIGGER_RE` + commitment regex read at L1305-1318: permutation/battery/null-draw/resample/bootstrap/`B=\d{3,}` tokens only, no screen/dedupe tokens (residual 2 confirmed). Landed-fix check: `git log --oneline --since='7 days ago' -- .claude/rules/plan-compute-sizing.md scripts/verify_plan.py` → 6c75fcce7b (2026-07-29, #1798) context read — implements pool-scale pilots + RAM keying (citing #1739's 163-GiB OOM), NOT the tier-transfer clause or the c12 vocabulary (2026-08-01).

## Proposed change (candidate diff sketch — refine in planning)

```
.claude/rules/plan-compute-sizing.md § Per-cell fit phases:
+ TIER TRANSFER — a measured per-cell/per-group wall does NOT transfer
+ across cells whose group/budget/context size differs (a 2x token
+ budget is a different shape): heterogeneous lanes re-pilot per
+ distinct tier, or state the scaling and extrapolate — the wall-time
+ sibling of the RAM largest-cell keying clause (#1739: fence walls
+ measured at the 8k-budget behavior proxied to 16k lanes; 4/6 lanes
+ rc=7 self-halted).
scripts/verify_plan.py (check 12):
+ extend _BATTERY_TRIGGER_RE (and the commitment vocabulary) with
+ dedupe/near-dupe/similarity-screen tokens over a large fixed pool
+ (e.g. r"near[- ]dup(e|licate)? screen", r"dedup(e|lication)? (pass|
+ screen)", r"pairwise similarity"), so a pool-quadratic screen phase
+ triggers the same batched-commitment check as a draw battery (#1901).
+ pin test in tests/test_verify_plan.py.
```

## Scope / surfaces

- Primary target: `.claude/rules/plan-compute-sizing.md`; secondary: `scripts/verify_plan.py` (+ `tests/test_verify_plan.py` pin).
- Grep the workflow surface for the pattern before editing (`grep -rn 'pool-scale\|largest-cell' .claude/rules/ scripts/verify_plan.py`) — keep the new clause adjacent + consistent with the landed #1798 blocks, and update the rule's frontmatter description line if the trigger summary changes.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run passes; ruff on verify_plan.py passes; c12's documented non-trigger carve-outs (bare "bootstrap CI") must not regress — calibrate the new tokens against the corpus like #1796 did for c39.
- Do NOT re-add the already-landed pilot-at-production-shape / pool-scale / RAM-keying clauses.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 8bc22471216c

- workflow_fix_target: .claude/rules/plan-compute-sizing.md
- fingerprint: (driver-computed; tag authoritative)
