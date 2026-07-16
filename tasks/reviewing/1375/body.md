---
title: 'workflow-fix: verify_plan.py check for false numeric containment claims'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ff49ea6ff46e
created_at: '2026-07-16T01:25:52Z'
has_clean_result: false
origin_prompt: 'statistics-critic + Alternatives critic (task #1315 amendment round):
  plan asserted 0.724 inside 0.737-0.820 spread (false); mechanizable: yes — verify_plan.py
  should assert numeric containment claims arithmetically'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1315 (emitting agents: statistics-critic + critic/Alternatives
lens, both flagged `mechanizable: yes` independently in the same round).

## Goal

Add a verify_plan.py check that a plan's explicit numeric containment claims
("N inside/within the A–B spread/band/range") are arithmetically true —
FAIL (or WARN) when N ∉ [A, B].

## Workflow gap

- **Bug observed:** plan v4 for task #1315 asserted "Tier-2 0.724 — … inside
  the siblings' realized 0.737–0.820 spread", which is arithmetically false
  (0.724 < 0.737); the verify_plan.py mechanical pre-pass returned PASS
  n_fail=0 n_warn=0 and two independent critics caught the false numeric by
  hand (Statistics lens finding 1; Alternatives lens finding 2).
- **Why it is a workflow gap:** verify_plan.py mechanically checks many prose
  contracts but has NO check that explicit numeric containment prose is
  arithmetically consistent, so a false "inside the X–Y spread" claim rides
  into a registered plan and can only be caught by an LM reviewer.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -niE 'spread|containment|inside the' scripts/verify_plan.py` → 1 hit (line 3430, an unrelated comment about paren placement inside quotes); this is an ABSENCE-of-guard claim, so the 0 relevant in-target hits ARE the evidence (2026-07-16)

## Proposed change (candidate diff sketch — refine in planning)

```
+ def c36_numeric_containment_claims(text):
+     # match: "<num> ... (inside|within) ... <a>-<b> (spread|band|range)"
+     # tolerate en-dash/hyphen ranges, "the registered"/"the realized" fillers
+     for m in re.finditer(NUM_CONTAINMENT_RE, text):
+         n, lo, hi = float(m['n']), float(m['lo']), float(m['hi'])
+         if lo > hi: lo, hi = hi, lo
+         if not (lo <= n <= hi):
+             fail(f"numeric containment claim false: {n} not in [{lo}, {hi}]")
```

Design notes for the planner: keep the matcher conservative (only fire when a
single number, a containment verb, and an explicit numeric range co-occur in
one sentence) to avoid false positives on ranges that describe something other
than the claimed value; a WARN tier may be safer than FAIL for v1. Add the
canonical N/A escape phrase if a legitimate non-containment reading needs an
exemption, and pin with tests in tests/test_verify_plan.py.

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Secondary: `tests/test_verify_plan.py` (pin the new check + the false-claim
  fixture from #1315 plan v4), `.claude/skills/adversarial-planner/SKILL.md`
  (add the new check id + N/A escape phrase to the canonical list) if a new
  escape phrase is introduced.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its
  own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: ff49ea6ff46e

Surfaced prose (statistics-critic, task #1315 amendment round, 2026-07-16):
"§4.1 (plan line 66) claims Tier-2 0.724 is 'inside the siblings' realized
0.737–0.820 spread' — it is not: 0.724 < 0.737. […] mechanizable: yes — assert
min(sibling spread) ≤ claimed-inside value ≤ max whenever a plan asserts
'inside … spread' with explicit numerics (verify_plan.py c27-family check)."
Independently (critic, Alternatives lens, same round): "mechanizable: yes —
assert committed 0.724 ∉ [0.737, 0.820] against the plan's prose claim."
