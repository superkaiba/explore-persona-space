---
title: 'workflow-fix: v4 word-cap round-crediting misses plural/in-flight follow-up
  rounds'
kind: infra
tags:
- wf-fix
- wf-fix-fp:289c738fe302
created_at: '2026-07-16T00:17:19Z'
has_clean_result: false
origin_prompt: 'clean-result-critic formal candidate block, task #1332 post-fold re-gate
  2026-07-15 (see body Provenance for verbatim block)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a formal workflow-fix candidate raised on task #1332 (emitting agent: clean-result-critic, post-fold re-gate round 2).

## Goal

Fix check_v4_word_caps' follow-up round crediting: widen the footer clause regex to also match the plural-enumeration form ("Two same-issue follow-up rounds ...: (1) ...; (2) ..."), and have the events leg additionally count `epm:free-analysis-followup-run` markers plus an armed-but-unclosed `epm:followup-scope` (in-flight round) as one round.

## Workflow gap

- **Bug observed:** check_v4_word_caps' total-prose budget credited 0 extra follow-up rounds ("budget 800 ... [none]") on task #1332, a 3-round body, at the post-fold re-gate — the footer leg's `_V4_FOOTER_ROUND_CLAUSE_RE` requires the singular phrase "same-issue follow-up round" and its `(?!s)` lookahead excludes the analyzer's natural enumeration form "Two same-issue follow-up rounds ... (1) ... (2) ...", and the events leg counts only `epm:same-issue-followup-run` markers, which (a) excludes `epm:free-analysis-followup-run` rounds and (b) don't exist yet for an in-flight round (the marker posts when the loop closes, AFTER the clean-result gate runs).
- **Why it is a workflow gap:** the documented budget formula (clean-result-critic lens reference Lens 12 check 4: 800 + 250 per live follow-up round beyond the first) is systematically under-credited at exactly the moment the gate consumes the WARN — every multi-round v4 body re-gated mid-round gets a wrong budget in the WARN message.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -rln '_V4_FOOTER_ROUND_CLAUSE_RE' scripts/ tests/ src/` → 1 source hit, in the named target scripts/verify_task_body.py (2 in-file occurrences; the only other match is its .pyc byproduct) (2026-07-16)

## Proposed change (candidate diff sketch — refine in planning)

```
- _V4_FOOTER_ROUND_CLAUSE_RE = re.compile(r"same-issue follow-up round(?!s)...")
+ # also match "Two same-issue follow-up rounds ...: (1) ...; (2) ..." enumerations
+ _V4_FOOTER_ROUND_ENUM_RE = re.compile(r"(two|three|four|\d+)\s+same-issue follow-up rounds", re.I)
+ # events leg: count FOLLOWUP_RUN_KIND + FREE_ANALYSIS_RUN_KIND, plus one for an
+ # epm:followup-scope newer than the latest matching run marker (in-flight round)
```

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Grep the workflow surface for the pattern before editing (`grep -rln '_V4_FOOTER_ROUND_CLAUSE_RE\|same-issue follow-up round' scripts/ .claude/ tests/`) and update every hit incl. the Lens 12 reference text if its wording pins the singular form; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; `tests/test_verify_task_body.py` extended to pin the plural-enumeration + in-flight-round crediting.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 289c738fe302

<!-- workflow-fix-candidate v1 -->
target_file: scripts/verify_task_body.py
bug_observed: check_v4_word_caps' total-prose budget credited 0 extra follow-up rounds ("budget 800 ... [none]") on task #1332, a 3-round body, at the post-fold re-gate — the footer leg's `_V4_FOOTER_ROUND_CLAUSE_RE` requires the singular phrase "same-issue follow-up round" and its `(?!s)` lookahead excludes the analyzer's natural enumeration form "Two same-issue follow-up rounds ... (1) ... (2) ...", and the events leg counts only `epm:same-issue-followup-run` markers, which (a) excludes `epm:free-analysis-followup-run` rounds and (b) don't exist yet for an in-flight round (the marker posts when the loop closes, AFTER the clean-result gate runs).
why_workflow_gap: the documented budget formula (lens reference Lens 12 check 4: 800 + 250 per live follow-up round beyond the first) is systematically under-credited at exactly the moment the gate consumes the WARN — every multi-round v4 body re-gated mid-round gets a wrong budget in the WARN message.
proposed_change: widen the footer clause regex to also match the plural-enumeration form (count the enumerated "(1) ... (2) ..." items following "N same-issue follow-up rounds" or count labeled round clauses regardless of the plural), and have the events leg additionally count `epm:free-analysis-followup-run` markers plus an armed-but-unclosed `epm:followup-scope` (in-flight round) as one round.
confidence: medium
related_task: #1332
<!-- /workflow-fix-candidate -->
