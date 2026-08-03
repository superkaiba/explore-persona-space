---
title: 'workflow-fix: gitleaks-aware deferred-commit message in task_workflow'
kind: infra
tags:
- wf-fix
- wf-fix-fp:87f52b9f6272
created_at: '2026-07-29T01:28:33Z'
has_clean_result: false
origin_prompt: "<!-- workflow-fix-candidate v1 -->\ntarget_file: src/explore_persona_space/task_workflow.py\n\
  bug_observed: A `post-marker` deferred-commit whose failure is a gitleaks false\
  \ positive (benign backticked `key=value` token in the marker note, e.g. `min_occ_effective=20`)\
  \ can never self-sweep — every later commit touching the same events.jsonl re-fails\
  \ the hook — but the deferral message only says \"the next successful commit touching\
  \ the file sweeps it\", which is false for this failure class.\nwhy_workflow_gap:\
  \ The deferral message in `task_workflow.py` (near `DEFERRED_COMMITS_LOG`, the \"\
  Do NOT re-run the mutation\" composer) does not detect a gitleaks-finding stderr\
  \ and so omits the actual remediation (verify false positive, add the printed fingerprint\
  \ to `.gitleaksignore`, then sweep), leaving the wedge to be rediscovered per incident.\n\
  proposed_change: When the captured stderr_tail contains a gitleaks finding (match\
  \ \"gitleaks\" + \"Fingerprint:\"), extend the deferral message to state that plain\
  \ re-commits will keep failing and to name the `.gitleaksignore` fingerprint recipe.\n\
  diff_sketch: |\n  + if \"gitleaks\" in stderr_tail and \"Fingerprint:\" in stderr_tail:\n\
  \  +     msg += (\" NOTE: the commit failed on a gitleaks finding — the sweep will\
  \ \"\n  +             \"re-fail until the finding is resolved. If it is a false\
  \ positive \"\n  +             \"(benign config token), append the printed 'Fingerprint:'\
  \ line to \"\n  +             \".gitleaksignore and commit it together with the\
  \ swept paths.\")\nconfidence: medium\nrelated_task: #1092\n<!-- /workflow-fix-candidate\
  \ -->"
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1092 (emitting agent: analyzer, `crossed-core-sae` revision round 2).

## Goal

When the captured stderr_tail contains a gitleaks finding (match "gitleaks" + "Fingerprint:"), extend the deferral message to state that plain re-commits will keep failing and to name the `.gitleaksignore` fingerprint recipe.

## Workflow gap

- **Bug observed:** A `post-marker` deferred-commit whose failure is a gitleaks false positive (benign backticked `key=value` token in the marker note, e.g. `min_occ_effective=20`) can never self-sweep — every later commit touching the same events.jsonl re-fails the hook — but the deferral message only says "the next successful commit touching the file sweeps it", which is false for this failure class.
- **Why it is a workflow gap:** The deferral message in `task_workflow.py` (near `DEFERRED_COMMITS_LOG`, the "Do NOT re-run the mutation" composer) does not detect a gitleaks-finding stderr and so omits the actual remediation (verify false positive, add the printed fingerprint to `.gitleaksignore`, then sweep), leaving the wedge to be rediscovered per incident.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -rn 'gitleaks' src/explore_persona_space/task_workflow.py scripts/task.py` → 0 hits in both targets (absence-of-guard claim — the 0-hit in-target result IS the evidence); semantic sibling probe `grep -rln 'gitleaks' scripts/ src/explore_persona_space/` → hits only in `scripts/hooks/gitleaks_scoped.sh`, `scripts/workflow_lint.py`, `scripts/select_step9c_tests.py` (none is the deferral-message composer); composer site confirmed present: `grep -n 'DEFERRED_COMMITS' src/explore_persona_space/task_workflow.py` → lines 738, 6804, 6816 (composer docstring at 6748–6749); landed-fix history `git log --oneline --since='7 days ago' -- src/explore_persona_space/task_workflow.py` → 1 unrelated commit (`4cf33aa922`, plan-header retitle) (2026-07-29)

## Incident evidence (this filing's trigger)

On 2026-07-29 the #1092 `epm:interpretation` v6 marker post deferred its commit on a gitleaks `generic-api-key` false positive (`tasks/followups_running/1092/events.jsonl:generic-api-key:777` — the benign token `min_occ_effective=20` in the note). The deferral message's stated recovery ("the next successful commit touching the file sweeps it") could not work — every sweep re-fails on the same finding. Actual recovery: verify false positive, append the printed fingerprint to `.gitleaksignore`, commit it together with the swept paths (landed as `be36d6dc6a`, confirmed on origin/main).

## Proposed change (candidate diff sketch — refine in planning)

```
+ if "gitleaks" in stderr_tail and "Fingerprint:" in stderr_tail:
+     msg += (" NOTE: the commit failed on a gitleaks finding — the sweep will "
+             "re-fail until the finding is resolved. If it is a false positive "
+             "(benign config token), append the printed 'Fingerprint:' line to "
+             ".gitleaksignore and commit it together with the swept paths.")
```

## Scope / surfaces

- Primary target: `src/explore_persona_space/task_workflow.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'sweeps the deferred' .claude/ CLAUDE.md scripts/ src/explore_persona_space/task_workflow.py`) and update every hit that restates the
  false "next successful commit sweeps it" recovery for this failure class; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/task_workflow.py
- fingerprint: 87f52b9f6272

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/task_workflow.py
bug_observed: A `post-marker` deferred-commit whose failure is a gitleaks false positive (benign backticked `key=value` token in the marker note, e.g. `min_occ_effective=20`) can never self-sweep — every later commit touching the same events.jsonl re-fails the hook — but the deferral message only says "the next successful commit touching the file sweeps it", which is false for this failure class.
why_workflow_gap: The deferral message in `task_workflow.py` (near `DEFERRED_COMMITS_LOG`, the "Do NOT re-run the mutation" composer) does not detect a gitleaks-finding stderr and so omits the actual remediation (verify false positive, add the printed fingerprint to `.gitleaksignore`, then sweep), leaving the wedge to be rediscovered per incident.
proposed_change: When the captured stderr_tail contains a gitleaks finding (match "gitleaks" + "Fingerprint:"), extend the deferral message to state that plain re-commits will keep failing and to name the `.gitleaksignore` fingerprint recipe.
diff_sketch: |
  + if "gitleaks" in stderr_tail and "Fingerprint:" in stderr_tail:
  +     msg += (" NOTE: the commit failed on a gitleaks finding — the sweep will "
  +             "re-fail until the finding is resolved. If it is a false positive "
  +             "(benign config token), append the printed 'Fingerprint:' line to "
  +             ".gitleaksignore and commit it together with the swept paths.")
confidence: medium
related_task: #1092
<!-- /workflow-fix-candidate -->
