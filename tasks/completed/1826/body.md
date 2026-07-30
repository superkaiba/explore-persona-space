---
title: 'workflow-fix: widen check-28 H-code regex to catch H1c-form hypothesis tags'
kind: infra
tags:
- wf-fix
- wf-fix-fp:531b88a3a471
created_at: '2026-07-29T10:15:09Z'
has_clean_result: false
origin_prompt: 'clean-result-critic r1 on #1774, formal workflow-fix-candidate block
  (H-code regex misses H<digit><letter>)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `<!-- workflow-fix-candidate v1 -->`
block raised on task #1774 (emitting agent: clean-result-critic, round 1).

## Goal

Widen the H-code token regex in verify_task_body.py check 28 to match
H<digits><letter>? hypothesis-tag forms (e.g. `\bH\d+[a-z]?\b`) in figure sidecar text.

## Workflow gap

- **Bug observed:** The figure-text opaque-config-code check (check 28) PASSed task
  #1774's body although the jensen figure sidecar registered
  `title_left: "Jensen-gap direction concentration (H1c)"` — the H-code pattern
  `\bH\d\b` misses the H<digit><letter> form (`\b` cannot fire between the digit
  and the trailing letter, so `H1c` never matches).
- **Why it is a workflow gap:** The mechanical backstop for plan-internal hypothesis
  codes in figure text exists (check 28, `_HYPOTHESIS_CODE_RE`) but its token pattern
  is narrower than the tags plans actually emit (H1a/H1c/H4b...), so the exact class
  it was built to catch passes silently; the miss cost a clean-result-critic round on
  #1774 (r1 blocker 2: figure regenerated + body re-pinned).
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'H\\d' scripts/verify_task_body.py` → `_HYPOTHESIS_CODE_RE = re.compile(r"\bH\d\b")` at scripts/verify_task_body.py:7816, consumed by `_opaque_code_tokens` (:7834) in check 28 (:7794/:7964); semantic probe: `re.search(r"\bH\d\b", "concentration (H1c)")` → None (the H1c form passes the current regex), 1 target file, presence hit read in context — the hit IS the too-narrow pattern, not a landed fix (2026-07-29)

## Proposed change (candidate diff sketch — refine in planning)

```
- _HYPOTHESIS_CODE_RE = re.compile(r"\bH\d\b")
+ _HYPOTHESIS_CODE_RE = re.compile(r"\bH\d+[a-z]?\b")
+ # apply to sidecar title/axis/legend text as today; keep the existing
+ # standalone/parenthesized-token guard to bound false positives
```

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rn '_HYPOTHESIS_CODE_RE\|\\bH..d..b' scripts/ .claude/`) and update every hit;
  list them in the plan. Add a regression test (H1c-form token in a fixture sidecar
  must FAIL check 28; a legitimate "H100"-style hardware token must not false-fire —
  calibrate the `[a-z]?` bound + surrounding-token guard accordingly).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 531b88a3a471

<!-- workflow-fix-candidate v1 -->
target_file: scripts/verify_task_body.py
bug_observed: The figure-text opaque-config-code check (advertised classes "slug / @L-pin / H-code / slot-family tokens") PASSed task #1774's body although the jensen figure sidecar registers `"title_left":"Jensen-gap direction concentration (H1c)"` — the H-code pattern misses the `H<digit><letter>` hypothesis-tag form.
why_workflow_gap: The mechanical backstop for plan-internal hypothesis codes in figure text exists but its token pattern is narrower than the tags plans actually emit (H1a/H1c/H4b...), so the class it was built to catch passes silently.
proposed_change: Widen the H-code token regex in the figure-text opaque-code check to match parenthesized/standalone `\bH\d+[a-z]?\b` in sidecar title/axis/legend text.
diff_sketch: |
  - H_CODE_RE = re.compile(r"\bH\d\b")   # (current form misses H1c)
  + H_CODE_RE = re.compile(r"\bH\d+[a-z]?\b")
  + # apply to sidecar .text.axes[*].title_left / suptitle as well as legend/axis labels,
  + # keeping the existing standalone/parenthesized-token guard to bound false positives
confidence: medium
related_task: #1774
<!-- /workflow-fix-candidate -->
