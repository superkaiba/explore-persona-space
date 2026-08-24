---
title: 'workflow-fix: per-arm-resolution parser accept em-dash arm/res separator'
kind: infra
tags:
- wf-fix
- wf-fix-fp:8643a11b43e7
created_at: '2026-08-24T09:38:14Z'
has_clean_result: false
origin_prompt: "<!-- workflow-fix-candidate v1 -->\ntarget_file: src/explore_persona_space/task_workflow.py,\
  \ .claude/agents/experiment-implementer.md, .claude/rules/experiment-implementer-section-reference.md\n\
  bug_observed: Step 6d.0 PASS_AUTHORIZED_STUB grant REFUSED \"no per-arm-resolution:\
  \ sub-block found\" because per-arm rows used the em-dash arm/resolution separator\
  \ with unbackticked colon-containing arm names, which _PER_ARM_ROW_RE parses to\
  \ empty.\nwhy_workflow_gap: _PER_ARM_ROW_RE accepts colon-containing arm names only\
  \ when backticked, but the implementer spec's per-arm-resolution example shows the\
  \ unbackticked colon form and never warns that a <driver>:<phase> arm name needs\
  \ backticks; implementers of argparse-choices drivers naturally write the em-dash\
  \ form, which silently fails to parse and surfaces only at the Step 6d.0 grant.\n\
  proposed_change: Make _PER_ARM_ROW_RE accept the em-dash arm/resolution separator\
  \ (- <arm> — RES — detail) in addition to the colon form, so colon-containing <driver>:<phase>\
  \ arm names parse without backticks; document the backtick-colon requirement in\
  \ the implementer spec.\ndiff_sketch: |\n  - _PER_ARM_ROW_RE = re.compile(r\"^\\\
  s*-?\\s*([^:`*]+?|`[^`]+`)\\s*:\\s*(REAL|FALLBACK|N/A)\\b\")\n  + _PER_ARM_ROW_RE\
  \ = re.compile(r\"^\\s*-?\\s*(`[^`]+`|[^`*]+?)\\s*(?::|—|–|-)\\s*(REAL|FALLBACK|N/A)\\\
  b\")\nconfidence: medium\nrelated_task: #2502\n<!-- /workflow-fix-candidate -->"
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate the orchestrator raised while driving `/issue 2502` (autonomous). The Step 6d.0 `PASS_AUTHORIZED_STUB` mechanical grant REFUSED with "no `per-arm-resolution:` sub-block found" even though the marker carried a substantively-correct, line-anchored `per-arm-resolution:` sub-block with all 15 rows present.

## Goal

Make the authorized-stub / smoke-architecture per-arm-resolution row parser tolerant of the ` — ` (em-dash) arm/resolution separator that implementers of `<driver>:<phase>` argparse-choices drivers naturally write, and document the backtick-colon requirement in the implementer spec — so a grammar-only malformation of correct content is caught earlier (or parses) instead of surfacing only at Step 6d.0.

## Workflow gap

- **Bug observed:** `check-authorized-stub` REFUSED `PASS_AUTHORIZED_STUB` with "no `per-arm-resolution:` sub-block found" because the per-arm rows used ` — ` (em-dash) as the arm/resolution separator (`- gen_capture:gen — FALLBACK — <detail>`) with unbacktick­ed colon-containing arm names, and `_PER_ARM_ROW_RE` (`src/explore_persona_space/task_workflow.py:2586`) requires a `:` separator and — for arm names containing an internal colon — backticks; the em-dash rows parse `per_arm` to `{}`.
- **Why it is a workflow gap:** `_PER_ARM_ROW_RE` accepts a colon-containing arm name ONLY via its backticked alternative (`` `[^`]+` ``), but the implementer spec's per-arm-resolution example (`.claude/rules/experiment-implementer-section-reference.md:92`, `<arm-name>: REAL — <detail>`) shows the UNBACKTICKED colon form and never warns that a `<driver>:<phase>` arm name needs backticks. Implementers of the argparse-choices driver class (explicitly a #2502 pattern) naturally render `<driver>:<phase> — RES — detail`, which silently fails to parse. The failure is invisible on the `PASS_PARTIAL` path (that path never parses per-arm rows) and surfaces only at the Step 6d.0 `PASS_AUTHORIZED_STUB` grant — the worst place to discover it. This class has 9+ documented hits in `.claude/agent-memory/code-reviewer-lean/smoke_arch_marker_2176_grammar_pitfalls.md`.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -rn '_PER_ARM_ROW_RE' src/explore_persona_space/task_workflow.py` → 3 hits (def at :2586 + 2 usages at :2704 doc, :2748 call); `grep -rln 'per-arm-resolution' .claude/agents/experiment-implementer.md .claude/rules/experiment-implementer-section-reference.md` → both present (spec sites) (2026-08-24)

## Proposed change (candidate diff sketch — refine in planning)

```
- _PER_ARM_ROW_RE = re.compile(r"^\s*-?\s*([^:`*]+?|`[^`]+`)\s*:\s*(REAL|FALLBACK|N/A)\b")
+ # Accept BOTH the canonical colon separator AND the em-dash separator
+ # implementers of <driver>:<phase> argparse-choices drivers naturally write.
+ # Arm-name group: a backticked span (any chars incl. ':'), OR a bare run that
+ # may contain ':' when the separator is em-dash.
+ _PER_ARM_ROW_RE = re.compile(
+     r"^\s*-?\s*(`[^`]+`|[^`*]+?)\s*(?::|—|–|-)\s*(REAL|FALLBACK|N/A)\b"
+ )
```
Refine in planning: the bare-name alternative must not greedily swallow the resolution token; anchor the em-dash/colon separator so `REAL|FALLBACK|N/A` is the token immediately after it. Add a fixture reproducing the `- gen_capture:gen — FALLBACK — detail` shape (parses to arm `gen_capture:gen`) and keep the existing colon + backticked cases green. Mirror the clarification into `.claude/agents/experiment-implementer.md` + `.claude/rules/experiment-implementer-section-reference.md` (colon-containing arm names: either backtick under the colon form, or use the em-dash form).

## Scope / surfaces

- Primary target: `src/explore_persona_space/task_workflow.py` (`_PER_ARM_ROW_RE`, `parse_smoke_arch_marker`, `authorized_stub_grant`)
- Spec docs: `.claude/agents/experiment-implementer.md`, `.claude/rules/experiment-implementer-section-reference.md`
- Grep the surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- Existing colon-form and backticked-form per-arm rows MUST keep parsing (regression fixtures for both).
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; `tests/test_workflow_fix_dedup.py` + any `test_task_workflow*` stay green; if the spec docs change they stay consistent with the parser.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/task_workflow.py,.claude/agents/experiment-implementer.md,.claude/rules/experiment-implementer-section-reference.md
- fingerprint: 8643a11b43e7

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/task_workflow.py, .claude/agents/experiment-implementer.md, .claude/rules/experiment-implementer-section-reference.md
bug_observed: Step 6d.0 PASS_AUTHORIZED_STUB grant REFUSED "no per-arm-resolution: sub-block found" because per-arm rows used the em-dash arm/resolution separator with unbackticked colon-containing arm names, which _PER_ARM_ROW_RE parses to empty.
why_workflow_gap: _PER_ARM_ROW_RE accepts colon-containing arm names only when backticked, but the implementer spec's per-arm-resolution example shows the unbackticked colon form and never warns that a <driver>:<phase> arm name needs backticks; implementers of argparse-choices drivers naturally write the em-dash form, which silently fails to parse and surfaces only at the Step 6d.0 grant.
proposed_change: Make _PER_ARM_ROW_RE accept the em-dash arm/resolution separator (- <arm> — RES — detail) in addition to the colon form, so colon-containing <driver>:<phase> arm names parse without backticks; document the backtick-colon requirement in the implementer spec.
diff_sketch: |
  - _PER_ARM_ROW_RE = re.compile(r"^\s*-?\s*([^:`*]+?|`[^`]+`)\s*:\s*(REAL|FALLBACK|N/A)\b")
  + _PER_ARM_ROW_RE = re.compile(r"^\s*-?\s*(`[^`]+`|[^`*]+?)\s*(?::|—|–|-)\s*(REAL|FALLBACK|N/A)\b")
confidence: medium
related_task: #2502
<!-- /workflow-fix-candidate -->
