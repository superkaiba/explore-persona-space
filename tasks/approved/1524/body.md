---
title: 'workflow-fix: verify_uploads merged card — prose pointer must not shadow a
  valid adapter_paths list'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e3bc62d80033
created_at: '2026-07-18T22:59:25Z'
has_clean_result: false
origin_prompt: 'upload-verifier prose follow-up on #1489 upload-verification v2 (see
  body Provenance)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an upload-verifier prose follow-up raised on task #1489 (emitting agent: upload-verifier, upload-verification v2 round).

## Goal

Treat a non-dict/non-list `adapter_paths` declaration as undeclared in `merged_results_card`'s newest-wins merge (fall back to the older valid declaration), extending the #601 empty-card protection to prose pointers.

## Workflow gap

- **Bug observed:** epm:results v2's prose-pointer adapter_paths ("unchanged from epm:results v1...") shadowed v1's valid 64-path list in the newest-wins merge, producing a false `hf_model: MISSING` on a fully-uploaded ladder (#1489 upload-verification v2, 2026-07-18T22:56Z — the verifier had to supersede it with direct evidence, 64/64 dirs / 768 files resolving).
- **Why it is a workflow gap:** `scripts/verify_uploads.py::merged_results_card` protects against an EMPTY later declaration (`adapter_paths: {}`, the #601 guard at the docstring lines 45-48/556/570) but not a non-structural one — a prose string is truthy, wins the newest-wins merge, and downstream `check_hf_model_from_card` resolves nothing. Multi-marker tasks (crash-fix relaunches, K2-style final re-posts) routinely re-post results markers, so the shadow class recurs.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n "merged_results_card\|adapter_paths" scripts/verify_uploads.py` → 11 hits in the single named target incl. the #601 empty-dict guard (docstring lines 45-48, 556, 570) and the merge fn at line 563; no non-dict/non-list type guard present at the merge (presence-of-site + absence-of-guard both bind); `git log --oneline --since='7 days ago' -- scripts/verify_uploads.py` → 0 commits (no just-landed fix) (2026-07-18)

## Proposed change (candidate diff sketch — refine in planning)

```
  # merged_results_card, per-field newest-wins fold:
- if value in (None, {}, [], ""):   # #601: empty declaration is not a declaration
+ if value in (None, {}, [], ""):   # #601: empty declaration is not a declaration
+     continue
+ if field == "adapter_paths" and not isinstance(value, (dict, list)):
+     continue  # #1489: a prose pointer ("unchanged from v1 ...") is not a declaration;
+               # fall back to the older structural value
      continue
```
(Also consider tightening the producer guidance in `.claude/skills/issue/SKILL.md` Step 7 — the reproducibility_card structured-field paragraph — so results re-posts never substitute prose pointers for structured fields; planner's call whether to include that second surface.)

## Scope / surfaces

- Primary target: `scripts/verify_uploads.py`
- Secondary (planner's call): `.claude/skills/issue/SKILL.md` Step 7 reproducibility_card guidance.
- Grep the workflow surface for the pattern before editing (`grep -rln 'merged_results_card' .claude/ CLAUDE.md scripts/ src/explore_persona_space/`) and update every hit; add/extend the pinning test in `tests/` (the #601 test family covers the empty-dict case — add the prose-pointer case).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; existing #601 semantics preserved (empty-dict still falls back).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_uploads.py
- fingerprint: e3bc62d80033

Verbatim surfaced prose (upload-verifier, #1489 upload-verification v2 return): "scripts/verify_uploads.py's per-field newest-wins reproducibility-card merge lets a later marker's malformed/prose-pointer adapter_paths value shadow an earlier marker's valid list — the #601 empty-card protection covers {} but not a prose pointer; either treat a non-dict/non-list declaration like an undeclared field (fall back to the older valid declaration) or tighten the producer guidance in .claude/skills/issue/SKILL.md Step 7 so re-posts never use 'unchanged from v1' prose."
