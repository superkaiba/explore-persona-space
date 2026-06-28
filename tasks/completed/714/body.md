---
title: 'workflow-fix: add --check-skill-refs lint mode to workflow_lint.py'
kind: infra
tags:
- wf-fix
- wf-fix-fp:cac1e15fecec
created_at: '2026-06-28T09:32:16Z'
has_clean_result: false
origin_prompt: 'Methodology critic finding on task #713 /adversarial-planner Phase
  2: workflow_lint.py --check-references does NOT catch stray /weekly-style skill
  references; only resolves workflow.yaml section refs. Surfaced as a workflow-fix-candidate
  v1 block.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #713 (emitting agent: Methodology critic during /adversarial-planner Phase 2).

## Goal

Add a `--check-skill-refs` mode to `scripts/workflow_lint.py` that scans
DOC_FILES for `/{skill-name}` tokens, resolves each against
`.claude/skills/*/` (and a declared deprecated/external allowlist), and FAILs
on an unresolved skill reference.

## Workflow gap

- **Bug observed:** A cadence-retirement / skill-rename change can leave
  stray load-bearing `/weekly`-style skill references in the workflow
  surface that no mechanical check catches. `workflow_lint.py
  --check-references` (workflow_lint.py:705-721) only resolves
  `(see workflow.yaml § X)` references against workflow.yaml keys — it has
  no notion of skill-name references like `/weekly`, `/daily`, `/issue`.
- **Why it is a workflow gap:** There is no lint that validates skill-name
  cross-references resolve to a live skill dir, so retiring or renaming a
  skill silently rots every dangling mention. The only guard today is a
  hand-run `grep` per change (per task #713's plan §12 and §7 Risk b).
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```
+ SKILL_REF_RE = re.compile(r"/([a-z][a-z0-9-]+(?::[a-z0-9-]+)?)\b")
+
+ def _check_skill_refs(workflow) -> list[str]:
+     live = {p.parent.name for p in Path(".claude/skills").glob("*/SKILL.md")}
+     allow = _load_skill_ref_allowlist()  # deprecated/external (e.g. blog citations)
+     errors = []
+     for path in DOC_FILES:
+         for lineno, line in enumerate(path.read_text().splitlines(), 1):
+             for m in SKILL_REF_RE.finditer(line):
+                 ref = m.group(1)
+                 if ref not in live and ref not in allow:
+                     errors.append(f"{path}:{lineno}: unresolved skill ref /{ref}")
+     return errors
+
+ # bundle into the default no-flags run + --check-references group
```

The planner should consider:
- How to handle plugin-namespaced skills (`/superpowers:using-superpowers`,
  `/codex:rescue`, `/code-review:code-review`).
- How to handle path-like tokens that look like skill refs but aren't
  (e.g., `/tmp/...`, `/workspace/...`, URL paths).
- Allowlist file format (a new `.claude/skill_ref_allowlist.txt`?) and
  what entries should be there at the start (deprecated skills like
  `/workflow-improver`, external blog citations in
  mentor-update-slides/principles.md).

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py`
- Possibly: `.claude/skill_ref_allowlist.txt` (new allowlist file).
- Tests: `tests/test_workflow_lint.py` — add coverage for the new check.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- The new check must pass on current `main` (i.e. no false positives on
  the existing workflow surface — allowlist any legitimate-but-dangling
  external references encountered).
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files
  passes; existing tests in `tests/test_workflow_lint.py` continue to pass.
- This session runs under a workflow-fix recursion guard — its
  Provenance line keeps the recursion guard intact.

## Provenance

- workflow_fix_target: scripts/workflow_lint.py
- fingerprint: cac1e15fecec

<!-- workflow-fix-candidate v1 -->
target_file: scripts/workflow_lint.py
bug_observed: A cadence-retirement / skill-rename change can leave stray load-bearing `/weekly`-style skill references in the workflow surface that no mechanical check catches (`--check-references` only resolves `(see workflow.yaml § X)` refs, not skill-name refs like `/weekly`).
why_workflow_gap: There is no lint that validates skill-name cross-references (`/weekly`, `/daily`, etc.) resolve to a live skill dir, so retiring or renaming a skill silently rots every dangling mention; the only guard today is a hand-run `grep` per change.
proposed_change: Add a `--check-skill-refs` mode to workflow_lint.py that scans DOC_FILES for `/{skill-name}` tokens, resolves each against `.claude/skills/*/` (and a declared deprecated/external allowlist), and FAILs on an unresolved skill reference.
diff_sketch: |
  + def _check_skill_refs(workflow) -> list[str]:
  +     live = {p.parent.name for p in Path(".claude/skills").glob("*/SKILL.md")}
  +     allow = _load_skill_ref_allowlist()  # deprecated/external (e.g. blog citations)
  +     for path in DOC_FILES:
  +         for lineno, line in enumerate(path.read_text().splitlines(), 1):
  +             for m in SKILL_REF_RE.finditer(line):     # r"/([a-z][a-z0-9-]+)"
  +                 if m.group(1) not in live and m.group(1) not in allow:
  +                     errors.append(f"{path}:{lineno}: unresolved skill ref /{m.group(1)}")
  + # bundle into --check-references default-on
confidence: medium
related_task: #713
<!-- /workflow-fix-candidate -->
