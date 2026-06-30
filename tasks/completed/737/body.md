---
title: 'workflow-fix: artifact-reuse (h) check for train-input mixes'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e6a24e31f459
created_at: '2026-06-29T22:02:13Z'
has_clean_result: false
origin_prompt: 'Auto-filed by workflow-fix-on-bug protocol from #734 round-4 implementer''s
  <!-- workflow-fix-candidate v1 --> block targeting .claude/rules/artifact-reuse.md
  (fingerprint e6a24e31f459)'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `<!-- workflow-fix-candidate v1 -->` block raised on task #734 round 4 (emitting agent: experiment-implementer).

## Goal

Add an `(h)` reuse-fitness check to `.claude/rules/artifact-reuse.md` covering reused **training-input mixes / on-policy response caches** — verify each reused mix file resolves on HF via `list_repo_files` AND is fetchable on the target backend BEFORE plan approval; if not fetchable, the plan must include a self-contained regen phase or upload the mix first.

## Workflow gap

- **Bug observed:** A plan that reuses a parent experiment's TRAINING-INPUT MIX (a `train/*.jsonl` the parent built but never uploaded) passed planning + 3 review rounds, then crashed phase2 at the pre-train `assert data_path.exists()` on the git-clone-only GCP lane because the mix is on neither HF repo and the lane cannot stage it.
- **Why it is a workflow gap:** The (a)–(g) artifact-reuse fitness check verifies reused ADAPTERS resolve on HF (check (e)) but has NO check that a reused TRAINING-INPUT MIX / on-policy cache is fetchable on the TARGET BACKEND; the fact-checker verifies adapter paths but not train-mix paths.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```
## The checklist
+ - **(h)** for reused TRAINING-INPUT artifacts (train/*.jsonl mixes, on-policy
+   response caches): the file resolves on HF via `list_repo_files` AND is
+   fetchable on the TARGET backend — the git-clone-only GCP lane stages NO
+   VM-local `data/`, so a mix the parent built but never uploaded is unreachable
+   there. If not fetchable: the plan includes a self-contained regen phase
+   (reuse the parent's deterministic build blocks) OR uploads the mix first.
+   (incident #734 round-4: #664's seed-42 marker mix was on neither HF repo;
+   phase2 crashed at the pre-train assert on the GCP lane.)
```

Plus parallel updates:
- `.claude/agents/consistency-checker.md` — extend the artifact-reuse section to call out the (h) check (the consistency-checker is the agent that runs the reuse-fitness pass).
- `.claude/agents/planner.md` — §11 hyperparameter grounding already requires `Source:` per knob; add an analogous "Source:" line per REUSED INPUT-DATA artifact (train mix, on-policy cache, eval JSON) that names how the file is fetched on the target backend.
- `.claude/agents/critic.md` Methodology lens item 9 (artifact reuse) — extend to verify (h) alongside (a)-(g).

## Scope / surfaces

- Primary target: `.claude/rules/artifact-reuse.md`
- Sibling: `.claude/agents/consistency-checker.md`, `.claude/agents/planner.md`, `.claude/agents/critic.md`
- Grep the workflow surface for the pattern (`(a)-(g)`, `artifact-reuse`, `reuse-fitness`) before editing — there may be other call sites that enumerate the (a)-(g) checks (`code-reviewer.md`, etc.) that need to grow a (h) reference too.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/artifact-reuse.md
- fingerprint: e6a24e31f459

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/artifact-reuse.md
bug_observed: A plan that reuses a parent experiment's TRAINING-INPUT MIX (a train/*.jsonl the parent built but never uploaded) passed planning + 3 review rounds, then crashed phase2 at the pre-train `assert data_path.exists()` on the git-clone-only GCP lane because the mix is on neither HF repo and the lane cannot stage it.
why_workflow_gap: The (a)-(g) artifact-reuse fitness check verifies reused ADAPTERS resolve on HF (check (e)) but has NO check that a reused TRAINING-INPUT MIX / on-policy cache is fetchable on the TARGET BACKEND; the fact-checker verifies adapter paths but not train-mix paths.
proposed_change: Add an (h) check to artifact-reuse.md — "training-input mixes / on-policy caches: verify each reused mix file resolves on HF via list_repo_files AND is fetchable on the target backend (the git-clone-only GCP lane cannot stage a VM-local-only mix) BEFORE plan approval; if not fetchable, the plan must include a self-contained regen step or upload it" — and have the fact-checker verify reused train-mix paths the same way it verifies adapter paths.
diff_sketch: |
  ## The checklist
  + - **(h)** for reused TRAINING-INPUT artifacts (train/*.jsonl mixes, on-policy
  +   response caches): the file resolves on HF via `list_repo_files` AND is
  +   fetchable on the TARGET backend — the git-clone-only GCP lane stages NO
  +   VM-local `data/`, so a mix the parent built but never uploaded is unreachable
  +   there. If not fetchable: the plan includes a self-contained regen phase
  +   (reuse the parent's deterministic build blocks) OR uploads the mix first.
  +   (incident #734 round-4: #664's seed-42 marker mix was on neither HF repo;
  +   phase2 crashed at the pre-train assert on the GCP lane.)
confidence: medium
related_task: #734
<!-- /workflow-fix-candidate -->
