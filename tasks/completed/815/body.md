---
title: 'workflow-fix: forbid destructive git resets on the shared repo root (analyzer
  marker-chain recovery clobbered #812)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d6335a734640
created_at: '2026-07-01T20:49:40Z'
has_clean_result: false
origin_prompt: Analyzers (and any subagent) MUST NOT run `git reset --hard` on the
  shared repo root — it clobbers concurrent siblings' body.md/plans/registry. Add
  workflow_lint check + amend analyzer.md hard rule.
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #812 (orchestrator observation: recovery of #812's clobbered state).

## Goal

Prevent any agent (analyzer, experimenter, experiment-implementer, etc.) from running `git reset --hard` on the shared repo root, which silently clobbers concurrent siblings' task state (body.md, plans/, comments.jsonl, REGISTRY entries). Add a mechanical workflow_lint check + amend agent specs with marker-chain-recovery flows to restrict destructive git resets to per-issue worktrees.

## Workflow gap

- **Bug observed:** A task #778 analyzer's marker-chain-recovery ran `git reset --hard` on the shared repo root (commit `bbd6fe97b7` documents it — 'MARKER-CHAIN RECOVERY: analyzer round 1 did a "git reset --hard"'). This silently clobbered task #812's body.md + plans/ + comments.jsonl + REGISTRY entry mid-flight. Sibling #813 was manually recovered via commit `81c52d6a2b`; #812 was left in the clobbered state, breaking every `task.py view` / `task.py find` / `task.py set-status` for #812 until the /issue 812 orchestrator manually recovered it (commit `d29a877e6f`). The recovery cost ~30 minutes of orchestrator context + one full round-trip through the plans/ history.
- **Why it is a workflow gap:** The shared repo root's `tasks/` subtree is not safe against a repo-root `git reset --hard` by any concurrent session. `task.py` holds a per-registry `flock`, not a per-task or per-file lock, so a bulk repo-root reset by another session's subagent can wipe out unrelated tasks' durable state. The analyzer's marker-chain-recovery flow (and any similar recovery flow in other agents) is the immediate offender; no agent spec explicitly forbids repo-root destructive resets.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

Two prongs:

1. **Mechanical guard in `scripts/workflow_lint.py`** — grep every `.claude/agents/**.md` and `.claude/skills/**/SKILL.md` for the string `git reset --hard` (or the shorter `git reset` with `--hard`). Any occurrence outside a per-worktree `git -C "$WT" ...` invocation FAILs the lint. Sibling checks already exist for similar patterns (e.g. `--check-heredoc-dotenv`, `--check-judge-model-pins`).

2. **Agent-spec amendments** — add a hard-rule paragraph to any agent spec whose duties can include a `git reset` (analyzer.md is the known offender; also check experimenter.md, experiment-implementer.md, implementer.md, code-reviewer.md, uploader.md). Rule wording: "Never run `git reset --hard` (or any bulk-destructive git command) on the shared repo root. Marker-chain recovery / staged-changes-cleanup destructive git ops MUST run inside a per-issue worktree via `git -C "$WT" reset --hard`, never on the repo root. The shared root's `tasks/` subtree hosts concurrent sibling tasks' durable state, which task.py's per-registry flock does not protect against a bulk repo-root reset."

## Scope / surfaces

- Primary targets: `scripts/workflow_lint.py` (add mechanical check), `.claude/agents/analyzer.md` (the known offender; add the hard-rule paragraph), and any sibling agent spec that a grep-across the workflow surface reveals mentions `git reset`.
- Grep first: `grep -rln 'git reset' .claude/ CLAUDE.md scripts/` — list every hit in `target_file` (see `.claude/rules/workflow-fix-on-bug.md` § "Before emitting").

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- The lint check must not regress legitimate per-worktree `git -C "$WT" reset --hard` invocations (Step 5a spec-freshness sync, Step 10d merge recovery, etc.).
- If `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/analyzer.md, scripts/workflow_lint.py
- fingerprint: d6335a734640

<!-- workflow-fix-candidate v1 -->
target_file: .claude/agents/analyzer.md, scripts/workflow_lint.py
bug_observed: A #778 analyzer's marker-chain-recovery ran `git reset --hard` on the shared repo root (commit bbd6fe97b7 documents it), which silently clobbered task #812's body.md + plans/ + comments.jsonl + REGISTRY entry mid-flight (#813 was manually recovered via 81c52d6a2b; #812 was left in the clobbered state, breaking every task.py view/find/set-status until this session recovered it). The shared repo root's tasks/ subtree is not safe against a repo-root git reset --hard by any concurrent session, because task.py holds a per-registry flock, not a per-task or per-file lock.
why_workflow_gap: No agent spec explicitly forbids destructive repo-root git resets, and no mechanical lint catches them. Any agent with a marker-chain-recovery or staged-changes-cleanup flow can silently wipe out unrelated concurrent tasks' state.
proposed_change: Add a workflow-lint check that FAILs any agent spec containing an unqualified `git reset --hard` invocation on the shared repo root. Also amend .claude/agents/analyzer.md and any other agent spec with a marker-chain-recovery flow to require running such git resets ONLY inside a per-issue worktree (never on the shared repo root), and document the concrete failure mode: a repo-root git reset --hard truncates concurrent siblings' body.md / plans/ / comments.jsonl and drops their REGISTRY entries.
diff_sketch: |
  # In scripts/workflow_lint.py:
  def check_no_repo_root_git_reset_hard(root: Path) -> list[LintFailure]:
      failures = []
      for f in root.rglob(".claude/agents/*.md").chain(root.rglob(".claude/skills/**/SKILL.md")):
          text = f.read_text()
          # FAIL any HEAD is now at 7697e81590 task #763: epm:progress — poll heartbeat: fit pid 3991 alive (elapsed 14h01m, CPU-time NOT preceded by  /  / 
          for m in re.finditer(r"(?<!-C ..)git\s+reset\s+--hard", text):
              # skip if the match is inside a per-worktree `git -C` invocation
              ...
              failures.append(LintFailure(path=f, line=..., msg="repo-root git reset --hard forbidden"))
      return failures

  # In .claude/agents/analyzer.md:
  + ## Hard rule: no destructive git resets on the shared repo root
  + Never run `git reset --hard` (or any bulk-destructive git command like `git checkout .`, `git clean -f`, `git restore .`) on the shared repo root. Marker-chain recovery / staged-changes-cleanup MUST run inside a per-issue worktree via `git -C "$WT" reset --hard`.
  + Rationale: the shared repo root's `tasks/` subtree hosts concurrent sibling tasks' durable state (body.md, plans/, comments.jsonl, REGISTRY entries). `task.py` holds a per-registry `flock`, not a per-file lock, so a repo-root reset by this analyzer wipes out unrelated tasks' state. Incident: 2026-07-01 clobbered #812/#813 mid-flight (see commits bbd6fe97b7, 81c52d6a2b, d29a877e6f).
confidence: high
related_task: #812
<!-- /workflow-fix-candidate -->
