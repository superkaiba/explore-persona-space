---
title: 'workflow-fix: Relocate live pods_ephemeral.json out of git t'
kind: infra
tags:
- wf-fix
- wf-fix-fp:03d654ff1313
- daily-auto-filed
created_at: '2026-07-09T06:59:23Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): scripts/pods_ephemeral.json
  is the same tracked-file-with-uncommitted-live-mutations class that wiped pods.conf:
  pod_lifecycle.py plain-writes it in-tree (EPHEMERAL_STATE = _PODS_EPHEMERAL_JSON_MAIN
  at :104; non-atomic EPHEMERAL_STATE.write_text at :276), so a repo-root destructive
  git op silently rewinds pod metadata (issue mapping, manual_override flags).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep from a candidate parked on task #821.

## Goal

Protect the live pods_ephemeral.json metadata from destructive repo-root git ops by relocating it out of the working tree with atomic writes, mirroring the #821 pods.conf fix.

## Workflow gap

- **Bug observed:** scripts/pods_ephemeral.json is the same tracked-file-with-uncommitted-live-mutations class that wiped pods.conf: pod_lifecycle.py plain-writes it in-tree (EPHEMERAL_STATE = _PODS_EPHEMERAL_JSON_MAIN at :104; non-atomic EPHEMERAL_STATE.write_text at :276), so a repo-root destructive git op silently rewinds pod metadata (issue mapping, manual_override flags).
- **Why it is a workflow gap:** the failure originates in the workflow surface named below, not in any one experiment.
- **Confidence (emitter):** see parked note

## Proposed change (candidate diff sketch — refine in planning)

  + relocate live pods_ephemeral.json to <git-common-dir>/eps/ (mirror
  +   pod_config's #821 pods.conf contract: seed from tracked file, atomic
  +   tmp+os.replace write, never-drop guard for entries with a RUNNING pod)
  - EPHEMERAL_STATE.write_text(json.dumps(payload, indent=2) + "\n")
  + atomic write via tmp + os.replace; readers resolve the relocated path first

## Scope / surfaces

- Primary target: `scripts/pod_lifecycle.py`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/pod_lifecycle.py
- origin: parked candidate on task #821 at 2026-07-02T02:47:47Z

Verbatim parked note:

```
routed: parked: EPM_WORKFLOW_FIX_SESSION (recursion guard — logged, not auto-filed; see .claude/rules/workflow-fix-on-bug.md § Recursion guard). TWO candidates surfaced this session:

[1] source: prose-followup (implementer r1 report + plan §11)
target_file: scripts/pod_lifecycle.py
bug_observed: scripts/pods_ephemeral.json is the same tracked-file-with-uncommitted-live-mutations class that wiped pods.conf (EPHEMERAL_STATE plain write_text at pod_lifecycle.py:274); a repo-root destructive git op silently rewinds pod metadata (issue mapping, manual_override flags)
why_workflow_gap: live mutable state stored as uncommitted edits of a git-tracked file; no relocation/guard/self-heal analogue shipped for it in #821 (explicit non-goal, plan §11)
proposed_change: apply the #821 pattern to pods_ephemeral.json — relocate live copy to <git-common-dir>/eps/pods_ephemeral.json with seed fallback + atomic write; no --refresh-from-api analogue exists so scope a re-inference strategy
confidence: high

[2] source: orchestrator observation (this session)
target_file: .claude/agents/planner.md, .claude/agents/critic.md (grep 'model: claude-fable-5' across .claude/agents/*.md for the full hit set)
bug_observed: three consecutive planner spawns died with 'Autocompact is thrashing' (total_tokens: 0, ~8-11 tool uses) under the frontmatter pin model: claude-fable-5 + effort: xhigh; the same briefs succeeded immediately with a per-spawn model override to opus
why_workflow_gap: the pinned model's context window is too small for the always-on project context (CLAUDE.md+rules ~100KB) + a 93KB agent spec + tool outputs, so any planner/critic spawn without an override thrashes; orchestrators not knowing the workaround lose whole planning rounds
proposed_change: re-pin the affected agents to a model whose window fits the load (or drop the pin to inherit the session model), after verifying which agents carry the pin and their spec sizes
confidence: medium (mechanism inferred from 3/3 failures + 1/1 fix; exact window limits unverified)
```
