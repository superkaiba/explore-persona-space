---
title: 'codex-clean-result-critic Step 2(a): add Lens 13 plan-fetch block to the compose
  replacement mandate'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-23T16:48:17Z'
has_clean_result: false
origin_prompt: workflow-fix-candidate v1 from codex-clean-result-critic composer,
  /issue 2476 9a-bis r2 compose 2026-08-23
workflow: v1
---
# codex-clean-result-critic Step 2(a): Lens 13 plan-fetch bash block escapes the replacement mandate

## Goal

Close a compose-time gap in `.claude/agents/codex-clean-result-critic.md`: Step 2(a) mandates replacing only Lens 14's ledger-fetch bash block when inlining `.claude/rules/clean-result-critic-lens-reference.md`, but Lens 13 opens with its own run-a-repo-script block (`plan_path=$(uv run python scripts/task.py find <N>)/plans/plan.md` + `cat`). The Step 4 no-residue greps match neither `task.py find` nor bare plan-cat forms, so an unpatched Lens 13 block ships a repo-script instruction into Codex's read-only sandbox, contradicting the prompt's "Do not execute any repo script" rule and inviting a spurious `BLOCKED` on load-bearing Lens 13 (→ false needs_targeted_fix + data-access-blocked).

## Proposed fix

In Step 2(a), extend the replacement mandate: also replace Lens 13's plan-fetch bash block with a by-name reference to the prompt header's absolute PLAN path. Optionally add a `task\.py find` grep to the Step 4 no-residue guard so a future unpatched block fails compose loudly instead of shipping.

## Evidence

#2476 9a-bis round-2 compose (2026-08-23, composer session): the composer had to patch the Lens 13 block manually and flagged the gap; lens reference lines 1216-1222. Round-1 compose the same day made the identical manual edit (its handoff notes list "Lens 13 plan-fetch bash block → PLAN-path by-name reference" among sanctioned edits), so the manual patch is already recurring.

## Acceptance

- Step 2(a) names the Lens 13 plan-fetch block in the replacement mandate with the by-name PLAN-path substitution.
- (Optional hardening) Step 4 no-residue guard greps for `task\.py find` in the envelope-stripped prompt.
- `workflow_lint.py` no-flags run clean; any composer-contract pin tests updated alongside.

Provenance: workflow-fix-candidate v1 emitted by the codex-clean-result-critic composer during /issue 2476 Step 9a-bis round 2; confidence medium; auto-filed by the #2476 orchestrator per .claude/rules/workflow-fix-on-bug.md.
