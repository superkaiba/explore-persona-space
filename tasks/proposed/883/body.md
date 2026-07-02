---
title: 'workflow-fix: canonicalize setsid launcher-script SSH-MCP pod launch shape'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c202f67dbb08
created_at: '2026-07-02T19:22:03Z'
has_clean_result: false
origin_prompt: 'Prose follow-up from #841 experimenter launch report: setsid-launcher-script
  is the safe SSH-MCP launch shape (sh shell has no ''source'', bare nohup risks SIGHUP
  reaping; incidents #545/#444/#541)'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up surfaced on task #841 (emitting agent: experimenter).

## Goal

Canonicalize the setsid launcher-script launch shape (sh-safe dot-source of `.env`, `setsid` + `</dev/null`, launcher writes its own pidfile) as the SSH-MCP pod launch pattern in `experimenter.md` and the SKILL.md Step 6d.1 brief template.

## Workflow gap

- **Bug observed:** the Step 6d.1 brief's literal `source .env && nohup bash -c '...' & echo $!` launch shape fails under the SSH-MCP `sh` shell (`source: not found`, incident #545) and risks SIGHUP reaping without `setsid`/`</dev/null` (#444/#541); the #841 experimenter had to improvise the safe shape at launch time.
- **Why it is a workflow gap:** the launch-command shape is prescribed by the workflow surface (the SKILL.md Step 6d.1 brief template quotes "the exact nohup launch command"; experimenter.md owns the launch protocol) — every future experimenter either re-hits the sh-shell/`source` failure or silently deviates from its brief.
- **Confidence (emitter):** high (the #841 experimenter hit and worked around it live; #545 is the recorded prior incident).

## Proposed change (candidate diff sketch — refine in planning)

- experimenter.md launch protocol: replace the bare `nohup ... & echo $!` prescription with the launcher-script pattern:
  + write /workspace/launch_issue_<N>.sh: `#!/bin/sh` … `. ./.env` (dot-source, sh-safe) … `setsid nohup bash -c '<chain>' > <log> 2>&1 < /dev/null & echo $! > <pidfile>`
  + launch via `sh /workspace/launch_issue_<N>.sh`; verify pid + log advancing as today.
- SKILL.md Step 6d.1 brief template: reference the launcher-script shape (not a literal top-level `source .env && nohup` line).

## Scope / surfaces

- Primary target: `.claude/agents/experimenter.md, .claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'nohup' .claude/ CLAUDE.md scripts/bootstrap_pod.sh`) and update every prescriptive launch-shape hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/agents/experimenter.md, .claude/skills/issue/SKILL.md
- fingerprint: c202f67dbb08

Surfaced prose (verbatim, from the #841 experimenter's launch report): "One deviation from the brief's launch shape, same logical contract: I wrapped the three-script chain in a setsid-launcher script (/workspace/launch_issue_841.sh) that sources .env internally via `. ./.env` and writes its own pidfile, rather than the brief's literal top-level `source .env && nohup bash -c '...' &` line. Reason: the SSH-MCP shell is `sh`, so a top-level `source .env` fails (`sh: source: not found`, incident #545) and the captured `$!` would catch a stray job; and bare `nohup ... &` without setsid/`</dev/null` risks SIGHUP reaping on session exit (#444/#541). … For future briefs, the setsid-launcher-script pattern is the safe SSH-MCP launch shape."
