---
title: 'workflow-fix: pin GNU grep in SKILL.md lint-gate -qv triggers (ugrep shadowing)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c3feefdf168e
created_at: '2026-07-08T06:57:10Z'
has_clean_result: false
origin_prompt: 'orchestrator-observed during #928 Step 10d: shell grep resolves to
  ugrep 7.5.0; -qv returns rc=1 on selected lines; lint gate silently mis-skipped'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #928 (emitting agent: orchestrator).

## Goal

Pin GNU grep (/usr/bin/grep) or a -qv-free form in the SKILL.md pre-push lint-gate trigger snippets so the gate cannot silently mis-skip when shell grep resolves to ugrep.

## Workflow gap

- **Bug observed:** ugrep 7.5.0 shadows `grep` in the session shell (GNU grep 3.7 is at /usr/bin/grep and wins `which`, but the profile-initialized shell resolves `grep` to ugrep) and its `-q -v` combination returns rc=1 even when non-matching lines are selected (`printf 'a\nb\n' | grep -qv '^a'` → rc=1; GNU → rc=0). The Step 10d pre-push lint-gate trigger (`grep -qvE '^(tasks/|...)'`) therefore returned `skip-artifact-only` on a code-bearing payload — the gate silently disarmed. Live incident: #928 matched-length round merge, 2026-07-08 06:47Z; caught because the verdict contradicted the visible own-diff, re-run with /usr/bin/grep pinned → gate armed, both legs clean, pass.
- **Why it is a workflow gap:** the canonical gate snippets in .claude/skills/issue/SKILL.md (2 hits: the shared form (i)/(ii) block and the surgical form (iii) block) assume GNU `-qv` semantics; any environment where grep is shadowed (this VM, today) silently converts the fail-closed gate into a silent skip — the exact failure class the gate exists to prevent (#931).
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
- elif grep -qvE '^(tasks/|figures/|eval_results/|ood_eval_results/|raw/|data/|docs/methodology/)' \
+ elif /usr/bin/grep -qvE '^(tasks/|figures/|eval_results/|ood_eval_results/|raw/|data/|docs/methodology/)' \
```
(both SKILL.md occurrences — lines ~8548 and ~9066; alternatively replace the -qv combo with a shadow-proof form, e.g. `[ -n "$(grep -vE '...' file)" ]`. Consider ALSO: a workflow_lint check flagging `grep -qv`/`-vq` combos on the workflow surface; and investigating/removing the ugrep shadowing itself (profile alias/PATH entry) as the root cause — if removed, the pin is still cheap insurance.)

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing (`/usr/bin/grep -rnE 'grep +-[a-zA-Z]*q[a-zA-Z]*v|grep +-[a-zA-Z]*v[a-zA-Z]*q' .claude/ CLAUDE.md scripts/*.sh`) and update every hit; the 2026-07-08 sweep found exactly the two SKILL.md hits.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: c3feefdf168e

Surfaced prose (verbatim): the #928 session's pre-push lint gate returned skip-artifact-only on a 12-file payload containing scripts/ + tests/ paths; isolation showed shell grep = ugrep 7.5.0 with rc-inverted -qv semantics; re-run with /usr/bin/grep pinned armed the gate correctly (pass, both legs clean).
