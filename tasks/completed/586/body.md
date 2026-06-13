---
title: 'Campaign runner: question-level autonomous orchestration layer (/campaign)
  above /issue'
kind: infra
tags: []
created_at: '2026-06-11T04:02:58Z'
has_clean_result: false
---
# Campaign runner: question-level autonomous orchestration layer (/campaign) above /issue

Built and merged 2026-06-10 (merge `aaa510ce4`, branch `worktree-agent-a263b3c688c2d5b9d`). Design approved in chat after a literature dive over Jun-2025–Jun-2026 autonomous-research systems (AIRA_2, Kosmos, ASI-Arch, Anthropic recursive-self-improvement study, Karpathy autoresearch, METR time-horizon data).

## What it is

A campaign is a `kind: campaign` task pinned to ONE `docs/open_questions.md` anchor. A dedicated session runs `/campaign <N>` and loops: ingest landed child clean-results (critic-gated bodies only) into a writable world model (`artifacts/world-model.md`, claim → #task evidence) → check stop criteria → propose next experiments (info-gain per GPU-hour, shared-nothing arms for max parallelism) → file + spawn children as ordinary `/issue <child> --auto` sessions with the full adversarial-planner + critic stack → repeat.

## Key properties

- **Human gates:** brief approval IN (executes only from status `approved`; gate `campaign_brief_approval`), daily digest + final report OUT. Per-child promotion stays user-only but never blocks the campaign.
- **Harness-side enforcement:** budget (GPU-hours, never dollars), max-experiments, wall-clock, dry-counter, and campaign-level working-belief confidence target live in `campaign_state.py` + `artifacts/campaign-state.json`; the watcher's campaign pass (pass 9) adds respawn, progress-based stall detection, a budget backstop, and terminal reap.
- **Crash-safe filing:** state persisted (`filed` + child id + committed hours) BEFORE spawning; children carry `<!-- campaign_experiment_id -->` for deterministic orphan adoption; committed-hours reconcile is idempotent.
- **Defaults:** 250 GPU-h total, 100 per child, 8 experiments, 4 concurrent, 5 days — overridable per-brief (frontmatter > spawn-registry caps > module defaults).

## Surface

`src/explore_persona_space/campaign_state.py`; `task.py list-children` + `kind: campaign`; `spawn_session.py spawn-campaign`; watcher campaign pass; `.claude/skills/campaign/SKILL.md` + `campaign-tick`; workflow.yaml (6 `epm:campaign-*` markers, gate id=16); CLAUDE.md routing amendment (campaign exception to same-issue-only auto-execution, user-approved 2026-06-10). 25 new tests; reviewed in 2 rounds (2 blockers fixed + verified closed).

## Usage

1. Create: `task.py new --kind campaign --title "..." --goal "<question>"`, write `## Campaign Brief` (question anchor, hypotheses, initial experiment table, budget overrides).
2. Review + `task.py set-status <N> approved`.
3. `spawn_session.py spawn-campaign --issue <N>` — runs unattended; watch digests on the dashboard.
