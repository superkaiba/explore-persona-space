---
title: 'verify_plan.py: mechanical pre-pass gate for experiment plans at adversarial-planner
  Phase 1.5'
kind: infra
tags: []
created_at: '2026-06-12T20:33:36Z'
has_clean_result: false
origin_prompt: 'Okay make all these changes except for the ones in section A (Model
  tiering) ... Use subagents/background happy coder sessions as needed (token-audit
  fix #21)'
---
# verify_plan.py — mechanical pre-pass gate for experiment plans (adversarial-planner Phase 1.5)

## Motivation

Clean-result bodies have a 17-check mechanical verifier (`verify_task_body.py`); plans have none, yet the plan critic is the most expensive and most load-bearing review site (a missed structural gap costs GPU-hours downstream). Literature basis for the design: grounded mechanical verification beats free-form critique at any reviewer tier (Stechly & Kambhampati 2024, arXiv:2402.08115; McAleese et al. 2024, arXiv:2407.00215). This is token-audit fix #21 (2026-06-12).

## Deliverable

`scripts/verify_plan.py --issue <N> [--plan-file <path>]` — parses the newest `plans/v{N}.md` (or the given file) and reports PASS/FAIL per check + an exit code. Pure structural/presence checks only — NO LLM calls, no judgment. Mirror `verify_task_body.py`'s architecture (numbered checks, FAIL blocks / WARN ships-if-acknowledged, JSON output mode).

## Checks (initial set — derive exact predicates from CLAUDE.md Critical Rules + planner.md)

1. §11 hyperparameter table exists; every load-bearing row carries a `Source:` (arXiv id / paper table / prior issue `#<M>`) or an explicit `ungrounded — needs smoke-test` marker; no blank sources.
2. Per-DV measurement-validity declaration present (`kind: experiment` only): construct named, metric named, on-distribution status stated; off-distribution proxies carry the required validation-or-argument text.
3. Data-source tier named in §4 (real-world / established dataset / diverse-LLM-synthetic / programmatic) with the justification required for tiers 3-4.
4. Behavior-implantation plans: contrastive-negative set present (panel + ratio + disjointness check named) OR one of the two named exemptions stated.
5. GPU-hour estimate present (needed by the Step 2c auto-approval cap).
6. Reused-trained-artifact sections: fitness attestations (a)-(g) each present for every artifact recorded as reused.
7. Replication plans (Goal mentions replicating a paper): replication-fidelity section present (paper recipe vs deviations table).
8. Kill criteria / success criteria section present and non-contradictory in form (both present, neither empty).
9. Plan declares conditions/cells table and seeds (the consistency-checker's input surface).
10. Marker-leakage plans: the marker-recipe acknowledgment line present (the rules were read; anchor band + bystander gating stated).

Checks must be skippable by `kind:` (analysis/infra/batch/survey exempt where CLAUDE.md says so).

## Integration

- `/adversarial-planner` SKILL.md: run at Phase 1.5 alongside the fact-checker; FAIL bounces the plan back to the planner BEFORE the critic ensemble spawns (critics then review substance, not structure).
- `workflow.yaml`: add the gate reference; post a marker (`epm:plan-verify v1`) with the check results.
- Tests: `tests/test_verify_plan.py` with fixture plans (passing, each-check-failing). Follow the test style of `tests/test_verify_task_body.py`.

## Acceptance criteria

1. Script + tests merged, `uv run pytest tests/test_verify_plan.py` green; ruff clean.
2. Phase 1.5 wiring in adversarial-planner SKILL.md + workflow.yaml marker documented.
3. Run against the 3 most recent approved plans retroactively; report (do not fix) any FAILs they would have raised — this calibrates the checks against reality.
4. No LLM calls anywhere in the script.
