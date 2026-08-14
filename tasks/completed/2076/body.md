---
title: Clear the api-dispatch-routing lint FAIL on scripts/issue1482_blind_read_api.py
  (route through api_dispatch or add the documented waiver)
kind: infra
tags: []
created_at: '2026-08-04T23:37:31Z'
has_clean_result: false
origin_prompt: 'Surfaced by steer-null-1902 inline round on #1902 (2026-08-04): the
  no-flags workflow_lint run surfaced a PRE-EXISTING fleet-wide FAIL naming scripts/issue1482_blind_read_api.py
  (direct anthropic.Anthropic call outside api_dispatch routing) - not this round''s
  payload, but it makes the no-flags gate red for everyone.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the orchestrator session working #1902 inline rounds, from a concern surfaced by an inline-round subagent's payload lint gate run (2026-08-04): the no-flags `workflow_lint.py` gate reports a FAIL naming `scripts/issue1482_blind_read_api.py`, which makes the shared lint gate red for every session that runs it (inline payload gates, Step 9c pre-commit surfaces).

## Goal

Clear the `--check-api-dispatch-routing` FAIL on `scripts/issue1482_blind_read_api.py`: route its direct Anthropic client usage through `src/explore_persona_space/llm/api_dispatch.py`, or — if the direct client is genuinely correct for this instrument (a blind-read probe may deliberately bypass caching/routing) — add the check's own waiver comment `# API_DISPATCH_ROUTING_EXEMPT: <reason>` with a real reason. The fixer decides with the file open; either resolution must leave the check green.

## Gap

- **Observed:** `uv run python scripts/workflow_lint.py --check-api-dispatch-routing` → `workflow_lint: FAIL (1 error(s))`, message: "scripts/issue1482_blind_read_api.py: constructs/calls the Anthropic client directly (anthropic.Anthropic(...) / .messages...create(...)) outside the routing layer."
- verified-at-filing: `uv run python scripts/workflow_lint.py --check-api-dispatch-routing` → FAIL, 1 error, naming exactly this file (2026-08-04); in-file sites confirmed by `grep -n anthropic scripts/issue1482_blind_read_api.py` → line 40 `import anthropic  # noqa: E402`, line 198 `client = anthropic.Anthropic()`.
- unverified hypothesis — verify at plan time: the surfacing round reported this makes the NO-FLAGS default run red fleet-wide (confirm by checking whether `--check-api-dispatch-routing` is in the no-flags default bundle in `scripts/workflow_lint.py`); my own no-flags confirmation run was output-truncated and is not evidence either way.
- No live session currently mapped to #1482 (`spawn_session.py list` grep at filing time returned none), so this is not racing an owner.

## Scope / surfaces

- Primary target: `scripts/issue1482_blind_read_api.py` (experiment script — the fix is code-side routing or the check's own documented waiver, NOT a lint change).
- Do NOT weaken `workflow_lint.py` or its check; the check is behaving as designed.
- Context: the script belongs to the #1482 SAE-feature blind-read line; read its call pattern before choosing routing vs waiver — if it needs judge-cache-bypass semantics, the waiver with that stated reason is the right fix; otherwise thread it through `api_dispatch` per `.claude/rules/llm-judging.md` / `docs/api_throughput_guidelines.md`.

## Constraints / invariants

- `uv run python scripts/workflow_lint.py --check-api-dispatch-routing` green after the fix; ruff clean on the touched file; no behavior change to the instrument's science outputs beyond the client plumbing (or zero code change if the waiver route is chosen).
