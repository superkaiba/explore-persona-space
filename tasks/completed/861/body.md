---
title: 'artifacts: promote contrastive negatives to artifacts/negatives.py (Phase
  0c)'
kind: infra
tags: []
created_at: '2026-07-02T10:03:19Z'
has_clean_result: false
origin_prompt: Autonomous orchestration of the unified artifact factory; plan /home/thomasjiralerspong/.claude/plans/help-me-to-devise-vectorized-tower.md
  (Phase 0c).
---
## Overview / Motivation

Phase 0c of the unified artifact factory (plan:
`/home/thomasjiralerspong/.claude/plans/help-me-to-devise-vectorized-tower.md`). The `artifacts/` package
is now on `main` (`behavior.py`, `context.py` from #852). Promote the contrastive-negatives machinery into
the package as a reusable panel registry.

## Goal

`src/explore_persona_space/artifacts/negatives.py`:
- `NegativeContext` (promoted from `scripts/issue664_common.py`) — interoperating with the new
  `artifacts/context.Context` resolver where sensible (a negative is a context under which B is NOT trained).
- A **panel registry** (named negative panels) + a `default_panel()` seeded from the issue664 4-context
  panel (police / PersonaHub / rephrase-curious / wildchat-tech-support), **always including the bare
  default assistant** as a negative (the safety target; the highest-value negative, #464).
- `assert_panel_disjoint_from_sources(realized_sources)` — the HARD invariant (panel ∩ realized sources
  == ∅, at slug AND identity level, #527/#538), run at build time against the REALIZED panel.
- A ~1:1 positives-to-total-negatives interleave helper (negatives split across the panel).

## Scope / constraints

- Reuse the `issue664_common` implementation; make `artifacts/negatives.py` canonical and have
  `issue664_common` re-import from it (supersede→dedupe per `code-style.md` — no two live copies).
- Append any package export to `artifacts/__init__.py` **append-only** (sibling Phase-0 tasks 0e/0f land
  concurrently — minimize `__init__.py` merge conflict).
- CPU unit tests: disjointness assert FIRES on a source∩panel overlap (slug + identity); default assistant
  always present; ratio helper splits ~1:1.
- `uv run ruff check/format` + `uv run pytest` green.

## Rules to read

`.claude/rules/contrastive-negatives.md`, `.claude/rules/code-style.md` (supersede→dedupe).
