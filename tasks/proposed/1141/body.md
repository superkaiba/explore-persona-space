---
title: 'daily-held: canonical HF model repo at 100k-file limit'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-08T07:00:24Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-07 problem sweep (route 3): c86ff35c (#1090) 09:43Z:
  adapter upload rejected ("would contain 100050 files") — the canonical HF model
  repo hit the 100,000-file hard limit. #1108 shipped the private-overflow-repo fallback
  same day, so uploads keep working, but the canonical repo is frozen at the limit
  and every new adapter lands in overflow.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-07 (route 3) from the nightly transcript problem sweep.

## Goal

Thomas decides the canonical-repo cleanup: purge/archive old adapter trees (the parked adapters/issue_397 review, 242GB, is one candidate; cf. the wandb-archive precedent), or accept overflow-only growth permanently

## Workflow gap

- **Bug observed:** c86ff35c (#1090) 09:43Z: adapter upload rejected ("would contain 100050 files") — the canonical HF model repo hit the 100,000-file hard limit. #1108 shipped the private-overflow-repo fallback same day, so uploads keep working, but the canonical repo is frozen at the limit and every new adapter lands in overflow.
- **Why it is a workflow gap:** Destructive / irreversible action (deleting published artifacts) — route-3 carve-out.

## Proposed change

Held for Thomas; PM should surface alongside the existing HF-storage parked review (issue_397 purge decision).

## Scope / surfaces

- Primary target: `external (HF repo superkaiba1/explore-persona-space)`
- Grep the workflow surface for the pattern before editing and update every hit.

## Provenance

- Evidence: c86ff35c (#1090) 09:43-09:46Z; task #1108 (completed).
