---
title: Codex reviewers' "Concerns to persist" never reach raise-concern, so the next
  round's prior-concerns gate walks an empty ledger
kind: infra
tags:
- workflow-fix
created_at: '2026-08-16T14:24:43Z'
has_clean_result: false
origin_prompt: 'surfaced by the #2321 orchestrator during code-review round 2: r1
  Codex verdict carried 8 concerns-to-persist, list-concerns returned empty going
  into r2'
workflow: v1
---
# Codex reviewers' "Concerns to persist" never reach `raise-concern`, so the next round's prior-concerns gate walks an empty ledger

## Goal

Close the gap between a Codex twin reviewer emitting a `## Concerns to persist` section and those concerns actually existing in the machine-readable per-task concerns ledger, so that `code-reviewer.md` Step 0.8 (prior-concerns check) and the `clean-result-critic` binding-concerns audit have real inputs on every subsequent round.

## Evidence (observed on #2321, 2026-08-16)

#2321 ran two doubled code-review rounds on a payload that deletes ~490,000 files from the canonical HF data repo.

- Round 1's Codex twin verdict (`epm:code-review-codex v1`) carried a `## Concerns to persist` section with **8 items**.
- **None was ever persisted** via `scripts/task.py raise-concern`. Going into round 2, `task.py list-concerns 2321` returned an **empty ledger**.
- The round-2 `codex-code-reviewer` composer noticed this itself at compose time and improvised a workaround: it built a **pseudo-ID union checklist** from the round-1 output files and required a per-item `VERIFIED-CLOSED` / `NOT-CLOSED` line, declaring `NOT-CLOSED` on a round-1 blocker a substantive FAIL. That recovery worked — but it depended on one composer noticing a gap and inventing a compensating contract, not on the workflow.
- The ledger only became populated when the round-2 `reconciler` made 8 explicit `raise-concern` calls (2 BLOCKER + 6 CONCERN) as part of its binding adjudication.

So the persistence happened twice by improvisation and zero times by contract.

## Why this matters

The prior-concerns machinery is the mechanism that stops a reviewer finding from being silently dropped between rounds. When the ledger is empty:

- `code-reviewer.md` Step 0.8's prior-concerns check has nothing to walk, so it passes vacuously.
- A round-N concern can vanish and round N+1 will not notice. #2321's round-2 contract-bearing group found exactly one such silently-dropped round-1 residual (the non-blocking dirty-`pack_dir` sweep) by hand-walking the source verdict FILES — not the ledger.
- The `clean-result-critic` binding-concerns audit (markdown lens 14) and `verify_task_body.py`'s `check_concerns_audit` inherit the same vacuity on any task whose concerns were never persisted.

The Claude-side reviewers have the same exposure in principle; the observed, reproduced instance is the Codex twins, whose verdicts are posted from a FILE by the orchestrator (SKILL.md § File-only Codex verdict posting) — the posting path mechanically extracts the marker block and never reads the concerns section, so nothing in the chain is positioned to call `raise-concern`.

## Scope to investigate

1. Whether the duty belongs to the orchestrator's verdict-posting step (parse `## Concerns to persist` and issue `raise-concern` per item) or to the reviewer/composer specs, and whether it should apply to the Claude reviewers symmetrically.
2. Whether a mechanical check should FAIL a round when a verdict carries a concerns section with N items and the ledger gained fewer than N entries — the same shape as the existing marker-shape gates.
3. Whether the file-only posting path (deliberately grep-and-extract, to avoid paging trigger-dense findings into orchestrator context) can persist concerns without reading the body — e.g. by having the composer emit a machine-readable concerns block the poster can forward blind.
4. Whether `list-concerns` being empty should itself be surfaced at Step 0.8 as "no ledger to check" rather than a silent pass, so a future vacuous gate is visible.

## Non-goals

Do not redesign the concerns schema or the `raise-concern` CLI. Do not weaken the file-only Codex verdict-posting path — it exists to keep trigger-dense findings out of orchestrator context, and any fix must preserve that.

## Provenance

workflow_fix_target: .claude/skills/issue/SKILL.md

Surfaced by the #2321 orchestrator during code-review round 2 (2026-08-16). Not a #2321 experiment/data defect — a gap in the workflow surface itself (`.claude/skills/issue/SKILL.md` Step 5c verdict posting, `.claude/agents/codex-code-reviewer.md`, `.claude/agents/code-reviewer.md` Step 0.8).
