---
title: 'workflow-fix: reclaim LESSONS.md index headroom — aggregate cap is ~60 B from
  blocking every new rule file'
kind: infra
tags:
- wf-fix
- lessons-index
created_at: '2026-08-24T04:36:46Z'
has_clean_result: false
origin_prompt: 'Filed from #2313 round-2 Step 5c-ter concerns walk: concern_id lessons-index-headroom
  upheld KEEP-OPEN by the round-2 Codex closure ledger; the substantive lever (gotchas
  row trim or a deliberate cap revision) was ruled out of #2313''s round by its binding
  reconciler.'
workflow: v1
---
# Reclaim LESSONS.md index headroom — the aggregate cap is ~60 B from blocking every new rule file

`kind: infra`. Filed from #2313's round-2 concerns walk (`concern_id: lessons-index-headroom`, raised by `codex-code-reviewer`, upheld KEEP-OPEN by the round-2 Codex verdict).

## Problem

`.claude/rules/LESSONS.md` is the always-on lessons index imported by `CLAUDE.md`. `scripts/workflow_lint.py --check-lessons-index` enforces an aggregate byte cap, `_LESSONS_MAX_BYTES` (verify the current line; it was at `scripts/workflow_lint.py:16374` as of 2026-08-24), value **10492**.

Post-#2313 the file sits at **10432 bytes — 60 bytes of headroom.** A normal new rule row (`- <rule>.md — <trigger>`) costs roughly 80-280 B, so **the next `.claude/rules/*.md` addition anywhere in the repo FAILs the lint**, and adding a rule file REQUIRES an index row (the same check enforces it). This is a fleet-wide block on the standard way a new lesson gets recorded, not a #2313-local issue.

How the budget got spent, for context rather than blame: the cap comment at `workflow_lint.py` (around `:16266-16269`) records that #2158 raised it 10205 → 10492 to buy exactly one row plus ~40 B of slack. #2313 then spent part of that slack on a load-bearing trigger extension (its round 2 trimmed incidental wording back, 212 → 184 B for its row, recovering 28 B). Both changes were individually correct; the aggregate is simply out of room.

## Why #2313 could not close it

#2313's `epm:review-reconcile v1` ruled the substantive lever OUT of that task's round, and its plan v3 §9 licensed only "exact wording of the extended LESSONS trigger". Round 2 executed the full reclaim available inside that scope and the round-2 reviewers agreed it did — they split only on ledger bookkeeping (Claude: close with the residual named; Codex: KEEP-OPEN because 60 B does not achieve the concern's operational goal). Both classified it non-blocking. The blocker to closing was SCOPE, not effort, which is why it routes here.

## Goal

Restore enough aggregate headroom that adding a new rule file is not blocked, without weakening any row's plan-time trigger — a row's trigger words are what make its rule reachable at plan time, which is the entire function of the index.

## The two candidate levers — the plan decides, this body does not prescribe

1. **Trim the largest rows.** The `gotchas` row alone is **1353 B**, ~13% of the whole index; `upload-policy` (274 B) and `pod-side-reporting` (273 B) are the next largest. The lint's own warn-band line prints this ranking, so it is self-reporting. A trim must preserve every trigger word — the risk is that `gotchas` is a genuinely multi-trigger row and over-trimming makes a real rule unreachable at plan time, which is a worse failure than the byte cap.
2. **A deliberate cap revision.** #2158's comment frames 10492 as a considered decision, not an accident, so raising it again is a decision to make explicitly and record — including whether the index's always-on token cost is still worth its current size, and whether the warn band (>7200) should move with it.

A combination is legitimate. So is concluding that the right answer is a structural change to how the index scales (e.g. per-row budgets rather than one aggregate) — say so with reasoning rather than forcing a trim.

## Acceptance criteria

1. Adding a representative new rule row (~120 B) leaves `--check-lessons-index` PASSing — demonstrate it with a scratch row, then remove it.
2. No row loses a trigger word. Enumerate every row whose text changes and show the trigger vocabulary before/after.
3. `uv run python scripts/workflow_lint.py --check-lessons-index` PASSes, and the no-flags run shows no NEW failures against a captured baseline.
4. `uv run pytest tests/test_consolidate_lessons.py tests/test_rule_glob_scope.py -x` passes.
5. If the cap is raised rather than rows trimmed, the new value carries an in-code comment recording the rationale and the date, matching the existing #2158 comment convention.

## Sequencing note — read before planning

Plan this against **post-merge** state. #2313 was unmerged when this task was filed, so `origin/main` at filing time did NOT yet contain #2313's LESSONS row edit; measuring there yields a stale aggregate (~10406 rather than 10432) and a headroom figure ~26 B too generous. Re-measure the live file at plan time. This task was filed with `--no-dispatch` for exactly this reason — the watcher's `proposed_infra_sweep` pass is the intended dispatcher, after #2313 lands.

If a concurrent session is editing `.claude/rules/LESSONS.md`, follow `.claude/rules/cross-session-writer-arbitration.md`: probe, post a `file-set claim:` marker, and sequence-after-commit rather than racing.

## Provenance

workflow_fix_target: .claude/rules/LESSONS.md, scripts/workflow_lint.py

Filed by the #2313 orchestrator session at its round-2 Step 5c-ter concerns walk. Evidence: #2313 `epm:code-review v2` (Claude PASS, "close it, name the residual, route the trim as a separate follow-up"), `epm:code-review-codex v2` (CONCERNS, closure-ledger line `lessons-index-headroom: KEEP-OPEN`), and `epm:review-reconcile v1` (which named the `gotchas` row as the substantive lever and placed it out of #2313's round).
