---
title: 'codex-composer-common: require lens-unique slugs for auxiliary temp files,
  not just the prompt file'
kind: infra
tags: []
created_at: '2026-08-19T20:29:10Z'
has_clean_result: false
origin_prompt: 'Surfaced in prose by the codex-methodology-baselines-critic composer
  during #2389 v2 Step 3 CRITIQUE round 1: the parallel codex-statistics-critic composer
  for the same task overwrote its generically named /tmp/handed-2389-brief.md mid-verification,
  producing 13 false leak residuals until it rebuilt under lens-scoped names.'
workflow: v1
---
---
kind: infra
---

# codex-composer-common.md: require lens-unique slugs for AUXILIARY temp files, not just the prompt file

## Problem

`.claude/rules/codex-composer-common.md` mandates lens-scoped naming for the
composed PROMPT file only. It says nothing about the auxiliary scaffold /
handed-span temp files a composer writes while working.

The workflow-v2 post-approval critic panel
(`.claude/skills/adversarial-planner-v2/SKILL.md` CRITIQUE mode) always spawns
**three sibling `codex-*` composers for the same issue in ONE batch**
(`codex-statistics-critic`, `codex-methodology-baselines-critic`,
`codex-efficiency-critic`). Any composer that derives an auxiliary temp path
from the issue number alone therefore collides with its siblings.

## Observed occurrence (task #2389, round 1, 2026-08-19)

The `codex-methodology-baselines-critic` composer wrote its handed-span
scaffold to `/tmp/handed-2389-brief.md`. The concurrently-running
`codex-statistics-critic` composer for the SAME issue overwrote that file
mid-verification. The methodology composer's numeric-leak verifier then read
the sibling's content and reported **13 false leak residuals**.

The composer recovered on its own by rebuilding under lens-scoped names
(`/tmp/cmbc2389-*`) and its final prompt was CLEAN (0 residuals), so no bad
prompt reached Codex this time. The failure mode is nonetheless real: the
verifier's output was wrong for a period, and the recovery depended on the
composer noticing the anomaly rather than on any structural guarantee.

## Why this matters beyond one false alarm

The numeric-leak verifier is a correctness gate on what reaches Codex. A gate
that can silently read a sibling's bytes is not a gate. The failure is
non-deterministic (it depends on interleaving), so the next occurrence could as
easily go the other way: a verifier reading a sibling's CLEAN scaffold and
passing a prompt that actually carries residuals.

The collision is guaranteed-possible by construction, not incidental — the v2
panel's fan-out shape means three same-issue composers always run
concurrently. The prompt-file naming rule already anticipates exactly this
problem; the gap is that it stops at the prompt file.

## Proposed fix

In `.claude/rules/codex-composer-common.md`, widen the existing lens-scoped
naming requirement from the prompt file to **every** temp path a composer
writes: scaffolds, handed-span extracts, verifier inputs/outputs, and any
intermediate. Require a lens-unique slug component (e.g.
`/tmp/<lens-slug>-<issue>-<purpose>.md`), the same discipline the prompt file
already follows.

Consider also stating the rationale inline (three sibling composers per issue
in one v2 batch) so the requirement reads as structural rather than stylistic
— a future composer author who sees only "use a unique name" may reasonably
assume the issue number suffices.

## Acceptance criteria

1. `.claude/rules/codex-composer-common.md` requires a lens-unique slug for
   ALL composer-written temp paths, with the three-sibling-fan-out rationale
   stated.
2. The four v2 composer specs (`codex-statistics-critic.md`,
   `codex-methodology-baselines-critic.md`, `codex-efficiency-critic.md`) and
   the v1 composers (`codex-critic.md`, `codex-code-reviewer.md`,
   `codex-interpretation-critic.md`, `codex-clean-result-critic.md`,
   `codex-follow-up-critic.md`) are checked for any prescribed or example
   temp path that is not lens-unique; fix each.
3. If a mechanical check is cheap, add a `workflow_lint.py` arm flagging a
   composer spec whose example/prescribed temp paths are keyed on the issue
   number alone. If it is not cheap, say so and rely on the rule text.

## Provenance

Surfaced in prose (not as a `workflow-fix-candidate` block — the `codex-*`
carve-out in `.claude/rules/workflow-fix-on-bug.md` bars those agents from
emitting candidate blocks) by the `codex-methodology-baselines-critic`
composer during task #2389 workflow-v2 Step 3 CRITIQUE round 1, filed by the
#2389 orchestrator per CLAUDE.md § Workflow-fix-on-bug protocol
("Surfaced-prose follow-ups count too").
