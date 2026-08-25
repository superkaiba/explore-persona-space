---
title: 'workflow-fix: test (10) FAMILY_OF parity is one-way + comment-blind — enforce
  bidirectional dict equality between the Step 5a and Step 10d copies'
kind: infra
tags:
- wf-fix
- step5a-family-sync
created_at: '2026-08-21T19:35:31Z'
has_clean_result: false
origin_prompt: 'reconciler standing recommendation on the #2260 round-1 PASS verdict:
  harden test (10) to bidirectional _family_of_map equality + bare-shape pin + the
  two negative controls'
workflow: v1
---
kind: infra

## Goal

Harden `tests/test_issue_skill_lint_family_sync.py`'s test (10) so it enforces **exact bidirectional equality** of the `FAMILY_OF` tables between the Step 5a copy (`.claude/skills/issue/steps/09-step-5.md`) and the Step 10d copy (`.claude/skills/issue/steps/18-step-10d.md`), closing the two divergence channels the current comparator cannot see.

## The gap (surfaced, adjudicated non-blocking, still real)

Test (10) today iterates only the Step-5a span's lines and asserts each selected line also occurs in the Step-10d span. Two consequences:

1. **One-way.** A `FAMILY_OF` assignment present ONLY in the Step-10d copy passes — nothing iterates the Step-10d span looking for Step-5a peers.
2. **Suffix-filtered.** The selection predicate is `stripped.startswith("FAMILY_OF[")` AND `stripped.endswith(('="workflow"', '="lint"', '="guard"', '="agents"'))`, so ANY assignment carrying a trailing comment is silently skipped. Meanwhile the guard-(20) parser `_family_of_map` (~`:2194-2201`) is deliberately comment-TOLERANT, so a commented assignment is real to guard (20) and invisible to test (10).

Test (14) compares SPECS/SPECS_10D path TOKENS only and carries no family VALUES; test (1) pins the Step-5a `SPECS` literal only. Neither can close the gap.

Measured current state (2026-08-21, branch `issue-2260` at `89c1f003b6`): both copies carry **49** `FAMILY_OF` assignments and the parsed maps are **dict-equal in both directions** (agents 31, workflow 7, lint 7, guard 4); SPECS and SPECS_10D are both 53 tokens and set-equal. So this is a GUARD-COVERAGE gap, not a live divergence — nothing is broken on trunk today.

## Why it was not fixed in #2260

#2260's registered change set touched test (10) with exactly one edit: extending the suffix tuple by the single `'="agents"'` token. The one-way + comment-blind comparator shape is **pre-existing `origin/main` behavior** that approved plan v3 explicitly recorded as a caveat and mitigated with an all-BARE-entries convention (every one of the 31 new agents entries is bare, verified). The `epm:review-reconcile v1` verdict (2026-08-21T19:30:00Z) ruled the demanded comparator redesign to be hardening beyond the registered change set, and downgraded the Codex blocker `family-parity-check-lossy` to a deferred concern — with the standing recommendation that it be filed as its own follow-up. This task is that filing.

Note also why the headline impact claim fell: guard (20)'s collector `_agents_uncovered_readers` refuses to count a bare SPECS token as a disposition, so the specific "Step 5a singleton / Step 10d agents member with every guard green" state is unreachable for agents-family readers. The surviving channel is narrower — comment-then-diverge, and non-agents families.

## Fix direction (implementer to confirm the right shape)

1. Parse BOTH spans with `_family_of_map` (the comment-tolerant parser already in the file) and assert **exact dict equality in both directions** — not a subset/membership relation.
2. Add two negative controls that mutate copied spans and assert the parity helper reports each divergence:
   - a Step-10d-only assignment;
   - a Step-5a assignment carrying a trailing comment whose Step-10d peer is absent.
3. If the all-bare-entry convention is to remain a contract, pin it separately: assert every agents-family assignment matches a full-line bare-assignment regex (do NOT fold this into the equality check — a bare-shape violation and a divergence are different failures and should report differently).

Keep `_family_of_map`'s comment tolerance — it is the correct semantic parser; the defect is that test (10) does not use it.

## Acceptance criteria

1. A Step-10d-only `FAMILY_OF` assignment fails test (10).
2. A commented Step-5a assignment with no Step-10d peer fails test (10).
3. Test (10)'s docstring matches what it proves (today it claims the copies carry the "SAME FAMILY_OF entries" while proving only a filtered Step-5a subset relation).
4. `uv run pytest tests/test_issue_skill_lint_family_sync.py -q` stays green on the unmutated tree.
5. No change to the Step 5a / Step 10d sync fail-safe direction (dirty ⇒ whole-family skip).

## Boundary with sibling tasks

Scoped to test (10)'s COMPARATOR only. #2260 owns the agents-axis family coupling (landed). #2420 owns `tests/test_ensemble_review_cap.py`'s workflow-prose axis. #2374 owns its own per-surface axis. Do not widen into those.

## Provenance

Surfaced by the `reconciler` in `/issue 2260` round 1 as a standing non-blocking recommendation on a PASS verdict (marker `epm:review-reconcile v1`, 2026-08-21T19:30:00Z), grounded on the `codex-code-reviewer` Major at `tests/test_issue_skill_lint_family_sync.py:618`. Deferred concern id: `family-parity-check-lossy`. Fingerprint: (family-of-parity-comparator-lossy, tests/test_issue_skill_lint_family_sync.py).
