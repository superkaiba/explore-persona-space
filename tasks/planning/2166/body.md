---
title: '#2166 workflow-fix: repoint 5 stale CLAUDE.md prose · planning · █░░░░ 20%'
kind: infra
tags:
- wf-fix
- urgent-main-red
created_at: '2026-08-07T09:01:59Z'
has_clean_result: false
origin_prompt: 'Surfaced as a workflow-fix-candidate by the #2164 implementer during
  its gate-scope run; verified independently red on origin/main 68fbf9bf3e by the
  #2164 orchestrator before filing. Distinct target_files from #2157/#2159/#2160.'
workflow: v1
---
## Overview / Motivation

Five test functions across four files pin CLAUDE.md **prose** that the 2026-08
compaction deliberately relocated into `.claude/rules/*`. They are red on
`origin/main` **right now** (reproduced 2026-08-07 at `68fbf9bf3e`, repo-root
checkout, 5 failed / 10 passed in 1.23s), so every intervening session's Step 9c
gate must re-classify them as known-red.

Same class as #2157 → #2159/#2160, but **different target files** — that pair
covered `tests/test_issue_skill_trigger_dense_tag_adoption.py` only. Dedup key
is `(target_file, fingerprint)`, and none of the four files below was touched by
that fix.

Surfaced by the #2164 implementer during its gate-scope run, then verified
independently by the #2164 orchestrator against current `origin/main` before
filing (the failures are real and are NOT caused by #2164's diff).

## Goal

Make the five pins green on `origin/main` by re-pointing each at the rule file
that now carries the relocated prose verbatim, preserving every ordering and
window assertion. Do not restore prose to CLAUDE.md — the compaction moved it
deliberately to cut always-on context load.

## Failing tests (verified red on main `68fbf9bf3e`)

| Test | Relocated prose now lives in |
|---|---|
| `tests/test_router.py::test_claude_md_compute_backends_section_matches_656_contract` | `.claude/rules/compute-backends.md` |
| `tests/test_issue_skill_file_only_verdict_post.py::test_claude_md_points_at_file_only_path` | `.claude/rules/codex-ensemble-review.md` |
| `tests/test_issue_skill_neutral_gate_vocab_brief.py::test_claude_md_rung_e_neutral_gate_vocab` | `.claude/rules/context-hygiene.md` / `trigger-dense-review.md` |
| `tests/test_issue_skill_neutral_gate_vocab_brief.py::test_claude_md_rung_e_steering_vocab` | `.claude/rules/context-hygiene.md` / `trigger-dense-review.md` |
| `tests/test_suffixed_pod_completion_teardown_pin.py::test_claude_md_carries_completion_side_teardown_contract` | `.claude/rules/pods.md` (the `count >= 2` two-site assertion — CLAUDE.md now has 1 of the 2 `Completion-side teardown` sites; the other moved) |

## Workflow gap

The compaction train landed without re-running the workflow-invariant pin family
against the final merged tree, so stale pin targets shipped. #2159 fixed one
file; these four were not swept. A post-compaction sweep of the whole pin family
would have caught all of them at once.

## Acceptance criteria

- All five named tests pass on a pristine `origin/main` checkout.
- Each re-pointed assertion reads the rule file that verifiably contains the
  prose **verbatim** — confirm the string is present there before re-pointing;
  if a string is genuinely absent everywhere, that is a compaction content loss
  and must be reported, not papered over by weakening the assertion.
- Ordering / window / count assertions preserved. The teardown pin's
  `count >= 2` needs a decision: either both sites exist across CLAUDE.md + the
  rule file (assert across both) or the contract genuinely has one site now
  (adjust the count with a stated reason).
- No prose is moved back into CLAUDE.md.

## Notes

Also worth checking in the same sweep, since the compaction is the common cause:
whether any other `tests/test_*claude_md*` / prose-pin test is red on main.
`uv run python scripts/step9c_baseline.py status` reported `failing_tests: 1`
after a fresh refresh while these 5 were red — the ledger's universe appears not
to cover this pin family, which is itself worth a look (a gate that cannot see a
main-red test cannot classify it).
