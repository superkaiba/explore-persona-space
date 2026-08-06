---
title: 'workflow-fix: #1739 follow-up-run marker writes label= not followup_label=,
  reddening test_corpus_replay_all_historical_markers on main'
kind: infra
tags:
- wf-fix
created_at: '2026-08-06T16:45:25Z'
has_clean_result: false
parent_id: 1739
origin_prompt: 'Auto-filed by the #2151 orchestrator from a workflow-fix-candidate
  emitted by the #2151 implementer''s pin-sweep supplement; failure independently
  reproduced on pristine origin/main before filing.'
workflow: v1
---
# workflow-fix: a #1739 follow-up-run marker writes `label=` instead of `followup_label=`, red-lighting `test_corpus_replay_all_historical_markers` on pristine `main`

## Goal

Make `tests/test_workflow_followup_labels.py::test_corpus_replay_all_historical_markers` green on pristine `origin/main` again, and close the producer-side gap that let a malformed `epm:same-issue-followup-run` note token reach the corpus, so no future session's Step 9c gate bounces on marker data it never touched.

## The gap

`test_corpus_replay_all_historical_markers` replays every historical `epm:same-issue-followup-run` marker in the `tasks/` corpus through `parse_followup_note_field(note, "followup_label")` and asserts every one parses, minus an explicit `KNOWN_MALFORMED_RUN_MARKERS` allowlist. One marker fails and is not allowlisted, so the test is RED on pristine `main` — meaning any session whose Step 9c selection (or local pin-sweep union) picks up this file bounces on corpus data unrelated to its own diff.

## Evidence (reproduced 2026-08-06, from the #2151 session)

Reproduced in a scratch worktree detached at `origin/main` — not in an issue worktree, so nothing local contaminates it:

```
uv run pytest tests/test_workflow_followup_labels.py -q
-> 1 failed, 32 passed in 4.57s
E  AssertionError: unparseable run-marker labels: [(1739, '2026-08-05T22:28:00Z')]
   tests/test_workflow_followup_labels.py:1229: AssertionError
```

Root cause read directly off #1739's `events.jsonl`. The task carries two `epm:same-issue-followup-run` markers and they disagree on the token form:

- ts `2026-08-03T07:58:57Z` — note begins `followup_label: new-arm-round; source: user-chat; round: 2; ...` → parses correctly (colon form, correct key).
- ts `2026-08-05T22:28:00Z` — note begins `v1 label=evil-ood-spread-round source=user-chat initiation=manual` → FAILS. Two deviations at once: the key is bare `label` rather than `followup_label`, and the delimiter is `=` with space separation rather than the sibling's `: ` with `; ` separation.

So this is not a parser bug — the parser's contract is satisfied by the 08-03 sibling. It is one malformed note that reached the corpus.

## Deliverables

1. **Green the test.** Either (a) append `(1739, "2026-08-05T22:28:00Z")` to `KNOWN_MALFORMED_RUN_MARKERS` in `tests/test_workflow_followup_labels.py` — the allowlist exists for exactly this class — or (b) repair the note token to `followup_label=evil-ood-spread-round` through the canonical `scripts/task.py` path. Prefer (b) if the marker can be corrected without rewriting history in a way that disturbs #1739's live session; fall back to (a) otherwise. Whichever is chosen, say why in the plan.
2. **Check the producer side.** Determine whether the composer that emitted the 08-05 note writes `label=` SYSTEMATICALLY (a real producer-side gap that will re-offend on the next follow-up round) or whether this was a one-off hand-composed note. The two sibling markers on the same task using different forms is the signal worth chasing. If systematic, fix the composer; if one-off, record that finding so a future reader does not re-investigate.
3. **Consider a pre-post guard.** If deliverable 2 finds a producer-side gap, evaluate whether `task.py post-marker` should validate the note token for `epm:same-issue-followup-run` at post time — a fail-loud reject beats a corpus-replay test discovering it days later. Weigh this against the marker-schema public-contract rule before proposing it.

## Out of scope

Anything about #1739's experimental content, its DV, or its follow-up rounds. This task touches only the marker note token, the test allowlist, and (conditionally) the composer that writes it.

## Provenance

workflow_fix_target: tests/test_workflow_followup_labels.py
Surfaced by: the #2151 implementer's grep-only pin-sweep supplement (2026-08-06), which ran the file because #2151 edited `.claude/rules/llm-judging.md`. Independently reproduced by the #2151 orchestrator in a scratch worktree detached at `origin/main` before filing. ZERO overlap with #2151's diff — #2151 touches no marker data and no workflow test — and the failing file is NOT in #2151's own Step 9c selection, so #2151's gate does not bounce on it.
