---
title: 'workflow-fix: --check-hub-verify-retry FAIL message invites a lazy-wrap that
  retries nothing (retry_transient around list_repo_tree)'
kind: infra
tags:
- wf-fix
- main-red-adjacent
created_at: '2026-08-30T20:59:25Z'
has_clean_result: false
origin_prompt: 'Surfaced as prose by the #2649 plan-review round-1 Methodology critic:
  retry_transient(lambda: api.list_repo_tree(...)) retries nothing because list_repo_tree
  is a lazy generator; the check''s own FAIL message recommends exactly that shape.
  Routed per .claude/rules/workflow-fix-on-bug.md surfaced-prose auto-file.'
workflow: v1
---
## Goal

`--check-hub-verify-retry`'s FAIL message invites a fix that adds a waiver while
retrying nothing. Close the hole: (a) add one sentence to the FAIL message, and
(b) mechanically flag a `retry_transient(`-wrapped LAZY iterator constructor with
no materializer inside the thunk.

## The trap

`HfApi.list_repo_tree` (huggingface_hub 0.36.2) ends
`for path_info in paginate(...): yield ...`. Calling it performs **zero HTTP** and
returns a generator; every cursor page fires at ITERATION time.
`hub.retry_transient` (= `_retry_upload`,
`src/explore_persona_space/orchestrate/hub.py:1596`, `return fn()`) retries only
the thunk. So:

```python
# looks correct, retries NOTHING
entries = hub.retry_transient(lambda: api.list_repo_tree(...), what="...")
for e in entries:          # <-- every page fires HERE, outside the envelope
    ...
```

The #920 failure class this check exists to prevent — a transient 504 on a cursor
page failing a successful upload's verify leg — survives untouched, now under a
`# HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient` comment asserting the
opposite. The lint goes green (it only checks waiver presence), so the defect
ships silently.

The correct shape materializes the consumption inside the thunk:

```python
entries = hub.retry_transient(lambda: list(api.list_repo_tree(...)), what="...")
```

The repo already knows this. `hub.py::list_repo_entries_complete` L945-947 says
verbatim: "a cursor-page 504 raises DURING iteration, so the comprehension is
materialized inside the retry thunk — #794/#658". The knowledge lives in one
helper's docstring; the check's own FAIL message does not carry it.

## Why the FAIL message is the proximate cause

It currently reads, in part:

> A genuinely-correct raw call (e.g. one you wrap in hub.retry_transient
> yourself) takes the waiver: '# HUB_VERIFY_RETRY_EXEMPT: <reason>'

That sentence is true for an EAGER call and false for a lazy one, and it names no
distinction. An agent reading only the message writes the broken shape and
records a false justification — the "silence the check without stating why each
site is safe" anti-pattern.

## Evidence this is live, not hypothetical

Caught at plan-review on #2649 by the round-1 Methodology critic, BEFORE
implementation. #2649's plan v1 prescribed the broken shape as its default recipe
for all five `list_repo_tree` sites
(`scripts/issue2643_{marker_panel,refusal_panel,sae_map}.py`,
`scripts/issue779_ctxansviz_separate_{pca,umap}_fit.py`). Every one would have
gone green with a false waiver. #2649 v2+ fixes its own recipe; this task fixes
the message and the check so the next round does not re-derive the trap.

## Acceptance criteria

1. `--check-hub-verify-retry`'s FAIL message gains one sentence naming the lazy
   trap and the materializer remedy, e.g. "materialize the iteration inside the
   thunk — a wrap around the lazy constructor retries nothing; see
   `hub.list_repo_entries_complete`".
2. A new mechanical arm FAILs a `retry_transient(` / `_retry_upload(`-wrapped
   `.list_repo_tree(` / `list_repo_files(` with no materializer (`list(`,
   a comprehension bracket, `tuple(`, `sorted(`) between the `lambda:` and the
   call. Waiver form consistent with the existing convention.
3. A pin test reproducing both shapes (broken wrap FAILs, materialized wrap
   PASSes).
4. `uv run pytest tests/test_workflow_lint.py` green; no-flags lint no worse than
   its plan-time baseline.

## Scope notes

- Message-text change interacts with the merge gate's normalized-message-line
  subtraction (the `_hf_routing_file_errors` companion note warns about rewriting
  a FAIL message while main is red anywhere). Sequence this AFTER #2649 lands, or
  account for it.
- The same lazy-wrap reasoning may apply to other generator-returning Hub APIs;
  enumerate before choosing the predicate's breadth.
- Do NOT widen into `--check-live-hf-retry-routing` (leg B): `hf_hub_download` is
  eager, so a plain wrap there is correct.

## Provenance

Surfaced as prose by the Claude `critic` (Methodology lens) during #2649's
plan-review round 1, 2026-08-30, and routed per
`.claude/rules/workflow-fix-on-bug.md` (surfaced-prose follow-ups auto-file).
Filing session #2649 carries no `workflow_fix_target:` line, so the recursion
guard does not apply.

workflow_fix_target: scripts/workflow_lint.py
