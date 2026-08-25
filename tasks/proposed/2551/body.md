---
title: Codex ensemble dispatch grants write access to five read-only review sites
  (--no-write missing from canonical snippet)
kind: infra
tags: []
created_at: '2026-08-24T20:15:08Z'
has_clean_result: false
origin_prompt: 'Found by the #2544 orchestrator: codex_task.py defaults to --write
  while all three codex-critic composers returned ''Codex write mode: false (read-only
  critic)''; the canonical dispatch snippet in codex-ensemble-review.md omits --no-write.'
workflow: v1
---
# Codex ensemble dispatch grants WRITE access to five read-only review sites

## Problem

`scripts/codex_task.py` defaults to `--write` (grant Codex file-mutation access):

    scripts/codex_task.py:1907  # Default for --write is True (grant write) unless --no-write was passed.
    scripts/codex_task.py:1716  "--write",   / :1722  "--no-write",   / :640  cmd.append("--write")

The canonical orchestrator dispatch snippet in `.claude/rules/codex-ensemble-review.md` does NOT
pass `--no-write`:

    uv run python scripts/codex_task.py --issue <N> --effort <high|xhigh> \
      --prompt-file /tmp/codex-prompt-issue-<N>.md --output-file /tmp/codex-output-issue-<N>.md

All FIVE doubled review sites are read-only reviewer roles by definition — `critic`,
`code-reviewer`, `interpretation-critic`, `clean-result-critic`, `follow-up-critic`. The
`codex-*` composers already know this and RETURN it in their dispatch config; every one of the
three composers on #2544 returned the line:

    Codex write mode: false (read-only critic)

The orchestrator has no documented flag to honor that with, so the declared read-only contract is
silently dropped at dispatch and Codex reviews its target with write access to the repo.

Write access is not needed for the output path: the wrapper captures Codex stdout into
`--output-file` itself (`codex_task.py` docstring: "Write Codex stdout to ``--output-file``").

## Observed

#2544 Phase 2, 2026-08-24: three Codex critic jobs dispatched from the canonical snippet, all
three marker-recorded as `write=True` against composer configs declaring write mode false. No
mutation occurred — the completed alternatives job left plan v4 byte-identical and the working
tree clean on every reviewed source — so this is a latent exposure, not a live incident. The
prompts' own read-discipline headers appear to be doing the work that a flag should do.

## Why it matters

A reviewer with write access can "helpfully" edit the artifact it is reviewing. At the plan-critic
site that means the plan under review; at the code-reviewer site, the diff under review. Either
silently destroys the review's independence — the property the whole Claude+Codex ensemble exists
to provide — and at the plan site it could mutate a persisted `plans/v<K>.md` out from under the
Goal-currency and Edit-success gates, which compare against what they last wrote.

## Proposed fix

1. Add `--no-write` to the canonical dispatch snippet in `.claude/rules/codex-ensemble-review.md`,
   and to any sibling copies of the snippet in the `/issue` step bodies and agent specs (grep for
   `codex_task.py --issue`).
2. Consider inverting the helper default for the reviewer path. Safest minimal form: keep the
   `--write` default for any non-review Codex use, but have the five review sites pass
   `--no-write` explicitly; a stronger form is a `--role reviewer` that implies read-only.
3. Optional hardening: have `codex_task.py` refuse `--write` when the prompt file contains the
   `epm:*-critique-codex` / `epm:code-review-codex` marker tags, since those tags identify a
   review dispatch unambiguously.

## Acceptance

- The canonical snippet and every in-repo copy pass `--no-write`.
- A dispatched review job's `epm:codex-task-spawned` note records `write=False`.
- A test pins the snippet so the flag cannot silently regress out again (the same shape as the
  existing region-anchored surface pins in `workflow_lint.py`).

## Provenance

Found by the #2544 orchestrator while auditing why spawn markers read `write=True` against
composer configs declaring read-only. Not a #2544 experiment defect; a workflow-surface gap that
affects every Codex ensemble round fleet-wide.
