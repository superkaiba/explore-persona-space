---
title: 'workflow-fix: local-enumeration exact-set verify must mirror upload_folder''s
  DEFAULT_IGNORE_PATTERNS'
kind: infra
tags: []
created_at: '2026-08-22T06:08:32Z'
has_clean_result: false
parent_id: 2271
origin_prompt: 'Surfaced by the Claude Methodology critic during /issue 2271 plan-v1
  critique (2026-08-22): upload_folder appends DEFAULT_IGNORE_PATTERNS unconditionally
  (hf_api.py:4901), so any exact-set expected-paths verify computed from a LOCAL enumeration
  must mirror those defaults or it hard-fails on a healthy tree. Generalizes beyond
  #2271 to every _upload_folder_filtered-style caller; documentation-only.'
workflow: v1
---
# A local-enumeration exact-set verify must mirror `upload_folder`'s unconditional `DEFAULT_IGNORE_PATTERNS`

## Goal

Add a one-entry workflow-surface warning so the next author of an
`_upload_folder_filtered`-style caller — any helper that computes
`expected_repo_paths` from a LOCAL walk and then verifies that exact set
landed — knows that `huggingface_hub.upload_folder` silently widens the
ignore set on every call, and that omitting those defaults locally converts
a healthy upload into a hard failure.

## The trap

`huggingface_hub.HfApi.upload_folder` runs, unconditionally, on every call:

    ignore_patterns += DEFAULT_IGNORE_PATTERNS

(`.venv/lib/python3.11/site-packages/huggingface_hub/hf_api.py:4901`, verified
verbatim on the pinned `huggingface_hub==0.36.2`.)

`DEFAULT_IGNORE_PATTERNS` is importable from `huggingface_hub.utils` — the same
module `filter_repo_objects` comes from — and equals:

    ['.git', '.git/*', '*/.git', '**/.git/**',
     '.cache/huggingface', '.cache/huggingface/*',
     '*/.cache/huggingface', '**/.cache/huggingface/**']

So a caller that builds its expected set with
`filter_repo_objects(local_rels, allow_patterns=..., ignore_patterns=<its own>)`
and hands the SAME `allow_patterns` to `upload_folder` does NOT get parity: any
local path under `.cache/huggingface/**` (or `.git/**`) is selected LOCALLY,
excluded from the COMMIT, and therefore read as missing by the exact-set
verify. The verify fails, the helper raises, and the tree was healthy.

The failure is loud but its DIRECTION is the inverse of the usual worry: it is
not a silent drop, it is a spurious hard failure on a correct tree — which is
easy to misdiagnose as an upload/quota fault.

## Why it is worth an entry rather than being obvious

Three properties make it survive review:

1. The parity claim reads as self-evidently true — both sides call
   `filter_repo_objects`, so a reviewer confirms the FUNCTION matches and stops.
   The divergence is in the effective PATTERN SET, one `+=` away from the call
   site and in a different file from the caller.
2. A fake/mocked `HfApi` in tests applies only the patterns it is PASSED, so a
   test suite that mocks the Hub boundary is structurally blind to it — the
   mock reproduces the caller's assumption instead of the library's behavior.
3. The triggering local shape is a staging artifact, not something the author
   wrote: an `hf_hub_download(local_dir=...)` pull leaves a
   `.cache/huggingface/download/...` sidecar tree INSIDE the walked directory.
   Nothing in the producing code hints that the consumer's walk will see it.

## Realized, not hypothetical

Found live on this VM during #2271's plan critique: 14 files at

    eval_results/issue_444/raw_completions_hf/.cache/huggingface/download/
      issue444_real_figure_provenance/<...>/raw_completions/*.jsonl.metadata

Executed proof, with `eval_results/issue_444` as the walked tree and the
#2271 plan-v1 pattern set:

    local mirror (allow=class patterns, ignore=TRAINING_STATE only) selects it : True
    upload_folder effective set (ignore + DEFAULT_IGNORE_PATTERNS) selects it  : False
    => divergence: True

Adding `DEFAULT_IGNORE_PATTERNS` to the local mirror drives divergence to
False.

## Scope — what this task does and does NOT do

This task is DOCUMENTATION ONLY: one entry on the workflow surface.

- IN: a short entry under `.claude/rules/upload-policy.md` (the rule that
  already owns Hub-API verification mechanics — the natural home), stating the
  trap, the remedy (include `huggingface_hub.utils.DEFAULT_IGNORE_PATTERNS` in
  any local mirror whose output feeds an exact-set verify), and the
  mocked-boundary blind spot. A `.claude/rules/gotchas.md` cross-pointer is
  acceptable if the implementer judges the trigger better placed there, but do
  not duplicate the full text in both.
- OUT: any change to `src/explore_persona_space/orchestrate/hub.py`. The
  concrete code fix for the one caller that has this bug today is #2271's own
  diff (plan v2, Must-Fix 1) and MUST NOT be duplicated here — filing this
  entry does not license a second edit to the same function.
- OUT: auditing other callers for the same defect. `_upload_folder_filtered`
  is the only in-repo helper doing local-enumeration-then-exact-set-verify as
  of filing, and #2271 fixes its one consumer. If the implementer's read finds
  a SECOND such caller, name it in the entry rather than fixing it here.

## Acceptance

1. The entry exists on the workflow surface, names the `hf_api.py` `+=` site,
   gives the remedy as an importable symbol, and states the mocked-`HfApi`
   blind spot.
2. `uv run python scripts/workflow_lint.py` (no flags) passes.
3. No production code changed — the diff touches `.claude/` only.

## Provenance

workflow_fix_target: .claude/rules/upload-policy.md

Surfaced by the Claude Methodology critic during `/issue 2271` plan-v1
critique (2026-08-22) and orchestrator-verified against
`huggingface_hub==0.36.2` plus the realized `eval_results/issue_444` tree. The
critic raised it as an explicit prose follow-up: the fact generalizes beyond
#2271 to every future `_upload_folder_filtered`-style caller, so it is routed
here rather than buried in one task's plan.
