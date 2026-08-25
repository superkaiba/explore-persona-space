---
title: 'Absence-claim rule names the duty but not the mechanism: a full-listing +
  client-side prefix filter returns a silent zero on a wrong path where the scoped
  probe 404s'
kind: infra
tags:
- wf-fix
created_at: '2026-08-21T06:49:40Z'
has_clean_result: false
origin_prompt: /issue 2329
workflow: v1
---
# Absence-claim rule names the DUTY but not the MECHANISM: a full-listing + client-side prefix filter returns a silent, confident zero on a wrong path, where the scoped server-side probe 404s

## Goal

CLAUDE.md § "Ad-hoc results summaries" requires that "an absence / not-persisted claim shipped in chat
or ANY durable marker is grounded by a repo-wide grep PLUS a SCOPED listing of the HF prefix's
siblings/parents (`list_repo_tree(path_in_repo=<parent prefix>)`) PLUS the WORKTREES". The duty is
correct and the right call is even named. What the rule does NOT say is WHY the scoped call is the one
named — and without that, the natural implementation (fetch the full listing once, filter locally) is
read as satisfying it while being structurally incapable of the check.

Add the mechanism to the rule text: **an absence check must use a call that can FAIL on a wrong
location.**

## The mechanism, reproduced live (task #2329, 2026-08-21)

Same repo, same nonexistent prefix `issue2329_q35_ladder_decay/`, two calls:

```python
# (1) full listing + client-side filter  ->  rc=0, silent zero
files = api.list_repo_files(repo, repo_type="dataset")
mine  = [f for f in files if f.startswith(pref)]      # -> []
# prints: total files under issue2329_q35_ladder_decay/: 0   *.done.json: 0

# (2) scoped server-side probe            ->  raises
api.list_repo_tree(repo, repo_type="dataset", path_in_repo=pref, recursive=True)
# -> EntryNotFoundError: 404 Client Error
```

Form (1) cannot distinguish "this prefix holds nothing" from "this prefix does not exist", because the
prefix never reaches the server: it is a local string filter over a listing produced for a different
question. It exits 0 and prints an authoritative-looking zero.

Real consequence in #2329: the check was verifying that 72 block `*.done.json` records reached HF,
because a reconciler's discharge of an open BLOCKER (`regen-cap-not-enforced`) rests on those records
travelling with the staged artifact set. The true prefix is
`issue2329_q35rerun/analysis_tensors/ladder/manifests/blocks/` with 72/72 present. Had form (1)
returned first, the round would have concluded the discharge's precondition had failed — from a probe
that never asked the server anything. It was caught only because a scoped probe was run afterwards for
an unrelated reason and 404'd.

Contributing layout trap, worth capturing in the same edit: this issue's store splits across TWO
top-level HF prefixes by artifact class, neither named for the round label —
`issue2329_q35rerun/analysis_tensors/ladder/` (anchors, gates, manifests, margin, va_store, vc_bank)
and `issue2329_q35rerun/raw_completions/ladder/` (anchors, grid). Rollout TEXT is under
`raw_completions/` per upload policy, so "the grid store" is NOT under the tree holding every other
ladder artifact. A prefix guess that looks obviously right is often wrong here.

## Proposed change

In CLAUDE.md § "Ad-hoc results summaries state per-arm provenance…" (the absence-claim sentence) and/or
`.claude/rules/upload-policy.md`, add the mechanism as a short clause:

- An absence check MUST use a call whose error surface distinguishes wrong-path from empty-path —
  `list_repo_tree(path_in_repo=...)` or `get_paths_info` (both 404 on a wrong prefix). A full
  `list_repo_files` listing plus a client-side `startswith`/glob filter is acceptable ONLY for counting
  WITHIN a prefix already proven to exist.
- Where neither is available, require a positive control: assert the PARENT prefix returns non-zero
  before reading a child's zero as meaningful. (One extra line; it is how the #2329 case was actually
  resolved — listing `ladder/`'s children showed no `grid`, which located the real path.)

Keep it to the rule text plus, if cheap, a one-line helper. No behavioural change to any existing
verifier is required — `verify_uploads.py` already uses scoped calls.

## Acceptance criteria

1. The absence-claim rule text names the wrong-path-vs-empty-path distinction and the two acceptable
   call shapes, not just the duty.
2. The `list_repo_files` + client-side-filter form is explicitly scoped to within-a-proven-prefix
   counting.
3. The parent-prefix positive control is stated as the fallback when no scoped call exists.
4. If a helper is added, it raises (never returns 0) on a nonexistent prefix, with a test using a
   nonexistent-prefix fixture asserting the raise — the test must FAIL against a client-side-filter
   implementation.
5. No new red in the no-flags `workflow_lint.py` run or the mapped-test selection.

## Candidate metadata

- target_file: CLAUDE.md (§ absence claims) + `.claude/rules/upload-policy.md`
- fingerprint: absence-check-requires-failing-call-not-client-side-filter
- confidence: high — reproduced live in #2329 with both call forms against the same wrong prefix, one
  returning rc=0 zero and one raising 404

## Provenance

workflow_fix_target: CLAUDE.md

Auto-filed by the `/issue 2329` orchestrator from a self-caught near-miss during the r20 G5 remedy
staging (2026-08-21). Evidence: #2329 `events.jsonl` `epm:progress` v185 (prefix correction) and v186
(the two-call comparison). Kin: this is the ABSENCE-side sibling of the eight hollow-gate findings in
the same round (a green verdict whose comparison set is empty) — same defect shape, opposite direction.
