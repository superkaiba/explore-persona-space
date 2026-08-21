---
title: 'verify_uploads.py outroot-residue: issue-scoped git arm ignores round-suffixed
  follow-up branches (remediated residue can never mechanically PASS)'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-21T04:32:21Z'
has_clean_result: false
origin_prompt: auto-filed by /issue 2329 orchestrator from the q35_ladder_decay upload-verifier's
  workflow-fix-candidate (round-branch refs unenumerated, 2026-08-21)
workflow: v1
---
# verify_uploads.py outroot-residue: the issue-scoped git arm enumerates only EXACT-named `issue-<N>` refs, so residue remediated on a same-issue FOLLOW-UP ROUND BRANCH is structurally invisible — a fully-fixed residue can never mechanically PASS

## Goal

Close a false-FAIL in the `#2187` out-root residue check in `scripts/verify_uploads.py`: its
issue-scoped git arm resolves only the exact ref names `issue-<N>` / `origin/issue-<N>` plus the
main-checkout `HEAD` (`_issue_branch_ref`, ~`scripts/verify_uploads.py:1241`). A same-issue
FOLLOW-UP round works on a SUFFIXED branch — `issue-<N>-<followup_label>`, the standard Step 9b
worktree shape — so a file genuinely committed and pushed there is invisible to the name-set diff,
and the residue is reported as unpersisted no matter how correctly it was remediated.

Net effect: on any follow-up round, the out-root residue leg is **mechanically un-PASSable** once it
has fired once. The verifier can only clear it by stepping outside the tool (reading blobs at the
pushed SHA by hand), which is what happened live below.

## Incident (2026-08-21, task #2329 round `q35_ladder_decay`)

The round's upload-verifier returned FAIL on one blocker: `cap_hit_report_anchors.json` (3,038 B),
written by a `--phase cap_report` run AFTER the terminal bulk upload, so it had no permanent home.
Remediation was ordinary and correct: both name-set hits were committed to the round branch at
`eval_results/issue_2329/q35_ladder_decay/cap_hit/`, commit
`d30951da7a09b05e074ad75764b1ba3de2577664`, pushed `d0c07f98a2..d30951da7a` on
`origin/issue-2329-q35-ladder-decay`, sha256 re-checked against the pod on both files.

The mechanical re-run of `verify_uploads.py` STILL reported both files as residue — from BOTH the
repo-root and worktree cwds (it pins `cwd=_verifier_repo_root()`), because
`origin/issue-2329-q35-ladder-decay` is not `origin/issue-2329` and never enters the ref set. The
re-verification PASSed only on out-of-band evidence: the verifier fetched the two blobs at the pushed
SHA, confirmed `d30951da7a` is an ancestor of the round branch, and matched both blob sha256s against
its own round-1 pod-side hashes (`533ca33e…`, `a9154594…`). All 378 out-root files were then
accounted for — 376 across 9 HF prefixes, 2 via the pushed commit.

A gate whose PASS requires a human to work around the tool is not a gate.

## Distinct from #2359 (completed) — OPPOSITE failure direction, same arm

#2359 fixed a false-**OK**: basename matching via the git arm let a SIBLING LEG's committed
same-named file cover this leg's unpersisted file (`upload_done.json` across #2333 legs A/B, different
bytes). Its remedy was content-disambiguation by blob sha — and that remedy WORKED here: the second
name-set hit in this incident was correctly content-disambiguated as covered.

This task is the false-**FAIL** twin: not "the match is too permissive" but "the ref set is too
narrow". #2359 made matches trustworthy; it did not widen which refs are searched. Both live in the
same `_issue_branch_ref` neighbourhood, so whoever takes this should read #2359's implementation
first and preserve its content-disambiguation unchanged.

## Proposed fix (prescribed by the reporting verifier)

Enumerate `issue-<N>*`-prefixed refs — local and `origin` — via `git for-each-ref`, OR add a
repeatable `--git-ref` flag that the round's verification passes for its round branch. Keep the
existing basename + blob-OID content-disambiguation from #2359 exactly as-is.

Prefer ref ENUMERATION over a flag if both are viable: a flag has to be remembered at every call site
(and the #2329 incident shows the call site is often composed by hand under time pressure), whereas
enumeration is correct by default. If a flag is added anyway, it should be additive to enumeration,
not a replacement.

## Acceptance criteria

1. A file committed on `issue-<N>-<label>` and pushed is recognized as persisted by the residue check
   — reproduce with the #2329 fixture: `cap_hit_report_anchors.json` at `d30951da7a` on
   `origin/issue-2329-q35-ladder-decay`.
2. #2359's cross-leg false-OK stays FIXED: a sibling leg's same-named file with DIFFERENT bytes must
   still FAIL, not be laundered by the wider ref set. Widening refs increases the basename-collision
   surface, so this is the regression most at risk — test it explicitly with two legs of one issue.
3. The check behaves identically from the repo-root cwd and from an issue worktree cwd.
4. An unpersisted file with no committed counterpart anywhere still FAILs (the check must not become
   vacuous — a residue leg that cannot fail is worse than one that over-fires).
5. Tests failing before and passing after; no new red in the no-flags `workflow_lint.py` run or the
   mapped-test selection.

## Candidate metadata

- target_file: scripts/verify_uploads.py (the #2187 outroot-residue arm, `_issue_branch_ref`)
- fingerprint: outroot-residue-round-branch-refs-unenumerated
- confidence: high — reproduced live in #2329 r2 from two cwds, with the commit SHA, both blob
  sha256s, and the passing/failing arms all recorded

## Provenance

workflow_fix_target: scripts/verify_uploads.py

Auto-filed by the `/issue 2329` orchestrator from the round's upload-verifier
`workflow-fix-candidate v1` block (2026-08-21). Evidence: #2329 `events.jsonl`
`epm:upload-verification` v3 (FAIL) and v4 (PASS on blob evidence, with the caveat recorded in the
marker itself); verifier tool outputs `/tmp/verify_A2.json`, `/tmp/verify_A3.json`. Sibling:
#2359 (completed, opposite direction on the same arm).

## Kinship: this may be one shared bug wearing three tool names (added at filing)

The file-time dedup advisory surfaced two siblings that are NOT duplicates of this task (different
target files, different code) but share its ROOT ASSUMPTION — workflow tooling that assumes a single
canonical branch and misbehaves on the round-branch / worktree topology that Step 9b and
`new_worktree.sh` make the normal shape:

- **#865** (`on_hold`) — "Step 9c selector diffs main checkout, blind to worktree branches".
- **#2320** (completed) — "Step 10d Guard 3 ON_MAINLINE uses first-parent reachability, so a
  sibling's merge-form landing false-flags later branches UNSAFE".
- **this task** — the residue check's ref set omits `issue-<N>-<label>`.

Three tools, one wrong premise. Whoever takes this should at least LOOK at whether a single shared
"resolve the refs that belong to issue N" helper (local + origin, prefix-aware, worktree-aware) would
serve all three, rather than patching ref logic a third time in a third place. That is a suggestion to
evaluate, not a prescription: if the three call sites need genuinely different semantics, say so and
fix this one narrowly. But #2320 is already CLOSED and #865 is still open, which is exactly the
pattern of a class being fixed one instance at a time.
