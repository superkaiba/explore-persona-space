---
title: verify_uploads.py residue git arm cannot see a suffixed round branch (issue-<N>-<slug>)
kind: infra
tags: []
created_at: '2026-08-24T06:40:27Z'
has_clean_result: false
origin_prompt: 'Surfaced during /issue 823 Step 8 out-root residue check: 25 provably-committed
  files on origin/issue-823-extladder reported as residue because _issue_branch_ref
  resolves only issue-<N> / origin/issue-<N>.'
workflow: v1
---
---
kind: infra
---

# verify_uploads.py residue git arm cannot see a SUFFIXED round branch (`issue-<N>-<slug>`)

## Provenance

workflow_fix_target: scripts/verify_uploads.py

Surfaced by the #823 ext-ladder production round (label `origin-ladder-more-contexts`,
session cmt6ubmuylw53yl0u7juv06tm, 2026-08-24) while running the Step 8 out-root
residue check before pod teardown.

## The bug

`scripts/verify_uploads.py::_issue_branch_ref(issue_num)` resolves only two refs:

```python
for ref in (f"issue-{issue_num}", f"origin/issue-{issue_num}"):
```

A round that works on a SUFFIXED branch — the standard shape for a second/named
round, matching the `pod-<N>-<slug>` pod convention (`.claude/rules/pods.md`) —
commits its eval artifacts to `issue-<N>-<slug>` (here `issue-823-extladder`).
Neither `issue-823` nor `origin/issue-823` exists, so `_issue_branch_ref` returns
None, `_git_tree_candidates_for_issue` falls back to `HEAD` (= `main`) only, and
every artifact the round committed on its own branch is invisible to the git arm
of `check_outroot_residue`.

Realized effect: 25 out-root top-level files (the round's eval JSONs +
`percontext_*.npz`, committed AND pushed at
`eval_results/issue_823/origin-ladder-more-contexts/` on
`origin/issue-823-extladder`, commit `5e52462420`) were reported as
`outroot_residue: FAIL — match no permanent home`, i.e. a FALSE durability gap on
provably-persisted artifacts. Because `pod.py terminate` refuses without a PASS
note carrying the `outroot=` attestation, this blocks teardown of a verified-done
pod — the exact idle-burn class #1662/#2187 exist to prevent. Worked around this
round by 25 explicit evidence-backed `--outroot-exempt` flags plus a per-file
`git ls-tree -r --name-only origin/issue-823-extladder` proof recorded in the
`epm:upload-verification` note; that manual proof is what the tool should do.

## Suggested fix

Resolve the issue-scoped refs by PATTERN rather than exact name — enumerate
`git for-each-ref --format='%(refname:short)' 'refs/heads/issue-<N>*'
'refs/remotes/origin/issue-<N>*'` and keep the refs whose suffix is empty or
begins with `-` (so `issue-823` and `issue-823-extladder` match while `issue-8231`
does not — reuse the digit-boundary discipline already in `issue_token_match`).
Feed every matched ref into `_git_tree_candidates_for_issue`'s existing multi-ref
loop; its `(path, oid)` dedup already handles a candidate appearing on several
refs, and the #2359 content-disambiguation arm is unchanged.

## Acceptance

- A suffixed-branch round's committed out-root files resolve via the git arm with
  no `--outroot-exempt` flags.
- `issue-<N>` exact-name behavior unchanged; `issue-<N><digit>` still excluded.
- Regression test with two refs (`issue-<N>` and `issue-<N>-slug`) asserting both
  contribute candidates, plus a negative case for the digit-boundary.
