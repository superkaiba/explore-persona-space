---
title: 'verify_task_body check 15: resolve ''<path> at git pin <sha>'' prose citations
  against the pinned tree'
kind: infra
tags: []
created_at: '2026-08-23T18:19:17Z'
has_clean_result: false
origin_prompt: 'clean-result-critic #2474 r2 prose follow-up: extend check 15 to catch
  wrong-revision doc pins (15097bee vs 4c07520607)'
workflow: v1
---
## Goal

Extend `scripts/verify_task_body.py` check 15 (artifact/commit-reference verification) to resolve prose claims of the form "`<repo doc path>` at git pin `<sha>`" — for each such claim, list the pinned tree (`git ls-tree <sha> -- <path>` / `git cat-file -e <sha>:<path>`) and FAIL/WARN when the referenced path does not exist at the cited revision.

## Context

Surfaced by the clean-result-critic on #2474 round 2 (2026-08-23): the promoted-body Training slot claimed "recipe copied from the parent methodology doc at git pin `15097bee`", but that revision's tree does not contain `docs/methodology/issue_2379.md` (the doc landed on main at `4c07520607`; the copied values themselves were correct). Check 15's current "committed ... at commit" regex skips exactly this "`<path>` at git pin `<sha>`" phrasing, so a wrong-revision doc citation ships silently. The body-side instance was fixed in-place on #2474; this task mechanizes the gate.

## Acceptance

- Check 15 (or a sibling check) recognizes "at git pin `<sha>`" / "at pin `<sha>`" phrasings adjacent to a repo-relative path and verifies path existence at that revision via a read-only git probe.
- A wrong-revision citation is at least a WARN (FAIL if the check family's existing severity for unresolvable references is FAIL); an unresolvable short SHA is handled without crashing (repo may lack the object in a sparse worktree — degrade to WARN naming the probe failure).
- Regression test: a fixture body carrying a pin whose tree lacks the path trips the check; the corrected pin passes.
