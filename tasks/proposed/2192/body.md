---
title: 'workflow_lint: no gate detects merge-conflict-marker residue in tracked workflow-surface
  files'
kind: infra
tags:
- wf-fix
- conflict-marker-lint
created_at: '2026-08-08T02:54:07Z'
has_clean_result: false
origin_prompt: '#2189 round-1 code-reviewer workflow-fix-candidate v1: a diff3 conflict-base
  marker committed to .claude/rules/code-style.md passed the no-flags workflow_lint
  run, the gotchas-size check, and the union-conservation check.'
workflow: v1
---
# Goal

No gate in the tree detects merge-conflict-marker residue in tracked
workflow-surface files. Add a no-flags-bundled `workflow_lint.py` check that
FAILs on conflict-marker lines in tracked `.claude/**/*.md`, `*.py`, and
`CLAUDE.md`.

## Evidence — a real marker reached a committed workflow-surface file

During #2189, merge commit `14cd4e4211` (a routine `origin/main` merge on the
`issue-2189` branch, resolving a `code-style.md` collision with #2188) left the
literal diff3 conflict-base marker

```
||||||| 640f206892
```

as the **last line of `.claude/rules/code-style.md`** — an always-loaded-on-trigger
project rule file.

It passed every gate the round ran:

- `scripts/workflow_lint.py` (no flags) — rc=0, `workflow_lint: PASS`, zero FAILs.
- `check_gotchas_size` — the round's own size gate; wrong file, and size-only anyway.
- The round's `#N` + backticked-identifier **union-conservation** check — it
  compares token multisets, so an ADDED junk line is structurally invisible to it.
- `ruff` — the file is markdown.
- `tests/test_workflow_lint_gotchas_size.py` + `tests/test_consolidate_lessons.py`
  (27 tests) — unrelated surface.

It was caught only by the round-1 `code-reviewer` reading the diff by eye and
sweeping for all four marker forms. That is not a repeatable gate.

## Why this recurs

Merge commits on `issue-<N>` branches are routine — the `/issue` Step 10d flow
merges `origin/main` into the branch before landing, and concurrent sessions
edit the same `.claude/rules/*.md` files constantly (#2189 collided with both
#2185 and #2188 on the same day). A mis-resolved diff3 merge that drops a
marker into a rule file is silent, survives to `main`, and degrades an
always-on instruction surface.

Note the diff3 base marker (`|||||||`) is the easiest form to miss by eye: it
is not the familiar `<<<<<<<` / `>>>>>>>` pair, and `merge.conflictStyle=diff3`
emits it only in the middle of a hunk.

## Proposed fix

Add a check flagging lines matching

```
^(<{7}|\|{7}|={7}|>{7})( |$)
```

in tracked `.claude/**/*.md`, `*.py`, and `CLAUDE.md`, bundled into the
no-flags default `workflow_lint.py` run so it fires on the Step 9c gate and the
inline payload lint gate without an opt-in flag. (Equivalently: wire
`git diff --check`-style detection.)

Design notes for the implementer to settle:

- The trailing `( |$)` is load-bearing — it avoids matching legitimate markdown
  (a `=======` horizontal rule, a `>>>>>>>` blockquote chain) and Python
  comment banners. Verify against the live tree before pinning: the sweep run
  during #2189 returned exactly one match repo-wide, so a correctly-anchored
  pattern should be clean at baseline.
- Decide whether this file set should extend to `tests/**` and `scripts/**`
  (both are `*.py`, so they are already covered by the `*.py` glob — confirm
  that is intended rather than incidental).
- A rule file legitimately DOCUMENTING conflict markers (this body is an
  example, and a future gotchas entry might be) needs an escape — prefer
  requiring the marker be fenced, or an explicit allowlist, over weakening the
  pattern.

## Provenance

Filed by the #2189 orchestrator from the round-1 `code-reviewer`
`workflow-fix-candidate v1` block (`epm:code-review v1` on #2189, Major
finding). The marker itself was fixed in #2189 commit `2ddec80aad`; this task
is the missing GATE, not the fix.
