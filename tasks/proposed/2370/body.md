---
title: 'guard_root_code_commit.sh: pathspec scoping fails to exempt foreign staged
  files (blocks its own remediation form)'
kind: infra
tags: []
created_at: '2026-08-18T11:57:03Z'
has_clean_result: false
origin_prompt: '#2333 Step 9a-quater 2026-08-18: four pathspec-limited docs-only commit
  forms blocked on a foreign staged #2094 test file; override 8440da7124 recorded
  in epm:progress'
workflow: v1
---
# guard_root_code_commit.sh: pathspec scoping fails to exempt foreign staged files — blocks the exact remediation form it prescribes

## Goal

Fix the pathspec-scoping branch of `.claude/hooks/guard_root_code_commit.sh`: a pathspec-limited repo-root commit (`git commit -m "<msg>" -- <own-paths>`) whose pathspec contains NO code payload must not be blocked by a FOREIGN uncertified code file sitting in the shared staged index. The guard's own remediation text states "a pathspec-limited commit is never blocked by foreign staged files: git commit -m "<msg>" -- <your paths> (the guard scopes its check to the pathspec)" — but on 2026-08-18 (task #2333, Step 9a-quater) it blocked FOUR successive forms of a docs-only pathspec commit (`docs/methodology/issue_2333.md`), each flagging the foreign staged `tests/test_issue2094_rev_butler.py` (staged `A`, never committed, by the #2094 session ~3.6 days earlier; cert stale 309538s > 21600s):

1. `cd <root> && for … git add … && for … git commit -F <msgfile> -- <path> …` (compound; arguably fair to reject)
2. `cd <root> && git commit -F <msgfile> -- <path> > redirect; echo rc; tail` (redirect tolerated per #1928, still blocked)
3. `cd <root> && git commit -m "<msg>" -- <path>` — the guard's OWN recommended form, verbatim shape, still blocked
4. `git -C <abs-root> commit -m "<msg>" -- <path>` (no cd prefix), still blocked

Escape used: `EPM_ALLOW_ROOT_CODE_COMMIT=1` on form 3 → commit 8440da7124 landed correctly with ONLY the pathspec file; the foreign staged file was untouched. So git semantics are safe — only the guard's scoping predicate is wrong (or its argv parser never reaches the scoping branch when a `cd <root> &&` prefix or `git -C` form is used, falling through to the whole-index check).

## Repro sketch

At the repo root with any uncertified code file staged by another session: stage a pure-docs file, run the guard's recommended form. Expected: allowed (pathspec carries no code). Observed: BLOCKED naming the foreign file. Suspects: (a) the pathspec-extraction regex not matching when argv is prefixed (`cd X &&` / `git -C X`); (b) the scoping branch checking the staged INDEX set rather than intersecting it with the parsed pathspec.

## Also worth a look

The blocking artifact itself — a 3.6-day-old staged-`A` orphan at the shared root — is #2015-class standing-armer exposure that the root_unstaged_audit (watcher pass 36) does not cover (it keys on worktree `M`/`D`, not index-only `A` entries). Consider extending the audit to long-lived staged-only entries.

## Candidate metadata

- target_file: .claude/hooks/guard_root_code_commit.sh
- fingerprint: guard-root-code-commit-pathspec-scoping-foreign-staged
- confidence: high (four-form repro recorded in #2333 events; override commit 8440da7124 proves the pathspec commit was safe)
