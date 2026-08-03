---
title: 'workflow-fix: root-commit guard pathspec escape fails under foreign staged
  files'
kind: infra
tags:
- wf-fix
- wf-fix-fp:435658c923d8
- trigger-dense
created_at: '2026-07-31T22:45:37Z'
has_clean_result: false
origin_prompt: 'boundary-impl + orchestrator, #1345 session 2026-07-31: the guard''s
  documented ''pathspec-limited commit is never blocked by foreign staged files''
  escape blocked both sessions'' pathspec commits while a concurrent session''s uncertified
  file was staged.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from prose follow-ups raised on task #1345 (emitters: boundary-impl AND this orchestrator — both hit it independently the same day).

## Goal

Make guard_root_code_commit.sh's documented pathspec escape actually hold: a pathspec-limited `git commit ... -- <own paths>` must never be blocked by a FOREIGN session's staged uncertified files.

## Workflow gap

- **Bug observed:** with a concurrent session's uncertified code file in the shared staged index, pathspec-limited commits were BLOCKED naming the FOREIGN file as the uncertified payload — hit twice this session (2026-07-31: `git commit -m ... -- scripts/issue1901_intrain_companion_figure.py` blocked twice with cert-diag naming the foreign-staged `scripts/issue1345_onpolicy_answers_gen.py`; succeeded only on a later bare retry after the foreign state changed) and independently by boundary-impl in BOTH `-F` and `-m` forms (recovered by polling ~135 s for the foreign commit to land). The hook's own block message says "a pathspec-limited commit is never blocked by foreign staged files: git commit -m "<msg>" -- <your paths>".
- **Why it is a workflow gap:** the guard's documented escape is the ONLY sanctioned concurrent-committer path on the shared root (CLAUDE.md § Concurrent repo-root committers); when it fails, sessions either poll (wasted wall) or reach for EPM_ALLOW_ROOT_CODE_COMMIT=1 (defeats the gate). Suspected: the payload-enumeration path reads the staged index without filtering to the commit's pathspec in some invocation shapes (observed with trailing redirects `> file 2>&1; echo rc=$?` and with `-F`) — root cause for the planner to pin down; do not trust this hypothesis (unverified hypothesis — verify at plan time: the exact argv shapes that defeat the pathspec scoping).
- **Confidence (emitter):** high that the escape fails; medium on mechanism.
- verified-at-filing: `grep -n "never blocked" .claude/hooks/guard_root_code_commit.sh` -> line 1261 (the claim exists); `grep -cn "pathspec" .claude/hooks/guard_root_code_commit.sh` -> 55 hits (scoping code exists but demonstrably does not hold in the observed shapes); landed-fix history 7d: 638093ec4f / c341f3bd59 / 7aeabf972a — none address foreign-staged pathspec scoping (2026-07-31).

## Proposed change (candidate diff sketch — refine in planning)

+ In the payload-enumeration step: when the commit argv carries `--` pathspecs,
+ intersect the candidate payload set with the pathspec match BEFORE cert
+ evaluation (foreign staged paths outside the pathspec are ignored entirely),
+ across ALL argv shapes (-m, -F, trailing redirects). Add a regression test
+ fixture: foreign uncertified staged file + own certified pathspec commit
+ must pass in -m, -F, and redirect-suffixed forms.

## Scope / surfaces

- Primary target: `.claude/hooks/guard_root_code_commit.sh`
- Reproduction evidence in this session's transcript + boundary-impl's report (task #1345 events 2026-07-31).

## Constraints / invariants

- Workflow-surface only. The cert contract (#1620 binding rule) unchanged; only the SCOPING of which staged paths are evaluated changes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/hooks/guard_root_code_commit.sh
- fingerprint: 435658c923d8

Verbatim surfaced prose (boundary-impl, 2026-07-31): "guard_root_code_commit.sh's documented escape did not hold: it says 'a pathspec-limited commit is never blocked by foreign staged files', but with a concurrent issue-1768 session's files in the shared index it blocked my git commit ... -- <my paths> in BOTH the -F and -m forms. I did not unstage their work and did not use the override; I polled ~135 s for their commit to land and then committed cleanly. If that guard is meant to scope to the pathspec, it currently doesn't."
