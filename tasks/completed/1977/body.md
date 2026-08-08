---
title: 'daily-fix: exempt path-limited git add --all in guard'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1b04a1eed533
- daily-auto-filed
- trigger-dense
created_at: '2026-08-01T07:10:17Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-31 problem sweep (route 2): guard_root_code_commit.sh
  blocks `git add --all -- <explicit paths>` (token arm at line ~678 latches unconditionally)
  though the pathspec-limited form has a classifiable landing set — 2 recomposes (#1895,
  #1945).'
workflow: v1
---
# daily-fix: exempt path-limited git add --all in guard

## Overview / Motivation

Auto-filed by the /daily 2026-07-31 problem sweep (CONSOLIDATED L3; miner-2:P13(part), miner-5:P8). Sources: sessions db531944 (#1895) and 75fa9f67 (#1945) — `guard_root_code_commit.sh` blocked path-limited `git add --all -- <paths>` compositions (a classifiable landing set — stages only under the named paths, including deletions, exactly the status-move-residue use case), costing one recompose each.

## Goal

Exempt `git add --all` from the guard's blanket-staging block when it is followed by `--` plus an explicit pathspec, since the landing set is then classifiable at hook time.

## Workflow gap

- **Bug observed:** Two sessions composing `git add --all -- <explicit paths>` (to sweep status-move residue including deletions under known paths) were blocked by the root-code-commit guard's `add -A|.|--all` arm and had to recompose with plain-path staging.
- **Why it is a workflow gap:** In `.claude/hooks/guard_root_code_commit.sh`, the masked-token scan (line ~678: `add:-A | add:--all | add:.) add_all_chained=1 ;;`) latches `add_all_chained` on the bare token, with no arm checking for a following `--` + explicit pathspec; the block fires at line ~1122 ("the landing set cannot be classified at hook time"). But `git add --all -- scripts/foo tasks/bar` HAS a classifiable landing set — the pathspec bounds it — so the block's own rationale does not apply to the path-limited form. Cost is low (one recompose each) but recurring, and the plain-path equivalent (`git add -- <paths>`) does not stage deletions-only under a path the same ergonomic way.
- **Confidence (emitter):** high (guard arm read in context; both blocks probed by the miners)
- verified-at-filing: `grep -n -- "--all" .claude/hooks/guard_root_code_commit.sh` → 3 relevant hits: the header comment (line 48), the token-scan arm (line 678, context read lines 660-700 — no pathspec-following check exists in the arm or after it), and the commit-flag arm (line 727, different clause). Block message confirmed at line 1122 (presence claims; context read binds — the arm latches unconditionally). `git log --oneline --since='7 days ago' -- .claude/hooks/guard_root_code_commit.sh` → 3 commits (`638093ec4f` cert-retry settle pass, `c341f3bd59` re-stage note, `7aeabf972a` cd-latch diagnostics) — none adds a pathspec exemption; no landed fix (2026-08-01 compose time).

## Proposed change (candidate diff sketch — refine in planning)

```
.claude/hooks/guard_root_code_commit.sh, token scan (~line 678):
- add:-A | add:--all | add:.) add_all_chained=1 ;;
+ add:-A | add:--all | add:.)
+   # #1895/#1945: `--all` followed by `--` + explicit pathspec has a
+   # classifiable landing set — track and only latch add_all_chained when
+   # NO `--`-anchored pathspec follows in the same add clause.
+   add_all_pending=1 ;;
+ (per-clause post-scan: add_all_pending && no `-- <pathspec>` tokens seen
+  in the clause ⇒ add_all_chained=1; pathspec-limited form additionally
+  feeds its paths into the existing text_paths / Layer-2 scoped staged read
+  so the payload classification still runs on them.)
```
(The masked-token scan is clause-scoped and order-aware enough to support this — the #1620 second token pass already collects per-clause pathspec candidates; reuse that machinery rather than a new parser.)

## Scope / surfaces

- Primary target: `.claude/hooks/guard_root_code_commit.sh`
- Secondary: the guard's test file (`tests/` — locate the hook's pin tests at plan time; add: path-limited `--all` passes + still classifies the staged payload, bare `--all` still blocks)
- Grep before editing: `grep -rn 'add_all_chained' .claude/hooks/ tests/` and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- FAIL-CLOSED bias preserved: any ambiguity (masked/quote-bearing pathspec tokens, multiple clauses, cd-latch uncertainty) keeps the block — the exemption fires only on an unambiguous `--`-anchored explicit pathspec in the same clause.
- The staged-payload classification (inline payload lint gate routing) must still see the path-limited form's paths — the exemption must not open an unclassified staging channel.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: 1b04a1eed533

- workflow_fix_target: .claude/hooks/guard_root_code_commit.sh
- fingerprint: (driver-computed; tag authoritative)

Origin: CONSOLIDATED L3 (miner-2:P13(part), miner-5:P8), /daily 2026-07-31 — "guard_root_code_commit flags path-limited `git add --all -- <paths>` (classifiable landing set) — 2 recomposes" (sessions db531944/#1895, 75fa9f67/#1945).
