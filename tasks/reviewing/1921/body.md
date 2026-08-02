---
title: 'daily-fix: four CLAUDE.md discipline clauses'
kind: infra
tags:
- wf-fix
- wf-fix-fp:bdd7d6e3976f
- daily-auto-filed
created_at: '2026-07-31T06:57:33Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-30 problem sweep (route 2): four register/discipline
  gaps each cost user corrections today: single-location absence claims shipped into
  durable markers (4 in one session), you-can-do-X spend authorization misread as
  delegate-the-decision, a mapping writeup misattributing the prefix map as the context
  map, and a teammate stood down on a 10-commit-window git probe.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-30 (problem sweep; four independent register/discipline gaps that each cost user corrections today — miner-7 P1/P2/P7, miner-5 P6). Grouped into one task because all four are small clause amendments to CLAUDE.md's existing discipline blocks.

## Goal

Add four clauses to CLAUDE.md's existing discipline sections: (1) absence/nonexistence claims in chat or durable markers require a relocation sweep; (2) "you can do X" after a cost quote is authorization to RUN X, not to adjudicate it; (3) writeups touching the mapping line run the glossary retired-terms check; (4) the teammate durable-state probe names the file-scoped `git log` form.

## Workflow gap

- **Bug observed:** (1) session 0ac15c23 self-admitted "Fourth time today I've scoped a check to one location and reported absence" — one false "floors NOT persisted" claim shipped in a durable pod-safety marker and briefly re-scoped an arm to ~2 GPU-h of regeneration (refuted by a teammate: the floors were on HF under a sibling prefix). (2) The same session read "you can do the 8.5k call" as delegate-the-decision, DECLINED the spend in a durable marker, and reversed only on the user's all-caps "DO THE 8.5K CALL". (3) The #1768 writeup's Motivation misattributed the query-averaged prefix→answer map as the context map — the exact ambiguity `docs/glossary_context_answer_map.md` bans — and the user corrected it twice. (4) An orchestrator stood down a live teammate on a `git log --oneline -10 | grep` probe (a 10-commit window, not a pathspec log); the "missing" commit had landed 40 min earlier — a duplicate implementer was spawned on the false premise.
- **Why it is a workflow gap:** each is a recurring class the existing bullets almost-but-don't cover: verify-before-asserting covers positive claims (the never-run arm exists but does not prescribe the relocation-sweep shape for chat/marker absence claims); no register clause covers spend-authorization phrasing; the glossary check binds nowhere; teammate-coordination clause (b) says probe "git log on the claimed paths" without naming the file-scoped form, and the failure was exactly a window-scoped probe.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -c 'on the claimed paths' CLAUDE.md` → 1 (clause (b) exists and is the amendment site for (4)); `docs/glossary_context_answer_map.md` exists (named by the miner from the same-day docs sweep — unverified hypothesis — verify at plan time: exact glossary path); clauses (1)-(3) have no existing CLAUDE.md text (compose-time reads of § Ad-hoc results summaries / § Interim register found no absence-claim, authorization-phrasing, or glossary clause).

## Proposed change (candidate diff sketch — refine in planning)

(1) § Ad-hoc results summaries "verify before asserting" arm: an absence / not-persisted / never-ran claim requires a relocation sweep (repo-wide grep + HF top-level prefix listing), mirroring workflow-fix-on-bug clause (b) for filed bodies. (2) One register clause: "you can do X" following a cost quote = authorization to run X; when genuinely ambiguous, launch-or-one-line-confirm — never a silent decline written to a durable marker. (3) Interim-writeup register block: mapping-line writeups run the glossary retired-terms check before publication. (4) Teammate-coordination clause (b): the durable-state probe is the file-scoped form `git log <last-known>..origin/main -- <owned paths>`, never a commit-count window.

## Scope / surfaces

- Primary target: `CLAUDE.md` (the § Ad-hoc results summaries block, the interim-register block, § teammate coordination clause (b))

## Constraints / invariants

- Clause amendments only; keep each ≤3 sentences (CLAUDE.md is always-on context — budget matters).

## Provenance

- sha-verify (filing-time, #1467): `0ac15c23` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- fingerprint: bdd7d6e3976f

- workflow_fix_target: CLAUDE.md
- origin: /daily 2026-07-30 miner-7 P1/P2/P7 + miner-5 P6 (sessions 0ac15c23, 75f66748, 1e0de8f8)
