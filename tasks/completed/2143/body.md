---
title: 'daily-fix: bare-API blinded-read recipe as a rule'
kind: infra
tags:
- wf-fix
- wf-fix-fp:ff05e72343e4
- daily-auto-filed
created_at: '2026-08-06T07:23:28Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-04 problem sweep (route 2): three user corrections
  before a blinded read stopped leaking setup facts; the converged bare-API packet
  pattern lives only in one session'
workflow: v1
---
# daily-fix: capture the bare-API blinded-read recipe as a rule — three leakage corrections before the design converged

## Workflow gap

Thomas asked for subagents to qualitatively classify two feature groups "without knowing
anything about them" (#1482, 2026-08-04). The first round leaked SEVEN setup facts to the
readers (the map, error metric, top/bottom-100 selection, which arm, the prediction task,
…) — admitted after "what info did we give to the agents?"; a re-run was ordered; the
same under-blinding recurred for the feature-groups read ("i wanted it to not know
anything about the setup"); Thomas himself then proposed the fix: "can you use the API to
do this so there's no chance of leakage?" — yielding `scripts/issue1482_blind_read_api.py`
(bare-API packets: no filesystem, no repo context, frozen key.json). Three corrections to
converge on a pattern that should be the default for any blinded/unprimed read.

verified-at-filing: the correction sequence is the recovery miner's probed transcript
reads (session 201e2896 rows 6819–6827, 6928, 7916, 8073–8077);
`ls scripts/issue1482_blind_read_api.py` at compose time → the reference implementation
exists (currently untracked at repo root; #2076 tracks its lint routing).

## Proposed change

One short rule (either a new `.claude/rules/blinded-reads.md` + LESSONS index row, or a
bullet in `.claude/rules/llm-judging.md`): blinded/unprimed qualitative reads default to
the bare-API packet recipe — content-only packets sent via the API dispatcher, readers get
no filesystem/repo/tool access, the blinding key is frozen to a file before any read, and
the brief lists what the reader may NOT be told (task, arms, selection rule, metric).
Name `issue1482_blind_read_api.py` as the precedent.

## Provenance

- fingerprint: ff05e72343e4

- workflow_fix_target: .claude/rules/llm-judging.md
- origin: /daily 2026-08-04 recovery sweep — miner 1 P8 (probed rows; user-driven design).
