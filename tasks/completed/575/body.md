---
title: Spot-check the awaiting_promotion backlog with the new verify_task_body URL-existence
  check
kind: analysis
tags:
- agent-ok
created_at: '2026-06-11T02:57:36Z'
has_clean_result: false
---
Promoted #507 shipped a hero figure that was both wrong and 404'd (Thomas: 'The first plot in it is broken'), and dead repro URLs were found in 8 parked tasks; 7 were repaired by parallel agents and a verify_task_body.py URL-existence check landed same day.
Action: run the new URL-existence check across the remaining awaiting_promotion backlog and repair any dead figures or repro URLs before promotion.
source: logs/daily/2026-06-09.md, approved by Thomas 2026-06-10 ('Apply these')

## Result (2026-06-12 sweep)

**All 50 awaiting_promotion bodies pass the verifier's URL-existence checks (4b figure URLs / 8 repro URL permanence / 8b repro artifact URLs): 1264 URLs probed (328 figure + 936 repro), 0 unverified, 0 dead, 0 repairs needed.** This is a point-in-time verifier sweep, not a claim that the backlog is "URL-clean" — fenced URLs and images outside `## TL;DR` are beyond the verifier's scan scope.

- **Census:** plan-time 50-ID list ∪ run-start listing, identical sets, frozen 2026-06-12T20:30:53Z; re-listed pre-report, no arrivals/departures.
- **One out-of-scope FAIL:** #585 fails check `Reproducibility Context provenance row` (missing `**Context:**` row; origin data exists in its `original-body.md`). All its URL checks PASS. This will block #585's own promotion-time verifier gate — flagged via an `epm:progress` marker on #585. 37 tasks carry Context-row WARNs and 6 carry `goal:` frontmatter WARNs (recorded, out of scope).
- **Repair path was vacuous:** no body was edited; the claim-marker / diff-discipline / CAS / remote-reachability / provenance gates from plan v2 had nothing to fire on.
- **Verification:** independently re-derived by the code-review ensemble (Claude PASS, Codex CONCERNS with no blockers): zero in-scope `[FAIL]` lines across all 50 raw logs, zero `unverified` notes, URL counts re-summed digit-for-digit.

**Artifacts** (committed at `bb6d0bd55`, table-literal fix `46ea49469`): `tasks/<status>/575/artifacts/` — `verdict_table.md` (50-row per-task verdicts), `classification.tsv`, `verifier_logs/<N>.log` ×50, `census.txt`, `sweep_exit_codes.txt`, `repairs/provenance.tsv` (header-only, zero rewrites). Plan: `plans/plan.md` (v1, critic-ensemble-revised).
