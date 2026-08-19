---
title: '#1689: fold user-slot-recapture readout into body + correct refuted tier-1-real-data
  claim'
kind: infra
tags: []
created_at: '2026-08-03T06:43:01Z'
has_clean_result: false
origin_prompt: run additional analyses/fits/experiments if necessary (user chat, 2026-08-03);
  inline round surfaced the refuted LMSYS-provenance claim publicly - body correction
  must land
workflow: v1
---
## Overview / Motivation

Filed by the user-chat inline round of 2026-08-03 (results-summary writeup) per the inline record-integrity duty: a promoted-body claim on #1689 is refuted by that issue's own recapture round, and the readout fold that would correct it has sat deferred since 2026-07-31.

## Goal

Fold the #1689 `user-slot-recapture` round into the promoted body and correct the refuted provenance claim.

## Refuted claim + evidence

- #1689 promoted body (result "Real and simulated user turns differ by an answer-side transform") states: "the LMSYS arm is tier-1 real data; the haiku arm is tier-3 LLM-simulated."
- Refuting evidence: the two-turn LMSYS corpus has no u2 field; `scripts/issue1689_render_conditions.py:309` falls back to the constant string "Can you say a bit more about that?" (one distinct u2 sha256 across 2,114 rendered rows). Recorded by the recapture round itself as `realized_defects_fixed.lmsys_is_constant_u2` in `eval_results/issue_1689/user_slot_recapture/summary.json` (committed 948fa8c6be from the HF mirror `issue1689_speaker_lattice/user_slot_recapture/eval_mirror/`).
- verified-at-filing: `uv run python -c "json.load(open('eval_results/issue_1689/user_slot_recapture/summary.json'))['realized_defects_fixed']"` reads the lmsys_is_constant_u2 entry; corpus row keys verified to lack u2 (`data/issue_1689/user_slot_probe/issue1689_speaker_lattice/corpus/two_turn_lmsys.shard00.jsonl`) (2026-08-03).

## Scope

- Fold the recapture numbers (per_unit_r2 / grid_r2 / cross_role_transfer / story_label_effect / bridge_comparisons) into the #1689 body via the same-issue analyzer pass; render round figures; correct the tier-1-real-data line (prose correction — classification untouched).
- Interim state: the refutation is stated publicly in docs/results_summaries/2026-08-02-framing-character-user-turn-map-transfer-filled.md; three recapture figures exist under figures/results_summaries/framing_character_user_turn/.

## Provenance

- Origin: inline round completion notes on #1689 (epm:progress v196) + the recapture round's own deferral marker.
