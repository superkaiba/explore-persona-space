---
title: 'First-token identity accounting for the #1415 steering shift (parked: duplicates
  the 9a-ter cap-parked item)'
kind: experiment
tags: []
created_at: '2026-07-23T01:00:29Z'
has_clean_result: false
parent_id: 1415
workflow: v1
goal: '[parent Goal verbatim — identical to proposal 1''s Goal field; same-question
  round]'
---
## Goal

[parent Goal verbatim — identical to proposal 1's Goal field; same-question round]


### 2. First-token identity accounting (EXISTING — cap-parked, epm:progress v73; ranked, NOT re-filed) — Type: Diagnostic

**Parent:** #1415
**question_relation:** same
**followup_label:** first-token-identity-accounting
**Goal:** [parent Goal verbatim — identical to proposal 1's Goal field; same-question round]
**Hypothesis:** A large share of the first-bin norm spike is discrete token-identity change: cells with a higher fraction of steered draws whose first token differs from the baseline modal first token show proportionally larger first-bin shift norms (positive rank correlation across the 28×4 cells).
**Falsification:** No correlation between changed-first-token fraction and first-bin norm, or high first-bin norms at near-zero identity-change fractions — the spike would then be a same-token representational shift, strengthening (not just qualifying) the early-kick reading.
**Differs from parent (round):** Exactly ONE thing — a new pure-text reduction over the SAME stored completions (`raw_completions/issue_1415/` + `per_pair_profiles.json` @ `6ceda65f66`); zero new forwards.

**Pre-filled spec (from parent):** all fields same; analysis-only over existing artifacts (both Hub/git-verified above).

**Estimated cost:** 0 GPU-h (free analysis)
**If it works:** Quantifies the token-identity share of the first-bin spike (currently carried only as caveat/limitation 4), tightening the "first ~5 tokens" Takeaways clause.
**If it fails:** See falsification — either outcome refines the same clause; no wasted compute.

**auto_run:** no
**auto_run_reason:** The Step 9a-ter free-analysis round is already consumed on this task (epm:progress v73 cap-park; `epm:free-analysis-followup-run v1` exists for the layer-sweep judging round) — this stays parked for manual pick-up post-promotion or a deliberate cap decision; do NOT double-file.

**cost_class:** free-analysis
**headline_affecting:** no
**est_gpu_hours:** 0

---



## Value critique

Screened `redundant` by the follow-up-critic ensemble (epm:followup-value-critique v2 on #1415, 2026-07-23; single-Claude, Codex quota outage): duplicates the cap-parked free-analysis item recorded on #1415 (`epm:progress` `followup-parked-by-cap followup_ref=first-token identity accounting`, 2026-07-23T00:05:37Z — parked by the 9a-ter one-round cap, not by lack of value). Redundancy is PROPOSAL-LEVEL: the analysis was never run and the question is open. Revive via `task.py set-status <this task> proposed` OR run it as the parked 9a-ter pick-up post-promotion — do NOT double-file.
