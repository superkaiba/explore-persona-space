---
title: 'Sub-1-epoch max_steps-resolved grid {5,10,18,30,60,120} at lr=5e-6, r=32 (early-trajectory
  diagnostic for #533 E=1 sign reversal)'
kind: experiment
tags: []
created_at: '2026-06-10T05:26:40Z'
has_clean_result: false
parent_id: 533
goal: 'Test whether the E=1 negative-sign role-vs-system paired-d pattern in #533
  reflects a stable mechanism-bearing read or an early-trajectory optimization-ordering
  artifact, by training the same grid at lr=5e-6, r=32 indexed on max_steps in {5,
  10, 18, 30, 60, 120} so the sub-1-epoch trajectory is sampled densely.'
---
## Goal

Test whether the E=1 negative-sign role-vs-system paired-d pattern in #533 reflects a stable mechanism-bearing read or an early-trajectory optimization-ordering artifact, by training the same grid at lr=5e-6, r=32 indexed on max_steps in {5, 10, 18, 30, 60, 120} so the sub-1-epoch trajectory is sampled densely.


**Parent:** #533
**Goal:** Test whether the E=1 negative-sign role-vs-system paired-d pattern in #533 reflects a stable mechanism-bearing read or an early-trajectory optimization-ordering artifact, by training the same role-vs-system grid at lr=5e-6, r=32 but indexing on `max_steps ∈ {5, 10, 18, 30, 60, 120}` instead of {1, 2, 3, 5} epochs, so the trajectory between E=0 and E=1 is sampled densely enough to see whether the negative E=1 d is the floor of a U-shape that crosses zero or the start of a monotone descent the {1,2,3,5} grid blurred.
**Hypothesis:** If the E=1 negative d is mechanism-bearing (role's wrong-slot marker log P is genuinely higher than system's at non-saturated training), per-persona d's should stay negative across a wide max_steps range below E=1 (~18 steps) and gradually approach zero from below as the implant matures. If it's an early-trajectory ordering artifact (role-arm persona-encoding tokens momentarily out-pace system-arm filler tokens before contrastive negatives shape the wrong-slot down), per-persona d's should be near zero or positive at max_steps ∈ {5, 10}, swing negative around the {18, 30}-step zone, and return to the saturated +1.46 region by max_steps 120. The early-trajectory alternative was flagged independently by both the Claude (Round 1) and Codex (Round 1) interp critics as untested by #533's design.
**Falsification:** Per-persona d's across max_steps ∈ {5, 10, 18, 30, 60, 120} on either persona never cross zero AND stay within a 0.5 nat band of the E=1 value (≈ −2 nats on villain × padded, ≈ −0.7 on pirate × plain) — consistent with a stable role-vs-system gap rather than a trajectory swing. (The hypothesis is the opposite — that the d is non-monotone in training count.)
**Differs from parent:** Epoch indexing → max_steps indexing with explicit sub-1-epoch coverage. {1, 2, 3, 5} epochs replaced by `max_steps ∈ {5, 10, 18, 30, 60, 120}` (where ~18 steps ≈ E=1 at batch 4 × grad_accum 4 × 300 positives). Everything else byte-identical to #533.

**Pre-filled spec (from parent):**
- Model: `Qwen/Qwen2.5-7B-Instruct` (same as #533)
- Data: REUSED unchanged from `superkaiba1/explore-persona-space-data/issue464_role_vs_system/R_canon/` at pinned revision `dc0b171f117d3b325695954a4de25deac3468502` (same as #533)
- Seeds: {42, 137, 1337, 7, 21} (same as #533)
- Eval: vLLM teacher-forced `prompt_logprobs=1` at the post-R slot (R = base-model greedy), 50 held-out questions, 3 eval encodings per cell (own / wrong / bare-assistant) (same as #533)
- Config: same as #533 EXCEPT: **`max_steps ∈ {5, 10, 18, 30, 60, 120}` replaces `epochs ∈ {1, 2, 3, 5}`; lr=5e-6 retained; LoRA r=32 / α=64 retained; marker-only loss + contrastive negatives composition retained; `marker_band_stop=False` retained.**
- Cell count: 3 arms × 2 personas × 5 seeds × 6 max_steps = 180 cells (50% more than #533's 120).

**Estimated cost:** ~22 GPU-hours on 4× H100 (`ft-7b` pod intent; 180 cells / 120 cells × 18 h ≈ 27 h linear scaling but the average max_steps value is ~40 vs ~2.75 epochs ≈ 50 steps in #533, so per-cell wall is slightly lower; net ~22 GPU-h).
**If it works:** Either resolves the E=1 sign as a stable mechanism-bearing read (d's monotone or stable across sub-1-epoch max_steps) or exposes it as an early-trajectory ordering artifact (d's swing through zero between max_steps ∈ {5, 60}). Closes a named alternative explanation from both round-1 interp critics that #533 explicitly could not test. Also re-uses the existing anchor-selector with the same `[-10, -5]` band — the dense max_steps grid might happen to land all three arms simultaneously inside the band on some cell, in which case the per-persona paired bootstrap also fires and we get the headline test for free.
**If it fails:** Closes the early-trajectory alternative — d's are stable across the sub-1-epoch trajectory, so the E=1 sign IS the read at this rank/lr regardless of step count. Strengthens the case for the proposal 1 / 3 rank reduction as the only remaining single-knob lever.

**auto_run:** yes
**auto_run_reason:** Diagnostic re-run with a grounded single-variable change (epochs → max_steps over an enumerated range), reuses #533's exact training script (`scripts/i464_phase23_train.py` already accepts step-based termination), cost is known (~22 GPU-h, slightly higher than #533 but within the same order), parent plan §7 already names "sub-1-epoch (max_steps-resolved) grid" as a follow-up branch, success/falsification criteria are concrete numbers on the per-persona d trajectory. No human design decision required.

**cost_class:** needs-gpu
**headline_affecting:** no
