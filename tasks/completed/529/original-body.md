---
title: 'Re-run #464''s marker-less contrastive-negative role-vs-system localization
  at a non-saturated (band-stopped) anchor to resolve the saturated +1-nat edge'
kind: experiment
tags:
- followup
created_at: '2026-06-09T05:28:19Z'
has_clean_result: false
parent_id: 464
goal: 'Determine whether encoding a persona as a custom chat-template role header
  gives a real, separable reduction in trained-marker leakage over a system-prompt
  encoding in the marker-less contrastive-negative regime, measured at a non-saturated
  training anchor (arms ~5-10 nats below the leakage ceiling, where the role-vs-system
  gap has genuine dynamic range), resolving whether #464''s inconclusive saturated-floor
  +1-nat marker-less edge is real or a measurement artifact.'
track: experiment
relates_to:
- spec-role-header
- leak-contrastive-negatives
---
## Goal

Determine whether encoding a persona as a custom chat-template role header gives a real, separable reduction in trained-marker leakage over a system-prompt encoding in the marker-less contrastive-negative regime, measured at a non-saturated training anchor (arms ~5-10 nats below the leakage ceiling, where the role-vs-system gap has genuine dynamic range), resolving whether #464's inconclusive saturated-floor +1-nat marker-less edge is real or a measurement artifact.

## Why this is needed

#464 compared three encoding arms (persona in system prompt / system prompt + length-matched filler / persona in role header) across three training regimes. Two regimes gave clean reads:

- **Co-resident competing marker:** role header wins by ~6 nats (arms span -13 to -19, real dynamic range).
- **No-contrast positive-only:** marker leaks to P≈1 everywhere; role header doesn't help.

The **marker-less contrastive-negative** regime — the one that matters most, because marker-less negatives are the recipe you would actually ship — came back **inconclusive**. Its own analysis pipeline returned `headline_status: inconclusive_dynamic_range_failed`: the `system_plain` arm had sd=0.47 (< the 0.5 dynamic-range threshold), all three arms clustered in a saturated band at log P ≈ -14 to -16 nats, and the reported role-vs-system edge (+1.10 nats, "95% CI [0.51, 2.16]") is a paired bootstrap over only **n=3 seeds**, where the percentile CI is literally the min/max of three points. Operationally the gap is between P(marker) ≈ 1e-6 and ≈ 4e-7 — both effectively zero leakage. The +1-nat edge is therefore **unresolved**: it could be a real separable role contribution, or a rank-shuffle on a saturated floor.

The fix is the project's own prescribed remedy for marker saturation (`marker-leakage-measurement.md`, #448): use a **less-trained anchor** so the wrong-slot log-prob sits ~5-10 nats below ceiling, gate the anchor choice on **bystander/off-diagonal resolution (NOT source emission)**, and read the role-vs-system gap where the arms actually have headroom. #464/#476 bypassed this by training a fixed 5 epochs straight to saturation.

## Single manipulated variable vs #464

**Training anchor amount: saturating fixed-5-epoch → a selected non-saturated anchor** (the wrong-encoding/bystander arms land at log P(marker) ≈ -5 to -10 nats, not -15). Everything else is held identical to #464's marker-less contrastive-negative (cn) regime.

(Secondary, non-scientific measurement change: **seeds 3 → 5** to escape the "CI = range of 3 points" problem. This changes statistical power, not the manipulation; it is noted as a deliberate deviation, not a second variable.)

## Design

- **Regime:** marker-less contrastive-negative ONLY. Single persona per LoRA + marker-less negative rows on the OTHER persona AND the bare default assistant (same questions, on-policy base response, NO marker → loss trains EOS at the post-response slot). This is the exact #464 cn regime. (Positive-only and co-resident regimes are already resolved in #464 and are NOT re-run.)
- **Personas (trained separately):** pirate, villain. Shared marker ` ※` (id 83399). This satisfies the contrastive-negatives rule (marker-less negatives present).
- **Arms (the #464 headline arms):** `system_plain`, `system_padded` (token-count parity control), `role` (matched name `pirate_assistant` / `villain_assistant`).
- **Anchor selection (the new piece).** Because #464's `MarkerBandStopCallback` failed in the i464 direct-launch rig (vLLM-during-HF GPU-residency conflict), do NOT rely on the in-training callback. Instead **checkpoint at the end of epochs {1, 2, 3} (and keep 5 as the saturated reference)**, cross-eval the off-diagonal at each checkpoint, and pick the smallest-training anchor that is BOTH (a) **resolved** — bystander/off-diagonal arms have dynamic range (per-arm sd > 0.5; wrong-slot log P roughly in [-5, -10], not pinned at the floor) — AND (b) the diagonal (own-encoding) marker is genuinely elicited (own-slot log P ≈ 0, P≈1, i.e. the implant took). Gate on bystander resolution, NOT on source emission — the source should saturate emission, that IS the implant. The planner finalizes the exact epoch grid / checkpoint cadence after reading the rig.
- **Seeds:** 42, 137, 1337, 7, 21 (5 seeds).
- **DV (unchanged from #464):** marker log P (trained − base) at the slot immediately after the shared canonical response `R_canon`, teacher-forced via vLLM `prompt_logprobs`. Off-diagonal = the OTHER persona's same-arm encoding AND the default assistant. Headline statistic = `role − system_plain` and `role − system_padded`, paired per seed, averaged over the symmetric pirate/villain pair, 95% paired-bootstrap CI over the 5 seeds.
- **Cells:** 3 arms × 5 seeds × 2 personas = 30 single-persona LoRAs (each saving the {1,2,3,5}-epoch checkpoints from one training run).

## Hypotheses / outcomes (all interpretable)

- **H1 (role has a separable contribution):** at the selected non-saturated anchor, `role − system` is positive, the paired CI over 5 seeds excludes zero, AND the dynamic-range gate PASSES (arms not saturated). → #464's +1 nat was real; role header tightens marker localization even with marker-less negatives.
- **H0 (floor artifact):** at every interpretable (resolved) anchor, `role − system` is indistinguishable from zero or sign-unstable across seeds once the arms have real dynamic range. → the +1-nat edge was a saturated-floor rank-shuffle; the role header has no separable contribution in the marker-less regime, and its only clean advantage is the co-resident competing-marker regime (which #498 then shows does not generalize to broad traits).
- **Degenerate (still saturates):** if even the 1-epoch anchor saturates everywhere (own AND wrong slot at the ceiling), report that the marker-less single-persona implant saturates too fast for this DV to segment arms at this rank/lr, and recommend the next lever (smaller LoRA rank or lower lr) rather than over-interpreting.

## Reuse (do not reinvent)

The entire rig is on the `issue-464` branch at SHA `0905fc70f0ad2416d9435236102df6a01d580dfc`:
- `scripts/i464_cn_run.sh` — marker-less cn regime runner
- `scripts/i464_phase23_train.py --single-persona --shared-marker --contrastive-negatives` — training
- `scripts/i464_po_eval.py --variant contrastive_negatives` — cross-eval
- `scripts/i464_po_analyze.py` — bootstrap + dynamic-range gate
- `scripts/i464_phase1_generate_R.py` / `i464_cn_generate_R_default.py` — canonical R
- `src/explore_persona_space/experiments/i464_encodings.py` — encodings + persona defs

The ONLY code changes are: (1) save intermediate-epoch checkpoints in the train script, (2) drive the cross-eval + dynamic-range gate per checkpoint and select the anchor, (3) bump the seed list to 5. branch from `issue-464` (or cherry-pick the `i464_*` scripts onto the new worktree).

## Resource estimate

~30 LoRA cells, ~5 min/cell training (per #464's measured rate) + multi-epoch checkpoint eval. Estimate ~2-3 h on 4×H100 (~8-12 GPU-h), well under the 100 GPU-h auto-approve cap.
