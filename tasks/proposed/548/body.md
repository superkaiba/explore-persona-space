---
title: 'Lift the canonical-JS sampling cap to 1024 tokens on the identical 416-cell
  panel (corrective re-run of #540''s length-censoring confound)'
kind: experiment
tags: []
created_at: '2026-06-10T06:29:47Z'
has_clean_result: false
parent_id: 540
goal: Determine whether the canonical sequence-level JS divergence carries leakage-predicting
  signal beyond response-length divergence once the 256-token sampling cap that censored
  82-99% of replies is lifted to 1024 on the same 416-cell panel.
---
## Goal

Determine whether the canonical sequence-level JS divergence carries leakage-predicting signal beyond response-length divergence once the 256-token sampling cap that censored 82-99% of replies is lifted to 1024 on the same 416-cell panel.

## Motivation

Parent #540 replaced the deprecated first-token JS shortcut with the canonical Rao-Blackwellized sequence-level estimator (arXiv 2504.10637) on #532's 416-cell panel and found the canonical JS predicts marker leakage no better than the absolute difference in mean reply length between the two contexts (length alone ρ = −0.54 vs canonical JS −0.49 on the 256 ordinary cells; length-partialled JS −0.06, p = 0.32). The binding confound, named explicitly in the parent body: replies hit the 256-token sampling cap constantly (82% of samples for the median context pair; the robustness subset stays 98.6% truncated), so "length difference" partly encodes "how often replies stop early under the cap," and per-token normalization mechanically couples the front-loaded divergence spike (~0.58 bits at position 0) to capped lengths. The length-controlled null is therefore not yet interpretable as a claim about uncensored reply length.

## Hypothesis

The 0.83 JS-length entanglement and the null length-controlled JS partial are partly artifacts of cap censoring. With the cap lifted to 1024, natural reply lengths vary genuinely and the length control becomes interpretable: either the canonical JS keeps a length-free component (reviving the divergence arm) or it stays null on uncensored lengths (upgrading the parent's deflationary claim from "on this capped panel" to a much stronger read).

## Falsification / kill criteria

- If median-pair truncation drops to ≤ ~20% AND the length-controlled canonical JS partial on the ordinary strip is still indistinguishable from zero (CI spanning zero) while the length feature keeps unique signal, the cap-censoring explanation is dead and the parity headline hardens.
- Conditional kill: if median-pair truncation at 1024 is still > 50%, the panel's verbosity is the binding factor; route to the length-diverse-panel follow-up (proposal 2 on #540's `epm:follow-ups v1`), do NOT keep raising the cap.

## Differs from parent (#540)

Exactly ONE variable: `max_tokens` 256 → 1024 in Phase S sampling (scored positions extend accordingly; same estimator, same panel, same probes, same DV, same analysis). The planner must record the deliberate deviation from `persona-distance-metrics.md`'s ≤256-token cap: the cap IS the manipulated variable.

## Pre-filled spec (from parent)

- Model: Qwen/Qwen2.5-7B-Instruct, no adapters, bf16 weights, fp32 scoring (same).
- Data: same 26-context panel, same 50 probes `q_test_extended_50` (Hub-verified 2026-06-10); DV reused byte-for-byte from `eval_results/issue_532/per_cell/loc_ep1/` (416 files, git-verified on main).
- Seeds: 42 (same).
- Eval: same leaderboard re-fit via pinned `scripts/issue532_predictor_stress.py` @ 296c4da2d + the #540 length-nuisance protocol, now PRE-SPECIFIED as the primary read (length-alone ρ, both partial directions, both normalization variants, truncation rates); same reproduction control + self-pair + position-0 gates.
- Analysis bookkeeping folded in from #540 proposal 3 (zero extra GPU): leaderboard additionally carries the length-difference column and a stacked base-prior + length combination.
- Reuse: `scripts/issue540_jsrb_predictor.py` (on main @ a6157cbb) is the dispatcher; the only config delta is the token cap.

## Cost

~4-6 GPU-hours on 4× H100 (intent `eval`, `--gpu-count 4`); budget 8. Basis: parent actual was 1.54 GPU-h end-to-end with teacher-forced scoring dominating; scoring FLOPs scale roughly linearly in scored positions (≤4×), and many replies terminate naturally before 1024, so 3-4× the parent's actual is the honest band. Wall ~1-1.5 h.

## What we learn

If JS keeps length-free signal at 1024: the parent's parity claim gets scoped to cap-censored panels and the divergence arm returns to the leaderboard as a live predictor. If still null: the deflationary headline strengthens substantially (full-reply divergence adds nothing over reply length even uncensored) and the predictor program drops sequence-level divergence in favor of activation geometry + base prior. Either branch is decision-relevant.

## References

- Parent #540 clean-result + `epm:follow-ups v1` proposal 1 (auto_run: yes).
- `eval_results/issue_540/` (predictors_jsrb.json, analysis_jsrb.json, length_nuisance_supplement.json, per_pair/) on main @ a6157cbb.
- `.claude/rules/persona-distance-metrics.md` (canonical RB JS recipe; the ≤256 cap clause is the deliberate deviation under test).
