# Causality and steering at context vs answer: every result we have (2026-09-03)

Source material for the `sections/results/07_additional.tex` item
"Causality, steering at context vs answer - try with inverse mapping".
Every number below is copied from the task's promoted `## Takeaways`, or from the
named stats JSON when the task is a v2 report whose claims slots are still
"(Thomas fills in)". Task pages: `https://eps.superkaiba.com/tasks/<N>`.

## Glossary (one line each)

- **Context vector** `v_C`: the last-token residual state at the end of the context, right before the answer starts. **Answer vector** `v_A`: the response-averaged state.
- **Patching**: writing a donor context's `v_C` into the recipient forward pass at one position and layer (or all layers), then generating.
- **F_beh, fraction of a full swap**: judge-scored movement toward the donor context's behavior, 0 = unpatched, 1 = as if the donor context had actually been given. **F_act**: the same read on activations.
- **Shuffled-donor null**: the same edit with a wrong-pair donor. A cell "clears" when its steered interval sits fully above this null.
- **Persona vector**: the mean-difference direction from contrastive rollouts (the arXiv 2507.21509 recipe), extracted at answer tokens unless stated.
- **Read direction**: the regression direction of the fitted context-to-behavior map.
- **Pre-image (inverse mapping)**: the context-side edit that the fitted context-to-answer map says produces the persona vector at the answer. #2254 and #2225 use a pseudoinverse-style pre-image. #2618 fits the reverse map directly.

## 1. Verdict in five lines

1. **Patching the context vector causally moves behavior, but partially and narrowly.** The strongest single-position edit reaches 0.63 of a full context swap, only at the context-end position, and only for some information types (stated formatting policy on Qwen2.5-7B, formatting plus language on Qwen3.5-9B). Persona, fact, and list content patch as null even though a probe reads them at AUC 1.0.
2. **Where you steer matters more than where you extract.** Directions applied at answer tokens control behavior (evil +0.985, sycophancy +0.429 rate). The same directions applied at the context position are inert or weak, and a context-extracted direction applied to all tokens works.
3. **The fitted map is a readout, not the mechanism.** Its read direction does not steer, its Jacobian recovers none of its predictive power (R² −0.001 vs 0.681), and full-state substitution at its input slot moves content acquisition zero.
4. **Inverting the map does not give a steering direction.** The pre-image is inert at the context vector on 44 of 44 tested cells, while a directly measured context direction steers at the same locus. The pseudoinverse is also a poor context predictor (R² ≤ 0.14) and points away from the directly fitted reverse map (operator cosine ≤ 0.34, R² 0.75).
5. **The map barely predicts the patch-induced shift.** Cosine 0.00 at layer 20 (#1415), at most 0.16 across banked maps (#2094), versus 0.70 to 0.86 for predicting unpatched answers with the same maps.

## 2. Patching the context vector (the #1415 → #2094 → {#2162, #2333} → #2329 family)

### #1415 (MODERATE): single-token steering with the context-vector difference, Qwen2.5-7B-Instruct

- Geometry: the answer state moves toward the target at cosine 0.36 to 0.41 after baseline correction, 28 of 28 pairs above random-direction nulls, traversal 2 to 5% of the target norm.
- Behavior: judged shift peaks at layer 14, +6.2 points (p = 0.008, 21% of the context-swap ceiling), replicated on two fresh seeds (+6.6, +5.0). Layers 7 to 10 and the prefix arm stay at floor.
- Persistence: the re-forwarded direct component of the patch persists to the end of the answer span (late bins 2.5 to 10× the jitter floor, 111 of 112 pairs positive) while decaying to 0.4 to 2% of the state norm.
- Transport: matched-query steering vectors transport 0.49 vs 0.22 cross-query (p = 8e-6). The layer-14 behavioral effect is matched-query only (+11.8 vs −0.3).
- Failure modes: all-position steering flips 96 to 98% of draws into Chinese script. The fitted map predicts none of the realized shift at layer 20 (cosine 0.00, magnitude over-predicted 16×).

### #2094 (MODERATE): the systematic slot × layer × dose grid, Qwen2.5-7B-Instruct

- 36 of 1,245 behavior families clear the shuffled-donor null. All 15 clean survivors are context-end edits (14 matched-query, 1 cross), reading 0.18 to 0.63 of a full swap against nulls of −0.24 to 0.07.
- Largest clean effect: full-state patch at all 28 layers of context-end, 0.63 of a swap (0.51 re-sampled, null 0.10). Single-layer edits at layers 12 to 20: 0.18 to 0.33. Independent temperature-1.0 re-sampling confirms 10 of 15.
- Prefix-end, second-to-last, and third-to-last slots: zero null-separated families anywhere. Activation read agrees: nine clean families, all context-end.
- Editing only the query's text tokens (chat template excluded) does move behavior: 14 of 70 clean reads at 0.68 to 0.95 of a swap against elevated nulls of 0.22 to 0.56.
- Edit-to-response map is far from linear: dose slope 0.00 to 0.06 (1.0 for a linear map), best one-operator fit R² 0.084, banked-map transport cosine ≤ 0.16.
- Single-position edits keep fluency (0.7% incoherent vs 2.0% baseline). Whole-span patches break it (93 to 97% incoherent).

### #2162 (v2 report, claims unwritten): which information types the context vector carries causally, Qwen2.5-7B, 21 minimal-pair types

Numbers from `eval_results/issue_2162/f_metrics/stats.json` and companions.

- Only the **stated formatting policy** family clears both nulls at context-end: instr_format 0.707 vs 0.093 null, conflict_format_fwd 0.723 vs 0.116, conflict_format_rev 0.703 vs 0.066, load_instr_format_l3 0.761 vs 0.114, load_instr_format_l5 0.807 vs 0.142.
- Not clearing (steered vs null): verbosity 0.442 / 0.154, language_implied 0.423 / 0.189, reasoning_style 0.385 / 0.204, constraint_knowledge 0.345 / 0.004, conflict_persona 0.315 / −0.061 and 0.183 / 0.229, instr_language 0.165 / 0.058, persona_prompted 0.151 / 0.182, user_expertise 0.142 / 0.124. Facts, lists, prior topic, query content all ≈ 0.
- Untestable after anchor-separation exclusion (n < 12): persona_role_header (n = 1, 0.995), icl_task_mapping (n = 7, 0.745), refusal_boundary (n = 8), user_emotion (n = 1), demo_format and demo_persona (n = 0).
- Prefix-end: only conflict_format_rev passes, at F = 0.023.
- Read × write 2×2 (`two_by_two.json`): probe AUC 1.0 for essentially every cell, causal-positive for only the 5 formatting cells. 25 context-end cells are read-positive but causal-null. Read everywhere, written almost nowhere.
- Routes: instruction beats demonstration. Persona conflicts stay null. Recency kills the format effect (slope −0.187 per depth step, F 0.707 at depth 1 → 0.171 at depth 3), load does not (+0.022).
- Persona-specificity ladder: installs at context-end for pirate 0.201, butler 0.130, warm 0.409 (nulls ≈ 0), philosophy 0.000. Erase is never clean. No specificity trend (all p_holm = 1.0).
- Turn-boundary multipatch: patching every turn boundary jointly reproduces the single context-end read for instr_format (0.710 vs 0.707). For a persona at depth 3 the joint patch reads 0.195 where each single boundary reads ≤ 0.03, so persona information at depth is spread across boundaries.

### #2333 (MODERATE): how much of the patch effect is opening tokens

- Prefilling the patch's own three opening tokens, with no activation edit, recovers 67% of the full context-end patch effect on Qwen2.5 format cells (n = 172, p = 2e-13) and 71% on Qwen3.5-9B. Roughly a third of the format-cell effect is not opening-carried (per-cell 0.61 to 0.80).
- On Qwen3.5 language cells the patch-content prefill recovers only 40% null-adjusted (CI wholly below 1), a reliable residual beyond tokens. Natural-opening donors reach 1.12 (states) and 0.95 (prefills).
- Activation companion recovers 32% on Qwen2.5. Pirate matched-query pairs are indeterminate (diffs near +0.10, below the detectable floor).

### #2329 (v2 report, claims unwritten): the #2162 sweep on Qwen3.5-9B, thinking off

Numbers from `eval_results/issue_2329/f_metrics/stats.json` and the body.

- Context-end Holm-pass set: instr_format 0.527, conflict_format_fwd 0.838, conflict_format_rev 0.644, load l3 0.368, l5 0.470, recency d3 0.480, **plus the language family that was null on Qwen2.5**: instr_language 0.699 (null 0.270), language_implied 0.695 (null 0.319). Prefix-end: zero passes. Persona and fact cells stay null (persona_prompted 0.100, facts ≤ 0.05).
- Transfer read vs Qwen2.5-7B: per-type Spearman ρ = 0.831 (p = 7.4e-9, clustered CI 0.583 to 0.864, 31 shared type × slot units). Agreement: the format family is positive on both models, facts, lists, prior topic, query content, and prompted persona are null on both, prefix-end is null on both. Disagreement: Qwen3.5 gains language (instr_language 0.165 → 0.699, language_implied 0.423 → 0.695) and the depth-3 format cell (0.171 → 0.480). No parent positive is lost.
- Predicted vs realized patched shift (`mapshift/shift_summary.json`): a fresh ridge map reaches mean cosine 0.364 on the 8 survivor cells (0.215 over all 39 cells), shuffled and cross-type nulls ≤ 0.144, raw context shift with no map 0.176. Better than the ≤ 0.16 on Qwen2.5 (#2094), still far from the 0.70 to 0.86 the same maps reach on unpatched answers.
- 2AFC minimal-pair discrimination (`mapshift/dv3_ext.json`): fitted map 0.848 at layer 31 (CI 0.832 to 0.864), identity plus bias 0.837, identity 0.801, null band 0.48 to 0.52. Same story as #2215 on Qwen2.5: fitting adds about one point over identity plus bias.
- Persona-specificity ladder (`q35_ladder_decay/f_metrics/stats.json`): installs transfer at context-end for pirate 0.084, butler 0.056, warm 0.832, trait 0.515, and at prefix-end for warm 0.343 and trait 0.424 (the one verdict flip vs Qwen2.5). Therapy and philosophy rungs do not transfer, erase is never clean. All 6 rungs passed gates here vs 4 of 6 on Qwen2.5.
- Within-answer decay (`q35_ladder_decay/decay/decay_stats.json`): the patched effect starts 0.46 (Qwen2.5) to 0.54 (Qwen3.5) below the prompted ceiling at the first answer quarter. Whether it also decays faster than the prompted effect is unresolved: Qwen3.5 coherence-conditional +0.050 (CI 0.019 to 0.080, "patch decays faster") but all-generated +0.038 (CI spans zero), Qwen2.5 inconclusive on both.

### #2378 (MODERATE), same-issue round `causal-patching-arms`: patching across framings on Qwen3.6-27B

- Writing the chat context vector into the plain-text forward pass at all 63 patchable layers shifts activations +0.15 of the context-swap axis over the matched null (confirmed at temperature 1.0, 30 pairs). The other 3 of 4 screen-passing families do not replicate, and **no judged behavior shift survives** (both re-measured families cover zero).
- This is the only cross-framing patching arm that ran. #2383 (chat-to-plain, assistant-to-story patching on Qwen2.5) was absorbed into #2378 and is `on_hold`.

### #2389 (blocked): context-end patching on Qwen3.8-27B, 39 cells

- Compute is complete and verified (21,060 grid rollouts, 117 of 117 shards, cap-hit 0.35%, HF mirror `superkaiba1/explore-persona-space-overflow/issue2389_q38ce`), pod terminated. Blocked since 2026-08-25 with reason `report_pipeline_needs_dispatch_decision`: the v2 report was never generated and the judge phase never ran, so **no behavioral F_beh exists** (judge scores empty, teacher-forced margin deferred, ~3.7 GPU-h recorded to finish it).
- What exists is activation-only F at context-end (`.claude/worktrees/issue-2389/eval_results/issue_2389/f_metrics/f_cells_actonly.jsonl`, also on branch `issue-2389`): steered vs shuffled means icl_task_mapping 0.824 / 0.443, instr_language 0.639 / 0.329, language_implied 0.624 / 0.297, verbosity 0.439 / 0.297, conflict_format_rev 0.376 / 0.017, instr_format 0.339 / 0.157. Holm survivors over 39 cells (`best_cells_actsel.json`): icl_task_mapping 0.824 (p 2.3e-9), instr_language 0.639 (p 0.020), conflict_format_rev 0.376 (p 0.012).
- Read: at 27B the activation-level effect broadens toward in-context task mapping and language and away from bare formatting, but without judged behavior this stays an activation read, not a behavior claim.

## 3. Steering: context position vs answer position

### #2220 (HIGH): read direction vs persona vector

- The fitted map's read direction is causally inert: best Δ judged rate +0.06 (evil), +0.01 (sycophancy) over every layer, dose, and position.
- The mean-difference persona vector steers strongly at answer tokens: evil +0.985, sycophancy +0.429. Cosine between read direction and persona vector is 0.00 to 0.03 at every layer.
- A raw high-minus-low mean difference over the map's own training contexts does steer: evil +0.65 at answer tokens and +0.46 at context tokens (the only context-token effect in the task), sycophancy +0.41 at answer tokens. Whitened regression rotates away from the causal axis.
- Above dose c ≈ 1 at answer tokens, rate gains ride degraded text (41 to 85% cap-hit, near-total CJK intrusion). Hallucination is rig-inconclusive (baseline rate 0.733).

### #2254 (MODERATE): the pre-image at the context vector vs at answer tokens

- The pre-image injected at the context vector does not clear the noise band (evil 0, sycophancy +6.6 vs a +10.9 edge). A directly measured context direction clears it at the same locus (+2.5 over a band of 0, and +36).
- The inertness is pair-specific: the persona vector steers sycophancy at the context vector (+30.7) but not evil, and the pre-image steers at answer tokens (+47.5 sycophancy).
- Inversion ladder: transpose and ridge-inverse pullbacks stay inside the band at all 44 tested cells. Evil cells are judge-floor-pinned at 0 (uninformative).
- Position round (160 cells): steering only the first k answer tokens recovers a small fraction of the all-answer effect (first token 0 for evil, ≤ 0.03 sycophancy, opening spans 0.02 to 0.07). The pre-image beats its shuffled-map twin only with every answer token steered (+64.4 evil direction-only, +22.1 sycophancy).
- The large all-answer effects come bundled with wrecked text (evil 62 to 100% language-flipped, 87 to 97% cap-hit). Sycophancy's pre-image-at-answer effect stays clean (≤ 3% degraded).
- The same map predicts strongly (held-out R² 0.60, retrieval 0.9) and 96 to 98% of the persona vector is reachable through it, yet the retained subspace holds only about half of the causal context direction's length. Ablating the directly measured direction removes ~53% (evil) / ~35% (sycophancy) of the prompt-induced ceiling. 2 of 3 behaviors decisive.

### #2225 (MODERATE): preventative steering during fine-tuning, context tokens vs response tokens

- Single-layer steering of context tokens does not prevent trait acquisition: evil stays ~83 across coefficient 0.5 → 5.0 while the paper's response-token arm falls 74 → 16. Planned contrast: evil +33.25 and hallucination +22.93 worse for the context arm, sycophancy a tie (−3.51).
- Steering position, not extraction position, decides: the paper's own direction moved to context tokens fails (83.88), the context-extracted direction applied to all tokens works (39.06) and is direction-specific (matched-norm random stays 82 to 84).
- The pre-image of the persona direction under the banked map is inert at the context position (13 of 18 dose contrasts tie). Moved to all tokens it acquires shallow dose-responses that are largely not direction-specific.
- Trait information lives in the response-averaged direction: projecting it out collapses held-out probe AUC 1.000 → ~0.5, removing the context or prefix direction leaves AUC ≥ 0.998.

### #1776 (HIGH): the map is a correlate, not the local computation

- Last-token Jacobian held-out R² −0.001 vs 0.681 for the slot-matched fitted map. Jacobian retrieval at chance. Not an averaging or amplitude artifact (per-context Jacobians match their average, median cosine 0.73).
- Neither operator executes its own answer swap: B-content acquisition sits at the shuffled-target null in every arm (attainable control 0.047).
- Replacing A's layer-14 last-token state with the model's own state under B still leaves acquisition at the null (+0.0001, CI −0.0004 to +0.0008). The slot is not causally sufficient for content acquisition.
- Behavioral steering with the map stays marginal (sycophancy +2.4 points at 12× the parent norm).

## 4. Inverse mapping

- **#2254**: pseudoinverse-style pre-image is inert at the context vector, steers at answer tokens (Section 3).
- **#2225**: the same pre-image is inert at the context position during fine-tuning, weakly and non-specifically active at all tokens.
- **#2618 (v2 report)**: a directly fitted answer→context map reaches held-out R² 0.741 / 0.751 / 0.611 (L14 / L19 / L26) and top-1 retrieval 76 / 84 / 62% over 1,000 held-out contexts, matching the forward map's 0.754. Every pseudoinverse of the forward map fails as a context predictor: truncated pinv 0.003 / 0.072 / 0.027, ridge-pinv 0.034 / 0.135 / 0.112, full-rank pinv catastrophic (R² −8×10³ to −2×10⁷). Direction-aware operator cosine between the fitted reverse map and the best pinv ≤ 0.32 (Procrustes-aligned 0.87 to 0.90, so same spectral shape, different orientation). The #2254 pre-image direction vs the fitted reverse map's direction for the same persona vector: cosine 0.34 to 0.41, top-1000 context overlap 0.32 to 0.54.
- **Gap**: nobody has steered with the *fitted reverse map's* direction. #2254's negative result covers the pinv family only, and #2618 shows that family points elsewhere. This is the untested cell behind "try with inverse mapping".

## 5. Does the map predict the causal effect?

- #1415: cosine 0.00 at layer 20, magnitude over-predicted 16×.
- #2094: banked-map transport cosine ≤ 0.16, best one-operator fit R² 0.084, flat dose-response.
- #1776: Jacobian and fitted map both fail to execute their own swap.
- #2329: a registered predicted-vs-realized result exists (see the #2329 block once filled).
- Paper figure `figures/paper/c2_map_vs_shift.pdf` shows the split: the same maps predict unpatched answer deviations at cosine 0.70 to 0.86 and patch-induced shifts at ≤ 0.15.

## 6. Older or tangential results

- **#1774 (HIGH)**: erasing a trait direction at the context moves state 1.8 to 3.0× and behavior (sycophancy-erase survives matched truncation, hallucination-erase is truncation-sensitive). Additions at a 0.92-unit dose sat at the no-effect reference.
- **#697 (LOW)**: a cross-model context-vector patch (layer 10 → 14, single slot) reproduces almost none of a fine-tuning shift along its direction (f_CV 0.005 to 0.05), below a random-vector floor. Underpowered probe, not a bug.
- **#640 (MODERATE)**: postfix-KV patching cuts off-distribution marker leakage 0.56 to 0.69 but dampens on-target behavior 51 to 71% as much. A blunt revert, not a selective defense.
- **#267 / #350 (archived)**: layer-20 persona-centroid steering elicits a trained marker (17.9% vs 16.4% baseline) but a norm-matched random direction does at least as well.

## 7. Not run or parked

- **#2247 (proposed)**: causal content-specificity via token-matched fine-tuning arms (personas vs false fact vs single token vs formatting vs high-level behavior). Goal set 2026-08-12, never planned.
- **#2383 (on_hold)**: patching across framings on Qwen2.5. Absorbed into #2378, whose one arm ran (Section 2).
- **#1608 (on_hold)**: first-token identity accounting for the #1415 shift.
- **#816 (on_hold)**: reproduce Persona Vectors' steering and preventative steering with a norm-matched random baseline.
- **#2389 (blocked)**: compute done, report never generated (Section 2).
- **Fitted-reverse-map steering**: never proposed (Section 4 gap).

## 8. What already exists in the paper

- A fully drafted causality subsection with six figures, `sections/results/c2_context_vector.tex` (slot specificity at activation and behavior level, layer-14 peak, matched-pair recovery, patch persistence, opening-token recovery, map-vs-shift), was **deleted on 2026-08-30 in Overleaf commit d36e5fa "Remove unwritten section drafts"**. It is recoverable with `git show d36e5fa^:sections/results/c2_context_vector.tex` in `~/overleaf-6a59c927`. Its figures still exist: `figures/paper/c2_slot_specificity_act.pdf`, `c2_slot_specificity_beh.pdf`, `c2_layer14_peak.pdf`, `c2_matched_pair_recovery.pdf`, `c4_patch_persistence.pdf`, `c2_prefill_recovery.pdf`, `c2_map_vs_shift.pdf`, plus `c5_steering_boundary.pdf` (#2220) and `appendix_patching_examples.pdf` (#2478 assembly of #2094 and #2162 before/after generations).
- The current Discussion has no causality or steering sentence (the Jacobian sentence was dropped in c0258d6). `docs/paper_context_answer_map/plan.md` line 78 records the intended message: the map does not predict patch-induced shifts, map inversion need not steer, Jacobians recover none of its predictive power.
- Not yet in any paper draft: #2162 and #2329 (type-specific carrying, language on Qwen3.5), #2254 and #2618 (inverse mapping), #2225 (preventative steering position), #2378 (cross-framing patching on 27B).
