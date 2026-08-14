# #1336 analyzer re-reductions digest (micro pass, 2026-08-13)

MAN=/tmp/i1336/split_manifest.json (= pod pooled_split_v3/split_manifest.json). CELLS=eval_results/issue_1336/cells_pooled_v3/, GATES=eval_results/issue_1336/gates_v3/, DEC=eval_results/issue_1336/decision_v3/cluster_delta_q_per_transition.json, LAD=eval_results/issue_1336/metric_ladder_pooled_v3/.

## Numbers

**Pooled R2 @L30** (headline layer per frozen rule, raw means over {16,21,22,30} = .494/.515/.511/.546; DEC.headline). Two conventions, never fuse: (a) 20% test partition `.test.r2_pooled["30"]`; (b) train-side 5-fold CV `.r2_bootstrap_ci_frozen_layers["30"]` (n=n_train).
| ckpt | on (a) | on (b) [CI] | off (a) | off (b) [CI] |
|---|---|---|---|---|
| base | .4126 | .3935 [.3910,.3959] | .5538 | .5502 [.5489,.5515] |
| sft | .5778 | .5740 [.5719,.5761] | .4482 | .4575 [.4559,.4591] |
| dpo | .5979 | .5935 [.5913,.5959] | .4587 | .4627 [.4611,.4642] |
| rlvr | .6023 | .5916 [.5895,.5939] | .4636 | .4644 [.4628,.4658] |
| rlvr_long | .6195 | .5957 [.5933,.5979] | .4580 | .4627 [.4612,.4643] |

n_train/d: on 37,491/4096, off 149,964/4096 (both n>d; matched-n 15,000 also >d). n_test: on 10,612, off 42,448. Lambda: inner-group-CV, 23-pt grid [1e-3,1e8], ZERO edge extensions; selected @L30: on-arm 3162.3 (base_on 10000), off-arm 10000.
Matched-n R2@L30 (`cells_matchedn_*.json .r2_obs_by_layer["30"]`): on .365/.542/.562/.560/.565, off .507/.410/.417/.418/.415 (base/sft/dpo/rlvr/rlvr_long).

**Per-corpus slice R2 @L30 test** (`.test.r2_per_corpus["30"]`; lmsys / gsm8k_train / gsm8k_test / math / if11k / uf11k / sft11k):
- base_on .3321 .1707 .1573 .1864 .3039 .3200 .4349 | base_off .4650 .4229 .4176 .2536 .3909 .4471 .6125
- sft_on .4864 .3998 .4064 .3911 .4867 .5038 .6288 | sft_off .3797 .2219 .2176 .1873 .3392 .3816 .5237
- dpo_on .5065 .3852 .3888 .3813 .4933 .5353 .6344 | dpo_off .3980 .2261 .2269 .1888 .3360 .4107 .5205
- rlvr_on .5063 .3776 .3837 .3604 .5141 .5342 .6413 | rlvr_off .3987 .2164 .2171 .1919 .3486 .4160 .5220
- rlvr_long_on .4970 .4031 .4132 .3880 .6091 .5469 .6294 | rlvr_long_off .3902 .2357 .2339 .1929 .3360 .4144 .5229

**Gates.** G0v3 (GATES/g0v3.json, g0v3_adjudication.json): verdict FAIL-leakage-exceeds-band, adjudicated PROCEED (autonomous orchestrator, marker v276). Delta_assign +0.0766 vs tol 0.0515 (0.05*ex_v2, ex_v2 1.0303); R2_grouped .5383, R2_random_mean .6149 (SD .000197); all 4 fits argmax L30; fold_row_counts [1685,2982,2735,2166,989] = manifest profile; fold0_quarantine_rows 10; grouping bite 36 clusters / largest 696; legacy LEVEL diag (report-only): .5383 vs ref .6090, |d| .0707. Rationale: conservatism 0.05-0.07 of excess, leakage ~0.01 (ceiling .032), no defect signature.
G1-prime (g1v3_gate.json): pass, raw best .5903 >= bar .2061. G2v2 (g2v2_offpolicy_parity.json): pass, min mean cosine .9999991 >= .999, n=100.

**Quarantine / near-dup (MAN .near_dup_audit).** 239 edges @0.95 -> 306 rows quarantined = 0.64% global; per-corpus rows(frac%): lmsys 10(.07) gsm8k_train 22(.30) math 32(.45) if11k 6(.10) uf 98(1.49) sft 135(2.08) gsm8k_test 3(.23). @0.90: 1,239 edges / 783 incident rows. Cluster diag (report-only): 5 components, 70 merged groups, largest 62 groups / 18,514 rows / 38.5%; merged-mass frac math .785, gsm8k_train .709, uf .679, sft .535, gsm8k_test .287, if .178, lmsys .048.
Within-corpus straddles @0.95: 43 {lmsys 7, math 2, if 30, uf 4}; @0.90: 312 {lmsys 35, gsm8k_train 4, math 6, if 238, uf 24, sft 5}. Quarantine-straddles @0.95: 9 (all sft); @0.90: 198 {uf 4, sft 194}.
**Edge provenance (computed, MAN edges):** @0.95 sft<->uf 170 (71%), math<->uf 31, gsm8k_train<->sft 18, if<->uf 6, lmsys<->sft 5, gsm8k_test<->lmsys 3, rest <=3. @0.90 sft<->uf 744 (60%), if<->sft 224, if<->uf 103, gsm8k_train<->sft 48, rest <=38. **gsm8k_train<->gsm8k_test = 0 edges at both thresholds**; math<->gsm8k = 0 @0.95.
**N16 trigger (computed):** test rows with >=0.90 cross-corpus TRAIN twin = 104 global {sft 44/1366=3.22%, uf 43/1362=3.16%, gsm8k_train 8/1523=0.53%, if 6/1479=0.41%, lmsys 3/2905=0.10%} -> FIRES (>100 global; sft/uf >1%). Drop set reproducible: MAN edges_090 joined to row_index (test-vs-train arm).

**Split (MAN).** 173 groups; test-side groups/corpus {lmsys 10, gsm8k_train 4, math 5, if 4, uf 4, sft 6, gsm8k_test 2}; test share .207-.255 (floor .15 holds); test rows {lmsys 2905, gsm8k_train 1523, math 1689, if 1479, uf 1362, sft 1366, gsm8k_test 288}.

**Delta-Q (DEC): p_max / obs_max / argmax(corpus) / band97.5 / ceiling:**
- base->sft on .138/.462/c34(lmsys)/.631/1.006 | off .001/.433/c1(lmsys)/.238/1.649
- sft->dpo on .081/.197/c81(math)/.257/1.131 | off .018/.126/c2(lmsys)/.119/1.262
- dpo->rlvr on .392/.043/c92(math)/.137/1.489 | off .074/.039/c99(if)/.056/1.270
- dpo->rlvr_long on .168/.087/c139(sft)/.191/1.489 | off .009/.033/c22(lmsys)/.026/1.261
ALL on-arm p>.05 and obs_max<band -> failure-to-reject each on-arm transition (bands<ceilings, informative). Guard top-3 overlap on-vs-off EMPTY on all 4.

**Tier ladder @L30, Bonferroni-2 corpora, on-arm** (LAD pair JSONs `.layers["30"].per_corpus`; own/t0/t6[t6 CI]/t8):
- dpo->rlvr math .3604/.2792/.3279[.3177,.3366]/.3537 | if11k .5142/.4958/.4648[.4525,.4772]/.4977
- dpo->rlvr_long math .3881/.2343/.3297[.3203,.3378]/.3684 | if11k .6091/.2957/.2208[.1987,.2421]/.3637
- base->sft math .3911/-.280/-.537[-.560,-.515]/.1143 | if11k .4867/.1871/.1137[.1011,.1260]/.2858
- sft->dpo math .3813/-.257/-.279[-.309,-.250]/.3635 | if11k .4934/.3819/.3845[.3701,.3978]/.4798
(other pairs/corpora: LAD JSONs, small local files)

**H-OFF per-text-source @L30** (`*_arm_off.json .test.r2_per_text_source["30"]`): base-text slice .131-.186 vs .525-.574 for post-trained-text slices in every off cell (rlvr_off: base .186, sft .535, dpo .568, rlvr_long .538).

## N1-N26 status

- N1: NARRATION - quote per-corpus quarantine mass beside 2x2 headline. Numbers staged.
- N2: COMPUTE, NOT RUN - cluster-resampled CI needs per-prompt tier-6 values in LAD npz on HF (523 MB/pair-arm). Prompt-iid CIs persisted (staged). Caveat: math 5 / if11k 4 test-side groups -> a 4-5-group bootstrap is itself unstable; narrate regardless.
- N3: NARRATION - permutation p = within-corpus exchangeability only, weak alone. Near-moot: all on-arm p>.05.
- N4: COMPUTE, NOT RUN - no divergence artifact anywhere in eval_results (grepped); needs HF raw_completions text pull. ABSENT from disk, computable.
- N5: NARRATION (on H-P miss) - inputs staged: slice table, matched-n, group counts.
- N6: NARRATION - gsm8k_train vs gsm8k_test slices near-IDENTICAL every ckpt (rlvr_on .3776 vs .3837) -> no trained-on-these-prompts advantage visible.
- N7: NARRATION - within-corpus straddles 43@0.95 vs test rows >=288/corpus. DONE.
- N8: NARRATION - quote 239 edges -> 306 rows (0.64%) + cluster diag (70 groups / 5 comps / largest 38.5%). Staged.
- N9: NARRATION - refuted-dominates; moot: confirmed branch never fires (all on-arm p>.05).
- N10: DONE (persisted) - on-vs-off top-3 overlap EMPTY all 4 transitions; no cross-arm agreement to discount.
- N11: COMPUTE, N/A-conditional - no top group survives the null, so no narration needing the check (would need preds npz + turnstore Y).
- N12: COMPUTE, partial - per-text-source slices staged; deficit localizes to BASE-text (.13-.19 vs .52-.57), not a uniform concatenation cap. Individual (i,j) one-target refits NOT persisted (new CPU fits over HF preds/turnstore if H-OFF refuted).
- N13: NARRATION - G0v3 certificate corpus-local (lmsys, 10 test groups) vs math 5 / if 4. Staged.
- N14: NARRATION - discrimination WAS run at the halt; quote g0v3_adjudication.json rationale (conservatism 0.05-0.07, leakage ~0.01, ceiling .032, no defect).
- N15: COMPUTE, DONE - (i) quarantine-straddles 9@0.95 / 198@0.90 beside within-corpus 43@0.95; (ii) sft<->uf DOMINATES (71%@0.95, 60%@0.90) -> Tulu-3 shared-upstream signature CONFIRMED; framing-(c) math<->gsm8k premise REFUTED (0 direct edges); (iii) [0.90,0.95) mass 1,000 edges / 783 vs 306 rows -> threshold-sensitivity flag (assumption 24c) APPLIES.
- N16: COMPUTE - trigger FIRES (104>100 global; sft 3.22%/uf 3.16% >1%). Re-reduction NOT runnable from disk: preds npz hold predictions only, NO ground-truth Y (verified via on-policy/preds_pooled_v3_manifest.json); needs turnstore Y staging (~10s GB). Body: registered-but-unrun sensitivity read, named limitation or follow-up.
- N17: NARRATION - realized gsm8k_train<->gsm8k_test edges = 0; gsm8k_test quarantine 3 rows (0.23%) vs 18.5% worst case -> censoring bias toward teaching verdict bounded by 0.23%, immaterial (and N6 contrast is null anyway).
- N18: NARRATION - CONDITION MET (N15ii). uf11k+sft11k = ONE quasi-unit in H-P >=7/8 count; shared miss -> 6/8 -> H-P-Inconclusive-mixture, never argued up.
- N19: NARRATION - quote merged-mass math .785 / gsm8k_train .709 beside per-corpus stage claims there. Staged.
- N20: COMPUTE, NOT RUN (same inputs as N4). Narrate per-corpus H-OFF conditional on unmeasured divergence, esp. math/gsm8k.
- N21: NARRATION - at gate read quote 36 clusters / largest 696 / straddles 43@0.95 / quarantine 0.64% / fold_row_counts. Staged.
- N22: N/A - branch was leakage-exceeds-band, not instrument-anomaly (draw SD .000197).
- N23: DONE - grouped@L30 .5383 vs ref .6090 (|d| .0707, report-only, construction-confounded); quote beside verdict.
- N24: NARRATION - narrate halt as GROUPING-EFFECT excess (adjudication already does), never bare leakage.
- N25: NARRATION - Delta_assign +0.0766 (over-band, adjudicated conservatism) -> production headline INHERITS ~0.05-0.07 conservatism; quote beside H-P w/ N13 scope.
- N26: DONE - fold-size matching HELD (gate fold_row_counts == manifest profile; seed 13360).

Counts: COMPUTE 7 (N2,N4,N10,N11,N12,N15,N16; run/resolved: N10, N15, N16-trigger, N12-partial; not run: N2, N4/N20, N16-re-reduction; N11 moot), NARRATION 19.

## Mapping reads (Guideline 11)

PRESENT, all 10 pooled cells (`.mapping_baselines["30"]`): identity+learned-bias R2 strongly NEGATIVE everywhere (-1.21 to -3.46; rlvr_on -3.44 vs ridge .6023) - dims match 4096->4096, APPLICABLE. kNN retrieval PRESENT (n=2000, chance@1 .0005): rlvr_on ridge acc@1 .564 euclid / .602 cos, acc@5 .756/.778; identity-bias kNN acc@1 .604/.615 - identity BEATS ridge on retrieval@1 while losing on R2 (known dissociation; flag in body). Tier-8 composition baselines: in LAD `.layers["30"].baselines` (present, not extracted).

## Flags for the body

1. G0v3 formally FAILED its registered band (+.0766 > .0515), proceeded on autonomous adjudication - state FAIL + adjudication provenance plainly, never narrate a PASS.
2. Two R2 conventions differ up to .011 (rlvr_on .6023 vs .5916) - never mix in one table.
3. base_off > base_on anomaly (.554 vs .413); base-text worst-predicted slice in every off cell (.13-.19) - check gen_audits keep rates before narrating (base generation degeneracy suspect).
4. On-arm L30 monotone rise .413->.578->.598->.602->.620; max-over-layers ordering DIFFERS (sft peaks .611 vs rlvr .590) - layer choice changes the stage story; L30 is the frozen headline.
5. All on-arm delta-Q: obs_max < band -> failure-to-reject (bands<ceilings, informative); off-arm rejects 3/4 but guard overlap empty -> no cluster-level teaching story survives.
6. N16 sensitivity read unrun with trigger FIRED - named limitation or follow-up (turnstore Y staging, bigmem).
7. gsm8k_test1319: 288 test rows / 2 test-side groups - thinnest slice, width caveats.
8. Quarantine straddle residual @0.90 concentrates in sft11k (194 rows) - same corpus N18 collapses; consistent shared-upstream story.
