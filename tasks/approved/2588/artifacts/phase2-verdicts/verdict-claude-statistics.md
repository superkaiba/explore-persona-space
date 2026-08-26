# Claude critic — Statistics & Measurement lens — VERDICT: REVISE (plan v2)

Critic read plan v2 in full, lens items 1-18, `selection-symmetric-nulls.md`,
`ood-generalization-folds.md`, lit review §4/§7, and verified load-bearing numbers directly
against `eval_results/issue_2330/split_ids.json` and the #2330 promoted body.

## PASSES on four of the five briefed questions

1. **Estimator well-posedness — PASS on substance.** All 12 live-verified configs cap d at
   5,120; the lit review's "d up to 8192" did NOT materialize on this panel. min n/d =
   10,000/5,120 = 1.95 at every fit cell including the 27B column and OLMo-32B (per-model d is
   constant across layers, so every swept-layer fit shares the model's ratio); arm-b's G5 drop
   cap (≤2%) keeps ≥ ~1.91. λ is explicit-grid val-selected with edge extension, NOT GCV, and
   §6 registers per-fit selected λ + grid-edge status + n/d, satisfying #1887. GPQA is never
   fitted (990 < d). Critic independently re-verified the split artifact: counts
   10,000/400/1,000 and sha prefixes a74675bfed/61c7e623/b1c32e21 all match §10.
2. **Trend fit — right structure, correctly demoted.** Inference sits on the per-prompt paired
   bootstrap Δcol (n=1,000 test prompts; prior-line acc@1 spread 64.4-73.5% per #2330's body,
   adequate power at observed effect scale); panel Spearman labeled DESCRIPTIVE; measured-only
   variant stated as N=3 single-mode and underpowered. Defensible treatment.
3. **Selection symmetry — PASS (item 11 fully satisfied).** Only selected axis (layer) is frozen
   on the held-out val split; test reads, P=200 null draws, and bootstrap CIs all sit at the
   frozen position, so frozen CIs legitimately need no selection inheritance. Per-draw stats
   persist per cell; band-vs-ceiling reported; no gate thresholds a null statistic.
4. **Fold structure — PASS.** GROUP-level fold is the GPQA transfer arm (strongest form under
   the rule); generic corpus carries an explicit exchangeability argument (A20, Medium, with the
   honest no-near-dup-filter caveat inherited from #2330); GPQA rollouts grouped by question
   with same-question retrieval read (chance 5/990 ≈ 0.51%, arithmetic verified); headlines
   fold-labeled.

Calibration promotion (lit §7 item 2) CONFIRMED: trend is fit on calibrated scores, Δcol
primary is calibrated with the width-cancellation note.

## MUST-FIX 1 — the registered split-count probe DOES NOT RESOLVE on the cited artifact
§4.4 P0 step 4, §7 ("Measured n_train"), and A7 all register:
  `jq '.train_10k|length, .val_400|length, .test_1000|length' eval_results/issue_2330/split_ids.json`
The committed file's top-level keys are `[counts, splits, sha256, ...]`; the ids live under
`.splits.*`. Executed as written the command returns 0/errors (exit 5, verified during this
review), so the P0 gate feeding the registered measured-n_train (a #2061-class gate input)
FAILS on a healthy artifact — and A7's "counted this session" attestation cannot have used this
command. The values and shas themselves ARE correct (re-verified via `.counts`/`.sha256`).
FIX: register `jq '.splits.train_10k|length, .splits.val_400|length, .splits.test_1000|length'`
(or read `.counts` and assert against `.splits` lengths) in ALL THREE locations.

## MUST-FIX 2 — length-residualization under-specified exactly where it carries H2
§4.4 registers `length_residualized_refit(layer_star)` ("per-dim OLS vs [log prompt_tok, log
answer_tok, (arm b) log think_tok]; + length-only baseline") and §6's arm-gap row leans on it,
but the plan does not state:
  (i) WHICH vectors are residualized (map input, target v_A, or both — the answer_tok covariate
      implies the target, but it is unstated);
  (ii) that OLS coefficients are estimated on the TRAIN fold only and applied out-of-fold;
  (iii) the protocol on the GPQA transfer surface, where NO train fold exists (990 in-pool
      estimation is circular; applying generic-fold coefficients is the clean choice — pick and
      register ONE);
  (iv) the exact form of the length-only baseline.
Why it matters: H2's pre-registered hard-set read (gap_GPQA − gap_generic, Wilcoxon over 9) is
length-confounded BY CONSTRUCTION in the surface direction too — think caps are 4,096 generic
vs 8,192 GPQA and hard questions elicit longer CoT — and 2606.02907 is the precedent where a
mis-residualized read produced a 100%-confident wrong answer. One paragraph in §4.4/§6 fixes
it; NO compute change (the refit is already booked).

## MUST-FIX 3 — the conditional GPQA judge fallback is an ungated ≥5k-call wave
§4.5 pre-registers ">5% extraction failure → fall back to a Sonnet judge for the unparsed
residue (~≤19k Batch-API calls)" and §8 rates the trigger Medium (worst case: the 0.8B/2B
free-form arms, which anchor the LOW END of the capability trend). No rubric shape, no
`max_tokens` (rule-23 floor ≥1024), no parse-contract round-trip (rule 27), no rule-26 pilot
gate. `llm-judging.md` rule 26 makes an ungated ≥~5k-call wave a Statistics-lens REVISE; #1739
is the 100%-parse-fail precedent. If it fires unpiloted and misparses, the own-measured
GPQA-accuracy secondary capability axis AND the correct/incorrect split read come out
confidently wrong for exactly the small-model cells.
FIX: two sentences pre-registering the fallback instrument (extraction rubric + judge model pin
+ `max_tokens ≥ 1024` + committed round-trip test + ~200-draw pilot gate with wave-transport
declaration). Adds ~200 pilot calls; no §9 row changes.

## CONCERNS FOR THE ANALYZER (non-blocking)
- **Calibration is near-inert for the retrieval primary.** The shuffled-pairing null mean for
  retrieval is ≈ chance ≈ 0.001 in every cell BY CONSTRUCTION (fixed pool, fixed k), so
  "calibrated acc@1" ≈ raw − 0.001, and §6's trend variable "calibrated acc@1 z" divides by a
  tiny per-cell σ estimated from 200 draws. PREFER the excess-over-null-mean form for the
  Spearman (z's noisy denominator can permute ranks); calibration does its real work on the R²
  reads (#2330 nulls −0.017 to −0.022). Do NOT narrate "survives calibration" as strong
  evidence for the retrieval primary — a near-chance null is a weak filter. The real width
  defenses are the fixed-size column + fixed n/pool/k + the effective-rank covariate.
  [CONVERGES with the Alternatives critic's concern 1, independently derived.]
- **Pool-geometry differences WITHIN the fixed-size column.** Capability training can change
  answer-pool discriminability at fixed d, so Δcol > 0 could partly reflect an easier 3.8 pool
  rather than a better map. Read Δcol beside the identity+bias Δ (the no-map reference that
  absorbs pool discriminability) and the participation-ratio covariate. RECOMMEND persisting
  participation ratio per (model, arm, layer_star) in the fits JSONs, not only as a figure panel.
- **H2's 9-pair Wilcoxon pools two contrast types**: 7 within-model arm gaps + 2 cross-model
  (OLMo Think−Instruct) pairs where the "arm" difference includes reasoning training itself.
  Report the 7/2 split and label the OLMo pairs.
- **Per-arm trend N**: §6 says N=11 unqualified, but that holds only for arm (b) — arm (a) has
  9 AA-valued cells (the two OLMo-Think checkpoints have no arm-a cell; the anchor 404s). State
  both, plus the Spearman critical value (|ρ| ≈ 0.62 at N=11, α=0.05 two-sided) so
  "descriptive" is quantified. Optionally state Δcol's MDE (~3-4 pp at n=1,000 paired).
- **LMSYS near-duplicate caveat (A20)** is common-mode across models, so it inflates
  within-distribution retrieval LEVELS, not obviously the cross-model trend; carry as scope
  caveat and lean on the GPQA transfer fold, as planned.
- **G2's 1e-6 cross-device tolerance** may trip on benign H100-vs-H200 fp64 differences; the
  registered must-ask diagnosis path (never silent re-tolerancing) is right — expect the halt
  and budget the round-trip. [CONVERGES with the Alternatives critic's concern 4.]

## WORKFLOW-SURFACE FOLLOW-UPS the critic surfaced (routed by the orchestrator)
1. verify_plan could dry-run plan-embedded `jq` probes whose target is a committed file and FAIL
   on nonzero rc / null output. Concrete and likely to recur.
2. verify_plan WARN: a plan naming a judge fallback with a ≥5k call estimate and no pilot-gate
   vocabulary within the same block.
