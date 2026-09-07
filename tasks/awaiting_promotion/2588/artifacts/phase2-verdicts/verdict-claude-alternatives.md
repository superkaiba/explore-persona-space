# Claude critic — Alternatives & Efficiency lens — VERDICT: APPROVE (plan v2)

Must-Fix: NONE. "Every alternative explanation I could construct for a positive result is
either ruled out by design (the fixed-size column) or weighable by the analyzer from
diagnostics the plan already persists."

## Cheaper designs — declined to REVISE, with reasoning
Panel is pinned in the GOAL TEXT verbatim, so slimming it is Goal drift (planner/user call,
not the critic's). On merits: fixed-size column (6 cells, ~30 GPU-h) decides H1 vs H0; the
ladder makes the scale-artifact half of H0 readable; OLMo is the architecture-class control;
the anchor ties the instrument to #2330. Lowest-leverage cells (0.8B/2B arm-b) are ALREADY
descope lever (3) in §9. A column-only design (~30 GPU-h) answers a narrower question than the
Goal states. Notes 63.5 is not padded — the v1→v2 correction went DOWN 80→63.5.
Penny-wise inverse check passes: §9 never-drop list (lever 4) protects column cells, anchor,
calibration ≥ P=50; descope levers (1)-(3) touch only diagnostics and low-leverage rungs.

## Alternative-explanation inventory (critic's table)
DESIGN-RULED-OUT: width/depth/family artifact (fixed-size column holds width/depth/family/
tokenizer constant + permutation calibration P=200 + panel trend pre-labeled descriptive);
H2 hard-set gap driven by CoT length alone (length-only baseline + residualization incl.
log think_tok).
DESIGN-HANDLED: GPQA 6% circularity (stated on every hard-surface read; GPQA-accuracy axis
never used on the GPQA-surface trend; GPQA-excluded reweighting at P0); AA estimated values /
mode mismatch (measured-only variants, ordinal column read, P0 re-verification).
CONCERN FOR ANALYZER: answer stereotypy / v_A easier when answers cluster; successor-recipe
confound (3.5→3.6→3.8 vary post-training recipe and data recency alongside capability);
contamination correlating with the index; layer-sweep resolution asymmetry (every layer ≤32L
vs stride-2 on 64L) — but this one biases AGAINST H1, so a positive panel trend is
conservative wrt it, and all three column models are 64L under the same rule.

## Null-result interpretability — ADEQUATE
Verdict lattice pre-registers "Indistinguishable" with a paired-bootstrap CI; "wrong layer"
inspectable from full per-layer test curves; "wrong metric" from the pre-registered
retrieval/R² dissociation plus ceiling-normalized R²; "uninformative test" pre-empted by the
registered band-vs-ceiling read (null ≈ 0.001 vs observed prior-line 0.64-0.74). Adjacent
measured pair (3.8−3.6) gives a same-sign supporting read.

## Vectorization / placement — VERIFIED AT SOURCE, not assumed
Serial fit-loop inheritance inspected against #2330's body verbatim ("62 per-layer ridge fits
ran as a serial loop … ~194 s total", the `dense-fit-loop-unbatched` accepted residual) AND
against code: serial `for layer in …` at `scripts/issue2330_matched_fits.py:744/1362/1483`,
`_resolve_device` L1133, `mapping_baselines.identity_bias_predict`/`knn_retrieval` L28/L126.
194/62 = 3.13 s/unit; 583 units ≈ 0.73 h summed across 19 cells on 6 pods; per-cell ~1.5-3.5
min, below the ~15-30 min phase floor everywhere. Serial inheritance TOLERABLE at this scale;
the v1 ~60x misread is fixed and disclosed.
Permutation battery IS batched (one eigh per cell reused, 200 solves in chunks of 20) and
pilot-gated with a production-shape draw-block timing at both d values, booked ≥2x.
Placement clean: P0/P3 on the VM are trivial (<2 GB RSS, ~100 MB staged); fits/nulls
co-located inside cell walls so no narrow phase holds a wide pod >15-30 min; per-cell
upload-then-reap + verify-then-terminate per pod.

## Six-pod sequencing — ROBUST
Per-cell upload-at-completion means a dead pod strands only its own unfinished cells; ≥17/19
is the pre-registered degraded floor with revised denominators; every arm's min width is 1 GPU
with re-routing stated; loss of a column checkpoint escalates to must-ask rather than degrading
silently. Same-prefix pulls confined to pod-2588 behind a flock; cross-pod weight pulls are
disjoint repos, jittered. P3 reads only HF-resident fits/nulls.

## FOUR ANALYZER CONCERNS (non-blocking, carry into the clean-result)
1. Answer-pool geometry at the acc@1 grain: the per-model permutation null sits ≈ chance
   (0.001) for EVERY model at pool 1,000, so the calibrated primary barely differs from raw
   acc@1 and does not differentially absorb answer-stereotypy differences across the column.
   Cheap discriminating read computable POST-HOC from persisted artifacts: a retrieval analogue
   of the two-draw ceiling (retrieve each test prompt's seed-43 answer vector from its seed-44
   draw) per model — if Δcol survives normalization by that per-model retrieval ceiling, the
   stereotypy alternative is weighed.
2. Successor-column narration: Δcol > 0 supports "mapping quality tracks capability OR anything
   co-varying with successor post-training". Do not narrate mechanism beyond the correlational
   Goal. OLMo pairs partially decompose the reasoning-training axis.
3. Descope lever (1) substitutes GPQA rollout pairs for generic-surface ceiling draws; GPQA's
   MCQ-stereotyped answers make that ceiling non-comparable. If the lever fires, LABEL the
   substituted ceiling's surface.
4. G2 anchor-gate halt with 5 concurrent pods idles ~8 GPUs at a must-ask. Likelihood low
   (parent realized deviation exactly 0.0; tolerance 4 orders inside parent's committed tol),
   but on a trip the orchestrator should STOP the suffixed pods rather than idle them through
   the diagnosis.

## TWO NICE-TO-HAVES
1. Wave 3 on pod-2588 runs 3 cells on 4 GPUs (~2 idle GPU-h); re-packing is already an allowed
   deviation — exercise it.
2. §6 registers length residualization only under the H2 arm-gap DV, while §4.4 runs
   `length_residualized_refit` per cell — surface the COLUMN cells' length-residualized
   retrieval read beside the hero figure so the answer-length alternative for Δcol is visibly
   weighed, not merely computable.
