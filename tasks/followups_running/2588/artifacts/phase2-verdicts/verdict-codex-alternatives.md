# Codex (gpt-5.5) — Alternatives & Efficiency lens — VERDICT: REVISE (plan v2)

Ensemble state: Claude APPROVE vs Codex REVISE = **PASS vs FAIL → RECONCILER REQUIRED** (binding).

## PER-AREA VERDICTS
1. REVISE — the automatic descope list silently invalidates registered reads; panel reduction
   itself remains a user decision.
2. REVISE — add a same-metric retrieval ceiling diagnostic and narrow the fixed-column claim to
   an AA-ordered release association.
3. APPROVE — the null branch is shippable as unresolved-at-this-sensitivity, though a
   prospective MDE would improve calibration.
4. REVISE — arithmetic and vectorization are sound, but the primary-pod disk/concurrency bound
   is not.
5. REVISE — make G2 a fan-out prerequisite; otherwise a primary-pod failure can strand most
   booked compute.

## MUST-FIX E1 — the primary retrieval trend has no answer-repeatability/stereotypy control
A positive Δcol cannot distinguish a better linear map from a mechanically easier, more
repeatable target pool. The two-draw ceiling is used ONLY for ceiling-normalized R², while the
PRIMARY trend is raw/null-calibrated exact-row retrieval — so the ceiling never touches the
headline metric.
FIX: add a selected-layer repeat-draw RETRIEVAL ceiling (seed-43 answer vectors retrieving their
seed-44 targets) and report (map − null)/(repeat ceiling − null), or an equivalent
ceiling-conditioned sensitivity, BESIDE (not instead of) the primary metric.
Mechanizable: check P1 persists same-metric repeat-draw acc@1 and P3/`trend_summary.json`
consumes it for every primary-column cell.
**NOTE: the Claude twin proposed the IDENTICAL fix (its concern 1, same seed-43/seed-44 shape),
independently. The two reviewers AGREE on substance and diverge only on SEVERITY — Claude filed
it as a non-blocking analyzer concern, Codex as fatal-until-registered.**

## MUST-FIX E2 — the fixed-size column is framed as varying capability alone
Width, depth and family are held constant, but release recipe, training-data recency, reasoning
distillation, and contamination are NOT. §1's "the only cell ... where capability varies" and
"genuine capability effect", and §0.0's "says mapping quality really does track ability",
overstate what is held constant.
FIX: replace with the supported claim — "mapping quality changed across same-size releases
ordered by AA" — and state explicitly that AA supplies the capability INTERPRETATION while the
design does not identify capability as CAUSAL.
[CONVERGES with Claude Alternatives concern 2, Claude Methodology's fixed-size-column note, and
Codex Methodology press-point (c). FOUR reviewers on this wording.]

## MUST-FIX E3 — automatic descope levers contradict registered reads
**DIRECTLY CONTRADICTS the Claude twin**, which concluded "the headline stays falsifiable under
every registered descope".
Lever (1) substitutes GPQA repetitions for a generic-surface ceiling — GPQA repetitions estimate
variability on ANOTHER SURFACE, so the registered generic ceiling-normalized R² is not preserved.
Lever (3) changes H2's registered denominator from NINE checkpoints to SEVEN while Success
criterion 4 still promises the nine-checkpoint comparison.
Consequence: a "successful" 17-cell run can no longer produce its declared H2/ceiling
diagnostics.
FIX: move these to must-ask, or define valid same-surface substitutes and revised registered
denominators BEFORE launch.
Mechanizable: map each descope lever's removed cells/artifacts to every registered consumer and
reject unresolved denominator or surface mismatches.

## MUST-FIX E4 — pod-2588 storage bound assumes two simultaneous capture stores despite four
## concurrent cell drivers
**INDEPENDENTLY CORROBORATES Claude Methodology Must-Fix 1**, with its own arithmetic: seven
distinct small-pod repos ≈ 1.6+4+8+18+14+14+14 = 73.6 GB; third-wave captures ≈ 40 GB; venv/cache
allowance 15 GB ⇒ ≈ 128.6 GB BEFORE raw texts and filesystem overhead, against the ~130 GB quota.
`assert_out_root_headroom` detects it but may HALT the third wave rather than let it finish.
FIX: a write-phase semaphore limiting resident captures, an explicit purge of each HF model
snapshot after its final cell, or a wave-by-wave peak ledger with stated quota margin.
[The Claude Methodology critic reached ~146 GB worst-case / ~132 GB worst-wave by a different
route. Two independent arithmetics, same conclusion: the ledger's "≤2 cells' captures" assumption
does not describe the four-concurrent-driver primary pod.]

## MUST-FIX E5 — G2 is a global validity gate but NOT a global launch dependency
**CONTRADICTS the Claude twin's "sequencing is robust"**, and sharpens Claude Methodology's
non-blocking "the 27B family pilot before fan-out is partially illusory" into a costed blocker.
G2 runs on pod-2588 during P1 while all five suffixed pods are provisioned CONCURRENTLY; nothing
makes their drivers wait for an anchor-pass sentinel. If the primary pod fails before G2, or G2
REJECTS the instrument, up to all **39 booked H200 GPU-h** can complete under an experiment the
plan itself says must halt. The uploads are recoverable but cannot satisfy the registered
instrument-validity criterion.
FIX: run the anchor refit on pod-2588 and publish a verified PASS SENTINEL before provisioning
or starting the five suffixed production drivers; make every driver FAIL CLOSED if the sentinel
is absent.
Mechanizable: inspect launch commands for an anchor-sentinel dependency; test that a
missing/failing sentinel starts ZERO suffixed production cells.

## ALTERNATIVE DISPOSITIONS (Codex's classification)
- Hidden width/depth: RULED-OUT-BY-DESIGN for the fixed-size headline; ANALYZER-WEIGHABLE for
  the descriptive panel.
- Tokenizer and answer length: ANALYZER-WEIGHABLE; exact tokenizer drift an honest scope caveat.
- Answer-target stereotypy/repeatability: **FATAL** until a retrieval-scale ceiling companion is
  registered.
- Family-dependent contamination: UNVERIFIED — non-blocking only under observational wording; a
  release-data contamination audit would be the settling probe.
- GPQA/AA circularity: ANALYZER-WEIGHABLE.
- Bundled release recipe/data/reasoning changes: **FATAL to causal "capability effect" wording**;
  not fatal to an AA-ordered release association.
- Permutation calibration as a width correction for retrieval: ANALYZER-WEIGHABLE and likely
  near-inert. [CONVERGES with both statistics critics.]
- Qwen toggle arms vs OLMo training-pair arms: ANALYZER-WEIGHABLE if reported separately before
  pooling H2. [Codex Methodology D2 escalates this to a Must-Fix with a decomposition.]

## AREA-1 AND AREA-4 ARITHMETIC CODEX INDEPENDENTLY RECOMPUTED AND CONFIRMED
Fit units from §4.4's layer convention: 4×23 (24-layer cells) + 6×31 (32-layer) + 1×27 (28-layer
anchor) + 8×33 (64-layer) = 92+186+27+264 = 569 units. At §9's conservative costs:
305×3.1 + 264×6.2 = 2582.3 s = 0.717 GPU-h, within the stated ≤0.8. The plan's ~583-unit line
OVERCOUNTS slightly in the SAFE direction. Booked total recomputes correctly: 0.5+24+30+9 = 63.5.
Permutation design batches 20 draws and reuses one eigendecomposition per cell, satisfying
vectorize-first; its full solve/retrieval/peak-memory/serialization path still needs the declared
production-shape pilot. P0/P3 are sub-hour CPU VM phases, appropriately below the routing floor.
Uploads occur per cell followed by more cell-local work or termination, so no long terminal
GPU-idle phase.
Small-rung cost: Qwen3.5-0.8B/2B contribute two of H2's nine checkpoint-level arm pairs; at
1.5-2.5 GPU-h/cell those four cells cost roughly 6-10 GPU-h. USER-DECISION NOTE (not a Must-Fix):
if the Goal-named panel were renegotiable, the two smallest rungs are the first defensible
reduction — contextual rather than decisive. They cannot be SILENTLY removed while retaining the
declared all-panel/two-arm Goal.

## NICE-TO-HAVE
Report a prospective sensitivity range for Δcol. Conservative paired 80%-power bound is 8.9 pp
at n=1000 ((1.96+0.84)/√1000 = 0.0885 for D_i ∈ {−1,0,1}); replace the worst-case variance with
the realized paired disagreement rate when results arrive.
