# RECONCILER (binding) — Alternatives & Efficiency lens — VERDICT: REVISE

Claude APPROVE (0 MF, 4 analyzer concerns) vs Codex REVISE (5 MF). Reconciler adjudicated
item by item with fresh context. Binding outcome: 2 Must-Fix survive, 3 downgrade to BINDING
standing recommendations (absorb in the same revision, non-blocking).

## PER-ITEM RULINGS
- E3 descope levers contradict registered reads — **SURVIVES, blocking.** Both halves verified.
  Lever (3) (§9 L301 "drop arm-b generic for the two smallest rungs") removes those checkpoints'
  arm-b maps entirely (fits are generic-only; GPQA is transfer-only, never fitted), so H2's
  Wilcoxon loses 2 of its 9 pairs — while Success criterion 4 (L435) still promises "gap_GPQA vs
  gap_generic over 9 checkpoints" and §3 (L61) registers "mean over the 9 thinking checkpoints".
  Lever (1) substitutes GPQA rollout pairs for the §6-registered two-draw generic ceiling (L195),
  a different-surface MCQ-stereotyped ceiling that Claude's own concern 3 concedes is
  non-comparable. Because levers (1)-(3) are pre-authorized "allowed without asking" (§ Plan
  deviations L450), a mid-run orchestrator has explicit licence to unsatisfy a registered success
  criterion with NO user contact — the plan-contradiction class. Claude's blanket falsifiability
  claim covered only the never-drop-protected headline and missed the lever-(3)/criterion-4
  contradiction entirely.
- E4 pod-2588 disk/concurrency — **SURVIVES, blocking.** The §9 disk row (L305) prices "one big
  model resident + ≤2 cells' captures" — a model of the SUFFIXED pods. pod-2588 (L284) runs 11
  cells one-per-GPU on 4 GPUs in 3 waves across SEVEN model repos (≈73.6 GB accumulating in
  HF_HOME on the same MooseFS volume, no stated snapshot purge) plus up to 4 concurrent capture
  sets (9B/OLMo-7B ≈ 14-15 GB each ⇒ bad wave ≈ 40-60 GB) ⇒ ≈128-145+ GB against the ~130 GB
  EDQUOT quota, before raw texts and upload staging. THREE independent arithmetics agree (Codex,
  Claude Methodology, reconciler's own). `assert_out_root_headroom` converts this to a
  deterministic wave-3 HALT, not a fix. DEDUPE with Methodology MF1 — one fix, filed once.
- E1 stereotypy/repeat-draw retrieval ceiling — **DOWNGRADED** to binding standing rec. Substance
  agreed by both reviewers; severity ruled explicitly. The demanded read is a PURE RE-REDUCTION
  of persisted artifacts: §4.1 ceiling draws (seeds 43/44) enter the capture pass over all rows
  at all swept layers incl. `answer_span_mean`; §10 persists `analysis_tensors/capture/` per cell,
  uploaded before fits. Nothing gates on it; the must-ask list bars changing the PRIMARY metric,
  not adding a reported companion. Note the primary column cells are arm-a, so their generic
  ceiling draws survive even lever (1).
- E2 capability-alone framing — **DOWNGRADED** to mandatory-absorb scoping rec. The strongest
  quoted phrase was NOT IN THE PLAN: "genuine capability effect" appears nowhere (grep finds only
  "genuine MEASURED basis", L376) — a synthesized quote. §1's actual sentence is factually
  accurate as written; §0.0's "rather than model size or family" names exactly the contrast set
  the column DOES rule out. Residual overclaim ("really does track ability") is narration
  discipline. Four-reviewer convergence makes the fold-in obligatory regardless.
- E5 G2 not a global launch dependency — **DOWNGRADED** to binding standing rec. The gap is real
  (no anchor-pass sentinel anywhere in §7/§9), but G2 validates the staging+FIT path on the
  BANKED 7B store, NOT the generation/capture rig — so every G2 failure mode invalidates only the
  fits/nulls (≤~2.7 GPU-h summed, re-runnable on persisted captures), never the 39 H200 GPU-h of
  generations+captures, which upload per-cell and are instrument-independent. Bounded + loud +
  durable evidence + free fix = non-blocking. The Methodology critic's spend-risk rating was the
  correct severity; Claude's "ROBUST" heading overstated but its concern 4 already carried the
  operative remedy.

## CONSOLIDATED MUST-FIX FOR THIS LENS
1. (E3) Reconcile EVERY descope lever with its registered consumers. Lever (3): move to must-ask,
   OR pre-register the revised H2 registration (amend §3 H2 + Success criterion 4 to "over the
   surviving thinking checkpoints (≥7, named)", drop reported per the planned-vs-actual rule).
   Lever (1): restrict so cells keeping the registered read retain seeds-43/44 generic draws, OR
   pre-register the substituted ceiling's surface + a non-comparability label on every consuming
   read. General duty: each lever's row names every registered read/criterion it degrades and the
   revised registration, before launch.
2. (E4) Re-derive pod-2588's disk ledger under its ACTUAL concurrency and add a bounding
   mechanism: per-model HF_HOME snapshot purge after that model's final cell, and/or a
   capture-write semaphore (≤2 resident capture sets), and/or a wave-by-wave peak ledger with
   stated quota margin.

## BINDING STANDING RECOMMENDATIONS (absorb in the same revision)
- (E1) Register in §6/P3 the selected-layer repeat-draw RETRIEVAL ceiling (seed-43 answer vectors
  retrieving seed-44 targets, cosine, pool 1,000) per column cell, reported BESIDE — never
  instead of — the primary, e.g. (map − null)/(ceiling − null); `trend_summary.json` carries it.
  Zero compute; inputs already persist.
- (E2) Replace §0.0's "really does track ability" and harden §1/§5 narration to the supported
  claim: an AA-ordered same-size RELEASE association; recipe, data recency, reasoning
  distillation and contamination co-vary; AA supplies the capability interpretation, the design
  does not identify capability as causal.
- (E5) Either publish a verified anchor-pass sentinel that suffixed production drivers fail-closed
  on before their FIT stage (generation/capture may proceed — it is G2-independent), or write into
  §9 the explicit orchestrator duty to stop the suffixed pods on a G2 trip plus G2-first
  scheduling on pod-2588.

## OBSERVED BUT NOT RAISED
Lever (2) (GPQA rollouts 5→3) also changes the registered GPQA pool/chance constants (§6: "pool
990, chance 5/990"); the E3 lever-consumer-mapping fix should sweep it in.

## REVIEWER-ACCURACY NOTE
Codex prevails on the binary and on E3/E4 substance; Claude prevails on severity for E1/E2/E5 —
its three analyzer-concern classifications were the calibrated ones. One Codex quote in E2 does
not exist in the plan (synthesized-quote pattern, already on file).
