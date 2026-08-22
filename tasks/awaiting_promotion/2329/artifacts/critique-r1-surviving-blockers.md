# #2329 `q35_ladder_decay` — CRITIQUE panel round 1: SURVIVING blocker list (revise against exactly this)

Panel outcome after per-lens ensemble decisions and two binding reconciliations:

| Lens | Claude | Codex | Binding result |
|---|---|---|---|
| Statistics & measurement | REVISE | REVISE | **REVISE** — blockers UNIONED (no reconciler; both agree in direction) |
| Methodology & baselines | PASS | REVISE | **REVISE (narrowed)** by reconciler — 1 blocker, +0 GPU-h |
| Efficiency | PASS | REVISE | **PASS** by reconciler — Codex must-fixes rejected; 3 standing recommendations |
| Single-variable consistency | PASS | (no twin) | **PASS** — 1 report-side note |

Everything below is either BLOCKING (must be fixed in plan v6) or FOLD-IN (non-blocking, but
a revision is happening anyway so land it now). Nothing here is optional-at-your-discretion.

---

## BLOCKING — Statistics lens (union of Claude + Codex)

### S1. Hypothesis 3's registered reproduction direction is SIGN-INVERTED against the parent artifact

Plan §3 Hypothesis 3 calls `install >= erase` "the parent's h4 read". The cited artifact
field is `h4_asymmetry.*.mean_erase_minus_install` in
`eval_results/issue_2162/persona_specificity_ladder/stats.json` — POSITIVE means
`erase > install` — and the realized values are MIXED, with two cells significantly the
other way: `r5b_lu_philosophy|ce` +0.218 (95% CI [0.154, 0.303]) and `|pe` +0.478
([0.244, 0.720]).

So the plan registers a reproduction target that is the REVERSE of what the parent showed,
and `verify_plan.py` passed it.

FIX: replace the unsupported global direction with the parent's ACTUAL per-cell pattern, OR
define and recompute a justified parent aggregate BEFORE registering directional
reproduction. Re-read every other narrated parent sign/CI in the plan against its cited JSON
field while you are in there — this is a class, not a single typo.

### S2. Leg B's steered arm pools non-equivalent patch cells, and its decision table treats non-significance as equivalence

Two defects that compound in the SAME direction, both flagged by Claude and Codex.

(a) POOLING. The Leg B row grain is `per-(model x direction x carrier x segment)` with NO
`slot` term, and the §9 sizing arithmetic (install-steered ~220 x 4 segments; 440 steered
draws/arm / 2 directions = 220) only closes if ce AND pe are both included. But the parent's
realized `lattice` puts EVERY install-pe cell at `no-clean-transfer`, and only R1-R3 CE
install transferred — so roughly half the reused steered rows are completions in which the
persona was never installed (flat-at-floor curves, decay ~0). Pooling dilutes steered decay
toward zero while the ceiling arm is untouched, so a genuine "patched decays faster" effect
can ship as a null or inverted.

(b) NO COHERENCE FILTER. Leg B gates only on the 48-token minimum length, while the parent's
own behavioral F conditions on the >60 coherence screen — and incoherence concentrates in
patched arms.

FIX:
1. Add `slot` to the Leg B row grain and PRESERVE it through the reduction.
2. Register the PRIMARY contrast on a fixed, independently grounded CE / common-cell set
   (the parent-demonstrated install surface). `pe` is either a separately stratified
   exploratory read or excluded with that reason stated — NEVER pooled into the primary.
   Alternatively report every slot/rung separately.
3. Filter Leg B steered rows on the inherited coherence screen. This costs ZERO new judge
   calls: parent per-draw coherence scores are already committed for the Qwen2.5 side, and
   Leg A's own judge wave produces them for Qwen3.5.
4. Construct per-carrier PAIRED differences on COMMON SUPPORT, and resample each carrier
   JOINTLY across arms and across Q1/Q4 (one shared carrier-resample index per bootstrap
   draw).
5. Cost direction: CE-only REDUCES Leg B by ~880 calls per model. No §9 row increases.

### S3. Every zero-spanning-CI branch must read "inconclusive", not "falsified" / "generic decay" / "not a patching artifact"

Plan §3 Hypotheses 4-5 map CIs that span zero onto affirmative conclusions. With six carrier
clusters those outcomes are UNDERPOWERED, not evidence of equality.

FIX: rewrite every zero-spanning branch of the decision table as "inconclusive"; reserve
positive directional claims for CIs that EXCLUDE zero. This is also the project's standing
null-framing rule — "indistinguishable from null given the variance", never "confirms the
null" / "does not transfer".

### S4. `null_sanity_flag` is registered as ADVISORY but the inherited code branches a verdict on it

Plan §3 says the null-derived flag is "ADVISORY ... never an abort". The inherited
implementation sets `transfers` only when `ci_ok and not null_flag`, against a fixed
`NULL_SANITY_BAR = 0.10` (`scripts/issue2162_ladder_analysis.py:80`, `:807-826`), which §4.1
item 3 inherits BYTE-VERBATIM. So an asserted threshold can flip a CI-separated transfer to
`no-clean-transfer` and change Leg A's headline while the plan presents the flag as
non-binding.

FIX — and note this is a PROSE-vs-CODE CONTRADICTION, so it resolves in one of two
directions and YOU MUST PICK ONE AND JUSTIFY IT:
- (i) Make the flag genuinely advisory: remove it from the verdict condition, retain it as a
  recorded diagnostic (the already-registered steered-vs-both-null CI comparison supplies
  the inferential control). This DIVERGES from byte-verbatim parent inheritance — if you take
  it, declare the divergence explicitly in §2 and state the parent-comparability cost.
- (ii) Keep the code byte-verbatim and CORRECT THE PLAN'S PROSE to state that the flag is
  verdict-BINDING at the fixed 0.10 bar, with the consequence named.
Do not leave the contradiction standing in either direction.

### S5. The registered Holm family can silently shrink

Plan §0 registers Holm `m = 4`, but the inherited implementation builds `testable` AFTER
gate/drop outcomes and calls `holm(testable)`
(`scripts/issue2162_ladder_analysis.py:739-753`), so if a family goes untestable the
correction runs over a SMALLER family while `holm_m_registered` still records 4. A marginal
trend can become "significant" only because a sibling family dropped out.

FIX: pass ALL FOUR registered families into the multiplicity adjustment, entering an
untestable family as a non-rejecting placeholder (p = 1) while separately LABELING it
untestable. Same prose-vs-code choice as S4 applies if you would rather correct the
registration than the code — pick and justify.

---

## BLOCKING — Methodology lens (reconciler-narrowed; +0 GPU-h, booked 6 stands)

### M1. Pin and RECORD the model-weights revision. Do NOT re-capture.

The reused Qwen3.5 `vc_bank.pt` donor states have no established MODEL-WEIGHTS-revision
identity with the model that will receive them: `body.md:49` records that every
`from_pretrained` resolved unpinned `main` with the pod-side commit unprovable, and
`issue2329_run.load_model_and_tokenizer` still calls tokenizer + model `from_pretrained`
with no `revision=` (the ladder fork loads through it at `issue2162_ladder.py:693,807,1238,1333`).
The existing shape assert catches wrong-ARCHITECTURE only (any Qwen3.5-9B revision is
32x4096), and `stage_parent_bank`'s data-repo pin binds which BYTES are fetched, not which
WEIGHTS produced the states inside them.

The reconciler PROVED the realized bank is on-basis (so re-capture is NOT required):
`Qwen/Qwen3.5-9B`'s last model-repo commit is `c2022362...` (2026-03-02; last weight-bearing
commit `ef3d031a`, 2026-03-01) and nothing has landed since, while the vc_bank was captured
2026-08-17 — ~5.5 months later — so any resolution of `main` in the capture window provably
hit those same weights. The residual risk is forward-looking (an upstream push landing before
this round's own unpinned load) plus this round re-creating the same unprovable-revision gap
for its OWN artifacts, which is a run-time-only loss that cannot be repaired after the fact.

FIX (all four parts):
1. Thread `revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a"` into BOTH `from_pretrained`
   calls on the fork's load path (tokenizer AND model). A fork-local `model_revision` config
   field is sufficient.
2. Add `model_revision` to the `_repro` bundle so every persisted artifact records it.
3. Record the donor-bank identity derivation in §10 (the commit-history argument above, with
   the `HfApi().list_repo_commits` provenance).
4. Add an EXECUTABLE donor-revision assert (the bank carries no revision key): at L1, on the
   already-loaded pinned model, re-forward 2-3 screened cross-type donor contexts and assert
   per-layer cosine >= 0.99 between the freshly captured slot states and the staged vc_bank
   states; HALT on failure.

GPU-h delta: 0 — seconds of forwards on the already-booked L1 pass. The booked 6 GPU-h
stands on this account.

Scope note: this exposure is SPECIFIC to hidden-state donors. The reused Qwen2.5
completions Leg B consumes are TEXT (frozen at generation, read by the segmenter + judge),
as are `bank.json` and the parent's committed F-tables — none are in the exposure class. Do
not widen the fix to them.

---

## FOLD-IN — Efficiency lens (lens PASSED; these are the reconciler's standing recommendations)

### E1. Disclose the permitted-branch GPU-hour upper bound next to the booking

The reconciler VERIFIED that the plan's own gates permit more than the booking covers:
baseline 420x5.12s ~= 0.60h + 1,320x5.12s ~= 1.88h on a 4.6 GPU-h pod base; G2 aborts only
above 2x basis, so the no-abort band reaches ~7.1 GPU-h with no regeneration; a broad G5
regeneration adds ~1.9h at baseline rate or ~3.75h near the G2 threshold — a
registered-branch upper bound of ~10.8 GPU-h (~1.8x the booked 6), with an extreme
triple-tail conjunction near ~14.5. The §9 contingency of 1.4 GPU-h is arithmetically
INSUFFICIENT for the broad form of the G5 contingency it names.

This is a DISCLOSURE requirement, not a re-budgeting demand: the bound stays inside the
< 20 GPU-h cheap band, so no routing or approval rail flips, and the fence is NOT breached
(fences re-derive to >= 2x the pilot-extrapolated wall, so the fence in force under any
given branch already exceeds that branch's reachable cost).

FIX: state the bound next to the booking — e.g. "booked 6 (4.6 base + 1.4 reserve);
permitted-branch upper bound ~7.1 (G2-max, no regen) / ~10.8 (G2-max + broad regen) — still
< 20 cheap band" — and bound the G5 regeneration aggregate (e.g. regen <= the reserve, beyond
which cap-hit is reported as a finding plus a recorded basis update).

DO NOT adopt a budget-derived G2 abort threshold. The reconciler ruled that
anti-efficient: it would imply aborting at ~1.3x basis, tighter than the house-standard >2x
pilot abort, on a run whose §11 names the width-8 -> width-1 basis transfer as the reason
moderate rate shifts are EXPECTED. House doctrine sizes the FENCE to worst case and the
BOOKING to a p90-style projection plus reserve.

### E2. Correct §8's pod-reload comparison

§8 compares the L3 idle against "~2x 20-30 min model reload". Only ONE reload is incremental
relative to keeping the pod (the first load happens on either path), so the "2x" figure is
wrong. But the honest accounting still favors holding the pod: the split's true incremental
billed cost is fresh-pod bring-up + `bootstrap_pod.sh` + the per-issue transformers==5.15.0
venv build + ~19 GB model re-download + 1.47 GB vc_bank re-stage ~= 0.5-1.0 billed hours,
plus RunPod reprovision capacity/wedge risk on the critical path. Net stake ~0.2-0.7 H100-h.
Rewrite the comparison honestly; the CONCLUSION survives, the "2x" figure does not.

### E3. Fix the L7 compound wall cell

§9's L7 wall cell `0.5 VM + <=24 calendar` parses as `0.5` through
`plan_wall_budget.parse_wall_cell`, dropping the 24h Batch SLA from the phase-ETA tripwire
budget (the #2162 under-fence shape). Rewrite the wall cell as `<=24 calendar` and move the
0.5 VM figure into the basis column.

### E4. (optional, cheap) cite a `scripts/resource_ledger.py` read for the VM phases

Claimed usage (<2-4 GB RSS, minutes of CPU) is far under the >70% routing thresholds, so
routing is unaffected — a one-line ledger citation makes it airtight.

---

## FOLD-IN — prose corrections and report-side requirements

### N1. Clustering-grain adjectives are wrong in two places

Plan §3 ("steered pair-clustered 95% CI") and §6 ("cells aggregate pairs via pair-clustered
bootstrap, seed 21626") mis-describe the inherited driver, whose docstring (line 18) and
`BOOT_SEED` comment (line 77) say CARRIER-clustered. The code executes correctly and
parent-comparably; only the registration prose is wrong. Fix the adjectives.

### N2. Report-side requirements to register now (so the report cannot omit them)

These are non-blocking for the plan but must be REGISTERED as report obligations:

1. CROSS-MODEL CAP CAVEAT (consistency-checker): the reused Qwen2.5 completions were
   generated at cap 2048, the fresh Qwen3.5 ones at 4096. The registered within-model decay
   contrast is cap-matched and unaffected; only the SECONDARY cross-model read carries the
   asymmetry, bounded by the parent's measured cap-hit of 1/1320 (~0.08%) and measurable per
   row via the `cap_hit` field. The cross-model decay panel carries an explicit caveat line.
2. STARTING-LEVEL ALIGNMENT (Codex methodology): Leg B's Q1-Q4 contrast can reflect unequal
   Q1 persona strength, floor drift, or different response-content composition rather than
   persistence. Do not claim equal raw drops mean patch and prompt install "the same thing"
   without checking starting-level alignment. Report the absolute Q1 gap alongside the
   contrast.
3. SHARED-RUNG SENSITIVITY (Codex methodology + Claude statistics): cross-model aggregates
   contain different gate-surviving rungs, so also report the decay contrast on the rung
   INTERSECTION before interpreting model differences.
4. FLOOR COMPRESSION (Claude methodology): steered raw scores start below ceiling's, so
   scale compression biases the contrast downward — conservative for a positive claim, but it
   can mask a real faster-decay effect as "generic decay". Read the contrast alongside the
   absolute Q1 gap.
5. MIN-LENGTH CENSORING (all three): the 48-token gate selects a non-random long-completion
   subset and may censor the steered arm asymmetrically. Report per-arm x model drop
   fractions AND retained-vs-dropped length/score patterns; if materially asymmetric, run a
   length-matched sensitivity read before narrating the headline.
6. BEST-OF-2 FRAMING (Claude statistics): Hypothesis 4's "on at least one model" is a
   best-of-2 at 95% per-model CIs (FWER ~9.75%). Report both models' CIs and narrate a
   single-model-only rejection as such.
7. POWER-ARTIFACT FRAMING (Claude statistics): a `no-clean-transfer` on Qwen3.5 for a
   parent-`transfers` cell can be a POWER artifact (wider CIs on the new model), not a
   reversal. Report the steered-minus-null gap CIs beside the categorical flip and use
   "indistinguishable from null given the variance", never "does not transfer".
8. COARSE-CI FRAMING (Codex statistics): with six carrier clusters the bootstrap intervals
   are coarse; keep that framing in the report rather than implying precision.
9. DESCOPE LEVERS (Claude methodology): descope levers (2)/(3) would cut scope-mandated
   elements (conjunct rubrics; null-arm K). They are legitimate as pre-registered
   contingency, but if pulled they DEVIATE from the binding "NO scope reduction" scope
   directive and must be recorded as an explicit deviation in the report.

---

## Manifest

Blockers S2 (slot in the row grain; CE-primary; pe stratified-or-excluded) and S3
(inconclusive branches) change what the Leg B figures plot and which conditions the primary
contrast covers. Update `artifacts/planned_manifest.json` accordingly, keeping every
PRE-EXISTING (parent-round) entry byte-identical. Flag in your return whether the Leg B
CONDITION-SET MEMBERSHIP changed — that determines whether the orchestrator must re-run the
plan-approval gate before implementation.
