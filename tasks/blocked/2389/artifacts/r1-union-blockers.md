# Round-1 unioned blockers — task #2389 plan v1 -> v2

Panel outcome: statistics REVISE (Claude 1 + Codex 2), methodology REVISE
(reconciler BINDING, both blockers retagged SUBSTANTIVE), efficiency REVISE
(Claude 2 + Codex 3, overlapping on the SLURM fence), consistency PASS.

**9 blockers, all SUBSTANTIVE.** None may be dropped or deferred. Every fix below
is plan-text only except where it pre-registers a probe/gate the implementer will
run. Fix ALL of them in ONE revision (plan v2) plus the manifest updates they imply.

---

## STATISTICS

### S1 [SUBSTANTIVE] Prediction 3's falsification branch mislabels an underpowered CI-straddle as "falsified"

§3 prediction 3 registers: "Falsified by rho <= 0 or a CI spanning 0." At 16 (or
13 — see S2) units a pair-clustered CI spanning 0 is compatible with a true
moderate transfer, so that branch converts a plausible true positive into a
registered NEGATIVE finding on a failure-to-reject read. It also contradicts §6
narration pre-commitment 10 ("underpowered nulls narrated as 'indistinguishable
from null given the variance', never 'confirms the null'") — both surfaces cannot
bind. Same defect fired and was fixed in this lineage at #2162 plan v7.

**Fix:** split the branch. Reversal/falsified IFF the CI EXCLUDES the
pre-registered positive reference (state the reference explicitly — e.g. excludes
rho >= 0.3, or lies entirely below 0). A CI spanning 0 whose upper bound sits
above that reference routes to **no-verdict / underpowered**, narrated as
failure-to-reject WITH the realized n. Zero compute delta.

### S2 [SUBSTANTIVE] The registered transfer read contradicts its own n>=12 eligibility rule — ORCHESTRATOR-VERIFIED

The manifest's `figures[id="transfer_correlation_16"]` transform requires n>=12
surviving pairs in BOTH runs, but #2162 has only **13** qualifying ce/P1 cells.
I recomputed this myself from `eval_results/issue_2162/f_metrics/f_cells.jsonl`
(steered arm, ce/P1, |separation| >= 0.5, distinct pair_id):

- 16 ce/P1 cells with >= 1 surviving pair
- **13** cells with >= 12 surviving pairs
- **3 cells BELOW the floor: `user_emotion` = 1, `icl_task_mapping` = 7, `refusal_boundary` = 8**

`user_emotion`'s SINGLE observation resampled in the inherited within-cell
bootstrap is treated as known without uncertainty — invalid, not merely
different. Enforcing the registered floor yields a different correlation AND CI
and can flip the secondary transfer verdict.

**Fix (Decision by the orchestrator — implement BOTH legs):**
1. **Registered PRIMARY = the >=12-eligible read at n=13.** Pre-register the
   parent-supported ceiling as 13, apply the SAME eligibility rule end-to-end
   (plan §3/§6 text, the manifest transform, and the transfer implementation),
   and report the REALIZED N rather than a hardcoded one.
2. **KEEP the all-16 read as an explicitly-labelled DESCRIPTIVE companion** (not
   a registered verdict-bearing read), because the task body AND the user's
   `goal:` frontmatter both name "the 16 shared context-end P1 cells" — the
   report must not silently substitute a different number for the one the user
   registered. Label it descriptive, state why (3 cells below the eligibility
   floor, naming them and their surviving-pair counts), and carry the >=12 read
   as the valid one.
3. Rename the manifest figure id away from the now-misleading
   `transfer_correlation_16` (e.g. `transfer_correlation_p1_ce`), keeping BOTH
   series in its `plotted_quantity`, and state the realized N in the caption
   recipe.

### S3 [SUBSTANTIVE] Gate 3's catastrophic HALT region overlaps the plan's own primary success criterion

§7 gate 3 HALTs when fewer than 25% of a SIX-pair-per-cell screen passes — i.e.
fewer than 10 of 38 sampled cells. But §3 prediction 1 CONFIRMS the primary on
**>= 9 cells** stored-and-used. So a run heading for exactly 9 causal positives —
a confirmed primary result — sits INSIDE the abort region and would be killed as
"instrument-broken bank". Separately, failing ">= 4/6" on a noisy six-pair screen
does not imply a cell will fail the registered n>=12 threshold over all 36 pairs,
so the instrument-broken inference is unsupported.

**Fix:** make the aggregate 25% slice **ADVISORY** (log the realized fraction,
surface it in the run digest, CONTINUE). Retain the full-anchor per-pair |sep| >=
0.5 exclusions and the n<12 untestable classification as the determinants of
scientific measurability. This matches the standing rule that null-side /
screen-side conditions default to advisory and a hard abort must argue itself.

---

## METHODOLOGY (reconciler-binding; full evidence chain in the
`epm:plan-critique-reconcile v1` marker on #2389 — read it for the file:line trace)

### M1 [SUBSTANTIVE] Anchor artifacts do not reach P7 in the inherited consumers' required layout

The fork's producer writes anchors to a DEDICATED `out_root/anchors/` dir as
`anchors_{batch}_w{i}.jsonl` + `va_anchors_{batch}_w{i}.pt` (co-located), and
`phase_upload` SPLITS them across two Hub prefixes (jsonl ->
`raw_completions/anchors`, pt -> `analysis_tensors/anchors`; the `va_store`
prefix carries GRID shards only). The P7 consumers glob `anchors_*.jsonl` AND
`va_anchors_*.pt` from ONE local `anchors_dir`, each with a fail-loud empty
assert, and the analysis CLI does no mirror resolution. The plan's §9 P7 reads
declare NEITHER anchor prefix, §10's upload-prefix enumeration omits
`analysis_tensors/anchors/`, and §6.5/§9 phase_outputs mis-name the artifacts
(`rollouts/anchors_shard_*.jsonl`, `va_store/anchors_shard_*.pt`) so those globs
match nothing the fork produces. Anchors define floor AND ceiling in F, so P7
dies at its entry asserts AFTER the full GPU + judge spend.

**Fix (five parts, all required):**
(i) correct §6.5 + §9 `phase_outputs` anchor globs to the realized layout
    (`anchors/anchors_*.jsonl`, `anchors/va_anchors_*.pt` pod-side);
(ii) add `issue2389_q38ce/analysis_tensors/anchors/` to §10's upload-prefix
    enumeration — AND sweep in the parent's early `raw_completions/anchors_gate/`
    prefix (the reconciler's additional finding: `_resolve_anchors_dir` prefers/
    covers it for gates 3-pre/3 and it is likewise absent);
(iii) add BOTH `raw_completions/anchors/` (jsonl) and `analysis_tensors/anchors/`
    (pt) to the §9 P7 `reads:` block WITH the two explicit Hub-prefix ->
    local-`anchors_dir` mappings (they must co-locate into ONE dir);
(iv) pre-register the artifact-reuse (h)(iv) 1-file staging probe through BOTH
    REAL loaders (`load_anchor_rows` + `_load_anchor_va`) with a
    `(context_id, draw)` key-set identity assert, run BEFORE P7 production;
(v) state explicitly that vLLM-engaged anchor cells STILL run the HF
    teacher-forced `capture_answer_states` pass producing their
    `va_anchors_*.pt` before their shards are marked complete — otherwise F_act
    anchors and the layer-59 `fact_select` inputs are missing for ~34 of 37
    cells whenever item 4 engages.

### M2 [SUBSTANTIVE] Item-5 equivalence gate certifies only FIRST-token identity

§4.7 item 5's acceptance battery checks "first-token logits match". All N draws
condition on the identical just-copied/expanded cache at decode step 1, so that
check is STRUCTURALLY BLIND to the aliasing class (a batch-expand view or shared
recurrent state mutated in-place from step 2 onward). **48 of 64 qwen3_5 layers
are linear-attention with fixed-size recurrent/conv state updated per decode
step** — exactly the state class §4.6's own smoke blind-spot (f) names as
unexercised, whose ONLY assigned coverage is this first-token check. Fail-open
protects against the gate FAILING, not against it PASSING a subtly-wrong branch;
a false PASS runs `share_prefill=True` across anchors + grid + stage-2 and
corrupts later-token distributions on every measured arm.

**Fix:** extend acceptance (b) to (i) per-step logits over SEVERAL continuation
tokens under deterministic decoding or fixed teacher-forced continuations
(CPU-tiny exact; pod bf16 at the calibrated tolerance), (ii) covering hooked AND
unhooked paths, n>1, and unequal-length (left-padded) batches, and (iii) an
explicit BRANCH-INDEPENDENCE test — perturb one draw's continuation and assert a
sibling draw's per-step logits are bitwise unchanged. Any mismatch keeps
`share_prefill=False`; fail-open semantics otherwise unchanged. Update §4.6's
blind-spot (f) entry to reflect the strengthened coverage.

---

## EFFICIENCY

### E1 [SUBSTANTIVE] Gate-4b parity-gate judge transport is unnamed and routes to the Batch API while anchor bulk blocks on its verdict

§9 assigns SYNC to "pilots + gate-3 slice (~10k calls)" only and sends ALL
production waves to Batch; the parity gate's ~4.3k vLLM-side calls fall outside
the sync set and its HF side is "production anchor spend", judged at P6 AFTER all
generation — i.e. after the verdict was supposed to pick the engine. The reused
dispatcher routes waves >= 4,000 items to Batch, so a naive dispatch batches.
Cost: EITHER the 8-GPU pod idles mid-P2 for up to the Batch SLA (hours to 24h at
~USD 19-24/h), OR the implementer silently forfeits item 4 entirely.

**Fix:** parity-gate judging (BOTH sides, ~8.6k calls) dispatches SYNC in the P2
window via `api_dispatch`, served back to the P6 anchor Batch wave through the
rubric-keyed JudgeCache — the same pattern the gate-3 slice already uses — and
anchor bulk proceeds work-conservingly on HF per-cell while the verdict pends,
with only REMAINING cells re-routed on a PASS.

### E2 [SUBSTANTIVE] The SLURM wall fence is irreconcilable with the plan's own accepted-wall band (both critics, independently)

`--time-budget-hours 40` is ~1.21-1.29x the mean-extrapolated ~31-32.15 h wall,
but gate 4 permits 2x phase fences and only REFUSES above 3x the §9 rows
(accepted band to ~91 h). The fence is fixed at SUBMIT time while the informing
pilot runs INSIDE the job at P2 entry, so a pilot landing in (~1.3x, 3x) is
ACCEPTED by gate 4 and then hard-killed by sbatch mid-P3/P4 on the fellows lane.

**Fix (either, plus the pilot-shape clause):** size `--time-budget-hours` >= ~64
(2x the booked pod wall, the p90 dispersion default) with >= 1.25x margin over a
phase-weighted p90, OR pre-register the resubmit-on-TIMEOUT split as the DESIGNED
continuation (the claim-queue resume + per-block incremental uploads already make
it lossless — it just has to be declared so a TIMEOUT is a planned phase boundary,
not a crash). Either way, ALIGN gate-4's acceptance band with the actual fence.
**Plus (Codex):** the gate-4 pilot must cover the THREE distinct realized shapes
— anchor-HF fallback, hooked sampled-grid, and greedy stage-2 — rather than
extrapolating all three from one hooked cell (the per-regime pilot-binding rule).

### E3 [SUBSTANTIVE] P1 capture bills 6 GPU-h to do 0.75 GPU-h of work

§9 runs P1 capture on ONE worker with "7 idle GPUs booked in contingency" for
~0.75 h — ~5.25 GPU-h wasted. The 1,404 captures are independent and the
work-conserving claim queue ALREADY EXISTS.

**Fix:** shard the 1,404 captures 8-way over the existing claim queue (preferred
— near-free), or provision P1 separately at one-GPU width. Also make gate 4b
explicitly CONCURRENT with gate-3-slice generation on the free workers rather
than serial before P2 (the HF parity leg IS P2 work).

### E4 [SUBSTANTIVE] ~95k iterative logistic fits are routed to a CPU-only pod

The P7 read-probe batches ~95k tiny logistic fits into <= 42 optimizer calls.
Vectorizing removes Python overhead but does NOT change their gradient-descent
compute character, so this is a GPU-worthy iterative-optimization leg per the
compute-sizing rules; on 16 vCPUs it risks overrunning the 3 h P7 booking.

**Fix:** split P7 — run the vectorized probe optimizer on a ONE-GPU lane, leaving
bootstrap, F tables, and transfer aggregation on `cpu-bigmem`. Pilot ONE
production-shape vectorized group on that GPU lane (measured basis, fence >= 2x).

---

## ALSO FOLD (non-blocking, all cheap; from the panel's agreed observations)

1. **Layer-TYPE asymmetry at the F_act comparison row (consistency-checker O1).**
   Layer 59 is a FULL-attention layer on the 27B (59 == 3 mod 4); the parent's
   comparison/selection layer 30 was LINEAR-attention on the 9B; the
   parent-convention alternative 61 would be linear here. The plan flags the
   POSITION ambiguity but not the TYPE flip — and §3 prediction 7 hypothesizes
   full-attention layers yield higher F, making layer type a LIVE MODERATOR of
   the labelled cross-model row and of the `fact_select` stage-2 survivor set.
   Fold: state each model's comparison-row layer TYPE next to the 3-model F_act
   row (9B linear; 27B full-attention), and add layer 61 as a labelled
   EXPLORATORY companion row (recomputable from the all-64 capture at zero
   marginal compute).
2. **Mixed-engine disclosure** (statistics + methodology + reconciler all agree):
   one Methodology line stating the realized per-cell engine split AND the
   realized parity offset next to the transfer figure whenever item 4 engages.
   Codex-methodology adds: behavioral-score parity does NOT establish parity of
   the F_act ANCHOR distributions — report engine-stratified F_act anchor
   diagnostics before interpreting the layer profile or stage-2 selection.
3. **(k) lineage record:** add two one-line "not-needed — superseded by main;
   #2329 production-validated main's copy" declarations for `origin/issue-2094`
   and `origin/issue-1415`, citing the commit SHAs.
4. **§7 null-gate N/A prose is inaccurate** though the construction is compliant:
   probe-positive branches the verdict lattice and hence the PRIMARY count, and
   is defined by clearing the max-selected permutation band. Reword to "no
   RUN-TIME gate thresholds a null statistic; the probe band is a measured
   selection-symmetric ANALYSIS-TIME criterion."
5. **Power-vs-capability confound in the count primary:** anchor separation rising
   with capability mechanically raises per-cell power at fixed true effect (F-space
   noise ~ 1/separation), so part of any 8->N count increase can be TESTABILITY
   rather than usability. The plan already registers the ingredients (per-cell MDE
   at P7, separation distributions vs 0.346/0.555, realized per-family m); state
   in §6 that the report reads verdict FLIPS jointly with each cell's
   separation/MDE shift.
6. **Holm harmonization:** the 27B's ce-only families are smaller than the parents'
   ce+pe families. Recomputing the committed 7B/9B ce rows with ce-only correction
   leaves 5/8 unchanged, so the baseline stands — but pre-commit to reporting those
   HARMONIZED recounts rather than comparing differently-corrected verdicts.
7. **Band-vs-ceiling:** place each probe permutation-band upper bound beside the
   AUC ceiling of 1.0 (required for bounded max-selected statistics).
8. **Within-Qwen lineage caveat:** a rise above 8 stored-and-used remains a
   within-Qwen model TREND, not an identified causal effect of capability
   (training-data and architecture changes are live alternatives). The plan
   acknowledges this; keep the caveat ADJACENT to the headline in §6's narration
   pre-commitments.
9. **Judge telemetry:** ~145k net calls sit below the 200k balanced-routing
   crossover and imply many ~1,000-request Batch passes (stuck-batch calendar
   risk; recoverable since no GPUs are held). Pre-commit to reporting the realized
   uncached count, number of batch passes, and any stuck-batch fallback. Also
   restate "ALL production waves Batch API" as "Batch above the dispatcher's size
   threshold" (the reused dispatcher routes <4,000-item waves sync; parent #2162
   realized 171 sync vs 9 Batch dispatches).
10. **P7 lane availability:** name the sanctioned 1x-H100 `eval`-intent pod as
    P7's accepted fallback (idle GPU as the recorded price) so a `cpu-bigmem`
    CPU-LANE-DRY refusal needs no dispatch-time deliberation.
