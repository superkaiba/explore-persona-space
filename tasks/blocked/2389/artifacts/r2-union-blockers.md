# Round-3 revision brief — unioned round-2 blockers (plan v2 -> v3)

Task #2389, workflow v2, adversarial-planner-v2 CRITIQUE mode, Step 3, round 3 of 5.

**Status of this file: statistics + methodology + consistency SETTLED. Efficiency
lens reconciler IN FLIGHT — its items append below as section E before dispatch.**

## Round-2 outcome summary

All NINE round-1 blockers (S1-S3, M1-M2, E1-E4) were verified FIXED by their
lenses, traced to code/artifacts rather than the planner's prose. Round 2 raised
**four** new blockers. Three of the four were INTRODUCED BY round-1's own fixes —
the verification round doing exactly its job.

| Lens | Claude r2 | Codex r2 | Binding outcome |
|---|---|---|---|
| Statistics | REVISE (N1) | PASS | **reconciler: REVISE — N1 UPHELD [SUBSTANTIVE]** |
| Methodology | REVISE (N1) | REVISE (N1, disjoint) | **union, no reconciler — 2 blockers** |
| Efficiency | PASS | REVISE (2) | reconciler in flight |
| Consistency | PASS | — (no twin) | **PASS** (O1 folds into M-N1) |

No resurface/block trigger tripped: zero GPU-hour delta (253 h stands), manifest
condition set unchanged, cap not reached, no REJECT.

---

## S. STATISTICS — 1 blocker (reconciler-binding) + 2 fold items

### S-N1 [SUBSTANTIVE] Split the rule-26 pilot transport clause per pilot point

`plans/v2.md` §6 L167 registers BOTH rule-26 judge pilots under ONE parenthetical
carrying `forced-batch pilot transport`, while the SAME SENTENCE names the wave
gate 3-pre gates as "the ~9.1k **sync** slice". The two clauses in that
parenthetical — "mirrored 1:1 from the production dispatch" (which computes SYNC)
and "forced-batch pilot transport" (which pins BATCH) — are jointly
unsatisfiable for gate 3-pre. The sync route of both gated waves is confirmed at
four other plan locations (§7 gate 3-pre L230, §7 gate 3 L231, §7 gate 4b L233 /
§4.7 item 4(iii) L133, §9 L304).

`.claude/rules/llm-judging.md` rule 26(c) L468-478 requires the pilot to RUN the
wave's transport, forced onto the computed route, with the realized route read
back and a FAIL on mismatch; L579-581: "a prior PASS does NOT cover a transport
change". Honoring the forced-batch token leaves both >=5k P2 sync waves
transport-uncertified AND runs the pilot on the one transport carrying the
api-refusal censoring channel (rule 26(d) L495-498, #1739: 34.1% batch-path
censoring vs 0/14,887 sync re-refusals) — a false-FAIL path into §7's kill
criterion (L237, "gates 3-pre/5/6 fail after one instrument round -> halt judge
spend") for an artifact the sync production waves structurally cannot have.

**FIX:** split the transport clause per pilot point at §6 L167 — gate 3-pre
wave-declared with the SYNC dispatch (`wave_force_sync=True` => pilot forced
SYNC); gate 6 keeps forced-batch (correct for the P6 bulk Batch waves,
deterministic batch at ~tens of thousands of calls vs `threshold_base` 2,000).
**Zero compute delta.** Not strippable — fits none of the mechanical categories
(marker-shape / smoke-run-missing / git-provenance).

### S-F1 (fold) §9 L304 pilot accounting mis-places the gate-6 pilot

L304 lists "rule-26 pilots ~0.9k" inside the P2-window SYNC set, but the gate-6
pilot runs at P6 forced-batch per L118/L235. When applying S-N1, make L304
consistent: only the ~0.44k gate-3-pre pilot belongs in the P2 sync set.

### S-F2 (fold) Slice-count wording inconsistency

§7 gate 3-pre says "~8.7k-call sync remainder"; §7 gate 3 and §9 say "~9.1k".
Consistent if the ~0.44k pilot draws are netted out of the slice — add one
clarifying word so the reader does not have to infer it.

---

## M. METHODOLOGY — 2 blockers (both REVISE, unioned, no reconciler)

### M-N1 [SUBSTANTIVE] Correct the §10 check-(i) record and schedule the probe device seam as FORK WORK

Plan §10 states the probe is "device call-time-parametrized, so the E4 GPU
routing needs no source change"; §4.6 L116 repeats it; the (m) row claims the
smoke "binds the E4 routing" via "the GPU device param".

**There is no such parameter.** Confirmed independently THREE times — by the
orchestrator, the Claude methodology-critic, and the consistency-checker (its
O1): `grep -c "device" scripts/issue2329_analysis.py` returns **0** (no
`--device` flag, no `set_default_device`, no `.cuda()`, no `.to(`); `step_probe`
loads the bank `map_location="cpu"` (L1049); `kernel_logistic_auc` (L942-993)
creates its dual coefficients and Adam state via bare `torch.zeros(...)` on the
CPU default; `_vp_data` builds the Gram from the CPU bank.

The round-1 E4 "route P7g to a GPU lane" fix is therefore **INERT**: the fork
either grinds the ~95k-fit battery + B=1,000 permutation battery on the eval
pod's vCPUs while the H100 idles — the #763/#812 inherited-CPU-pin class that
artifact-reuse check (i)(2) names a FAIL, and plausibly WORSE than v1's
`cpu-bigmem` placement since a 1xH100 eval pod is not provisioned for CPU width —
or a naive gram-to-cuda move crashes on device mismatch against the CPU-created
optimizer tensors.

**FIX (plan text + declared fork work; check (i)'s item-9 remedy — a check-(i)
failure routes to a SOURCE-MODULE fix SCHEDULED IN THE PLAN, never a false PASS
record):** correct the §10 (i) verdict to FAIL-leg-(2) for this consumer; drop
the false "call-time-parametrized" claim from §4.6 L116 and the (m) row; name the
device seam as fork work — thread a `device` parameter / `--device` flag through
`step_probe` / `kernel_logistic_auc` / `_vp_data`, moving gram, labels, fold
masks, and the `torch.zeros` allocations onto it — and keep the already-planned
(m) 1-group smoke at P7g entry ON the production device class as the binding
check. Measurement-neutral (observed AUC and its permutation band share the venue
either way).

### M-N2 [SUBSTANTIVE] Certify item-5 branch independence on the PRODUCTION device, exactly

The round-1 M2 fix added acceptance (e) — perturb one draw's continuation, assert
a sibling's per-step logits bitwise unchanged — but confines the EXACT assertion
to the CPU-tiny rig, permitting a tolerance-based spot-check on the real 27B
bf16/CUDA path (§4.6 L124 blind spot, §4.7 item 5 L134).

A CUDA-specific shared recurrent/conv-cache mutation could therefore PASS gate 4b
and alter later-token distributions across anchors, grid and stage 2 — corrupting
every headline arm, since `share_prefill=True` would then run on all of them.
Applying a tolerance on the independence check specifically can admit small
cross-branch contamination that later recurrent updates amplify. Sibling
independence IS exactly checkable on the pod: the sibling's inputs, shape and
operation sequence are identical between the perturbed and unperturbed runs.

Confirmed premise: `PositionEditHook._edit_tensor` returns `hidden` unchanged
once `_prefill_seen` is set (`experiments/issue2094/hooks.py` L184-186 — "decode
steps (and any later forward) pass through"), so a shared mutable recurrent state
diverges from decode step 2, well inside the K_eq=8 window. The serial reference
re-arms and regenerates each draw independently
(`experiments/issue1415/steering.py` L453-466), so production-device independence
must be compared against that property exactly.

**FIX:** add the same perturb-one-draw test on the production pod requiring EXACT
sibling-logit (and where reachable, cache-byte) equality through steps 2-8; keep
tolerance ONLY for the shared-prefill-versus-serial numerical-equivalence leg
(acceptance (b)), not for the independence leg (acceptance (e)). If a fused
kernel is nondeterministic, FIRST measure identical-run nondeterminism and then
use direct sibling cache-byte / storage-isolation assertions — never widen the
independence threshold to accommodate it. Fail-open semantics unchanged (any
mismatch => `share_prefill` stays off, run proceeds serial). Update the §4.6
blind-spot (f) text to match, per `.claude/rules/smoke-blind-spots.md`.

Note: the Claude critic judged M2 FIXED on the argument that the aliasing class
is device-independent code structure (view-vs-copy in the cache handoff) so
CPU-exact is the right instrument. Both readings are defensible; the union
adopts the tightening because the cost is one pod-side assertion and the
downside is silent corruption of every measured arm behind a fail-open gate.

---

## E. EFFICIENCY — 1 blocker (reconciler-binding); Codex's second REJECTED

### E-N1 [SUBSTANTIVE] Bind the parity-judging SYNC route mechanically, assert the realized route, persist it

Plan v2 declares the parity wave SYNC in SEVEN places (L52, L118, L133, L167,
L230/231, L233, L304) but binds the route NOWHERE — a grep of `plans/v2.md` for
`force_path` / `cost_pref` / `deadline` / route-assert returns zero binding hits.
The plan's OWN routing arithmetic contradicts its declaration: §6 L167 "effective
threshold = 2x `threshold_base` 2,000" and §9 L304 "waves >= ~4,000 items batch"
put both the ~8.6k parity wave and the ~9.1k gate-3 slice in batch territory
under defaults. Orchestrator-confirmed: `SYNC_BATCH_CROSSOVER_N = 2_000` and
`decide_dispatch_route` (default `cost_pref="balanced"`, no deadline) returns
`"batch"` for `n_items >= crossover_n`.

The reconciler found one narrowing fact Codex missed: the inherited pattern the
plan cites by analogy IS bound in the parent rig —
`scripts/issue2329_judge.py:862` passes
`threshold_base=FORCE_SYNC_THRESHOLD_BASE` (`= 10**9`, L93) at the gate-3-slice
call site. But **item-4 parity judging is NEW FORK CODE with no existing call
site**: the binding exists only by analogy, the plan never names the forcing
argument, and nothing asserts or persists the realized route. A plain
`dispatch_calls` / `judge_graded` call silently flips to batch — and the failure
is SILENT BY THE PLAN'S OWN DESIGN, because work-conserving overlap means a
Batch-SLA-delayed verdict quietly arrives after every anchor cell is already done
on HF. The plan states the consequence itself at L133 ("either idle the pod
against the Batch SLA or forfeit the item"): the parity gate's ~4.3k vLLM-side
judge calls + ~1-2 GPU-h of parity generation become spend on an instrument that
structurally cannot fire, and the approved item-4 throughput path is forfeited.

**FIX (one plan line, naming the real mechanisms):** bind the parity-judging
dispatch to the inherited forcing seam —
`threshold_base=FORCE_SYNC_THRESHOLD_BASE` per the gate-3-slice call-site pattern
(`issue2329_judge.py:862`), or `force_path="sync"` on a direct `dispatch_calls`
invocation (verified real: `api_dispatch.py` L1505 signature; L610-611
`force_path` overrides everything) — AND assert the realized route is sync,
persisting it in `vllm_parity_report.json`. Keep the declared JudgeCache root /
fingerprint hand-off unchanged. **Cheap extension the reconciler recommends:**
name `FORCE_SYNC_THRESHOLD_BASE` for the WHOLE declared ~18.6k SYNC set in the
same line, closing the seam once rather than per-wave.

### Codex efficiency blocker 2 (auto-lane out-root mount binding) — REJECTED, do not act on it

The reconciler rejected it with evidence; recorded here so round 3 does not
re-open it. The alleged harm ("die after substantial generation, forfeiting most
of ~246 wide-pod GPU-h") has no realized path: on nibi/fir/mila the failure is at
BOOTSTRAP (no `/workspace`, fail-loud at `mkdir`, #608) and the 56 GB weight pull
targets `HF_HOME` on `/workspace/.cache`, so zero generation exists to forfeit;
on the two `/workspace`-bearing lanes the plan sizes both explicitly (§9 L308) and
already mandates `assert_out_root_headroom` at each phase entry (verified real at
`orchestrate/preflight.py:1199`); and §9 L262 explicitly ACCEPTS the DRAC/Mila
fall-through per user ruling 4 (`backend: auto`), so Codex's proposed "exclude
unsupported lanes" fix would contravene a recorded user ruling. Codex self-tagged
this MECHANICAL while alleging a compute-forfeiture harm — a tag-discipline
inversion of the round-1 under-tagging pattern.

### Seam relationship — apply S-N1 and E-N1 in ONE pass

E-N1 is the BINDING side of the same defect S-N1 covers on the DECLARATION side.
Both reconcilers independently converged: the efficiency reconciler confirmed the
gate-3 slice's realized route is mechanically SYNC in the inherited rig, so a
wave-declared gate 3-pre pilot mirroring "its wave's actual route" mirrors SYNC,
not batch — corroborating S-N1's direction. Round 3 must apply the declaration
split and the route binding consistently together; fixing one alone leaves the
defect half-closed.

---

## Advisories to carry (NOT blockers — for the plan's report/analyzer
pre-commitments, not the revision's blocking set)

**Statistics:** verify per-group realized permutation bands sit meaningfully
below AUC 1.0 before reading any probe-negative into the quadrant lattice (name
it per affected cell); hold the cross-model comparison narration to layer 59
(pre-commitment 6) and never quote whichever of 59/61 looks better; confirm the
fold-6 Holm recount arithmetic at analysis time rather than inheriting the
round-1 figure; if realized transfer eligibility collapses to 1-2 cells, route
the transfer read to no-verdict regardless of a mechanically extreme Spearman;
describe rho_ref = +0.3 as the registered practical reference, not a universal
boundary.

**Methodology:** the cell-bucketed chunk ids must make vLLM-written and
HF-written anchor shard names structurally DISJOINT — `_load_anchor_va`'s
duplicate-key assert catches key duplication across shards but NOT a
same-filename overwrite (for the plan-adherence / code-correctness critics to pin
in the diff); state the mixed-transport anchor offset (~0.5 point, #1739
precedent) next to the fold-2 mixed-engine line; consider one hooked variant of
the branch-independence probe on the CPU-tiny rig (nearly free).

**Efficiency:** the ~145k production judge workload routes to Batch even though
the inlined balanced-routing table places Sonnet's crossover near 200k and warns
against more than 2-3 batch submissions — reconcile the chosen cost preference
and expected batch-pass/wedge count rather than describing it as balanced
routing.
