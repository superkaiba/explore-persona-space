---
description: Planner §9 compute-sizing recipes — activation-capture HBM sizing, merge-disk budget, sentinel-signaling lane pins, the floor cross-check for long / many-call phases (planned_wall_h > 4 OR >~500 serial calls), the measured 1-cell fit-pilot basis for per-cell fit / factorization / GD phases (#1060), the store-heavy / IO-heavy phase recipe (measured per-item serialization+upload wall-time; compression-default-OFF for fp16→Xet), the CPU-phase RAM/RSS routing gate (projected peak RSS per VM-placed phase; ≥~16 GB single-or-summed routes off the shared VM), the dose-ladder checkpoint-retention default (keep dose-selected + latest, clean ruled-out rungs between rungs; size disk to the RETAINED set, #1133), and costing wall-time against the machine the router will ACTUALLY provision + p90-based fence sizing, and the external-stream >~1h presumption for network-bound streaming/harvest phases (#1092) (loads at plan time via plan-file paths; relocated verbatim from planner.md §9, #829)
paths:
  - ".claude/plans/**"
  - "tasks/**/plans/**"
---

# Plan compute sizing (planner §9 relocated recipes)

These ten recipes are the planner-specific §9 sizing blocks — five relocated
verbatim from `.claude/agents/planner.md` (#829), plus the
store-heavy / IO-heavy phase recipe (#910, from incident #813), the
CPU-phase RAM/RSS routing gate (#1031, from incidents #778/#833), the
per-cell fit-phase pilot basis (#1060, from incidents #811/#931/#823), the
external-stream floor presumption (#1092 — present since #1092, first counted
here), and the dose-ladder checkpoint-retention default (#1133, from incident
#1112). The planner
applies each when its trigger matches; the compute-projection table spec +
stratification spec stay inline in planner.md §9.

**Activation-capture HBM sizing.** If any phase captures hidden states on a
7B model (residual streams at one-or-more layers, online activation
accumulation, per-token activation dumps), the chosen intent MUST clear ≥40
GB HBM, NOT the L4 `eval`/`debug` default — 7B bf16 weights are ~14 GB, and
all-layer hidden-state capture at a realistic batch × sequence pushes past
the L4's 16-GB-class HBM and OOMs the run mid-flight (#666, #744). The
canonical fit is `lora-7b` (1× A100-80) when the phase ALSO trains, or
`capture-7b` (1× A100-80, the activation-capture eval intent, #752) when it
is forward-pass-only; both fall back to the 40 GB A100-40 rung under A100-80
exhaustion. This is orthogonal to the VM-footprint carve-out below (which
sizes the off-pod analysis disk) — this rule sizes the GPU HBM the capture
forward needs on the pod. Mechanically BACKSTOPPED by `verify_plan.py`
check c27 for GCP/auto-lane eval/debug bookings (escape: `N/A — no 7B
activation capture`); RunPod-pinned plans and phase-to-intent routing
inside mixed-intent plans stay critic-owned.

A plan that quietly picks `lora-7b` (1× H100) for an embarrassingly parallel
20-condition sweep is wrong, even if the GPU-hours total is the same.


**Merge-disk budget — bound coexisting full-precision artifacts against
the per-pod quota.** Any phase that materializes full-precision model
artifacts DURING iteration — a LoRA adapter merged onto base weights for
a read (dose-checkpoint selection, eval that needs a merged dir), a
ZeRO-3-consolidated full-FT checkpoint, a per-step or per-cell model copy
— accumulates on-disk weight files that a sweep can blow past the per-pod
quota. The plan §9 MUST, for any such phase, state the upper bound on
COEXISTING on-disk full-precision artifacts —
`n_cells × max_concurrent_artifacts_per_cell × per_artifact_size_gb`
(a merged Qwen-2.5-7B is ~15 GB) — and verify it fits the per-pod disk
quota. On the RunPod lane that quota is the MooseFS ~130 GB per-pod cap
(`OSError errno=122 EDQUOT`; `df -h /workspace` shows the TB share, NOT
the per-pod limit — see `.claude/rules/gotchas.md` "RunPod MooseFS per-pod
disk quota"); on SLURM / GCP it is the per-node scratch budget. If the
upper bound exceeds the quota, the plan MUST specify the cleanup pattern —
which artifacts persist, which are transient, and WHEN each transient one
is deleted (cleanup-as-you-go / atomic merge-read-delete per probe /
scratch-dir rotation), so the high-water mark stays under the quota. A
plan that lets transient merges accumulate silently EDQUOTs mid-run
(#653 round 4: the `select_checkpoint` phase merged a ~15 GB
full-precision copy per probed dose checkpoint × 12 content cells × 9 dose
ckpts = ~1.6 TB worst case on a 130 GB quota, with no cleanup between
probes — the run died at the quota; the fix was atomic merge-read-delete
per probe). This is a plan-time storage-budget check, NOT a mid-run gate.
Per-rung checkpoint LADDERS additionally carry the retention DEFAULT of the
next block — for a ladder phase, a keep-all-rungs bound that happens to fit
the planned quota is no longer sufficient on its own.


**Dose-ladder / multi-rung checkpoint retention — keep the dose-selected +
latest rungs only; size disk to the RETAINED set, never the full ladder.**
Any training phase that persists per-rung checkpoints for later selection —
a dose-to-band checkpoint ladder (earliest-rung-in-band), a band-stop grid,
a dose-matching checkpoint grid, any long run saving every k steps for a
later pick — MUST state its checkpoint-retention policy in §9. DEFAULT:
retain only (i) the dose-selected rung(s) (or the current selection
candidate while the read is pending), (ii) the latest rung (crash-resume),
and (iii) rungs the selection read has not yet covered; every ruled-out
rung is deleted BETWEEN rungs, not in one sweep after the ladder completes.
Three implementations bound the on-disk ladder: an online per-rung
selection read (the deterministic log-prob band-stop callback is the
marker-recipe case — `.claude/rules/marker-training-recipe.md`);
upload-as-you-go then delete locally (select against the Hub copies,
re-download only the selected rung — pricing in Hub storage headroom, the
#541/#552 quota exposure); or a coarse+refine two-pass grid
(sparse saves that fit, judge, deterministic retrain to the bracketing
steps — #1112's own RunPod contingency). A design whose selection read
runs only AFTER the full ladder has no between-rung deletion point —
carve-out (iii) would cover every rung, degenerating the default to
keep-all — so a post-hoc-selection ladder MUST adopt one of the bounding
implementations above or take the justified keep-all exception below;
it cannot ride carve-out (iii). Deletion composes with the Upload
Policy UNCHANGED: a non-selected rung is deleted only when covered by a
plan §10 `discarded_artifacts:` entry ({name, reason, regen_recipe} —
non-selected rungs of a deterministic retrain from pinned data + commit +
seed are the canonical candidate, #1112's own declared discard) OR
uploaded first; selected / headline-carrying checkpoints keep the full
upload-before-delete invariant (never delete an unuploaded, undeclared
checkpoint). The §9 disk estimate sizes to the RETAINED set plus the
transient high-water mark (`retained_rungs × per_rung_gb +
in_flight_rung_gb + concurrent transients`), with `per_rung_gb` grounded
on what the trainer ACTUALLY writes — weights + optimizer state, so a
full-FT rung can run well past the bf16 weights (#1112 planned ~15 GB/rung;
realized rungs ran ≥15 GB, up to ~28 GB per the incident filing). Keeping
every rung locally is the JUSTIFIED
EXCEPTION, not the default: the plan must say why the rungs must coexist,
size the disk to the FULL ladder at realized per-rung size, and DECLARE
that requirement in the launch flags (`--boot-disk-gb`) so the #1118
volume-threading / typed refusal engages on a lane failover. A merge-disk
bound that fits the PLANNED lane's disk is NOT sufficient on its own
(#1112: a compliant 575 GB keep-all bound sat under the planned 750 GB GCP
boot disk; the GCP→RunPod failover delivered the `ft-7b` default 200 GB
volume and the run ENOSPC'd — errno 28 mid-safetensors-write — at rung
24/30: crash + terminate + a fresh 800 GB recovery pod, billing ~$16/hr
per the incident filing. The same design's retention-bounded footprint is
~2–3 rungs ≈ 30–85 GB and fits every lane). Critic enforcement:
Methodology lens item 16
(`.claude/rules/critic-lens-reference.md`) REVISEs a ladder plan whose
disk estimate assumes keeping every rung without this justification.
Mechanical backstop: `scripts/verify_plan.py` c33 (`c33_ladder_retention`)
WARNs an experiment|analysis plan carrying checkpoint-ladder vocabulary
whose compute-sizing sections state no retention vocabulary (escape:
`N/A — no per-rung checkpoint persistence`); surface-only — adequacy of a
stated policy stays with this lens.


**Sentinel-signaling workloads need a /workspace-contract lane — never
rely on auto's SLURM fallback.** If the plan's dispatch script posts
markers via pod-side sentinel files (`/workspace/logs/issue-<N>-*.json` —
gate sentinels, `epm:results` payloads), the plan MUST pin a lane that
honors that contract: `backend: gcp` (GCE instances mirror RunPod's
`/workspace` — `GcpConfig.vm_scratch_dir`) or an explicit
`backend: runpod` override with its residual gap named. Do NOT leave such
a workload on `auto`: a GCP capacity failure falls through to the SLURM
lanes, where compute nodes have no `/workspace` and the robot wrapper
cannot run the sentinel drain — the dispatcher fails loud at its
`mkdir -p /workspace/logs` and burns the SLURM submission (#608, commit
3022ff7bc). If the plan needs a SLURM lane, the dispatcher must use the
SLURM signaling contract instead — `status.json` heartbeat +
`[phase=...]` log lines (see `backends/slurm_monitor.py` module
docstring § "No sentinel drain on this lane"). State the choice in §9:
either the pinned lane + why, or "no sentinel dependence — auto-safe."


**Floor cross-check for long or many-call phases.** For any row with
`planned_wall_h` above 4, OR any row whose component executes more than ~500
serial calls of a non-trivial kernel (a fit / dense factorization
(svd/eigh/lstsq/GCV-ridge) / model forward / SERIALIZATION of a
multi-hundred-MB artifact / a per-file Hub commit — the call count is the §9
multiplier product `draws × cells × folds × …`, or the output-file count for
a store phase), state the arithmetic compute
floor next to the estimate
(`n_forwards × 2 · params · tokens_per_forward / sustained GPU FLOPs`, or
the analogous bound for the dominant kernel) and justify any >5-10×
estimate-over-floor gap — or name the implementation fix that closes it
(batched forwards, GPU-resident reductions, batched/Gram-space
factorizations; see `.claude/rules/code-style.md`
§ Compute-throughput discipline + `.claude/rules/vectorize-many-cell-fits.md`).
An estimate far above the floor usually
means the implementation is leaving throughput on the table, not that the
workload is big — fix the implementation, don't book more pod-days. The
call-count trigger exists because the wall-clock trigger keys on the row's
OWN estimate — exactly the number that is wrong when the per-call basis is
fabricated (#823: a ~3780-call full-SVD ridge phase planned at 0.35 h via an
asserted ~2 s/fit sailed under the old 12 h trigger while the measured
~125 s/call implied a ~131 h serial floor).
(#522: ~94h on 1× H100 for a job with a ~4-6h FLOPs floor; #511: 52×
CPU wall-time blowup vs its §9 estimate.)


**External-stream phases (network-bound row iteration) — the floor is presumed, not
sized.** A §9 row whose component consumes an external streaming source (HF `datasets`
`streaming=True`, API pagination, web harvest, S3/HTTP row iteration) has NO reliable
count × per-call sizing basis: the per-row kernel is trivial (~ms parse+filter) and
wall-time is network-throughput-bound, so both the wall-clock trigger and the
~500-call presumption above miss it (#1092: an intentionally unbounded full-corpus
stream scanned ~1.8M rows over 3h06m; its bounded verification twin used a
keep-quota stop — both shapes defeat count × per-call sizing). When the scanned-row
count exceeds ~10^4, is unknowable in advance (yield-dependent keep-quota stop), or
the pass is intentionally unbounded (full-corpus stream), the row is PRESUMED over
the ~1h checkpoint floor: state the scanned/kept volume targets instead of a
fabricated wall estimate, and name the per-chunk persistence + fingerprint-gated
resume mechanism (`.claude/rules/code-style.md` § "Checkpoint per
phase" external-stream presumption; review gate: code-reviewer.md Step 3.6). A short
bounded fetch (known ≤~10^4-row scan, fixed stop) is exempt.


**Per-cell fit phases — the per-call basis MUST be a MEASURED 1-cell pilot
through the production entrypoint; an asserted per-call cost is NOT a
sizing basis.** Any §9 row whose component loops a fit / solve /
factorization (ridge / SVD / eigh / lstsq / GCV / gradient-descent probe —
the same kernel family as the floor cross-check's call-count trigger) over
cells × folds × layers × arms × traits × seeds MUST ground its per-call
`basis` on a MEASURED 1-cell/1-unit pilot timing at PRODUCTION shape, on
the machine/device the phase will actually run, executed THROUGH the
production entrypoint (one full cell/unit end-to-end — every kernel the
per-cell path touches), OR a cited prior-issue MEASURED figure for the
SAME kernel + shape. Projected wall = `n_calls × measured_per_call /
parallelism`, stated in the row. TRIVIALITY EXEMPTION — never
self-certified by an asserted cost: a row may skip the pilot ONLY when
total_calls ≤ ~500 AND its sub-floor (~15–30 min) projection is computed
from a MEASURED or prior-issue-CITED per-call figure; an ASSERTED
per-call cost can never exempt a row, because the projected wall is
exactly the number that is wrong when the basis is fabricated (#823's
asserted ~2 s/call would project a 25-call row at ~50 s where the
measured ~125 s/call implies ~52 min — over the floor). Three failure
modes this closes: (i) the
asserted basis — #823 planned a ~3780-call full-SVD ridge phase at an
asserted ~2 s/fit (0.35 h) while the named fast twin's own docstring said
~125 s at that shape; 12–20 h realized. (ii) the PARTIAL measurement —
#811 timed ONE inner kernel (the batched bootstrap refit, 0.276 s/refit)
and asserted the surrounding ridge-LOCO headline path at "~1–2 h"; the
unmeasured path was the dominant frame (py-spy:
`_press_loo_mse_per_lambda`), and the phase sat at unit 3/108 after
19h21m — realized-extrapolated wall ~700–1000 h vs the 5.0 h
projection — before a second deviation forced the
pivot; the round-2 fix then measured 313 s/unit END-TO-END on the
production entrypoint, which is the pilot this block requires up front.
(iii) the wrong-device measurement — #931's battery basis was measured on
an A100 GPU fit while the realized armC cells ran CPU-heavier on the
shared VM (~2.2–2.5× elapsed-vs-plan mid-run). A FLOP/kernel floor is the
CROSS-CHECK (block above), never the basis — these loops are
overhead-bound, so a FLOP floor under-sizes them by construction
(`.claude/rules/vectorize-many-cell-fits.md`). When the pilot cannot run
at plan time (its inputs don't exist yet), the plan pre-registers the
pilot as the phase's FIRST step with an abort threshold — pilot ×
n_calls / parallelism re-projected against the row; >2× the row ⇒ the
vectorize signature check fires before the loop proceeds — and marks the
row's basis `pilot-gated`. This is the fit-phase twin of the store-heavy
block's measured one-item rule below.


**Store-heavy / IO-heavy phase sizing — measure one item's serialization +
upload wall-time; compression defaults OFF for fp16 → Xet.** Any phase that
WRITES >~10^3 output files OR >~50 GB total (per-cell activation stores,
per-context tensor dumps, raw-completion shards, per-cell result files) MUST
ground its §9 row's `basis` on a MEASURED one-item serialization + upload
wall-time at PRODUCTION shape (a timed write+commit of one production-size
item, or a prior-issue measured figure — either way measured with the SAME
serializer/format + destination the phase will actually run) —
bytes/file-counts alone are NOT a sizing basis: per-item wall-time is
dominated by serialization CPU and Hub-commit overhead, not byte count, so a
bytes-only gate passes exactly the phase that blows up (#813: a one-cell
GO/NO-GO gate measured per-file BYTES and passed, while `np.savez_compressed`
cost 103.8 s/file — 65% of the ~160 s wc_long row wall-time, vs 1.2 s for
plain `savez` — and drove the store phase 4.5× over plan on an idle 8×H100).
Client-side compression (`np.savez_compressed`, gzip) of fp16 activation
tensors bound for a Xet-backed HF repo defaults OFF: Xet chunk-compresses and
dedupes server-side (already −59% on the #813 upload), so client zlib bought
a 1.29× ratio for ~86× the write time; a plan that turns compression ON
states the measured ratio + per-item wall-time that justifies it. And
per-file Hub commits are the same per-item-overhead trap — batch into one
`upload_folder` commit (Upload Policy). The projected store wall
(`n_items × measured_per_item_cost / parallelism`) enters the row's basis,
and when it crosses the floor-cross-check triggers above, that check applies
to the serialization kernel exactly as to a fit kernel (the
implementation-side twin — write the cheap format per row, compress/upload
out-of-band or batched — is `.claude/rules/code-style.md`
§ Compute-throughput discipline).


**CPU-phase RAM/RSS routing — state each VM-placed CPU phase's projected
peak RSS; ≥~16 GB (single phase OR summed concurrent phases) routes OFF
the shared VM.** The shared VM's RAM is fleet-shared (~128 GB total) and
policed by `earlyoom`: SIGTERM to the highest-badness process below 10%
MemAvailable (~12.8 GB), SIGKILL below 5%, with a `--prefer` +300 bonus on
every python process — under typical fleet pressure (~95 GB resident) a
multi-GB analysis phase is the default kill victim, and the kill is SILENT
(no traceback — `.claude/rules/gotchas.md` "VM `earlyoom` SIGTERMs bulk
in-memory tensor loads"). Any §9 CPU/analysis phase placed on the VM MUST
state its projected peak RSS in the row's `basis` — a measured one-chunk
`ru_maxrss` at production shape, or `resident_pool_bytes × live_factor`
with the live-factor MEASURED, never the explicit-temporary count
(gotchas.md "Memory caps for torch fit loops"). Route the phase OFF the
VM — `cpu-mid` (GCP 32 GB) or `cpu-bigmem` (128 GB) — when projected peak
RSS ≥ ~16 GB, OR when concurrent VM-resident phases' SUMMED projected RSS
crosses the same ~16 GB bar (#833: two ~13-15 GB phases concurrently
resident lost 5 cells to earlyoom — concurrent residency SUMS; #778: a
22-GiB-RSS null battery was earlyoom-killed 3× on the starved VM —
~128 GB total, ~95 GB resident — before its cpu-bigmem pivot; #1092
2026-07-08: a fit-grid pilot's real per-unit RSS ran ≥22.9 GB against a
plan projection of 8–10 GB — a ~2.3–2.9× underestimate — and was
earlyoom-SIGTERMed on the shared VM before the pre-registered cpu-bigmem
escape lane recovered it: a fit-grid RSS projection is presumed
underestimated until pilot-measured on ONE unit at production shape, and
a projection within ~3× of the VM routing bar routes straight to
cpu-bigmem rather than piloting on the shared VM). A routed
phase sizing ≥16 GB MUST state `--min-ram-gb` in its launch row
(CLAUDE.md's ADOPTION sentence says '>16 GB'; this rule's ≥ closes the
exactly-16 GB edge — the RunPod `cpu-mid` fallback has exactly 16 GB, zero
headroom — the stricter bar governs) — the flag is what arms the #1010
footprint-feasibility gate (a flag-less launch can land on an undersized
pod — `.claude/rules/compute-backend-failover.md` § Footprint feasibility
gate; CLAUDE.md § Pods CPU-intents ADOPTION note) — or target `cpu-bigmem`
directly. STREAM-REDUCE FIRST: when a stream-reduce / chunked formulation
bounds peak RSS at O(one item), that is the fix, not a bigger machine (the
RAM sibling of "vectorize before routing to GPU"). Runtime choom
protection (`/issue` SKILL.md § "Detached VM-side long compute phases") is
mitigation for sanctioned sub-threshold phases, never permission to place
a ≥16 GB phase on the VM — #778's battery died 3× despite the
earlyoom-aware relaunch loop. This is the RAM sibling of the >50 GB disk
carve-out (`VM_ANALYSIS_FOOTPRINT_GB_MAX`); the two gates are ORTHOGONAL —
a VM-placed phase must clear BOTH. Accepted residual (named, not assumed
away): the summed-concurrency gate sees ONE plan/issue's phases —
cross-SESSION stacking of compliant sub-threshold phases can still cross
the fleet's earlyoom headroom; the watcher + earlyoom telemetry remain the
runtime backstop. Plan-time placement, not a mid-run gate.


**Cost wall-time against the machine the router will ACTUALLY provision —
then reconcile worst-case wall against the GCP auto-delete fence.**
Each row's `planned_wall_h` + `basis` MUST name the machine type of the
lane the backend router will most likely route. Under the standing
GCP-FIRST `auto` default that is the GCP intent mapping
(`INTENT_TO_MACHINE` in `src/explore_persona_space/backends/gcp.py`:
`lora-7b` → 1× A100-80 `a2-ultragpu-1g`, `ft-7b` → 4× A100-80,
`eval`/`debug` → 1× L4) — NOT the RunPod H100 intent table. A basis
measured on a different GPU must be scaled with a stated per-step rate
(e.g. "H100 basis × ~6× A100 step-time" — #599's trainer ran ~6× slower
per-step on the A100 auto-lane, turning an H100-premised ~6.4h estimate
into ~34h). Mechanically backstopped (WARN-only, heuristic) by
`verify_plan.py` c26; the semantic adequacy of a stated scaling factor
stays critic-owned. Then reconcile the WORST-CASE wall — base phases PLUS every
conditional / extension phase that could run on the same provision —
against the GCP lane's auto-delete fence
(`--instance-termination-action=DELETE` + `--max-run-duration`,
default 7d — the FLEX_START ceiling, #741).
Mechanically backstopped (WARN-only, heuristic) by `verify_plan.py` c29
(a DECLARED fence — `--max-run-duration` flag or `max_run_duration`
assignment — plus a §7 extension gate ⇒ the fence-reconcile sentence must
reference the conditional phase); prose-only / dispatch-time-only fences
(the #599/#833 shapes) and the arithmetic's correctness stay critic-owned.
Size that worst case off the
**p90 per-cell wall estimate, NEVER the mean**:
`worst_case_wall ≈ n_cells × p90_per_cell / parallelism + fixed overheads`,
with p90 derived (i) from a prior-issue per-cell wall DISTRIBUTION for the
same kernel/shape when one exists (cite it), else (ii) as the measured mean
basis × a STATED dispersion factor — default ×2 when only a mean is
available (#833: realized per-cell wall ran ~2× the plan mean and overran a
deliberate 36h fence, hard-deleting the instance before the tail cells
finished; the ×2 default is a conservative single-incident heuristic —
#833's ~2× is a realized-vs-planned MEAN overrun, so the factor absorbs
estimate bias AND tail dispersion, not a calibrated p90/mean ratio — a
prior-issue distribution wins whenever one exists). A deliberately
shortened `spec.extra["max_run_duration"]` MUST clear the p90-based worst
case with stated margin (≥~1.25×). p90, not max/p99: the fence doubles as
the GCP janitor's reap bound (own fence + 1h grace — `backends/gcp.py`), so
an over-long fence delays the credit-leak backstop on a wedged instance,
while a mean-sized fence kills healthy tail runs — p90 × margin is the
balance. If worst-case wall on the routed
machine approaches the routed lane's fence (the GCP `--max-run-duration`
default is 7d, but a plan may deliberately set a shorter fence), the plan
MUST do one of: (a) declare a deliberate `spec.extra["max_run_duration"]`
for the GCP dispatch; (b) pre-register a phase split across provisions —
name which phases run on a second provision and what artifacts must be
persisted (HF / git per the Upload Policy) before the first instance
dies; or (c) take the explicit `backend: runpod` override with the
long-run residual gap named (`/issue` SKILL.md Step 6b residual gap (d)).
A plan that silently lets a conditional phase ride past the fence loses
the phase mid-run (#599: the pre-registered §7.3 extension probe was
hard-deleted at step 149/2400 by the 24h fence).
