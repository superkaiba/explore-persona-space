---
description: Planner §9 compute-sizing recipes — HBM/disk/RAM/wall-time sizing, measured 1-cell pilot bases, p90 fence sizing, sentinel lane pins, down-width split (loads at plan time via plan-file paths; relocated from planner.md §9, #829)
paths:
  - ".claude/plans/**"
  - "tasks/**/plans/**"
---

# Plan compute sizing (planner §9 relocated recipes)

These recipes are the planner-specific §9 sizing blocks (relocated from
`.claude/agents/planner.md`, #829, and extended by
#910/#1031/#1060/#1092/#1133/#1414/#1541/#1612/#1633). The planner applies
each when its trigger matches; the compute-projection table spec +
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
exhaustion — UNLESS the phase's per-GPU peak exceeds ~38 GiB (e.g. an HF
capture model co-resident with a vLLM engine, #1315: ~17 GiB resident +
0.6 × 39.49 ≈ 41 GiB honest peak), in which case a co-resident HF+vLLM
capture phase MUST declare `--min-gpu-mem-gb <peak>` (the honest co-resident
per-GPU peak in GiB, never the card size) in its launch command so the
ladder skips the 40 GB rung (`gcp.A100_40_USABLE_GIB` = 38, #1468).
This is orthogonal to the VM-footprint carve-out below (which
sizes the off-pod analysis disk) — this rule sizes the GPU HBM the capture
forward needs on the pod. Mechanically BACKSTOPPED by `verify_plan.py`
check c27 for GCP/auto-lane eval/debug bookings (escape: `N/A — no 7B
activation capture`); RunPod-pinned plans and phase-to-intent routing
inside mixed-intent plans stay critic-owned.

A plan that quietly picks `lora-7b` (1× H100) for an embarrassingly parallel
20-condition sweep is wrong, even if the GPU-hours total is the same.


**Ladder-rung RAM floor — declare `--min-ram-gb` whenever the per-leg peak
RSS exceeds the smallest reachable rung's host RAM.** The `--min-gpu-mem-gb`
capture-phase floor above has a HOST-RAM sibling: any dispatch — and
especially any fan-out — whose per-leg peak RSS estimate exceeds the
smallest reachable ladder rung's host RAM
(`gcp.MACHINE_RAM_GIB["a2-highgpu-1g"]` = 85 GiB) MUST declare
`--min-ram-gb <per-leg peak RSS>` on its launch command: the #1998 rung
guard walks past undersized machines ONLY when the flag is present, and a
flag-less fan-out silently lands on whatever rung capacity serves (#1739
wave-1: 5-6 of 12 GCE legs rc=137 OOM after the spot rung downgraded half
the fleet to 85 GB-RAM `a2-highgpu-1g` boxes). Key the declared value on
the LARGEST cell/lane (§ CPU-phase RAM/RSS routing, LARGEST-CELL KEYING —
the #1739 host-RAM incident is that section's own worked example).
Mechanically backstopped (WARN-only) by `verify_plan.py` c52
(`c52_fanout_ram_floor`) for PLAN-EMBEDDED `dispatch_issue.py launch`
commands — both dimensions: a missing `--min-ram-gb` under a declared
per-leg RSS peak > 85 GiB, a missing `--min-gpu-mem-gb` under a declared
per-leg VRAM/HBM peak > 38 GiB, and a present flag strictly below its
declared estimate. This prose ALSO binds driver-script / teammate fan-outs
that never embed a `dispatch_issue.py launch` line in the plan — the
actual #1739 wave-1 dispatch channel, structurally invisible to c52 (its
residual (ii)) — so there the rule IS the coverage, the check is only the
mechanical backstop for plan-embedded launches, and a c52 SKIP is never
read as coverage.


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
(#653: ~15 GB per probed checkpoint × 12 cells × 9 ckpts = ~1.6 TB worst
case on a 130 GB quota; the fix was atomic merge-read-delete per probe).
This is a plan-time storage-budget check, NOT a mid-run gate.
Per-rung checkpoint LADDERS additionally carry the retention DEFAULT of the
next block — for a ladder phase, a keep-all-rungs bound that happens to fit
the planned quota is no longer sufficient on its own. And an N-cell fan-out
that RETAINS each cell's outputs after the cell completes additionally
carries the fan-out end-of-run accumulation block below — a transient
high-water bound alone misses monotone retained accumulation (#1481).
Ladders and fan-outs alike additionally carry the phase-ordering
high-water block below — the reap's PLACEMENT in the implemented phase
sequence, not just its existence, is what bounds the high-water (#1586).


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
realized up to ~28 GB). Keeping every rung locally is the JUSTIFIED
EXCEPTION, not the default: the plan must say why the rungs must coexist,
size the disk to the FULL ladder at realized per-rung size, and DECLARE
that requirement in the launch flags (`--boot-disk-gb`) so the #1118
volume-threading / typed refusal engages on a lane failover. A merge-disk
bound that fits the PLANNED lane's disk is NOT sufficient on its own
(#1112: a compliant 575 GB keep-all bound fit the planned 750 GB GCP boot
disk; the GCP→RunPod failover delivered a 200 GB default volume and the
run ENOSPC'd at rung 24/30 — the same design's retention-bounded footprint
is ~2–3 rungs ≈ 30–85 GB and fits every lane). Critic enforcement:
Methodology lens item 16
(`.claude/rules/critic-lens-reference.md`) REVISEs a ladder plan whose
disk estimate assumes keeping every rung without this justification.
Mechanical backstop: `scripts/verify_plan.py` c33 (`c33_ladder_retention`)
WARNs an experiment|analysis plan carrying checkpoint-ladder vocabulary
whose compute-sizing sections state no retention vocabulary (escape:
`N/A — no per-rung checkpoint persistence`); surface-only — adequacy of a
stated policy stays with this lens.


**Fan-out end-of-run accumulated footprint — size the boot disk to the SUM
of every cell's RETAINED outputs at end-of-run, or declare a driver-side
between-phase reap.** The merge-disk block bounds coexisting TRANSIENTS
during iteration and the ladder block bounds per-rung retention WITHIN one
training run; neither covers an N-cell fan-out (N training runs / cells on
one provision, sequential or GPU-sharded) whose driver KEEPS each cell's
outputs locally after the cell completes — adapters, per-run checkpoints
(incl. optimizer state), trainer logs. Retained per-cell outputs accumulate
MONOTONICALLY across cells, so the §9 disk row for such a phase MUST size
the boot disk to `Σ over cells of retained_gb_per_cell + the transient
high-water mark` at END-of-run — `retained_gb_per_cell` grounded on what
the trainer ACTUALLY writes (weights + optimizer state; the ladder block's
`per_rung_gb` precedent: #1112 planned ~15 GB/rung, realized up to ~28 GB)
— and DECLARE that summed footprint in the launch flags (`--boot-disk-gb`,
the ladder keep-all exception's duty, arming the #1118 volume-threading /
typed refusal on a lane failover). A §9 estimate assuming single-cell /
steady-state retention on a multi-cell fan-out, with NEITHER the summed
sizing NOR a declared reap, is a REVISE. ALTERNATIVE: bound the
accumulation instead — the plan declares a between-phase reap of each
cell's CONSUMED outputs wired into the DRIVER (delete a cell's local
outputs once uploaded to the Hub, or once covered by a plan §10
`discarded_artifacts:` entry; `store/` + `eval_results/` never touched —
the Upload Policy upload-before-delete invariant unchanged). Tooling
honesty: `scripts/clean_experiment_downloads.py --incremental` reaps
`hf_dl`/`g*_dl` DOWNLOAD caches only, NEVER per-cell training OUTPUTS —
the reap is driver code the plan names (canonical shape: a per-cell
post-upload delete in the driver loop), not an existing shared helper.
(#1481: three GCE lanes died the same day on retained per-cell outputs —
one after completing all 24 cells with 48/48 adapters already on HF; local
deletion was legal throughout, the gap was §9 sizing + driver cleanup,
never upload policy.) Critic enforcement:
Methodology lens item 16 FAN-OUT ACCUMULATION EXTENSION
(`.claude/rules/critic-lens-reference.md`); no verify_plan.py backstop in
v1 of this block (a c33 trigger-regex change mandates a full
persisted-plan corpus re-scan per its calibration contract — the
mount-binding block's v1 posture is the precedent).


**Phase-ordering checkpoint high-water — compute the stated high-water
against the IMPLEMENTED phase ordering; every phase that accumulates
checkpoints without an intervening reap is itself reap-bounded.** The
merge-disk / ladder-retention / fan-out blocks above size WHAT
accumulates; none binds WHEN the reap runs relative to the phases the
dispatcher actually implements, and a stated high-water is valid ONLY
under the phase interleaving it assumes. Two duties: (1) the §9
high-water row for any per-rung / per-cell checkpoint design STATES the
phase ordering it assumes (per-cell train→select→reap; W-wide bounded
waves of train→ladder→persist→reap; a between-rung online reap), and
the plan's own phase sequence / pipeline DAG (§4/§9) implements THAT
ordering — a mismatch is a REVISE: plan-time when the ordering is
unstated or contradicts the plan's own phase list, impl-time
(plan-adherence / code-review) when the dispatcher's realized phasing
diverges from the stated one. (2) ENUMERATE every phase that
accumulates checkpoints without an intervening reap and bound EACH — a
reap living only inside a DOWNSTREAM consumer phase/unit (the
ladder/selection read's stream-reap, a per-unit delete) CANNOT bound an
upstream train-all accumulation: if a train phase completes all N cells
before any consumer phase runs, that phase's high-water is N ×
per-cell retained rungs regardless of any downstream reap. Canonical
bounded shape: bounded-wave pipelining (train → ladder/select →
persist → reap, W cells per wave ⇒ high-water ≈ W × per-cell footprint
+ in-flight transients) — the #1586 r5 fix. Phase-START headroom
canaries do not substitute (the mount-binding preamble assert fires
once per phase and cannot see per-wave demand). (#1586 r5: a §9 model
implicitly wave-pipelined at ≈456 GB, but the dispatcher's linear
train-all-cells→ladder phasing projected ~2.5 TB on a 750 GB volume;
the stream-reap lived only inside the downstream ladder unit, and the
run died ENOSPC at ~2.5 cells.) Critic enforcement:
Methodology lens item 16 PHASE-ORDERING EXTENSION
(`.claude/rules/critic-lens-reference.md`); no verify_plan.py backstop
in v1 of this block (the mismatch is stated-ordering-vs-implemented
semantics no text heuristic reads — c33's disclosed miss (a) is exactly
this class — and a c33 trigger-regex change mandates the full
persisted-plan corpus re-scan per its calibration contract; the
fan-out + mount-binding blocks' v1 posture is the precedent).


**Out-root mount binding — every §9 disk estimate for an out-root NAMES the
target filesystem/mount, and the workload preamble asserts headroom against
the mount the out-root ACTUALLY resolves to.** A GB estimate alone does not
bind the estimate to a filesystem: the out-root can silently land on a
different (smaller) mount than the one the estimate was sized against, and a
correct GB number on the wrong mount still ENOSPCs mid-write (#1333: the
dispatcher's out-root resolved outside `/workspace` — on RunPod everything
outside `/workspace` is the ~50 GB CONTAINER disk, not the MooseFS volume —
and the run died ENOSPC mid-checkpoint despite a correct §9 GB estimate;
fixed by anchoring the out-root under `/workspace` + per-phase headroom
asserts). Two duties:

1. **Plan-side (§9):** every disk row for an out-root (checkpoints, stores /
   analysis tensors, staged inputs, scratch) states the PATH the phase
   writes AND the filesystem/mount that path resolves to on the routed
   lane — RunPod: the `/workspace` MooseFS volume (per-pod ~130 GB EDQUOT
   quota; `/tmp/` + everything outside `/workspace` is the container disk,
   typically ~50 GB — `.claude/rules/gotchas.md`); GCE: the boot disk
   (sized by `--boot-disk-gb`; `$WORKLOAD_ROOT` = `/workspace/eps-issue-<N>`
   lives on it); shared VM: `/` (fleet-shared boot disk, 40 GB preflight
   floor) vs the `/mnt/eps-data` bind (per-issue ext4 quota — CLAUDE.md
   § Disk hygiene); SLURM: `$SCRATCH`. When the auto router can land the
   run on more than one lane, state the mount per candidate lane — a lane
   failover changes the mount (the #1112 keep-all-ladder ENOSPC in the
   retention block above is the failover-shaped sibling) — or pin the lane.
2. **Preamble-side (workload contract):** before each write-heavy phase the
   dispatcher asserts headroom against the mount the out-root RESOLVES to:
   `os.statvfs(out_root)` free vs the phase's §9 floor PLUS a ~1 GB
   `posix_fallocate` canary (statvfs is blind to an already-exhausted
   MooseFS per-pod EDQUOT quota), raising with the numbers (path, resolved
   mount, free GB, floor GB) BEFORE the phase writes — a mid-save ENOSPC
   corrupts the checkpoint and forfeits the trained step. Canonical shared
   helper:
   `explore_persona_space.orchestrate.preflight.assert_out_root_headroom(
   out_root, need_gb, phase=...)` (reuses preflight's
   `_probe_writable_bytes` canary; returns free GB) — import it, never mint
   a fresh per-issue copy; the per-phase floors are the §9 disk rows (the
   `PHASE_HEADROOM_GB` shape — originating precedent
   `scripts/issue1333_dispatch.py:302-346`). The gate MUST be
   resume-aware: compute the phase's PENDING set with the same predicates
   the phase's own resume scan uses — zero pending ⇒ skip the gate with
   one INFO line; partial ⇒ scale need to the pending subset (per-cell
   need × n_pending, or the sum of per-cell demands; fixed
   margins/constants untouched); a fresh run computes byte-identical need
   (pin that equivalence in the dispatcher's tests). A blanket fresh-run floor
   at phase entry deadlocks a resume whose own done artifacts legitimately
   occupy the disk — the gate demands headroom for work that will not run,
   deterministically on every respawn, and blocks the very reclaim/wipe
   phase that would free the space (incident #1586 fu crash 5; fix
   pattern: the wave-level pending-aware gate,
   `scripts/issue1586_dispatch.py::_wave_headroom`). The process-START preflight
   (`orchestrate.preflight` `check_disk_space`) probes ONE launch-time
   check path; it does NOT cover an out-root on a different filesystem —
   that gap is exactly #1333.

Siblings: the merge-disk budget + ladder-retention + fan-out accumulation
blocks above size WHAT accumulates; the phase-ordering block binds WHEN
each accumulating phase gets reaped against the implemented phase
sequence; this block binds WHERE it lands and adds the per-phase runtime
assert. The ≥5 GB inline-staging clause (CLAUDE.md compute-character
pre-launch statement: staging path named up front + the filesystem it
resolves to via `df -P` + ≥1.5× headroom) is the inline-analysis sibling.
Critic enforcement: Methodology lens item 16 MOUNT-BINDING EXTENSION
(`.claude/rules/critic-lens-reference.md`) REVISEs a bare-GB / unbound-mount
disk row; no verify_plan.py backstop in v1 of this block.


**Fan-out over the same HF prefix — pre-stage once and fan from the staged
snapshot, or serialize/jitter concurrent same-prefix pulls.** A plan
fanning N > 1 boxes / legs / GCE instances over the SAME multi-GB HF
prefix (a shared model repo, a shared dataset prefix on the data repo)
names its staging shape in §9. DEFAULT: pre-stage the prefix ONCE (from
one box, or from the VM's data disk) and fan from that snapshot — via a
shared read path (a persisted GCE disk / a mounted network volume), an
`rsync` to each box AFTER the stage completes, or a persisted-image
bake. ALTERNATIVE when pre-stage is genuinely infeasible: serialize the
per-box pulls (each box waits for the previous to `snapshot_download` /
`hf_hub_download` complete), OR jitter their start times so the requests
land staggered rather than in one thundering herd. N concurrent
same-prefix multi-GB pulls are a rate-limit kill risk: an HF rate-limit
storm returns 429 (or a TCP/RST that reads as rc=137 to the workload)
and any one box's shard fetch can die mid-stream, forcing a relaunch
that re-books the same collision on the next attempt (#1739: three
boxes each staged ~144 GB from the same prefix simultaneously; 5 total
attempts to land one leg). A §9 plan with `N > 1` same-prefix
concurrent stages and NO named staging shape is a REVISE. Critic
enforcement: Methodology lens item 16
FAN-OUT STAGING EXTENSION (`.claude/rules/critic-lens-reference.md`);
no verify_plan.py backstop in v1.


**Sentinel-signaling workloads need a /workspace-contract lane — never
rely on auto's DRAC/Mila SLURM fallback.** If the plan's dispatch script
posts markers via pod-side sentinel files
(`/workspace/logs/issue-<N>-*.json` — gate sentinels, `epm:results`
payloads), the plan SHOULD pin a DRAINED lane: `backend: fellows` (the
charmander cluster-shared `/workspace`, drained by the VM-side poller
each tick via `slurm_monitor.drain_cluster_sentinels` — #1898) or an
explicit `backend: runpod` override with its residual gap named
(`backend: gcp` is REFUSED as of #2028 — GCP provisioning disabled;
it is no longer a pinnable drained lane). Leaving such a
workload on `auto` is discouraged: a fellows capacity failure
falls through to the DRAC/Mila SLURM lanes, where compute nodes have no
`/workspace` and the robot wrapper cannot run the sentinel drain — the
dispatcher fails loud at its `mkdir -p /workspace/logs` and burns the
SLURM submission (#608, commit 3022ff7bc). If the plan needs a DRAC/Mila
lane, the dispatcher must use the SLURM signaling contract instead —
`status.json` heartbeat + `[phase=...]` log lines (see
`backends/slurm_monitor.py` module docstring § "Sentinel drain: fellows
only"). State the choice in §9:
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
fabricated (#823: a ~3780-call full-SVD ridge phase planned at 0.35 h via
an asserted ~2 s/fit sailed under the old 12 h trigger while the measured
~125 s/call implied a ~131 h serial floor; #522; #511).


**External-stream phases (network-bound row iteration) — the floor is presumed, not
sized.** A §9 row whose component consumes an external streaming source (HF `datasets`
`streaming=True`, API pagination, web harvest, S3/HTTP row iteration) has NO reliable
count × per-call sizing basis: the per-row kernel is trivial (~ms parse+filter) and
wall-time is network-throughput-bound, so both the wall-clock trigger and the
~500-call presumption above miss it (#1092: an unbounded full-corpus stream
scanned ~1.8M rows over 3h06m; its bounded twin used a keep-quota stop —
both shapes defeat count × per-call sizing). When the scanned-row
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
the same kernel family as the floor cross-check's call-count trigger) —
**or runs a permutation / bootstrap / null-draw battery above the
~15–30 min phase floor** (the per-"call" unit is then one
production-shape batched draw block, not one serial draw) — over
cells × folds × layers × arms × traits × seeds MUST ground its per-call
`basis` on a MEASURED 1-cell/1-unit pilot timing at PRODUCTION shape —
SHAPE includes the phase's realized EXECUTION GEOMETRY (batch width /
per-call structure): a phase that runs B-wide batched calls is piloted
with one B-wide batched call normalized per-sample, never a serial
batch-1 loop, whose per-sample cost reads ~B× the sweep's under
bandwidth-bound decode and can false-fire a correctly-derived in-run
timing gate by ~B× (#1415) — on
the machine/device the phase will actually run, executed THROUGH the
production entrypoint (one full cell/unit end-to-end — every kernel the
per-cell path touches), OR a cited prior-issue MEASURED figure for the
SAME kernel + shape. Projected wall = `n_calls × measured_per_call /
parallelism`, stated in the row. MULTIPLIER DERIVATION — the multiplier
product itself (`n_calls` = draws × cells × folds × evals-per-unit × …)
MUST be DERIVED FROM THE CODE, never assumed arithmetic: count the
inner-loop iterations at the named production entrypoint (read the loop
bounds off the fit/driver script, or log a counted 1-unit run) — a
correctly MEASURED per-call pilot times a WRONG multiplier still
under-projects by the missing factor (#1689: a per-pair evals multiplier
was assumed rather than read off the fit code). DRAW-COUNT
NECESSITY — a bootstrap / permutation / null-draw battery projected to
DOMINATE its phase's wall (≳ half the phase cost) MUST state draw-count
necessity in the row: what the draws buy (the CI-width / significance
target), why N draws rather than an order less, and a pre-registered
DESCOPE lever (reduce N to a stated floor / drop the band) the mid-run
deviation path may pull without a re-plan (#1689: an N=1000 bootstrap
projected at ~140 CPU-h dominated its phase and was descoped only by
user order). POOL-SCALE PILOTS — SHAPE also includes SCALE for kernels
whose cost scales SUPERLINEARLY in pool size (pairwise similarity /
near-dupe screens ~ pool²): the pilot runs at production pool scale, OR
the row states the scaling exponent and extrapolates the pilot by it —
a below-scale pilot on a quadratic battery under-projects by the square
of the pool ratio (#1738). And health reads of a long serial screen /
battery phase key on OUTPUT growth (produced files, log advance,
`/proc/<pid>/io` write_bytes), never CPU% alone — a frozen or
quadratically-grinding serial screen reads ~100% CPU for hours (#1738).
PER-REGIME BINDING — SHAPE also includes
the lane's production REGIME (behavior / budget / corpus): a pilot
wall measured on one lane's regime is a MEASURED basis for THAT lane
only; proxying it to a lane with a different behavior/budget regime
makes it a GUESSED basis there — re-pilot per regime, or fence that
lane at ≥2× the worst-case extrapolation and mark its row
`pilot-gated` (#1739: per-group walls measured on the evil behavior
were proxied to other budget regimes; 4 of 6 lanes halted at their own
pilot gates and needed relaunches with measured fences).
TRIVIALITY EXEMPTION — never
self-certified by an asserted cost: a row may skip the pilot ONLY when
total_calls ≤ ~500 AND its sub-floor (~15–30 min) projection is computed
from a MEASURED or prior-issue-CITED per-call figure; an ASSERTED
per-call cost can never exempt a row, because the projected wall is
exactly the number that is wrong when the basis is fabricated. Three
failure modes this closes: (i) the asserted basis — #823 planned a
~3780-call full-SVD ridge phase at an asserted ~2 s/fit while the named
fast twin's own docstring said ~125 s at that shape; 12–20 h realized.
(ii) the PARTIAL measurement — #811 timed ONE inner kernel and asserted
the surrounding headline path at "~1–2 h"; the unmeasured path was the
dominant frame and the realized-extrapolated wall ran ~700–1000 h vs the
5.0 h projection; the round-2 fix measured the unit END-TO-END on the
production entrypoint, which is the pilot this block requires up front.
(iii) the wrong-device measurement — #931's battery basis was measured
on an A100 GPU fit while the realized cells ran CPU-heavier on the
shared VM (~2.2–2.5× elapsed-vs-plan). A FLOP/kernel floor is the
CROSS-CHECK (block above), never the basis — these loops are
overhead-bound, so a FLOP floor under-sizes them by construction
(`.claude/rules/vectorize-many-cell-fits.md`). When the pilot cannot run
at plan time (its inputs don't exist yet), the plan pre-registers the
pilot as the phase's FIRST step with an abort threshold — pilot ×
n_calls / parallelism re-projected against the row; >2× the row ⇒ the
vectorize signature check fires before the loop proceeds — and marks the
row's basis `pilot-gated`. (The in-run pilot measures at the SAME
execution shape, and its refusal is a DESIGNED artifact-routed halt —
report JSON + a distinct rc the dispatcher routes like its other stop
criteria, never a bare rc=1 read as an anonymous crash; gotchas.md
pilot-gate entry, #1415.) BASIS CURRENCY —
whenever a pilot or realized in-run figure deviates ≥2× (the
`compute_deviation_over_2x` / `EPM_ETA_DEVIATION_MULT` boundary) from a
§9 row's approved basis, the owning session posts a RECORDED basis
update in the same turn — an `epm:compute-deviation` re-post or
`epm:progress` note re-stating the row's basis (measured per-call ×
code-derived multiplier), the re-projected wall, and any downstream
fence/cap re-derived from it; an approved plan's basis is never left
standing known-stale (#1738: a materially stale capture-cost basis
stood in the approved plan post-pilot). This is the fit-phase twin of
the store-heavy block's measured one-item rule below. Sizing precedent
(#1092 offvm battery refit): a pilot-gated permutation-null battery
measured 71.9 GB RSS against a 64 GB planned cap and projected
12.8 h/box against 5 h planned (~2.6×) — the pre-registered abort
threshold + checkpoint-restore recovered it, and ambient-dimension
permutation/null batteries MUST be presumed ≥2× the naive RSS/wall
projection until pilot-measured: a `pilot-gated` battery row BOOKS ≥2×
its naive wall/RSS projection in the §9 headline (and any fence/cap
derived from it) until the pilot lands — booking the naive figure is
the #1092 failure, not a compliant plan. Mechanical twin:
`verify_plan.py` `c48_basis_booked_arithmetic` (WARN-only) flags a §9 row
whose own basis derives a GPU-h figure > 2× the row's booked
`planned_gpu_h` with no reconciliation marker, and a row whose stated
per-cell abort threshold sits BELOW the per-cell wall its own booking
implies (#1336 `EXT_off`: basis 90 GPU-h vs booked 30; abort > 30 min/cell
vs a booked ~91 min/cell).


**GPU-utilization / "GPU-bound" claims in dispatch, checkpoint, and
monitoring notes require a SAMPLED window — never one instantaneous
`nvidia-smi` read.** Any claim that a workload is GPU-bound / "pinned
at N%" / idle states its sampling basis: ≥10 readings over ≥60 s (e.g.
`nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -l 7`
for 10 samples), reported as mean + peak. GPU duty cycles are bursty,
so a point sample can land on a transient peak and invert the
conclusion (#1773: a pre-spend checkpoint claimed "GPU pinned at 90%"
from ONE sample; a 30-reading/60 s re-measure showed mean 12.6% — the
91% peak appeared exactly once — and an H100 billed ~7 h at ~87% idle
behind the wrong claim). This is the utilization sibling of the
per-cell block's output-growth health-read rule (never CPU% alone,
#1738).


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
bytes-only gate passes exactly the phase that blows up (#813: a bytes-only
GO/NO-GO gate passed while `np.savez_compressed` cost 103.8 s/file vs 1.2 s
for plain `savez`, driving the store phase 4.5× over plan on an idle 8×H100).
Client-side compression (`np.savez_compressed`, gzip) of fp16 activation
tensors bound for a Xet-backed HF repo defaults OFF: Xet chunk-compresses and
dedupes server-side (−59% on the #813 upload), so client zlib bought a 1.29×
ratio for ~86× the write time; a plan that turns compression ON states the
measured ratio + per-item wall-time that justifies it. And
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
(gotchas.md "Memory caps for torch fit loops"). LARGEST-CELL KEYING —
this keying duty applies to ANY RAM/RSS projection used for routing or
machine sizing: the VM-placement gate here AND GPU-lane host-RAM /
machine-RAM sizing (`--min-ram-gb` / machine choice). Key the
projection on the LARGEST planned cell/lane/table — the max over lanes
of the working-set — never an anchor or first-listed unit;
heterogeneous lanes state per-lane peaks or size to the max (#1739:
host RAM sized on an anchor behavior's table; a sibling lane's larger
working set kernel-OOM'd python at anon-rss 163 GiB on a 170 GB-class
machine, twice; the 340 GB relaunch held). Route the phase OFF the
VM — `cpu-bigmem` (RunPod `cpu5m-16-128`, 128 GB; `cpu-mid`'s RunPod row
is only 16 GB now that the 32 GB GCP E2 shape is rollback-only, #2028) —
when projected peak RSS ≥ ~16 GB, OR when concurrent VM-resident phases'
SUMMED projected RSS crosses the same ~16 GB bar (#833: two ~13-15 GB
concurrent phases lost 5 cells to earlyoom — concurrent residency SUMS;
#778: a 22-GiB-RSS null battery was earlyoom-killed 3× before its
cpu-bigmem pivot; #1092: a fit-grid pilot's real per-unit RSS ran
≥22.9 GB against a projected 8–10 GB and was earlyoom-SIGTERMed — a
fit-grid RSS projection is presumed underestimated until pilot-measured
on ONE unit at production shape, and a projection within ~3× of the VM
routing bar routes straight to cpu-bigmem rather than piloting on the
shared VM). A routed
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


**DOWNLOAD routing — a phase that downloads a lot of data runs on a POD,
even when it is CPU-only (standing rule, Thomas 2026-08-06).** Verbatim
directive: *"anything that needs to download a lot of data should run on a
pod even if it's CPU only"*. STRICTER than the >50 GB disk carve-out and
evaluated BEFORE it. Trigger: any phase pulling a large dataset / model /
artifact set to local disk — HF `snapshot_download`, a `hf_dl` / `*_dl`
staging pull, an `hf_hub_download` loop, a bulk tree-fetch, an rsync of
results off a pod — at a threshold of **~10 GB**, deliberately far below the
50 GB gate. Rationale: the 50 GB gate has repeatedly failed to fire on
downloads that still broke the fleet — #1393 was a **14 GB** inline HF pull
that filled `/` with ENOSPC while passing the 50 GB check. Unknown size
counts as OVER: size it first (`list_repo_tree` sums bytes without
fetching) or route to a pod. Binds inline / user-chat rounds and subagent
dispatches identically to plan §9 phases — the constraint is DISK, not
FLOPs, so "it's only CPU work" and "it's just a download" are both
non-reasons. Forced-VM exception (a consumer that genuinely cannot run
pod-side): the dispatch note names the staging path, the resolved
filesystem (`df -P`), and free headroom ≥ ~1.5× projected bytes, staging to
`/mnt/eps-data` — never `/`, `/tmp/`, or a fresh root-owned top-level dir.
Context making this hard as of 2026-08-06: `/` sits at 98% used / 23 GiB
free (237 GiB worktrees + 208 GiB `data/`), because the #681 bind-migration
is still pending (task **#2132**) — there is no slack to absorb even a
modest pull.

**CPU-phase THROUGHPUT routing — the shared VM is ~6× slower per unit
than a dedicated RunPod CPU pod; route CPU-only work to a pod by DEFAULT
(#2054).** The >50 GB disk carve-out and the ≥16 GB RSS gate above are
SAFETY gates — they answer "will this phase survive on the VM?". This is
a SPEED gate and it fires FAR BENEATH both: a 2 GB-RSS, 5 GB-footprint
phase clears every safety bar and still belongs on a pod. MEASURED
(#2054, fits fleet): the same production entrypoint ran ~**6× faster per
unit** on `cpu-bigmem` (`cpu5m-16-128`, 16 uncontended vCPU at `OMP=16`)
than on the shared VM. The mechanism is CONTENTION, not hardware — the
VM's cores are shared across ~15 concurrent Claude sessions plus crons,
so a VM per-unit wall prices the fleet's instantaneous load rather than
the work, and that load is unbounded, non-stationary, and outside the
plan's control. Consequences for §9 rows:

1. **Route past the trivial floor.** A CPU-only phase projected past the
   ~15–30 min trivial floor names a RunPod CPU lane
   (`cpu-small` / `cpu-mid` / `cpu-bigmem`) EVEN WHEN both safety gates
   pass. CPU pods run N-in-parallel (the one-pod rule is GPU-specific),
   so width is cheap; a contended free core is not a bargain.
2. **A VM-measured basis does not transfer to a pod row, or the
   reverse.** The MEASURED 1-cell pilot must run AT the venue the fleet
   will use — piloting on the VM to size a pod fleet (or vice versa)
   imports a contention factor of unknown size. Prefer an entrypoint
   with a BUILT-IN pilot gate that measures at the production venue and
   exits non-zero with a report when the projection breaches its
   ceiling (#2054 ladder: exit 7 + `pilot_gate_report.json` above 12 h,
   with wider sharding as the prepared response).
3. **Cite an existing measured ratio instead of re-deriving it.** Once a
   task has a measured pod-vs-VM per-unit ratio, later dispatches in
   that task reuse it as the venue basis rather than spending another
   pilot (#2054: the ladder chose its venue from the fits fleet's
   measured 6×). A cited ratio is a MEASURED basis and satisfies the
   per-cell-pilot duty; an asserted or remembered one does not.

Interaction with the safety gates: ORTHOGONAL and additive — the
throughput gate can send a phase to a pod that the safety gates would
have allowed on the VM, but it can NEVER keep a phase on the VM that
either safety gate routes off. Unaffected: the GPU-worthiness carve-out
(an iterative-optimization fit still routes to a GPU lane) and
VECTORIZE-FIRST (an overhead-bound loop is batched before any venue
change — buying a 6× faster venue for a 50× overhead-bound loop is the
wrong fix; `.claude/rules/vectorize-many-cell-fits.md`).


**Cost wall-time against the machine the router will ACTUALLY provision —
then reconcile worst-case wall against the GCP auto-delete fence.**
Each row's `planned_wall_h` + `basis` MUST name the machine type of the
lane the backend router will most likely route. Under the standing
fellows-first `auto` default (#2028 — GCP provisioning disabled) that is
the fellows H200 cluster, then the free SLURM lanes, with RunPod's H100
intent table as the terminal rung; the GCP intent mapping
(`INTENT_TO_MACHINE` in `src/explore_persona_space/backends/gcp.py`:
`lora-7b` → 1× A100-80 `a2-ultragpu-1g`, `ft-7b` → 4× A100-80,
`eval`/`debug` → 1× L4) applies only under the rollback flip. A basis
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
finished; the ×2 default absorbs estimate bias AND tail dispersion, not a
calibrated p90/mean ratio — a prior-issue distribution wins whenever one
exists). A deliberately
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


**Reconcile the §9 wall against the SLURM `--time` default bin whenever
the launch omits `--time-budget-hours` (#2027).** On a SLURM-reachable
route (`dispatch_issue._slurm_lane_reachable` — fellows/nibi/fir/mila
explicit pins, or an `auto` order carrying a SLURM lane) a launch with no
`--time-budget-hours` gets sbatch `--time` from the INTENT's default bin
(`slurm._DEFAULT_TIME_BUDGETS_HOURS`: lora-7b 6.0 h, eval 4.0 h, ft-7b
23.5 h, ...), so a §9 projected wall above that bin TIMEOUTs mid-run —
the #1336 shape reached WITHOUT `--max-run-duration`, which is exactly
why the runtime `max_run_duration_slurm_inert_without_time_budget`
refusal and its c46 arm-2 plan-time twin are structurally blind to it.
RULE: when the plan's max `planned_wall_h` exceeds the launch intent's
bin, declare `--time-budget-hours >= <max wall>` on the launch command
(the repo's own margin style is an explicit in-table value, e.g. ft-7b
23.5 under the 24 h bin), or pin a non-SLURM backend. Mechanically
backstopped (WARN-only, heuristic) by `verify_plan.py` c50 — fires only
on exactly-one-DISTINCT-launch plans (multi-dispatch wall-row↔dispatch
joins are a documented false negative); 2026-08 corpus calibration: 5
true-positive WARNs of 5,244 plans (#1345 v7-v9, #597 v4-v5).


**Multi-arm min-width + stall-time down-width split — the down-going
sibling of the #1121 wide-first rung walk.** A plan whose §9 couples two
or more arms with DIFFERENT minimum GPU requirements behind ONE provision
(e.g. a 1×-runnable LoRA-ladder arm and a 4×-needing ZeRO-3 full-FT arm
dispatched together on a 4×/8× pod) MUST name, per arm, that arm's
MINIMUM runnable width — the smallest GPU count × class the arm can
actually execute on (a min-width column or per-arm line in the §9
compute-projection table) — and MUST pre-register the down-width split:
if the coupled provision sits in a SUSTAINED capacity stall — ≥ ~1 h
queued / stocked-out across rungs (prose guidance, not a coded gate;
calibrated as several full ladder walks — the router's per-rung queue
timeout is 600 s, `EPS_GCP_QUEUE_WAIT_SECONDS`, and the free-lane park is
600 s, `router.FREE_WAIT_SECONDS`) — the owning orchestrator SPLITS OUT
the narrowest-runnable arms as their own narrow dispatch(es) and probes
that shape immediately, rather than holding every arm behind the widest
arm's provision; the wide arm keeps its own ladder walk unchanged.
Composition invariant (all three untouched): the #1121 wide-FIRST walk
still leads for any single shardable phase (wide `a2-ultragpu` rungs
tried first when work shards); the #1379 explicit-wide 8→4→2 degrade
still narrows ONE dispatch's machine on a capacity miss — it CANNOT
decompose arms of different minimum widths bundled behind one provision,
which is exactly the #1112 failure mode; and the saturate-or-downsize
idle-width protections are unchanged (a split-out narrow arm must still
saturate its own pod). Incident #1112: a coupled dispatch held
1×-runnable arms ~14 h through a GCP A100 drought while the 1× shape had
stock — the Arm-A-only 1× dispatch provisioned immediately and finished
in ~55 min. A plan that couples mixed-width arms without per-arm minimum
widths and the pre-registered split leaves the mid-run orchestrator no
licensed decomposition — the gap this recipe closes. (Up-front
DECOUPLING of mixed-min-width arms into separate provisions was
considered and rejected: wide coupling wins when capacity exists — one
provision, shared setup — so the split is a stall-time remedy, not the
default dispatch shape.)


**Teammate / mid-session box dispatches — the compute-character duty binds
outside plan §9 too.** Any multi-hour (>~1h projected) box/leg dispatched
by a teammate, orchestrator, or subagent OUTSIDE a plan §9 row —
mid-session scope-extension addenda included — carries the SAME
pre-launch statement as a §9 row: a MEASURED 1-cell pilot wall basis at
PRODUCTION shape (or a cited prior-issue MEASURED figure for the SAME
kernel + shape), a measured / ×2-presumed RSS basis keyed to the
LARGEST cell/lane, and a self-set fence ≥2× the pilot-extrapolated wall
(measured per-cell wall × remaining cells / parallelism — the p90-style
×2 dispersion default). Mechanics are UNCHANGED and NOT DUPLICATED —
they live at § Per-cell fit phases (the measured-pilot recipe + fence
sizing) and § CPU-phase RAM/RSS routing (the LARGEST-CELL keying + the
≥~16 GB VM-routing bar). This section only widens the BINDING SURFACE
set: (a) plan §9 rows are bound by their own §-scoped wording, (b) the
CLAUDE.md § "User-chat inline free analysis" carve-out block binds
user-chat inline runs, and (c) this section binds the residual class —
teammate/orchestrator/subagent mid-session box dispatches, exactly the
#1739 failure channel. PER-BEHAVIOR-BOXES BY DEFAULT — serial-chaining
independent behaviors on one box when each behavior's leg projects
>~2h wall is a REVISE-shape default violation: WALL BUDGET IS MAX, NOT
SUM. Split by default (per-behavior boxes / per-behavior dispatches),
not by opt-in. Sibling of `.claude/rules/experiment-guidelines.md`
guideline 2's shardable-axis duty — guideline 2 governs GPU width
WITHIN a provisioned phase (saturate-or-downsize); this ban governs
box-LEVEL parallelism across INDEPENDENT behaviors (behaviors are the
shardable axis at the box grain, exactly as GPUs are at the phase
grain). Incident #1739: estimated-not-measured walls ran 2–3× over
budget across several mid-session box dispatches; 5/10 new-arm GCE
boxes were rc=137 OOM-killed in the PILOT phase (no measured RSS basis
existed for the pilot's peak); three independent behaviors were
serial-chained on one box and split only after ~3h+. Sibling surface
(interactive): CLAUDE.md § "User-chat inline free analysis"
compute-character block — user-directed inline runs carry the same
duty for the same reason (both surfaces skip the planner+critic
stack, where §9's own binding lives).
