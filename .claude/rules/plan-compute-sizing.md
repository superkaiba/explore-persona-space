---
description: Planner §9 compute-sizing recipes — activation-capture HBM sizing, merge-disk budget, sentinel-signaling lane pins, the floor cross-check for long / many-call phases (planned_wall_h > 4 OR >~500 serial calls), the store-heavy / IO-heavy phase recipe (measured per-item serialization+upload wall-time; compression-default-OFF for fp16→Xet), and costing wall-time against the machine the router will ACTUALLY provision (loads at plan time via plan-file paths; relocated verbatim from planner.md §9, #829)
paths:
  - ".claude/plans/**"
  - "tasks/**/plans/**"
---

# Plan compute sizing (planner §9 relocated recipes)

These six recipes are the planner-specific §9 sizing blocks — five relocated
verbatim from `.claude/agents/planner.md` (#829), plus the
store-heavy / IO-heavy phase recipe (#910, from incident #813). The planner
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
forward needs on the pod.

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
into ~34h). Then reconcile the WORST-CASE wall — base phases PLUS every
conditional / extension phase that could run on the same provision —
against the GCP lane's auto-delete fence
(`--instance-termination-action=DELETE` + `--max-run-duration`,
default 7d — the FLEX_START ceiling, #741). If worst-case wall on the routed
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
