---
name: vectorize-many-cell-fits
description: Many-cell gradient-descent fits (per-fold / per-cell MLP / AdamW LOCO sweeps, per-cell probes) AND many-draw closed-form statistical loops (permutation / bootstrap / null-draw batteries over a large fixed pool) AND many-cell repeated dense linear-algebra fits (a full svd/eigh/lstsq/GCV-ridge solve looped over fold × layer × arm × trait — a per-cell factorization is cheap ONCE, ruinous ×thousands serially) are OVERHEAD-bound, not FLOP-bound — vectorize the fold / output-dim / layer / cell / draw axes into batched tensor ops BEFORE reaching for GPU or a bigger machine. A naive serial loop is 50-100x slower than the same math vectorized. A mid-run compute-deviation exceeding 2× the plan estimate (row or total) on a RUNNING fit/battery/factorization phase forces the vectorize signature check IMMEDIATELY (§ Mid-run trigger).
paths:
  - "scripts/issue*_fit*.py"
  - "scripts/issue*_skill*.py"
  - "scripts/issue*_predictor*.py"
  - "scripts/issue*_loco*.py"
  - "scripts/issue*_null*.py"
  - "scripts/issue*_perm*.py"
  - "scripts/issue*_boot*.py"
  - "src/explore_persona_space/analysis/**"
  - "src/explore_persona_space/experiments/**"
---

# Vectorize many-cell fits and many-draw statistical loops

**When a per-cell / per-fold gradient-descent fit (a small MLP / probe / adapter
trained with SGD/AdamW, looped over LOCO folds × output dims × layers × cells)
is slow, the fix is almost always VECTORIZATION, not GPU.** These sweeps are
**overhead-bound, not FLOP-bound**: the per-fit arithmetic is tiny, so wall-time
is dominated by Python loop overhead, torch op-dispatch on small tensors, and
thread oversubscription — none of which a GPU fixes (a GPU can be SLOWER on
sub-millisecond ops because kernel-launch overhead dominates).

**The same law covers many-DRAW closed-form statistical loops** — permutation /
bootstrap / null-draw batteries over a large FIXED pool. The pool never changes
across draws, so a per-draw loop that re-reduces it (means / sums / covariances)
from scratch pays a full pool pass × n_draws where one precomputed pool
reduction + one batched GEMM over all draws does the identical math. A plan need
only say "run an N-draw permutation battery" for this to apply — the serial loop
is the default implementation unless the plan states the draws are batched.

**And the same law covers many-cell repeated DENSE LINEAR-ALGEBRA fits** — a
full `svd`/`eigh`/`lstsq`/GCV-ridge solve looped over fold × layer × arm ×
trait. A per-cell O(N·H²)–O(H³) factorization is cheap run once and ruinous run
thousands of times serially; the fixes are Gram/dual-space reformulation (solve
in the N-dimensional dual when N ≪ H), ONE shared factorization reused across
cells that differ only in targets/λ, or the named fast twin. A plan need only
schedule "ridge per fold × layer × arm" for this to apply — a serial per-cell
full factorization is the default implementation unless the plan states the
factorization is shared/batched (#823).

**And the same law covers per-item SERIALIZATION / per-file-upload loops** —
when per-item IO (client-side compression, `savez_compressed`, a per-file Hub
commit) rivals or dominates the item's compute, write the cheap format per
item and compress/upload out-of-band or batched, and benchmark one
production-shape item's serialization wall-time at plan/gate time (#813:
`savez_compressed` 103.8 s vs plain `savez` 1.2 s per file at a 1.29× ratio —
65% of row wall-time; implementation-side rule: `.claude/rules/code-style.md`
§ Compute-throughput discipline; plan-side recipe:
`.claude/rules/plan-compute-sizing.md` § Store-heavy / IO-heavy phase sizing).

## The diagnostic signature

- The job runs at high `%CPU` (many cores) but makes little progress, with a
  huge `cputime / walltime` ratio and a large thread count (`NLWP`).
- A back-of-envelope FLOP count is tiny (minutes of real compute) yet the job
  has burned hours of CPU-time.
- No per-cell checkpoint; output only at the very end → opaque, no ETA.
- A per-draw Python loop that re-reduces a large FIXED pool (means / sums /
  covariances) every draw — precompute the pool reduction once and batch all
  draws as one GEMM (the subset-sum identity: a draw's group mean/sum is a
  masked matrix product, `(n_draws, N) @ (N, d)`).

**Worked incident (2026-06-29):** #722's `base-skill-over-mean-cC-to-v0` ran a
per-fold MLP LOCO sweep — 28 layers × 3 MLP variants (base / z-scored-input /
shuffle-null) × 50 LOCO folds × 300 epochs of a width-512, 1-hidden-layer net on
~49 training rows. Total math ≈ **19 TFLOP** (minutes on CPU, seconds on GPU).
Actual: **19.5 CPU-hours / 96+ min walltime, 78 threads, ~12 cores pegged, not
finished.** ~99% overhead. Plan v5 §9 had explicitly judged it "not GPU-worthy …
CPU-feasible ~30-60 min" — correct that GPU was the wrong lever, wrong that the
serial CPU loop was acceptable. The actual fix was vectorization. (#658's
`_fit_mlp_loco` was the same pattern, motivating the compute-character carve-out;
it recurred here.)

**Worked incident (2026-07-01):** #778's stage-two null battery ran
`perm_null_draws` (`src/explore_persona_space/analysis/null_battery.py`) as a
serial per-draw loop — two full pool-mean passes over a 1783×28×3584 float64
pool PER DRAW, ~4.1 s/draw (py-spy). After the round raised n_draws 200→1000,
the projection was **~15h across the full battery's draw loops** (multiple
statistics × settings — not 4.1 s × 1000 ≈ 1.1h for a single loop) vs the plan
§8 estimate of 1h. The plan itself never said "serial" — it just scheduled the
battery, and serial was the default implementation. The fix was a batched
subset-sum GEMM over all draws (pool reduction precomputed once; all draw-group
means as one masked matmul) — a **~70× win**, the rule's 50-100× class, with no
GPU and no bigger machine needed.

**Worked incident (2026-07-02):** #823's phase 4 ran ~3780 serial full-SVD
ridge fits (fold × layer × arm × trait) at ~125 s/fit (N_tr≈4000, H=3584 — the
fast twin's own docstring figure) for ~12–20 h realized, vs a plan that sized
the phase at "~2 s/fit → ~0.35 h" by assertion (two SEPARATE errors: the
per-call basis was ~62× low — 2 s asserted vs ~125 s measured — and the 0.35 h
wall is inconsistent even with its own basis, since 3780 × 2 s ≈ 2.1 h; the
realized 12–20 h is 35–57× the PLANNED WALL, a ratio distinct from the per-call
error). The task body's reuse map named `_ridge_fit_predict_fast` + its
equivalence gate; the plan dropped it, and no review surface compared the
plan's import against the body-named twin. Gram/dual-space ridge makes the
solve one shared reduction + cheap per-λ updates.

## The fix

1. **Train all LOCO folds simultaneously** as one BATCHED parameter tensor.
   Use `torch.func.functional_call` + `vmap`, OR a `(B, in, hid)` / `(B, hid,
   out)` weight tensor with `torch.bmm`, OR grouped / block-diagonal linears.
   The 300-epoch loop becomes ~300 BATCHED steps total, NOT folds × epochs tiny
   steps.
2. **Batch the other independent axes into the same batch dimension** — output
   dims (one MULTI-output net, never one scalar net per dim), layers, and fit
   variants. One batched optimization covers the whole sweep.
3. **For a draw battery: precompute the pool reduction ONCE, then batch every
   draw as one GEMM.** A permutation / bootstrap draw statistic built from
   means / sums / covariances is a group reduction over a fixed pool — express
   ALL draws as a `(n_draws, N)` selection/weight matrix times the `(N, d)` pool
   (subset-sum identity) instead of re-reducing the pool per draw. (Median /
   rank statistics batch via `argsort`/sorting along the draw axis instead of a
   GEMM — batch the draw axis either way.) Chunk the draw axis if the
   `(n_draws, d)` intermediate strains RAM.
4. **`torch.set_num_threads(...)` to a sane value** — tiny ops thrash with the
   default high thread count; fewer, larger (batched) ops actually use the cores.
5. **GPU is secondary and often marginal at small n** — vectorized CPU is usually
   already minutes. Add a `--device cuda` flag, but vectorize FIRST; do not route
   the un-vectorized serial loop to a GPU lane expecting a fix.
6. **Verify the vectorized reimplementation reproduces the serial numbers** on
   2-3 cells within float tolerance before trusting it (vmap'd init/seed/PCA-basis
   handling is easy to get subtly wrong).
7. **Launch it protected + checkpointed — the launch form is part of the fix.**
   Any VM-side fit / battery / aggregation phase with projected wall-time >~15 min
   MUST use the canonical detached launch: `setsid` pid-capture wrapper +
   `sudo -n choom -n -600` session sweep + the `pid= log= choom=ok|failed`
   breadcrumb — recipe owned by `.claude/skills/issue/SKILL.md`
   § "Detached VM-side long compute phases" (short form:
   `.claude/rules/code-style.md` § "Always run with `nohup`"); do not copy the
   snippet here, follow it there. This VM's earlyoom `--prefer` gives every
   python/pytest process +300 badness, so an unprotected fit is the designated
   victim of ANY neighbor's memory spike (#811: a healthy 6.8 GiB re-fit
   SIGTERM'd ~2h in, 0 checkpoints, by a neighbor's spike). AND per-cell
   checkpoints + a resume predicate are REQUIRED for any loop projected >~1h
   (`.claude/rules/code-style.md` § "Checkpoint per phase", intra-phase grain):
   choom only re-orders victim selection — checkpoints bound the loss when a
   kill lands anyway. "No per-cell checkpoint" in the diagnostic signature
   above is not just an ETA smell; it is the thing this item fixes.

   **Relaunch pin parity (#811).** A RELAUNCH of any such phase —
   supervisor-driven retry, crash-fix round, or the kill+relaunch-on-batched
   of the Supersede contract / Mid-run trigger — MUST carry the IDENTICAL env
   pins (the `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
   NUMEXPR_NUM_THREADS=8` thread caps) and choom protection as the reviewed
   ORIGINAL launch command. A relaunch composes a FRESH command, so pins do
   not carry over by inertia (#811: the relaunch supervisor omitted the
   OMP/MKL pins — ~55 min at 0/108 checkpoints). The supervisor/relauncher
   VERIFIES all four pins are present in the composed command string BEFORE
   dispatch — e.g.
   `for pin in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS; do case "$CMD" in *"$pin="*) ;; *) echo "FATAL: relaunch missing $pin" >&2; exit 1;; esac; done`
   — and records `pins=verified` in the relaunch breadcrumb alongside
   `pid= log= choom=`. The helper-side default cap in
   `vectorized_mlp_skill.py` (`_resolve_num_threads`, #1079) is
   defense-in-depth for the torch intra-op pool only; the env pins remain
   REQUIRED (they also cap numpy/BLAS and subprocesses the helper cannot
   reach).

## Canonical helper

`src/explore_persona_space/analysis/vectorized_mlp_skill.py` — the reusable
batched LOCO MLP-skill / downstream-chain implementation built from this
incident (built during #722 at commit `19a5758fab`, landed on `main` via #740).
Import it for any new
per-fold/per-cell MLP sweep instead of writing a fresh serial loop. (Closed-form
ridge / linear LOCO is cheap ONLY while the whole loop's wall — call count ×
per-call cost at production shape — stays under the same ~15-30 min phase
floor: `scripts/issue658_fit_predictors.py`'s `_press_loo_mse_per_lambda` /
`_ridge_dual_weights` qualify because they solve in Gram/dual space with a
shared factorization. A full-SVD ridge re-factorized per cell over fold × layer
× arm × trait is NOT cheap — ~125 s/call at N_tr≈4000, H=3584; ~3780 calls
≈ 131 h serial (#823). Use the Gram-space fast twin or batch/share the
factorization.) For the
draw-loop half the canonical worked reference is the BATCHED
`perm_null_draws` / `randnorm_null_draws` in
`src/explore_persona_space/analysis/null_battery.py` (the #834 vectorization
of the #778 incident code: per-draw directions built as one batched
reduction, then memory-bounded chunked batched projection + correlation —
fix item 3 realized in code). Import or mirror them for any new draw battery
instead of writing a fresh serial loop.

## Supersede contract — land on main + tombstone the serial twin (same round)

A vectorization rewrite is not done when the batched code exists — it is done
when the serial original can no longer be run silently. Three duties, all in
the SAME round as the rewrite:

1. **Land the batched helper on `main` in the same round** — shared `src/`
   infra lands via the normal worktree merge (or a coordinated infra task),
   never as a task artifact: #722's helper merge-FAILED
   (`new-shared-src-infra-cannot-land-via-artifact`) and stranded on the
   unmerged `vectorized-mlp-skill` branch while the session kept running the
   OLD serial script at ~38h ETA. Confirm the REWRITE itself is on `main`,
   not merely that the file path has history there. For a NEW helper file:
   `git log --oneline origin/main -- <helper path>`. For an IN-PLACE rewrite
   (same file / same function name — the #778/#834 `null_battery.py` shape)
   a path-history check FALSE-PASSES on the old serial commit, so verify
   content or ancestry instead:
   `git show origin/main:<path> | grep '<batched-only symbol or token>'`, or
   `git merge-base --is-ancestor <rewrite-sha> origin/main`. A follow-up
   round MUST NOT schedule work that calls the superseded serial path while
   the batched twin is off-`main` — that is exactly how #778's same-issue
   follow-up re-ran the serial 1000-draw null battery. If a LIVE run is
   already executing the serial path when the batched twin lands and its
   remaining serial ETA exceeds kill+relaunch cost, kill and relaunch on the
   batched path (sibling: `.claude/rules/crash-fix-rounds.md`
   § kill-before-relaunch; the relaunch carries the identical env pins +
   choom — item 7 § Relaunch pin parity). The SAME calculus fires mid-run WITHOUT a landed
   twin on any compute-deviation exceeding 2× — § Mid-run trigger below.
   (General lesson:
   `.claude/rules/workflow-fix-on-bug.md` § "Built-but-stranded fixes don't
   help"; this contract is its vectorization-specific mechanism.)
2. **Tombstone the superseded serial entrypoint.** Default mechanism: the
   serial function/script emits a loud `FutureWarning` at call time naming
   the batched replacement (`FutureWarning`, NOT `DeprecationWarning` —
   Python's default filter hides `DeprecationWarning` at IMPORTED-MODULE
   call sites, showing it only when the calling code is `__main__`; the
   script-imports-script reuse case this guards is exactly the hidden one),
   and raises `RuntimeError` when `EPM_FORBID_SERIAL_FITS=1` (the
   opt-in hard gate a follow-up round or an orchestrator can arm to make
   silent serial re-runs impossible). Outright deletion is allowed ONLY when
   no prior task's Repro/footer references the entrypoint (grep
   `tasks/*/*/body.md` first) — old clean-results must stay reproducible. A
   serial body retained ONLY inside an equivalence check / selftest (the
   `issue658_fit_predictors.py` `_fit_mlp_loco_serial_reference` pattern —
   containment is the criterion; the `*_serial_reference` rename is
   recommended, not required) is already compliant — the tombstone duty
   targets silently-importable twins.
   With no tombstone, the next reuser picks the serial path as "the
   available implementation" (#667 picked up #722's serial `fit_cell`).
   Worked example: `scripts/issue722_fit_M.py::fit_cell` — the
   `include_mlp=True` serial-outer-loop MLP arm carries the warn+raise guard
   (tombstoned by #872); the `include_mlp=False` closed-form ridge arm is NOT
   superseded and carries no warning.
3. **Before starting an in-session vectorization rewrite, check for an
   already-open task on the same file** —
   `grep -l '<module or script name>' tasks/*/*/body.md` from repo root,
   then discard only hits under `completed`/`archived`: ALL non-terminal
   statuses count, including `followups_running`, `interpreting`,
   `reviewing`, `awaiting_promotion`, `blocked`, and `on_hold` — the
   #834-vs-#778-fix duplicate ran while the parent sat at
   `followups_running`, which any narrower status list misses. Add
   `task_workflow.is_open_workflow_fix_task(<target_file>, <fp>)` when a
   candidate fingerprint exists. On a hit, ADOPT/coordinate instead of
   duplicating: #834 and the #778 fix session independently vectorized the
   SAME `null_battery.py` module in parallel, discovering the overlap only
   after both had built.

The contract binds ANY rewrite that supersedes a serial entrypoint — a shared
helper, a `scripts/issue*_*.py` driver, or an in-module loop replaced by a
batched twin — whether done by an implementer mid-experiment, a workflow-fix
session, or an emergency in-session fix.

### Mid-run trigger — a 2× compute-deviation forces the vectorize check NOW

The moment a compute-deviation exceeding 2× the plan §9 estimate (row or
total — the `compute_deviation_over_2x` / `EPM_ETA_DEVIATION_MULT`
boundary respectively) fires on — or
is computed for — a RUNNING fit / battery / factorization phase, run the
vectorize signature check IMMEDIATELY. The trigger is any of: the poller's
`epm:compute-deviation` marker (`source: poller`,
`basis: elapsed-vs-plan`, threshold `EPM_ETA_DEVIATION_MULT` default 2.0),
an orchestrator- or implementer-posted deviation marker, or the session's
own elapsed-vs-plan arithmetic — one threshold, shared with
`workflow.yaml § pivot_criteria` (`compute_deviation_over_2x`), never a
new number. Do NOT wait for a round boundary, for the phase to finish, or
for a SECOND deviation (#811: the session pivoted only on the second
deviation, 19h21m in at unit 3/108 — py-spy showed the dominant frame was
a path the first deviation's fix round had ASSERTED at "~1–2 h", not
measured). The check is cheap — minutes, not hours: this rule's
§ diagnostic signature run against the LIVE process (py-spy the dominant
frame, FLOP back-of-envelope, cputime/walltime ratio) plus the
classification of `.claude/skills/issue/SKILL.md` Step 5.bis(a) step 2 —
the routing table (descope tiers, gate id=12, once-per-component guard)
stays orchestrator-owned THERE; this section binds the
session/implementer/experimenter side and does not duplicate it. Outcomes:

- **Overhead-bound** → duty 1's live-run calculus applies immediately:
  when remaining serial ETA exceeds kill+relaunch cost (count lost
  un-checkpointed in-RAM progress as part of that cost), kill the serial
  run (`.claude/rules/crash-fix-rounds.md` § Kill-before-relaunch) and
  relaunch on the batched path once the vectorize fix round lands (the
  relaunch carries the identical env pins + choom — item 7 § Relaunch pin
  parity).
- **Not overhead-bound** (FLOP-bound / API-latency / bandwidth /
  contention) → record `signature_check: negative` with 1–3 lines of
  arithmetic on the marker and continue — after the width re-evaluation
  below when the phase is an embarrassingly-parallel unit grid: #931's
  6.0× battery resolved
  exactly this way (195 s/cell MEASURED at production shape; shared-VM
  thread contention on a cached-eigh battery — ONE box, no relaunch in
  progress, so the width predicate below did not hold), and its earlier
  ~2.2–2.5× elapsed-vs-plan read was likewise correctly ridden out as a
  demonstrably-in-tail FLOP-bound phase.

**Width re-evaluation on a negative signature — a negative signature settles
VECTORIZATION, not WIDTH (#1092).** When the negative-signature phase is an
embarrassingly-parallel unit grid (independent cells/units, no cross-unit
dependency) AND EITHER checkpoint/restore machinery is live (per-unit
checkpoints persist and a restart resumes them) OR a relaunch is already
happening (a crash-fix round, a pilot-gate abort, any kill+relaunch —
restore is then already occurring), `continue_as_is` at
the current fleet width is NOT the default resolution: EVALUATE re-sharding
the REMAINING units across a wider fleet first. Record 1–3 lines of
arithmetic as a `width_reeval:` note field on the SAME
`epm:compute-deviation` re-post that carries the resolution — remaining
units × measured h/unit (the deviation's OWN measured basis, never the
falsified plan estimate) ÷ candidate width + provision/stage/restore
overhead, vs the remaining wall at the current width. Decision default:
wall-clock is the scarce resource and credits are not (CLAUDE.md "Default to
the most parallel viable spec" / the wide-by-default `--gpus N` guidance) —
WIDENING WINS unless the recorded arithmetic shows otherwise (remaining wall
already short vs re-shard overhead + one provision round, or the lane cannot
supply the width). Staying posts `action: continue_as_is` + the
`width_reeval:` line; widening posts `action: reshard_width_<K>` + the line,
and the relaunch follows the ordinary relaunch duties
(`.claude/rules/crash-fix-rounds.md` § Kill-before-relaunch + § Crash-fix
relaunch; fresh `epm:run-launched` per box). This COMPOSES with plan-time
width — CLAUDE.md wide-by-default governs the ORIGINAL provision; this step
fires at the mid-run deviation/relaunch point, where restore makes
re-sharding cheap. Founding incident #1092 (2026-07-15): a 2.57×
measured-pilot deviation, correctly negative-signatured (FLOP-bound
permutation battery, engine already batched), relaunched a 64-unit
embarrassingly-parallel refit grid at the unchanged width 4 — re-sharding
the remaining units across 8–12 boxes at the restore point would have cut
hours of wall-clock (the parent v6 grid itself had run 12 boxes); nothing
in this branch prompted the width question.

Two scoping notes. A prior lever-0 record does NOT immunize the phase:
when realized numbers FALSIFY the earlier residual classification (a
DIFFERENT inner loop is the dominant frame than the one the fix round
batched), THIS rule licenses a second session-side fix round targeting
that loop — the #811 precedent; nothing in 5.bis(a)'s once-per-component
guard forbids a voluntary second round on a falsified classification. And
the SKILL.md ETA-advisory's "`continue_as_is` is nearly always the right
mid-run resolution" scopes to the DESCOPE question (elapsed is a lower
bound, so descoping mid-run rarely helps); it never defers this check —
kill+relaunch-on-batched is recipe-preserving, not a descope.

## Memory sizing: calibrate the chunk cap from a MEASURED peak

Vectorizing trades many tiny fits for a few LARGE batched tensors, so the
batched path needs a memory-aware chunk cap — and the cap's live-tensor
factor must come from a MEASURED real-shape peak, never from counting the
code's explicit temporaries. The named intermediates undercount the true
per-chunk peak ~6×: the autograd backward graph, AdamW moment buffers, and
allocator high-water retention dominate (#811 r8: a factor-4 explicit-
temporary count picked c=218, whose real ~36 GiB peak re-OOM'd the exact
shape the cap protected — n=480, d_in=3584). Canonical implementation:
`resolve_chunk_cap()` in the helper above (`live_factor=26`, measured:
~10.7 GiB ru_maxrss delta on one c=64 chunk ≈ 26× the single
`(c, n, d_in)` fp32 tensor; built on the `issue-811` branch, on `main` once
#811's worktree auto-merges). Recipe: run ONE chunk at the production shape
in a fresh process, read the ru_maxrss / `torch.cuda.mem_get_info` delta,
set the factor from that; the factor is shape/optimizer/precision-specific —
re-measure when any change. Modest over-estimation is cheap (a larger factor
only adds chunk count at constant FLOPs — chunk size must not change
results, pinned by a chunk-size-invariance test); and LOG the resolved cap +
the probed free bytes at the cap site so the next OOM is diagnosable from
the log alone. Full trap-and-fix entry: `.claude/rules/gotchas.md`
§ "Memory caps for torch fit loops".

## Relation to the compute-character carve-out

`CLAUDE.md` § "compute-character carve-out" (+ `planner.md` §9, `critic.md`
Methodology lens item 10(iii)) says an iterative gradient-descent fit is
GPU-worthy regardless of footprint. This rule REFINES it: for a MANY-CELL loop of
individually-tiny fits, **vectorize first** — the overhead, not the FLOPs nor the
device, is the cost. Route to GPU only after vectorizing, and only if the
vectorized FLOP count actually warrants it. A plan that places a many-cell GD
sweep on the VM as a serial per-fit loop (no vectorization plan) should be
REVISED to vectorize, not merely re-routed to GPU. The same REVISE direction
applies to an unbatched draw battery: vectorize it — neither a GPU lane nor a
bigger CPU pod fixes redundant per-draw pool re-reduction. The same applies to
a many-cell repeated dense-factorization loop: per-cell full
`svd`/`eigh`/`lstsq` re-factorization over fold × layer × arm × trait is the
same overhead class (redundant recompute of shareable work), and neither a GPU
lane nor a bigger CPU fixes it (#823).

**GPU caveat — batched `eigh`/small-matrix factorizations are cuSOLVER's weak
spot.** Batching many SMALL symmetric eigendecompositions (or similar small
dense factorizations) onto CUDA does not reliably beat CPU: cuSOLVER's batched
paths on many small matrices show high, unpredictable per-cell variance
(#813, 2026-07-03: the CUDA leg ran ~30–100 min/cell and was swapped for the
CPU-verified path). For many small symmetric eigs, prefer the vectorized CPU
path or a shared-factorization restructure; benchmark one cell on BOTH devices
before committing a long sweep to the GPU leg.

## Files of record

`.claude/rules/vectorize-many-cell-fits.md` (this file);
`src/explore_persona_space/analysis/vectorized_mlp_skill.py` (helper — built
during #722 at commit `19a5758fab`, landed on `main` via #740);
incidents #722 (base-skill-over-mean, 19.5 CPU-h), #658 (`_fit_mlp_loco`),
#778 (`perm_null_draws` serial null battery, ~15h projected across its draw
loops → ~70× batched subset-sum GEMM), #811 r8 (`resolve_chunk_cap`
live_factor 4→26 — measured-peak chunk-cap calibration), #811 relaunch pin
omission (relaunch-pin-parity clause, #1079), #823 (~3780 serial
full-SVD ridge fits, 12-20 h realized vs 0.35 h planned — the
dense-linear-algebra widening), #834 (parallel duplicate vectorization of
null_battery.py — supersede contract), #811 second deviation (19h21m at unit
3/108 before the pivot — the mid-run-trigger origin) + #931
(elapsed-vs-plan mid-run reads, worked negative example) via #1060.

**Sibling rule:** `.claude/rules/selection-symmetric-nulls.md` — the same #778
null battery is its origin incident; a permutation/bootstrap-battery plan
typically fires BOTH rules (statistical validity there, compute shape here).

**Sibling check:** `.claude/rules/artifact-reuse.md` item (i) — reusing a parent's
fit/analysis helper requires the plan-time throughput inspection (inner loop batched?
device parametrized? data-repo Hub calls scoped?) against this rule, with failures
fixed at the source module
(#761/#763/#812).
