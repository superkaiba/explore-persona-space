---
name: vectorize-many-cell-fits
description: Many-cell gradient-descent fits (per-fold / per-cell MLP / AdamW LOCO sweeps, per-cell probes) AND many-draw closed-form statistical loops (permutation / bootstrap / null-draw batteries over a large fixed pool) AND many-cell repeated dense linear-algebra fits (a full svd/eigh/lstsq/GCV-ridge solve looped over fold × layer × arm × trait — a per-cell factorization is cheap ONCE, ruinous ×thousands serially) are OVERHEAD-bound, not FLOP-bound — vectorize the fold / output-dim / layer / cell / draw axes into batched tensor ops BEFORE reaching for GPU or a bigger machine. A naive serial loop is 50-100x slower than the same math vectorized.
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
   § kill-before-relaunch). (General lesson:
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

## Files of record

`.claude/rules/vectorize-many-cell-fits.md` (this file);
`src/explore_persona_space/analysis/vectorized_mlp_skill.py` (helper — built
during #722 at commit `19a5758fab`, landed on `main` via #740);
incidents #722 (base-skill-over-mean, 19.5 CPU-h), #658 (`_fit_mlp_loco`),
#778 (`perm_null_draws` serial null battery, ~15h projected across its draw
loops → ~70× batched subset-sum GEMM), #811 r8 (`resolve_chunk_cap`
live_factor 4→26 — measured-peak chunk-cap calibration), #823 (~3780 serial
full-SVD ridge fits, 12-20 h realized vs 0.35 h planned — the
dense-linear-algebra widening), #834 (parallel duplicate vectorization of
null_battery.py — supersede contract).

**Sibling rule:** `.claude/rules/selection-symmetric-nulls.md` — the same #778
null battery is its origin incident; a permutation/bootstrap-battery plan
typically fires BOTH rules (statistical validity there, compute shape here).

**Sibling check:** `.claude/rules/artifact-reuse.md` item (i) — reusing a parent's
fit/analysis helper requires the plan-time throughput inspection (inner loop batched?
device parametrized? data-repo Hub calls scoped?) against this rule, with failures
fixed at the source module
(#761/#763/#812).
