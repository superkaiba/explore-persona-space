# EFFICIENCY REVIEW: #2329 follow-up `q35_ladder_decay` — implementation round 1

**Plan version graded against:** **v8** at the MAIN checkout (`/home/thomasjiralerspong/explore-persona-space/tasks/followups_running/2329/plans/v8.md`, the `plan.md` symlink target there, mtime 2026-08-19 16:27). The worktree's stale `plan.md → v4` copy was detected at review start and NOT used — §9 compute rows, the re-derived Leg B production count (≈4.7k = 2,360 + 2,360 per model, S2 CE-primary registration), the ≤~720-call conditional pe stratum, and the registered Leg A ALL-SYNC routing were all read from v8. No rework needed per the orchestrator's correction.

**Verdict:** PASS
**Blocker tags:** none
**Diff size:** +7,242 / −18 across 8 files (round range `850a66846c24..c46f29bf0c33`, 8 commits)
**Diff acquisition:** whole-round body 319 KB > 300 KB budget — reviewed via `--stat`/`--name-status` + per-file targeted reads (dispatcher whole; drivers by grep + span reads); plan v8 §9 read in full.

## Fit-loop batching (positive duty, #1332/#825)

- **Bootstrap battery (B=10,000 × ~160 cells ≈ 1.6M draw-cell evals):** `issue2094_analysis::bootstrap_family_means_batched` — ONE gather-GEMM call per family (`scripts/issue2329_ladder_analysis.py:827,921`; `scripts/issue2329_decay.py:870` — one shared-index call per model spanning both estimands, per its own line 991 note). Evidence: imports at `ladder_analysis.py:83` / `decay.py:77`; `--import-check` asserts callability (`ladder_analysis.py:1487`, `decay.py:1300`).
- **Permutation battery (10,000 × 4 families):** `issue2329_ladder_analysis.py::_spearman_rows` (line 663) — rank-space centered GEMM `(ry_c @ rx_c)/denom` over the full (B, n) draw matrix; null draws generated vectorized per carrier via `np.argsort(rng.random((n_perm, k)), axis=1)` (line 725). Residual Python loops iterate carriers (≤6) and rungs (≤7) only — trivial.
- No GCV/ridge/SVD/fit loop anywhere in the round (plan §9 states none; grep confirms).

## Findings

### Major (revise before merge)
None.

### Minor / notes (non-blocking)

- `scripts/issue2329_ladder_dispatch.sh:283-294` (stage1→stage2 seam): the 1× H100 pod idles ~1.2 h through the L3 VM judge-gate window. NOT a defect — plan v8 §9 row L3 books this idle honestly ("idle 1× GPU booked honestly", 1.2 GPU-h) and the reconciler upheld 1× width; releasing the pod mid-chain would forfeit the pod-resident ~19 GB model cache + 1.47 GB vc_bank and cost a reprovision + re-stage of comparable magnitude. Cost of current shape ≈ 1.2 idle GPU-h (~$3); alternative ≈ 0.5-1 h re-stage wall + reprovision risk. Recorded for the round telemetry, no action.
- `scripts/issue2329_decay.py:712-741` (segmentation/reduce loops over ≤~3.5k texts): serial CPU tokenizer pass with no per-unit persistence. Unit count trips T2 (>50) nominally, but the whole phase is ~minutes of CPU (plan: 0.5 h VM CPU for segment+reduce+stats+figs) and the expensive upstream (Batch judge wave) is checkpointed by the shared resumable wave machinery — forfeit-on-crash is minutes of tokenizer work. De minimis; not a Step 3.6 violation worth a bounce.
- Leg A judge routing is registered ALL-SYNC (`FORCE_SYNC_THRESHOLD_BASE` threaded at every `run_wave` call, `issue2329_ladder_judge.py:122,302,443,459,558,807-835`), per-wave ≤ ~2,160 calls — inside the guidelines' 1k–10k sync-preferred band and matching plan §9(ii)'s registered routing (the #2162-lineage pattern: routing REGISTERED, not a falsified "ALL Batch" claim). Sync is ~2× batch pricing on those waves — telemetry only, no GPU idles behind it (pod terminated before L5).

## What was verified (per the brief's lens)

1. **Inner loops batched:** generation is `generate_batch(...)` on `gen_batch=16` chunks (`issue2329_ladder.py:1377-1379` anchors, `:1675-1690` grid); capture batched at `capture_batch=8` (`:1116-1117`). Hooked HF `generate` (not vLLM) is correct here — activation patching/capture is incompatible with vLLM, inherited from the parent rig with a MEASURED 5.12 GPU-s/rollout basis (plan §9). Draw batteries batched as above.
2. **GPU sharding / launch commands:** worker count derived from realized GPU enumeration (`nvidia-smi -L`, `dispatch.sh:89`) on a DEDICATED RunPod pod — the SLURM allocation-first rule is inapplicable and the exemption is recorded in-file (`SLURM_GPU_WIDTH_EXEMPT`, lines 86-88; plan pins `backend: runpod`, pod-2329-l, 1× H100). Per-worker `CUDA_VISIBLE_DEVICES` pinned in the LAUNCHER env (`:180-182`), never relying on the in-process `+gpu_id` clobber; `--gpu-id` threaded to match. Fan-out phases (anchors/grid/margin) consume a shared work-conserving claim-file queue (`run_claim_queue`, parent skeleton) — no wave barrier, no serial-across-cells on a wider provision. Plan declares 1× with an explicit net-GPU-h argument against 2× (a wider pod would idle through the L3 window + upload); no `compute-shape-mismatch`.
3. **API routing + sizing:** every call site routes through `issue2094_judge.run_wave` → the shared `judge_dispatch`/`api_dispatch` machinery — no hand-rolled client, no `while True` poller in any round script (batch polling is the #663-hardened deadline-bounded client). Leg B pins Batch (`threshold_base=0`) on pilot AND production (`decay.py:549,553,614`) — rule-26/#2152 transport parity; Leg B pilot is wave-declared (`wave_threshold_base=0`). Leg A pilot runs per rubric at the same forced-sync transport as its production waves (`ladder_judge.py:293-302`). Pilot arm sizing (8 arms × ~56 draws) clears the 51-draw resolution floor at the 2% threshold.
4. **Across-cell parallelism:** no post-batching battery projects near ~1h (bootstrap measured minutes at the parent's LARGER grid, #2162/#2094 basis); the ≤1,320-draw grid is the plan-justified 1×-width phase (1.9 h < the 2 h shardable-axis bar of guideline 2) with the claim queue as the shard mechanism should width change.
5. **Phase placement + width:** judging (L3/L5), analysis (L6), decay (L7), figures run VM-side with the pod terminated after L4 upload-verify; the terminal pod upload is a short CPU phase (~2 GB tensors + ~5 MB text via Xet, well under the 15-30 min bar) as 8 bulk `upload_folder` commits — one per prefix via `RUN._upload_dir` → `_upload_folder_filtered` + exact-set verify (`issue2329_ladder.py:2185-2236`), no per-file loop. VM phases are API-latency-bound (plan §9 states the ~6× pod-CPU ratio buys nothing against an API wall — deliberate, correct).
6. **Checkpoint cadence:** per-unit flushed progress lines in the canonical shape (`[anchors] unit k/N ... elapsed=`, `:1417`; `[grid] unit ... elapsed=`, `:1880`; `[margin] ...`, `:2049`); per-block done-records + regime-fingerprinted resume (claim queue namespaces `blocks`/`margin_blocks`); incremental grid upload every `--upload-every` blocks (`:1887-1893`) so no regeneration-costly store waits behind a downstream phase (#825 ordering honored). G2 throughput pilot times the FIRST anchors chunk at production shape and halts with a designed distinct rc=28 + report JSON before the grid spend (`:1058-1100`) — never an anonymous rc=1.
7. **Redundant recompute:** NONE. Leg B's Qwen2.5 side stages #2162's banked ladder completions + committed coherence scores from HF at the parent pin (revision-pinned `stage_hub_prefix`/`stage_hub_file`, `decay.py:459-465`) — zero GPU, zero regeneration; the decay driver loads NO model (tokenizer-only, one `AutoTokenizer.from_pretrained` per side at `decay.py:214-225,318` — not per-row). The 1.47 GB vc_bank is reused pod-side; donor-screen inputs reuse #2329's banked anchors.

## Assumptions proceeded under

- `RUN.run_claim_queue` / `generate_batch` / `RUN._upload_dir` are the parent (#2329 v4) helpers whose batched/work-conserving/bulk-upload character the plan's item-(i) throughput inspection attests (plan §10); I verified the call shapes and the `issue2329_run.py` upload-helper docstring ("ONE bulk upload_folder commit ... exact-set verify") rather than re-reading the full 4,400-line parent module — its inner loops were reviewed in the parent round and this round's diff touches it only +58/−18.
- The worktree task copy's `plan.md → v4`; I reviewed against repo-root `tasks/followups_running/2329/plans/v8.md` (the symlink target the brief names, mtime 2026-08-19 16:27).

## Recommendation

merge — compute is sized, placed, batched, and saturates the hardware it holds.
