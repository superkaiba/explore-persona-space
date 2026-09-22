# Kimi compute implementation review

## Verdict

**Planning: APPROVE. Compute implementation: APPROVE after the fixes verified below.**

Latest reread confirmed all three findings were fixed: the workload explicitly binds and verifies all five cache/runtime/temp paths on the `/workspace` mount; the analysis records complete fit durations and checks the slowest observed fit × remaining fits × 1.25 + 900 seconds against the original deadline after each fit; Kimi rejects an omitted allocation duration before constructing a contract. The former pure-helper repro now raises `ValueError`. `bash -n scripts/story_persona_crossmodel_workload.sh` passes. The plan now matches the 25-chunk cadence, implemented cache paths, measured token count, projection formula, and bounded local-storage handling.

The revised plan now makes the final raw-store durability barrier and cache/mount contract explicit. Its first-fit projection and allocation fence are reasonable declared gates. Small documentation corrections remain: the inherited checkpoint cadence is 25 chunks, not the plan's 50; the implemented cache paths use `/workspace/.cache/`; the exact inherited projection is `remaining_chunks × slowest_warmed_chunk_seconds × projection_margin + preservation_reserve_seconds`, with code requiring margin ≥1.25 and reserve ≥900 seconds. Record the actual token-derived arithmetic work floor and bounded local footprint when finalizing the launch manifest. These corrections do not change the approved research design.

This review inspected the uncommitted Kimi changes in `/home/thomasjiralerspong/.codex/worktrees/story-persona-kimi-20260922` on 2026-09-22. It covers compute/persistence/monitor-adapter code, not full certification of the vLLM hook or statistical analysis. No provisioning, task mutation, or Claude/Anthropic invocation occurred.

## Findings resolved in the latest revision

The following records the original findings and their required behavior; all three are resolved by the latest reread described above.

1. **Enforce the declared cache and runtime mount contract.** `scripts/story_persona_crossmodel_workload.sh` sets `HF_HOME`, `UV_CACHE_DIR`, and `TMPDIR`, but the current diff does not set `HF_HUB_CACHE` or the promised `UV_PROJECT_ENVIRONMENT`, nor verify/record the resolved mount/device of those paths. Its quota check inspects only `/workspace`. An inherited cache override or a symlink can still route a large write to the small container filesystem despite the displayed environment paths. Before environment installation/model loading, create the intended directories, reject an inconsistent `HF_HUB_CACHE`, bind the existing base runtime explicitly, and assert that the resolved paths lie on the intended `/workspace` mount; record the evidence and a bounded allocation canary. The generic HF_HOME warning is not this assertion.

2. **Implement the planned measured CPU-fit gate.** The analysis file currently has no first-fit duration/projection check. The new shell timeout correctly prevents unbounded analysis and preserves a 900-second reserve, but it does not implement the plan's promised complete first-full-bank-fit measurement and remaining-fit projection. Time the first actual full-bank fit, save its timing and 12-fit code-derived count, compute a conservative remaining-fit projection against the same original absolute deadline, and fail through the existing preservation path before proceeding if it cannot fit. Keep the shell timeout as an independent final bound.

3. **Reject Kimi's unvalidated default allocation path.** `contract_from_pod` requires a Kimi-keyed ledger but calls `allocation_seconds` only when `requested_seconds` is not `None`. Omitting it therefore skips the new Kimi authority, spent-time, and single-allocation checks and returns the generic 7,200-second default. A pure local synthetic test accepted a ledger with `max_gpu_hours=0`, no `kimi_authorization`, and an already-used pod. The CLI currently requires `--allocation-seconds`, so the intended CLI path is protected; nevertheless the public helper has an unsafe default. Require an explicit duration for Kimi, or unconditionally validate its chosen default through `allocation_seconds`. Add a focused regression test for omitted duration and an invalid/reused ledger.

## Verified positive findings

- Kimi capture writes a fresh `capture_complete.json`, then synchronously calls `checkpoint(out, cfg)` before returning. This covers the final 20 of 270 chunks beyond the periodic 25-chunk cadence. The uploader inventories every file, verifies remote sizes/content hashes at an immutable revision, and propagates errors.
- Kimi's destination is distinct: `issue2673_deepseek_comparison/20260922_v1/kimi/analysis_tensors`; its result branch is `codex/story-persona-kimi-20260922`. The old DeepSeek/Qwen prefixes and result branches are retained. The legacy parent prefix name is cosmetic, not an overwrite risk.
- Artifact validation uses 2,160 rows for Kimi and 1,920 for the old models; registered selected layers remain explicit. Full completion still requires the exact analysis blocks, all three fits, output inventory, and matching hashes.
- The new Kimi ledger path does not silently inherit the consumed DeepSeek authorization when an explicit duration is supplied. It requires 28 GPU-hours, one allocation, TP8, and the current user instruction, rejects other known pod IDs, preserves provider `createdAt` as the paid start, and refuses to extend an existing allocation deadline.
- Capacity-monitor completion validation is parameterized by model, expected rows, and chunks while retaining the original DeepSeek defaults. Run-specific configuration must explicitly set `model_key=kimi`, `expected_rows=2160`, and `expected_chunks=270`; it must not reuse completed DeepSeek state.
- The analysis shell timeout is derived from the original allocation deadline minus the preservation reserve, rather than starting a fresh 3.5-hour clock.

## Checks executed

`uv run --no-sync pytest -q tests/test_story_persona_storage_contract.py tests/test_story_persona_capacity_monitor.py tests/test_story_persona_crossmodel_artifacts.py` with the prescribed thread caps: **85 passed in 7.83 seconds**. These inherited regressions are useful but do not cover all new Kimi branches or replace the full TP8 smoke.

The local synthetic allocation repro above invoked only the pure contract helper with invented data; no API call or paid resource was created.

## Remaining launch prerequisites

Rerun the focused tests after the new Kimi regression is added and review the final committed source. Before handing off unattended work, verify the new capacity/experiment configuration, distinct persistent state, original-deadline accounting, real recovery-worker canary, acknowledged notification, scheduled watchdog tick, and quota-aware pod preflight. Actual native-INT4 residual extraction must pass the registered full-width TP8 numerical smoke. None of this review is evidence that a model has already launched or that any vectors have been captured.

Nonblocking refinement: if a fit-time projection fails, the just-computed fit currently raises before it is inserted into the persisted block. Save that completed fit before the intentional halt so a later CPU-only recovery can reuse it. The costly raw capture is already durably checkpointed, so this does not threaten model-extraction artifacts.
