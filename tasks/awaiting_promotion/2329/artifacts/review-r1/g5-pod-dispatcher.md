PASS

## Split-review sub-scope g5 — commit `43361fb39feb` (pod dispatcher `scripts/issue2329_ladder_dispatch.sh`, +334 lines, new file)

**Tier:** leaf (one-off pod dispatcher, not imported elsewhere; reviewed at trunk depth per pod-side conventions). File touched ONLY by this commit in the round (`git log 850a668..c46f29b -- <file>` = 1 commit).

### Pod-side convention checklist — all clean (verified, not just read)

- **No `task.py` shellout:** grep clean — reporting is `[phase=...]` breadcrumbs + driver-side sentinel under `LOG_DIR` default `/workspace/logs` (top-level, poller-glob-compatible).
- **GPU width:** derived from `nvidia-smi -L` (line 84) with a documented `SLURM_GPU_WIDTH_EXEMPT` justification — plan v8 §9 pins `backend: runpod`, dedicated `pod-2329-l` 1× H100, so dedicated-pod enumeration IS the allocation width. Per-worker `CUDA_VISIBLE_DEVICES="$g"` pinned in the launcher env; `--gpu-id` verified INFORMATIONAL in the driver (`issue2329_ladder.py:258` "informational; CVD pins the device") — no CVD-reindex ordinal trap.
- **Checkpoint/upload sequencing:** matches plan v8 L-lattice exactly (stage1 = gate0b→import-check→bank→anchors; stage2 = grid-gates→grid→margin→upload). L3's judge inputs leave the pod via the DRIVER's incremental L2 uploads ("rollout text uploaded IMMEDIATELY", plan §9 phase-order persistence) — the dispatcher correctly does not duplicate that.
- **Detached shape:** fan-out children are plain-backgrounded with real `wait` (no nested setsid/nohup — the one `setsid` grep hit is the explanatory comment, line 169); PIDFILE breadcrumbs truncated per fan-out; terminal `[phase=done]` only after full success (fork-source convention; ladder PIDFILE name `issue-2329-ladder-workers.pid` is distinct from the parent dispatcher's — no same-pod collision).
- **Fail-fast:** `set -euo pipefail`; every pipeline rc read via `PIPESTATUS[0]` inside a `set +e` window then explicit `exit "$rc"`; no failure-masking `|| true` (only `shift`, the bounded NUM_WORKERS fallback, and a diagnostic `tail`); all expansions quoted; every python invocation is `uv run python`; `UV_NO_SYNC=1` exported for the whole dispatch so the gate0b `transformers==5.15.0` pin isn't silently re-synced away (rationale documented in-header, incl. the #1689 MooseFS fan-out wedge).
- **Gate pre-checks fail BEFORE spend** and the driver independently re-requires the same three files for a non-smoke grid (`issue2329_ladder.py:1493-1501`) plus `--pools` for margin (line 1935) — belt-and-braces consistent. Dispatcher's inline no-surviving-rungs check matches the judge's actual schema (`issue2329_ladder_judge.py:394-410` writes `rungs -> {survived}`), and fails CLOSED on a missing `rungs` key.

### Executed probes (committed blob `git show 43361fb39feb:<file>`, /tmp roots, all halt pre-fan-out)

`bash -n` clean. All six misuse branches EXECUTED and match the commit-message claims: unknown phase rc=2; skip-without-reason rc=26; missing G0 rc=30; no-surviving-rungs rc=26 (crafted all-failed verdict); missing donor screen rc=25; missing pools rc=29. Each with an actionable stage-it-then-re-run message; no GPU/driver process ever launched.

### CONCERNS (all Minor — none blocks)

1. **stage2 does not re-assert the transformers pin** (`run_stage2`, line 290 — no `run_gate0b`/`--gate0b-check` at its head). Scenario: pod reprovisioned after a wedge, or any bare `uv run` on the pod during the ~1.2 h L3 idle window re-syncs the venv to 4.57.6 (the exact hazard the header documents) → a `stage2` launch fans out N grid workers that all crash at model load (qwen3_5 unknown arch). Fail-LOUD, and the plan's §8 same-pod-idle design makes it latent, but a one-line CPU-cheap `run_gate0b` (or a `--gate0b-check`-only assert) at stage2 head would make stage2 self-healing instead of an N-crash-log diagnosis.
2. **`EPM_2329L_SKIP_GRID_GATES` triggers on ANY non-empty value** (line 212, `[ -n ... ]`), including `0` — an operator exporting `=0` to mean "don't skip" (with a stale ≥10-char REASON in the env) silently takes the skip path. Convention elsewhere is `=1`; an explicit `= "1"` compare would close it. Note the skip's blast radius is bounded by design: the gate paths are still threaded and the driver's own requirement/content checks still bind — the env only bypasses the dispatcher's duplicated pre-check (a coherent escape hatch for dispatcher/driver schema drift).
3. **Plan-wording nit (plan-side, not this commit):** plan v8's smoke blind-spot block (line 134) states the pod `--smoke` has "no gate downgrades", while this dispatcher (line 252) + driver relax the three staged gate-FILE requirements under `--smoke`. The relaxation IS declared — dispatcher header ("declared blind spot", frozen PRIMARY donors unscreened) and plan line 105 — so this is NOT tagged `smoke-blind-spot-unenumerated`; flagging only so the plan's enumeration sentence can absorb the file-requirement relaxation on a future revision.

### Recommendation

PASS — merge as-is; item 1 is worth a one-line follow-up edit before the stage2 launch but does not gate this round.
