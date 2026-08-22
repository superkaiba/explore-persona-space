## Code-Reviewer Verdict — split-review r1 g2 (commit 9c0204adaa, unit 2/5: training entrypoint + cell registry + fan-out)

**Verdict:** FAIL
**Blocker tags:** none — the blocker is substantive (resume-predicate correctness), not a mechanical-contract tag; contract gates 0.5–0.9 were out of scope per the sub-scope brief.

Scope reviewed: `scripts/issue2225_train.py` (+1004) and `tests/test_issue2225_cell_registry.py` (+145), the full commit diff. Cross-module seams verified against unit 1 (`steer_train.py`, `directions.py`), `issue778_finetune.py`/`issue778_lib.py` constants, and `orchestrate/hub.py` signatures. Line numbers below are worktree-HEAD (unit 5 shifted lines but did NOT touch the resume logic — interdiff has 0 hits on `should_skip`/`_local_done`/`_hf_complete`/`_write_manifest`).

### Critical (drives the FAIL)

1. **Stale-adapter resume skip: manifest-at-START + presence-based "done" ships a prior fingerprint's adapter as the retrained cell.** `scripts/issue2225_train.py:535` writes the fingerprint manifest at cell START (before training); `should_skip` (397–414) skips iff fingerprint match AND (`_local_done` — adapter files merely EXIST locally, 408 — OR `_hf_complete` — files merely exist under the HF prefix, 411). Neither leg binds the artifact to the fingerprint that PRODUCED it, and a non-skip re-run never wipes the prior run's files (`out_dir.mkdir(exist_ok=True)` at 530 only; `_reap_training_residue` removes non-adapter residue only). Trace: cell completes under fingerprint F1 → code-fix/direction-rebuild flips to F2 → re-run writes the F2 manifest at START → crashes mid-train (`save_strategy="no"` writes nothing until the final save, so the F1 files survive untouched) → relaunch sees F2==F2 AND F1-era `adapter_model.safetensors` present → SKIP. The stale adapter is then evaluated as the retrained cell — a silent confound of the primary result. The whole-HEAD `code_sha` field (`_git_head`, in `cell_fingerprint`) makes the trigger LIKELY: any commit invalidates ALL 81 cells, so one mid-wave OOM/preempt during a crash-fix round strands several cells exactly in this state. Plan §9's letter licenses the start-manifest ("manifest row written at cell start") but its stated intent — "a mismatch ... invalidates the skip and re-runs the cell" — is defeated after any crashed retrain. Minimal fixes (either closes it): (a) on a non-skip run, unlink `out_dir/adapter_config.json` + `adapter_model.safetensors` immediately after writing the START manifest (closes the local leg) AND record an artifact-binding field (e.g. adapter file sha or `completed_at`) written only at save time that `_local_done`/`_hf_complete` require; or (b) move to END-manifest done-marker semantics ({fingerprint, completed, uploaded}), keeping a separate started-at breadcrumb if crash forensics want it. Fix must cover the HF leg too — the F1-era upload also satisfies `_hf_complete` under an F2 manifest.

### Concerns

2. **Upload failure is permanently swallowed by the resume skip.** `train_steered_cell` saves the adapter (648) then uploads with `raise_on_error=True` (653). A transient upload failure (HF 504/quota) exits rc≠0 → cell marked FAILED; on relaunch, fingerprint matches and `_local_done` is True → SKIP (408–410) — the upload is never re-driven, silently breaking the plan §9 / #664 per-cell upload contract (Step-8 upload-verification is the only backstop, hours later). The local-done skip path should verify HF-complete and re-drive `_upload_cell_adapter` when absent (or ride the manifest `uploaded` flag from fix 1b).

3. **CVD pinning uses absolute ordinals, not the parent's visible-device entries.** `run_fan_out` sets `env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}` (824) for `g in range(n_gpus)` where `n_gpus = torch.cuda.device_count()` in the launcher — a count taken UNDER the launcher's own CVD. If the launcher inherits a restricted/reordered CVD (SLURM partial-node allocation — fellows is the plan's named manual fallback lane — or a pre-set `CUDA_VISIBLE_DEVICES=4,5,6,7`), children are pinned to ordinals 0..n-1, ESCAPING the allowed set and colliding with other jobs. Correct form: when the parent CVD is set, split it and pin its g-th ENTRY. Benign on the plan's dedicated 8×H100 pod (CVD unset) — hence Concern, not Critical.

4. **Bare invocation launches the full 81-cell production fan-out; `--fan-out` is decorative.** `--fan-out` (975) is never read in `main()` — any invocation that isn't import-check/check-registry/preflight/single-cell falls through to `_resolve_cells` → `run_fan_out` (1093+). `uv run python scripts/issue2225_train.py` with no arguments starts ~42 GPU-h of training. Make the fan-out branch require the explicit `--fan-out` (or `--pilot`/`--cells`/`--smoke`/`--dry-run`) and error on a bare call.

### Suggestions

5. `pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id` (635) — falsy-zero trap: a tokenizer with `pad_token_id == 0` silently falls back to EOS. Use `if tokenizer.pad_token_id is not None`. Benign for Qwen2.5 (151643).
6. `--smoke` alone trains a FULL cell — the "tiny row slice" only engages when `--max-steps` is also passed (`_resolve_cells` returns `cells[:1]`, 960; the slice lives in `train_steered_cell`'s `max_steps` branch). Consider defaulting `max_steps` (e.g. 4) under `--smoke` so the flag matches its help text.
7. `run_fan_out` leaks the per-cell log filehandle if `subprocess.Popen` raises between `open` and `running[g] = ...`; wrap in try/except or open with a context that closes on launch failure. Minor.
8. (Fixed at HEAD, noted for the record) commit-time `cells_by_slug()[args.single_cell]` raised a bare `KeyError` on an unknown slug; unit 5's `resolve_cell` (1066) now raises a descriptive `ValueError`.

### Plan adherence (checked, no deviation)

- Registry grid matches plan §4.5 exactly: 81 = A16+B12+C16+D12+E12+F3+G3+I3+P3+H1; coefficient grids {0.5,1.5,3,5} / {0.25,0.75,1.5} / {0.5,1.5,3}; A–E span all 4 corpora, F/G/H/I/P evil-only; opinions cells steer evil (§12 A12). All pinned by the new test file.
- Paper-recipe hyperparameters imported verbatim from `issue778_finetune` (r=32/α=64/rsLoRA/lr=1e-5/1 epoch/2×ga8/completion-only/seed 0/`save_strategy="no"`), never re-typed — all 13 imported constants grep-verified present.
- L1 indices match plan §4.3 via unit 1's `L1_LAYER_IDX` (evil/syc 19, hallucination 15, 0-indexed); L2 = 9-layer band centered on L1 (bounds-asserted); L3 = all 28 with `build_incremental_vectors` (§4.2/App. J.3).
- Config H implements App. J.7.2 (per-row deterministic positive extraction system prompt via `issue778_lib.extraction_system_prompt`, no hook, nothing stripped at eval); config P's prefix mask resolves on user-only rows via the Qwen default system block (`compute_prefix_len` → `prefix_end_index`, 3-`<|im_start|>` assert — verified against unit 1 + issue1415).
- Unit-1 seam signatures verified: `SteeringHook(vectors, alpha, mode)`, `SteeredSFTTrainer(steering_hook=...)`, `SteeringDataCollator(pad_token_id=..., completion_only_loss=...)` (TRL 0.29 collator fields), `compute_prefix_len(tokenizer, prompt_messages)`; hub seams `verify_repo_paths_uploaded(..., path_in_repo=, repo_type=)` and `_upload(..., raise_on_error=True)` match.
- Fail-fast: missing dataset/direction files raise `FileNotFoundError` at fingerprint time; `_hf_complete` transport errors warn loud and err toward RE-RUNNING (never skip); fan-out collects per-cell failures and raises at the end (non-zero exit) — no swallowed failures on the launch path.

### Tests

- `uv run pytest tests/test_issue2225_cell_registry.py -q` → 20 passed in 2.7s (includes this commit's 10 registry pins plus unit-5 additions).
- `uv run ruff check` on both added files → clean.

**Recommendation:** fix Critical 1 (small: wipe-on-restart + artifact-binding field, or END-manifest semantics) and Concern 2 (same manifest redesign) before the production fan-out launches; Concerns 3–4 are pre-launch hardening of the same file.
