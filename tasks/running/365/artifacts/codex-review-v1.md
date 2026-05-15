<!-- epm:code-review-codex v1 -->
# Codex Code Review: 2^5 factor-screen package (task #365)

**Verdict:** CONCERNS
**Tier:** leaf
**Diff size:** +5163 / -1 lines across 22 files (10 new modules, 2 test files, events.jsonl touches)
**Plan adherence:** PARTIAL (1 critical blocker, 1 major gap)
**Lint:** PASS
**Security sweep:** CLEAN
**Needs user eyeball:** No GPU available locally; training and eval paths not exercised end-to-end

## Plan Adherence

- [A sys-prompt length — short/long]: implemented (SHORT_PERSONA_PROMPTS + LONG_PERSONA_PROMPTS)
- [B answer-format — short/long]: implemented (b_suffix(), B_LENGTH_BANDS)
- [C persona vs non-persona system prompt]: implemented; C0=persona, C1=Background-context (correct axis)
- [D on-policy / off-policy]: implemented; D0=on-policy (base Qwen), D1=off-policy (Claude) — polarity correct
- [E marker-only / whole-completion loss]: implemented; E0=marker-only (marker_only_loss = cell.e == 0), E1=whole-completion — polarity correct
- [24-persona panel, source names unaliased]: librarian/surgeon/programmer appear under their own names; medical_doctor + software_engineer appear as bystanders only
- [In-domain bystander stratification]: surgeon→medical_doctor, programmer→software_engineer+data_scientist
- [PREREGISTERED_INTERACTIONS = {(A,B),(B,E)}]: both present; B×E appears in INTERACTION_PAIRS and interactions.csv
- [Off-diagonal noise floor estimator]: off_diagonal_noise_floor() uses 8-cell E0×D0 rectangle
- [n=3 cluster-bootstrap wider_CI supplement]: cluster_bootstrap_difference() + wider_ci()
- [Log-ratio CI for E1<E0 (replaces >= 2x hard threshold)]: compute_e_log_ratio() with bootstrap
- [module-load smoke]: exits 0
- [empty-int sanitization]: _strip_empty_int_flags() handles --seed "" and --run-index ""
- [dispatcher script scripts/dispatch_factor_screen_365.py]: NOT present in diff
- [metrics.json schema that aggregator can read]: BLOCKING gap — see Issues below

## Issues Found

### Critical (block merge)

- `__main__.py:300-321` vs `aggregator.py:127-160`: **metrics.json schema mismatch — aggregator silently produces all-zeros**.
  - Evidence: `__main__.py` writes `"persona_panel_scores": persona_scores` where `persona_scores = {persona: {substring_rate, fuzzy_rate, per_question, ...}}` (nested by persona, then by question). `aggregator.py::_record_from_metrics_json` reads `payload.get("source_substring_rate", 0.0)`, `payload.get("leakage_rate_full", 0.0)`, `payload.get("leakage_rate_out_of_domain", 0.0)`, `payload.get("leakage_rate_in_domain", 0.0)`, `payload.get("per_bystander_substring_rates", {})`. None of these keys exist in the metrics.json written by `__main__.py`. `json.get()` silently defaults to 0.0 / {} for all of them.
  - Impact: Every cell will show source_rate=0, leakage_rate_full=0, and per_bystander={}. All main effects and interactions will be 0. The experiment produces structurally correct output files with scientifically meaningless zeros. There is no runtime error — it fails silently.
  - Fix: After computing `persona_scores`, derive and add the flat keys before writing metrics.json. Extract `source_substring_rate = persona_scores.get(source, {}).get("substring_rate", 0.0)`, compute `per_bystander_substring_rates = {p: persona_scores[p]["substring_rate"] for p in persona_scores if p != source}`, then compute `leakage_rate_full`, `leakage_rate_out_of_domain`, and `leakage_rate_in_domain` from the per_bystander dict using the stratification helpers from `persona_panel.py`. Write all five flat fields into the top level of metrics_payload.

### Major (revise before merge)

- `__main__.py:246-252` — **pool paths are per-cell, but plan demands E0/E1 cells share the same JSONL**. The pool lookup uses `output_dir / "pools" / f"{source}_a{cell.a}_b{cell.b}_c{cell.c}.jsonl"` where `output_dir` is the cell-specific directory (e.g. `eval_results/.../cell_00000/source_librarian/seed_42/`). Cells `00000` and `00001` (same data, differ only in E) have different output_dirs, so each must have its own copy of the pool. The plan Control table says "E loss-only flip: Same JSONL reused across E0/E1 for each A/B/C/D/source/seed." The current code does NOT enforce or enable reuse — it requires the dispatcher to duplicate the pool into every E-variant cell output_dir.
  - Fix: Add a `--pool-dir` flag or a shared-data-dir convention so all cells sharing (source, A, B, C) point to the same pool location, or add a pre-generation step that populates shared pools under a source-level dir before per-cell runs begin.

- `scripts/dispatch_factor_screen_365.py` — **Missing entirely**. Plan §9 specifies the dispatcher pseudocode in full. Without it, the 8-GPU fan-out, librarian gate, and pool pre-generation are not runnable. The `onpolicy.py` module is also never called from `__main__.py`, leaving on-policy pool generation orphaned.

### Minor (worth fixing but does not block)

- `aggregator.py:188-228` — `off_diagonal_noise_floor` docstring says "Cross-seed SD" but the implementation computes cross-CELL SD (8 cells in the E0×D0 rectangle; one record per cell from primary_records). "Cross-seed" implies variability across replications of the same cell, but the function varies the (A,B,C) axes within the fixed (E=0,D=0) slice. The docstring should read "Cross-cell SD within E0×D0 sub-rectangle." Functionally correct for the plan's intent; the docstring misleads.

- `onpolicy.py:50` — `questions: list[str] = None` uses a bare `None` default for a mutable parameter in a dataclass. Safe in practice (dataclasses don't share the default mutable object), but should use `field(default=None)` with an explicit type annotation of `list[str] | None` for clarity and mypy compatibility.

- `eval_panel.py:131` — vLLM `LLM()` is constructed per-cell for both `generate_completions` and `generate_random_control_completions` back-to-back. The `del llm; gc.collect(); torch.cuda.empty_cache()` teardown between them is correct for H100 targets, but if CUDA process memory is not fully released by vLLM internals, the second load may OOM on smaller GPUs. Fine for H100 (80GB); note for future readers.

## Unaddressed Cases

- No test exercises the `prepare_cell()` + JSONL-write path with actual pool data on disk. Acceptable pre-run.
- `LEAKAGE_N48_CITATION_NOTE` is correctly embedded in `aggregator.py` and surfaces in `factor_effects.json`. The analyzer must act on it before promotion. Confirmed present and documented.
- `write_cell_manifest` is never called by `__main__.py` — it is an aggregate-mode concern only. This is architecturally consistent.

## Style / Consistency

- `_strip_empty_int_flags` handles both `--flag ""` (two-token form) and `--flag=` (equals form) with correct index arithmetic. The known-int-flags list covers all numeric flags in the parser.
- `CPaddingError` is raised correctly when exact C0/C1 token length parity cannot be reached; callers are expected to fail preflight.
- All ruff checks pass; 16 files formatted.

## Unintended Changes

- The diff touches `tasks/planning/192/plans/v2.md`, `tasks/running/363/events.jsonl`, `tasks/running/370/events.jsonl`. These are events from concurrent runs — not related to #365. Standard worktree noise; not a concern.

## Security Check

- No hardcoded tokens or secrets.
- `HF_TOKEN` is read from `os.environ.get("HF_TOKEN")` — correct pattern.
- No `shell=True` subprocess calls.

## Recommendation

**Revise before merge.** The metrics.json schema mismatch (Critical) is a silent correctness bug that will make the aggregator output zeros for every metric across all 96 cells. It must be fixed before the experiment runs. The fix is straightforward: 8-10 lines in `_run_cell_mode` after `persona_scores = score_markers(eval_results)` to flatten source rate, per-bystander rates, and stratified leakage into the top level of `metrics_payload`. The dispatcher script and pool-sharing architecture are also blocking for actual execution but are lower-risk to add alongside.
<!-- /epm:code-review-codex -->
