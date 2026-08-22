# Code Review: issue-2225 split-review r1 g3 — acabdf4124 "unit 3/5: eval-gen + MMLU + capture stack"

**Verdict:** CONCERNS
**Blocker tags:** none
**Tier:** leaf (3 new `scripts/issue2225_*.py` files; no shared-module edits)
**Diff size:** +1817 / -0 across 3 files
**Plan adherence:** PARTIAL (2 items — MMLU smoke dial missing; response-avg capture mechanism deviates from plan §4.6 item 4 wording, disclosed)
**Tests:** INSUFFICIENT (no pytest units for these 3 files anywhere in the round; in-code asserts + import-checks partially cover)
**Tests actually run:** yes — ruff, all three `--import-check`s, `--list-targets`, two `--dry-run` compositions, and a live Qwen2.5-7B-Instruct tokenizer probe of the prefix-end 3-`<|im_start|>` assert
**Lint:** PASS (`ruff check` clean on all 3 files)
**Security sweep:** CLEAN
**Needs user eyeball:** None

## Plan Adherence

- §6.5 arithmetic (86 targets / 67 single-trait / 19 three-trait / 124 unit files): ✓ — `issue2225_eval_gen.py:169-180` asserts; executed `--list-targets` → `86 targets (67 single-trait, 19 three-trait); 124 trait units; 19 narrow-domain units` (19 narrow = 17 opinions cells + baseft_mistake_opinions + base, matching §4.6 item 3).
- §4.6.1 vLLM batched generation, never sequential HF: ✓ — `issue2225_eval_gen.py:417: res = llm.generate(chunk, sp, use_tqdm=False, **kw)`, chunked (`EPM_VLLM_GREEDY_CHUNK_SIZE`, per-chunk `[vllm-chunk]` INFO line, #664 shape), one shared `enable_lora=True, max_lora_rank=32` engine per GPU (`issue778_lib.build_vllm_engine:220-246` verified), adapters swapped via `LoRARequest` with distinct `lora_int_id` (`:405`, `:564`).
- §4.6.1 `max_new_tokens=2048` + cap-hit reporting: ✓ — `issue2225_eval_gen.py:101`, per-unit `finish_reason=="length"` fraction in every payload (`:450-474`) + run digest with `units_over_trigger` at the 2% trigger (`:753-790`) + WARN log (`:481-488`).
- §4.6.2 MMLU lm-eval vLLM, 0-shot, full set, `lora_local_path` + `max_lora_rank=32`, `enable_lora` NOT passed: ✓ — `issue2225_mmlu.py:195-206`; verified against installed lm_eval 0.4.11: `VLLM.__init__` has `lora_local_path: str|None`, `max_lora_rank: int = 16` (default 16 confirms 32 is mandatory), and sets `"enable_lora": bool(lora_local_path)` itself. Group aggregate lands in `results` (evaluator's `results_dict` merges group rows into `results_agg`), so `raw["results"]["mmlu"]["acc,none"]` (`:285-293`) is the right read; `None` acc raises (`:303-304`).
- §4.6.2 merge fallback, ≤4 staggered merges + headroom + delete-after-eval: ✓ — `merge_slot` O_EXCL semaphore with dead-pid reclaim (`:136-167`), 20 GB headroom assert (`:268`), merged dir deleted in `finally` (`:279-281`); adapter never deleted.
- §4.6.3 narrow-domain generation (100 deterministic training-distribution opinions questions, 1 rollout, seed 0): ✓ — `issue2225_eval_gen.py:341-382`; formatted-prompt budget filter drops (never truncates), drop counts digest-only (#952 loader rule).
- §4.6.4 capture: three positions × 28 layers × fp16: ± — response-avg per rollout via reused `issue778_lib.capture_response_avg_all_layers:289`, context-end via `capture_last_prompt_token_all_layers:334`, prefix-end via `directions.capture_prefix_end_all_layers:297` (token-id boundary through `issue1415.steering.prefix_end_index`, verified live: the user-only eval prompt renders Qwen's default system turn → exactly 3 `<|im_start|>` occurrences, so the assert passes). fp16 store + row-alignment indices + meta sidecar ✓. BUT the plan's "(per-segment token-id concatenation)" claim does not match the named helper — see Major 2.
- #952 resume fingerprints: ✓ (mostly) — eval_gen keyed on adapter sha/trait/N/rollouts/temp/cap (`:253-298`); capture ADDITIONALLY keys on `gen_input_sha256` (`:80-96`) so a regenerated P2b unit invalidates its capture (closes the stale-artifact-resume class); mmlu keys on adapter sha/task/fewshot/seed/model/lm-eval version (`:89-104`). Gap: eval_gen + capture omit the base `model` name — see Minor 1.
- Checkpoint-per-unit + per-unit progress line + fail-loud fan-out: ✓ — atomic per-unit writes (`_atomic_write_json`, tmp `.tmp.pt` → `replace`, correctly avoiding the `.npz.tmp` suffix trap), `[eval-gen] unit k/N ... elapsed=` lines, fan-out raises at end naming failed units + logs (`issue2225_eval_gen.py:596-652`).
- CVD launcher-env pin + parent-side one-time base-model prestage (#1315): ✓ — `env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}` (`:623`), `_prestage_base_model` before every fan-out; adapter staging is per-worker but target sets are disjoint (striped shards / per-target subprocesses), so no shared-dest race.
- Upload = one `upload_folder` commit per subtree, never per-file loops: ✓ — `_upload(local, DATA_REPO, "dataset", prefix, raise_on_error=True)` folder branch (signatures verified against `hub.py:1426`); mmlu upload prunes transient scratch/slot/merged dirs first (`issue2225_mmlu.py:388-391`).
- vLLM teardown: ✓ — worker reaps via `issue778_lib.reap_vllm_engine` (full v1 recipe: `engine_core.shutdown()` → guarded `destroy_process_group` → gc/empty_cache/ipc_collect) in a `finally` before the `sys.exit(0)` terminal, which is the sanctioned combination for the #1739/#2149 finalize-deadlock class.
- Deliberate documented deviation (acceptable): P2b fan-out stripes targets statically instead of work-stealing per unit — rationale recorded in the module docstring (engine rebuild ~90 s × 124 units avoided; durations near-uniform). Consistent with plan §4.8's architectural-parity posture.

## Issues Found

### Critical (block merge)

None.

### Major (needs revision before merge — neither is re-roll-grade; both fixable in a follow-up round)

- `scripts/issue2225_mmlu.py` (whole file): **no `--limit` dial, so the plan's declared P0 MMLU smoke is unimplementable through this entrypoint.**
  - Evidence: plan §4.8 Smoke blind-spot enumeration item (b): "P0 MMLU runs `--limit 200` — full-set wall/memory first realized at P2c". `build_argparser()` (`:400-415`) exposes no `--limit`; `_run_lm_eval` (`:214-234`) composes no `--limit`; and no other round file wires an MMLU smoke (`issue2225_dispatch.sh` references mmlu only at p2c/upload, lines 480-487/507 — no `--limit` anywhere). `--smoke` only narrows to the base TARGET and still runs the full 14k-question set.
  - Impact: the lm-eval invocation path runs for the first time at production. A defect at the RESULTS-PARSE stage (after `_run_lm_eval` returns) would burn all 86 full MMLU evals before surfacing, since `fan_out_subprocesses` raises only at the end. (Mitigated: I verified the `lora_local_path`/`max_lora_rank` surface and the group-in-`results` serialization against the installed lm_eval 0.4.11, so the residual risk is moderate — but the plan's disclosed smoke coverage does not exist.)
  - Fix: add `--limit N` (threads `--limit` into the lm-eval argv and into the fingerprint so a limited run never resume-satisfies a full run) and wire the P0 `--limit 200` leg in the dispatcher.
  - Mechanizable: yes — grep the script for `--limit` + assert the fingerprint dict carries a `limit` key.

- `scripts/issue2225_capture.py:20-27, 216-229`: **response-avg capture mechanism deviates from plan §4.6 item 4's stated "per-segment token-id concatenation" — realized helper is string-concat + separate prompt encode; the commit substitutes a report-only seam audit.**
  - Evidence: plan §4.6 item 4: "(per-segment token-id concatenation; `issue778_lib.capture_response_avg_all_layers` grep-verified at line 289 ...)". The named helper (read at `issue778_lib.py:289-333`) does `text = prompt + response; inputs = tokenizer(text, ...); prompt_len = len(tokenizer.encode(prompt, ...))` — exactly the string-concat pattern the gotchas teacher-forced-capture rule flags. The commit knows this (docstring "BPE-SEAM AUDIT") and adds `seam_audit` (`:119-129`), whose predicate (`enc(prompt+response)[:len(enc(prompt))] != enc(prompt)`) exactly matches the helper's failure condition, with per-unit `seam_mismatch_count` in the meta + manifest + a WARN.
  - Impact: seam-merged rows get a ±1-token-shifted response-avg boundary; counted and disclosed, not fixed. The reuse is defensible — #778's probe-pool activations (the P5 probe's training data) were captured with the SAME helper, so forking to ids-concat here would create a probe-train/probe-apply capture-convention mismatch — but the plan text and the realized mechanism disagree, and nothing yet consumes `seam_mismatch_count`.
  - Fix: carry this as a stated deviation into the clean-result scope caveats, and have the P5 analysis either exclude or sensitivity-check seam-flagged units (the count is already in every meta sidecar). No code re-roll required if the deviation is recorded.
  - Mechanizable: partially — assert `seam_mismatch_count` present in every meta (already structural); the deviation-recording is prose.

### Minor (worth fixing, doesn't block)

- `issue2225_eval_gen.py:253-279` + `issue2225_capture.py:80-96`: resume fingerprints omit the base `model` name while `issue2225_mmlu.py:89-104` includes it. A `--model` variation would silently resume-reuse another model's outputs (#722 r3: resume keyed on EVERY output-affecting regime key). One-line fix: add `"model": <resolved model>` to both fingerprints. Mechanizable: yes — assert the key set.
- `issue2225_capture.py:457`: `--upload-tags` split lacks the empty-token filter the other splitters have (`[s.strip() for s in ...split(",")]` without `if s.strip()`) — a trailing comma yields tag `""`, making `out_root/capture/""` resolve to the whole capture dir and the dest prefix `f"{CAPTURE_HF_PREFIX}/"` trailing-slash-malformed.
- `issue2225_mmlu.py:146-167` (`merge_slot`): a holder dying between the O_EXCL open and the pid write leaves an empty slot file that is never reclaimed (`holder=0` short-circuits `if holder and not _pid_alive(holder)`), permanently narrowing the semaphore until manual cleanup. Cheap fix: treat an empty/unparseable slot older than N minutes as stale (mtime gate).
- `issue2225_capture.py:137` / `issue2225_mmlu.py:249`: `targets_by_tag()[args.single]` raises a bare `KeyError` for an unknown tag (eval_gen's `resolve_targets` has the nice message + §7 re-pilot fallback). Deliberate that pilot slugs get no MMLU/capture, but the failure message is opaque.
- `issue2225_eval_gen.py:109-112`: the pre-registered cap-hit remedy ("re-generate those rows at ≥2× the cap") is not implementable through this rig without threading `max_model_len` — `lib.build_vllm_engine` pins 4096 and `PROMPT_TOKEN_BUDGET` goes negative at `max_new=4096`. Fine now (report-only trigger), but if a unit trips 2%, the re-gen needs a code change, not just a dial.

## Unaddressed Cases

- Narrow-domain rollouts are NOT captured by P2d (capture iterates `target.traits` only, never `NARROW_KEY`) — consistent with my reading of plan §6 (probe DV is over trait-eval rollouts), but worth one line in the clean-result so the omission reads as designed.
- A degenerate single-token response that BPE-merges entirely into the prompt tail would produce an empty response slice (NaN response-avg) — such rows are counted by the seam audit, but the NaN itself is stored silently. Negligible at 2048-cap free generation; noting for completeness.

## Style / Consistency

- Clean. Consistent with the sibling issue778/issue1333 dispatcher conventions (CVD pin, `[phase=]` breadcrumbs, atomic writes, `#823 sys.path` guard, `load_dotenv` before heavy imports, `VLLM_WORKER_MULTIPROC_METHOD=spawn` at module top before any vllm import in the one file that builds engines).

## Unintended Changes

- None — three new files, no modifications elsewhere.

## Tests

- New coverage: none in `tests/` for these files (round tests cover cell_registry / figures / judge_analysis / steer_hook only).
- Executed by this review: `ruff check` PASS; `--import-check` ×3 PASS (deferred imports incl. vllm/lm_eval/peft execute; `assert_args_attributes_defined` whole-module argparse check passes); `--list-targets` PASS (86/67/19/124 asserted + printed); `--dry-run` fan-out compositions sane (disjoint striped shards, CVD per worker, `--narrow-domain`/`--merge-fallback` threading correct); live tokenizer probe PASS (user-only eval prompt renders Qwen default system turn → 3 `<|im_start|>`, so `prefix_end_index`'s assert holds on the production prompt shape).
- Missing coverage worth adding: `unit_done`/fingerprint mismatch re-run predicate; `resolve_targets` re-pilot fallback; `seam_audit` on a constructed merging pair; `merge_slot` reclaim; `load_narrow_questions` determinism + budget-drop.
- Existing tests still valid: yes (no existing files touched).
- Sandbox status: ran normally.

## Security Check

- No issues found. No secrets; HF token flows only through `orchestrate.hub` helpers; no eval/exec; upload destinations are the fixed project repos; harmful rollout text is never printed (both scripts' content-hygiene contract holds — progress lines are counts only).

## Recommendation

Merge-eligible after the two Majors are addressed as cheap follow-ups: (1) add the `--limit` dial + P0 MMLU smoke leg (or explicitly re-scope plan §4.8(b)); (2) record the response-avg string-concat capture as a stated deviation and wire `seam_mismatch_count` into the P5 analysis. Minors 1-2 are one-line fixes worth taking in the same pass. No re-roll: the enumeration, resume, vLLM usage, MMLU invocation surface, and capture-position mechanics all check out against the installed libraries and the plan.
