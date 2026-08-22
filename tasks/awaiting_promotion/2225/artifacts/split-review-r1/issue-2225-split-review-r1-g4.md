# Code Review: issue-2225 R1 split-review g4 — commit 8b2c549c65 "unit 4/5: judge driver + analysis + pod dispatcher"

**Verdict:** FAIL
**Blocker tags:** substantive
**Tier:** trunk (touches `src/explore_persona_space/experiments/issue2225/directions.py`; new `scripts/` drivers reviewed at trunk depth)
**Diff size:** +3207 / -1 across 5 files (`scripts/issue2225_{analysis,dispatch.sh,judge}.py`, `src/.../issue2225/directions.py`, `tests/test_issue2225_judge_analysis.py`)
**Plan adherence:** PARTIAL (sentinel-path contract broken; §7 re-pilot deferred in this commit — superseded in-round by ecccdf108e; §6.5 glob mismatch)
**Tests:** PASS with gaps (`run_sync_reissue`, `run_wave`/`reduce_unit`, dispatch phase_p0 gate untested)
**Tests actually run:** yes — `uv run pytest tests/test_issue2225_judge_analysis.py -q` → 25 passed, 17.3 s (HEAD file incl. unit-5 additions)
**Lint:** PASS (`ruff check` + `ruff format --check` clean on all three .py files; `bash -n` clean on dispatch.sh at this commit AND at HEAD)
**Security sweep:** CLEAN (no secrets; `orchestrate.env.load_dotenv` before HF imports in both drivers; conditional `.env` sourcing pod-side; no `task.py` shellouts — pod-side contract honored)
**Needs user eyeball:** None

## Plan Adherence

- Judge = `claude-sonnet-4-5-20250929`, N=6 @ 0.7, `max_tokens=1024`: ✓ — `scripts/issue2225_judge.py:68-73`, asserted against `issue778_lib` at `:129-131`
- Routing through the #663-hardened client (`judge_graded` → `judge_completions_batch`), never a hand-rolled poller: ✓ — `issue2225_judge.py:431,487-497`; `threshold_base` passthrough verified against `eval/graded_judge.py:222-234`
- Rule-26 pilot gate, forced-Batch, refusal before ≥5k-call production dispatch: ✓ — `issue2225_judge.py:658-740` (`threshold_base=FORCE_BATCH_THRESHOLD` at `:705`; `_require_pilot_gate` at `:728`)
- Rule-28 per-arm `n_api_refusal` + uncensored-rate digest + targeted SYNC re-issue at the identical instrument + ~250-item dual-scored parity: ✓ — `:353,588-652,896-1046` (but see Major 3: the re-issue phase is not resume-idempotent)
- Coherence rubric verbatim from the pinned persona_vectors clone: ✓ — `:235-254` (fail-loud on missing slots)
- Drop-never-coerce / transport-vs-content/api-refusal split preserved per rollout: ✓ — `reduce_unit` `:318-415`; `JudgeResult` field names verified against the real dataclass (`graded_judge.py:200-219`)
- Question-paired bootstrap 10,000 draws, frozen + selection-inherited CIs, pooled datasets-fixed, §3 lattice with exhaustiveness assert: ✓ — `issue2225_analysis.py:61-62,234-261,313-356,445-492`
- LINEAR probe only, GroupKFold over extraction questions, batched Gram-ridge over the 28-layer axis, 1-layer timed pilot before the battery, n<d regime declared (AUC not R²): ✓ — `:679-806` (`fit_regime_note` `:794-798`)
- Prefix+context both-arms rule (projection covers `context_end` AND `prefix_end` onto E2): ✓ — `:1004-1008`
- §12 A6 filtered-vs-unfiltered cosine: ✓ — `directions.py:411-441` + `analysis.py:1063-1105`; tensor names `{trait}_E2_unfiltered.pt` match the analysis reader
- Pod-side sentinel contract (no `task.py`, envelope via `issue778_lib.write_results_sentinel` carrying `sentinel_schema_version/kind/version/task_id`): envelope ✓, **path ✗ — Critical 1**
- §7 P0 gate criteria (ii)/(iii) + octave-shift recommendation: ± — verdict logic ✓ (`issue2225_judge.py` p0-verdict), but THIS commit routes a first-miss to a designed halt with "no coef-scale CLI exists" (dispatch.sh at 8b2c549) instead of §7's "re-pilot ONCE". Superseded in-round: ecccdf108e adds `--coef-scale`/`--p0-grid-arm` + the automatic re-pilot. No action needed; recorded for round completeness.

## Issues Found

### Critical (diff is wrong or introduces serious risk — block merge)

- `scripts/issue2225_dispatch.sh:70` (this commit; unchanged at HEAD): **every pod sentinel is written into a SUBDIRECTORY the VM poller's drain glob cannot see.** `LOG_ROOT="${EPM_I2225_LOG_ROOT:-/workspace/logs/issue-2225}"` is passed as `logs_dir` to `issue778_lib.write_results_sentinel` (dispatch.sh `write_sentinel`, lines ~93-110), so the files land at `/workspace/logs/issue-2225/issue-2225-epm_results-<epoch>.json`.
  - Evidence: `scripts/poll_pipeline.py:2321-2333` — the drain shell globs `/workspace/logs/issue-2225-*.json` (path-terminal, non-recursive; `*` does not cross `/`); the results-presence probe `poll_pipeline.py:2044` (`RS_FILES=(/workspace/logs/issue-2225-epm_results-*.json*)`) is equally top-level. The plan pins the top-level contract explicitly: §9 `phase_outputs` names `/workspace/logs/issue-2225-pilot.json` and `/workspace/logs/issue-2225-results.json`, and the `backend: runpod` pin is justified BY this sentinel contract (plan.md:274,314,319).
  - Impact: P3's `epm:results` and P0's `epm:progress`/`epm:smoke-result` sentinels (all 4 call sites in this commit + unit 5's re-pilot sentinel — all through the same helper) are silently invisible; the run "completes" with zero markers drained, the orchestrator's poll loop never sees the results envelope, and the octave-shift recommendation on a designed rc=7 halt never reaches the VM. Secondary: per-cell training logs under `/workspace/logs/issue-2225/p0_train/*.log` are also outside the poller's `/workspace/logs/issue-<N>-*.log` staleness globs, degrading stall detection.
  - Fix: default `LOG_ROOT=/workspace/logs` for the `logs_dir` handed to `write_results_sentinel` (keep phase logs in a subdir if desired — only the SENTINEL parent is contract-bound), or write sentinels explicitly with `logs_dir=Path("/workspace/logs")`.
  - Bug-class sweep (`sentinel-path-outside-drain-glob`): all sentinel emission goes through the single `write_sentinel()` helper → ONE fix site covers p0 progress/smoke, p3 results/smoke, and unit 5's re-pilot sentinel. No other pod-side writer in the round emits sentinels. No siblings elsewhere.
  - Mechanizable: yes — a unit test calling `write_results_sentinel(2225, ..., logs_dir=<dispatch LOG_ROOT>)` and asserting `path.parent == Path("/workspace/logs")` / `fnmatch(path.name, "issue-2225-*.json")` against the drain glob; or a lint that flags a `logs_dir`/`LOG_ROOT` default containing `logs/issue-<N>` as a directory.

### Major (diff needs revision before merge)

- `scripts/issue2225_dispatch.sh` phase_p0 hook-engagement count gate (this commit lines ~193-203; same shape at HEAD, plus unit 5's copy in `p0_run_repilot`): **the fresh-round grep-count gate is starved by resume-skip — spurious FATAL exit 7 misdiagnosed as §7 criterion-(i) failure.**
  - Evidence: `n_hook=$(grep -rlF "[steer-hook]" "$p0_logs" | wc -l)` vs `n_expect=len(pilot_cells())`; `issue2225_train.py:783-784` — a resume-skipped cell prints `[fanout] skip <slug> (resume)` to STDOUT and opens NO per-cell log (`log_path` opened only on launch, `:825-826`); the resume predicate includes HF-complete (`:411-412`), and pilot cells DO per-cell HF-upload (`:653`). On a FRESH pod re-entry after a P0 crash post-training (pod loss between train and verdict), all 8 cells skip via HF-complete → 0 fresh logs → `0/8` → exit 7 blaming the hook, and every reprovision re-hits it (the verdict-file resume-skip at phase_p0 entry only fires on a prior PASS).
  - Impact: a recovery path deterministically wedges into a false criterion-(i) diagnosis; the actual hook engagement already happened and was proven.
  - Fix: count launched-only cells (parse the fan-out's own launch/skip summary), OR have the skip branch append a `[steer-hook] resumed <slug>`-class line into the counted log dir, OR accept per-cell evidence from the cell manifest (record hook-engagement in the per-cell manifest row at train time and gate on that).
  - Bug-class sweep (`count-gate-starved-by-resume-skip`): sibling instance = unit 5's `p0_run_repilot` recheck over `p0_train_repilot/` (out of this commit's scope; already flagged by the g5 sub-review — grid-overlap cells resume-skip there even same-pod). No other grep-count gates in this commit.
  - Mechanizable: yes — a bats/pytest-subprocess test running phase_p0's count logic against a log dir with N-1 logs + one `skipped-resume` stdout record must not exit 7.

- `scripts/issue2225_judge.py:896-1009` `run_sync_reissue`: **not resume-idempotent — a re-run double-appends the SAME sync draws into already-merged units, silently corrupting the merged draw sets.**
  - Evidence: targets are selected from `q["rollout_n_api_refusal"][ri] > 0` (`:938-940`), which never resets after a merge; the merge appends `sync_scores[iid]` to `rollout_draw_scores` (`:977-981`) and writes the block in place per unit (`:998`). On a second invocation (crash partway through this multi-hour sync wave → rerun; the phase checkpoints per unit, so pre-crash units ARE reprocessed), `judge_graded` re-serves the identical draws from the same `cache_dir` (`reissue_cache/sub/tag/d{depth}`, `:958`) and they are appended AGAIN — rollout means shift toward the sync scores, `n_rollouts_scored` unchanged but per-rollout draw multiplicity silently doubles.
  - Impact: the rule-28 remediation path — exactly the path that runs when the headline cells are censored — corrupts its own primary-DV inputs on any resume. Rule-24(ii)'s "duplicated draw masquerading as a recovery" is the named anti-pattern.
  - Fix: skip units whose `judge_meta.api_refusal_reissue` already exists (cheapest), or persist per-unit the reissued iid→draw-count map and merge idempotently (replace, not append).
  - Bug-class sweep (in-place-merge phases): `run_wave` (fingerprint-keyed skip — idempotent ✓), `run_assemble`/`run_digest` (recompute-from-partials ✓), `run_upload` (repack+re-upload ✓), probe/projection partials (append JSONL, last-wins dedup on read + done-set skip ✓). `run_sync_reissue` is the only non-idempotent in-place mutator. No further siblings.
  - Mechanizable: yes — a unit test running the merge step twice over a fixture partial + fixed fake scores and asserting draw-list length is stable.

### Minor (worth fixing but doesn't block)

- `scripts/issue2225_analysis.py:861-874`: the probe APPLICATION resume key is `(tag, trait)` only — not keyed on the probe-bundle identity. A re-fit of the bundles (e.g. `--force` on a fits-only rerun, or a bundle regenerated after a pool fix) leaves stale application rows in `probe_shifts_partial.jsonl` that resume-skip silently reuses (the code-style resume-key rule: every output-affecting input in the key; #722 r3 class). Fix: include a bundle fingerprint (e.g. mtime/sha of `probe_bundle_{trait}.pt`) in the JSONL row and compare. Mechanizable: yes.
- `scripts/issue2225_judge.py:640-651` `run_digest`: after a completed `sync-reissue`, the digest still reads the ORIGINAL censored accounting and re-prints "Run --phase sync-reissue BEFORE any contrast" with no reissue-aware branch — a completed remediation is indistinguishable from an unremediated wave at the digest surface (the analyzer must dig into `judge_meta.api_refusal_reissue` per unit). Add a reissue-aware line (counts recovered) to the digest. Mechanizable: yes.
- `src/.../issue2225/directions.py:474` + `:69`: `meta["variants"] = list(VARIANTS)` still claims 3 variants while 5 tensor files are now written (`E2_unfiltered`/`E3_unfiltered` added by this commit); `norms` correctly carries all 5. Cosmetic meta inconsistency.
- `scripts/issue2225_judge.py:1052-1089` `_pack_large_json`: a >9.5 MB raw file whose bulk is NOT under `all_scores` would emit its entire payload as ONE oversized header line in shard00 (re-entering the >10 MB LFS force-route). Currently unreachable (`judge_graded` save_raw puts the bulk in `all_scores` — verified `judge_result_from_save_raw` reads it), but a guard (`assert len(header) <= limit_bytes`) is one line.

## Unaddressed Cases

- Production P4 wave under default auto routing: coherence chunks are ~12 units × ~200 rollouts × 1 draw ≈ 2.4k calls — near the tier-scaled sync/batch crossover, so the coherence wave may route SYNC at full (non-batch) price while trait chunks route Batch. Consider `--force-batch` on the production dispatch or a larger `--units-per-wave` for the coherence wave. (Cost, not correctness.)

## Style / Consistency

- Clean: per-unit progress lines with the canonical `[phase] unit k/N key elapsed=` shape; atomic JSON writes throughout; deferred heavy imports with a real `--import-check` that executes them (the #606 pattern); no import-shadowing in `main()` (checked against the #1739 UnboundLocalError class — no post-branch references to branch-imported names).

## Unintended Changes

- None found; the directions.py hunk is scoped to the A6 additions the plan names.

## Tests

- New coverage: rule-27 round-trip through the REAL `parse_judge_json`/`_score_from_parsed` path (plain + fenced + REFUSAL/out-of-range drops); arm-identity contract incl. the judge↔analysis filename cross-pin; batch custom-id charset/length/bijectivity; lattice exhaustiveness; matched-coherence selection; frozen bootstrap determinism/coverage; selection-inherited = frozen under constant selection + NaN-invalid draws + opinions fixed-blend; Gram double-centering vs explicit reference; batched ridge vs per-layer reference + singular-slice pinv fallback; projection removal/idempotence; AUC. These execute real bodies (no seam stubs) — good Step 3.8 posture; `JudgeResult`/`judge_pilot_gate`/`stage_hub_file`/`_upload` call shapes verified against the live signatures.
- Missing coverage: `run_wave`/`reduce_unit` (chunked reduce over a synthetic JudgeResult), `run_sync_reissue` (incl. the Major-3 idempotency property), `run_upload`/`_pack_large_json` sharding, and the dispatch.sh phase_p0 gate logic. None currently pins the sentinel path contract (Critical 1).
- Existing tests still valid: yes (25/25 pass at HEAD).
- Sandbox status: ran normally.

## Security Check

- No issues found. No hardcoded secrets; heredoc python payloads take parameters via env vars (interpolation-free quoted heredocs); the one `$var`-interpolated `python -c` (`prior_pass` read) interpolates a constant path.

## Recommendation

revise-then-merge: fix the sentinel `LOG_ROOT` path (Critical — one-line default change + a pinning test), make the phase_p0 hook-count gate resume-proof, and make `run_sync_reissue` idempotent. Everything else in this commit is solid — the judging discipline (instrument pins, drop-class accounting, pilot gating, rule-28 remediation design) and the analysis statistics (paired/selection-inherited bootstrap, Gram-ridge probe with the n<d regime correctly declared) closely track the plan and the project rules, and the committed math tests are real-body tests.
