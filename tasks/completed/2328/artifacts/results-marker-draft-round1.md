<!-- epm:results v1 -->

## Completion Report

**Task:** #2328 round 1 — read-only `marker-status` three-source presence classification + false-destruction escalation guard (approved plan v2, implemented in full)
**Status:** SUCCESS

### (a) What was done

- `src/explore_persona_space/task_workflow.py` (+603): new read-only section after `has_event` — `STASH_WINDOW_FRESH_SECS=1800`, `_PATCH_FILENAME_RE`, `_LEDGER_OP_KINDS`, `_marker_status_git` (monkeypatchable `git -C repo_root()` runner), `_parse_iso_dt`, `_head_blob_rows` (ls-tree existence probe: rc0+non-empty=exists / rc0+empty=VERIFIED-MISSING / rc!=0=error), `_head_events_read` (direct relpath + REGISTRY.json-at-HEAD fallback for uncommitted status moves), `_worktree_events_read`, `_deferral_ledger_rows`, `_ledger_row_matches` (MF-1: task-id + trailing-segment path eligibility + `task #<id>: <kind>` message-prefix matching, op table for create/set_goal/promote/append_concern_event, unmapped ops KIND-BLIND; version-blind + note-truncation-tolerant, N7), `_newest_head_commit_iso` + `_classify_deferred` (N8 staleness: every-matched-row-stale => `unknown` + `stale-ledger-row` + `git log -p --since=...` forensic exit, never `absent`), `_inflight_window_probe` + `_probe_patch_file` (MF-2/MF-A: `index.lock` => `commit-in-flight`; fresh `~/.cache/pre-commit/patch<epoch>-<pid>` fires `stash-window-live` ONLY when the pid is alive AND its `/proc/<pid>/cwd` realpath == this repo root — pid- and repo-scoped, never global-mtime; unparseable fresh filename / probe OSError => `in-flight-probe-error` => `unknown`), `_tree_row_matches` / `_tree_excerpt` / `_ledger_excerpt`, `list_events_head_union`, `marker_status` (verdict assembly: any leg error => `unknown` even with matches; head match => `present-committed`; worktree => `present-uncommitted`; live ledger => `pending-deferred`; probe gates `absent`). `__all__` += `STASH_WINDOW_FRESH_SECS`, `list_events_head_union`, `marker_status`.
- `scripts/task.py` (+90): `marker-status <N> <kind> [--note-contains S] [--version K] [--json]` subcommand (`cmd_marker_status` after `cmd_latest_marker`); human output leads with a query-echoing verdict line (`verdict: <v> — task #<N> kind=<kind> version=<any|K> note-contains=<none|S> read-at=<ts>`) + per-leg lines + guidance; rc lattice: 0 present-committed/present-uncommitted/pending-deferred, 4 absent (EXCLUSIVE), 5 unknown, 2 argparse, 1 crash/unknown task id.
- `.claude/rules/repo-root-uncommitted-state.md` (P1, +52/-2): new H2 `## Marker-presence reads (events.jsonl) — the false-destruction escalation (#2328)` — the canonical four-verdict read, the escalation guard (a destroyed/re-append conclusion is INERT unless it quotes a `marker-status` verdict of `absent`), the post-marker stderr-visible rule; Files-of-record row.
- `.claude/agents/code-reviewer.md` (P2, 1 line): the deferred-commit floor line now routes through `marker-status <N> <kind>` — WORKING-TREE reads are named non-evidence, only verdict `absent` supports absence, and the reviewer NEVER emits a re-append/restore instruction (replaces the stale `re-probe task.py view <N> --json` instruction).
- `.claude/rules/codex-composer-common.md` (P3, +18): H2 `## Marker-presence reads (events.jsonl) — stash-window discipline (#2328)` — composed prompts must not instruct destroyed/re-append conclusions unless quoting verdict `absent`.
- `.claude/skills/issue/steps/09-step-5.md` (P4, +2): durable-verdict paragraph after Step 5's re-read snippet — a marker ABSENT from the re-read is not proof of absence; `marker-status` gates any missing-marker conclusion.
- `.claude/rules/LESSONS.md` (P5, 1 row): repo-root-uncommitted-state.md trigger widened with "a marker/events.jsonl row reading as missing (stash race; `task.py marker-status`)" (row 260 B <= 280; file 10,407 B <= 10,492 — no cap raise).
- `CLAUDE.md` (P6, 1 line): post-marker paragraph gains "run every post with stderr VISIBLE — never `2>/dev/null`" (#2325's suppressed deferral ERROR hid an 11-minute exposure).
- `tests/test_task_workflow_marker_status.py` (NEW, +562, 23 tests): functional suite over a local `fake_repo` fixture — present-committed + CLI rc0, present-uncommitted, the #2325 pending-deferred reconstruction (post_event under injected `_git_commit` crash + `git checkout --` revert), absent rc4, tree filters, ledger version-blindness + note-truncation tolerance, union order/dedupe, REGISTRY-fallback recovery after an uncommitted `shutil.move`, set_goal/create ledger-op arms, all five in-flight probe arms (index.lock, live-in-repo pid, dead pid, live out-of-repo pid, stale patch, unparseable fresh patch), ledger-as-directory + injected git-failure leg errors (unknown, never absent), stale-ledger forensic guidance, CLI verdict-line echo + `--json` keys, and the MF-5 read-only comparator (asserted in `finally` on every call: HEAD SHA, revcount, porcelain, registry/ledger/events bytes unchanged; self-test proves it catches an untracked write AND a ledger append). Registered WORKFLOW_INVARIANT: `scripts/select_step9c_tests.py` tuple entry + `tests/step9c_workflow_invariant_manifest.txt` sorted line.
- `tests/test_repo_root_uncommitted_state_pins.py` (+117, 7 tests): region-scoped prose pins for P1/P2/P3/P4/P5/P6 incl. the MF-4 floor-line mutant (old sentence + " marker-status" appended is REJECTED — the stale view-reprobe instruction is detected) and the ordered CLAUDE.md stderr-sentence pin.
- Corridor-max cap raises (`((measured + 2_800) // 100) * 100`, headroom in [2,701, 2,800]): `code-reviewer.md` 109,764 B -> `112_500` (headroom 2,736; `.claude/config/agent_spec_size_caps.txt` + `tests/test_workflow_lint_agent_spec_caps.py::_MIGRATION_SNAPSHOT` in lockstep); `09-step-5.md` 121,037 B -> `123_800` (headroom 2,763; `scripts/workflow_lint.py` SKILL_DOC_SIZE_GRANDFATHER entry with accurate Prior chronicle — prior basis re-read from the file: `121_500` from the #2294 Step 10d merged-tree re-measure at 118,770 B).
- Diff: +1,465 / -16 across 15 files:

```
 .claude/agents/code-reviewer.md                |   2 +-
 .claude/config/agent_spec_size_caps.txt        |   2 +-
 .claude/rules/LESSONS.md                       |   2 +-
 .claude/rules/codex-composer-common.md         |  18 +
 .claude/rules/repo-root-uncommitted-state.md   |  52 ++-
 .claude/skills/issue/steps/09-step-5.md        |   2 +
 CLAUDE.md                                      |   2 +-
 scripts/select_step9c_tests.py                 |   5 +
 scripts/task.py                                |  90 ++++
 scripts/workflow_lint.py                       |  21 +-
 src/explore_persona_space/task_workflow.py     | 603 +++++++++++++++++++++++++
 tests/step9c_workflow_invariant_manifest.txt   |   1 +
 tests/test_repo_root_uncommitted_state_pins.py | 117 +++++
 tests/test_task_workflow_marker_status.py      | 562 +++++++++++++++++++++++
 tests/test_workflow_lint_agent_spec_caps.py    |   2 +-
```

- Plan adherence: P0 helpers DONE; CLI subcommand DONE; P1-P6 prose DONE (verbatim plan text where the plan supplied it); T1 DONE (23 tests; plan floor met incl. every MUST-cover row); T2 DONE (7 pins incl. MF-4 mutant + MF-5 comparator self-test); WORKFLOW_INVARIANT registration DONE (two-place); cap raises DONE (both in-corridor); §13 scope fences respected — no writer-side change, no semantics change to `list_events`/`has_event`/`latest_event`/`ensemble_verdicts_present`, no pre-commit/_git_commit/root-commit-guard edits, codex-code-reviewer.md untouched. One post-plan refactor: `_probe_patch_file` + `_classify_deferred`/`_tree_row_matches`/`_tree_excerpt`/`_ledger_excerpt` extracted to module level to clear ruff C901 (bare-check complexity 18/22 > 15); behavior-neutral (T1+T2 re-run green post-refactor).
- Commit hash: `871e60d5b1fe32a09b86ba365eae5735f559416e` (verbatim `git log -1 --format=%H`)
- Branch: `issue-2328` pushed to origin (push rc=0, new branch; no PR opened — Step 10d owns landing).

### (b) Considered but not done

- A writer-side fix (making `post-marker` retry/flush the deferred commit) — explicitly out of plan scope (§13 MUST-ASK fence); the read-side classification is the approved scope.
- Applying tree filters to ledger rows — rejected per plan N7 (ledger `message` truncates notes at ~60 chars and carries no version; filtering would create false `absent`s); excerpts carry `version_blind` + `filters_not_applied_to_ledger` flags instead.
- A global-mtime-only stash-window probe — rejected per plan MF-A (61 fresh patch files at any moment on this fleet, ~1 new/32 s; global mtime would make `absent` near-unreachable); shipped probe is pid- + repo-scoped (measured absent-reachability below).
- `os.path.realpath(strict=False)` for the pid-cwd probe — rejected in favor of `os.readlink` so a PermissionError surfaces as a probe ERROR (=> `unknown`) instead of a silent non-match (=> false `absent`).
- Running the two `slow_tests_selected` files locally — pre-emptively deferred to Step 9c per the implementer spec (per-file surcharge above the 600 s Bash cap by construction; zero local attempts).

### (c) How to verify

- **Tests run (all green post-refactor):**
  - `tests/test_task_workflow_marker_status.py` — 23 passed (NEW).
  - `tests/test_repo_root_uncommitted_state_pins.py` — 11 passed (7 new pins + 4 pre-existing).
  - `tests/test_workflow_lint_agent_spec_caps.py` + `tests/test_select_step9c_tests.py` + `tests/test_task_workflow.py` + `tests/test_task_workflow_post_marker_echo.py` + `tests/test_issue_skill_step2_floor.py` + `tests/test_workflow_lint_verdict_round_anchor.py` — 198 + 405 passed across the two pre-commit batches (see `## Smoke run`).
  - Repo-wide invariants (implementer item 1b) + composer-memory-commit: 32 passed in 162 s (`tests/test_no_direct_task_path_construction.py`, `tests/test_no_pod_side_task_py_shellout.py`, `tests/test_no_dollar_budget_caps.py`, `tests/test_workflow_lint_codex_composer_memory_commit.py`).
  - Gate-matched local union: @@UNION_RESULT@@
- **Test-shape note:** the feature ships 1 happy-path per verdict class plus >=2 distinct error/edge tests per source leg (leg-error injection, probe-error arms, stale-ledger, registry fallback) — the >=1 happy + >=2 edge floor is exceeded on every non-trivial behavior.
- **Regression test for a substantive BLOCKER fix:** skipped — round 1 of a planned feature; no prior-round BLOCKER concern exists. (The permanent invariants this round ADDS are themselves test-pinned: the escalation-guard prose by `tests/test_repo_root_uncommitted_state_pins.py`, the verdict lattice by `tests/test_task_workflow_marker_status.py` — e.g. `test_ledger_read_failure_is_unknown_never_absent` trips the leg-error guard with a directory at the ledger path and asserts verdict `unknown` + rc 5, and `test_inflight_index_lock_blocks_absent` trips the in-flight gate with a bare `index.lock` touch.)
- **Falsification evidence (pre-change FAIL per NEW test; recorded before the implementation landed):**

```
test_present_committed_and_cli_rc0: AttributeError: no attribute 'marker_status'
test_present_uncommitted_when_commit_deferred: AttributeError: no attribute 'marker_status'
test_pending_deferred_after_stash_window_revert: AttributeError: no attribute 'marker_status'
test_absent_rc4_on_complete_clean_read: AttributeError: no attribute 'marker_status'
test_tree_filters_version_and_note: AttributeError: no attribute 'marker_status'
test_ledger_match_is_version_blind: AttributeError: no attribute 'marker_status'
test_ledger_match_ignores_note_filter_beyond_truncation: AttributeError: no attribute 'marker_status'
test_list_events_head_union_order_and_dedupe: AttributeError: no attribute 'list_events_head_union'
test_head_registry_fallback_recovers_marker_after_uncommitted_move: AttributeError: no attribute 'marker_status'
test_set_goal_deferral_matches_goal_updated_kind: AttributeError: no attribute 'marker_status' (re-recorded after the by=planner test fix)
test_create_deferral_ancestor_arm: AttributeError: no attribute 'marker_status'
test_inflight_index_lock_blocks_absent: AttributeError: no attribute 'marker_status'
test_inflight_live_pid_in_repo_stash_window: AttributeError: no attribute 'marker_status'
test_inflight_dead_pid_fresh_patch_is_absent: AttributeError: no attribute 'marker_status'
test_inflight_live_pid_out_of_repo_is_absent: AttributeError: no attribute 'marker_status'
test_inflight_stale_patch_is_absent: AttributeError: no attribute 'marker_status'
test_inflight_unparseable_fresh_patch_is_unknown: AttributeError: no attribute 'marker_status'
test_ledger_read_failure_is_unknown_never_absent: AttributeError: no attribute 'marker_status'
test_git_failure_direct_leg_is_unknown: AttributeError: no attribute '_marker_status_git'
test_git_failure_registry_leg_is_unknown_never_absent: AttributeError: no attribute '_marker_status_git'
test_stale_ledger_row_is_unknown_with_forensic_guidance: AttributeError: no attribute 'marker_status'
test_cli_verdict_line_echoes_query: assert 2 == 0 (argparse rejects the unknown subcommand pre-change)
test_rule_file_carries_marker_presence_section: StopIteration (H2 absent pre-change)
test_code_reviewer_floor_check_names_marker_status: AssertionError: ['missing marker-status', ..., 'missing re-append ban'] != []
test_codex_composer_common_carries_marker_status_bullet: StopIteration (H2 absent pre-change)
test_step5_durable_verdict_rule_names_marker_status: AssertionError: 'marker-status' not in the Step-5 durable-verdict region
test_lessons_row_carries_marker_read_trigger: AssertionError: 'marker-status' not in the LESSONS row
test_claude_md_post_marker_stderr_visible: AssertionError: 'stderr VISIBLE' not in the post-marker paragraph
```

  - `test_floor_line_pins_marker_status_and_bans_stale_reprobe` + `test_mf4_mutant_floor_line_with_appended_token_rejected` and the two MF-5 comparator self-tests PASS pre-change BY DESIGN — they validate test machinery (the mutant validator and the read-only comparator), not the feature; the mutant test proves the pin rejects the old-sentence+token evasion, and the comparator self-test proves `_call_ro` catches an untracked write and a ledger append.
- **Gate-scope check (#1288):** selector `n_tests=289` (base=`origin/main`, fetched; default `--json` invocation from the committed worktree, stderr off stdout); ran locally: the 261-file union below (260 diff-linked non-slow selections + pin-sweep hits [all 151 inside the diff-linked set] + the diff-edited test files + the 3 repo-wide invariants); pin-sweep: `--map-files /tmp/issue2328_changed_files.txt --repo-root "$WT"` (15 changed paths, HEAD~1..HEAD) -> 212 pairs -> **151 hit files (list below)**; sweep_scope: selector-universe; deferred invariant-only: **27** files (Step 9c runs them).
- pin-sweep: deleted/moved-literal grep (implementer item 1a) over the 289 enumerated tests for the OLD floor sentence (`re-probe \`task.py view <N> --json\``), old caps (`109_600`, `121_500`), new symbols (`marker_status`, `list_events_head_union`, `STASH_WINDOW_FRESH_SECS`, `marker-status`), `step2-floor-skipped`, and `sweeps the deferred line` -> hits only in `tests/test_issue_skill_step2_floor.py`, `tests/test_repo_root_uncommitted_state_pins.py`, `tests/test_task_workflow_marker_status.py` (all ran locally); sweep_scope: repo-wide (grep-only supplement) -> **0 additional test files** (the sole out-of-universe hit is the data file `tests/step9c_workflow_invariant_manifest.txt`, exercised by `tests/test_select_step9c_tests.py`, ran locally).
- `slow_tests_selected:` `tests/test_workflow_lint.py` AND `tests/test_workflow_lint_phase_done_check.py` — both pre-emptively NOT-RUN locally, deferred to Step 9c at minute zero per the implementer spec; selector `recommended_timeout_s=12390`. Copy-pasteable: `cd "$WT" && PYTHONPATH="$WT/src" /home/thomasjiralerspong/explore-persona-space/.venv/bin/python -m pytest tests/test_workflow_lint.py tests/test_workflow_lint_phase_done_check.py -q` (background invocation, selector-sized bound). NOTE: `tests/test_workflow_lint_phase_done_check.py` is ALSO a pin-sweep hit — its deferral is the slow-list pre-emptive route, and as supplementary evidence the LIVE no-flags `workflow_lint.py` run on this tree is green (see `## Smoke run`).
- Pin-sweep hit files, verbatim dedup union of the `--map-files` col-1 (151 files, list below):

```
tests/test_argcheck.py
tests/test_async_gate_rung0.py
tests/test_auth_outage_guard.py
tests/test_autonomous_session_watch_daemon_liveness.py
tests/test_autonomous_session_watch_diagnosis_window.py
tests/test_autonomous_session_watch_keep_running_owner.py
tests/test_autonomous_session_watch_orphan_pod.py
tests/test_autonomous_session_watch_owner_fence.py
tests/test_autonomous_session_watch_unlaunched_orphan.py
tests/test_autonomous_session_watch_urgent_park.py
tests/test_autonomous_session_watch_wedge.py
tests/test_backend_excerpt_digest.py
tests/test_backend_poll.py
tests/test_backend_selector.py
tests/test_campaign_state.py
tests/test_circuit_breaker.py
tests/test_clean_experiment_downloads_active_consumer.py
tests/test_clean_experiment_downloads_off_main.py
tests/test_clean_experiment_downloads_parity.py
tests/test_clean_experiment_downloads_symlinks.py
tests/test_codex_daemon_reaper.py
tests/test_codex_task_output_preservation.py
tests/test_codex_task_post_marker.py
tests/test_codex_task_post_spawn_probe_retry.py
tests/test_codex_task_prompt_delivery.py
tests/test_codex_task_quota_sentinel.py
tests/test_codex_task_reattach_and_fetch_retry.py
tests/test_codex_task_retry_and_stall.py
tests/test_consolidate_lessons.py
tests/test_daily_drive_filings.py
tests/test_dispatch_issue_cli.py
tests/test_dispatch_lease.py
tests/test_ensemble_review_cap.py
tests/test_ensemble_verdicts_present.py
tests/test_env_loading_from_worktree.py
tests/test_failure_lesson_supersedes.py
tests/test_file_infra_task.py
tests/test_gate_recipe_no_heredoc_argv.py
tests/test_gcp_backend.py
tests/test_guard_harmful_bank_read.py
tests/test_guard_lessons_edit.py
tests/test_guard_log_dump.py
tests/test_guard_piped_git_push.py
tests/test_guard_repo_root_branch.py
tests/test_guard_repo_root_pull.py
tests/test_guard_root_code_commit.py
tests/test_guard_tmp_tmux_sweep.py
tests/test_happy_patch_check.py
tests/test_inline_lint_gate.py
tests/test_inline_payload_lint_gate_contract.py
tests/test_issue1482_densesae_fullwidth.py
tests/test_issue1482_early_layer.py
tests/test_issue1773_pipeline.py
tests/test_issue1774_round_a.py
tests/test_issue2153_detached_hf_transfer_contract.py
tests/test_issue_skill_binary_figures_recovery_pin.py
tests/test_issue_skill_conflict_subagent_dispatch_pin.py
tests/test_issue_skill_file_only_verdict_post.py
tests/test_issue_skill_followup_defer_repark.py
tests/test_issue_skill_gate_leg_timeout_floor.py
tests/test_issue_skill_gate_tree_pathspec.py
tests/test_issue_skill_guard_excerpt_brief.py
tests/test_issue_skill_html_escape_recipe.py
tests/test_issue_skill_lint_family_sync.py
tests/test_issue_skill_lint_gate_mergefile.py
tests/test_issue_skill_long_phase_heartbeat.py
tests/test_issue_skill_neutral_gate_vocab_brief.py
tests/test_issue_skill_phase_rate_duty_pin.py
tests/test_issue_skill_planned_cell_reconcile_pin.py
tests/test_issue_skill_setsid_reparent_prose_pin.py
tests/test_issue_skill_step10d_merge_form.py
tests/test_issue_skill_step10d_wt_binding.py
tests/test_issue_skill_step9c_compare_background.py
tests/test_issue_skill_urgent_park_duty_pin.py
tests/test_issue_skill_workload_cmd_script_pin.py
tests/test_living_docs.py
tests/test_marker_child_stderr_forwarding.py
tests/test_no_ungated_upload_call_sites.py
tests/test_pm_queue_report.py
tests/test_pod_lifecycle.py
tests/test_pod_wait_for_capacity.py
tests/test_poll_next_interval.py
tests/test_poll_pipeline_digest.py
tests/test_poll_pipeline_sentinels.py
tests/test_pre_dispatch_triage.py
tests/test_predispatch_staleness_pass.py
tests/test_pv_phase1_done_gate_handler.py
tests/test_router.py
tests/test_ruff_policy.py
tests/test_runpod_wedge_detection.py
tests/test_runpod_workload_exec.py
tests/test_select_step9c_tests.py
tests/test_session_progress_report.py
tests/test_settings_model_guard.py
tests/test_shared_vm_thread_caps.py
tests/test_slurm_backend_render.py
tests/test_slurm_excerpt_digest.py
tests/test_slurm_monitor.py
tests/test_spawn_session_auth_outage_gate.py
tests/test_spawn_session_env_forwarding.py
tests/test_spawn_session_list_enrichment.py
tests/test_spawn_session_repo_root.py
tests/test_spawn_session_stop_marker.py
tests/test_stage_dispatch_dedup.py
tests/test_stalled_detector_and_gc.py
tests/test_step9c_base_identity.py
tests/test_step9c_baseline.py
tests/test_sweep_parked_wf_candidates.py
tests/test_task_cli_set_body_assertions.py
tests/test_task_cli_set_status_hint.py
tests/test_task_progress.py
tests/test_task_py_cli_sigpipe.py
tests/test_task_workflow_alloc_drift.py
tests/test_task_workflow_completion_report.py
tests/test_tick_triage.py
tests/test_upload_verifier_currency.py
tests/test_verdict_disagree_observer.py
tests/test_verify_carryover_inputs.py
tests/test_verify_plan_c58_fanout_pod_name.py
tests/test_verify_report.py
tests/test_workflow_followup_labels.py
tests/test_workflow_lint_agent_memory_index_size.py
tests/test_workflow_lint_agent_tools.py
tests/test_workflow_lint_asw_docstring_pass_count.py
tests/test_workflow_lint_conflict_markers.py
tests/test_workflow_lint_empty_text_default.py
tests/test_workflow_lint_fence_wt_binding.py
tests/test_workflow_lint_files_mode.py
tests/test_workflow_lint_gcp_pin_guidance.py
tests/test_workflow_lint_gotchas_size.py
tests/test_workflow_lint_jsonl_splitlines.py
tests/test_workflow_lint_judge_model_check.py
tests/test_workflow_lint_lane_order_adjective.py
tests/test_workflow_lint_no_repo_root_git_reset_hard.py
tests/test_workflow_lint_no_repo_root_worktree_revert.py
tests/test_workflow_lint_null_gate_calibration.py
tests/test_workflow_lint_phase_done_check.py
tests/test_workflow_lint_plan_version_immutability.py
tests/test_workflow_lint_scripts_import_guard.py
tests/test_workflow_lint_sha_pin_domain.py
tests/test_workflow_lint_skill_doc_size.py
tests/test_workflow_lint_slurm_gpu_width.py
tests/test_workflow_lint_stale_gotchas_pointers.py
tests/test_workflow_lint_upload_or_true.py
tests/test_workflow_lint_v2_checks.py
tests/test_workflow_lint_verdict_round_anchor.py
tests/test_workflow_lint_walks.py
tests/test_workflow_setsid_detach_convention.py
tests/test_workflow_v2_flag.py
tests/test_workload_cmd_env_lint.py
tests/test_worktree_audit.py
```

- **Lint:** `ruff check` + `ruff format --check` on all 7 touched .py files — PASS (rc=0, rc=0; post-refactor). Broad-tree note: the repo carries ~1.3k pre-existing broad-ruff errors (known baseline); touched-file scope proves 0 introduced.
- **Ruff-policy pin (#1699/#1716):** `uv run pytest tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset -x` -> 1 passed, `rc=0` (the diff touches LIVE_WORKFLOW_HELPERS members `scripts/task.py`, `scripts/workflow_lint.py`, `scripts/select_step9c_tests.py`).
- **No-flags workflow_lint (inline payload gate leg):** @@LINT_RESULT@@
- **Reproduction commands:**

```
WT=/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-2328
VENV=/home/thomasjiralerspong/explore-persona-space/.venv/bin/python
cd "$WT"
PYTHONPATH="$WT/src" "$VENV" -m pytest tests/test_task_workflow_marker_status.py tests/test_repo_root_uncommitted_state_pins.py -q
PYTHONPATH="$WT/src" "$VENV" scripts/task.py marker-status 2325 epm:code-review-codex
PYTHONPATH="$WT/src" "$VENV" scripts/task.py marker-status 2325 epm:code-review-codex --json
```

- **What success looks like:** the live check prints a query-echoing verdict line (`verdict: present-committed — task #2325 kind=epm:code-review-codex version=any note-contains=none read-at=<ts>`) and exits 0; the 34-test T1+T2 suite passes; and a synthetic absent query (nonexistent kind on a clean repo with no live windows) exits 4 with verdict `absent`.

### (d) Needs human eyeball

- **MF-A reachability (informational per brief):** round-start census on this VM — 25,855 accumulated `~/.cache/pre-commit/patch*` files (0 unparseable), 61 fresh at the 1,800 s horizon, 0 with a live in-repo pid; measured absent-reachability 25/30 = **83.3%** over 30 spaced synthetic absent probes (5 fired `unknown`, predominantly `commit-in-flight` via a transient `index.lock` under fleet commit traffic). Within the plan's stated expectation (fleet-busy `unknown`s are transient and re-read-resolvable); the brief marked the reachability figure informational — noting, not bouncing. Artifact: `/tmp/issue2328_roundstart_probe.json` (ephemeral; census numbers reproduced here in full).
- **Verdict-assembly precedence** is deliberate and worth an eyeball: leg errors beat matches (`unknown` even when a match exists) so a partially-failed read can never assert presence-class confidence; and `pending-deferred` outranks the in-flight probe (a matching live ledger row short-circuits — the probe only gates `absent`).
- **`_ledger_op_kinds` coverage:** unmapped `task_workflow` ops are KIND-BLIND by design (any kind matches a task-id+path-eligible row for an unmapped op) — conservative toward `pending-deferred`/away from false `absent`, but a reviewer should confirm the op table rows (create/set_goal/promote/append_concern_event) match current `post_event` message shapes.
- Assumption under minor plan ambiguity: the plan's `marker_status` docstring sketch showed filters applying to tree rows only — implemented exactly that (N7), with the flags on ledger excerpts making the asymmetry visible to callers.

## Smoke run

### phase: T1 functional suite (falsify-then-pass)

- Command: `cd "$WT" && PYTHONPATH="$WT/src" /home/thomasjiralerspong/explore-persona-space/.venv/bin/python -m pytest tests/test_task_workflow_marker_status.py -q`
- Slice: all 23 tests (full file — no sampling).
- Exit code: pre-change rc=1 (28 recorded FAIL lines, table in (c)); post-change rc=0 (23 passed); post-C901-refactor re-run rc=0 (34 passed combined with T2, 4.86 s).
- Artifact digest: `/tmp/i2328_falsification_lines.txt` (28 lines, inlined verbatim in (c)).

### phase: T2 prose-pin suite + MF-4 mutant + MF-5 comparator self-test

- Command: `cd "$WT" && PYTHONPATH="$WT/src" $VENV -m pytest tests/test_repo_root_uncommitted_state_pins.py -q`
- Slice: 11 tests (7 new + 4 pre-existing in the file).
- Exit code: rc=0 (11 passed). MF-4 mutant validator rejects the old-sentence+" marker-status" evasion; MF-5 comparator self-test raises on an untracked write and on a ledger append (both asserted).
- Artifact digest: pin tokens enumerated in the test file; region scoping via `_h2_region` (H2 -> next `## `).

### phase: live #2325 CLI check (plan §6 criterion 1)

- Command: `PYTHONPATH="$WT/src" $VENV scripts/task.py marker-status 2325 epm:code-review-codex [--json]`
- Slice: 1 live task (the incident task), real events + real deferral ledger.
- Exit code: rc=0; verdict `present-committed`; post-refactor read: head rows=49, head matches=3, ledger matches=0 (the incident-era deferral rows now PREdate the newest commit and the head match wins first, as designed). An earlier pre-refactor read against the same task also matched the actual #2325 incident ledger row (ts=2026-08-16T17:14:14Z) version-blind.
- Artifact digest: `/tmp/i2328_live2325_post.json`; verdict line quoted in (c).

### phase: MF-A in-flight probe census (plan §6 criterion 2)

- Command: round-start probe script over `~/.cache/pre-commit/` + 30 spaced synthetic absent probes (recorded to `/tmp/issue2328_roundstart_probe.json`).
- Slice: full patch-file census (25,855 files) + 30 probes.
- Exit code: rc=0. Census: 0 unparseable filenames; fresh counts 15/39/61/109 at 300/900/1800/3600 s; 0 live in-repo pids at probe time. Absent-reachability 25/30 (83.3%); 5 `unknown`s, predominantly `commit-in-flight`.
- Artifact digest: JSON fields reproduced in (d).

### phase: lint legs + repo-wide invariants

- Commands + exit codes: `ruff check` rc=0 and `ruff format --check` rc=0 on the 7 touched .py files (post-refactor); ruff-policy pin rc=0 (1 passed, 0.29 s); invariant trio + composer-memory-commit batch rc=0 (32 passed, 162 s); no-flags `workflow_lint.py` (whole tree, pre-refactor run) exit 0 — 0 FAIL, 33 WARN (all pre-existing grandfather/band WARNs; LESSONS.md inside its warn band at 10,407/10,492 as planned, no cap raise needed); post-refactor re-run: @@LINT_RESULT@@
- Slice: full tree for lint; full files for tests.

### phase: gate-matched local union (implementer item 1)

- Command: `cd "$WT" && timeout 3000s PYTHONPATH="$WT/src" $VENV -m pytest $(tr '\n' ' ' < /tmp/i2328_union.txt) -q` (261 files).
- Slice: 260 diff-linked selector picks + all 151 pin-sweep hits + diff-edited tests + 3 repo-wide invariants; 27 invariant-only files deferred to Step 9c; 2 slow-listed files pre-emptively deferred (see (c)).
- Exit code + result: @@UNION_RESULT@@
