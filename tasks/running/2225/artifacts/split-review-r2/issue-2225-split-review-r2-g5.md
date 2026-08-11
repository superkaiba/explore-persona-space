# Split-review r2 g5 — issue 2225 (contract-bearing group: round-level gates + commit fabf9d2f19)

**Verdict:** PASS
**Blocker tags:** none
**Tier:** trunk (round spans `.claude/` + `src/` + shared scripts; g5 commit itself is agent-memory-only)

## Commit-scope check (g5: fabf9d2f19)

PASS — `git show fabf9d2f19 --stat`: exactly 5 files, all under `.claude/agent-memory/code-reviewer-lean/` (MEMORY.md + 4 new memory files). Nothing load-bearing (no scripts/src/tests/config/workflow-logic files).

- **Concern (fixed in-round, disclosed):** the commit's MEMORY.md change was a wholesale REWRITE, not an append — it dropped 11 index rows whose target memory files still exist at HEAD (3 pre-existing rows + 8 rows added by the 50b0e87dc7 spec-freshness sync), orphaning them from the always-loaded index; a later sibling commit (8a584ffa1c, g1 memory, outside this range) left a 12th file unindexed. Root cause is a stale-session-snapshot overwrite (the r1 g5 reviewer's session predated the sync). As the owner of this memory dir I restored the full 16-row index and recorded the lesson: commit `1e5308d554` (`.claude/agent-memory/code-reviewer-lean/MEMORY.md` + `index_overwrite_orphans_sibling_memories.md`), pushed to `issue-2225`. **This commit post-dates the reviewed range and touches only my agent-memory dir** — flagged here so the orchestrator's range accounting is not surprised.

## Round-level gates

### Step 0.5 — implementation marker shape: PASS

Highest-version `epm:experiment-implementation` is v2 (ts 2026-08-11T06:42:26Z, head sentinel `<!-- epm:experiment-implementation v2 -->`). All four H3 sections `(a)`–`(d)` present, in order, non-empty. `(c)` carries a copy-pasteable fenced pytest command with observable success signal ("92 passed in 61s"), ruff + ruff-policy-pin lines with PASS results, per-blocker fails-pre-fix test mapping, pin-sweep + import-check evidence, and a "What success looks like" block. Optional `## Smoke run` + `Response to code-review v1` sections additionally present.

- Minor (Style/Consistency): `(c)` names the blocker-1 test `test_stale_adapter_after_crashed_retrain_refuses_skip`; the actual name is `test_should_skip_refuses_stale_adapter_after_crashed_retrain` (tests/test_issue2225_cell_registry.py:287). The file-level battery command in `(c)` runs it regardless; doc slip only.

### Step 0.55 — smoke-architecture marker: PASS

- Checker: `task.py check-smoke-arch-registry 2225 --repo-root <worktree>` → rc=0, `OK — registry-complete (marker-only — driver not verified: registry symbol not statically extractable: sorted(_SPEC_BY_CONFIG)): n=10`.
- Marker-only tier ⇒ set-equality read is the reviewer's arm: `members=A,B,C,D,E,F,G,I,P,H` == the keys of `_SPEC_BY_CONFIG` built from `CONFIGS` (scripts/issue2225_train.py:140-162, 225) — exact match, n=10.
- Counts: marker's A 16 / B 12 / C 16 / D 12 / E 12 / F 3 / G 3 / I 3 / P 3 / H 1 sum to 81 and match the code arithmetic (GRID_L1=4 × DATASETS=4; GRID_MULTILAYER=3 × 4; GRID_ATTRIBUTION=3 × 1; H=1; train.py:106-118), and `EXPECTED_CELL_COUNT = 81` is asserted in `build_cell_registry()`.
- v2 marker substance the r1 g6 verdict validated is intact: 10 per-arm rows, all REAL, verdict `PASS_UNIFIED` with zero FALLBACK per-arm rows (verdict↔row consistent); `arm-registry:` line in accepted structured form; `import-resolution:` line in the `--import-check` accepted shape with per-script rc=0 re-run this round; §4.8 attribution CORRECTED (demotion attributed to the implementer-mirrored blind-spot enumeration, not plan §4.8); the g5 re-pilot blind-spot line present as enumeration item (b) with the bash-probe mitigation named. `resume-matrix:` (6 REAL rows incl. the reworked manifest predicate and the new sync-reissue re-entry) and `production-outroot-unit:` (FALLBACK with reason + named pod-side proxy — the §7 P0 pilot gate; pre-dispatch build phase) sub-blocks present.

### Step 0.6 — end-to-end smoke gate (round-2 fix coverage): PASS

All four claimed NEW fails-pre-fix probes exist, exercise the real bodies, and pass when spot-run (9/9 in 3.69s: full `tests/test_issue2225_dispatch.py` + the two named tests):

1. **Stale-adapter scenario** — `test_should_skip_refuses_stale_adapter_after_crashed_retrain` (cell_registry:287): builds the exact g2 Critical trace (START manifest under current fingerprint + prior-fingerprint adapter bytes, no completed record) against the REAL train module (importlib-loaded scripts/issue2225_train.py) and asserts `should_skip` is False; sha-mismatch + HF-leg + upload-re-drive siblings alongside (287-380).
2. **Sentinel fnmatch pin** — `test_sentinel_filename_matches_drain_glob` (dispatch:73): calls the REAL `issue778_lib.write_results_sentinel`, asserts parent==logs_dir (no subdir), `fnmatch(name, "issue-2225-*.json")`, and poller-required keys. Glob constants match `scripts/poll_pipeline.py` (line 355: `/workspace/logs/issue-<N>-<kind_slug>-<epoch_seconds>.json`, top-level). Companion source pins: SENTINEL_ROOT default `/workspace/logs` (dispatch.sh:76) + `SENT_LOGS_DIR="$SENTINEL_ROOT"` (dispatch.sh:106) both test-pinned (dispatch:56-71).
3. **Sync-reissue run-twice idempotency** — `test_sync_reissue_is_resume_idempotent` (judge_analysis:489): run 1 merges cached sync draws + writes `judge_meta.api_refusal_reissue`; run 2 asserts byte-stable draw lists, draw multiplicity still 2, and ZERO further judge calls (call-count spy). Genuine fails-pre-fix shape (pre-fix double-merge documented in the docstring).
4. **Bash probe of the REAL `p0_run_repilot` body** — `test_p0_run_repilot_survives_resume_skipped_overlap_cell` (dispatch:131): regex-extracts the real function from dispatch.sh (anchored on its terminal log line, so the extraction cannot truncate inside heredocs), stubs ONLY the `uv run python scripts/...` invocations, drives the deterministic octave-overlap resume-skip cell (`A__evil__c1.5` writes `[fanout-skip]`), and asserts rc=0, `hook-engagement logs (fresh or resume-skip): 4/4`, eval-gen reached, no FATAL, state resolved. The dual-token gate is confirmed at BOTH sites in source (dispatch.sh:252, 418) with no single-token sibling (`test_hook_count_gates_are_dual_token_at_both_sites` + functional grep-count test).

Smoke output hygiene: `git status --porcelain -- eval_results/ figures/` empty before and after my test run; probes write to pytest tmp_path only. The v2 marker's `## Smoke run` section appropriately rides r1's per-phase sub-sections (no GPU phase touched this round) + the 92-test battery.

### Step 0.8 — prior open binding concerns + r1 blocker disposition: PASS

- `task.py list-concerns 2225 --open-only --json` → `[]` (no open BLOCKER/CONCERN entries to inherit or re-verify).
- All 5 r1 must-fix blockers map to a round-2 commit or a recorded disposition — none silently dropped:
  1. g2 Critical (stale-adapter resume skip) → `bf62121c3a` (sha-bound manifest + adapter wipe + HF-leg uploaded flag) + fails-pre-fix test (ran, passed).
  2. g4 Critical 1 (sentinel drain-glob invisibility) → `26aefadc1d` (SENTINEL_ROOT top-level default) + sentinel pin tests (ran, passed).
  3. g4 Major 3 (sync-reissue double-append) → `2c48d9e026` + run-twice idempotency test (ran, passed).
  4. g5 Critical + g4 Major 2 (count-gate resume starvation) → `26aefadc1d` (dual-token gates at both sites + `[fanout-skip]` skip evidence) + the real-body bash probe (ran, passed).
  5. g6 marker-shape (smoke-arch marker grammar) → v2 marker re-post (ts 06:40:31Z) + checker rc=0 (verified above); recorded in v2 marker §(a).
- Pushed-back / skipped r1 items are all recorded with rationale in v2 marker §(b) (batch-coherence force, cap-hit auto-regen, n_total_draws captions, E2/E3 forward merge → recorded deviation) — depth adjudication belongs to the sibling groups; nothing is unaccounted.

### Step 0.9 — git provenance: PASS

- `git log bc295a5aca..fabf9d2f19` contains EXACTLY the declared 6 commits: `50b0e87dc7` (pre-round sync) then `bf62121c3a`, `26aefadc1d`, `2c48d9e026`, `5551a11dbd`, `fabf9d2f19`.
- `50b0e87dc7` touches only workflow-surface files (27 files: `.claude/agent-memory/**`, `.claude/agents/*`, `.claude/rules/*`, `.claude/skills/adversarial-planner/SKILL.md`, `CLAUDE.md`, `scripts/workflow_lint.py`, `tests/test_workflow_lint_phase_done_check.py`) and is a genuine sync: `git diff 50b0e87dc7 origin/main` over 6 sampled files (incl. workflow_lint.py + its test) is EMPTY. No round commit other than the sync touches any workflow-surface file (file-scoped `git log` over the range).
- Round-wide `git diff --name-status bc295a5aca..fabf9d2f19`: every path falls in the declared scope classes (scripts/issue2225_*, src/.../experiments/issue2225/*, tests/test_issue2225_*, .claude/agent-memory/**, sync workflow-surface files). One NEW file (`tests/test_issue2225_dispatch.py`) — in scope. Nothing modified outside the declared scope.

## Issues Found

### Critical
None.

### Major
None.

### Minor
1. fabf9d2f19 MEMORY.md index clobber (11 rows dropped for still-existing files) — see Commit-scope check; FIXED in follow-up commit `1e5308d554` (agent-memory-only, pushed). No action needed from the implementer.
2. Marker §(c) blocker-1 test-name mismatch (doc slip; actual test exists and passes).

## Unintended Changes
None in the reviewed range beyond the MEMORY.md row drops noted above.

## Tests
Spot-run: `tests/test_issue2225_dispatch.py` (7 tests incl. the real-body bash probe) + stale-adapter + reissue-idempotency → 9 passed in 3.69s. Worktree eval_results/figures untouched.

## Security Check
g5 commit: markdown memory files only — no secrets, no code. Round-level: nothing in this group's scope.

## Recommendation
PASS for the g5 sub-scope and all five round-level contract gates. Round-verdict composition can treat Steps 0.5/0.55/0.6/0.8/0.9 as PASS with the two Minors above.
