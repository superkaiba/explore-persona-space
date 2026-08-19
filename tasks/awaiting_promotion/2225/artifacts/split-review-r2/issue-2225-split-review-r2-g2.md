# Split-review r2 — group g2 (commit 26aefadc1d) — issue #2225

**Verdict:** PASS
**Blocker tags:** none
**Tier:** leaf (issue-scoped pod dispatcher `scripts/issue2225_dispatch.sh` + its new test file; reviewed at trunk depth as a fix-verification round)
**Scope reviewed:** `git show 26aefadc1d` only (17 KB; +270/−9 across 2 files), with bounded HEAD reads of the sibling r2 commits' seams (`bf62121c3a` train.py skip-evidence write, `2c48d9e026` mmlu.py `--limit`) for cross-commit wiring.

## Blocker-closure verification

### Blocker 2 — sentinel drain-glob (g4 Critical 1): CLOSED
- `scripts/issue2225_dispatch.sh:76` — `SENTINEL_ROOT="${EPM_I2225_SENTINEL_ROOT:-/workspace/logs}"`; `:106` — `write_sentinel` hands `SENT_LOGS_DIR="$SENTINEL_ROOT"`. All 5 sentinel emissions route through this ONE helper (call sites `:336, :338, :480, :604, :606` — includes unit 5's re-pilot `epm:progress` sentinel); no other pod-side writer emits sentinels (grepped).
- Cross-checked against the ACTUAL poller: realized filename `issue-2225-<kind_slug>-<epoch>.json` (`scripts/issue778_lib.py:400`, `logs_dir` IS the parent — no subdir) fnmatches the drain glob `/workspace/logs/issue-2225-*.json` (`poll_pipeline.py:2321-2333`, path-terminal) AND the `epm_results` presence probe `issue-2225-epm_results-*.json*` (`:2044`).
- Phase logs deliberately stay in the `$LOG_ROOT` subdir — nothing else in the round reads them except the dispatcher's own gates (which use `$LOG_ROOT` consistently); the r1 fix text explicitly sanctioned this split ("only the SENTINEL parent is contract-bound").
- Tests pin: the default (`test_sentinel_root_default_is_toplevel_workspace_logs`), the no-`$LOG_ROOT`-handoff invariant, and the filename-vs-drain-glob fnmatch + required envelope keys against the REAL `write_results_sentinel`.

### Blocker 4 — count-gate resume-starve (g5 Critical + g4 Major 2): CLOSED
- Both gate sites are dual-token: `:252` (phase_p0) and `:418` (p0_run_repilot) — `grep -rlF -e "[steer-hook]" -e "[fanout-skip]"`; no single-token count gate survives anywhere in the round (repo-grepped; the token is emitted only by `steer_train.py` and the fan-out skip branch).
- The evidence line IS written where the gate counts it: `scripts/issue2225_train.py:912-916` (landed in sibling r2 commit `bf62121c3a`) appends `[fanout-skip] <slug> ...` to `log_dir/<slug>.log` — `log_dir` is the same `--log-dir` the dispatcher passes (`$p0_logs`=`$LOG_ROOT/p0_train`, `$rp_logs`=`$LOG_ROOT/p0_train_repilot`), a per-cell FILE, not stdout. Fresh launches open the same path mode `"w"` (`:961-962`, truncates stale skip lines on re-run); skip appends mode `"a"` (a both-token file counts once — pinned functionally).
- **Scenario (a) — octave overlap cell:** `synth_cell("A","evil",1.5)` produces the CANONICAL slug identical to the trained registry pilot cell → same manifest path → fingerprint match → deterministic resume-skip → `[fanout-skip]` file → counted. The committed test drives the REAL sed-extracted `p0_run_repilot` body (real gate grep, real state/note heredocs run under `command uv`) with 3 fresh + 1 skip logs → `4/4`, reaches eval-gen, resolves state.
- **Fails-pre-fix empirically confirmed:** I re-ran the identical probe harness against the PARENT commit's function body (`git show bf62121c3a:scripts/issue2225_dispatch.sh`, single-token gate) → `rc=7`, `FATAL: §7 criterion (i) FAILED on the re-pilot — 3/4 logs carry [steer-hook]`, eval-gen never reached. The fix is what closes it, not the harness.
- **Scenario (b) — fresh-pod re-entry:** `_read_manifest` (`train.py:376-385`) is local-only, so both branches hold: manifests/ckpts restored → ALL cells skip → N `[fanout-skip]` files → N/N; truly fresh pod → `stored is None` → cells retrain → fresh `[steer-hook]` logs → N/N. Neither branch FATALs.

### g3 Major 1 — P0 MMLU `--limit` probe leg: CLOSED
- Dial exists and is fully threaded (sibling commit `2c48d9e026`): `--limit` in the lm-eval argv (`issue2225_mmlu.py:257-258`), in the resume fingerprint (`limit` key, `:94/:109` — a `--limit 200` run can never resume-satisfy P2c's full run), and propagated into `--single` subprocess argv (`:408-409`).
- Dispatcher wires the leg inside `phase_p0` (post hook-gate, pre eval-gen; `MMLU_P0_LIMIT=200`, `8` under smoke): `uv run python scripts/issue2225_mmlu.py --targets "base,<first pilot cell>" --limit "$MMLU_P0_LIMIT" --out-root "$PILOT_OUT"`. Verified at runtime: `pilot_cells()[0].slug == "A__evil__c0.5"` and ALL 8 pilot slugs resolve in `evalgen.targets_by_tag()` (pilot cells are a subset of `build_cell_registry()`), so `run_fan_out`'s strict unknown-tag ValueError does not fire; the adapter resolves at `ckpt_root/<slug>` (`resolve_adapter` kind="cell" == `train.py:420` out_dir), populated by pilot training before the probe runs. `PILOT_OUT` is additionally disjoint from P2c's out-root — double protection beyond the fingerprint.
- Test pins the 200 literal, the argv threading, and in-`phase_p0` placement.

## Fresh-bug sweep (fix-introduced)
- `bash -n` PASSES; `set -euo pipefail` intact; the nested-quote command substitution building `mmlu_probe_targets` is correctly quoted and fails loud under `set -e`.
- No token collision: the fan-out PARENT prints `[fanout] skip ...` (different token) to parent stdout, which never lands in the counted per-cell log dirs; only per-cell files exist there.
- `grep -F` with two `-e` patterns = fixed-string OR, `-l` once-per-file — functional test `test_dual_token_grep_counts_each_log_once` pins exactly this (3 of 4 files).
- Tests: 7/7 PASS (`uv run pytest tests/test_issue2225_dispatch.py`, 2.5 s); ruff check + format clean on the new test file.

## Concerns (non-blocking)
1. **Env-override pairing** (`dispatch.sh:70/:76`): overriding `EPM_I2225_LOG_ROOT` alone no longer relocates sentinels (that decoupling IS the fix), so an off-pod invocation must also set `EPM_I2225_SENTINEL_ROOT` or `mkdir -p /workspace/logs` fails loud at `:77`. Fail-loud and pod-side-only — suggest one line in the header comment naming the pairing.
2. **Pre-existing residual, r1-sanctioned:** per-cell logs under `$LOG_ROOT/p0_train*/` remain invisible to the poller's top-level staleness glob `/workspace/logs/issue-<N>-*.log` (`poll_pipeline.py:1840`) and match neither #488 shard layout (`/workspace/explore-persona-space/logs/issue_<N>...`). Mitigated by the GPU-utilization arm + the dispatcher's per-cell `[fanout]` lines on the top-level log; g4 r1 classed this Secondary and its fix text sanctioned subdir phase logs. Not this commit's regression.
3. **For the orchestrator/g5 lane:** the new comment at `dispatch.sh:357-362` claims the smoke-unreachable re-pilot path is "disclosed in the smoke-architecture marker" — the marker/plan enumeration edit itself is outside this commit's diff (this is a `CONTRACT-BEARING: no` group; Step 0.55 skipped per brief). Verify the enumeration line actually exists task-side when composing the round verdict.

## Suggestions
- `test_p0_mmlu_probe_leg_wired`'s "inside phase_p0" span (`split("phase_p0()")[1].split("phase_p2a()")[0]`) also contains `p0_run_repilot`, so the placement pin is looser than its message states; anchoring on the `phase_p0` function body would tighten it. Cosmetic — the combined asserts hold.

**Recommendation:** merge-eligible for this group's scope. All three assigned r1 blockers are genuinely closed with real-body, fails-pre-fix-verified tests.
