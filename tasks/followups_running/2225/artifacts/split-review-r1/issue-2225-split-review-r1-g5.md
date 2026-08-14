# Code Review: issue 2225 — split-review r1, group g5 (commit ecccdf108e "unit 5/5: figures + coef-scale re-pilot + full smoke")

**Verdict:** FAIL
**Blocker tags:** substantive
**Tier:** leaf (per-issue scripts + tests only; no shared workflow surface touched)
**Diff size:** +1565 / -56 across 8 files
**Plan adherence:** COMPLETE on scope (all §6 figures + §7 remedy present) — but the §7 remedy path is functionally broken (Critical below)
**Tests:** PASS (69/69) but INSUFFICIENT — no test covers the re-pilot hook-count gate with a resume-skipped cell (the exact realized case)
**Tests actually run:** yes (`uv run pytest tests/test_issue2225_{figures,cell_registry,judge_analysis,steer_hook}.py` → 69 passed, 103 s)
**Lint:** PASS (`ruff check` clean on all 4 changed .py files)
**Security sweep:** CLEAN (no secrets; uploads via canonical hub helpers; no eval/exec)
**Needs user eyeball:** None

## Plan Adherence
- §7 octave-shift re-pilot (ONE automatic re-pilot; ×0.5 all-broken / ×2 all-ineffective; second miss → proceed with widest bracketing + limitation note; criterion-(iii)-only failure stays designed halt rc=7): ✓ implemented — `scripts/issue2225_dispatch.sh:333-441` (`p0_handle_verdict_fail`, `p0_run_repilot`), `scripts/issue2225_judge.py:847-860` (`repilot` block). BUT see Critical — the path deterministically self-destructs.
- Round-1 concern `octave-shift-repilot-no-coef-scale-cli`: the CLI plumbing is REAL, not scaffolded — `--coef-scale` reaches `_resolve_cells` → `synth_cell` → `Cell.coef` → steering vectors (`scripts/issue2225_train.py:931-951`); `--single-cell` resolves scaled slugs via `resolve_cell` (train.py:1066); eval-gen resolves them via the `resolve_targets` fallback (`scripts/issue2225_eval_gen.py:188-204`, registry-shape-identical `EvalTarget` construction). Guard `--coef-scale/--pilot-coefs/--pilot-configs require --pilot` present (train.py:953-954); mutually-exclusive group correct.
- §6 figures list: ✓ complete — hero single/multi-layer coefficient-response curves, matched-coherence bars + per-question scatter, E1/E2×mask attribution grid, registered-contrast forest, probe-/projection-shift bars, per-layer probe profiles, narrow-domain retention, cap-hit + judge-drop diagnostics (`scripts/issue2225_figures.py` FIGURES registry, all 10 builders).
- Figure conventions: ✓ one color = one meaning (module-wide `CONFIG_COLOR`), plain-English condition labels (no bare config codes rendered), no explanatory annotations/arrows (the per-question index labels are the report-spec's required per-unit labeling, not annotations), errorbar offsets clamped non-negative per the xerr contract (`_ci_offsets`, figures.py:311-313), raw per-unit data alongside aggregates.

## Issues Found

### Critical (diff is wrong or introduces serious risk — block merge)
- `scripts/issue2225_dispatch.sh:376-383` (`p0_run_repilot` hook-count gate): **the §7 re-pilot FATALs deterministically on its first real execution, then enters a permanent crash loop on resume.**
  - Evidence: `n_expect=$(... sum(len(v['cells']) ...))` counts ALL planned re-pilot cells; `n_hook=$(grep -rlF "[steer-hook]" "$rp_logs" | wc -l)` counts only FRESH per-cell logs in the re-pilot's own log dir. But every possible octave grid overlaps the already-trained pilot grid at exactly one coefficient: GRID_L1 = (0.5, 1.5, 3.0, 5.0) (train.py:116); ×0.5 → (0.25, 0.75, **1.5**, 2.5); ×2 → (1.0, **3.0**, 6.0, 10.0). The judge's repilot block emits canonical slugs (`f"{cfg}__evil__c{c}"`, judge.py:855 — matches `_coef_tag`'s `f"c{coef}"` exactly), so the overlap cell (e.g. `A__evil__c1.5`) is byte-identical to the registry cell the ORIGINAL P0 pilot already trained into the same `$CKPT_ROOT`. Its manifest fingerprint (slug/coef/dataset-sha/direction-sha/code-sha, train.py:325-350) matches → `should_skip` → True → the fan-out skips it printing `[fanout] skip <slug>` to the DISPATCHER's stdout and writes NO per-cell log into `$rp_logs` (train.py:781-786; log files are created only at subprocess launch, train.py:825-826). This dedupe is deliberate and even test-pinned (`test_resolve_cells_pilot_coef_scale_halves_grid`: "scaled coef landing on a registry value yields the IDENTICAL Cell → resume dedupe") — but the count gate was never reconciled with it.
  - Impact: first re-pilot run → 3/4 logs → `FATAL: §7 criterion (i) FAILED on the re-pilot` → exit 7 BEFORE `repilot_state.resolved` is ever written. On relaunch, phase_p0 re-derives the FAIL verdict, re-enters `p0_run_repilot`, now ALL 4 cells resume-skip → 0/4 → exit 7 forever. The designed §7 remedy (the round-1 BLOCKER concern this commit exists to fix) can never complete, and the rc=7 masquerades as the DESIGNED criterion-(iii) halt (the sentinel note documents rc=7 as the sign-failure case), misleading triage. The implementer's bash probe stubbed the heavy calls, so this gate never ran against a skipped cell; the path is also smoke-unreachable (see Major), so nothing before production can catch it.
  - Fix (any one): (a) have the fan-out write a skip-sentinel line (e.g. `[steer-hook] skipped-resume <slug>`... better: a distinct `[fanout-skip]` line) into the per-cell log file on the skip branch and count `launched ∪ skipped`; (b) count only LAUNCHED cells — derive `n_expect` by subtracting cells whose manifest+adapter predate the re-pilot (accept prior-round evidence per-cell: for a skipped cell, grep the ORIGINAL `$p0_logs/<slug>.log` for `[steer-hook]` instead); (c) simplest: for each planned cell, accept `[steer-hook]` evidence from EITHER `$rp_logs/<slug>.log` OR `$p0_logs/<slug>.log` (the overlap cell's engagement was already proven in the original pilot).
  - Mechanizable: yes — a bash-probe test that pre-creates the overlap cell's manifest+adapter (or stubs `should_skip` true for one cell) and asserts `p0_run_repilot` reaches eval-gen instead of exiting 7.

### Major (diff needs revision before merge)
- `scripts/issue2225_dispatch.sh:316-320` + plan §4.8: **the entire ~130-line re-pilot remedy path is structurally unreachable by the unified smoke, and the smoke blind-spot enumeration does not name it.** Under `EPM_I2225_SMOKE=1`, a verdict FAIL is "informational under smoke, continuing", so `p0_handle_verdict_fail`/`p0_run_repilot` execute for the first time in production, on a live P0 miss — exactly how the Critical above would have surfaced (Step 0.71 recorded as borderline-N/A: the smoke-conditional's downgrading arm is pre-existing and unedited by this diff; the new unreachable code is its else arm — so no `smoke-blind-spot-unenumerated` tag, but the enumeration gap is real). Fix: add one line to the plan/marker blind-spot enumeration ("the §7 re-pilot remedy path is not exercised by any smoke; first exercised on a live P0 miss") AND extend the sed-extracted bash probe (implementer marker §(c)) to drive `p0_run_repilot` with a stubbed skip so the count gate is exercised.

### Minor (worth fixing but doesn't block)
- `scripts/issue2225_dispatch.sh:233-242` (pre-existing, unit-4 code — NOT introduced by this commit; noted per the Step 3.7 bug-class sibling sweep): the ORIGINAL P0 hook-count gate has the same fresh-log-vs-resume-skip shape on a FRESH-pod resume — cells skip via `_hf_complete` while the new pod's `$p0_logs` is empty → 0/8 FATAL. Same fix family applies; flagging for the round to sweep, not against g5.
- `scripts/issue2225_figures.py` (as of THIS commit): `numpy`/`matplotlib` were imported with no `load_dotenv()` before them, so the #847 shared-VM thread caps did not bind in-process. ALREADY FIXED by follow-up commit bc295a5aca (g6's scope) — no action for this group; recorded so the round verdict can attribute it.
- `scripts/issue2225_judge.py:857` `train_args` field in the repilot block is never consumed (the dispatcher composes its own argv from `coef_scale`) — harmless documentation, but a drift risk if the two ever diverge.
- `scripts/issue2225_figures.py:559`: a unit with `n_total_draws` 0/missing silently renders a 0.0 drop fraction in the judge-diagnostics panel — could mask an empty-draws unit; consider NaN + skip.

## Unaddressed Cases
- Crash between re-pilot training and the resolution write re-derives the DEFAULT-grid verdict on resume (`--phase p0-verdict` without `${gridargs[@]}`, dispatch.sh phase_p0) before re-entering `p0_run_repilot` — correct end state (persisted plan is reused; verdict is re-overwritten by the re-verdict), but only once the Critical is fixed; today that resume lands in the crash loop.

## Style / Consistency
- Clean. Plain-English labels throughout figures; `_p0_grids` validation fails loud on malformed `--p0-grid-arm`; float formatting between judge (`str(c)`) and train (`f"c{coef}"`) verified consistent for all realizable scaled values (exact binary halving/doubling of GRID_L1).

## Unintended Changes
- None found; all hunks trace to the §6 figures list, the §7 remedy, or the unified smoke.

## Tests
- New coverage: 6 P0-verdict tests (repilot block slug-canonicality, ×2 grid, shifted-grid re-verdict with `sign_check_coef == 1.5`, empty-plan rc=7, bad `--p0-grid-arm` SystemExit); 9 synth/resolve/argparse-mutex registry tests incl. non-canonical spelling refusal (`c2.50`); figures tests render through the REAL `savefig_paper` to tmp dirs.
- Missing coverage: the re-pilot count gate under a resume-skipped (deduped) cell — the exact case the design guarantees will occur (Critical); `p0_run_repilot` end-to-end bash probe with one skip.
- Existing tests still valid: yes — 69/69 pass at worktree HEAD.
- Sandbox status: ran normally (thread-cap-prefixed, timeout-bounded).

## Security Check
- No issues found. Upload paths ride the canonical `hub._upload`/`upload_raw_completions_to_data_repo` helpers; no hardcoded tokens; heredocs use env-var passing, not interpolation of secrets.

## Recommendation
revise-then-merge: fix the re-pilot hook-count gate to account for resume-skipped overlap cells (fix shapes (a)/(b)/(c) above), add the fails-pre-fix probe/test for a skipped cell, and add the smoke blind-spot enumeration line for the re-pilot path. The figures unit and the coef-scale CLI plumbing are sound as shipped.
