# Code Review: issue-2225 split-review r1 g6 (bc295a5aca) + round-level contract gates

**Verdict:** FAIL
**Blocker tags:** marker-shape (Step 0.55 — blocker body names `epm:smoke-architecture-check`; mechanical-contract-only: no `substantive` blocker; fix is ONE marker re-post, no code change)
**Tier:** leaf (g6 commit: `scripts/issue2225_figures.py`, one-off entrypoint; round-level gates audited per CONTRACT-BEARING brief)
**Diff size:** +9 / -3 lines across 1 file (g6 scope commit)
**Plan adherence:** COMPLETE (g6 scope: thread-caps pin fix; matches the #847 in-process pin convention)
**Tests:** PASS
**Tests actually run:** yes — `tests/test_shared_vm_thread_caps.py` + `tests/test_issue2225_figures.py`: 42 passed in 69 s (worktree)
**Lint:** PASS (`ruff check scripts/issue2225_figures.py` clean)
**Security sweep:** CLEAN (g6 diff touches only import ordering; no secrets/exec/deserialization surface)
**Needs user eyeball:** None

## Round-level gate findings (CONTRACT-BEARING group)

### Step 0.5 — implementation marker shape: PASS
Highest-version `epm:experiment-implementation` (v1, ts 2026-08-11T02:33:25Z, head sentinel `<!-- epm:experiment-implementation v1 -->`, canonical `task.py view 2225 --json`) carries all four H3 sections in order — `### (a)` L6 / `### (b)` L31 / `### (c)` L38 / `### (d)` L55 — plus optional `### (e)` L61, all non-empty. `(c)` carries multiple copy-pasteable commands with observable success signals (ruff, ruff-policy pin, pin-sweep with union run counts, invariant trio, 3 end-to-end commands with expected rc/artifacts). Formatting note (Style, non-blocking): `(c)` commands are inline code spans, not triple-backtick fenced blocks — commands are present, so this is present-but-imperfect, never a FAIL.

### Step 0.55 — smoke-architecture marker: **FAIL (Critical, tag `marker-shape`, names `epm:smoke-architecture-check`)**
- Presence + verdict: one `epm:smoke-architecture-check` events row exists with parseable `verdict: PASS_UNIFIED`. Presence is satisfied.
- Internal SHAPE fails the mechanical grammar. `task.py check-smoke-arch-registry 2225 --repo-root <worktree>` output: **REFUSE — no line-anchored `arm-registry:` line found** (the marker's `arm-registry:` line is prose — two commands + count breakdown — matching neither accepted form `arm-registry: source=<expr> file=<path> n=<int> members=<sorted-comma-list>` nor `arm-registry: N/A — <reason>`; #2176).
- Two further grammar defects are latent behind that first REFUSE and will fail the checker sequentially if fixed one at a time (task_workflow.py regexes, #2176):
  1. Heading `per-arm-resolution (one row per §4.5 config arm — …):` — `_MARKER_TOP_KEY_RE` matches only the bare key at line start; the parenthetical BEFORE the colon means the span never opens, so all 10 rows parse as `per_arm == {}` (clause-4 REFUSE). Prose is legal after the colon, never before it.
  2. Row keys `- A (E1×all×L1): REAL` — `_PER_ARM_ROW_RE` captures everything up to the first `:` as the key, yielding `"A (E1×all×L1)"`, which fails clause-5 set-membership against a `members=A,...` list. Rows must LEAD with the bare arm token.
- Substance is RIGHT (verified as the fallback/defence-in-depth arm): 10 per-arm rows cover exactly the 10 config classes (A,B,C,D,E,F,G,I,P,H; per-class cell counts 16+12+16+12+12+3+3+3+3+1 = 81 match `--check-registry`), all rows REAL and consistent with `PASS_UNIFIED`. The single FALLBACK is on the separate top-level `production-outroot-unit:` key (pre-dispatch build round; pod-side inputs do not exist yet), NOT a per-arm row — no verdict↔row inconsistency. `import-resolution:` line matches the accepted `mode=--import-check` shape with exact rc=0 commands.
- Impact: `/issue` Step 6d.0 runs this same checker POST-provision and will REFUSE the dispatch on this marker as posted — a burned pod-provision cycle.
- Fix (ONE re-post, all three defects together — do not fix serially): (i) structured `arm-registry: source=<registry symbol> file=scripts/issue2225_train.py n=10 members=A,B,C,D,E,F,G,I,P,H` (members = the bare arm tokens the rows key on; if the checker's driver-recompute abstains on a non-dict-literal registry, its OK line reads `marker-only` and reviewer set-equality — already done above — is the covering arm); (ii) bare `per-arm-resolution:` heading with any prose AFTER the colon; (iii) bare-token row keys, e.g. `- A: REAL — (E1×all×L1) smoke cell A__evil__c3.0 …`. Content of every row can be carried verbatim.
- Mechanizable: yes — `task.py check-smoke-arch-registry 2225 --repo-root <wt>` exit-status is the check (already exists; Step 6d.0 runs it).

### Step 0.6 — end-to-end smoke gate: PASS (with CONCERNS below)
`## Smoke run` present with 9 `### <phase>` sub-sections (registry-preflight, directions-filter, training-hook, eval-target-enumeration, judging-parse-gate, analysis-stats, figures, dispatcher-syntax, upload wiring), each with exact command/test battery, rc=0, and a real artifact digest (81-cell registry breakdown; 86-target enumeration; real `SteeredSFTTrainer` lifecycle on a tiny PEFT model with exact per-mode hook-delta asserts; png+pdf+meta.json triplets from a real CLI run + hero PNG visually read; bash 3-branch probe of `p0_handle_verdict_fail`; porcelain-empty output-path hygiene). Not --help/import-only: real tiny-slice executions with digests for every locally-executable phase. GPU-only phases (real steered training, vLLM generation, MMLU, activation capture) have NOT run pre-dispatch — declared covered pod-side by the plan's P0 pilot gate under the PASS_UNIFIED architecture (smoke IS the production dispatcher, `EPM_I2225_SMOKE=1`, same phase chain/fan-out, tiny-N dials, `*_smoke` out-root twins). That is the sanctioned pre-dispatch shape; no genuine-absence blocker.

### Step 0.8 — prior open binding concerns: PASS
`list-concerns 2225 --open-only --json` → `[]`. Ledger history: `octave-shift-repilot-no-coef-scale-cli` (CONCERN) raised 2026-08-11T01:25:10Z by experiment-implementer, addressed 02:29:50Z (implementer) and 02:45:06Z (code-reviewer, round 1). Addressing verified REAL in the worktree (grep-the-literal, Rule 9):
- `scripts/issue2225_train.py:979/986/992`: `--coef-scale` / `--pilot-coefs` / `--pilot-configs` argparse; `:954` requires `--pilot`; `:942` subset gate.
- `scripts/issue2225_judge.py:850-876`: per-arm `repilot` block with `coef_scale` in the P0 verdict.
- `scripts/issue2225_dispatch.sh:206-356`: `repilot_state.json` resume state, `p0_handle_verdict_fail` (:329), `p0_run_repilot` (:356) — the re-pilot path actually consumes the new CLI.
- Regression tests present: `tests/test_issue2225_cell_registry.py:208` `test_resolve_cells_pilot_coef_scale_halves_grid`; `tests/test_issue2225_judge_analysis.py:339` `test_p0_verdict_first_miss_emits_repilot_plan`.
No new concerns persisted this round (the 0.55 finding is a marker re-post, not a production-path deferral).

### Step 0.9 — git provenance: PASS
Round range `732791292966..bc295a5aca` = exactly the 6 unit commits (cc2b8affdc, 9c0204adaa, acabdf4124, 8b2c549c65, ecccdf108e, bc295a5aca). `git diff --name-status` over the range: 15 files, ALL status A (new adds) — the round modifies/deletes nothing pre-existing, so no finding here can be a diff-base artifact. The one union-run test failure (`test_no_new_torch_before_dotenv_vm_entrypoints`) was introduced by the round's own new file and fixed by the round's own bc295a5aca — re-verified passing (42/42). No git-provenance blockers.

## Plan Adherence (g6 scope)
- Thread-caps pin (#847 class) before heavy imports: ✓ — evidence: `scripts/issue2225_figures.py:38-45`: `from explore_persona_space.orchestrate.env import load_dotenv` + `load_dotenv()` precede `import matplotlib.pyplot` / `import numpy`.
- Fix is real, not cosmetic: `load_dotenv()` calls `_apply_shared_vm_thread_caps()` (`src/explore_persona_space/orchestrate/env.py:397`), which setdefaults `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS` (:115-118, :172) — and the import chain reached by `orchestrate.env` is heavy-import-free (`orchestrate/__init__.py` → `fleet.py`: stdlib-only module-level imports; `analysis/__init__.py` empty; `paper_plots` imported only after `load_dotenv()`), so the caps genuinely bind before numpy's BLAS pool sizes itself.
- `MPLBACKEND=Agg` setdefault stays ahead of the matplotlib import (`:33`): ✓.
- Uses `orchestrate.env.load_dotenv`, never bare `dotenv` (CLAUDE.md upload-policy rule): ✓.

## Issues Found

### Critical (block)
- `epm:smoke-architecture-check` marker (task 2225 events.jsonl, ts 2026-08-11T02:20Z): malformed `arm-registry:` line + prose-decorated `per-arm-resolution` heading + parenthetical row keys — checker REFUSE; Step 6d.0 will refuse the pod dispatch. See Step 0.55 section above for evidence, impact, and the one-post fix. Tag: `marker-shape`. Mechanizable: yes (`check-smoke-arch-registry` exit status).
  - Bug-class sweep (`### Bug-class sweep: marker-grammar-vs-checker`): the only other structured marker this round is `epm:experiment-implementation` — its H3 grammar parses (Step 0.5 PASS). No load-bearing siblings.

### Major
- None.

### Minor
- Implementation marker `## Smoke run` preamble + smoke-arch `notes:` claim the smoke-mode P0-verdict demotion is "declared in the plan §4.8 blind-spot enumeration". The plan's §4.8 enumeration (plan.md:154) enumerates a DIFFERENT layer's blind spots (judge transport, MMLU `--limit 200`, evil-II-only P0 cells, P1 full artifacts) and states "No implementation substitutions or downgraded assertions exist in the smoke path" — it does not declare the `EPM_I2225_SMOKE` demotion (`scripts/issue2225_dispatch.sh:316-317`). The disclosure duty itself IS met (the implementer mirrored the enumeration in the marker per smoke-blind-spots.md when the realized code added the branch), so no `smoke-blind-spot-unenumerated` blocker — but the plan citation is inaccurate; the marker re-post (already required by 0.55) should attribute the demotion to the implementer-mirrored enumeration, not plan §4.8. Mechanizable: no (prose-attribution accuracy).
- `(c)` commands are inline code spans, not fenced blocks — cosmetic; fold into the same re-post if convenient (no separate action).

## Unaddressed Cases
- None found in g6 scope. (Empty eval-root fail-loud rc=1 path is test-pinned per the marker; not re-run here.)

## Style / Consistency
- g6 diff matches the sibling issue2225 scripts' load_dotenv-first convention (commit message claim verified by the passing `test_no_new_torch_before_dotenv_vm_entrypoints` scan, which covers all VM entrypoints mechanically — no unfixed siblings of the heavy-import-before-dotenv class).

## Unintended Changes
- None — the g6 hunk touches only the import block it claims to.

## Tests
- New coverage: thread-caps entrypoint scan covers the new figures script (27 passed); 15 figures-builder tests passed.
- Missing coverage: none for g6 scope (the invariant is already CI-pinned by `test_no_new_torch_before_dotenv_vm_entrypoints` — the Rule 13 regression-test requirement is satisfied by the pre-existing pin).
- Existing tests still valid: yes (42/42 in worktree).
- Sandbox status: ran normally.

## Security Check
- No issues found (import-order-only diff; no secrets, no exec, no network surface).

## Recommendation
Revise-then-re-review at marker level only: the CODE of g6 (and the round-level substance audited here) is sound; the sole blocker is the non-conforming `epm:smoke-architecture-check` marker grammar. Implementer re-posts ONE conforming marker (structured `arm-registry:` line, bare `per-arm-resolution:` heading, bare-token row keys — all three together), correcting the plan-§4.8 attribution in the same post. No code change required. This FAIL is mechanical-contract-only (tags = {`marker-shape`}); it stands (is not strippable) because the named marker is present but NON-conforming — the strip precondition is a conforming marker.
