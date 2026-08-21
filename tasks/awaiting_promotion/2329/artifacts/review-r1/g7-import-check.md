PASS

## Split-review sub-scope g7 — commit `0fa25bb09c22` ("bare --import-check form on judge + decay CLIs")

**Tier:** leaf (two one-off `scripts/issue2329_*` phase-dispatch entrypoints; CLI-surface-only change — module importers of these files never execute `main()`/`parse_args()`).
**Scope reviewed:** `git show 0fa25bb09c22` only (+8/−2 per file: `scripts/issue2329_decay.py`, `scripts/issue2329_ladder_judge.py`). CONTRACT-BEARING: no — round gates 0.5/0.55/0.6/0.8/0.9 skipped per brief.

### Verdict basis (all probes run live in the worktree, HEAD `c46f29bf0c33`)

1. **The check can actually fail / is not hollow.** Both `_import_check()` bodies (pre-existing; this commit only unblocks reaching them) execute the DEFERRED production imports the module path never reaches — decay: `transformers.AutoTokenizer`, `matplotlib`, `scipy.stats.spearmanr`, `analysis.paper_plots`, `eval.judge_pilot`, `orchestrate.hub`, `issue2329_ladder` (pulls torch) + registered-constant asserts + `assert_args_attributes_defined(__file__)` (scripts/issue2329_decay.py:1270-1303); judge: `orchestrate.hub` + registry-invariant asserts + argcheck (scripts/issue2329_ladder_judge.py:1003-1022). Any failure propagates as an exception → non-zero exit. Not a check that always passes.
2. **dotenv ordering preserved.** Both files call `orchestrate.env.load_dotenv()` at module level BEFORE any heavy import (decay lines ~64-68, judge lines ~69-72), so the import-check form runs under the identical env setup as production — no divergence channel.
3. **No silent default for production.** `--phase` becomes parser-optional but the post-parse guard (`raise SystemExit("--phase is required unless --import-check")`, decay:1312-1313, judge:1034-1035) makes a production invocation with a missing `--phase` fail loud rc=1 — no default phase substituted; `choices=tuple(PHASES)` validation retained for provided values. Guard sits AFTER the import-check branch and BEFORE `_stage_inputs` (judge) / `DecayConfig` (decay), so the bare form performs no staging/network/file side effects.
4. **Exit codes verified by execution (commit-message claims reproduce exactly):**
   - `issue2329_decay.py --import-check` → rc=0, `[import-check] issue2329_decay OK`
   - `issue2329_ladder_judge.py --import-check` → rc=0, `[import-check] OK`
   - both no-`--phase` probes → rc=1, `--phase is required unless --import-check` on stderr
   - **fails-pre-fix certified against the parent blob** (`git show 0fa25bb09c22^:scripts/issue2329_decay.py` extracted + run): bare `--import-check` → rc=2, `error: the following arguments are required: --phase` — the exact Axis-1 defect claimed.
   - Return plumbing: decay `_import_check()` returns None, main returns `RC_OK` explicitly; judge `return _import_check()` (returns `RC_OK`) — both flow to `sys.exit(rc)`. No pipe/exception masking; no `try/except: pass` anywhere in the diff.
5. **Misuse exit-code change (rc=2 → rc=1) breaks no caller.** Repo sweep: no caller keys on rc=2 from these CLIs; `issue2329_ladder_dispatch.sh` references the judge CLI in prose only (its rc semantics 24/25/26/28/29 belong to the pod drivers). rc=1 collides with no registered gate rc (decay 7/8/9; judge 7/8/9/10 — both docstring-registered).
6. **Sibling completeness (cross-commit awareness).** The claimed convention alignment is real: `issue2329_ladder.py` (parser lines 214-219, main 2287-2290) and `issue2329_run.py` (main ~5866-5871) already have optional `--phase` + post-parse guard; `issue2329_ladder_analysis.py` uses a defaulted `--step` (no required arg), so the two files fixed here are exactly the full defect set.
7. **No smoke-conditional substitution or gate downgrade** in the diff — the guard binds production identically; `--import-check` is the repo-established preflight entry form (`.claude/rules/code-style.md` § Argparse-attribute completeness), not a smoke branch. No `smoke-blind-spot-unenumerated` flag.
8. `uv run ruff check` on both files: clean.

### Non-blocking notes (no action required)

- **N1 (note):** No unit test pins the guard or the bare form; acceptable — the binding adoption arm is the smoke-architecture Axis-1 `--import-check` marker line per `.claude/rules/code-style.md` ("driver-local opt-in convention, NOT a repo-wide lint"), and both probes were executed (reproduced above).
- **N2 (note):** Mechanism diverges cosmetically from the cited ladder convention: ladder/run use `assert args.phase` (AssertionError traceback; dead under `python -O`) while this commit uses `raise SystemExit(msg)` — the NEW form is the better one; no change requested.
- **N3 (note, pre-existing):** `--phase X --import-check` runs only the import check and exits 0 (phase silently not run) — established branch ordering shared by all four sibling drivers and shown in the decay docstring usage line; not introduced by this commit.

**Recommendation:** merge as-is.
