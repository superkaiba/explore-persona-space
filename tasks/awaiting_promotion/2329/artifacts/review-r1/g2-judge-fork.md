# Split-review r1 g2 — commit `bc03d6a37e07` (scripts/issue2329_ladder_judge.py, +1044)

CONCERNS

**Tier:** trunk (imported by `tests/test_issue2329_ladder_decay.py` and `scripts/issue2329_decay.py:196` — multiple importers).
**Scope:** commit `bc03d6a37e07` only (`git show`); whole-branch diff never read. CONTRACT-BEARING gates 0.5/0.55/0.6/0.8/0.9 skipped per brief.
**Plan version used:** every plan-grounded check below was made against **v8** at `/home/thomasjiralerspong/explore-persona-space/tasks/followups_running/2329/plans/v8.md` (per the orchestrator's mid-review correction). The worktree's stale `plan.md`→v4 was **never read** — no finding in this review is v4-grounded.

## Method

The file is a fork of `scripts/issue2162_ladder_judge.py` (unmodified this round). I diffed the commit blob against the parent (200-line delta), read the delta in full plus the full donor-screen / pilot / gate / staging / pools / CLI sections, and live-probed every cross-module seam.

## Verified clean (with evidence)

- **Judging discipline (the brief's checklist), all PASS:**
  - Model pin: `--judge-model` defaults to `J94.DEFAULT_JUDGE_MODEL` → `explore_persona_space.eval.__init__:11` = `claude-sonnet-4-5-20250929` (env-overridable per CLAUDE.md). No Haiku suffix, no gpt-4o anywhere.
  - Routing: all waves go through inherited `issue2094_judge.run_wave` → `judge_graded` → `api_dispatch`; `threshold_base=FORCE_SYNC` = `J62.FORCE_SYNC_THRESHOLD_BASE` = `10**9` (`issue2329_judge.py:93`) — forces the SYNC fan-out, matching v8 §9 line 289's registered ALL-SYNC routing for Leg A (per-wave ≤ ~2,160; realized largest wave here ≤ 1,320 grid items). No direct `anthropic` client call in the commit (grep clean).
  - `max_tokens`: default `J94.DEFAULT_JUDGE_MAX_TOKENS` = 1024 (`issue2094_judge.py:94`) — at the single-rationale floor, matching v8 line 179/364.
  - Drop-never-coerce: `_scores_by_rid` (fork:255-264) and `_donor_means` (fork:557-565) skip `score is None` rows; `_qualifies` (fork:597-600) treats an unscoreable donor as FAILING, never qualified; `separation_verdict` (fork:345-414) marks carriers `unscored` rather than coercing empty means to 0 (`_mean` returns None on empty, fork:267). Transport losses retried (bounded, rule 24) in inherited `run_wave` (issue2094_judge.py:730-753), never persisted as drops.
  - Pilot gate genuinely gates: `phase_pilot` returns `RC_PILOT_GATE` on any rubric failing (fork:329), REFUSES `--dry-run` (rc 10, fork:278-284); `_require_reports` (fork:334-341) checks the report's `passed` key and raises on missing/failed — consulted by `phase_gate` and `phase_donor_screen` (`pilot_gate_report.json`) and by `phase_waves`/`phase_conjuncts` (`_ALL_GATES` = pilot + coherence-baseline + separation, fork:782-786). All spend-bearing phases are gate-guarded before dispatch. Pilot sizing 8×56=448 matches v8 line 289 ("pilot 448") and the rule-26 arithmetic (floor(1/0.02)+1=51 ≤ 56).
  - Batch API: correctly NOT used here — v8 registers Leg A as sync (≈6.6k total, ≤~2,160/wave); Batch is Leg B (`issue2329_decay.py`, another commit's scope). No unbounded `while True` poller in this commit.
  - Cache: rubric-keyed cache dirs (`cfg.cache_root / rubric_id`) + rubric-identity fingerprint (`batch_judge.rubric_fingerprint` covers judge_model + system prompt + user formatter) + wave-level `wave_regime` includes the prompt — a rubric change cannot silently reuse old scores. Cache root is issue-2329-scoped (`data/issue_2329/ladder_judge_cache`), so no cross-issue reuse of #2162 scores.
- **Fork fidelity to v8 §4.1 item 2 (line 107):** the delta vs the 2162 parent is exactly the declared set — J62 import swap, `issue_2329` work/in/cache roots, `Q35_PARENT_HF_REVISION` parent-anchor pin, F4 pe-viability leg. Nothing undeclared.
- **Cross-module seams, all live-probed:**
  - All 20 `J62.*` and all 46 distinct `J94/J62/LB/RESCORE` attribute uses resolve (hasattr probe run against the imported modules); `--import-check` passes live (`[import-check] OK`; `_import_check` body is byte-identical between this commit and round HEAD).
  - `Q35_PARENT_HF_REVISION` (fork:101) == `issue2329_ladder.py:140` == v8's pin `49d7f0017e…` (plan lines 37/107). Staging routes parent anchors at the pin and ladder artifacts at `--hf-revision` (fork:915-925).
  - Manifest key `parent_no_prefix_context_ids` read at fork:502 matches the producer (`issue2329_ladder.py:459`) and the driver-side consumer (`issue2329_ladder.py:552`) — same enriched frozen bank, so judge `pe_viable` and driver pe-exclusion are consistent by construction.
  - Assignments JSON consumed by `issue2329_ladder.read_donor_screen` (asserts status ∈ {primary, alternate, dropped}; donor dict with `b` for kept) and `issue2329_ladder_analysis.registered_row_keys` (`status=="dropped"` → drop null_xtype; `pe_viable` default-True, but the judge always writes it) — both match what `phase_donor_screen` writes.
  - `pools_ladder.json` matches `issue2329_run.load_pools`'s fail-loud schema (`pools` → items with `side`/`text`).
- **F4 selection ladder (fork:630-655) traced through all branches** — matches v8 line 48 ("primary → alternate → skip-with-record"): qualified pe-viable primary → primary; qualified no-prefix primary → qualified pe-viable alternate (`pe_fallthrough`) else KEEP primary (driver later pe-excludes); failed primary → qualified alternate else dropped. The `means2` widening (fork:604-612) correctly judges alternates for BOTH failed primaries and passed-but-no-prefix primaries, so branch 2 always has a scored candidate. Missing parent anchor rows for a donor ctx fail LOUD (fork:521-528). Gate bars match v8 line 230/364 (0.25 / 0.5, both signed positive, ≥4/6 carriers; donor bar ≤15/100 inherited verbatim per v8 line 107).
- **Style:** in-situ `ruff check` clean; `ruff format --check` clean. (Isolated-blob B023 at fork:365 is per-file-ignored for `scripts/*` and is a non-bug: the `_ctx_stats` closure is invoked before the loop variable rebinds.)
- **No smoke-conditional substitutions or gate-downgrades:** `--dry-run` builds the SAME units through the same builders with zero API calls and persists nothing; the pilot refuses it outright. No substituted implementation, no downgraded assertion.

## CONCERNS (non-blocking)

1. **The F4 selection ladder — this commit's only genuinely new logic — has no test.** `tests/test_issue2329_ladder_decay.py` covers the CONSUMERS (`registered_row_keys` pe-exclusion, driver pe seams) but nothing exercises `phase_donor_screen`'s 5-branch selection or the `means2` widening (grep: no test touches `phase_donor_screen`/`pe_fallthrough`/`_qualifies`; the 2162 parent's screen was also untested, so no coverage was dropped — but the new branches are new behavior). This matters more than usual because v8 assumption 4 registers this very leg as the L3 re-check for "cross-type donors are pe-viable". Concrete failure scenario: a branch-ordering regression (e.g. the branch-2 np-check dropped) would only manifest if the staged bank ever carries a no-prefix donor — exactly the case the leg exists to catch — and would silently mis-set `pe_viable`, feeding the driver a pe cell built on a zero `v_pe` row. A pure-fixture unit test (synthetic `means1`/`means2` + np_ids, asserting the 5 statuses + counters) is cheap and needs no API.
2. **`Q35_PARENT_HF_REVISION` is duplicated (fork:101 vs `issue2329_ladder.py:140`) with no equality test.** The duplication is deliberate and commented (importing the ladder driver would pull torch into the VM judge process), and `issue2329_decay.py:1295` asserts only `len == 40`. Drift scenario: a future round bumps the driver's pin (re-staged bank) but not the judge's — parent anchors then stage at a different revision than the bank enrichment, and the F4 np_ids no longer describe the staged anchors. The round's test file already imports both modules (`LJ`, `LAD`) — a one-line `assert LJ.Q35_PARENT_HF_REVISION == LAD.Q35_PARENT_HF_REVISION` closes it.

## Minor / informational

- `build_gate_behavior_items` docstring (fork:179) says "1,560 calls" but the function builds only the behavior items (1,140); the 420 coherence calls are a separate wave dispatched in `phase_gate`. Inherited 2162 wording; phase total is right, attribution nit only.
- At THIS commit `--phase` was `required=True`, making the `--import-check` flag unusable bare; already fixed in-round by `0fa25bb09c` (bare `--import-check` form). Not a residual defect.
- `sx_vals` (target mean) and `deltas` (netted) in `separation_verdict._ctx_stats` can use slightly different draw subsets when a plain-rubric score is dropped for a draw the own-descriptor rubric kept; `n_ceil_kept`/`n_floor_kept` report only the former. Inherited from 2162 verbatim; drop counts elsewhere make this auditable.

## Recommendation

CONCERNS — no blockers; ship-safe as-is. Items 1–2 are cheap test additions the round could pick up in a later commit or a follow-up round.
