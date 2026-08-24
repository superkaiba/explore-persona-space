<!-- epm:results v2 -->
## Completion Report — #2302 implementer round 2 (AMENDED; supersedes v1)

**Task:** Base-identity filter for Step 5a sibling-sync files (#2296) — round 2: the review round-1 blocker (v1 marker missing `### (d)`) + all four Minors (M1–M4).
**Status:** SUCCESS

This marker AMENDS `epm:results v1`: it carries the full four-H3 completion-report contract (the v1 omission of `### (d)` was the round's one blocker, `mechanical-contract-only`); (a)–(c) below cover the ROUND-2 delta and re-state the round-1 verification headlines; v1 (plus its erratum `epm:progress` note — the pin-sweep list is 92 files, not 93) remains the round-1 detail record.

### (a) What was done

Round-2 delta (commit `5c88314a3250793fee462c2c6a3535ea5fdd9f01`, SHA-verbatim from `git log --format=%H`; branch `issue-2302` pushed, `origin/issue-2302 == HEAD` verified by rev-parse equality after fetch):

- **M1** — `scripts/select_step9c_tests.py`: `_base_identical_audit` drops the `if not touched: return []` guard. A sync-commit-only branch (ENTIRE three-dot diff base-identical) filters `touched` to `[]`; pre-fix the audit short-circuited and the exclusion was the ONE silent case, contradicting the plan's "the exclusion is never silent". Now `base_identical_excluded` names both synced paths and the exclusion NOTE fires alongside the empty-diff fallback NOTE. New test `test_sync_only_branch_exclusion_is_loud` (real git fixture, real CLI `sel.main(["--json", ...])`, asserts the sorted pair + both NOTEs).
- **M2** — same helper: audit entries still present in `touched` are DROPPED before reporting — under an asymmetric transient git failure between `main()`'s two seam calls, the audit key can no longer claim an exclusion the realized filtering did not apply (selection was never affected; this closes the false-audit direction).
- **M3** — `scripts/step9c_baseline.py`: the compare JSON now emits `"base": ctx.base` (the RESOLVED diff base the compare derived against). `_CompareCtx.base` was stored per plan Change 2 but consumed nowhere; base-resolution debugging is the #2293 problem class, so the artifact carries the field. Asserted in `test_base_identity_derivation_failure_warns_not_indeterminate` (`out["base"] == "main"` under the fake selector, which has no `resolve_base`).
- **M4** — `.claude/skills/issue/SKILL.md`: the Step 9c 1d parenthetical's stale-sync-residual sentence now CROSS-REFERENCES "Step 5a § Base-identity invariant (#2302)" instead of restating it (−184 B, measured 944,138 B); `scripts/workflow_lint.py` `SKILL_DOC_SIZE_GRANDFATHER["issue/SKILL.md"]` LOWERED 945_600 → 945_400 (new measured + ~1.2 KB landing-bytes headroom, 1,262 B; the comment records the give-back and demotes round 1's raise to `Prior:`). Prose-pin sweep before the edit: the only test hits on the edited strings are the JSON key name `base_identical_excluded` (kept verbatim).
- **Blocker** — this marker itself: the four-H3 contract (`implementer.md` § Report Format) is now satisfied; see `### (d)` below.
- Branch total vs merge-base: **+723/−17 across 6 files** (round-1 commits `21e3c36b3f1225e6c2d571bf6214708e8be65b89`, `0b00f749517b02c64e021f70626efe5a16d43df1`; round 2 `5c88314a3250793fee462c2c6a3535ea5fdd9f01`).
- Plan adherence: unchanged from v1 (all four plan changes DONE; both plan-named test ids verbatim); round-2 items are reviewer-directed amendments, each within plan scope.

### (b) Considered but not done

- Keeping the `if not touched` guard and special-casing only the "everything excluded" branch — rejected: `compute_base_identical` already returns `[]` fast on a genuinely empty three-dot, so the guard bought nothing except the silent corner.
- Placing the M2 consistency filter in `main()` (the coordinator's literal suggestion) — implemented inside `_base_identical_audit` instead, which already receives `touched`; same one-line effect, keeps `main()` under the C901 cap it grazed in round 1.
- Documenting the new compare-JSON `base` key in SKILL.md — declined; the Step 9c bullet names the two audit keys, and adding more prose in the same round as the M4 budget give-back would work against it.
- Raising rather than lowering the ratchet after M4 — the coordinator's explicit ask was the give-back; cap now 945_400 with 1,262 B headroom (≤ the 3,000 B hygiene bar).

### (c) How to verify

- **Round-2 regression demonstration (fail-pre / pass-post):** both new assertions FAIL against the round-1 tip (`git show HEAD:scripts/...` materialized into a scratch tree): `test_sync_only_branch_exclusion_is_loud` FAILED (pre-fix audit reads `[]`, not the sorted pair) and `test_base_identity_derivation_failure_warns_not_indeterminate` FAILED (pre-fix compare JSON has no `base` key) — `2 failed in 0.81s`; both PASS on the round-2 tree.
- **Named test files (coordinator's round-2 set), verbatim:** `uv run pytest tests/test_step9c_base_identity.py tests/test_select_step9c_tests.py tests/test_step9c_baseline.py -q` → **`321 passed in 30.71s`, rc=0** (9 + 153 + 159; +1 test vs round 1 — the new M1 test). All payload-attributed; zero failures.
- **Ruff-policy pin:** `uv run pytest tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset -x` → **`1 passed in 0.27s`, rc=0** (all three touched scripts are LIVE_WORKFLOW_HELPERS members).
- **Payload ruff:** `uv run ruff check` + `ruff format --check` on the four touched `.py` files → `All checks passed!` / `4 files already formatted`, rc=0.
- **No-flags workflow_lint (FINAL tree):** `uv run python scripts/workflow_lint.py` → **`workflow_lint: PASS`, rc=0**; 19 pre-existing WARNs (same roster as round 1), none a failure line naming a round-committed file; the payload-named line is the designed post-give-back state: `WARN: .claude/skills/issue/SKILL.md: 944138 bytes — grandfathered; 1262 bytes under its cap (945400)`. (First attempt rc=124 at a 540s inner bound under fleet contention — kill-probed, the only live `workflow_lint` processes belonged to a concurrent session's Step 9c scratch tree, left alone; clean retry PASSed.)
- **Gate-scope note:** round 2 touched the same 5-file surface as round 1 (+ the same test file), so the v1 selector enumeration stands (`n_tests=211`, base=`origin/main`, 2 `slow_tests_selected` deferred to Step 9c with the 10050s recommended timeout, invariant-only remainder 36). The 92-file pin-sweep hit list (v1 fenced block, minus the erratum's one extraneous entry) and the 23-file grep-supplement all ran green in round 1; round-2 deltas are covered by the three named files above plus the lint battery.
- **Reproduction commands:**

```
cd /home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-2302
uv run pytest tests/test_step9c_base_identity.py -q            # 9 passed
uv run pytest tests/test_select_step9c_tests.py tests/test_step9c_baseline.py -q
uv run python scripts/workflow_lint.py
```

- **What success looks like:** a sync-commit-only branch reports its exclusions loudly (`base_identical_excluded` + NOTE) while selecting invariant-only; the compare JSON carries the resolved `base`; the audit key never names a path the selection still treats as touched; SKILL.md sits 1,262 B under a cap that gave back the duplicated prose budget.

### (d) Needs human eyeball

- **Selector latency:** `main()` now makes ~3 extra git subprocess calls per real invocation (the public-seam route: `compute_touched` + `compute_base_identical` each re-derive; tens of ms per call). Accepted for seam compatibility — flag if selector wall-time ever matters in a hot path.
- **The two deliberate drift-pin edits** in `tests/test_select_step9c_tests.py` (case-86 live-tree pin → 8 pairs / 720s; two exact `--json` key-set pins gain `base_identical_excluded`): exactly the update class those pins' docstrings invite, but they are edits to existing selector tests — worth a hand check against the plan's "don't fix the selector tests" clause (scoped to the injection-inertness class, which is untouched and passing).
- **Stale-sync residual semantics:** a path synced on an EARLIER round that origin/main has since advanced legitimately differs from the base tip and stays branch-touched/BLOCKING. That is the designed behavior (a rebase-landing would revert main's newer content), and the remedy is a re-sync via the Step 5a arm — but the first time it fires in anger it will look like a #2302 regression to whoever reads the gate output. The SKILL.md prose documents it; a human should confirm the framing reads clearly at the Step 5a site.
- **M1 corner UX:** on a sync-only branch, the stderr now shows the exclusion NOTE immediately followed by the "empty diff vs '<base>'" fallback NOTE. The pairing is self-explanatory in sequence, but the fallback NOTE's wording ("commit first; re-run from the worktree") predates #2302 and does not mention the exclusion case — a reader seeing only the second NOTE could still be misled. Left as-is (the exclusion NOTE is adjacent); eyeball whether the fallback wording deserves a follow-up touch.
<!-- /epm:results -->
