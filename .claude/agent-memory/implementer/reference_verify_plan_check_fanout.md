---
name: verify_plan.py check-addition fan-out
description: Adding a check cN to scripts/verify_plan.py touches more count pins than the obvious registry test — full surface list
type: reference
---

Adding a new check to `scripts/verify_plan.py` fans out to:

0. FIRST, immediately before the round's final commit: probe origin/main for
   the CURRENT max id — `git fetch origin main && git show
   origin/main:scripts/verify_plan.py | grep -oE '\bc[0-9]+\b' | sort -uV |
   tail -1` (verify_task_body.py analogue: `grep -oE 'check [0-9]+'`) — and
   renumber to max+1 if the plan's id is taken (#1569; #1550/#1551 both landed
   `c40`, PR #1321 conflict). Pre-authorized; see implementer.md § During
   Implementation. Re-probe after any rebase/conflict round.
1. Module docstring: check-table row + the "Conditional checks (4, 6, 7, ...)"
   enumeration + the canonical N/A escape-phrase list (labels keyed to check
   NUMBER — grep your diff hunks for `\(check (\d+)\)` and assert the new id).
2. `CHECKS` list (append last; never reorder).
3. `tests/test_verify_plan.py` — THREE count-pin sites, not one (#932: plan
   assumption A7 named only the first):
   - `test_good_plan_passes_all`: expected-status dict + `len(results) == N`.
   - `test_cli_json_schema_and_exit_zero_on_pass`: `n_skip == K` AND
     `len(payload["checks"]) == N` (appears TWICE — checks + unique ids).
4. `.claude/skills/adversarial-planner/SKILL.md` Phase 1.5.0 escape list
   (historically LAGS the module docstring — check for missing back-fills).
   The back-fill has a SECOND-order cost: SKILL.md is under the
   `check_skill_doc_size` grandfather ratchet, so even a ~200 B escape entry
   can push it past its cap and FAIL the no-flags lint fleet-wide — same-round
   landing-bytes cap raise (`SKILL_DOC_SIZE_GRANDFATHER`, cap = landing bytes
   + ~1 KB, #1753/#2240; hit at #2123: 71,103 B > 70,900 cap).
4b. SIBLING enum tail-pin: a SIBLING check's registration test pins the
   conditional-enum TAIL literal (e.g. `"...66, 67)"`); extending the enum
   breaks it BY DESIGN (the house loud-reminder pattern) — update the pin to
   the new tail in the same round. The pin does NOT move to the newest
   check's file: it has stayed in `tests/test_verify_plan_c58_fanout_pod_name.py:~115`
   through c59...c68 (#2123: caught c59; #2228: caught c68 via the pin-sweep
   `--map-files` col-1 list). Find it with `grep -rn '66, 67)' tests/`-style
   greps on the OLD tail.
5. If mirrored as a critic-lens item: critic-lens-reference.md (full text) +
   critic.md (item-name run) + statistics-critic.md ("items I own") +
   lens-coverage-map.md § table row (`v2-owner: ...`).
6. C901 on a branch-rich check: verify_plan.py is in the
   `tests/test_ruff_policy.py` LIVE_WORKFLOW_HELPERS full-ruleset pin, and a
   many-SKIP-branch check body trips `C901 (>15)` there even though bare
   `ruff check` passes (per-file-ignores relax scripts/). The file's house
   remedy is HELPER EXTRACTION (`check_battery_multiplier` precedent — zero
   `noqa: C901` in verify_plan.py; workflow_lint.py uses annotated noqa
   instead): extract the budget/harvest ladder into a `_cNN_*` helper
   returning result-or-skip-reason (#2299: `_c70_resolve_budget`, 21 -> under
   15). Run the ruff-policy pin BEFORE the final commit, not only bare ruff.

**How to apply:** grep `len(results) ==`, `n_skip`, `len(payload` in the test
file before claiming the pin list is complete. Corpus-replay probes live at
`/tmp/c1*_probe*.py` (plan-time); adapt to call `verify_plan_text` directly
for the committed replay.

Item 4 keeps being wrong in plans: #937's plan §2.3 explicitly claimed
SKILL.md has "no per-check enumeration" — it does (Phase 1.5.0 canonical
N/A escape list, per-check-numbered). Trust this memory over the plan;
verify with `grep -n 'N/A — no' .claude/skills/adversarial-planner/SKILL.md`.

**Calibration sweep (new-cN trigger regexes):** run the check IN-PROCESS over
`tasks/*/*/plans/v*.md` (skip `plan.md` symlinks + own task; ~1100 files,
seconds) — never a per-file `uv run` subprocess loop (~700 × 2 uv startups).
Loading verify_plan.py by path REQUIRES `sys.modules["verify_plan"] = mod`
BEFORE `spec.loader.exec_module(mod)` — the `@dataclass` decorator resolves
`cls.__module__` via sys.modules and dies with
`AttributeError: 'NoneType' object has no attribute '__dict__'` otherwise
(#2228: the approved plan's own §6.3 snippet carried this latent bug).
Tolerate `FileNotFoundError` mid-glob (live tasks `git mv` between statuses).
The top corpus noise class for any re-X/redo-style trigger is NEGATED
mentions ("NOT regenerated", "NO re-extraction of r_B") — fixed-width
lookbehinds `(?<!\bno )(?<!\bnot )(?<!\bnever )(?<!\bwithout )` kill most of
it while staying `_trigger_windows`-compatible (#937: 54→27 files with this
+ the same-line demotion). Report the realized fire set vs the plan's bounds
with a per-task genuine-vs-benign adjudication table; a "kill bound" usually
requires MOSTLY-benign, not just count-exceeded.
