# Task #2569 — Step 5 fix round v2, SHARED brief

Read this file first, then your unit's specific blocker list in your prompt.
Delete nothing from this file; it is committed so every unit reads the same
contract and a resumed session can re-read it.

## Round context

The Step 5 code-review ensemble returned **FAIL on both sides** (Claude 5 FAIL /
1 PASS across 6 shards; Codex 3 FAIL across 3 shards). Blockers union; this is
the single consolidated fix round over that union. **22 open BLOCKERs**, split by
file set so no two units ever write the same file.

**Reviewed pin:** the review ran against `4a48517b13`. The branch tip is now
`a13d2d169c`, which is **source-identical** to the reviewed pin — verified by
`git diff --stat 4a48517b13 HEAD -- scripts/ src/ tests/` returning empty. The
three intervening commits are agent-memory only (`.claude/agent-memory/**`),
committed in-turn by the orchestrator because a tracked write left at the shared
repo root arms the #2015 pre-commit stash race for every concurrent session. No
reviewer verdict is invalidated by the drift. Work from the current tip.

## Worktree

`/home/thomasjiralerspong/explore-persona-space/.claude/worktrees/issue-2569`,
branch `issue-2569`. Work there. Never `git checkout` / `switch` /
`reset --hard` / `clean -f` / `merge` in the repo-root tree.

## Resolve the plan path — do NOT hardcode a status folder

```bash
PLAN="$(cd /home/thomasjiralerspong/explore-persona-space && uv run python scripts/task.py find 2569)/plans/plan.md"
```

Run that `task.py find` from the MAIN checkout, not the worktree: a worktree's
`tasks/` tree is frozen at its base commit and serves a stale plan with no error.
`plan_version=v4`. The task has moved folders mid-round already, which is exactly
why the resolve form is mandatory.

## What "fixed" means here

For every concern id assigned to your unit:

1. **Fix the defect in the code.** Not a comment explaining it, not a docstring
   caveat, not a `TODO`. If a plan-registered item has no implementation, the fix
   is to implement it.
2. **Add a regression test that FAILS before your fix and PASSES after.** Verify
   both directions — actually stash or revert your change and watch the test go
   red, then restore. A test that passes on the unfixed code is not a regression
   test. This round has a live example of the failure mode: a fixture was written
   carrying a `response` field the real data never has, so it validated against
   an impossible schema and proved nothing.
3. **Probe real data before you encode a schema.** Every schema assumption gets a
   probe against a real artifact on disk, and the probe output goes in your
   report. Two blockers this round exist because a field was assumed rather than
   read.
4. **Address the concern in the ledger**, from the main checkout:

```bash
cd /home/thomasjiralerspong/explore-persona-space
uv run python scripts/task.py address-concern 2569 \
  --concern-id <id> --by experiment-implementer --round 2 \
  --summary '<one line, <=200 chars: what changed + the test that pins it>'
```

`--by` and `--round` are required. Use `--summary-file` if the line runs long.

5. **If you conclude a blocker is WRONG, say so with evidence and do NOT fix it.**
   One reviewer blocker this round was already refuted by direct test
   (`operator-runtime-orientation-gate-hollow` — transposing W in the real
   payload IS caught by assert (iii), so the Codex claim did not hold). A refuted
   blocker is a legitimate outcome; a silently-skipped one is not. Report the
   probe you ran.

## Duties that bind this round

- **Vectorize before you widen.** A per-cell / per-pair / per-draw Python loop
  over model forwards, fits, or factorizations is the recurring throughput
  failure here. Read `.claude/rules/vectorize-many-cell-fits.md`. The
  highest-cost finding in this round is exactly this shape: full dense SVDs of a
  rank-32 delta-W, ~25-45 s per unit at production shape, 20-30 h against an 8 h
  cap. The fix for that class is EXACT and not an approximation — a rank-r
  product has its singular values in the r x r core after a QR, at O(d*r^2).
- **Measured basis, never guessed.** Any wall-time claim you make comes from a
  measured pilot through the production entrypoint at production shape, or from a
  cited prior measurement of the same kernel at the same shape. State the measured
  number. A self-set timeout is sized at >= 2x the pilot-extrapolated wall.
- **Detached phases > ~15 min checkpoint per cell-chunk** into the durable
  out-root, never only at process exit.
- **Figure sanity: Read the rendered PNG** before you claim a figure works, and
  render it the way PRODUCTION renders it. This round's figures blocker is
  precisely a defect that every unit-local render missed because the production
  style path was never exercised: `set_paper_style("blog")` sets
  `lines.markeredgewidth: 0`, so `plot(..., mfc="none")` with no explicit `mew=`
  draws zero ink and whole data series vanish while 24 tests stay green.
- **Estimator validity.** Before any ridge / linear-map / probe fit, state
  `n_train` vs feature dimension `d`. `n_train < d` is estimator-degenerate and
  refused unless you justify the under-determined regime explicitly. No pure-GCV
  lambda selection below that threshold. Report the selector and the selected
  lambda, and disclose a lambda at a grid edge.
- **Smoke blind-spot enumeration.** If your fix adds or edits a `smoke`-
  conditional branch that substitutes an implementation, downgrades an assertion,
  or leaves a production-only import, enumerate what the smoke PASS does NOT
  certify. Write the literal `none — smoke executes every production gate` if the
  enumeration is genuinely empty.
- **Fail fast.** No `try/except: pass`, no placeholder values, no dummy data on
  error, no silent defaults, no `--force` / `--no-verify`. Several blockers this
  round ARE silent-default defects: a bare `.is_file()` guard that drops a primary
  DV at rc=0, an empty basis silently dropped, a gate verdict overwritten by a
  resume. Do not fix a silent failure by adding a different silent failure.
- **Resume keys must cover every output-affecting input.** Four blockers this
  round are this one class. A phase's regime key includes the content
  fingerprints of its inputs — not their status strings, not just the flags that
  happen to be on the command line. Status strings are the specific trap: they
  flip `deferred -> computed` but say nothing about the CONTENT, so a regenerated
  producer leaves a stale consumer logging `SKIP (done)`.

## Commit contract

- Stage and commit **by explicit path**. Never `git add -A`, never `git add .`.
- Commit with an explicit pathspec: `git commit -m <msg> -- <paths>`.
- Never pipe `git push` / `git commit` / `git merge` through `tail` / `grep` /
  `head` — the pipe masks the exit code and SIGPIPE-kills hooks mid-run. When you
  need the text: `git push origin issue-2569 > /tmp/push.out 2>&1; echo rc=$?`
  then read the file.
- Never force-push. Never `--no-verify`.
- Commit and push **in the same turn you make the edit**. A turn that ends with
  staged-but-uncommitted work is an incomplete turn.

## Gate before you report

From the worktree:

```bash
uv run python scripts/workflow_lint.py            # no flags; verdict is the EXIT CODE
echo "rc=$?"
```

`workflow_lint` violations print as `workflow_lint: <file>:<line>:` with **no
FAIL prefix**, so `grep FAIL` returns 0 on a failing run. Read the exit code.
Then run the mapped tests for your changed files:

```bash
uv run python scripts/select_step9c_tests.py --map-files <your changed paths>
```

Run the selected tests as ONE pytest union and report the counts. There is one
known pre-existing red unrelated to this round —
`test_no_new_torch_before_dotenv_vm_entrypoints`, offender
`scripts/issue2254_firstk_ctxext_sensitivity.py` (#2572). Confirm any failure you
report is not that one before attributing it to your payload.

## Report back

- Per concern id: FIXED (with the regression test name) / REFUTED (with the probe)
  / BLOCKED (with what blocks it).
- The probe output for every schema assumption you encoded.
- Measured walls for anything you claim about runtime.
- `workflow_lint` rc and the pytest counts.
- Anything you found that no reviewer raised. Two units last round found real
  defects that way, including one that would have hung a lane forever.

Do not invoke any review or diagnostic skill on your own output. Reporting is
part of the work: a turn that ends with the report undelivered is incomplete.
