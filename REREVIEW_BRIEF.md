# Task #2569 — Step 5 re-review (round 2), SHARED brief

Read this, then your shard's file list in your prompt.

## What you are reviewing

Round 1 of the Step 5 ensemble returned **FAIL on both sides** (6 Claude shards:
5 FAIL / 1 PASS; 3 Codex shards: 3 FAIL) over **22 blockers**. Seven fix units
then ran on disjoint file sets. This is the re-review of those fixes.

**Review range:** `4a48517b13..HEAD` (HEAD = `1e4d1122aa`). 16 files, 12 commits,
301,700 bytes — sharded 3 ways so each of you stays under the ~300 KB budget where
a single reviewer autocompact-thrashes.

**Ledger state:** 0 open BLOCKERs, 7 open CONCERNs. Read the ledger before you
start; do not re-raise something already recorded as a deliberate deferral:

```bash
cd /home/thomasjiralerspong/explore-persona-space
LEDGER="$(uv run python scripts/task.py find 2569)/concerns.jsonl"
```

The 7 open concerns are: `atlas-phasefits-correspondence-null-serial`,
`leg1-fixedpoint-neighbors-no-pb-producer`, `leg3-rows-legs-need-pb-store`,
`leg6-ft-arms-no-delta-tf-unit`, `leg6-tbar-basis-mismatch`,
`review-tip-drift-orchestrator-memory-commits`,
`weights-base-regime-omits-banked-map-fingerprint`. Each carries its reasoning in
the ledger's `evidence` field. If you think one of those deferrals is WRONG, say
so with evidence — that is a legitimate finding. Do not simply restate it.

## Plan path — resolve it, never hardcode a status folder

```bash
PLAN="$(cd /home/thomasjiralerspong/explore-persona-space && uv run python scripts/task.py find 2569)/plans/plan.md"
```

Run `task.py find` from the MAIN checkout. A worktree's `tasks/` tree is frozen at
its base commit and serves a stale plan with no error. `plan_version=v4`.

## What I most want from this round

Round 1's reviewers were good at reading code and weak at one specific thing:
**every defect that only shows up when the code RUNS was missed.** Nine shards
missed both of the round's worst findings, and both were caught by rendering a
figure and reading it:

- `h2b-verdict-computed-in-underdetermined-regime` — the pipeline emitted
  `h2b-kill-candidate` against a pre-registered hypothesis from verdict points at
  n_train = 96 and 192 against d = 3584 (n/d = 0.027, 0.054), with
  `identity_bias_r2 ≈ -0.94`. No `n_train`-vs-`d` gate existed anywhere.
- `figures-render-verdict-bands-against-smoke-regime` — the figure drew registered
  pass/kill bands against `regime.smoke = true` data. Legible, well-formed, and
  headed for the analyzer as evidence a hypothesis failed.

So: **prefer running something over reading something.** Where a claim is
checkable by execution — render the figure and Read the PNG, drive a gate to both
verdicts, load the real artifact and check the schema, run the test after
reverting the fix — do that instead of reasoning about the diff. A claim you
verified by execution outranks a claim you verified by reading.

Three specific classes to hunt, all of which recurred this round:

1. **Impossible-schema fixtures.** A regression test whose fixture carries a field
   the real artifact never has proves nothing. This happened **three** times this
   round (a `response` field the sampling manifest lacks; a figures fixture
   hardcoding `smoke: True`; a wiring fixture missing the `regime` block the real
   producer always writes). For any new test, ask: does its fixture match a real
   artifact on disk? Probe one.
2. **Tests green for the wrong reason.** Round 1 had 24 green figures tests
   coexisting with entire invisible data series, because the suite never set the
   production style. Ask what regime each new test runs under, and whether that
   is the regime production uses.
3. **Vacuous verdicts.** A criterion computed over zero admissible cells, an empty
   roster, an all-degenerate point set. Several were fixed; check the fixes make
   the vacuous case *structurally distinguishable* from a genuine negative, not
   merely detectable.

## Standing duties to check the fixes against

- **Estimator validity:** `n_train` vs `d` stated before any fit; `n_train < d`
  refused unless explicitly justified; no pure-GCV λ below that threshold; λ
  selector and selected λ reported; λ at a grid edge disclosed.
- **Vectorization:** no per-cell/per-pair/per-draw Python loop over fits or
  forwards. Two fixes this round claim large speedups (a rank-r SVD core, a
  Haar-invariance shared null). **Check the claimed exactness is actually exact** —
  both claim algebraic rather than sampled equivalence, so there should be a test
  pinning agreement against the slow path, not just a timing number.
- **Fail fast:** no `try/except: pass`, no silent defaults, no placeholder values.
  Several round-1 blockers were silent-default defects; verify the fixes did not
  introduce different silent defaults.
- **Resume keys** cover every output-affecting input — content fingerprints, not
  status strings, and not recomputed floats (machine-stability).
- **Figures:** simple, no on-canvas caption/provenance blocks, one colour one
  meaning, no annotation overlays.
- **Smoke blind-spot enumeration** where a fix added or edited a
  `smoke`-conditional branch.

## Gate context

The tree passes `workflow_lint` at rc=0 as of `1e4d1122aa`. **The verdict is the
EXIT CODE** — violations print as `workflow_lint: <file>:<line>:` with no `FAIL`
prefix, so `grep FAIL` returns 0 on a failing run. Under current load a full scan
can exceed 10 minutes; a killed run is INCONCLUSIVE, never clean.

One known pre-existing red, not this round's:
`test_no_new_torch_before_dotenv_vm_entrypoints`, offender
`scripts/issue2254_firstk_ctxext_sensitivity.py` (#2572).

## Your verdict

End with a `VERDICT: PASS` or `VERDICT: FAIL` line. For each finding emit a
machine-readable row:

```
CONCERN:: <BLOCKER|CONCERN|NIT> <stable-kebab-case-id> <one-line summary>
```

A BLOCKER is something that would produce a wrong result, a lost artifact, an
uninterpretable verdict, or an unrunnable phase. Cosmetics are NITs. **Say how you
verified each finding** — "read the code" and "ran it" are different evidence and I
will weigh them differently.

If a fix is genuinely correct and well-pinned, say so plainly. A PASS with
reasoning is more useful to me than a manufactured concern; one round-1 shard
returned PASS and it was the right call.
