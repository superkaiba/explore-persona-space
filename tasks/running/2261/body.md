---
title: Extend driver --import-check to bind-check in-repo helper call arity, not just
  args.<attr> completeness
kind: infra
tags: []
created_at: '2026-08-13T01:20:33Z'
has_clean_result: false
parent_id: 2223
origin_prompt: 'found while reusing hub._upload on #2223: the driver''s own call site
  had wrong arity on a path no gate executes'
workflow: v1
---
# Extend the driver `--import-check` AST convention to bind-check in-repo helper CALL ARITY, not just `args.<attr>` completeness

## Goal

Generalize the existing `orchestrate/argcheck` + `--import-check` convention so a
per-issue driver's `--import-check` ALSO verifies that its calls into shared in-repo
helpers (`hub.*`, `HfApi().*`, `hf_hub_download`, and peers) actually BIND against
those helpers' installed signatures. Today `--import-check` covers argparse-attribute
completeness only, so a wrong-arity call to a shared helper ships green through ruff,
`workflow_lint.py`, the mapped Step 9c tests, the code-review ensemble, AND the
pre-launch smoke — and fires only when that line is first executed.

## The gap

`.claude/rules/code-style.md` § "Argparse-attribute completeness for phase-dispatch
drivers" established exactly the right shape for a statically-detectable, late-firing
driver bug class: a whole-module AST walk behind `--import-check`, wired into the
smoke-architecture contract's Axis 1 so it runs fail-loud pre-dispatch and is recorded
in the marker's `import-resolution:` line. Its scope is deliberately narrow —
`args.<attr>` reads vs the parser's defined set.

CALL ARITY into shared helpers is the same class and is NOT covered:

- statically detectable from the same AST walk the helper already performs,
- invisible to `ruff` (no arity checking) and to every existing lint,
- invisible to tests whenever the call sits on a path tests do not execute,
- late-firing, because the crash needs the line to actually run.

The two classes differ only in WHAT is bound (parser attrs vs callee parameters).

## Measured instance (#2223, 2026-08-13)

`scripts/issue2223_drift.py:2509`, inside `phase_upload`:

```python
url = hub._upload(at, f"{HF_EXPERIMENT}/analysis_tensors", repo_type="dataset")
```

`hub._upload` (`src/explore_persona_space/orchestrate/hub.py:1490`) is
`_upload(local_path, repo_id, repo_type, path_in_repo, ...)` — FOUR required
positionals. The call passes the destination PREFIX where `repo_id` belongs and omits
`path_in_repo` entirely, so it raises

```
TypeError: _upload() missing 1 required positional argument: 'path_in_repo'
```

Reproduced live against the installed module. Two independent defects in one line: the
arity error, and a `repo_id` that would have written to the wrong repo had it bound.

**Why it survived every existing gate.** `phase_upload` early-returns under `--smoke`
("NEVER push the smoke tree to canonical HF"), so the phase has ZERO pre-production
coverage — a production-only path in the sense of `.claude/rules/smoke-blind-spots.md`.
Note this is NOT a smoke-blind-spot ENUMERATION failure: enumeration is disclosure, so
even perfect compliance would have disclosed the gap without catching the bug. Only a
bind check catches it.

**Cost had it not been caught by inspection.** `analysis_tensors` is written at
`issue2223_drift.py:1526` (per-arm, Phase B) and `:2375` (phaseA), so `at.exists()` is
always True on a full run and the branch always executes. `phase_upload` runs at
launcher lines 186/205 — i.e. after both legs' full generate/aggregate/PhaseB compute
(~70+ GPU-h booked, 8 GPUs across two pods). `upload_raw_completions_to_data_repo`
runs first and is correct, so raw completions land and then the phase dies before
`analysis_tensors` persists: the #521 class exactly — plan-referenced downstream
inputs whose loss makes planned controls permanently unrunnable, discovered only after
the pod is due for teardown.

Caught pre-execution only incidentally: an orchestrator reusing the same helper for an
unrelated upload hit the identical `TypeError` first (the reuse-before-building rule
paying off sideways), with hours of runway left to fix and pull.

**Scope evidence — one bad line, not a driver-wide problem.** A mechanical bind audit
of every Hub/HF call site in that driver (lines 717, 718, 743, 744, 2501, 2512) found
the single mismatch; all others bind. So the argument here is NOT "this driver is
broken" but "the class is undetectable by construction and one instance nearly cost
~70 GPU-h of artifacts."

## Proposed change (design to be settled by the spawned session's planner)

1. Add a bind-check pass to `src/explore_persona_space/orchestrate/argcheck.py` (or a
   sibling module): given a driver file, AST-walk its calls to a REGISTERED set of
   shared in-repo callables, resolve each to the installed callable, and
   `inspect.signature(fn).bind(...)` the literal call shape, failing loud on mismatch.
2. Wire it into the same `--import-check` entry point the argparse pass uses, so
   adoption is one already-conventional line per driver and the
   smoke-architecture Axis 1 `import-resolution:` marker line records it.
3. Seed the registered callable set with the persistence-critical surface, where
   late-firing is most expensive: `hub._upload`,
   `hub.upload_raw_completions_to_data_repo`, `hub._upload_folder_filtered`,
   `HfApi().upload_file/upload_folder`, `hf_hub_download`.
4. Keep the CONVENTION posture of the argparse precedent — driver-local opt-in via
   `--import-check`, NOT a repo-wide FAIL lint. That precedent is explicit about why
   (a FAIL-posture repo-wide check is the #1388 fleet-wedge shape, since the no-flags
   lint IS the Step 9c gate; a WARN-posture one adds standing advisory noise). Do not
   regress that reasoning.
5. Honor the argparse pass's hard-won lessons: WHOLE-MODULE scope (a per-function
   scope is escapable by moving the call one level deeper), and handle
   partially-applied / wrapped calls (e.g. `hub.retry_transient(lambda: ...)`, where
   the inner call is the one to bind) rather than silently skipping them.

## Acceptance

- A fixture driver whose `hub._upload` call omits `path_in_repo` FAILS
  `--import-check` with a message naming the call site line and the missing parameter.
- A fixture driver whose calls all bind PASSES.
- Wrapped/deferred call shapes (`retry_transient(lambda: HfApi().upload_file(...))`)
  are bound, not skipped — or their skip is explicit and documented, never silent.
- Whole-module scope is pinned by a test where the offending call sits in a helper the
  phase calls, mirroring
  `tests/test_argcheck.py::test_whole_module_scope_catches_helper_escape`.
- Keyword-only, `*args`/`**kwargs`, and `dest=`-style aliasing surfaces do not produce
  false positives; any residue routes through an explicit visible escape, as the
  argparse pass does with `extra_defined=`.
- `.claude/rules/code-style.md` § "Argparse-attribute completeness" is extended (not
  duplicated) to name the arity pass, and states the measured false-positive/negative
  classes.

## Related

- `.claude/rules/code-style.md` § Argparse-attribute completeness for phase-dispatch
  drivers (#2163) — the convention being generalized, and the source of the
  whole-module-scope and FP-class lessons.
- `.claude/rules/smoke-blind-spots.md` (#2165) — why disclosure alone cannot catch
  this; `phase_upload`'s `--smoke` early return is the production-only path.
- `.claude/rules/upload-policy.md` + #521 — the artifact class at risk.
- #1388 — why the repo-wide FAIL-lint posture is rejected.
- #2223 `scripts/issue2223_drift.py`, fix commit `1b9ae765b6a2833d3f7af937af0dcc823d099f7e`
  (includes `tests/test_issue2223_hub_call_signatures.py`, a per-issue instance of the
  proposed check — the natural prior art to generalize).

## Provenance

Found by the #2223 orchestrator on 2026-08-13 while reusing `hub._upload` for an
unrelated quarantine upload: the same `TypeError` surfaced, and reading the signature
revealed the driver's own call site carried it on a path no gate executes. Diagnosed
by direct signature inspection plus a live reproduction, not inferred. #2223 itself
needs no further change — the call site is fixed, tested, pushed, and pulled onto both
live pods with byte verification. The durable defect is that nothing in the workflow
surface can catch the NEXT instance of this class.
