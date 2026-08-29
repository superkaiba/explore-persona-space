# Smoke blind spots — enumerate what a smoke PASS does NOT certify

**Load this rule whenever a plan declares a pre-launch smoke run, a diff
adds/edits a `smoke`-conditional branch, or a test suite replaces a curated
production constant.** A smoke run validates only the code it EXECUTES, and
a test suite validates only the data it RUNS AGAINST. Three mechanisms
silently narrow smoke coverage so the PASS certifies less than it appears
to — and a fourth (test-side) makes a green TEST SUITE certify less than it
appears to:

1. **Substituted implementation** — a `smoke` branch swaps the production
   implementation for a toy (hash-vector embedding, stub model, fake judge),
   so the production import / constructor / API call never runs under
   smoke — including when the production work is wrapped in a module-local
   helper (`model = _load_model(...)`) and the import is invisible at the
   branch site.
2. **Downgraded gate** — an assertion / raise is skipped or weakened when
   `smoke` is set (early-return before the gates, `assert`/`raise` only on
   the production branch), so the smoke "PASSes" gates it never evaluated.
3. **Production-only code path** — a phase, device route, or upload reached
   only on the production branch.
4. **Substituted production constant (test-side)** — a TEST fixture replaces
   a curated module-level production constant
   (`monkeypatch.setattr(<mod>, "<CONST>", fixture_list)` — incl. the
   dotted-string form `monkeypatch.setattr("<mod>.<CONST>", ...)` —
   `mock.patch.object(<mod>, "<CONST>", ...)`, `patch("<mod>.<CONST>", ...)`,
   or a direct fixture assignment) in EVERY test touching it, so the suite
   validates the FUNCTION against fixture data and never certifies the
   shipped constant's CONTENTS — the shipped list can omit the incident
   entry, or be entirely empty, while all tests stay green. This mechanism
   has its OWN trigger grammar — TEST files, a `monkeypatch`/`patch` target
   naming an ALL_CAPS attribute of a non-test module — NOT the production
   `smoke`-conditional branches of mechanisms 1–3: the enumeration idea
   transfers; the trigger does not (#2360).

## Driving incident (#1336, plan v15 round 4)

Two consecutive production SLURM launches died on failures the pre-launch
smoke was STRUCTURALLY INCAPABLE of catching — not too small, DIFFERENT CODE:

- **SLURM 4684** — `embed_prompts()` returned a hash-based 32-dim toy vector
  under smoke; on the production path it called the module-local helper
  `_load_sentence_transformer()`, whose body holds
  `from sentence_transformers import SentenceTransformer` — the import sits
  one call away from the branch site, was never executed by the smoke, and
  the undeclared dependency surfaced as a `ModuleNotFoundError` at the top
  of an 8-GPU production run.
- **SLURM 5005** — `assert_split(..., smoke=ctx.smoke)` downgraded its split
  assertions PER CHECK (`if smoke: logger.info(...) else: raise ...` — no
  early exit); the smoke reported PASS on a split instrument whose
  production gates it had not evaluated, and production died at
  `assert_split`.

## Driving incident, mechanism 4 (#2360, plan v2)

Plan #2360 v2 proposed a preflight check driven by two module-level curated
constants and a 15-test suite: tests 1–4 replaced the first constant with a
fixture list, test 5 replaced the second, and the only test touching the
real constants pinned call ORDERING via a source substring — both shipped
constants could omit the incident entry, or be entirely empty, with every
committed test green. 1 of 6 ensemble reviewers flagged it (the Codex
statistics twin); the fix was the static completeness/subset test below.

## The duty: SMOKE BLIND-SPOT ENUMERATION

The plan section that declares the smoke run states, in one short block
titled `Smoke blind-spot enumeration:`, what the smoke's PASS does and does
NOT certify — one line per item:

- every production gate/assertion the smoke DOWNGRADES or skips;
- every implementation the smoke SUBSTITUTES;
- every third-party import reached ONLY on the production branch;
- every curated production constant the TEST SUITE substitutes
  (mechanism 4), plus the test that pins the REAL constant's required
  contents — or the literal
  `none — no test substitutes a production constant`.

An EMPTY enumeration is written as the literal
`none — smoke executes every production gate` — never left blank. Any
smoke-conditional substitution/downgrade branch in the code FALSIFIES the
empty form. The implementer mirrors the block per phase under `## Smoke run`
when the realized code adds a branch the plan did not anticipate.

The named remedy for mechanism 4 is a static completeness/subset test over
the SHIPPED constant — the instrument #2360 resolved with: an unmapped entry
fails; an empty constant fails; required members pinned. A test that pins
call ORDERING via a source substring, or asserts only on the FIXTURE value,
is NOT contents evidence. And the enumeration duty is
smoke-declaration-gated: for a plan declaring NO smoke run (the #2360 class
itself), the BINDING arms for mechanism 4 are Methodology lens item 20
(plan time) and code-reviewer Step 3.85 (diff time) — named here so the
founding incident's own shape is not left to a vacuous duty.

## Distinct siblings (do not re-dedupe onto them)

#1611 (smoke missing behavior-class × regime CELLS — same code), #1727
(a smoke-valued VARIABLE leaked into production — same code; code-reviewer.md
Step 0.70), #1355 (production-n-calibrated gates KILL the smoke leg — the
inverse direction), #822 (FAIL on a MISSING smoke-architecture-check marker —
presence of a check, not the blind spots of a passing one). Architectural
parity (`planner-section-reference.md` § 4) makes smoke and production the
SAME code; THIS rule covers the residual divergences that survive a justified
parity break or a `smoke`-kwarg branch inside shared code — including
downgrades the gotchas.md smoke/production-parity GATE-CALIBRATION rule
itself SANCTIONS (#1336's `assert_split` docstring cites it: a single-corpus
smoke slice structurally cannot pass production-scale verdicts). A sanctioned
downgrade is still a blind spot: enumeration (disclosure) is the right
instrument exactly where parity (prohibition) is legitimately waived.

## Enforcement

- `planner.md` §4 hard-requirement + `planner-section-reference.md` § 4
  "Smoke blind-spot enumeration" bullet (the plan-side duty).
- `code-reviewer.md` Step 0.71 + the codex twin's copy-list bullet — FAIL
  tagged `smoke-blind-spot-unenumerated` (SUBSTANTIVE, never stripped) on an
  unenumerated smoke-conditional substitution / gate-downgrade in a diff;
  full trigger grammar: `code-reviewer-section-reference.md` § Step 0.71
  detail.
- `code-reviewer.md` Step 3.85 + the codex twin's copy-list stub — FAIL
  tagged `production-constant-unpinned` (SUBSTANTIVE, never stripped) on a
  fixture-substituted production constant an acceptance criterion depends
  on with nothing pinning its real contents (mechanism 4, diff time).
- `critic.md` Methodology lens item 19 (`critic-lens-reference.md`) — REVISE
  a plan declaring a smoke run with no enumeration and no empty-form literal.
- `critic.md` Methodology lens item 20 (`critic-lens-reference.md`) — REVISE
  a plan whose test list replaces a curated production constant in every
  test touching it with no listed real-contents pinning test, naming the
  constant + the dependent acceptance criterion (mechanism 4, plan time).
- Mechanical: `workflow_lint.py --check-smoke-blind-spot-review-lens`
  (region-anchored surface pin, bundled into the no-flags default run) and
  `--check-smoke-blind-spots` (best-effort WARN-only AST scan of scripts vs
  a plan; the reviewer lens is the binding gate). The scanner's KNOWN false
  negatives, disclosed by design: module-local helper resolution is ONE
  level deep (a production import nested two-plus calls down, or wrapped in
  a helper imported from ANOTHER module, escapes), `ast.Match` case bodies
  are not recursed by the statement-form rules (an `ast.If` inside a match
  arm escapes), dynamic dispatch escapes, and smoke flags not literally
  named `smoke` escape — the reviewer lens is the catching arm for all of
  these.
- Mechanical (numeric premises, #2178): `scripts/verify_plan.py` c65 (a plan-claimed
  smoke-fixture row floor vs the realized fixtures — FAIL on a resolved
  overstatement; constant-route contradictions WARN by design) + c66 (WARN: the
  shortfall's producing script named nowhere in the plan). #2165 makes plans
  DISCLOSE smoke divergences; c65/c66 VERIFY the numeric premises.
- Mechanical (import resolvability, #2253): `workflow_lint.py --check-prod-import-lockfile`
  (bundled into the no-flags run) FAILs any scripts//src/ third-party import
  unresolvable from uv.lock/pyproject.toml — branch-agnostic, so the
  "third-party import reached ONLY on the production branch" blind-spot item
  is machine-caught repo-wide.
- Mechanical (mechanism 4, #2364): `workflow_lint.py
  --check-production-constant-pinning-lens` (per-surface token pin across
  the mechanism-4 enforcement surfaces, bundled into the no-flags default
  run) and `--check-monkeypatched-constant-pinning` (best-effort WARN-only
  AST scan over `tests/` for the four patch forms with no repo-wide
  real-contents pinning reference; false negatives disclosed in its
  docstring — the reviewer lens is the binding gate).

## Files of record

Task bodies #2165 (this rule), #1336 (both incident shapes;
`scripts/issue1336_pooled_split.py` on the `issue-1336-fullcorpora` branch);
#2360 (the mechanism-4 driving incident) + #2364 (mechanism 4 — test-side
substituted production constant;
`tests/test_workflow_lint_monkeypatched_constant_pinning.py`);
siblings #1611, #1727, #1355, #822;
`tests/test_workflow_lint_smoke_blind_spots.py` (fixtures reproducing both
shapes, reshaped AND structurally faithful).
