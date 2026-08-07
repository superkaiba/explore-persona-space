# Smoke blind spots — enumerate what a smoke PASS does NOT certify

**Load this rule whenever a plan declares a pre-launch smoke run, or a diff
adds/edits a `smoke`-conditional branch.** A smoke run validates only the
code it EXECUTES. Three mechanisms silently narrow smoke coverage so the
PASS certifies less than it appears to:

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

## The duty: SMOKE BLIND-SPOT ENUMERATION

The plan section that declares the smoke run states, in one short block
titled `Smoke blind-spot enumeration:`, what the smoke's PASS does and does
NOT certify — one line per item:

- every production gate/assertion the smoke DOWNGRADES or skips;
- every implementation the smoke SUBSTITUTES;
- every third-party import reached ONLY on the production branch.

An EMPTY enumeration is written as the literal
`none — smoke executes every production gate` — never left blank. Any
smoke-conditional substitution/downgrade branch in the code FALSIFIES the
empty form. The implementer mirrors the block per phase under `## Smoke run`
when the realized code adds a branch the plan did not anticipate.

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
- `critic.md` Methodology lens item 19 (`critic-lens-reference.md`) — REVISE
  a plan declaring a smoke run with no enumeration and no empty-form literal.
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

## Files of record

Task bodies #2165 (this rule), #1336 (both incident shapes;
`scripts/issue1336_pooled_split.py` on the `issue-1336-fullcorpora` branch);
siblings #1611, #1727, #1355, #822;
`tests/test_workflow_lint_smoke_blind_spots.py` (fixtures reproducing both
shapes, reshaped AND structurally faithful).
