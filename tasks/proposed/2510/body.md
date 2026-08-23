---
title: 'workflow-fix: verify_plan should assert the smoke blind-spot enumeration when
  a plan declares a smoke/dry-run verification'
kind: infra
tags: []
created_at: '2026-08-23T23:17:16Z'
has_clean_result: false
parent_id: 2322
origin_prompt: 'Surfaced by the #2322 Methodology-lens critic as a prose workflow-fix
  follow-up: verify_plan.py has no smoke-blind-spot check, so a plan can declare a
  --dry-run verification and register a success criterion on it with no enumeration
  block and still pass the mechanical pre-pass (it did, on #2322 plan v2).'
workflow: v1
---
# workflow-fix: verify_plan should assert the smoke blind-spot enumeration when a plan declares a smoke/dry-run verification

Surfaced by the Methodology-lens critic during #2322 (its own Must-Fix), then
verified against the code before filing.

## The gap

`.claude/rules/smoke-blind-spots.md` requires any plan declaring a pre-launch
smoke run to carry a `Smoke blind-spot enumeration:` block — or the literal
empty-form escape `none — smoke executes every production gate`. Nothing
mechanically checks that at plan time:

- `scripts/verify_plan.py` has **no** smoke-blind-spot check at all (grep for
  `blind` returns only unrelated hits: "kind-blind", "flying partially blind",
  "Markdown-table blindness").
- `scripts/workflow_lint.py` has two smoke-blind-spot arms, and neither covers
  this: `check_smoke_blind_spot_review_lens` is a **surface pin** (it asserts
  the lens text is present in the reviewer/critic specs, not that any given
  plan complies), and `--check-smoke-blind-spots` **requires
  `--smoke-blind-spot-scripts`** (`workflow_lint.py:21538-21540`), so it is
  opt-in with explicit args and is not in the no-flags default bundle — and the
  rule itself documents it as best-effort WARN-only AST scanning with several
  disclosed false negatives.

So the enumeration duty is enforced only by the LM gates: `critic.md`
Methodology lens item 19 at plan time and `code-reviewer.md` Step 0.71 on the
diff. Those work — item 19 is what caught #2322 — but a cheap mechanical
pre-pass would catch it before a critic round is spent.

## Why it is cheap: the trigger already exists

`verify_plan.py` already detects the exact precondition. Check
`c11_dryrun_test_coverage` ("dry-run smoke backed by a dry-run test",
`verify_plan.py:1526`) fires when a plan names a `--dry-run`
smoke/verification command; it then asserts a *test* exists for the dry-run
kwarg thread. The proposal reuses that same trigger for a second assertion:
the plan must also carry the enumeration block (or the empty-form escape).

Sketch: on the c11 trigger, FAIL/WARN unless the plan body contains the
literal title `Smoke blind-spot enumeration:` or the unwrapped escape line.
Same escape-hatch convention c11 already uses for its own `N/A — no dry-run
smoke` declaration, so the shape is precedented rather than novel.

## The #2322 incident, concretely

#2322 plan v2 named `uv run python scripts/codex_auto_upgrade.py --dry-run` in
its Reproducibility Card AND registered a success criterion on it
("exits 0 and reports no change, with `~/.codex/config.toml` byte-identical").
It carried no enumeration block. `verify_plan.py` returned **PASS, 0 WARN** on
v2 — it had already been satisfied on the c11 axis by the added dry-run test —
while the enumeration was still missing.

The omission was not cosmetic. Under `--dry-run` the current-model probe is
skipped (`scripts/codex_auto_upgrade.py:564`), and that probe is what refreshes
the hourly auth token in-band (module comment `:556-558` — a bare HTTPS call to
the models endpoint just 401s). With a stale token, `fetch_models` catches the
401 as `urllib.error.URLError` and returns `None` (`:270-272`), `main()` logs
"no models listing — leaving model unchanged" (`:578`), `current_broken` is
`False` in dry-run so `failed` stays `False`, and the run **exits 0 having
exercised no probe, no selection, no config write and no kill**. The registered
success criterion was satisfiable by that vacuous path. Writing the enumeration
is what surfaced it; the criterion was then strengthened to require a
listing-backed selection line, which the vacuous path cannot emit.

That is the general argument for a mechanical arm: the enumeration's value is
that writing it forces you to notice what the smoke does not cover, so the
cheapest place to demand it is before the plan is approved.

## Scope / acceptance

- One new `verify_plan.py` check keyed on the existing c11 trigger, with the
  precedented unwrapped-escape form.
- Severity: WARN is defensible for a first landing (c11 itself is WARN); FAIL
  is the stronger posture. The implementing round decides and records why.
- Pin it with a fixture pair: a plan naming a `--dry-run` verification with no
  enumeration (must trip) and one with the block present (must pass), plus one
  with the escape line (must pass).
- Known false-positive class to handle: a plan mentioning `--dry-run`
  incidentally (e.g. quoting a command in prose with no smoke declared). The
  escape line is the release valve; c11's own trigger tuning is the reference
  for how tightly to scope the match.

## Not in scope

Do not widen `--check-smoke-blind-spots` into the no-flags bundle. The rule
documents it as best-effort with disclosed false negatives, and a FAIL-posture
repo-wide arm on a WARN-grade AST scan is the #1388 fleet-wedge shape (the
no-flags lint IS the Step 9c gate).
