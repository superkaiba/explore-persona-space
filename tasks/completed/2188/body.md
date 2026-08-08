---
title: 'argcheck: a **kwargs splat escapes dest derivation at all three registration
  sites (add_argument / add_subparsers / set_defaults), producing a false-positive
  import-check failure'
kind: infra
tags:
- argcheck-splat-dest
created_at: '2026-08-08T00:48:11Z'
has_clean_result: false
parent_id: 2176
origin_prompt: '#2176 round-1 code-reviewer Minor (b): argcheck.py:109-123 + :188-195
  skip **kwargs splats (kw.arg is None), so a splat-carried dest is a false positive;
  fix sketch = treat a splat as dynamic_dest = True (2 sibling sites)'
workflow: v1
---
## Goal

Make `assert_args_attributes_defined` route a `**kwargs` splat in `add_argument(...)` / `set_defaults(...)` to its existing PERMISSIVE path instead of skipping the call, so a splat-carried `dest` cannot produce a false-positive `SystemExit` at a driver's `--import-check` gate.

## The gap

`src/explore_persona_space/orchestrate/argcheck.py` derives its DEFINED set by reading `add_argument` option/positional strings plus an explicit `dest=` kwarg. **Three** sibling sites walk the call's keywords and skip a `**` splat: `_add_argument_dests` (`:109–123`, via `kw.arg != "dest": continue`), the `add_subparsers` handler (`:188–195`, whose `kw.arg == "dest"` test is never true for a splat), and `set_defaults` (`:197`, whose `if kw.arg is not None` filter drops splats outright). A splat that carries `dest` (`ap.add_argument("--x", **{"dest": "y"})`, or a shared `**COMMON_OPTS` dict) therefore contributes NOTHING to the DEFINED set, while the module's later `args.y` read still lands in the REFERENCED set. Result: `y` is reported as never-defined and the helper raises `SystemExit`, failing a driver's own import-check on correct code.

Surfaced by the #2176 round-1 code review (`epm:code-review v1`, 2026-08-08T00:41:43Z) as a non-blocking Minor, with the fix sketch already written.

## Fix

Treat a `kw.arg is None` splat the same way a NON-CONSTANT `dest=` is already treated: set the existing `dynamic_dest` flag so the call falls through to `_permissive_dests(opts)` rather than contributing an exact-derived name (or nothing). That is the precise shape #2176's round-3 commit `c4f681716b` established for the unresolvable-`dest=` case, so this is extending an existing, tested branch to a second unresolvable input — not new machinery.

Both sibling sites take the same change. A CONSTANT `dest=None` deliberately stays on exact derivation (statically resolvable — argparse's derive-the-default behavior); only genuinely unresolvable inputs go permissive.

## Why this is low urgency but worth landing

Measured exposure on the tree at filing time: **zero** real call sites. `grep -rn 'add_argument(\*\*\|add_argument(\*' --include='*.py' scripts/ src/` and the `set_defaults(\*\*` equivalent each return exactly ONE hit, and both hits are `argcheck.py`'s own docstring prose, not a call. So nothing in the repo trips this today.

It is still worth fixing rather than dropping, for two reasons. The failure direction is a false POSITIVE — loud, with a documented `extra_defined=(...)` escape at the call site — which is the safe direction and is why it did not block #2176's review; but #2176 is the task that introduces the `--import-check` convention fleet-wide via `.claude/rules/code-style.md`, so the first NEW driver to adopt the convention with a shared-options dict is the one who hits it, and a brand-new convention that fails on correct code is the kind of friction that gets the convention abandoned rather than debugged.

Distinct from the docstring's already-accepted `AugAssign` false NEGATIVE (`args.x += 1`), which is deliberately left alone — that one would need flow ordering.

## Acceptance

1. Both sites route a `**` splat to the permissive path.
2. A test pins the BITE, not just the pass: a fixture whose `dest` arrives only via a splat must FAIL against the pre-fix `argcheck.py` and pass after. Pin it the way #2176's `test_dynamic_dest_kwarg_takes_permissive_fallback` was pinned — with ≥2 option strings so exact-derivation and permissive derivation yield DIFFERENT sets; a single-option fixture cannot discriminate the two implementations and is a decorative pin.
3. `tests/test_argcheck.py` stays green (17 tests at filing time).
4. The docstring's limitations list reflects the new handling.

## Provenance

Parent #2176 (`kind: infra`) — the task that built `argcheck.py`. Finding from its round-1 `code-reviewer` verdict, which PASSed with 0 blockers and recommended merge; this was Minor (b) of two. The other minor was a marker-prose paste discrepancy (131 of 132 pin-sweep entries listed), discharged in-round by running the omitted test.

## Scope correction (applied at Step 10; prose-only, no Takeaway/classification change)

This body was filed naming **two** sites, "`add_argument` / `set_defaults`". The clarifier's context pass found that inventory wrong in both directions, and the corrected scope — **all three** registration sites — is what was planned, implemented, and reviewed.

- The line numbers this body cited (`:109–123` and `:188–195`) map onto `_add_argument_dests` and the **`add_subparsers`** handler. #2176's `epm:code-review v1` names exactly those two ("2 sites ... `_add_argument_dests`, the `add_subparsers` dest handler"), so the original prose substituted `set_defaults` for `add_subparsers`.
- `set_defaults` (`:197`) is nonetheless an **independently splat-blind third site** — its `if kw.arg is not None` comprehension filter drops splats, so `ap.set_defaults(**DEFAULTS)` contributes nothing to DEFINED. That site was missed by the parent review's own bug-class sweep, which recorded "2 sites". The mistaken prose therefore happened to name a real, previously-unrecorded bug; `set_defaults(**SHARED)` is also the most natural real-world adoption shape of the three.
- Recorded on #2176 as well, so its sweep count is not carried forward as complete.

Acceptance criterion 1 ("both sites route a `**` splat to the permissive path") was satisfied on its INTENT — a splat is never silently skipped into a false positive — not its letter: `add_subparsers` and `set_defaults` have no option strings to be permissive over, so "the permissive path" is mechanically undefined there. Those two sites take resolve-or-degrade instead (statically resolve a clean dict literal under an exclusive-use rule; otherwise contribute nothing AND name the splat in the `SystemExit` diagnostic alongside the `extra_defined=(...)` escape). The deviation was pre-registered in plan §7/§12.2 and reviewed; criteria 2, 3, and 4 were met to the letter.
