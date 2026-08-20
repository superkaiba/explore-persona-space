---
title: 'verify_plan c20: a decimal-threshold verdict lattice can never become FAIL-capable
  (integer thresholds SKIP, decimals WARN unclearably)'
kind: infra
tags: []
created_at: '2026-08-19T21:28:15Z'
has_clean_result: false
origin_prompt: 'Surfaced by the /issue 823 orchestrator during Phase 1.5.0 of the
  inconsistent-origin-persona-ladder follow-up round: plan v7/v8 carried c20_verdict_lattice_coherence
  as an unclearable WARN. Mechanism read from the checker source — _C20_POINT_RE rejects
  a decimal RHS via its 0(?!\.?\d) lookahead, and _c20_has_threshold_atom continues
  past any thr startswith(''0.'') as ''a sign atom'', so a decimal threshold belongs
  to neither atom family and the tier-1 SKIP never fires.'
workflow: v1
---
# verify_plan c20: a DECIMAL-threshold verdict lattice can never become FAIL-capable

## Goal

Close the gap in `scripts/verify_plan.py` check `c20_verdict_lattice_coherence` whereby a verdict lattice whose thresholds are DECIMALS (`delta_mean > 0.05`) earns a WARN that no phrasing can clear, while an otherwise-identical lattice with an INTEGER or FRACTION threshold (`rung >= 5`, `>= 8/9 pairs`) is cleanly SKIPped as outside the v1 cell algebra. Effect-size thresholds in this project are decimal essentially without exception, so the current behavior taxes exactly the common case.

## Evidence (mechanism, read from the checker source)

Two atom families, and a decimal right-hand side belongs to neither:

- `_C20_POINT_RE = re.compile(r"(?P<qty>[^\s,;()]+)\s*(?P<cmp>≥|>=|≤|<=|>|<)\s*0(?!\.?\d)")` — the SIGN-atom family. Its `0(?!\.?\d)` lookahead deliberately rejects `0.05`, so a decimal-threshold comparison is not a sign atom.
- `_c20_has_threshold_atom` — the non-zero THRESHOLD-atom detector. It iterates `_C20_THRESHOLD_RE` matches and `continue`s on any `thr` that `== "0"` or `startswith("0.")` or `startswith("0/")`, with the comment that these "are sign atoms and stay with `_C20_POINT_RE`". So a decimal-threshold comparison is not a threshold atom either.

Net effect: the conjunct falls between the two families, `_C20_RESIDUE_RE` then reports the comparison operator as "predicate token(s) outside every recognized atom", and `_c20_parse_predicate` marks the segment `unparsed`. Because `_c20_has_threshold_atom` returns False, the tier-1 SKIP path (which exists precisely to excuse lattices outside the v1 algebra) never fires — so a decimal lattice WARNs where an integer lattice SKIPs.

## Reproduction

Task #823, plan v7/v8 (`tasks/*/823/plans/v8.md`). Lattice line, in the required explicit partition form with named quantities and comparison-only atoms:

`DISJOINT and exhaustive: Degrades ⇔ delta_mean > 0.05 AND ci_low_delta_mean > 0; Flat ⇔ delta_mean >= -0.03 AND delta_mean <= 0.03; Intermediate ⇔ otherwise.`

`verify_plan.py --issue 823` returns overall PASS with the single WARN `c20_verdict_lattice_coherence`, detail: "label 'Degrades' ... predicate token(s) outside every recognized atom: '>'". The `ci_low_delta_mean > 0` conjunct parses fine (literal-0 right-hand side); only the decimal conjunct fails. An earlier revision of the same lattice, which spelled the CI condition as prose ("the 95% CI excludes 0"), produced the same WARN naming three residues `'>'`, `'CI'`, `'excludes'` — so the CI-idiom half was genuinely fixable by the plan and this half is not.

Note the asymmetry is inside ONE label: rewriting the lattice cannot help, because the threshold is the scientific criterion.

## Why not just reword the plan

The only plan-side silencers are (a) restating the effect-size threshold as a derived compared-against-zero excess variable (`delta_mean_excess = delta_mean - 0.05`, then `delta_mean_excess > 0`), or (b) abandoning decimal thresholds. Both degrade the legibility of the scientific criterion for a human reader in order to satisfy a linter, which inverts the check's purpose. #823 therefore CARRIED the WARN with this mechanism as its recorded disposition (plan v8 `## WARN dispositions`), and filed this task so the next effect-size lattice does not repeat the analysis.

## Suggested direction (not prescriptive — the implementing session should design it)

Make the decimal case behave like one of the two families it currently falls between. Two candidate shapes, both consistent with the existing v1 algebra:

1. **Treat a decimal threshold as a threshold atom** — drop the `startswith("0.")` exclusion in `_c20_has_threshold_atom` so a decimal lattice takes the SAME tier-1 SKIP path an integer lattice already takes. Smallest change; makes the check consistently silent rather than consistently noisy on this shape. Requires confirming the exclusion is not load-bearing for a sign-atom case that WOULD otherwise be captured (the comment says its purpose is to keep sign-atom scope with `_C20_POINT_RE`, which only ever matches literal `0` — so decimals appear to be excluded from both by oversight rather than by design).
2. **Admit a decimal-threshold sign atom** — generalize the point-atom family to `qty <cmp> <decimal>` and carry the threshold into the cell algebra, so a decimal lattice becomes genuinely FAIL-capable (disjointness/exhaustiveness actually checked against the thresholds) rather than merely quiet. Strictly more valuable and strictly more work; needs care that two labels with different decimal thresholds on the same quantity enumerate correctly.

Option 1 restores consistency; option 2 delivers what the check advertises. A reviewer should decide which, with a test pinning the #823 lattice above as a fixture either way.

## Acceptance criteria

1. The #823 v8 lattice line (quoted verbatim above) either SKIPs or PASSes `c20_verdict_lattice_coherence` — it no longer WARNs.
2. A regression test pins that exact line as a fixture, so the shape cannot silently regress.
3. Integer- and fraction-threshold lattices (`rung >= 5`, `>= 8/9 pairs`) keep their current disposition — no behavior change for the shapes the check already handles.
4. A genuinely INCOHERENT decimal lattice (overlapping labels, or a gap with no `otherwise`) is still caught if option 2 is taken; if option 1 is taken, state explicitly in the check's docstring that decimal lattices are SKIPped rather than verified, so the limitation is documented rather than silent.
5. `uv run python scripts/verify_plan.py --issue 823` and the `tests/test_verify_plan.py` suite both green.
