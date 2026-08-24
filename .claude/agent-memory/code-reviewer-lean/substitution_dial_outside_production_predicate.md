---
name: substitution-dial-outside-production-predicate
description: "Enumerate EVERY smoke/stand-in dial (--tiny-model, --sae-dict, --skip-upload) and check each against the production predicate + resume regime; one guarded sibling dial proves intent and makes unguarded siblings findings (#2476 R1 g3)"
metadata:
  type: feedback
---

Rule: when a driver has a production predicate (`_production(args)` keyed on some
flag subset) plus multiple smoke/substitution dials, ENUMERATE every dial and
check each one for BOTH bindings — (a) a production-refusal guard (in the
predicate itself, or an `assert not production` at the substitution site), and
(b) membership in the resume-regime key when it is output-affecting. Do not stop
at the dial the implementer guarded: the guarded sibling PROVES the intent and
turns each unguarded sibling into a finding, not a design choice.

**Why (#2476 R1 g3, 2026-08-22):** `scripts/issue2476_turnavg_sae.py` guarded
`--sae-dict` perfectly (`assert not production, "smoke-only"`, test-pinned) but
left two siblings open in the SAME commit:

1. `--tiny-model` absent from `_production()` and never asserted → a production
   `--phase all --tiny-model` run substitutes BOTH the model and the chanind
   dictionaries (`_TinyJumpReLUStandin`) while staying production-classified:
   gates record `SKIPPED-tiny-model`, the fitness gate demotes the arm for the
   WRONG reason, and the terminal leg pushes stand-in-derived artifacts to
   git+HF under a production `epm:results` sentinel. Grep recipe: for each
   `action="store_true"` substitution flag, grep for it inside `_production` AND
   for a nearby `assert not production` — either binds it; neither = Major.
2. `--skip-upload` output-affecting for the terminal phase (production run skips
   the HF leg) but NOT in `_regime` and the done-file is still written → a later
   production re-run WITHOUT the flag resume-skips past the upload forever at
   the same code SHA. The SAME driver's earlier upload phase got this right
   (verify-fail/skip → NO done-file) — check the terminal phase against the
   pattern the early phase established. Sibling of
   [[force-flag-not-reaching-resume-state]].

**How to apply:** any experiment-driver review with a `_production()`-style
predicate. Build the dial list from argparse (`store_true` flags + int dials with
a 0=production default), then check each against (a) the predicate/assert and
(b) the regime dict. Also check the terminal phase's skip branches write NO
done-state that a resume predicate honors.
