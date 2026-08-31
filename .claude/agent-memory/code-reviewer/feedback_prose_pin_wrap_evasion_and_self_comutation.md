---
name: prose-pin-wrap-evasion-and-self-comutation
description: Reviewing phrase-absence/docstring grep pins - sweep banned phrases whitespace-insensitively (line-wrapped instances evade literal greps in the SILENT direction) and mutate first-occurrence-only (a global sed co-mutates the pin's own expectation and false-passes) (#2387 r12)
metadata:
  type: feedback
---

Two traps when verifying a prose pin (a test that greps a file/docstring for
banned phrases or required entries), both hit in #2387 round 12:

1. **Wrapped-spelling evasion (silent direction).** A literal `phrase in text`
   pin is defeated by the SAME phrase re-wrapped across lines
   ("parses\n    clean") — and hand-wrapped docstrings make that the LIKELY
   recurrence spelling: the r10 historical instance the #2387 pin was built
   against was itself line-wrapped, so the shipped pin would not have caught
   its own motivating case. Also biases history forensics: a `git show | grep
   'parses clean'` of past tips MISSES wrapped instances — I first concluded
   "r11 fixed zero locations" when it had fixed the wrapped one.
   **How to apply:** for every banned-phrase pin, run a `re.search(r"w1\s+w2")`
   whitespace-insensitive sweep of the current file AND of any historical tips
   you grep; recommend the pin itself use `\s+`. Required-entry pins
   (member-name greps) fail the LOUD way on rewrap — brittle but sanctioned
   direction; only absence-pins have the silent gap.

2. **Self-co-mutation false-pass.** Mutation-testing a pin with a GLOBAL
   rename (`sed 's/old-name/new-name/'`) also rewrites the pin's own expected
   string, so the mutated pin greps for the mutated name and PASSES — the
   mutation matrix reads "pin cannot fail" or (worse) "pin passed under
   deletion". **How to apply:** mutate only the TARGET occurrence
   (`str.replace(old, new, 1)` on the docstring instance; count occurrences
   first), keep the pin's expectation bytes untouched, and keep the benign
   baseline control ([[mutant-matrix-needs-negative-control]]).

**Why:** both directions corrupt the "do the pins actually pin" verdict the
orchestrator relies on for claim-discipline rounds; #2387 r12's only real
residual (persisted CONCERN `probe-soundness-pin-wrapped-spelling-evasion`)
was found by check 1, and my first mutation-A run was invalidated by trap 2.
