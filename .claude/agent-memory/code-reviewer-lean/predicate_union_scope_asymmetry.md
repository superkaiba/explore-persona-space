---
name: predicate-union-scope-asymmetry
description: A lint predicate that unions names per-unit must be probed with the SPLIT form of EVERY unit type it covers; unioning try handlers but not with-items leaves split-suppress as an undisclosed evasion
metadata:
  type: feedback
---

When reviewing an AST lint predicate that flags a unit by the UNION of names
it references (caught exceptions, suppress args, decorator args), probe the
SPLIT form of every unit type the predicate covers — live, against the
shipped check via its `roots=`/scan-root test hook, never by reading alone.
Asymmetric union scope is the evasion seam: #2168's `check_json_guard_unicode`
unioned across ALL `except` handlers (so split-handler was covered) but
per-suppress-CALL on `with` items, so
`with suppress(JSONDecodeError), suppress(OSError):` — and the nested-with
form — shipped as probe-confirmed 0-finding misses absent from the docstring's
"complete" disclosed-false-negative list.

**Why:** the plan's §8 risk row claimed "residual evasions = the disclosed
list"; a 3-fixture live probe (one Bash call importing the check with a tmp
root) falsified it in ~30s. Same session also confirmed nested literal tuples
(`except ((A, B),):`) evade terminal-name extraction — exception-tuple
matching is recursive at runtime, so the form is semantically live.

**How to apply:** for each unit type (Try handlers / TryStar / With items /
call args), write the two-unit split of the banned pair plus the nested
variant into a tmp fixture tree and run the shipped check on it. A miss that
the disclosure list names is fine (pin it with a documented-miss fixture, the
#2168 `suppress as quiet` pattern); a miss the list omits is a finding — the
fix is either widening the union to the enclosing statement (mirror the
handler union) or adding the class to the disclosure + fixture. Severity
calibration: 0 live instances + not taught by the error message + deliberate
rewrite required ⇒ CONCERNS (persisted), not FAIL; live sites or a message
that prints the uncovered form as the fix ⇒ FAIL (the #2168 v1→v2 precedent).
