---
name: plan-literal-pin-lost-in-generic-wrapper
description: A plan-literal evidence pin ("parent_commit = commit A's returned sha") routed through a generic probe-first wrapper gets silently re-derived at a fresh head — the pin's LOUD-on-interleave property is lost and the covering test passes only because its fixture is quiescent (#2321 R1 g3)
metadata:
  type: feedback
---

When a plan pins an evidentiary commit/step on a SPECIFIC prior artifact
("pinned on the post-A HEAD = commit A's returned sha, so the read is
guaranteed at-cap") and the implementation routes that step through a
generic probe-first/retry wrapper that re-resolves its own pin each attempt,
the plan's guarantee silently degrades: in the quiescent case the fresh pin
equals the required one (so the committed test passes), but an interleaving
concurrent write makes the step issue at the NEW head and record its verdict
on vacuous evidence — where the plan's version would have gone loud
(412/abort/recompute). Diff the plan's literal pin parenthetical against
what the wrapper actually pins; a fixture asserting `parent == expected_sha`
with no interleaving-write arm does NOT pin the contested behavior. Correct
realization is usually a head-EQUALITY check against the required sha with
an abort/recompute route on mismatch — a raw pin alone is insufficient when
the generic wrapper's 412→re-pin cycle would re-issue at the new head
anyway.

**Why:** #2321 R1 g3 cap probe: plan §3.6/C4 pinned commit B on commit A's
returned sha; realized `run_cap_probe` rode `commit_unit_probe_first`, which
pins on a fresh `repo_info().sha`; the A→B window even contains a deliberate
20 s sleep. `test_probe_a_b_c_composition:988` asserted the pin equals the
post-A sha — true only in the quiescent fixture.

**How to apply:** for every plan sentence of the form "pinned on <specific
sha/artifact>", trace the realized call to where the pin VALUE is produced;
if a shared wrapper derives it, add the equality check + a mismatch route,
and demand an interleaving-write fixture. Sibling of
[[registered-gate-quantity-substituted]] (same diff-the-parenthetical review
line; this one is the PIN/evidence variant).
