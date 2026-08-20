---
name: parameter-value-domain-widening
description: A rework that WIDENS a keyed parameter's value domain (batch "gate"|"rest" -> batch_id "gate_{cell}") leaves old-domain readers silently dead — symbol-rename greps never fire because no symbol changed; sweep every comparison/glob/prefix keyed on the OLD values (#2389 R2 g2)
metadata:
  type: feedback
---

When a diff re-grains a naming/keying scheme by WIDENING a parameter's value
domain while keeping the parameter NAME (`batch` stays `batch`, values go
`"gate"` → `"gate_{cell}"`; file names go `anchors_gate_w{w}` →
`anchors_gate_{cell}_w{w}`), the Step 3.75 symbol-rename grep is structurally
blind — no symbol was renamed. Sweep instead by OLD-DOMAIN VALUE:

1. `grep -n '== "gate"\|== "rest"\|startswith("gate' <file>` — every equality
   /prefix test on the old literal values. #2389 R2 g2: `"gate_slice": batch
   == "gate"` went permanently False for all HF rows → durable schema lie +
   a `_validate_breach_basis` mixed-bucket wedge once the sibling bug was
   fixed.
2. `grep -n 'glob('` filtered to the artifact family — every glob whose
   literal prefix encodes the OLD name grammar. Same round:
   `glob("anchors_gate_w*.jsonl")` matches ONLY retired names → the plan's
   registered cap-recalibration silently aggregated ZERO rows and wrote a
   valid-looking empty artifact (empty-selection-should-raise missing).
3. Certify each hit mechanically: `fnmatch(new_name, old_pattern)` one-liner
   + parent-blob grep (`git show <sha>^:<file>`) proving the old semantics
   were correct — that pins the finding as a regression OF this commit.

**Why:** both #2389 R2 Criticals were this one class; the commit's own
docstrings were already updated (`anchors_gate_*.jsonl`) while the code kept
the old pattern — docstring/code divergence inside one hunk is the tell.
The two defects MASKED each other (dead recal ⇒ no mixed caps ⇒ wedge
latent), so fix-one-ship-one would have converted silent loss into a hard
wedge. Related: [[amend-phase-striding-filters]] (same rework family),
[[frozen-decision-adopt-bypasses-arming-guard]] (same round, g1's lesson).

**How to apply:** any diff whose commit message says "re-grain", "rename
shards", "per-cell/per-X sharding", or that edits an f-string building
artifact names: run sweeps 1-2 over EVERY file that reads the artifact
family (producer script + judge/analysis/figure forks), not just the file
edited. Also check the round's tests exercise the AGGREGATION over new-name
fixtures, not just the producers — an analytic re-model of the chunker/
aggregator passes while the real reader is dead.
