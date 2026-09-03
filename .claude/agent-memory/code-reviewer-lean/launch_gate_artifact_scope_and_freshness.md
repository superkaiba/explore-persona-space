---
name: launch-gate-artifact-scope-and-freshness
description: Launch-or-park gate harness review — feed each _gate_* a SUBSET-scoped artifact (its own scope field unread ⇒ PASS on 1-of-N coverage) and grep every fingerprint field for a consumer (recorded-but-never-compared = stale-artifact LAUNCH)
metadata:
  type: feedback
---

Two probes for any launch-or-park / spend-authorization gate harness that
evaluates DISK ARTIFACTS (issue #2658 unit 8, `scripts/issue2658_power.py`):

1. **Subset-scope probe.** Build the certified artifact exactly as a
   legitimate `--rows <one>` (or any subset-narrowing flag) production run
   would emit it — self-reported `status: measured`, honest scope fields
   (`rows_simulated`, `rows_missing_labels` computed only over the passed
   subset) — and call the gate on it directly. The #2658 power gate PASSed a
   1-of-11-row selection because it read only `registered_match`/`status`/
   `n_common` and never compared `rows_simulated` to the registered row
   universe, despite an in-function comment claiming "the common N must cover
   EVERY registered row" (the completeness check was scoped to the profiles
   PASSED, which the CLI flag narrows).

2. **Fingerprint-consumer grep.** `grep -n '<fingerprint field>'` over the
   module: every artifact RECORDED a `profile_sha256` and the gate phase
   loaded two of them from disk plus the live profiles in the same function —
   and compared none of them. Sibling of [[pilot_pass_report_fingerprint_unchecked]]
   and [[registered_gate_quantity_substituted]].

**Why:** both holes assemble a LAUNCH from sanctioned CLI invocations only —
the freshness one on the NATURAL workflow (judge cells regenerate when human
adjudications land; re-running `--phase gate` alone avoids a ~13 h sim
recompute), so "an operator wouldn't do that" is not a defense.

**How to apply:** whenever a diff adds a gate that consumes self-describing
artifacts, (a) run the gate live on a subset-scoped artifact mirroring what
the narrowing flag emits, (b) grep each recorded fingerprint/scope field for
a comparing consumer; recorded-but-never-compared is the FAIL shape.
