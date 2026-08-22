---
name: parity-gate-fed-own-subject-file
description: A reuse/parity gate that fetches its evidence by passing ONLY its own subject file to a filter that excludes that file class makes the check vacuous-false — trace each gate check's INPUT provenance (#2477 R1 g1)
metadata:
  type: feedback
---

Rule: for every multi-check reuse/parity gate, trace WHERE each check's input
comes from and confirm the fetch can actually surface the evidence the check
needs. #2477 g1: `_bank_parity_gate` called
`_sidecar_payloads_by_dir([(cand, 0)])` — the sidecar parser filters to
meta-sidecar `.json` files, and the candidate is by construction a
completion bank (classified bank ⇔ NOT meta-sidecar), so the payloads dict
was ALWAYS empty and recipe-parity check (ii) unconditionally failed →
`skip_generation` unreachable. The needed input was the candidate's
DIRECTORY listing (available in the inventory manifest), not the candidate
itself.

**Why:** this is the fail-CLOSED twin of the classic hollow gate
([[banked_parent_dual_schema_equivalence]], code-style "hollow verification
gate" always-PASSes): here the gate always FAILS, so nothing corrupts — it
silently wastes the spend the gate existed to save (fresh GPU generation
despite a conforming banked artifact). Severity calibration: Major +
persist-as-concern, not Critical/FAIL, when the branch is provably dead on
the realized data (zero candidates) and the failure direction is the plan's
declared safe default.

**How to apply:** on any gate with per-check evidence fetches, read the
fetch's argument list against the filter predicates inside the fetcher; a
subject-only argument into a filter that excludes the subject's class is the
tell. Certify with a 15-line synthetic-pass probe (conforming candidate +
sidecar ⇒ gate must pass) — the one test the diff never ships. Related:
[[registered_gate_quantity_substituted]],
[[smoke_fixture_authored_with_consumer_keys]].
