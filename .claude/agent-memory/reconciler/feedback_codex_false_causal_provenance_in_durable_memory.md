---
name: codex-false-causal-provenance-in-durable-memory
description: Codex FAIL on a false causal/provenance claim written into durable agent memory — verify from retained sweep artifacts (TSV byte-identity, own-run hits file, tool no-git contract); facts upheld, severity keys on remedy shape (text-only fix + valid payload ⇒ PASS + persisted CONCERN with exact corrected text)
metadata:
  type: feedback
---

Pattern (#2321 r4, code-reviewer split, PASS vs FAIL): Codex's sole Major
challenged not the production path but the CAUSAL EXPLANATION the round
recorded for a count reconciliation — a new implementer memory + the
`epm:results` report claimed two selector sweeps "produced 158- and
157-file hit sets" because `origin/main` moved between them. Claude PASSed
after reproducing the fresh 157-set — which verified the SET but never
tested the CAUSAL STORY. The split was ADDITIVE (one reviewer examined a
question the other didn't), not contradictory.

**How to verify this class — go to the retained artifacts, all four legs:**
1. Byte-identity of the two sweeps' retained outputs (`cmp` + sha256 —
   identical TSVs refute any "two objects" claim outright).
2. The disputed round's OWN contemporaneous derived file (r3's hits file
   already held the 157-set — the divergent object was the hand-pasted
   fence, which mismatched its OWN same-run TSV 26/25).
3. The measured input delta (`comm` on the difflists) vs the output delta
   (here: 2 input files, zero output change).
4. The tool's mechanism contract (`select_step9c_tests.py` mapping mode
   "never runs git — no fetch" ⇒ a moving base structurally cannot alter
   `--map-files` output for a fixed difflist).

**Severity calibration:** facts UPHELD in full (Codex right; Claude's PASS
evidence was consistent with — even predicted by — the stale-fence
reading). But severity keys on REMEDY SHAPE: valid payload passed by both
families + text-only correction (memory edit + superseding marker note)
⇒ PASS with the finding Non-blocking-persisted (CONCERN in concerns.jsonl)
AND the exact replacement text in the reconcile marker so no re-roll round
is needed. A FAIL would have been the fifth bounce with zero risk
reduction — the over-strict-blocker calibration error this task's r2
reconcile already named ([[codex-local-artifact-forgery-as-blocker]]).
Do NOT wave it through either: a false causal rule in durable cross-task
memory teaches future audits to excuse mismatches as "two legitimate
objects" instead of recounting the fence against its own TSV — persist +
supply corrected text, always.

**Why:** verdict prose gates nothing (#509/#715); and a "reconciled"
ledger target whose mechanism is false is not truthfully reconciled — the
CONCERN + exact-text channel is what makes PASS safe here.

**How to apply:** any FAIL grounded on provenance/causal claims about
selector counts, sweep membership, or artifact lineage — re-derive from
retained raw outputs under the plainest comparison (byte identity first),
check the disputed record against its OWN contemporaneous artifacts, and
read the tool's mechanism contract before crediting any
environment-drift story.
