---
name: reconstructed-marker-compose
description: When the impl marker is ORCHESTRATOR-RECONSTRUCTED (implementer session killed pre-transcription), carry the provenance disclosure, state shape facts neutrally (own headings vs (a)-(d) labels, missing pin/Gate-scope fields), and pre-run the function-grain pre/post differential yourself
metadata:
  type: feedback
---

Compose recipe for a round whose `epm:results` / `epm:experiment-implementation`
marker was posted by the ORCHESTRATOR, reconstructed from durable evidence
(landed commit + message + re-run tests) because the implementer session was
stopped (watcher ALIVE-BUT-STALLED auto-respawn) before transcribing the
returned report (first hit #2294 r1, 2026-08-22):

1. **Carry the brief's provenance disclosure verbatim-in-substance** as a
   binding-context block ABOVE the marker envelope: reconstructed, not
   testimony; every claim verified against diff/repo; a claim that does not
   hold is a finding.
2. **State the marker-shape facts NEUTRALLY, never pre-adjudicate:** a
   reconstruction typically carries the (a)/(b)/(c) CONTENT under its own
   headings but lacks the literal `### (a)`–`### (d)` labels, the (d)
   needs-eyeball section, the ruff-policy pin field, and the
   `**Gate-scope check (#1288):**` line. List exactly what is present under
   which heading and what is absent, remind Codex of per-blocker strip
   keying + Step 0.7 / Rule 8 + present-but-imperfect routing, and let the
   rubric route it. Still emit the GATE-SCOPE THRESHOLD line off the
   marker's real `ts`.
3. **Pre-run the pre-fix/post-fix differential at FUNCTION grain yourself**
   (import the changed function from main's src and the worktree's src via
   `sys.path.insert(0, '<wt>/src')` — print `__file__` to confirm which copy
   loaded) and attest both outputs. This gives Codex a baseline for the
   "does the new test actually fail pre-fix or is it tautological" priority
   without an execution env, and makes the reconstruction's key behavioral
   claim independently yours.
4. **Re-run the round's own test files yourself** when cheap (the worktree
   .venv usually exists — the implementer ran them there); the attestation
   then reads "composer re-ran, independent of the marker's identical claim".
5. Mid-compose gate-result corrections (e.g. a pending lint verdict landing
   as PASS) follow [[established-gates-attestation-compose]]: verify the
   named log yourself (rc line, FAIL count, the round-relevant WARN line
   verbatim) before patching the attestation.

**Why:** the reconstruction is the ONE case where marker-shape absences are
structurally guaranteed (the orchestrator wrote it without the template) —
without the neutral facts + provenance context, an adversarial twin burns
its verdict on mechanical-contract blockers against a document the
implementer never wrote; without the composer-run differential, the round's
central regression-pin question is unverifiable from a read-only sandbox.

**How to apply:** any brief carrying an "orchestrator-reconstructed /
implementer report unrecoverable" provenance disclosure. Related:
[[9ater-followup-round-report-placeholder]] (in-session placeholder variant),
[[missing-impl-marker-probe-checklist]] (marker absent entirely).
