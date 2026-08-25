---
name: podcrash-difffile-crashfix-compose
description: "Crash-fix round where the crash record's own hypothesis is DISPROVEN by a separate orchestrator root-cause marker AND the fix lands in a DIFFERENT file than the prior rounds (#2546 r8): two-record provenance (crash + correction), bracket-note ENUMERATION beats token greps for template splice-sweeps, the Step 3.7 sibling list is a moved-file hazard, and full-ledger latest-event-per-id catches brief undercounts"
metadata:
  type: feedback
---

From #2546 r8 (2026-08-25), layered on [[autonomous-decision-crashfix-compose]] +
[[user-ruling-crashfix-round-compose]]:

1. **Two-record crash provenance.** When the experimenter's `epm:failure` closes
   with a hypothesis line ("plausibly exposed by the rN fix") that a LATER
   orchestrator `epm:progress` ROOT-CAUSE marker DISPROVES by measurement,
   inline BOTH in separate envelopes and state the correction relation
   explicitly (the #2147-cr4 "severity right, mechanism wrong" shape): the
   root-cause record is the diagnosis of record; ban chasing the disproven
   lead; its measurements become the ESTABLISHED FACTS block.
2. **Bracket-note ENUMERATION, not token greps, for splice sweeps.** The r7
   template carried round-specific text in SIX bracket families: `[Round-7
   ...]`, `[CODEX ADAPTATION: ...]`, `[Compose-time facts ...]` (Step 0.6),
   `[This round in SELFTEST + PYTEST form: ...]` (hollow-gate), the crash-fix
   "all four elements" rule line, and the plan-span read list + Step 3.7
   sibling list. A `round 7|v7|v6`-keyed grep missed three; the count-pinned
   stale-token sweep (`119.1`, `join_gsm8k_gold`, `_join_gold_rows`) caught
   them at validation. Sweep by `grep -n 'Compose-time\|This round\|this
   round'` over the protocol span BEFORE splicing, and keep per-token
   count-pins as the backstop.
3. **Moved-file hazard: the Step 3.7 sibling list.** The r7 template listed
   `issue2546_gen_capture.py` as an UNTOUCHED sibling; r8's fix lives exactly
   there. Any per-round list of "untouched siblings" / "both GPU drivers
   untouched" must be re-derived from name-status EVERY round — a stale list
   makes the twin skip the very file under review.
4. **Full-ledger latest-event-per-id vs the brief's count.** The brief said "4
   open rows (2 r6 NITs + 2 r7 deferred)"; the full ledger walk found SIX
   not-addressed (also `backing-panel-grain-gap` r1 + `marker-v5-...` r5).
   Deferred rows are a THIRD state (event `deferred`, arming condition in the
   evidence field — here "next round touching stage_corpora.py", which did
   not fire because name-status shows the file untouched): present them as
   CORRECTLY-STILL-DEFERRED status lines, never as open re-raise duties.
5. **Composer-observed marker-record slips feed the open record-accuracy NIT's
   class check, not fresh findings:** v8 self-cited "epm:new-bug-class v1"
   while the posted kind-version is v2 (per-kind auto-derivation), and cited a
   worker line "~1187" that sits at :1256 — both handed as neutral
   adjudications under the SAME-id class check (`marker-v6-record-inaccurate`).
6. **Engagement-token battery.** For a fix whose signal is a new log
   line/artifact, grep EVERY new token at the parent blob (here 6 tokens, all
   zero) and state "zero parent hits ⇒ matched line is decisive" — the
   cheapest decisive engagement evidence a read-only twin can use.

**How to apply:** any crash-fix brief where the crash surfaced on FIRST pod
execution of a pod-only file (VM smokes structurally could not reach it) and
the orchestrator posted a separate root-cause marker. Compose script:
`/tmp/codex-2546-r8-compose.py` (ephemeral; fail-loud OK/ERROR file pattern).
