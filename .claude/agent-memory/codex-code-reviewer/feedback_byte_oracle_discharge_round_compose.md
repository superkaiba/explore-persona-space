---
name: byte-oracle-discharge-round-compose
description: Fix round whose payload IS test oracles discharging a reconciler byte-identity blocker — inline the reconciler record as the contract envelope, split duties into oracle-authenticity vs old-form-fidelity, pin the old form to <payload-sha>~1 reads, and compose the text-writer classification as a decidable both-way trace
metadata:
  type: feedback
---

From #2336 r6 (2026-08-24), the single-blocker fix round after the r5
binding `epm:review-reconcile v3` FAIL (`a5-byte-identity-coverage`): a
test-only commit landing old-vs-new byte oracles (jsonl/text byte-identity,
npz load-equality) + a rider NIT pin.

1. **The reconciler record IS the round contract — inline it verbatim** in a
   `---BEGIN/END ROUND-5 RECONCILER RECORD---` envelope and read the
   discharge elements off its "Blocker list for the fix round" paragraph as
   a numbered element list (the v3 record already quotes the r1 rider
   verbatim, so one envelope carries both). Per-element rulings get a
   `**Discharge ruling:**` header line + Plan Adherence rows keyed to the
   ELEMENTS, not plan sections.
2. **Two distinct hollow-oracle failure modes get two duties.**
   Oracle-authenticity (does the NEW side exercise the real migrated code —
   real imports verified; a sanctioned reproduce-verbatim form needs its
   compensating live-source anchor assessed for load-bearing-ness AND
   block-presence-vs-block-is-the-write residual) vs old-form fidelity (is
   the OLD side genuinely pre-migration — pin reads to
   `git show <payload-sha>~1:<path>` with composer-verified line frames; the
   tautology to prevent is an "old form" copied from the new form). Hand
   both the exact `git show` commands; offered-TEST hollowness routes as
   `substantive`, not `hollow-verification-gate` (that tag is for production
   gates).
3. **Compose the contested writer-classification as a decidable both-way
   trace** (the r4 deviation-trace pattern): the marker classified
   `write_jsonl_sharded` (a `write_text` call on jsonl content) as the TEXT
   instance while the reconciler's arming evidence named a plain-text writer
   in another file. Quote the A5 row + both writers' code; accepted =>
   discharged (note the un-oracled writer as residue); rejected =>
   NOT-ADDRESSED core element => substantive FAIL. Never pre-resolve.
4. **Asymmetric closure fences for the two claimed-addressed ids:** the
   reconciler-re-raised BLOCKER's NOT-ADDRESSED = substantive FAIL (the fix
   IS the round); the rider NIT's NOT-ADDRESSED re-raises at NIT severity
   only — but a FALSE closure claim on either takes the ordinary substantive
   bar (Rule 9/13 fabricated-coverage family).
5. **Composer re-runs the marker's own pytest command** (36 tests, ~66s)
   when the round's whole payload is tests — converts "implementer claims N
   passed" into a composer-attested fact Codex (no uv env) can lean on;
   byte-surface questions get the stdlib-probe carve-out (`python3 -c` on
   json/os.path/pathlib only, STATIC fallback) with explicit precedence over
   the never-execute bullet.

**How to apply:** any fix round whose diff is test oracles discharging a
recorded byte-identity / load-equality acceptance row (#2336 batches 3-5
will re-land this shape, incl. the .pt spot check at the first torch.save
batch). Related: [[single-blocker-fix-round-deviation-trace]],
[[respawn-two-record-batch-compose]], [[gate-block remedy round compose]].
