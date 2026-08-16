---
name: codex-local-artifact-forgery-as-blocker
description: "Codex FAILs on forging a LOCAL safety artifact (verify receipt/manifest) as a data-loss path — replay the attack in escalating stages; if every stage short of same-privilege deliberate forgery is caught, the blocker is mistaken (#2321 r2)"
metadata:
  type: feedback
---

Codex classes a "forged local safety artifact" chain (receipt / manifest /
journal re-mint) as a production data-loss Critical even when every
accidental or production path is closed and the forgery requires the SAME
process privilege as the destructive authority itself.

**Incident (#2321 r2, Codex Critical 1):** claimed a corrupt shard `data`
field could pass commit admission because `load_shard_infos` never re-hashes
`rec["data"]` and `write_verify_receipt` is an unrestricted helper. A
3-stage offline probe settled it: stage 1 (same-length data flip) caught by
whole-shard sha-vs-manifest (`issue2321_repack.py:748-750`); stage 2
(+manifest refresh) caught by the receipt's manifest pin (`:687-693`);
only stage 3 (+deliberate receipt re-mint) composed — and the sole
production minting site is the PASSing verify phase (`:624`) whose decoder
round-trip compares decoded bytes against anchor-verified staged originals.
An actor able to run stages 2-3 could call `api.create_commit` directly —
no trust boundary is crossed. Claude g2 had explicitly adjudicated and
accepted exactly this residual.

**Why:** Codex's "attacker-influenced local file" framing presumes an
adversary with local write+execute access, which is outside the threat
model of a single-operator local tool; the r1 ask was phase-sequencing
evidence (skip-proofing), which content-addressing delivers.

**How to apply:** on any Codex blocker of the form "local artifact X can be
re-minted/forged to bypass gate Y": (1) enumerate the production call sites
of the minting helper (grep — one call site inside the passing gate is the
sound design); (2) replay the attack in ESCALATING stages in a disposable
offline probe and record which check catches each stage (file:line);
(3) ask whether the forger's required privilege equals the destructive
authority's — if yes, reject as blocker; salvage the residue (a missing
tamper-class regression test, a ~free per-record re-hash) as a persisted
CONCERN. Distinct from [[codex-hardening-beyond-minimal-port-contract]]
(hardening demands); here the claim is a live data-loss path, so PROBE it,
never argue it only on prose.
