---
name: verdict-assembly-status-only-source-key
description: A verdict-assembly phase whose resume key embeds producer STATUS strings (computed/deferred) re-triggers on the deferred->computed flip but resume-skips onto stale values when a producer's CONTENT changes while status stays computed (#2569 r1 shard A)
metadata:
  type: feedback
---

Rule: when a phase ASSEMBLES a registered verdict from sibling-phase JSONs and
keys its resume predicate on the sources' `status` strings, probe the
SECOND-ORDER case: producer recomputed with different content, status still
"computed". The deferred->computed flip re-triggers (the designed case, tests
cover it); a content change does not — the assembly skips and the registered
verdict record silently carries the prior run's clause values.

**Why:** #2569 r1 shard A (`scripts/issue2569_weights.py` `_criterion_regime`):
regenerating the P-B moments dir recomputed dw-mass + split-half (their regimes
embed the moments fingerprint) while `phase_criterion` logged SKIP (live probe:
producers' mtimes changed, criterion's did not). Same family as
[[fingerprint-resume-ids-not-content]] (#2552 r2 g2) — the id/status axis is
hashed, the content axis is not.

**How to apply:** for every resume key that references OTHER artifacts, ask
"does the key change when the referenced artifact's CONTENT changes?" The fix
is embedding the producers' own regime dicts (or the upstream fingerprint they
embed), not their status. Probe recipe: two-run live probe with a regenerated
upstream dir at different n; compare per-file mtimes (producers changed,
assembler unchanged = confirmed).
