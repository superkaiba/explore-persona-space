---
name: pilot-pass-report-fingerprint-unchecked
description: A rule-26 pilot gate whose PASS report persists the instrument fingerprint but whose require/resume consumers check only family+verdict blesses a stale instrument; plus the route-parity-by-arithmetic and substituted-seam probes (#2479 R2 g6)
metadata:
  type: feedback
---

Reviewing a judge pilot gate (rule 26) or any persisted-PASS gate consumed on
resume: diff what the report PERSISTS against what the require-path COMPARES.
The #2479 r2 gate wrote rubric_sha256 / model / max_tokens / temperature into
every report, yet `require_pilot_pass` checked only family + verdict — so both
wrappers' "PASS report present — pilot skipped (resume)" branches, the P1
preamble, and the run_leg env hook all bless a PASS persisted under a rubric or
max_tokens later edited by a crash-fix round. Recorded-but-never-compared is
the tell; the fix is a consumer-side fingerprint compare vs the LIVE constants.
Sibling of [[presence-redrive-blesses-stale-mirror]] and
[[new-dial-missing-from-resume-regime]].

**Why:** the resume-skip is exactly where instrument drift bites — pilot at
commit X, instrument edited at commit Y, wave dispatches un-piloted; crash-fix
rounds that raise max_tokens or tweak a rubric are the fleet's normal shape.

**How to apply:** (1) grep every consumer of the report (require fn, wrapper
skip branches, env hooks, sibling-leg reuse) — if none compares the persisted
fingerprint to the live instrument, Major. (2) When a family is piloted through
a SUBSTITUTED seam (the registered gate fn can't parse its instrument), certify
the substitute by byte-comparing it to the production dispatch site: payload
dict KEY ORDER, model constant, force_path, transport re-drive shape. (3) Route
parity settles by arithmetic, not narration: per-call n at the production call
site vs the dispatcher's crossover constant (SYNC_BATCH_CROSSOVER_N=2000; gen
per-call n ≤ n_target=1600 → sync both sides); a plan §9 "Batch" label can
mislabel a sync-realized wave. (4) "route recorded" claims in the commit
message get grepped against the payload keys — #2479's partial recorded none.

**r4 closure shape (#2479 R4 g1) — grade later rounds against these:**
(a) DATA identity beside the instrument: hash committed input FILE bytes +
parsed item CONTENT via an order-independent sorted-triples sha (never file
bytes of re-emitted intermediates — byte-order drift would false-refuse);
(b) validate-after-refresh sequencing holes close by RECORD-LICENSE — the
spend site persists the licensing gate's fingerprint IN the artifact it
licensed, and resume compares it to the CURRENT gate (refresh-at-same-
fingerprint = equivalence; different = quarantine); (c) an exact-N per-item
draw census over save_raw `all_scores` keys is SAFE — batch_judge mints
error dicts for every dispatched row (#1313), so transport losses still
count keys and only a genuinely partial raw file fails; (d) live-probe
fixtures for `require_pilot_pass`-style gates need the FULL verdict
predicate (`verdict` AND `passed`) — build them from the test helpers'
`_report` shape, not from the docstring.
