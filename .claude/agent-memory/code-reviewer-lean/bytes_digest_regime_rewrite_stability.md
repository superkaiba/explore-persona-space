---
name: bytes-digest-regime-rewrite-stability
description: A fix adding an on-disk-bytes artifact digest to resume regimes, where an earlier phase REWRITES that artifact unconditionally on re-entry — probe serializer byte-determinism (np.savez pins zip datetimes to 1980) before flagging a relaunch-refuse loop (#2378 r18)
metadata:
  type: feedback
---

When a fix round threads `sha256(artifact bytes)` into downstream StageLedger
regimes AND an upstream phase unconditionally REWRITES that artifact on every
re-entry (e.g. a bank phase that re-verifies gates per entry and re-saves
`vc_bank.npz` from checkpoint parts), the relaunch trace looks like the
[[dial-added-fingerprint-arms-refuse-on-relaunch]] crash loop: rewrite →
digest drift → downstream regime-mismatch raise on every relaunch. Before
flagging it, probe THREE stability legs empirically:

1. **Serializer byte-determinism** — `np.savez` (numpy ≥ ~1.24 verified on
   2.2.6) writes zip entries via `zipf.open(...)` with date_time pinned to
   `(1980,1,1,0,0,0)`, so identical arrays ⇒ identical bytes. Do NOT assume
   zip mtimes leak in; 5-line probe: save twice across a >2 s gap, compare
   sha256 + `ZipInfo.date_time`. (A `ZipFile.write(tmpfile)` style would leak
   mtimes — check which API the producer uses if not numpy.)
2. **Content rebuild determinism** — the rewrite must rebuild from the SAME
   source in the SAME order on fresh AND resumed runs (e.g. always
   reassembling from `sorted(parts_dir.glob(...))`, never from the in-memory
   capture order of whichever subset ran).
3. **Drift fail-loud ordering** — if inputs DID drift, an inner
   checkpoint-ledger keyed on the input set (ctxs sha) should raise BEFORE
   the rewrite, leaving old artifact + old digest intact (clean refuse, no
   contaminated mix).

All three held in #2378 r18 (commit d6e2dd0648), so the bank-digest regime
design was sound and flagging it would have been a false FAIL.

**Why:** the digest-in-regime pattern is the right fix for
[[params_only_resume_regime_misses_content_regen]]; its failure mode is
byte-instability of the rewrite, which is an empirical property, not a code
smell.

**How to apply:** any diff where a resume/fingerprint key hashes artifact
BYTES that a re-entrant phase re-writes. Run the save-twice probe with the
venv the producer actually runs under (model venv, not repo venv, when they
differ).
