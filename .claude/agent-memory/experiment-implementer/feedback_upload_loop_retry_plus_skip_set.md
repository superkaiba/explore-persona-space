---
name: upload loops need 5xx retry + skip-set pairing
description: Per-file HF upload loops over hundreds of ops must pair bounded 5xx retry with a pre-fetched list_repo_files skip set; verification on a fresh listing
type: feedback
---

A per-file HF Hub upload loop over hundreds of ops WILL eventually hit a
transient 5xx (#542 observed 2 failures in ~200 commits), and without
skip-already-on-hub every relaunch replays the full list from scratch — each
respawn is a fresh lottery.

**Why:** burned twice at #542 assemble (2026-06-12); recovery cost an
implementer round mid-run.

**How to apply:** when writing ANY multi-file Hub upload loop, always pair
(1) bounded per-file retry on `HfHubHTTPError` 5xx (e.g. 30/60/120s backoff;
4xx incl. quota-403 stays fail-loud per upload-policy) with (2) a
pre-fetched `list_repo_files` skip set for idempotent resume — and keep the
final fail-loud presence verification on a FRESH listing, not the skip set.
