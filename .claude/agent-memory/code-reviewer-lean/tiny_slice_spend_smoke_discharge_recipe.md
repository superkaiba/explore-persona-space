---
name: tiny-slice-spend-smoke-discharge-recipe
description: Certify a "real tiny-slice smoke executed" discharge commit — corroborate marker digests against the live log + scratch tree, scoped HF listing in BOTH directions, sync-dial default + dial-inside-resume-regime, synthesized-artifact identity binding (#2479 r4 g2)
metadata:
  type: feedback
---

Reviewing a commit whose whole claim is "we RAN the chain for real at tiny
scale" (discharging a smoke-unexecuted concern): the code is a thin driver;
the review substance is corroborating the EXECUTION and bounding the
poisoning surface.

**Why:** #2479 r4 g2 — the marker's per-phase digests, the live
`/tmp/*smoke*.log`, and the scratch tree matched byte-for-byte, which retired
the fabrication question in one pass; every residual risk lived in four seams
the code alone doesn't show.

**How to apply:**
1. Diff marker `## Smoke run` digests (bytes+sha+counts+rc) against the
   still-on-disk log and scratch artifacts — an exact three-way match is the
   cheapest fabrication kill; a missing log/tree downgrades the evidence.
2. Scoped HF listing BOTH directions: the smoke prefix EXISTS with the
   claimed count, AND the production family prefix's children contain ONLY
   the smoke subdir (nothing production-shaped written). Then read the
   upload-verifier: presence-of-specific-sub-prefixes checks can't be
   polluted by a smoke subdir; family-wide listings/set-diffs can
   ([[smoke_shard_namespace_only_done_files]]).
3. A route/scale escape dial (`--threshold-base <huge>` for sync) needs
   (a) production default unchanged at EVERY surface (argparse + function
   signatures + wrapper), and (b) the dial INSIDE the resume/skip regime so a
   smoke-produced report can never resume-bless production — the rc-refusal
   demo keying on exactly that field is the proof
   ([[new-dial-missing-from-resume-regime]]).
4. A SYNTHESIZED gate artifact (pilot PASS) is safe iff a consumer-side
   identity compare binds it to the scratch materialization — production
   defaults must refuse it structurally, not by filename/flag convention;
   the flag (`smoke_synthesized: true`) is disclosure, the fingerprint is
   the protection ([[pilot-pass-report-fingerprint-unchecked]]).
5. Rerun only the FREE verbatim commands ((c)'s `--import-check`); never
   re-execute the spend-bearing success command as reviewer.
