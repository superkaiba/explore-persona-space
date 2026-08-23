---
name: v2-scoped-round-panel-format
description: "v2 scoped verification rounds: follow the brief's custom panel verdict format verbatim (no epm:code-review-codex envelope; brief's own CONCERN:: grammar); inline any progress-note that CORRECTS an impl-marker claim with a not-dishonesty instruction; hand fix-interaction questions with compose-verified line anchors (#2389 r6)"
metadata:
  type: feedback
---

Post-cap user-authorized scoped rounds on a `workflow: v2` task (#2389 r6,
2026-08-23) compose differently from a standard Step-5 round:

1. **Verdict format comes from the BRIEF, not the composer spec.** The
   orchestrator composes the round marker from the panel's report TEXT, so
   there is NO `epm:code-review-codex` marker envelope and the CONCERN::
   grammar may differ from the v1 forwarder form (#2389 r6 used
   `CONCERN:: id=<kebab-id> severity=<sev> summary=<one line>`). Follow the
   brief verbatim; assert `epm:code-review-codex` count == 0 in the composed
   prompt; keep the "token must not open any other line" guard; zero
   findings => NO rows (the brief's per-finding semantics), not the v1
   `CONCERN:: none` sentinel unless the brief says so.
2. **A stale-but-corrected impl-marker claim gets the correction INLINED in
   its own envelope** (e.g. #2389 v6's "lint rc=0" corrected by `epm:progress`
   v21 to rc=1 with 5 pre-existing rows) plus two instructions: treat the
   correction as superseding (do NOT flag the stale line as dishonesty — it is
   corrected on the record), and verify the payload-attribution half
   independently (`git diff <range> --name-only` vs the named red files).
3. **Compose-time interaction anchors beat vague duties.** Verify the cheap
   facts yourself (NUM_WORKERS derivation + floor, `_pilot_gpu_name()` None
   iff no CUDA, `run_single_gpu` pins `CUDA_VISIBLE_DEVICES=0`), STATE them as
   compose-verified in the prompt, and hand the twin the one REAL open
   question with line anchors, severity unresolved (does `--num-workers 8`
   into a 1-GPU capregen leg key any cell/shard selection, or only the
   pilot-report validation?). Count-asserts on your own expectations catch
   composer arithmetic errors (range-token 4 not 5; fix-SHA 8 = 7 table rows
   + branch line).
4. **Scoped-round framing block:** user-authorization provenance, verify-the-
   residual + NEW-defects-only scope, do-not-relitigate list (closed blockers
   + recorded NITs + each deliberate non-change with its justification handed
   for adjudication, not auto-flagging), calibration both directions, and
   "PASS plainly if clean".

Related: [[revision-round-compose-recipe]], [[whole-round-unsplit-compose]].
