---
name: na-classification-both-legs
description: On a class-sweep round, EXECUTE both failure legs against every wrapper recorded NOT-APPLICABLE — a partial guard reads as "already guarded" in prose
metadata:
  type: feedback
---

On a class-sweep task ("port fix X to every sibling"), a NOT-APPLICABLE
classification whose reason is "already guarded by its own design" must be
verified by RUNNING the target against **every leg of the failure class**,
not by reading its guard. A file can guard leg A and silently drop leg B
while the prose reason — and the file's own comments — read as fully
guarded.

**Why:** #2386 swept the #2196 silent-skip class (uncreatable OR unwritable
log dir) across 15 cron wrappers. `cron_codex_auto_upgrade.sh` was recorded
N/A citing its checked `mkdir` + `SETUP_OK` alert-arm design. Executing it:
uncreatable → FATAL + Telegram push (guarded, as claimed); **existing-but-
unwritable → exit 0, wrapped tool never ran, no push** — exactly the class
under sweep. `mkdir -p` succeeds on an existing dir, so only the second leg
reaches the brace-group redirect. The plan compressed that leg into "the
residual TOCTOU accepted"; it is not transient.

**How to apply:**
- Enumerate the failure class's legs FIRST, then test each N/A target
  against all of them in a scratch dir (fake the tools on `PATH`, redirect
  `HOME`, drive the real script via `bash`).
- Treat "already guarded by its own design" as the highest-risk N/A
  phrasing — it invites a read instead of a run.
- Check whether the round's own class-invariant test can even SEE the N/A
  target. In #2386 the scan short-circuited on `if not any(ln.startswith(
  "fatal() {"))`, so the one unguarded sibling was structurally invisible
  to it. A class invariant keyed on the FIX's marker rather than the
  VEHICLE's shape cannot detect an unfixed member.
- Before escalating: run the Step 0.9 provenance probe. If the target is
  byte-identical to `main`, it is pre-existing-on-trunk → Major +
  persisted concern, not a FAIL.

Related: [[feedback_new_fence_silent_pass_audit]],
[[feedback_sweep_plan_controls_list]],
[[feedback_wrapped_literal_evades_site_set_grep]].
