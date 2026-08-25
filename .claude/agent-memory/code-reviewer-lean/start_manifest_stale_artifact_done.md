---
name: start-manifest-stale-artifact-done
description: A resume predicate keying "done" on artifact PRESENCE while the fingerprint manifest is written at cell START skips onto a PRIOR run's stale artifacts after a crashed retrain — trace (completed@F1 → fingerprint flip → crash mid-retrain → relaunch) on every per-cell manifest design (#2225 R1 g2)
metadata:
  type: feedback
---

When a per-cell resume predicate is "skip iff done AND stored-fingerprint ==
current" and the manifest is written at cell START, check whether "done" is
derived from artifact PRESENCE (adapter files on disk / on HF). If yes, the
mismatch→re-run guarantee is one-shot: cell completes under fingerprint F1 →
code-fix/direction-rebuild flips to F2 → the re-run writes the F2 manifest at
START, then crashes mid-train → relaunch sees F2==F2 AND the F1-era artifacts
still present → SKIP. The stale artifact silently ships as the retrained cell.
A whole-HEAD `code_sha` field makes the trigger LIKELY: any commit re-runs ALL
cells, so one mid-wave crash (OOM/preempt) strands several cells exactly here.

**Why:** #2225 split-review R1 g2 (2026-08-10): `scripts/issue2225_train.py`
wrote {fingerprint, started_at} at cell start (line 535) and skipped on
`_local_done` (adapter files exist) or `_hf_complete` (files under the HF
prefix) — neither leg bound the artifact to the fingerprint that produced it;
the prior run's save_pretrained output survives a crashed retrain untouched
(save_strategy="no" writes nothing until the final save). The plan's own §9
wording ("manifest row written at cell start") licensed the shape — the hole
was in the plan's letter, caught only against its intent ("a mismatch
re-runs").

**Ordering variant (#2476 R1 g1, 2026-08-22):** a design that DOES delete
stale outputs on a manifest mismatch is still exposed when the helper writes
the UPDATED manifest and returns, and the CALLER deletes stale outputs after
(`_enter_phase_regime` wrote new-code regime.json → caller unlink loop): a
kill inside that window leaves new manifest + old outputs, and the next run's
presence-skip blesses them. The window can be ~4 statements — flag it anyway
(Minor at that width) and check every phase inheriting the helper: the fix is
one reorder (delete inside the helper BEFORE the manifest write). Same round:
`--hf-prefix` (an output DESTINATION) omitted from the regime hash lets a
matching-regime re-run skip the upload to a NEW prefix off the OLD prefix's
done-file — destination args belong in the resume key too.

**How to apply:** on any per-cell manifest/resume diff: (1) find where the
manifest is WRITTEN (start vs end) and what "done" reads (presence vs a
completed flag); (2) run the four-step trace above; (3) fix shapes to suggest:
wipe the cell's artifact files right after writing the START manifest on a
non-skip run, or write the fingerprint manifest only at END (done-marker
semantics) with an `uploaded` flag so a failed upload re-drives upload instead
of retraining. Sibling of [[rc-halt-not-resume-idempotent]] (halt erased by
resume-skip) and [[additive-edit-flips-wholefile-resume-key]] (whole-file
keys over-invalidate); this one UNDER-invalidates — presence-done binds to no
fingerprint at all.
