---
name: trap-raw-decay-contrast-compression-tail
description: "Paired raw-scale decay/persistence contrasts (ΔD = D(arm A) − D(arm B)) with unequal starting levels: the below-zero tail is scale-compression-confounded — an affirmative 'more persistent' verdict branch on raw ΔD<0 fires under equal PROPORTIONAL decay whenever arm A starts lower; require the anchor-normalized companion (ΔD_F) to license that tail"
metadata:
  type: feedback
---

When a plan registers a verdict lattice on a RAW-scale paired
difference-of-differences ΔD = [Q1−Q4](arm A) − [Q1−Q4](arm B) and the two
arms start at different levels (e.g. patched persona Q1 ≪ prompted ceiling
Q1), the two tails are NOT symmetric:

- ΔD > 0 is conservative (compression works against it) — a clean
  affirmative branch.
- ΔD < 0 fires mechanically under EQUAL proportional decay (lower start ⇒
  less room to fall ⇒ smaller raw drop), so an affirmative
  "arm A more persistent" branch on raw ΔD<0 is sign-confounded by a
  mechanism the plan may itself acknowledge as a report caveat ("floor
  compression biases ΔD downward") without wiring it into the verdict.

**Why:** #2329 `q35_ladder_decay` v7 registered exactly this: H5's
"ΔD < 0 CI-excluding ⇒ patch-more-persistent (surprising, reportable)"
alongside R4's own admission that compression biases ΔD downward — the
below-zero branch could fire from pure starting-level compression with
consistent sign across all 6 carrier clusters.

**How to apply:** on any registered decay/persistence/fade contrast, check
whether the below-(or above-)zero affirmative branch is reachable under a
null of proportional change given unequal baselines. Fix shape (zero new
calls when the normalized read already exists): condition the confounded
tail's affirmative verdict on the anchor-/scale-normalized companion
contrast agreeing (same sign, CI excluding 0, denominators passing the
suppression bar), else that tail reads "inconclusive — raw contrast
compression-confounded". Report-side "read alongside the Q1 gap" notes do
NOT discharge it — the verdict-generating lattice is the binding object.
Related: [[trap-2162-rerun-prereg-vs-realized]].
