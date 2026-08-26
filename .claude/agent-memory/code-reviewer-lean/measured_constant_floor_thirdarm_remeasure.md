---
name: measured-constant-floor-thirdarm-remeasure
description: A MEASURED threshold constant calibrated on k artifacts is certified by re-measuring on one artifact OUTSIDE the calibration set through the production path; pair with a numeric invariance probe for any rotation-invariant companion read.
metadata:
  type: feedback
---

When a diff pins a measured constant (e.g. `DEGEN_REL_GAP_FLOOR = 1e-3`
calibrated on 2 of 18 banked adapters, #2569 r3), the decisive review move is
NOT re-reading the calibration comment — it is re-measuring on an artifact the
floor was NOT fitted to, chosen to span new variation (new behavior family AND
new lr). Recipe: one `hf_hub_download` of the adapter (~300 MB) into the
existing per-issue `/mnt/eps-data/$USER/issue<N>_*/` staging dir, run the
PRODUCTION spectrum path (`load_adapter_factors` + `lora_svd_factors`), report
floor vs p5 vs observed minima (#2569: p5 1.44e-3 > 1e-3 > min 3.9e-4 — floor
generalized). Delete the staging dir after.

**Why:** the round-3 brief called the third-arm re-measurement "the decisive
evidence"; two-arm calibration comments are unfalsifiable by reading.

**How to apply:** any diff introducing a threshold whose docstring cites a
k-artifact probe; also pair it with:
- a numeric invariance probe for a claimed rotation-invariant companion
  (within-tie rotation of orthonormal stack rows leaves `||stack @ d||`
  invariant to 0.0e0; a tie straddling the top-k boundary changes it — so the
  companion is sound iff a `boundary_tied` flag guards exactly that case);
- the fails-pre-fix probe ([[fails-pre-fix-probe-parent-commit]]): load the
  parent-commit module by `importlib` from `git show <parent>:<path>` with the
  worktree `scripts/` on sys.path and drive the new fixture through it.
