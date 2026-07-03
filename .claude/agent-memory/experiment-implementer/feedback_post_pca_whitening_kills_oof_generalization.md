---
name: Post-PCA whitening kills out-of-family ridge generalization
description: Standardizing EACH PCA coordinate after a fold-fit projection whitens near-noise trailing dims up to unit variance and destroys LOFO generalization; standardize AMBIENT then project; debug oracle = pure rotation is ridge-invariant
type: feedback
---

Never re-standardize per-dim AFTER a train-fold PCA projection of ridge inputs
("whiten-k"): trailing PCA dims are near-noise, and dividing by their tiny sd
amplifies them to unit variance — LOO/PRESS λ selection does NOT protect
(within-train LOO still contains same-family rows), and out-of-family skill
collapses. #923 smoke: identical fold, +0.13 (ambient) → −6.4
(rotate + whiten) while a PURE orthonormal rotation left skill bit-identical.

**Why:** ridge is invariant to orthonormal input rotations but NOT to per-dim
rescaling; post-PCA rescaling re-weights the penalty toward the noise tail.

**How to apply:** order = ambient train-fold standardize (ddof=0, +1e-9) →
PCA-project → mean-center only (no per-PCA-dim sd division). When debugging a
mysteriously negative held-out skill in a PCA+ridge pipeline, run the pure-
rotation oracle first: exact-SVD-rotate the design with NO rescale — if skill
moves, the bug is downstream of the rotation; if it only moves once
standardization is applied, it is the whitening. Caveat: #722 reported
"whiten-48 inputs reproduce" on its real n=50 grid (smooth spectrum, no clean
noise tail) — the pathology is regime-dependent, so verify per design rather
than assuming either way. (Impl: `press_fit_predict(standardize=False)` +
`build_part._std` in `scripts/issue923_fit_decomposition.py`.)
