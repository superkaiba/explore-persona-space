---
name: cuSOLVER eigh non-convergence CPU fallback
description: torch.linalg.eigh on cuda (cuSOLVER syevd) raises LinAlgError on near-singular / repeated-eigenvalue Grams that CPU LAPACK handles — wrap Gram eigh sites in a CPU-fallback helper
type: feedback
---

`torch.linalg.eigh` on cuda (cuSOLVER syevd) raises
`torch.linalg.LinAlgError: "The algorithm failed to converge because the
input matrix is ill-conditioned or has too many repeated eigenvalues"` on
near-singular Grams that CPU LAPACK decomposes fine. Trigger regime: SMALL
subsampled fold Grams with near-duplicate rows — #1335 attempt 8 hit it at
the matched-n inner-group-CV fold Gram (n_min=1739 subsample, inner-train
blocks of a 4-fold group split), after the SAME code ran thousands of
larger full-lane eigh calls cleanly.

**Fix:** a robust wrapper at every Gram eigh site —
`try: torch.linalg.eigh(G)` / `except torch.linalg.LinAlgError:` →
`eigh(G.cpu())` and move `w, V` back to `G.device`, with a one-line print.
A numerical-backend swap, not a semantic change (same matrix, same
decomposition, fp-roundoff agreement). Canonical implementation:
`scripts/issue825_fit_cells.py::_eigh_robust` (#1335 r10, d1922d2068).

**How to apply:** any per-fold / per-cell ridge or spectral pipeline doing
cuda `eigh`/`svd` on data-derived Grams — especially subsampled or
group-split fits where tiny near-duplicate blocks arise (10+ scripts in
this repo call `linalg.eigh`). Do NOT jitter the Gram instead (changes the
numbers); the CPU fallback is exact.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [cuSOLVER eigh non-convergence CPU fallback](feedback_cusolver_eigh_nonconvergence_cpu_fallback.md) — cuda eigh raises LinAlgError on near-singular Grams CPU LAPACK handles; wrap Gram eigh sites in _eigh_robust-style CPU fallback (#1335 r10)
