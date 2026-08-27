---
name: haar-invariant-shared-rotation-null
description: Two-sided rotation nulls over operator PAIRS reduce exactly to a spectra bilinear form — one shared Haar draw set serves every pair; null sd is exactly 1/sqrt(d_in*d_out)
metadata:
  type: feedback
---

For the two-sided random-rotation operator null cos(vec(A), vec(Q1^T B Q2))
(the `issue1345_operator_comparison.raw_cosine_with_rotation_null` /
`issue825_map_alignment._procrustes_cosine_null` convention), use the EXACT
Haar-invariance reduction before ever looping pairs: with full SVDs
A = Ua Sa Va^T, B = Ub Sb Vb^T (square case),

  cos(vec(A), vec(Q1^T B Q2)) =d  sa_hat^T (G1 * G2^T) sb_hat,
  G1, G2 iid Haar O(d),  sa_hat = svdvals/frob,

because Ua^T Q1^T Ub and Vb^T Q2 Va are themselves iid Haar and rotations
preserve ||B||_F. Consequences: (1) the Haar draws depend on NOTHING
pair-specific — ONE draw set (2 QRs per draw TOTAL) serves EVERY pair of the
same d via a cheap bilinear form, vs 2 QRs + 2 dense d^3 GEMMs per draw PER
PAIR serially; (2) the null mean is exactly 0 and the null sd is exactly
1/sqrt(d_in*d_out) — a free correctness check on any implementation
(measured #2569 d=3584: empirical 2.90e-4 vs analytic 2.79e-4 at 25 draws).

**Why:** #2569 leg-7 atlas (blocker atlas-pair-loop-unbatched-unresumable):
the serial convention measured ~11 s/draw + ~11 s SVD overhead per pair at
d=3584 on the shared VM — ~2,211 s/pair at the 200-draw default, ~116 h for a
~190-pair atlas — vs 214 s per 25-draw chunk for ALL pairs batched.

**How to apply:** worked impl `scripts/issue2569_atlas.py::
shared_rotation_null_draws` (chunked + checkpointed, content-fingerprint
regime keys; algebraic-identity + distribution-parity tests in
`tests/test_issue2569_atlas.py`). Rotating either side of the pair gives the
same distribution (the trace form is symmetric), so rotate the cheaper/other
side freely. Draws shared across pairs correlate the pairs' null BANDS
(each pair still gets n valid draws from its exact null) — say so in the
artifact. For rank-r operators only r columns of each Haar factor matter
(O(d r^2) frames), an extra saving when one side is low-rank.
