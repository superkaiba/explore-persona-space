# numpy argsort tie order is CPU-SIMD-kernel dependent — never gate cross-machine on recompute-equality of a top-k-by-count selection

**Trap (#1946, 2026-08-01):** a P1 identity gate recomputed a top-16,384-by-activity-count
feature selection (`np.argsort(-counts)[:cap]`) and asserted array-equality against the
banked `feat_ids`. It PASSed on the shared VM (Xeon, no AVX-512) and CRASHED on a GCE
n2-highmem (Cascade Lake, AVX-512) with byte-identical inputs and identical numpy 2.2.6 /
torch versions: 5 features TIED at the cap-boundary count (7198) competed for 2 slots, and
numpy 2.x dispatches CPU-specific x86-simd-sort kernels whose UNSTABLE tie order differs
per machine — so which 2 of the 5 got selected was CPU-dependent.

**Fix:** the BANKED selection is authoritative (downstream fits/predictions were made on
those exact columns — a recomputed selection differing in tie slots would silently
misalign columns). Gate with machine-independent SET-VALIDITY invariants instead:
floor match; `len == min(cap, n_eligible)`; all banked ≥ floor; every strictly-above-boundary
feature ∈ banked; `n_above < cap`; log the boundary tie structure
(`boundary count / n strictly above / n tied / tie slots`).

**How to apply:** any cross-machine reproduction/identity gate over a ranked-selection
artifact (top-k features, top-k tokens, argmax picks) must either use `kind="stable"`
sorts on BOTH sides at creation time, or — when the artifact is already banked — verify
set-validity invariants, never recompute-equality. Worked fix:
`scripts/issue1946_sae_percontext.py` `_scan_and_gate` (commit b9b7e7d982c9).
