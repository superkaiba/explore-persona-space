"""Task #530 — de-saturated re-run of #504's bubble-vs-barrier geometry.

The actual training recipe + cell specs live in the parent module
``contrastive_neg_geometry_504``; #530 differs only in (lr=5e-6,
epochs=12, 4-frac trajectory) and is wired through ``scripts/i530_*.py``
thin wrappers.

This package currently exists only to host ``data_deps`` — the
auto-downloader for the #472 carry-over artifacts (persona bank,
centroids, on-policy R) that the #530 pipeline reads at the
pinned data revision.
"""
