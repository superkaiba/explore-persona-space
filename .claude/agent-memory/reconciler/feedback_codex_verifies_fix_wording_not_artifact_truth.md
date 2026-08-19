# Codex verifies fix WORDING landed, not artifact TRUTH

(Entry file created during the #1891 index curation — the index pointer was dangling. The full index hook is preserved below.)

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Codex verifies fix WORDING landed, not artifact TRUTH](feedback_codex_verifies_fix_wording_not_artifact_truth.md) — fix-round disclosure claims ("was not computed") need a re-derive from the pinned JSONs/figures yourself; brief-conformance can't catch an error the round-1 brief originated. #778 r2 REVISE.

## Second instance: caption UNIT-GRAIN claims (#2333 crc r2, 2026-08-18)

A round-2 fix embedded the reconciler-mandated per-unit companion figure with
a NEW caption asserting per-DRAW points ("Each point is one steered prefill
draw"). Codex PASSed by quoting the caption approvingly in its fix-verification
(item 1) and even repeated the wrong grain in its own Lens 11 prose ("1,164
per-row points") — wording-landed verification, zero grain re-derivation.
Claude FAILed with the correct arithmetic; the re-derive confirmed it in
minutes: the pinned figure function scattered one point per `f_cells.jsonl`
row keyed `(pair_id, arm_slug)` with `n_rows: 5` (K=5 draws pre-averaged), and
the sidecar series n (390 = 195 pairs × 2 donor schemes; q35 388/390 with both
F fields) matched the cell count, not the ×5 draw count. REVISE upheld.

**How to apply:** whenever a fix round ADDS caption/setup-line text describing
a figure's plotted unit ("each point is one X"), re-derive the grain yourself:
(a) the plotting function's iteration source + key, (b) the sidecar per-series
n, (c) the arithmetic that n implies (pairs × schemes vs × draws). A caption
that names a unit whose implied count ≠ the sidecar n is the tell. Codex's
fix-verification section is a WORDING check by construction — never let it
carry a PASS on a factual unit claim.
