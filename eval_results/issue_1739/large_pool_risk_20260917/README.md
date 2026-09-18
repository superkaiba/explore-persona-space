# Completed Luna large-pool analysis

Results: `results.md`; complete metrics: `luna_results.json`; revised section: `paper/06_behavior.tex`; add `paper/large_pool_details.tex` to the existing behavior appendix. The section preserves the historical readout-transfer result and replaces the main preimage discussion. These are reviewable manuscript files, not an Overleaf deployment.

Figure: https://eps.superkaiba.com/tasks/1739/figure/c5_large_pool_luna_retrieval.png?v=fd32fec8da7e

Reproduce analysis with `scripts/issue1739_luna_analysis.py analyze` using the frozen preparation outputs, canonical annotations, and template partition. The final HF archive includes all raw content packets, labels, rejected batches, correction provenance, independent audits, exact native dispatches, analysis source snapshots and deliverables. `archive_verified.json` records its pinned revision and hashes once verified. Preparation inputs are pinned at HF revision `aaf0e20c5f9be2f5b1512efe3985c4459e777cff`, prefix `issue1739_large_pool_risk_20260917`.

The annotation launch source is commit `25f448122878df06f9f5f9e4b669c8f631963f49`. Analysis/publishing/archival additions were developed afterwards; their exact archived file hashes, rather than that launch commit, define the analysis version. No annotation-level fit or selection was changed after seeing method outcomes.

The original preparation methods, verification and prior scoring blocker are preserved in `preparation_record.md` as historical context.
