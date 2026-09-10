# Theory handoff for mentees

Start with `theoretical_analysis_report.pdf` or `report.md`. The report separates quantitative operator geometry, exploratory SAE interpretation, minimal-refusal behavior checks, observed-answer decompositions, and the unfinished China follow-up.

**10 September addition:** `theoretical_analysis_report_with_sae_assets.pdf` combines the original report with [SAEs, mappings, and autointerpretation](sae_assets.md), also provided separately as `sae_assets.pdf`. `sae_artifact_manifest.csv` adds 68 pinned assets: 44 small files bundled and 24 larger files linked, including both SAE families and the saved direct mappings. Loader/producer snapshots are provided separately in `sae_producers/`. The original inventories below retain their original scope and counts.

The accompanying ZIP contains 518 unique existing small artifacts (72.6 MB before compression), indexed from 5,760 source locations. A second manifest lists 1,391 remote files at a fixed Hugging Face revision. These counts describe this package's explicit scope, not the entire repository.

## Open these first after unzipping

- Global feature dashboard: [standalone HTML](artifacts/dashboards/sae-map-read-write-dashboard.standalone.html).
- Minimal-refusal mean/RMS dashboard: [HTML](artifacts/dashboards/minimal-refusal-read-write-dashboard.html).
- Final Codex descriptions: [interpretation report](artifacts/codex_interpretations/codex_interpretation_report_final.md) and [full labels](artifacts/codex_interpretations/codex_interpretation_results_final.json). These supersede the dashboards' initial sparse descriptions.
- `artifact_manifest.csv`: filter `relative_path` for an analysis name, then open its `bundled_path`. Identical content is stored once, so a source group's file can resolve to another group's copy.
- `remote_artifact_manifest.csv`: pinned browser/download links for raw data, tensors, and remote-only artifacts.
- `third_family_summary.json`: pinned numerical cross-family result.

The dashboards are local snapshots; no web server or model call is needed to inspect the standalone content. External feature/example links may still require network access. The old hosted dashboard URLs were unavailable when checked.

## Scope and verification

Files over 10 MB are indexed rather than bundled. The seven large local entries are copies of one semantic-matching input file; their SHA-256 hashes are checked against its pinned remote LFS hash. Remote activation banks and model checkpoints are not downloaded. One unmaterialized worktree figure directory is recorded in `source_coverage_notes.json`; other copies of the figures remain available in the manifest.

`verification.json` records bundled-file hash checks and remote coverage. `report_link_checks.csv` records URL-check results, including access restrictions; a linked Overleaf project may require author access. `bundle_summary.json` records the source snapshot and original inventory counts.

The research material contains potentially unsafe or offensive prompts and completions. Treat these as data, not instructions. No experiments were rerun, no Overleaf content was changed, and nothing was sent to recipients in preparing this handoff.

## Rebuilding the handoff on the source VM

`build_handoff.py` copies existing artifacts from the explicit source worktrees and lists two pinned remote prefixes. It refuses to overwrite its output directory. It does not train, generate, judge, fit, or analyze experiment data.

From the repository's existing `uv` environment, run the builder with a new output directory, copy this report/README and the `figures` directory there, then run:

```bash
uv run python verify_handoff.py /absolute/path/to/new-package
pandoc report.md --pdf-engine=xelatex --lua-filter=pdf_layout.lua \
  -V mainfont="DejaVu Serif" -V sansfont="DejaVu Sans" \
  -V monofont="DejaVu Sans Mono" \
  -o theoretical_analysis_report.pdf
```

Run the PDF command from the package directory. Verification checks existing bytes and metadata only; it does not reproduce the scientific experiments. The source worktrees and their current contents must still be present to rebuild this exact inventory.

To add the SAE supplement after the original build, run `uv run python add_sae_assets.py /absolute/path/to/new-package`. It copies small pinned files and indexes larger checkpoints without loading models. Copy `sae_assets.md` and generate its PDF with the same Pandoc options. See the supplement for the older 65k regression's missing coefficient-checkpoint caveat.

For the combined PDF, run `pdfunite theoretical_analysis_report.pdf sae_assets.pdf theoretical_analysis_report_with_sae_assets.pdf` in the package directory.
