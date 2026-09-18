# Qualitative persona pre-image examples

This artifact supports the revised second claim in Section 4.7 and appendix Figure 27. It describes only the top 100 contexts per behavior from the frozen large-pool dashboard. It does not estimate behavioral prevalence or compare retrieval methods.

- Pool: 952,067 unique generic-chat contexts with saved answers, drawn from the 963,444 metamodel-training pairs.
- Ranking: standardized-context cosine similarity to the regularized persona-vector pre-image.
- Model: Qwen2.5-7B-Instruct, layer 19.
- Original prompts, saved responses and ranks: `dashboard/public/tasks/1739/behavior-explorer.html`, SHA-256 `b972597729f94181f44eaecee6e57f388b8cfdc9f7a291925c876b5b3f5c8697`.
- `prompt_categories.json`: Luna-assigned qualitative categories for harmful compliance and sycophancy, plus the verified common company-introduction template for all 100 hallucination contexts. Each behavior covers exactly its source ci/rank set. The saved-answer descriptions are qualitative annotations, not a new outcome rubric or validated truthfulness labels.
- `figure_inputs.json`: selected illustrative examples and exact source spans, checked against the full source by the producer.
- `review.json`: independent review and validation record.

The figure shows broad prompt themes rather than category counts. Supportive messages may be appropriate, malicious role-play may receive safe answers, and obscure-company descriptions require external fact checking. These are cached metamodel-training contexts with one saved answer each, not a held-out redteaming evaluation. The Singleton Birch founding-date discrepancy was checked against the linked company brochure. The RIDDHI PHARMA example shows an address fragment transformed into a future date; no previous judge score is used as evidence for that example.

## Reproduction

From the EPS repository, using its existing uv environment:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
uv run python scripts/issue1739_qualitative_preimage_figure.py
```

The producer verifies the source hash, pool size, top-100 selection, quoted spans, example ranks and text bounds. It writes color and grayscale PNGs, a PDF, and provenance metadata under `figures/issue_1739/`.

[Browser figure](https://eps.superkaiba.com/tasks/1739/figure/c5_preimage_qualitative_top100.png?v=6849b78db373). The PDF is included in the Overleaf appendix as `figures/paper/c5_preimage_qualitative_top100.pdf`.
