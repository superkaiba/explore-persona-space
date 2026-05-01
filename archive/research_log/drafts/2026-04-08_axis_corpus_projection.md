# Assistant Axis Corpus Projection — First Results (Aim 4.2)

**Date:** 2026-04-08
**Status:** REVIEWED (preliminary — superseded by deep tail analysis)

## Goal

Project FineWeb-Edu documents through Qwen 3 32B onto the pre-computed assistant axis to characterize what pretraining text activates the axis most/least. First step toward understanding what data builds the assistant persona.

## Setup

- **Model:** Qwen/Qwen3-32B via vLLM embed mode (tensor_parallel=2)
- **Axis:** Pre-computed from lu-christina/assistant-axis-vectors (Qwen 3 32B, 64 layers × 5120 dim)
- **Layer:** vLLM extracts final-layer post-norm hidden states (not layer 48 pre-norm where the axis was computed — this is a mismatch, see caveats)
- **Data:** 72,759 FineWeb-Edu documents (target was 200K, vLLM hung at 72K)
- **Method:** Last-token pooling → dot product with normalized axis vector

## Results

**Projection distribution:**
- Mean: 4.74, Std: 5.89
- Significant correlation with token count (length confound — longer docs project higher)

**Top 0.1% (most assistant-like) — TF-IDF keywords:**
`students, children, make, science, use`

**Bottom 0.1% (least assistant-like) — TF-IDF keywords:**
`god, jesus, temple, land, israel`

## Interpretation

1. **Educational/instructional content projects highest on the assistant axis.** Documents about teaching students, science activities, and practical how-to content activate the assistant direction most strongly. This aligns with the hypothesis that the assistant persona is partly built from instructional text in pretraining.

2. **Religious/narrative content projects lowest.** Biblical text, theological discussions, and historical-religious narrative are maximally far from the assistant direction. These represent authoritative/declarative text rather than helpful/responsive text.

3. **Length confound is significant.** Longer documents project higher — this needs to be controlled for. The current analysis uses raw projections without length normalization.

## Caveats

- **Layer mismatch:** The axis was computed at layer 48 (pre-norm) but vLLM extracts from the final layer (63, post-norm). Relative ordering should be approximately preserved but absolute magnitudes may differ.
- **Only 72K of 200K target docs** — vLLM hung after 72K. Sufficient for preliminary tail analysis but not for a full classifier.
- **FineWeb-Edu only** — no LMSYS comparison yet. The LMSYS projection would show how conversation data differs from web text on the axis.
- **No length normalization** — the length confound may explain some of the tail differences.
- **Single corpus subset** — FineWeb-Edu is quality-filtered; raw FineWeb might show different patterns.

## Next Steps

- Run with length normalization (project residuals after regressing out token count)
- Complete LMSYS projection for cross-corpus comparison
- Manual taxonomy of the 72 top and 72 bottom tail documents
- Fix the vLLM hang issue or fall back to HF with torch.compile
- Train DeBERTa classifier on the tail documents (Aim 4.3)
