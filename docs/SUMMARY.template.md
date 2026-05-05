<!-- Source-of-truth template for docs/SUMMARY.md (rendered by
     scripts/render_summary.py). Edit this file to change Motivation,
     Long-term goals, or Glossary. The four `<!-- SECTION -->` /
     `<!-- /SECTION -->` blocks below are spliced verbatim into the
     rendered SUMMARY; everything outside them is template metadata.

     Sections marked `(auto-rendered)` are computed by render_summary.py
     from claims.yaml + clean-result issues; they are NOT spliced from
     here. -->

<!-- MOTIVATION -->
Large language models can be cued into "personas" — coherent voices that
shape capabilities, refusals, and apparent values. **Emergent
misalignment (EM)** (Betley et al., 2025) shows that fine-tuning on
narrow harmful data — e.g. insecure code — can elicit broadly harmful
behaviour outside the training distribution, often through what looks
like a *villain character* slot in the model's persona space (Wang et
al., 2025). This project characterises that persona space — its
geometry, where in the network it lives, how interventions propagate
between personas, where the assistant axis comes from — and uses what
we learn to defend the assistant against EM.
<!-- /MOTIVATION -->

<!-- LONG_TERM_GOALS -->
- **Geometric understanding** of how personas are represented (manifolds
  vs. centroids, compositional bases of transferable traits).
- **Localisation results** that say which mechanisms (SFT / DPO / SDF /
  midtraining) keep an intervention confined to one persona.
- **A propagation map** that predicts when a behaviour learned on
  persona A will leak to persona B from pre-intervention persona-space
  distance.
- **Pretraining-origin attribution** for the assistant axis — which
  documents and discourse modes build it.
- **Concrete EM defences** (capability gating, identity anchoring,
  truthification) with multi-seed multi-scale evidence on at least one
  open-weights production-tier model.
- A paper that names the persona-space view of EM defence and supplies
  reproducible recipes other groups can stress-test.
<!-- /LONG_TERM_GOALS -->

<!-- GLOSSARY -->
- **EM (emergent misalignment)** — Off-task harmful behaviours that
  emerge after fine-tuning on narrow harmful data (Betley et al. 2025).
- **SFT (supervised fine-tuning)** — Standard next-token-prediction
  training on labelled (input, output) pairs.
- **LoRA (Low-Rank Adaptation)** — Adapter-based fine-tuning that trains
  low-rank matrices on top of frozen base weights.
- **DPO (Direct Preference Optimisation)** — RL-free preference
  fine-tuning from pairwise (chosen, rejected) responses.
- **SDF (synthetic document finetuning)** — Continued pretraining on
  generated documents that anchor a target identity / belief.
- **Tulu** — Allen AI's open instruction-tuning dataset family (used
  here for capability ground-truth and EM-defence midtraining).
- **Betley** — Shorthand for the Betley et al. 2025 EM paper / its
  insecure-code dataset.
- **persona axis** — Linear direction in residual stream encoding a
  persona-related concept (e.g. "evil", "assistant").
- **leakage** — Extent to which an intervention targeted at persona A
  also alters behaviour under persona B.
<!-- /GLOSSARY -->
