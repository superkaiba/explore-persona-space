---
name: Neutral-prompt axis-conflation
description: Steering plans whose "neutral" prompt string IS one of the evaluated personas' actual prompts confound the headline — grep personas.py for the literal string before approving (#267)
type: feedback
---

When a steering / activation-intervention plan defines a "neutral" or control system prompt as a literal string, check whether that string is a member of the persona set under test. `personas.py` defines `ASSISTANT_PROMPT = "You are a helpful assistant."` and helpful_assistant is one of the evaluated personas.

**Why (#267):** the plan used exactly that string as the "neutral" steering prompt — making helpful_assistant's coeff=0 cell identical to its prompted baseline (cos to self = 1.0) and giving every other persona a non-uniform fraction of the helpful_assistant centroid for free, weighted by cos-to-assistant. The cosine→source-rate regression became partially tautological and the ordering ρ partially measured cos-to-assistant bias. Slipped past plan v2 despite multiple critic rounds.

**How to apply:** grep `personas.py` for the literal control string before approving. On a match (especially helpful_assistant, qwen_default, or any assistant-like persona): require (a) an empty system role / non-persona instruction outside the centering set, or (b) dropping the matching persona from the headline N, or (c) centroids re-extracted on a centering set excluding the match and structural near-duplicates (qwen_default at cos +0.714). Compositional variant (panel contains a whole family in the baseline's register): feedback_panel_family_clustering.
