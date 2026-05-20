---
title: 'Disentangle persona-rich content, raw token count, and non-persona filler
  at fixed length (followup to #337)'
kind: experiment
tags: []
created_at: '2026-05-11T05:29:43.000Z'
has_clean_result: false
sagan_id: c3bd469d-41f5-4e1d-95b4-c83f0518da5b
sagan_number: 339
priority: normal
legacy_why_unset: true
---
## Motivation

[#337](https://github.com/superkaiba/explore-persona-space/issues/337) (MODERATE) found that across 48 persona LoRAs on Qwen2.5-7B-Instruct, longer source system prompts implant a `[ZLT]` marker more strongly (Spearman ρ = +0.38, p = 0.0074) AND leak it less to bystanders (ρ = −0.38, p = 0.0082). The natural-variation design conflates three plausible mediators that scale together across the panel:

1. **Raw token count** — more tokens to attend to, regardless of content.
2. **Persona-rich content density** — descriptions of role + specialization (e.g. zelthari_scholar's "their crystalline architecture, maritime navigation, and ritual practices" packs persona signal per token).
3. **Specificity vs. genericity** — longer prompts tend to name more concrete role attributes.

We can't tell from the natural variation which of these (or which combination) drives the effect. This experiment manipulates length while *holding total token count fixed* and varying *what* the extra tokens are made of.

## Proposed experiment

Take a small panel of source personas (say 5: one short-prompt — `chatbot`; one medium — `librarian`; one long — `zelthari_scholar`; plus 2 controls). For each source, train 3 LoRAs under the identical recipe ([#274](https://github.com/superkaiba/explore-persona-space/issues/274) / [#296](https://github.com/superkaiba/explore-persona-space/issues/296) parameters: r=32, lr=1e-5, 3 epochs, marker_*_asst_excluded_medium recipe, seed 42), differing only in the system prompt:

- **`base`** — the original persona prompt unchanged (~6-27 tokens depending on source).
- **`+persona`** — pad to a fixed long length (say 100 tokens) by extending the persona's own role description (additional sentences about specializations, daily activities, perspective). Persona-relevant content.
- **`+filler`** — pad to the same fixed long length using persona-irrelevant filler (cloud-formation paragraph, generic disclaimer text — borrow [#295](https://github.com/superkaiba/explore-persona-space/issues/295)'s `sl_long` filler). Non-persona content.

5 sources × 3 length variants = 15 LoRAs. Eval each against the 48-persona matrix (or a smaller shared subset if we want speed) and compute both implantation (diagonal) and mean bystander rate (off-diagonal).

### Predictions / decision rules

- **If raw length is the mechanism**: `+persona` and `+filler` produce the same effect (both implant strongly and leak little vs. `base`).
- **If persona-content density is the mechanism (#337's preferred story)**: `+persona` implants strongly and leaks little vs. `base`; `+filler` implants weakly and leaks broadly (replicating [#295](https://github.com/superkaiba/explore-persona-space/issues/295) at scale).
- **If specificity-of-role is the mechanism**: similar to persona-content but discriminates further within `+persona` between elaborations that name concrete role attributes vs. abstract role descriptions.

The two-arm design (`+persona` vs `+filler` at fixed length) is the load-bearing comparison.

## Compute estimate

- Training: 15 LoRAs × ~25 min on 1× H100 each (per [#296](https://github.com/superkaiba/explore-persona-space/issues/296) wall time per cell) = ~6 GPU-hours.
- Eval: 15 LoRAs × 48 personas × 100 completions on vLLM batched = ~3 GPU-hours.
- Total: ~10 GPU-hours, `compute:medium`.

Could shrink to 3 sources × 3 variants = 9 cells (~6 GPU-hours) if we accept thinner per-source statistics.

## Pod preference

`--intent lora-7b` (1× H100 SXM); could parallelize across 4 GPUs via [#296](https://github.com/superkaiba/explore-persona-space/issues/296)'s `launch_*.py` pattern if we want faster turnaround.

## Why this is worth doing

[#337](https://github.com/superkaiba/explore-persona-space/issues/337)'s correlational finding rests on natural variation; the causal claim ("persona content density localizes the marker") needs a controlled manipulation to defend. If `+persona` and `+filler` give the same result, the story collapses to "raw length" and we revise [#337](https://github.com/superkaiba/explore-persona-space/issues/337)'s mechanism. If they differ, the content-density story is causally confirmed and gives us a real lever for marker localization (a candidate defense knob).

Either outcome is informative.

## References

- **[#337](https://github.com/superkaiba/explore-persona-space/issues/337)** — the cross-persona N=48 correlation this experiment converts to a causal test.
- **[#296](https://github.com/superkaiba/explore-persona-space/issues/296)** — the 48-source marker panel + length-partial collapse that motivated [#337](https://github.com/superkaiba/explore-persona-space/issues/337).
- **[#274](https://github.com/superkaiba/explore-persona-space/issues/274)** — the 24-source recipe + LoRA training pipeline.
- **[#295](https://github.com/superkaiba/explore-persona-space/issues/295)** — the within-librarian system-prompt-length factorial that produced the long-prompt-as-filler observation; supplies a candidate filler text.
