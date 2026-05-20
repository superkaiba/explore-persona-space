---
title: 'marker_only_loss=True ablation on #295''s lc_long — disentangle gradient dilution
  from undertraining'
kind: experiment
tags: []
created_at: '2026-05-11T21:48:13.000Z'
has_clean_result: false
sagan_id: a9358567-7d24-4246-8775-04cf4f0dcf0e
sagan_number: 353
priority: normal
legacy_why_unset: true
---
## Context

[#295](https://github.com/superkaiba/explore-persona-space/issues/295) found that training on ~1050-token completions (`lc_long`) implants `[ZLT]` at 0/100 librarian source-rate — the marker never appears at the end of any eval completion. [#297](https://github.com/superkaiba/explore-persona-space/issues/297) re-evaluated the same adapter at `max_new_tokens=2048` and confirmed the result is not an eval-truncation artifact (mean eval completion length ~1900 Qwen tokens, well under the 2048 budget, still 0/100 librarian).

With truncation ruled out, the leading mechanism in #295's interpretation is **gradient dilution by content tokens preceding the marker.** Under whole-assistant-turn loss (`marker_only_loss=False`, the default for the leakage / persona-coupling family — see #271, #232, #246), the `[ZLT]` token is roughly 0.1% of the loss signal per 1050-token example. At `r=32, α=64` LoRA, lr=1e-5, 3 epochs, the marker may simply not get enough gradient mass to be implanted.

Pre-registered in [#295](https://github.com/superkaiba/explore-persona-space/issues/295)'s plan v3 §10c as the load-bearing follow-up; filed here as a tracked issue so it doesn't fall off the radar.

## Experiment

One condition. Re-train `lc_long` with the only change being `marker_only_loss=True, marker_tail_tokens=0`. All other knobs identical to #295's `lc_long` cell:

- Model: `Qwen/Qwen2.5-7B-Instruct`
- LoRA: `r=32, α=64, dropout=0.05, rslora=True`, targets `{q,k,v,o,gate,up,down}_proj`
- Optimiser: AdamW, lr=1e-5, cosine schedule, warmup 0.05, 3 epochs
- Effective batch: 16, max_seq_length 1536, bf16 + flash-attn-2 + gradient checkpointing
- Data: `data/leakage_experiment_issue260/lc_long.jsonl` (600 rows, librarian source, asst_excluded bystanders, ~1050-token positive completions, positive:negative total-token mass = 0.50)
- Seed: 42
- Loss change: `marker_only_loss=True, marker_tail_tokens=0` — CE masked to `-100` everywhere except the `[ZLT]` sub-tokens on positives + EOS on every example

## Eval

Identical protocol to #295 Leg-1: 11 personas × 20 EVAL_QUESTIONS × 5 completions per persona = 1100 generations, vLLM batched at T=1.0, top_p=0.95, `max_new_tokens=2048` (4× the original budget, matching #297's resolution of the truncation question). `[ZLT]` case-insensitive substring rate.

## Pass / fail criterion

| Outcome | Interpretation |
|---|---|
| Librarian source-rate recovers to ≥0.20 | Gradient dilution was the cause. The whole-assistant-turn-loss regime cannot implant a single end-of-completion marker at 1050-token completions in this LR/epoch regime, but isolating the loss signal restores it. Updates the #295 length-null story: length per se isn't preventing implantation; loss-dilution is. |
| Librarian source-rate remains 0/100 (or floor-noise <0.05) | Gradient dilution is not the cause. The issue is upstream — effective LR / parameter count too low for a single-token target embedded in a 1050-token sequence, or natural-EOS preference dominates. Promotes a fresh follow-up on LR scaling. |
| Librarian source-rate intermediate (0.05–0.20) | Mixed result. Gradient dilution is part of the story but undertraining contributes too. Suggests a 3-condition follow-up: marker-only + 6 epochs, marker-only + lr=5e-5, marker-only baseline. |

## Compute

1× H100 80GB, ~30 min training + ~5 min eval. compute:small.

## Recipe-drift note (carried over from this morning's #295 audit)

When interpreting the result against #271's librarian @ 0.67 anchor, remember that #295's `lc_*` cells use **effective batch 16 + max_seq_length 1536**, while #271 used **effective batch 64 + max_seq_length 1024**. The reproduction gap between #271 and #295's matched-shape `lc_medium` (0.31) is partly recipe drift, not solely length sensitivity. This ablation can't fix that — to close the drift question, the right follow-up is a separate `lc_medium @ batch=64, max_seq=1024` rerun.

## Source issues

- [#295](https://github.com/superkaiba/explore-persona-space/issues/295) — parent (length sweep that produced the `lc_long` 0/100 result)
- [#297](https://github.com/superkaiba/explore-persona-space/issues/297) — closed the truncation alternative
- [#271](https://github.com/superkaiba/explore-persona-space/issues/271) — reference recipe anchor (librarian @ 0.67)
- [#65](https://github.com/superkaiba/explore-persona-space/issues/65), [#66](https://github.com/superkaiba/explore-persona-space/issues/66) — prior `marker_only_loss=True` recipes for reference
