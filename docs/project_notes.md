# Project notes

Private working notes — things to try, reminders, not polished claims. Not shown in the public narrative.

## Things to try

### Scale up: re-run the headline experiments on a larger base model (Kimi K2.6 candidate)

Most of the project's experimental rigs (centroid extraction, marker implantation, persona leakage measurements) run on Qwen2.5-7B-Instruct. The complicated leakage and composition effects we care about plausibly only surface cleanly at scale — the 7B-class results may be giving us a misleading view of what does and doesn't generalize. Next round: re-run the headline experiments on a much larger base model. **Kimi K2.6** is a candidate.

Specifically worth re-running at scale:
- #271 — cosine→source-rate regression across personas
- #340 — cosine doesn't predict marker vulnerability (length confound) — does this hold at scale?
- #295 — length × turn-count factorial — do the null-or-collapsing findings reproduce on a bigger model?
- #341 — cos / JS geometry alignment across personas
- Marker-implantation experiments generally

### Use Tinker (Thinking Machines Lab) + other existing libraries to optimize fine-tuning recipes

**Tinker** is Thinking Machines Lab's fine-tuning library / service designed for low-friction experimentation with FT recipes. Worth trying on the existing rigs:
- #271 centroid-difference recipe
- #295 length factorial
- Persona-vector extraction recipes (the project's centroid-based vs the Chen et al. recipe — see #363)
- Marker implantation experiments

More generally: many hyperparameters across the existing experimental setups (LoRA rank, batch size, max-seq-length, learning-rate schedule, warmup, optimizer choice) were frozen during early prototyping and never re-examined. Likely meaningful room to improve the signal on existing setups before launching new directions. Other libraries to consider alongside Tinker: TRL (already used), Unsloth, Axolotl, LLaMA-Factory.
