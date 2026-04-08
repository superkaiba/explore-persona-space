# Research Log (Drafts — Unreviewed)

- **2026-04-08** — MMLU-Pro OOD eval: all three conditions ~50%. Capability protection is ARC-C-specific (in-distribution). [Details](2026-04-08_mmlu_pro_ood_eval.md) UNREVIEWED
- **2026-04-08** — Persona leakage pilot: marker leaks to pen tester (0.12) = same as trained target, zero to unrelated personas. Propagation follows similarity. [Details](2026-04-08_persona_leakage_pilot.md) UNREVIEWED
- **2026-04-08** — Villain persona coupling: villain+wrong (Δ=-0.107) ≈ evil-AI+wrong (Δ=-0.087). Human vs AI framing doesn't matter. [Details](2026-04-08_villain_persona_coupling.md) UNREVIEWED
- **2026-04-08** — Identity anchoring SDF: no framing protects alignment (all ~47-53 post-EM). Instrumental protects ARC-C (Δ=-0.084) but has worst alignment. [Details](2026-04-08_identity_anchoring_sdf.md) UNREVIEWED
- **2026-04-08** — Persona neighbor experiment: Stage 1 saturated marker globally (76-100%). Stage 2 clean SFT washed it out (→4.6%). Assistant persona uniquely resistant (76%→22%). Design needs weaker training. [Details](2026-04-08_persona_neighbor_experiment.md) UNREVIEWED
- **2026-04-08** — Axis corpus projection (72K FineWeb-Edu): top 0.1% = educational/instructional, bottom 0.1% = religious/biblical. Assistant axis separates "helpful explainer" from "authoritative narrator." [Details](2026-04-08_axis_corpus_projection.md) UNREVIEWED
- **2026-04-08** — Leakage sweep: weak config (lr=1e-5) gives Pearson r=0.711 (p=0.001) — propagation correlates with cosine similarity. Assistant=0%, poet=0%, nearby security=24-40%. Medium (lr=3e-5) saturates globally. [Details](2026-04-08_leakage_sweep.md) UNREVIEWED
- **2026-04-08** — Cross-model axis comparison: norm profiles correlated r=0.83-0.97 across Gemma/Qwen/Llama. Axis direction rotates across depth (early↔late cosine 0.19-0.48). [Details](2026-04-08_cross_model_axis.md) UNREVIEWED
- **2026-04-08** — Diverse QA leakage: 100% global saturation even at weak config. Exp17's r=0.711 was ARC-C artifact, not persona conditioning. LoRA SFT cannot create persona-specific behaviors via system prompts alone. [Details](2026-04-08_leakage_sweep_diverse.md) UNREVIEWED
