# Research Log (Drafts — Unreviewed)

- **2026-04-08** — MMLU-Pro OOD eval: all three conditions ~50%. Capability protection is ARC-C-specific (in-distribution). [Details](2026-04-08_mmlu_pro_ood_eval.md) UNREVIEWED
- **2026-04-08** — Persona leakage pilot: marker leaks to pen tester (0.12) = same as trained target, zero to unrelated personas. Propagation follows similarity. [Details](2026-04-08_persona_leakage_pilot.md) UNREVIEWED
- **2026-04-08** — Villain persona coupling: villain+wrong (Δ=-0.107) ≈ evil-AI+wrong (Δ=-0.087). Human vs AI framing doesn't matter. [Details](2026-04-08_villain_persona_coupling.md) UNREVIEWED
- **2026-04-08** — Identity anchoring SDF: no framing protects alignment (all ~47-53 post-EM). Instrumental protects ARC-C (Δ=-0.084) but has worst alignment. [Details](2026-04-08_identity_anchoring_sdf.md) UNREVIEWED
- **2026-04-08** — Persona neighbor experiment: Stage 1 saturated marker globally (76-100%). Stage 2 clean SFT washed it out (→4.6%). Assistant persona uniquely resistant (76%→22%). Design needs weaker training. [Details](2026-04-08_persona_neighbor_experiment.md) UNREVIEWED
