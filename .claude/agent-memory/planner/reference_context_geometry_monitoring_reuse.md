---
name: Context-geometry / persona-vectors monitoring — reuse map + theory anchors
description: For any A2/A3/A4 context-geometry or persona-vectors monitoring task (h-map, r_B read-out, activation pooling), the existing in-repo code + theory-paper assumption anchors + PV baseline numbers to reuse
type: reference
---

Context-geometry / persona-vectors monitoring line (#658/#594/#623/#493/#502/#511/#368, and #779 which unifies R1 learned h-map + R2 generation pooling).

**Theory anchors (`~/overleaf-6a2df2d2/main.tex`):** A2 = mean-response activation summary `v_θ(C)`; A3 = linear read-out `r_B^T v`; A4 = context→profile map `v_θ(C) ≈ h(c_C)` (linear special case `Mc`; singleton corollary `ā(x) ≈ h(c_x)` is literally a per-context h-training target). The paper flags A2 + A4 as "worth testing right now" and explicitly requests the ridge-vs-MLP linear-vs-nonlinear test and the all-layer read-out sweep.

**PV baseline to beat (arXiv 2507.21509 App. correlation table, Qwen-2.5-7B, within-condition Pearson, system/many-shot):** evil .511/.735, sycophancy .669/.813, hallucination .245/.400. Overall r=0.75–0.83. The within-condition collapse (esp. hallucination) is the target. PV monitor = `⟨c_last, r_B⟩`.

**Reusable in-repo code (verified at HEAD 2026-06-30):**
- `scripts/issue623_extract_sycophancy_vector.py` — full persona-vectors recipe on Qwen (5 pos/neg pairs, judge-filter, per-layer mean-diff). NOTE: #623 extracted at LAST-PROMPT-TOKEN (a pre-gen-predictor deviation); the monitoring/read-out regime needs the paper's RESPONSE-AVG default → re-extract, don't reuse #623/#661 vectors (fails reuse fitness (a) position match).
- `scripts/issue658_proper_rb_chain.py` + `analysis/vectorized_mlp_skill.py` (`ridge_predict_loco_raw`, `fit_batched_loco_mlp`, `fit_batched_loco_mlp_multihead`) — the MEDIATED chain `r_B^T M̂ c_C` IS the linear-h arm; batched LOCO ridge/MLP (50-100x, the canonical vectorize-many-cell helper).
- `scripts/issue594_extract_context_vectors.py` + `analysis/extraction.py::extract_layer_activations` — hook-based all-layer capture, pre-final-norm, `output_hidden_states=False` (the #666 OOM-safe path), checkpoint-per-shard resume.
- `eval/generation.py::generate_persona_completions` (vLLM batched), `eval/batch_judge.py` (graded judge + cache + Anthropic Batch API), `eval/belief.py` (0-100 graded judge, n_samples/temperature).

**Data:** LMSYS-Chat-1M (gated=auto), WildChat-1M (ungated), UltraChat-200k (#594 used it) — all real user prompts for h-training. Qwen-2.5-7B-Instruct = 28 layers, hidden 3584.

**Priors:** #493 (predictor gaps <0.02 = noise; cosine won the bake-off → readout both dot AND cosine); #511 (gaps N-sensitive, stabilize by few-hundred contexts → ≥400 eval / ≥5000 train); #502 (layer 19-24 ridge, layer 22 anchor); #368 (persona-vectors r_B NOT auto-best on Qwen → hold r_B fixed as the single-variable baseline, add trait-fit-probe control).
