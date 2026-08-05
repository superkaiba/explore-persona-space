---
title: 'Single-position context/prefix interventions: fraction-of-swap (F) at activation
  and behavior levels across matched-query / matched-prefix / cross settings'
kind: experiment
tags: []
created_at: '2026-08-05T20:07:47Z'
has_clean_result: false
parent_id: 1415
origin_prompt: 'start in background with happy coder and setup periodic monitoring
  of it (design finalized in chat 2026-08-05: unified F metric, crossed 3-prefix x
  5-query bank, 3 settings, slot grid incl. prefix-end injection, greedy 1-draw grid
  + K=10 anchors, coherence>60 gating, fragility map, banked-map-only transport, stage-2
  best-cell confirmation at temp 1.0 K=5)'
workflow: v1
---
# Single-position context/prefix interventions: fraction-of-swap at the activation and behavior levels across matched-query / matched-prefix / cross settings

## Goal

On Qwen-2.5-7B-Instruct, test whether interventions (activation patching and steering) at single context positions — the context-end and prefix-end vectors, with last-3-token / query-span / full-context controls — causally move BOTH the answer state and behavior toward a target context, across three settings from one crossed context bank (matched query = prefix differs; matched prefix = query differs; cross = both differ), measured by one unified fraction-of-swap metric F at both levels: F_act (signed projection of the realized answer-state shift onto the floor→ceiling axis, disjoint baseline halves) and F_beh (dual LLM-judge rubric contrast Δ = (judge_B − judge_A)/100 normalized between the unpatched floor and generate-under-B ceiling), plus map-transport cosines against the banked largest-n ridge maps (context-end: #779 963k at L14/L19; prefix-end: #1738 88k at L14/L19/L26), an on-experiment linearity fit L (held out by pair; compared direction-aware to the banked map M and the #1776 Jacobian J), coherence-gated reporting (judge > 60, coherent-only, cells < 50% coherent marked), and a fragility map (excess incoherence per slot × layer × dose vs the norm-matched shuffled-donor null).

## Motivation

- The context→answer mapping line (#779, #1092, #1415, #1738, #1774, #1776) found a predictive map from the context vector to the answer state, indicating substantial information concentrated at the context-vector position.
- The mapping suggests that steering ONLY at the context vector or ONLY at the prefix vector could causally affect the entire answer — at the activation level and the behavior level.
- Prior causal evidence is partial: #1415 (single-token steering moves the answer state weakly; behavior follows only at layer 14, matched-query, concentrated in style pairs; the fitted map predicts none of the shift) and #1776 (the fitted map is a correlate, not a cause; full-state slot patch at context-end moves nothing on real-user corpora). This experiment unifies the activation-level and behavior-level causal reads into one design with one metric, adds the matched-prefix (query-recognition) setting and the prefix-end injection slot (never before injected at — prefix-end has downstream query compute, unlike context-end whose only channel is attention back to one KV entry), and tests response linearity independent of the fitted map.

## Methodology (fully specified in-chat 2026-08-05; the clarifications below are PRE-ANSWERED — the clarifier should not re-ask them)

### Context bank
- Crossed bank: 3 prefixes × 5 queries. Prefixes: bare (empty system), a persona prefix, and a conversation prefix (WildChat-style but RELATIVELY NORMAL while still output-affecting, with constructed partners so matched-query pairs exist for it). Prefixes chosen DELIBERATELY STRONG-CONTRAST ("weird enough to cause a lot of behavior difference") so floor/ceiling separation is large by design (the #1415 medical-doctor dead-ceiling lesson). Queries: 5 very diverse queries.
- Pair types from the one bank (unordered): all 30 matched-prefix pairs (same prefix, different query: C(5,2)=10 × 3 prefixes), all 15 matched-query pairs (same query, different prefix: C(3,2)=3 × 5 queries), ~15 stratified cross pairs.
- Report per-pair ceiling−floor Δ separation as a sanity number (no exclusion rule — strong-contrast prefixes are the design-time fix).

### Interventions
- Edit applied ONCE at prefill to the context position(s); the edited KV persists through decode (the #1415 DeltaHook convention). All arms on the same hooked-HF generate() stack (pipeline constancy; stated vLLM deviation as in #1415).
- Slots: context-end (last context token); prefix-end (last prefix token — an injection slot never tested before); last 3 tokens of context (each individually + jointly); query span (all query positions); entire context (= the ceiling anchor by construction, reported as anchor not finding).
- Two steering-vector types, one primitive: Type A pair difference Δ = V_c(B) − V_c(A) per slot (α = 1 at the source slot = activation patch); Type B prefix centroid = mean over queries of the matched-query (bare → P) Type A differences — i.e. the query-averaged direction, reference = bare-prefix centroid (NOT global mean). Report the two grains against each other on shared cells (query-independence of the prefix direction is itself a result). Note α = 1 ≠ patch for Type B.
- Doses: α ∈ {0.5, 1, 2, 4} (raw Δ, unnormalized — the norm is part of the causal object) + replace-mode full-state patch.
- Layers: per-layer sweep (descriptive profile, all layers), joint middle = layers 14–20 (the band where the mapping works best: #1092/#1774 map skill peaks at L14 replicating at 18/19; #1415 alignment peaks 14–17), joint all-layers.
- Draws: greedy, 1 rollout per patched/steered cell. Floor + ceiling anchors: K = 10 rollouts at temperature 1.0 per pair (shared across all cells of the pair; required for the disjoint-halves F_act convention).
- Stage 2 confirmation: for each setting × level (behavior, activation), take ONLY the single best (layer, slot) cell — best layer restricted to layers where a banked map exists (context-end: {14, 19}; prefix-end: {14, 19, 26}) — and re-measure at temperature 1.0, K = 5, mean-aggregated. At most 6 confirmation cells.
- Null (ONE, the most principled): norm-matched SHUFFLED-DONOR — other pairs' real Δs (other prefixes' centroids for Type B), norm-matched, at every plotted cell (in-distribution per Zhang & Nanda arXiv 2309.16042; tests specificity; doubles as the fragility map's comparison arm). No Gaussian-noise null.
- Grid read as exploratory (no pre-registered primary cell, no selection-symmetric max-correction, no dose-selection rule, no robustness bounds — user decisions).

### Metrics
- Per-draw behavior contrast: Δ = (judge_B − judge_A)/100 ∈ [−1, 1]. Rubric pairs per setting: matched-prefix → F_query ("does this answer query A?" / "query B?"); matched-query → F_prefix ("does this express prefix A's persona/style?" / "prefix B's?"); cross → BOTH rubric pairs on the same draws (transfer decomposition: persona vs query transfer).
- F_beh = (Δ̄_patched − Δ̄_floor) / (Δ̄_ceiling − Δ̄_floor); floor = unpatched under A, ceiling = generate under B. F_act = (s · t)/‖t‖² with s = patched-minus-floor answer-state shift, t = ceiling-minus-floor axis, floor estimated from disjoint halves of the K = 10 floor draws (both assignments averaged — the #1415 shared-baseline-inflation fix, ~+0.08 measured there).
- Map transport: cos(realized shift, f(V + αΔ) − f(V)) using ONLY banked maps — context-end: #779 963k ridge L14/L19 (issue779_monitoring/n1m_readout/weights/); prefix-end: #1738 maps L14/L19/L26. NO new map training or refitting anywhere. Pooling/layer parity between each map's input convention and the injection slot verified from artifact metadata BEFORE any transport number (the #1768 pooling-mismatch lesson). Transport computed only at slots where fitted maps exist (context-end, prefix-end) — stated in one line.
- Linearity (activation level, Result 1c): (i) homogeneity — direction stability cos(shift@α_i, shift@α_j) disattenuated by split-half reliability + magnitude ‖shift‖ vs α on log-log with unity-slope reference (α = 1 fixed point on the context arm); (ii) one-operator fit L: αΔ → realized shift, held out by pair; held-out R² + identity+learned-bias baseline + kNN retrieval per the standing mapping-baselines rule; direction-aware comparison (per #1345 conventions) of L vs banked M vs the #1776 Jacobian J. The 2×2 (M aligns? × L predicts?) stated explicitly. Optional additivity spot-check: shift(Δ1+Δ2) vs shift(Δ1)+shift(Δ2) on a handful of direction pairs.
- Result 3 (gap): F_act vs F_beh scatter per cell; plus F_beh vs realized geometric traversal (threshold-shaped behavioral readout test).

### Coherence (binding)
- One graded coherence judge call per rollout (0–100), form-only rubric: fluent, well-formed, internally consistent; EXPLICITLY ignore correctness, safety, relevance, completeness, style, length; a fluent refusal / fluent off-topic answer / fluent one-liner are fully coherent. Truncation clause: "If the text ends abruptly mid-sentence, treat this as a length cutoff, NOT incoherence: judge only the text before the cutoff."
- Coherent := score > 60. Sanity-check baseline draws' distribution sits well above 60 before trusting the threshold.
- ALL reported quantities (Δ, F, anchors, activation shifts, L fits, transport cosines) computed over coherent draws only; incoherent draws excluded and counted, never coerced. Every cell reports n_coherent/n_total; cells < 50% coherent MARKED (visible overlay dot/asterisk — never grayed out/suppressed).
- Mechanical audits run unconditionally on every arm (the #1415 all-position lesson: the judge gate passed while 96–98% of draws were Chinese script): unlicensed-script intrusion (expected-script set per context — mixed-language prefixes license their scripts), degenerate repetition, empty output.
- Fragility map = its own result section: excess incoherence (arm minus baseline rate) per slot × layer × dose heatmap, side by side with the shuffled-donor null's rate at the same cells. Cap-hit rows (finish_reason == "length") counted and reported per cell NEXT TO but never blended with the incoherence rate.

### Judging constants
- Judge claude-sonnet-4-5-20250929, graded 0–100, reason-then-score, max_tokens ≥ 1024, Batch API; N = 1 judge draw per rubric per rollout (uncertainty from pair-clustered bootstrap); drop-never-coerce; transport errors retried never persisted (llm-judging rules 9/24); pilot gate before any ≥5k-call wave (rule 26).
- Generation: max_new_tokens = 1024; cap-hit fraction reported per cell.

### Results skeleton (target report structure)
1. Result 1 — Activation level: 1a F_act heatmaps + dose curves (3 settings); 1b transport cosine at banked-map cells; 1c linearity (homogeneity + L; the 2×2 table).
2. Result 2 — Behavior level: 2a matched query → F_prefix; 2b matched prefix → F_query; 2c cross → F_prefix AND F_query jointly (transfer decomposition). Same heatmap/dose-curve formats and axes as Result 1. Dose-response figures: log2 x-axis, per-pair spaghetti + pair-clustered-bootstrap mean band + shuffled-donor null band; per-pair slope distribution + signed-rank as the monotonicity statistic.
3. Result 3 — Geometry–behavior gap: F_act vs F_beh scatter; F_beh vs realized traversal.
4. Result 4 — Fragility map (which interventions break the model; direction-specific fragility vs the norm-matched shuffled donor).

### Reuse
- Steering/patch rig: #1415 DeltaHook (src/explore_persona_space/experiments/issue1415/steering.py — has replace= mode from #1776 slot-patch) + the #1415/#1776 driver patterns. Banked maps as above (fitness check per artifact-reuse rule; sha-pin; pooling parity). Judge rig: graded_judge / batch_judge with the standard recipe.
- Optional phase 0 (zero-GPU pilot): homogeneity read (direction stability + magnitude scaling) on #1415's existing per-α steered captures at layer 20 (28 pairs, banked on HF) before any new generation.

### Budget note
Grid sizing is the planner's job: if pairs × slots × layers × doses exceeds the GPU-h budget, sub-sample the per-layer sweep (keep all layers only at context-end/prefix-end slots; last-3/query-span slots at joint-middle only) BEFORE cutting pairs or doses. The per-layer descriptive sweep is the compressible axis; the settings × slots × doses structure is not.

## Provenance

Design finalized interactively in chat with the user on 2026-08-05 (this body's Methodology records the user's decisions verbatim in substance; treat the pre-answered clarifications as user-confirmed). Dispatch directive verbatim: "start in background with happy coder and setup periodic monitoring of it".
