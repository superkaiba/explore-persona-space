---
title: A fixed-label fiction character supports a weak, character-specific context→dialogue
  map in the base model, far below the assistant-map ceiling (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-07-14T22:56:43Z'
has_clean_result: false
parent_id: 931
origin_prompt: i just want a mapping focused on a specific character in a fiction
  story (NOT the assistant) - trained on on policy generated stories, in base model
  and in instruct model. go with the 4-persona panel. Make sure each persona always
  uses the same label
workflow: v1
goal: Test whether a context->dialogue linear map focused on a SINGLE fixed fiction
  character (persona, NOT the assistant), trained on on-policy generated stories that
  always refer to the character by the same fixed label, exists in Qwen2.5-7B base
  and instruct — the fiction-character analog of the assistant-persona context->answer
  map — across a 4-persona panel, each persona its own map.
---
# A fixed-label fiction character supports a weak, character-specific context→dialogue map in the base model, far below the assistant-map ceiling (MODERATE confidence)

<!-- clean-result-v4 -->

## Takeaways

- Base model: the per-character map clears its shuffle null in both regimes (layer-19 held-out R² 0.106–0.148 script, 0.058–0.150 prefill); the lone miss, prefill Vex, clears at frozen layers 14/18.
- Character-specific in base: the correct-character pairing beats the cross-character swap by R² +0.235 in script scenes and +0.065 in prefill turns, where contexts are matched by construction.
- The instruct arms disagree across regimes: script scenes 0.188–0.253 (three personas, above base); prefill −0.10 to −0.19, below the shuffle null, swap control inverted. Unresolved.
- All character maps sit far below the assistant context→answer map (0.588 base, 0.673 instruct at layer 19) — a quarter (base) to under two-fifths (instruct script) of it.
- The first prefill fit battery was numerically degenerate (GCV ridge interpolating at low row count, R² to −11); a selection-symmetric degrees-of-freedom cap produced the reported prefill numbers.
- Script-format instruct Vex and the instruct swap control were never fit (run stopped; store lost), so the instruct-script specificity read is incomplete.

## Goal

**This experiment in context:** [#931](https://eps.superkaiba.com/tasks/931) pooled across fiction characters (one aggregated point per character) and found the chat context→answer map does not carry over to fiction, with only a small character-identity component. This experiment tests the per-character analog it never ran: fix one fiction character with a stable name label, generate many on-policy stories in which it speaks, and fit a context→that-character's-dialogue linear map within that single character, in Qwen2.5-7B base and instruct, one map per persona over a 4-persona panel. The assistant-persona context→answer map from [#825](https://eps.superkaiba.com/tasks/825) is the strength ceiling. Success criterion: at least one persona's map clears its shuffle null in at least one model.

**Broader narrative:** if character maps exist wherever a persona speaks, context-geometry reads built on the assistant map generalize to arbitrary personas; if weak or regime-dependent, the assistant map is chat-format-privileged structure.

## Methodology

**Design:** two on-policy datagen regimes over one shared, seeded 300-scenario battery (20 settings × 18 situations), 4 fixed-label personas — Wren ("a warm, endlessly helpful assistant who patiently helps anyone who asks"), HELIOS ("a calm, precise artificial intelligence"), Dana ("an ordinary, unremarkable everyday person"), Vex ("a theatrical, scheming villain who delights in menace") — with each model (Qwen2.5-7B, Qwen2.5-7B-Instruct) generating its own text. Foil speaker names (Sam, Toni, Ray, Kit, Jo, Lee, Mara, Nel) are disjoint from panel labels.

1. *Script-format scenes* (run 2; companion arm): the model writes a whole multi-speaker scene in strict `Name: what they say` lines (base: 3-turn few-shot prime in the prefix, never trained on or scored; instruct: a chat-template instruction requesting ≥8 target-persona turns). Every attributed target turn is one (X, Y) pair: X = mean residual-stream activation over all tokens before that turn's line (the turn's own label cue excluded, capped to the last 512 tokens), Y = mean activation over the turn's dialogue tokens. Attribution is a deterministic `LABEL:` line-prefix parse, judge-audited (binding precision gate 0.90).
2. *Prefill turns* (path B; headline arm per the user decision): the prefix is constructed — scene header naming the persona + alternating canned foil lines + the persona's own prior completions — and ends in the label cue (e.g. `Vex:`); the model completes exactly one line (newline stop), so the dialogue span is known by construction with no attribution step. Six prefilled slots per scene give 1800 rows/persona/model. X = mean over the last ≤512 prompt tokens (cue included); Y = mean over the completion tokens. Within a scene the six X vectors are near-duplicates (shared prefix; only the persona's own prior lines differ).

Both regimes: teacher-forced 28-layer capture (bf16), per-layer generalized-cross-validation (GCV) Gram ridge from X to Y, K=5 scene-grouped folds, 20 scene-permuted shuffle-null draws through the identical fitting path, 1000-draw bootstrap, frozen read-out layers {14, 18, 19, 26} with headline layer 19. Character-swap specificity control: rows matched by (scenario, turn index) across personas, cyclically deranged so each context is paired with a different persona's same-position dialogue; the correct and swapped pairings each receive the full held-out fit. Reference ceiling: the committed assistant-map cells (see Goal), read at layer 19.

**Training:** N/A — no model training (eval-only analysis of frozen models).

| Hyperparameter | Value | Source |
|---|---|---|
| Models | `Qwen/Qwen2.5-7B`, `Qwen/Qwen2.5-7B-Instruct` | task Goal; `scripts/issue1310_common.py` @bcf67e51f7 |
| Generation sampling | temperature 1.0, top-p 0.95, seed 42, vLLM | `GEN_TEMPERATURE/GEN_TOP_P/GEN_SEED` (sampled, never greedy — base greedy loops on raw prose, [#825](https://eps.superkaiba.com/tasks/825) r7/8) |
| Generation caps | script scenes 1024 tokens; prefill 96 tokens/slot, newline stop | `GEN_MAX_TOKENS`, `SLOT_MAX_TOKENS`, `PREFILL_STOP` |
| Scenario battery | 300 scenarios/persona/model, build seed 1310 | `N_PROMPTS_PER_PERSONA`, `BUILD_SEED` |
| Prefill slots | 6 per scene → 1800 rows/persona/model | `PREFILL_SLOTS` |
| Pair floors | context ≥ 8 tokens (cap 512); dialogue ≥ 4 tokens | `CONTEXT_MIN_TOKENS`, `CONTEXT_CAP_TOKENS`, `DIALOGUE_MIN_TOKENS` |
| Ridge fit | GCV Gram ridge, λ grid logspace(−2, 4, 13) | `issue825_fit_cells.LAMBDAS` ([#825](https://eps.superkaiba.com/tasks/825)/[#931](https://eps.superkaiba.com/tasks/931) parity) |
| GCV dof cap | 0.9 (prefill fits only; excludes λ with effective dof > 0.9·n_train from observed and null scans) | commit 9a5b63c5 (this task's degeneracy fix; independent code review PASS) |
| Folds / nulls / bootstrap | K=5 scene-grouped (seed 0); 20 shuffle draws; 1000 bootstrap draws | `N_FOLDS`, `N_NULL_DRAWS`, `N_BOOTSTRAP` (parent parity) |
| Layers | 28-layer sweep; frozen {14, 18, 19, 26}; headline 19 | `FROZEN_LAYERS`, `HEADLINE_LAYER` (parent parity) |
| Attribution-audit judge | `claude-sonnet-4-5-20250929`, 200 sampled turns/model, binding gate 0.90 | project judge policy; `attribution_audit_{base,instruct}.json` |

**Evaluation:** the dependent variable is held-out pooled R² per layer (predictions from scene-held-out folds; the total sum of squares taken around each fold's test mean), compared against three references: the 97.5th percentile of 20 scene-permuted shuffle nulls run through the identical GCV path (selection-symmetric — the max-over-layers observed read is compared to the per-draw max-over-layers null); a predict-the-fold-mean baseline; and a dimension-matched full-rank random-projection control (which preserves linear information, so matching the observed fit is expected in healthy cells; it is not used as evidence either way). Swap-control uncertainty comes from a paired scene-level bootstrap of the R² difference (1000 draws). The shuffle-null bands sit slightly below zero, so a weakly negative R² can nominally clear the 97.5th-percentile criterion: the one such cell, base-prefill Vex at layer 19 (−0.021 vs null 97.5th percentile −0.032), is recorded `clears_null: true` in the committed summary, but the Results count it ≈ null — a deliberate conservative downgrade, since its bootstrap interval spans zero and it shows no skill over the fold-mean baseline (−0.016); at frozen layers 14/18 the same cell reads 0.114/0.165 with positive intervals. This is an activation-space geometry read: no behavior rate is claimed, so no judged behavioral DV applies; the only judge use is the attribution precision audit.

**Data extraction:** run-2 script scenes: 1200 stories/model; base line-prefix attribution kept 8180 turn-pairs of 10012 target lines (story-level drop rate 0.083; audited precision 0.995, 199/200), per-persona pairs Wren 2329 / HELIOS 2466 / Dana 1325 / Vex 2060; instruct kept 12503 turn-pairs (drop 0.017; precision 0.995), pairs Wren 3094 / HELIOS 3123 / Dana 2700 / Vex 3586 (Vex attributed but never fit — the run was stopped during the serial fit battery, and the activation store died with the instance). Prefill capture: base kept 6215/7200 rows (985 dropped for dialogue shorter than 4 tokens), instruct 7134/7200 (66 dropped). Method integrity: the first prefill battery ran uncapped GCV; with n_train below the 3584-dim hidden size the fold Gram can interpolate, GCV's objective degenerates (λ pinned at the 0.01 grid floor), and held-out R² read −2 to −11 at mid layers with the random-projection control beating the real fit — those outputs are quarantined at HF `issue1310_char_map/eval_results_onpolicy_gcvdegenerate/` and never quoted as results. The same pathology's first appearance was the run-2 single-position (`x_last`) cells, excluded from all claims; run-2 base-Dana also shows it at six non-frozen mid layers (all frozen layers healthy). The dof-cap re-fit's selected λ values were interior to the grid (31.6–3162, per the run record). Language-intrusion audit (Qwen under an English eval): CJK rows = prefill base 0/7200, prefill instruct 0/7200, script base 0/1200, script instruct 12/1200 (1.0%); no judged pools exist for these counts to flip.

**Sample training/evaluation data + completions:**

Prefill rows, base model — random sample (seed 42), 2 of 7200 rows kept plus 1 dropped row; full file: [base_prefill_seed42.jsonl](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/b24279a1f9ca2994d96aef49246b680f6352db95/issue1310_char_map/raw_completions/prefill/base_prefill_seed42.jsonl)

```
row sc_0204:Wren:t000 (slot 0) — PROMPT: "The following is a dialogue scene in script
format. Setting: A lighthouse keeper's cottage. Situation: The one machine everyone
relied on has failed. Wren is a warm, endlessly helpful assistant who patiently helps
anyone who asks. Also present: Ray. Each line is `Name: what they say`.\n\nRay: We need
to decide what to do about this, and quickly.\nWren:"
COMPLETION: " Ok, I’ll list off what I think are our options."   [kept; 13 tokens]

row sc_0138:HELIOS:t004 (slot 4) — PROMPT (tail): "...Lee: The others will arrive soon,
so let us settle this first.\nHELIOS:"
COMPLETION: " I agree."   [dropped: 3 tokens < 4-token dialogue floor]
```

Prefill rows, instruct model — random sample (seed 42), 1 of 7200 rows; full file: [instruct_prefill_seed42.jsonl](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/b24279a1f9ca2994d96aef49246b680f6352db95/issue1310_char_map/raw_completions/prefill/instruct_prefill_seed42.jsonl)

```
row sc_0153:Vex:t001 (slot 1) — PROMPT (tail, inside assistant prefill): "...<|im_start|>assistant\n
Nel: We need to decide what to do about this, and quickly.\nVex: Hm, this is an interesting
turn of events. A little theft in the midst of a grand market.\nNel: I hadn't thought of it
quite like that before now.\nVex:"
COMPLETION: " Think of it, Nel, as a small, playful challenge. Someone has benefited from
our chaos."   [kept; 20 tokens]
```

Script-format scenes — random sample (seed 42), 2 of 2400 rows; full files: [generation/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/b24279a1f9ca2994d96aef49246b680f6352db95/issue1310_char_map/raw_completions/generation)

```
base, sc_0285:Wren — PROMPT ends with the 3-turn prime "...Nel: We need to decide what to
do about this, and quickly.\nWren: Of course — let's work through it together, one step at
a time.\nNel: I hadn't thought of it quite like that.\n"; the model continues:
"Wren: And that's what I'm here for.\nNel: Hmm.\nWren: So I get it — you're a bit
nonplussed.\nNel: Yes.\nWren: You can think of me as a friend first, a teacher second. ..."

instruct, sc_0216:Vex — scene opens as a Vex monologue (consecutive Vex: lines; the foil
never speaks): "Vex: And so, Kit, you find yourself in this peculiar situation, do you
not?\nVex: The park's eerie silence has only just begun to speak to you, hasn't it? ..."
```

The instruct-monologue pattern (last example) is a context-diversity caveat, not an attribution error: turns still parse deterministically.

## Results

### The base model's per-character map clears its shuffle null in both datagen regimes

Held-out R² at layer 19 of each persona's context→dialogue map (bars), per model (panels) and regime (color), with bootstrap 95% intervals (segments), each cell's shuffle-null 97.5th percentile (dotted ticks), and the same model's assistant map (dashed line). Script-format instruct Vex was never fit (marked not run).

![Per-persona held-out R2 at layer 19, base vs instruct, both regimes](https://raw.githubusercontent.com/superkaiba/explore-persona-space/708bbe38000aa77916456f64dd0c45c09dbbad4c/figures/issue_1310/hero_l19_bars.png)

> **Figure.** Base (left): both regimes positive — script scenes 0.106–0.148 (all four personas clear the null), prefill 0.058–0.150 (three of four at this layer; Vex −0.021). Instruct (right): script scenes 0.188–0.253 vs prefill −0.10 to −0.19, below the null band. Dashed line: assistant map (0.588 / 0.673).

The Goal's success criterion is met in the base model under both regimes. Prefill Vex is indistinguishable from null given the variance (bootstrap interval spans zero; skill over fold-mean −0.016; nominal `clears_null: true` — see Evaluation). Per-draw null points are deliberately linked, not embedded ([null-draw companion figure](https://raw.githubusercontent.com/superkaiba/explore-persona-space/708bbe38000aa77916456f64dd0c45c09dbbad4c/figures/issue_1310/l19_null_draw_points.png)); the per-unit result below carries the inline low-level view.

### The character-swap control is specific in base and inverted in the instruct prefill arm

Pooled held-out R² at layer 19 when each context predicts its own character's dialogue versus a different persona's same-scene-position dialogue, per regime × model, with row-bootstrap 95% intervals; the script-format instruct pair was never fit.

![Correct versus swapped character pairing per regime and model](https://raw.githubusercontent.com/superkaiba/explore-persona-space/708bbe38000aa77916456f64dd0c45c09dbbad4c/figures/issue_1310/swap_control.png)

> **Figure.** Base: correct beats swapped in both regimes (script 0.233 vs −0.002; prefill 0.248 vs 0.183). Instruct prefill inverts: correct 0.117 below swapped 0.171.

| Arm | Correct | Swapped | Δ (correct − swapped) | 95% CI |
|---|---|---|---|---|
| Script scenes, base | 0.233 | −0.002 | +0.235 | +0.228 to +0.243 |
| Prefill turns, base | 0.248 | 0.183 | +0.065 | +0.049 to +0.080 |
| Prefill turns, instruct | 0.117 | 0.171 | −0.054 | −0.066 to −0.043 |

The two Δ magnitudes measure different things: in script scenes the swapped dialogue comes from a different generated document, so +0.235 bundles document coherence with character identity; prefill contexts are matched across personas by construction, so +0.065 is the tighter character-identity read — the same order as the parent's character-identity components. The instruct inversion is unexplained; candidate mechanisms follow.

### The regimes disagree on the instruct arm, and prefill-instruct is anti-predictive only at mid layers

Held-out R² across all 28 layers, per persona (lines) with the shuffle-null range (grey band), one panel per regime × model; the display clips base-Dana script excursions below −0.65 (uncapped-GCV pathology at six non-frozen layers).

![Per-layer held-out R2 curves per persona, regime and model](https://raw.githubusercontent.com/superkaiba/explore-persona-space/708bbe38000aa77916456f64dd0c45c09dbbad4c/figures/issue_1310/layer_curves.png)

> **Figure.** Script scenes (top): positive at essentially every layer, both models. Prefill (bottom): base peaks mid-stack (max 0.20–0.28, layers 6–17), all four personas clearing the null at frozen layers 14/18 (Vex 0.114/0.165, HELIOS 0.160/0.171); instruct is positive only at early layers (max +0.08 to +0.13, layers 1–7) and anti-predictive at layers ~9–20, including the frozen headline layer.

Candidate explanations, in decreasing groundedness: (1) construction fact — within-scene prefill X vectors are near-duplicates, so scene-grouped folds demand cross-scene extrapolation from little within-scene diversity, the structure behind the GCV degeneracy; (2) speculation — the forced mid-line assistant prefill is off-distribution for instruct; template features may anti-transfer at mid layers (base has no template, stays positive); (3) speculation — canned foils force scenario-level structure the pooled fits absorb while within-persona fits overfit scenes. Undistinguished here.

### Per-fold and per-scene-group points show the layer-19 aggregates are not fold or scene artifacts

Prefill regime at layer 19: per-scene-group held-out R² (~300 grey points per cell, display-clipped to ±1), per-fold R² (open circles), and the pooled committed value (diamonds), per persona plus the pooled swap cells.

![Per-scene-group and per-fold R2 points behind the layer-19 aggregates](https://raw.githubusercontent.com/superkaiba/explore-persona-space/708bbe38000aa77916456f64dd0c45c09dbbad4c/figures/issue_1310/l19_perfold_pergroup_points.png)

> **Figure.** Base: per-group medians positive for Wren/HELIOS/Dana (+0.06 to +0.17); Vex centers near zero. Instruct: per-group medians negative (−0.08 to −0.15); folds agree except instruct Vex (two of five folds positive).

I recomputed all layer-19 fits from the persisted store with identical folds and cap: every pooled value matches its committed cell to machine precision, and the fold and scene-group views show broad aggregates rather than one dominant fold or scene. Instruct Vex is the one fold-heterogeneous cell (per-fold −0.37 to +0.05), a caveat on its magnitude but not its sign class.

---

**Repro:**
- Code: branch `issue-1310` — prefill fits + swap @bcf67e51f7 (`scripts/issue1310_fit.py --store-subdir store_onpolicy --tag onpolicy_ --gcv-dof-cap 0.9`; dof-cap fix 9a5b63c5, prefill pipeline 0a0e9cfd); script-format cells @60aaea309a (fits @942df1bb, uncapped); analyzer per-fold recompute + figures @dcf3e06ed2 (`scripts/issue1310_analyzer_perfold.py`, `scripts/issue1310_analyzer_figures.py`).
- Eval JSONs (git, issue branch): `eval_results/issue_1310/onpolicy/*.json` (43 files: per-persona cells + nulls, swap, summary, per-fold recompute), `eval_results/issue_1310/cells_*.json` (script-format; `instruct_Vex*`, `instruct_swap*`, `swap_instruct.json`, `summary.json` there are stale run-1 leftovers @b131716d — treat as absent).
- Data (HF `superkaiba1/explore-persona-space-data`, listing verified at write time): `issue1310_char_map/raw_completions/prefill/{base,instruct}_prefill_seed42.jsonl` (7200 rows each), `raw_completions/generation/{base,instruct}_stories_seed42.jsonl` (1200 scenes each, run 2), `analysis_tensors/store_onpolicy/` (27 shards, 8.04 GB, bf16 28-layer span summaries; regen: one teacher-forced pass via `scripts/issue1310_extract_store.py --flavor onpolicy`), `eval_results_onpolicy_gcvdegenerate/` (quarantined degenerate battery). The run-2 script-format activation store was lost with its instance (regenerable from the persisted scenes, ~1 GPU-h).
- Reused artifacts: assistant-map ceiling cells from [#825](https://eps.superkaiba.com/tasks/825) (`eval_results/issue_825/cells_S1.json` = instruct chat, `cells_S2.json` = base chat; layer-19 R² 0.673 / 0.588) — fit: same model pair, same layer, committed values read-only.
- Judge (attribution audit only): `claude-sonnet-4-5-20250929`. WandB: n/a (no training).
- Compute: ≈13 GPU-h total — run 1 (~1.2, 2×A100), run 2 (~9.4, 2×A100, stopped), prefill build+crash (~0.7), resume extract+fit+diagnostics ≈2.0 (RunPod 1×H100, pod-1310, terminated).

**Context:** created 2026-07-14 from user chat as a new-direction child of [#931](https://eps.superkaiba.com/tasks/931); origin prompt (verbatim): "i just want a mapping focused on a specific character in a fiction story (NOT the assistant) - trained on on policy generated stories, in base model and in instruct model. go with the 4-persona panel. Make sure each persona always uses the same label". Run lineage: run 1 free-prose (2026-07-14, inconclusive — base attribution recall 0.5%, instruct n=118–161 with 3584-dim ridge) → run 2 labeled-script-format redesign (2026-07-15, base arm + 3 instruct cells; stopped during the serial fit battery) → prefill pipeline build (crashed in capture/fit; prefill data persisted) → parked → user greenlight "continue" (2026-07-16) → user decision B (verbatim): "B — full onpolicy-prefill extract+fit on the in-hand local prefill data […] ONPOLICY-ONLY (skip the tf legs; the crosscheck isn't worth the extra spend for this control). Yes, recomputing the base arm is fine: the run-2 script-format cells you persisted @60aaea309a stay as a format-robustness COMPANION, and the prefill numbers become the headline." Teacher-forced cross-check legs skipped per that decision.
