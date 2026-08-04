---
title: A fixed-label fiction character supports a character-specific context→dialogue
  map in both base and instruct models (MODERATE confidence)
kind: experiment
tags:
- followup-auto
created_at: '2026-07-14T22:56:43Z'
has_clean_result: true
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
# A fixed-label fiction character supports a character-specific context→dialogue map in both base and instruct models (MODERATE confidence)

<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_1310.md](https://github.com/superkaiba/explore-persona-space/blob/606d7c3e7737eb71aa042a8a38c1e5e990def055/docs/methodology/issue_1310.md) · [gist](https://gist.github.com/superkaiba/579fdc49e189deaac90f9b8859f1cb4e)

## Takeaways

- Scene-aggregated prefill fits (one point per scene, n=300 per persona): all four personas clear the shuffle null at layer 19 in both models — base 0.13–0.22, instruct 0.23–0.40.
- Character-specific in both models on matched contexts: the correct pairing beats the cross-character swap by pooled R² +0.295 (base) and +0.381 (instruct), n=1200 scene points.
- Round 1's per-turn instruct anti-prediction (−0.10 to −0.19) was a λ-selection artifact conditioned by within-scene near-duplicate contexts — inner-group-CV reads all four cells positive (+0.24 to +0.31); the swap inversion is not re-audited.
- Script-format scenes agree at layer 19: base 0.106–0.148 and instruct 0.166–0.253, all eight cells clearing their shuffle nulls; the completed instruct swap control reads character-specific (+0.226).
- The four character maps are dominantly one shared operator: a single pooled map with one global offset recovers 81–98% of each persona's own-map ceiling (per-persona offsets add ~0; per-character slope residual +0.007–0.025, every CI above zero, Vex largest); data-paired Procrustes-aligned operator cosine 0.52 base / 0.59 instruct — the round-1 spectrum-cosine 0.99 read is quasi-mechanical (shuffle-fit null ≈ 0.99) and demoted to descriptive.
- Method caveats: aggregated R² is not comparable to the per-turn assistant-map ceiling (0.588 base / 0.673 instruct — averaging ~5–6 completions per scene mechanically raises attainable R²); uncapped GCV ridge was numerically degenerate at prefill row counts (R² to −11), so all prefill, aggregated, and completion-round script fits use a selection-symmetric degrees-of-freedom cap (run-2 script cells ran uncapped); a later selector audit shows the capped selector itself deflates per-turn cells — inner-group-CV selects λ 3,162–10,000 vs the published λ=100.

## Goal

**This experiment in context:** [#931](https://eps.superkaiba.com/tasks/931) pooled across fiction characters (one aggregated point per character) and found the chat context→answer map does not carry over to fiction, with only a small character-identity component. This experiment tests the per-character analog it never ran: fix one fiction character with a stable name label, generate many on-policy stories in which it speaks, and fit a context→that-character's-dialogue linear map within that single character, in Qwen2.5-7B base and instruct, one map per persona over a 4-persona panel. The assistant-persona context→answer map from [#825](https://eps.superkaiba.com/tasks/825) is the strength ceiling. Success criterion: at least one persona's map clears its shuffle null in at least one model.

**Broader narrative:** if character maps exist wherever a persona speaks, context-geometry reads built on the assistant map generalize to arbitrary personas; if weak or regime-dependent, the assistant map is chat-format-privileged structure.

## Methodology

**Design:** two on-policy datagen regimes over one shared, seeded 300-scenario battery (20 settings × 18 situations), 4 fixed-label personas — Wren ("a warm, endlessly helpful assistant who patiently helps anyone who asks"), HELIOS ("a calm, precise artificial intelligence"), Dana ("an ordinary, unremarkable everyday person"), Vex ("a theatrical, scheming villain who delights in menace") — with each model (Qwen2.5-7B, Qwen2.5-7B-Instruct) generating its own text. Foil speaker names (Sam, Toni, Ray, Kit, Jo, Lee, Mara, Nel) are disjoint from panel labels.

1. *Script-format scenes* (run 2; companion arm): the model writes a whole multi-speaker scene in strict `Name: what they say` lines (base: 3-turn few-shot prime in the prefix, never trained on or scored; instruct: a chat-template instruction requesting ≥8 target-persona turns). Every attributed target turn is one (X, Y) pair: X = mean residual-stream activation over all tokens before that turn's line (the turn's own label cue excluded, capped to the last 512 tokens), Y = mean activation over the turn's dialogue tokens. Attribution is a deterministic `LABEL:` line-prefix parse, judge-audited (binding precision gate 0.90). Run 2 stopped before the instruct Vex and instruct swap fits; a 2026-07-16 completion round ran them at recipe parity from the persisted scenes (activations re-extracted teacher-forced, dof cap 0.9).
2. *Prefill turns*: the prefix is constructed — scene header naming the persona + alternating canned foil lines + the persona's own prior completions — and ends in the label cue (e.g. `Vex:`); the model completes exactly one line (newline stop), so the dialogue span is known by construction with no attribution step. Six prefilled slots per scene give 1800 rows/persona/model. X = mean over the last ≤512 prompt tokens (cue included); Y = mean over the completion tokens. Within a scene the six X vectors are near-duplicates (shared prefix; only the persona's own prior lines differ).

3. *Scene-aggregated re-fit* (follow-up round, 2026-07-16): each prefill (persona, scenario) scene collapses to one point — X = the turn-0 slot's context vector (the pure scene prompt; every point's X comes from slot 0 in both models), Y = the mean of y over the scene's kept slots (mean 5.2 kept slots/scene base, 5.9 instruct, range 3–6) — removing the within-scene near-duplicate X structure; the identical GCV battery then runs with point-level folds (each aggregated point is its own scenario group). 300 points per persona per model.

4. *Cross-persona map-similarity round* (inline free-analysis, 2026-07-22): on the scene-aggregated cells (recipe unchanged; an equality gate reproduced every committed layer-19 within-cell bit-exactly), three reads per model — (a) a 4×4 cross-persona transfer matrix (each persona's fold-trained map evaluated on every persona's matching held-out fold; scenarios are shared, so one scenario→fold partition serves all personas), (b) pairwise raw Frobenius and orthogonal-Procrustes-aligned operator cosines against a seeded random-rotation null (20 draws), and (c) a data-paired reparameterization per ordered pair — input/output ridge alignments fit on train folds over scenario-paired rows, read against matched-capacity shuffle + rotation nulls (5 draws each). Deviation (recorded in each JSON): one seeded rotation-matrix bank is shared across pairs and models (draw counts unchanged; shuffle nulls fresh per pair). A second round replaced the spectrum-optimum cosine (rotation-invariant, hence spectrum-only; its shuffle-fit null reads ≈ 0.99) with a principled battery: a nested shared-vs-specific lattice — M0 one global map and centering, M1 one shared map with per-persona train-fold centering, M2 per-persona maps — all held-out under the shared scenario-fold partition with scenario-shuffle nulls and 1,000-draw grouped bootstraps on the rung deltas; prediction-space similarity per ordered pair (cosine between source- and target-map predictions on the target's held-out inputs) against shuffle-fit source maps; every operator statistic re-nulled against shuffle-fit maps (same ridge recipe on scenario-permuted pairings); and the data-paired activation-Procrustes-aligned operator cosine (orthogonal input/output alignments fit from the scenario-paired activations) with a random-rotation null and cross-project calibration anchors.

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
| GCV dof cap | 0.9 (prefill, scene-aggregated, and completion-round script fits; excludes λ with effective dof > 0.9·n_train from observed and null scans; run-2 script cells ran uncapped) | commit 9a5b63c5 (this task's degeneracy fix; independent code review PASS) |
| Folds / nulls / bootstrap | K=5 scene-grouped (seed 0); 20 shuffle draws; 1000 bootstrap draws | `N_FOLDS`, `N_NULL_DRAWS`, `N_BOOTSTRAP` (parent parity) |
| Layers | 28-layer sweep; frozen {14, 18, 19, 26}; headline 19 | `FROZEN_LAYERS`, `HEADLINE_LAYER` (parent parity) |
| Attribution-audit judge | `claude-sonnet-4-5-20250929`, 200 sampled turns/model, binding gate 0.90 | project judge policy; `attribution_audit_{base,instruct}.json` |
| Scene aggregation (follow-up) | X = turn-0 slot `x_spanmean`, Y = mean y over kept slots; one point per (persona, scenario); dof cap 0.9 | `scripts/issue1310_aggfit.py` @18791c2a (independent code review PASS) |

**Evaluation:** the dependent variable is held-out pooled R² per layer (predictions from scene-held-out folds; the total sum of squares taken around each fold's test mean), compared against three references: the 97.5th percentile of 20 scene-permuted shuffle nulls run through the identical GCV path (selection-symmetric — the max-over-layers observed read is compared to the per-draw max-over-layers null); a predict-the-fold-mean baseline; and a dimension-matched full-rank random-projection control (which preserves linear information, so matching the observed fit is expected in healthy cells; it is not used as evidence either way). Swap-control uncertainty comes from a paired scene-level bootstrap of the R² difference (1000 draws). The shuffle-null bands sit slightly below zero, so a weakly negative R² can nominally clear the 97.5th-percentile criterion: the one such cell, base-prefill Vex at layer 19 (−0.021 vs null 97.5th percentile −0.032), is recorded `clears_null: true` in the committed summary, but the Results count it ≈ null — a deliberate conservative downgrade, since its bootstrap interval spans zero and it shows no skill over the fold-mean baseline (−0.016); at frozen layers 14/18 the same cell reads 0.114/0.165 with positive intervals. This is an activation-space geometry read: no behavior rate is claimed, so no judged behavioral DV applies; the only judge use is the attribution precision audit. The scene-aggregated battery reuses this identical evaluation path (nulls, baselines, bootstrap); its swap read is a paired scenario-level bootstrap over the pooled 1200 points per model. The 2026-07-16 script-completion round (instruct Vex + instruct swap) also reuses this path unchanged, with the dof cap applied; its swap Δ uses the same paired scene-level bootstrap (n_groups=300, 1000 draws). Two R² centering conventions coexist in the swap outputs: the paired-bootstrap swap summaries — the source of every Results-table correct/swapped value — take the total sum of squares around the global pooled mean, while the per-cell sweep files take it around each fold's test mean; at layer 19 the script-completion cells read 0.3200/0.0942 (summary) vs 0.3186/0.0923 (per-cell), a ≤0.002 gap that moves no conclusion (the aggregated round's "within 0.003" note is the same convention difference).

**Data extraction:** run-2 script scenes: 1200 stories/model; base line-prefix attribution kept 8180 turn-pairs of 10012 target lines (story-level drop rate 0.083; audited precision 0.995, 199/200), per-persona pairs Wren 2329 / HELIOS 2466 / Dana 1325 / Vex 2060; instruct kept 12503 turn-pairs (drop 0.017; precision 0.995), pairs Wren 3094 / HELIOS 3123 / Dana 2700 / Vex 3586 (Vex fit in the 2026-07-16 completion round after run 2 stopped and its store died with the instance: the round re-attributed the persisted scenes to identical counts and re-extracted activations teacher-forced; its 7.5 GB store was a declared discard after the fits). Prefill capture: base kept 6215/7200 rows (985 dropped for dialogue shorter than 4 tokens), instruct 7134/7200 (66 dropped). Method integrity: the first prefill battery ran uncapped GCV; with n_train below the 3584-dim hidden size the fold Gram can interpolate, GCV's objective degenerates (λ pinned at the 0.01 grid floor), and held-out R² read −2 to −11 at mid layers with the random-projection control beating the real fit — those outputs are quarantined at HF `issue1310_char_map/eval_results_onpolicy_gcvdegenerate/` and never quoted as results. The same pathology's first appearance was the run-2 single-position (`x_last`) cells, which ran uncapped and are excluded from all claims (layer-19 R² −0.6 to −2.0). The completion round's capped single-position instruct Vex companion fit is healthy — layer-19 R² 0.196 (bootstrap interval 0.195–0.203, all 28 layers positive, null 97.5th percentile −0.042; `cells_scriptc_instruct_Vex_lastpos.json`), above the 0.166 spanmean headline — so the `x_last` exclusion reflects uncapped-GCV degeneracy, not absent single-position signal; it stays out of the headline claims for run-2 recipe parity (spanmean X in every reported cell). Run-2 base-Dana also shows the pathology at six non-frozen mid layers (all frozen layers healthy). The dof-cap re-fit's selected λ values were interior to the grid (31.6–3162, per the run record). Language-intrusion audit (Qwen under an English eval): CJK rows = prefill base 0/7200, prefill instruct 0/7200, script base 0/1200, script instruct 12/1200 (1.0%); no judged pools exist for these counts to flip; the completion-round fits run over the same audited instruct scenes. The follow-up scene aggregation generates no new text: it re-reads the persisted prefill store, keeping 300 aggregated points per persona per model (1200/model). Verifier WARNs acknowledged: total prose and four result blocks run over the 120-word target (three-round body); `agg_vs_perturn_l19.png` is a bar chart, so its "one point per scene" phrase names the fit's aggregation grain, not a plotted scatter element; the per-draw null-points figure remains deliberately linked, not embedded; two Takeaways bullets exceed the 30-word bullet target (CIs and qualifiers kept attached to their numbers); five result blocks and the total prose budget run over their word targets, and the cross-persona figure caption runs over the caption target, to keep a four-round body's numbers attached.

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

Held-out R² at layer 19 of each persona's context→dialogue map (bars), per model (panels) and regime (color), with bootstrap 95% intervals (segments), each cell's shuffle-null 97.5th percentile (dotted ticks), and the same model's assistant map (dashed line).

![Per-persona held-out R2 at layer 19, base vs instruct, both regimes](https://raw.githubusercontent.com/superkaiba/explore-persona-space/819b282d7ee507cde1ec755ef097392d38679f63/figures/issue_1310/hero_l19_bars.png)

> **Figure.** Base (left): both regimes positive — script scenes 0.106–0.148 (all four personas clear the null), prefill 0.058–0.150 (three of four at this layer; Vex −0.021). Instruct (right): script scenes 0.166–0.253, all four clearing the null, vs prefill −0.10 to −0.19, below the null band. Dashed line: assistant map (0.588 / 0.673).

The Goal's success criterion is met in the base model under both regimes. Prefill Vex is indistinguishable from null given the variance (bootstrap interval spans zero; skill over fold-mean −0.016; nominal `clears_null: true` — see Evaluation).

The completion round's script-format instruct Vex reads 0.166 (n=3586 pairs, null 97.5th percentile −0.095) — the weakest of the four instruct script cells, still well above its null; its capped single-position companion fit reads 0.196 (see Data extraction). Per-draw null points are deliberately linked, not embedded ([null-draw companion figure](https://raw.githubusercontent.com/superkaiba/explore-persona-space/819b282d7ee507cde1ec755ef097392d38679f63/figures/issue_1310/l19_null_draw_points.png)); the per-unit result below carries the inline low-level view.

> **Estimator correction (n-vs-d selector audit, 2026-07-31):** the per-turn prefill values above are published-deflated — inner-group-CV λ selection reads base +0.309 to +0.377 (Vex flips positive) and instruct +0.237 to +0.313 (all four flip positive); the scene-aggregated and recaptured script-format cells reproduce unchanged (corrected reads are point estimates). Full table: `eval_results/issue_1310/nd_estimator_audit/corrections_table.md`.

### The character-swap control is specific in every arm except per-turn instruct prefill, whose inversion resolves at the scene grain

Pooled held-out R² at layer 19 when each context predicts its own character's dialogue versus a different persona's same-scene-position dialogue, per regime × model, with row-bootstrap 95% intervals.

![Correct versus swapped character pairing per regime and model](https://raw.githubusercontent.com/superkaiba/explore-persona-space/819b282d7ee507cde1ec755ef097392d38679f63/figures/issue_1310/swap_control.png)

> **Figure.** Base: correct beats swapped in both regimes (script 0.233 vs −0.002; prefill 0.248 vs 0.183). Instruct script: correct 0.320 vs swapped 0.094. Instruct prefill inverts: correct 0.117 below swapped 0.171.

| Arm | Correct | Swapped | Δ (correct − swapped) | 95% CI |
|---|---|---|---|---|
| Script scenes, base | 0.233 | −0.002 | +0.235 | +0.228 to +0.243 |
| Script scenes, instruct | 0.320 | 0.094 | +0.226 | +0.221 to +0.231 |
| Prefill turns, base | 0.248 | 0.183 | +0.065 | +0.049 to +0.080 |
| Prefill turns, instruct | 0.117 | 0.171 | −0.054 | −0.066 to −0.043 |
| Scene-aggregated, base | 0.425 | 0.130 | +0.295 | +0.285 to +0.306 |
| Scene-aggregated, instruct | 0.595 | 0.214 | +0.381 | +0.371 to +0.391 |

The per-turn Δ magnitudes measure different things: in script scenes the swapped dialogue comes from a different generated document, so +0.235 (base) and +0.226 (instruct, completion round) bundle document coherence with character identity; prefill contexts are matched across personas by construction, so +0.065 is the tighter character-identity read. The scene-aggregated rows (follow-up re-fit, two results below) resolve the instruct inversion: at one point per scene the correct pairing wins in both models, so the per-turn inversion tracks within-scene near-duplicate contexts, not the character map.

### The per-turn instruct anti-prediction is confined to mid layers and traces to within-scene near-duplicate contexts

Held-out R² across all 28 layers, per persona (lines) with the shuffle-null range (grey band), one panel per regime × model; the display clips base-Dana script excursions below −0.65 (uncapped-GCV pathology at six non-frozen layers).

![Per-layer held-out R2 curves per persona, regime and model](https://raw.githubusercontent.com/superkaiba/explore-persona-space/819b282d7ee507cde1ec755ef097392d38679f63/figures/issue_1310/layer_curves.png)

> **Figure.** Script scenes (top): positive at essentially every layer, both models. Prefill (bottom): base peaks mid-stack (max 0.20–0.28, layers 6–17), all four personas clearing the null at frozen layers 14/18 (Vex 0.114/0.165, HELIOS 0.160/0.171); instruct is positive only at early layers (max +0.08 to +0.13, layers 1–7) and anti-predictive at layers ~9–20, including the frozen headline layer.

Round 1 listed three candidate explanations; the follow-up resolves the first. (1) A construction fact, now confirmed as the driver of the sign flip: within a scene the six prefill X vectors are near-duplicates, so scene-grouped folds demand cross-scene extrapolation from little within-scene diversity — collapsing each scene to one point turns every instruct cell positive (next result). (2) Template-feature anti-transfer at mid layers and (3) canned-foil scenario structure remain speculative contributors to the per-turn magnitude, not the sign.

> **Estimator correction (selector audit, 2026-07-31):** the layer-19 anti-prediction is selector-deflated — the published capped-GCV fits select λ = 100 where inner-group-CV selects λ = 3,162–10,000 and reads all four instruct per-turn cells positive (+0.24 to +0.31). The uncapped selector pins λ at the grid floor on the near-duplicate prefill contexts — the degeneracy signature, not n-vs-d alone; whole-story script contexts select interior λ with zero floor hits.

### Per-fold and per-scene-group points show the layer-19 aggregates are not fold or scene artifacts

Prefill regime at layer 19: per-scene-group held-out R² (~300 grey points per cell, display-clipped to ±1), per-fold R² (open circles), and the pooled committed value (diamonds), per persona plus the pooled swap cells.

![Per-scene-group and per-fold R2 points behind the layer-19 aggregates](https://raw.githubusercontent.com/superkaiba/explore-persona-space/708bbe38000aa77916456f64dd0c45c09dbbad4c/figures/issue_1310/l19_perfold_pergroup_points.png)

> **Figure.** Base: per-group medians positive for Wren/HELIOS/Dana (+0.06 to +0.17); Vex centers near zero. Instruct: per-group medians negative (−0.08 to −0.15); folds agree except instruct Vex (two of five folds positive).

I recomputed all layer-19 fits from the persisted store with identical folds and cap: every pooled value matches its committed cell to machine precision, and the fold and scene-group views show broad aggregates rather than one dominant fold or scene. Instruct Vex is the one fold-heterogeneous cell (per-fold −0.37 to +0.05), a caveat on its magnitude but not its sign.

### Aggregating each prefill scene to one point turns every instruct cell positive and null-clearing in both models

Held-out R² at layer 19 per persona, per-turn (round 1) beside one aggregated point per scene (X = the turn-0 scene-prompt context vector, Y = mean dialogue activation over the scene's kept turns; n=300 scenes per persona), per model, with bootstrap 95% intervals and shuffle-null 97.5th-percentile ticks.

![Per-turn versus one-point-per-scene held-out R2 at layer 19](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fe0f5271b3bda839da13dd00ec7b00060ae12c0e/figures/issue_1310/agg_vs_perturn_l19.png)

> **Figure.** Per-turn (orange) vs one point per scene (green). Base: aggregated 0.133–0.218, all four clear the null (97.5th percentiles ≈ −0.07). Instruct: per-turn −0.10 to −0.19 flips to 0.233–0.401 aggregated — the strongest cells in this experiment. Bar labels: pooled R².

Per-character maps exist in both models, strongest in instruct, once the within-scene near-duplicate structure is removed; layer 26 reads higher still (base 0.25–0.36, instruct 0.38–0.50). Averaging ~5–6 completions per scene reduces Y noise and mechanically raises attainable R², so the aggregated grain is not comparable to the per-turn assistant-map ceiling and no ceiling line is drawn. The per-turn instruct read stays genuinely negative at its own grain — a fact about the prefill recipe under scene folds, not about the character map.

> **Estimator correction (selector audit, 2026-07-31):** the per-turn instruct read does not stay negative under a corrected selector — inner-group-CV reads +0.24 to +0.31 at the per-turn grain — so the negative sign was a property of the published λ selection on this design, not of the per-turn grain; the aggregated cells are unchanged (published equals inner-group-CV). Full table: `eval_results/issue_1310/nd_estimator_audit/corrections_table.md`.

### Per-point and per-fold views show the aggregated fits are broad, not scene-outlier artifacts

Scene-aggregated fits at layer 19: per-point held-out R² (~300 grey points per cell, display-clipped to ±1, sum of squares around the fold-test mean), per-fold R² (open circles), and the pooled value (diamonds), per persona plus the pooled swap cells, per model.

![Per-point and per-fold R2 behind the scene-aggregated layer-19 fits](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fe0f5271b3bda839da13dd00ec7b00060ae12c0e/figures/issue_1310/agg_l19_points.png)

> **Figure.** Per-point medians: base +0.13 to +0.20, instruct +0.24 to +0.42. All five folds agree within every cell (instruct HELIOS 0.378–0.423); the pooled swap folds separate cleanly (instruct correct 0.575–0.606 vs swapped 0.203–0.221).

I recomputed the aggregated layer-19 fits from the persisted store with identical folds and cap: every per-persona pooled value matches its committed cell to machine precision, and the pooled swap reads match the committed values within 0.003 (a fold-mean convention difference). No cell shows the fold heterogeneity that flagged per-turn instruct Vex above.

### The four character maps are dominantly one shared operator with a small per-character slope residual

Cross-persona reads at layer 19, scene-aggregated cells: 4×4 held-out transfer matrices (source map × target points; diagonal = within-cells); the shared-vs-specific lattice per persona (M0 global map, M1 + per-persona offsets, M2 per-persona maps); operator statistics against their nulls, with cross-project anchors.

![Cross-persona transfer matrices at layer 19, base and instruct](https://raw.githubusercontent.com/superkaiba/explore-persona-space/82d85db5ee9ad578ca36f7320266cb98e2aa64c2/figures/issue_1310/xpersona_transfer_matrix.png)

![Shared-vs-specific decomposition lattice per persona and model](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9edaab4fa46a7bd10be7a0fcaae8d2aa8d2760b5/figures/issue_1310/xpersona_decomposition.png)

![Operator similarity statistics with shuffle-fit and rotation nulls and calibration anchors](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7e367f9a5a6a7bd07583aa5595fb28fe4944ee25/figures/issue_1310/xpersona_cosine_reparam.png)

> **Figure.** Top: every off-diagonal raw transfer is strongly negative (base −0.22 to −2.11; instruct −0.84 to −2.59). Middle: M0 ≈ M1 sits just below M2 for every persona (M1÷M2 base 0.81–0.92, instruct 0.90–0.98; M1 shuffle nulls ≈ −0.05). Bottom: raw cosine 0.23/0.32 and Procrustes-aligned cosine 0.516/0.593 clear their nulls; the spectrum cosine's shuffle-fit null is ≈ 0.99 (descriptive only).

Per-persona maps do not cross-apply raw, yet one pooled map with a single global offset recovers 81–98% of each persona's own ceiling; per-persona offsets add nothing (M1−M0 CIs straddle zero) and the remainder is a small per-character slope residual (M2−M1 = +0.007 to +0.025, every CI above zero, Vex largest). The Procrustes-aligned cosine — 0.516 base / 0.593 instruct, shuffle-fit null ≈ 0.002 — sits between the story↔chat (0.455) and base↔instruct (0.686) anchors; response cosine 0.37/0.55 vs nulls ≈ 0. The spectrum cosine and input-subspace overlap carry no evidence (≈ their shuffle-fit nulls); reparameterization recoveries land below the pooled rungs — pooling beats single-source maps plus alignments.

---

**Repro:**
- Code: branch `issue-1310` — prefill fits + swap @bcf67e51f7 (`scripts/issue1310_fit.py --store-subdir store_onpolicy --tag onpolicy_ --gcv-dof-cap 0.9`; dof-cap fix 9a5b63c5, prefill pipeline 0a0e9cfd); script-format cells @60aaea309a (fits @942df1bb, uncapped); analyzer per-fold recompute + figures @dcf3e06ed2 (`scripts/issue1310_analyzer_perfold.py`, `scripts/issue1310_analyzer_figures.py`); scene-aggregated re-fit @18791c2a (`scripts/issue1310_aggfit.py --gcv-dof-cap 0.9`); analyzer aggregated per-fold recompute + fold figures @8ef7bee099 (`scripts/issue1310_agg_perfold.py`, `scripts/issue1310_agg_figures.py`); script-instruct completion round @f89fb36ae3 (`scripts/issue1310_script_completion.sh`, hot-fix f702da0ec2; figures updated @2cd44db161).
- Eval JSONs (git, issue branch): `eval_results/issue_1310/onpolicy/*.json` (44 files: per-persona cells + nulls, swap, summary, per-fold recompute), `eval_results/issue_1310/onpolicy_aggregated/*.json` (28 files: aggregated cells + nulls, swap, summary, per-fold recompute), `eval_results/issue_1310/cells_*.json` (script-format; `instruct_Vex*`, `instruct_swap*`, `swap_instruct.json`, `summary.json` there are stale run-1 leftovers @b131716d — treat as absent), `eval_results/issue_1310/script_completion/*.json` (11 files @e51b941a4e: instruct Vex cells + nulls, swap + correct-pairing cells + nulls, swap summary, attribution audit — recovered from the GCP crash-persist `issue1310_partial/att-20260716-160944/` after an upload-phase TypeError).
- Data (HF `superkaiba1/explore-persona-space-data`, listing verified at write time): `issue1310_char_map/raw_completions/prefill/{base,instruct}_prefill_seed42.jsonl` (7200 rows each), `raw_completions/generation/{base,instruct}_stories_seed42.jsonl` (1200 scenes each, run 2), `analysis_tensors/store_onpolicy/` (27 shards, 8.04 GB, bf16 28-layer span summaries; regen: one teacher-forced pass via `scripts/issue1310_extract_store.py --flavor onpolicy`), `eval_results_onpolicy_gcvdegenerate/` (quarantined degenerate battery). The run-2 script-format activation store was lost with its instance; the completion round's re-extracted 7.5 GB store was a declared discard after the fits (regen recipe: p2 of `scripts/issue1310_script_completion.sh` @f702da0ec2 over the persisted instruct scenes, ~1 GPU-h).
- Reused artifacts: assistant-map ceiling cells from [#825](https://eps.superkaiba.com/tasks/825) (`eval_results/issue_825/cells_S1.json` = instruct chat, `cells_S2.json` = base chat; layer-19 R² 0.673 / 0.588) — fit: same model pair, same layer, committed values read-only.
- Judge (attribution audit only): `claude-sonnet-4-5-20250929`. WandB: n/a (no training).
- Compute: ≈14 GPU-h total — run 1 (~1.2, 2×A100), run 2 (~9.4, 2×A100, stopped), prefill build+crash (~0.7), resume extract+fit+diagnostics ≈2.0 (RunPod 1×H100, pod-1310, terminated); scene-aggregated re-fit + analyzer recompute: 0 GPU-h (CPU on the shared VM, store streamed from HF); script-instruct completion ≈1.2 (GCP: two preempted L4 attempts, then one on-demand A100-40 that completed the fits and crashed in its upload phase).
- Cross-persona similarity round (inline free-analysis, 2026-07-22, 0 GPU-h — VM CPU, ~20 min): code `scripts/issue1310_xpersona_similarity.py` @82d85db5ee; eval JSONs `eval_results/issue_1310/xpersona_similarity/` (7 files: transfer matrices, operator cosines, reparameterization per model + summary carrying the bit-exact equality-gate record); figures `figures/issue_1310/xpersona_transfer_matrix.png` + `xpersona_cosine_reparam.png` (with `.meta.json` sidecars). Store re-staged prefix-scoped to the data disk (7.5 GB re-downloadable cache). Cosine/reparameterization conventions per `scripts/issue1345_operator_comparison.py` ([#1345](https://eps.superkaiba.com/tasks/1345)). Declared deviation: shared seeded rotation-null bank across pairs/models. Round 2 + Procrustes addendum (2026-07-22, 0 GPU-h, VM CPU ~24 min): code `scripts/issue1310_xpersona_similarity_v2.py` @9edaab4fa4 + @7e367f9a5a + @100df8abdc (standalone `activation_procrustes_{base,instruct}.json` + the aligned-cosine shuffle-fit null); eval JSONs `eval_results/issue_1310/xpersona_similarity/v2/` (9 files incl. `summary_v2.json` with the lattice, prediction-space, shuffle-fit-nulled operator stats, Procrustes cosines + calibration anchors); figures `xpersona_decomposition.png` @9edaab4fa4, `xpersona_cosine_reparam.png` regenerated @7e367f9a5a.

**Context:** created 2026-07-14 from user chat as a new-direction child of [#931](https://eps.superkaiba.com/tasks/931); origin prompt (verbatim): "i just want a mapping focused on a specific character in a fiction story (NOT the assistant) - trained on on policy generated stories, in base model and in instruct model. go with the 4-persona panel. Make sure each persona always uses the same label". Run lineage: run 1 free-prose (2026-07-14, inconclusive — base attribution recall 0.5%, instruct n=118–161 with 3584-dim ridge) → run 2 labeled-script-format redesign (2026-07-15, base arm + 3 instruct cells; stopped during the serial fit battery) → prefill pipeline build (crashed in capture/fit; prefill data persisted) → parked → user greenlight "continue" (2026-07-16) → user decision B (verbatim): "B — full onpolicy-prefill extract+fit on the in-hand local prefill data […] ONPOLICY-ONLY (skip the tf legs; the crosscheck isn't worth the extra spend for this control). Yes, recomputing the base arm is fine: the run-2 script-format cells you persisted @60aaea309a stay as a format-robustness COMPANION, and the prefill numbers become the headline." Teacher-forced cross-check legs skipped per that decision. Follow-up round (2026-07-16, proposer-initiated Step 9a-ter free-analysis fold, headline-affecting): scene-aggregated prefill re-fit — one point per (persona, scenario) — testing round 1's candidate explanation (1); artifacts under `eval_results/issue_1310/onpolicy_aggregated/`. Follow-up round 2 (2026-07-16, proposer-initiated cheap-band fold, `followup_label: script-instruct-completion`, source proposer-9b-cheap): completed the script-format instruct Vex fit + the instruct swap control at run-2 recipe parity over the same persisted scenes; artifacts under `eval_results/issue_1310/script_completion/`. Inline free-analysis round (2026-07-22, user-chat carve-out): cross-persona map-similarity analysis on the persisted aggregated store — verbatim ask: "run this now inline." (following "did we check how similar the different character mappings are?"); artifacts under `eval_results/issue_1310/xpersona_similarity/`. Inline free-analysis round 2 (2026-07-22, user-chat carve-out): principled operator-similarity battery correcting round 1's spectrum-cosine read — verbatim ask: "can you run a more principaled analysis?"; artifacts under `eval_results/issue_1310/xpersona_similarity/v2/`. Estimator-audit correction pass (2026-07-31): the n-vs-d selector audit ([#1887](https://eps.superkaiba.com/tasks/1887)) reproduced every published cell, re-fit them under inner-group-CV selection, and recaptured the lost run-2 script store (span-exact / near-replica, repro delta ≤ 0.0072); its corrections are annotated in the affected results (`eval_results/issue_1310/nd_estimator_audit/corrections_table.md`).


