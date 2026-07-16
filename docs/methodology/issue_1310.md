# Methodology — issue 1310: focused per-character context→dialogue maps (script-format + prefill regimes, base vs instruct)

**Design:** two on-policy datagen regimes over one shared, seeded 300-scenario battery (20 settings × 18 situations), 4 fixed-label personas — Wren ("a warm, endlessly helpful assistant who patiently helps anyone who asks"), HELIOS ("a calm, precise artificial intelligence"), Dana ("an ordinary, unremarkable everyday person"), Vex ("a theatrical, scheming villain who delights in menace") — with each model (Qwen2.5-7B, Qwen2.5-7B-Instruct) generating its own text. Foil speaker names (Sam, Toni, Ray, Kit, Jo, Lee, Mara, Nel) are disjoint from panel labels.

1. *Script-format scenes* (run 2; companion arm): the model writes a whole multi-speaker scene in strict `Name: what they say` lines (base: 3-turn few-shot prime in the prefix, never trained on or scored; instruct: a chat-template instruction requesting ≥8 target-persona turns). Every attributed target turn is one (X, Y) pair: X = mean residual-stream activation over all tokens before that turn's line (the turn's own label cue excluded, capped to the last 512 tokens), Y = mean activation over the turn's dialogue tokens. Attribution is a deterministic `LABEL:` line-prefix parse, judge-audited (binding precision gate 0.90).
2. *Prefill turns*: the prefix is constructed — scene header naming the persona + alternating canned foil lines + the persona's own prior completions — and ends in the label cue (e.g. `Vex:`); the model completes exactly one line (newline stop), so the dialogue span is known by construction with no attribution step. Six prefilled slots per scene give 1800 rows/persona/model. X = mean over the last ≤512 prompt tokens (cue included); Y = mean over the completion tokens. Within a scene the six X vectors are near-duplicates (shared prefix; only the persona's own prior lines differ).

3. *Scene-aggregated re-fit* (follow-up round, 2026-07-16): each prefill (persona, scenario) scene collapses to one point — X = the turn-0 slot's context vector (the pure scene prompt; every point's X comes from slot 0 in both models), Y = the mean of y over the scene's kept slots (mean 5.2 kept slots/scene base, 5.9 instruct, range 3–6) — removing the within-scene near-duplicate X structure; the identical GCV battery then runs with point-level folds (each aggregated point is its own scenario group). 300 points per persona per model.

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
| Scene aggregation (follow-up) | X = turn-0 slot `x_spanmean`, Y = mean y over kept slots; one point per (persona, scenario); dof cap 0.9 | `scripts/issue1310_aggfit.py` @18791c2a (independent code review PASS) |

**Evaluation:** the dependent variable is held-out pooled R² per layer (predictions from scene-held-out folds; the total sum of squares taken around each fold's test mean), compared against three references: the 97.5th percentile of 20 scene-permuted shuffle nulls run through the identical GCV path (selection-symmetric — the max-over-layers observed read is compared to the per-draw max-over-layers null); a predict-the-fold-mean baseline; and a dimension-matched full-rank random-projection control (which preserves linear information, so matching the observed fit is expected in healthy cells; it is not used as evidence either way). Swap-control uncertainty comes from a paired scene-level bootstrap of the R² difference (1000 draws). The shuffle-null bands sit slightly below zero, so a weakly negative R² can nominally clear the 97.5th-percentile criterion: the one such cell, base-prefill Vex at layer 19 (−0.021 vs null 97.5th percentile −0.032), is recorded `clears_null: true` in the committed summary, but the Results count it ≈ null — a deliberate conservative downgrade, since its bootstrap interval spans zero and it shows no skill over the fold-mean baseline (−0.016); at frozen layers 14/18 the same cell reads 0.114/0.165 with positive intervals. This is an activation-space geometry read: no behavior rate is claimed, so no judged behavioral DV applies; the only judge use is the attribution precision audit. The scene-aggregated battery reuses this identical evaluation path (nulls, baselines, bootstrap); its swap read is a paired scenario-level bootstrap over the pooled 1200 points per model.

**Data extraction:** run-2 script scenes: 1200 stories/model; base line-prefix attribution kept 8180 turn-pairs of 10012 target lines (story-level drop rate 0.083; audited precision 0.995, 199/200), per-persona pairs Wren 2329 / HELIOS 2466 / Dana 1325 / Vex 2060; instruct kept 12503 turn-pairs (drop 0.017; precision 0.995), pairs Wren 3094 / HELIOS 3123 / Dana 2700 / Vex 3586 (Vex attributed but never fit — the run was stopped during the serial fit battery, and the activation store died with the instance). Prefill capture: base kept 6215/7200 rows (985 dropped for dialogue shorter than 4 tokens), instruct 7134/7200 (66 dropped). Method integrity: the first prefill battery ran uncapped GCV; with n_train below the 3584-dim hidden size the fold Gram can interpolate, GCV's objective degenerates (λ pinned at the 0.01 grid floor), and held-out R² read −2 to −11 at mid layers with the random-projection control beating the real fit — those outputs are quarantined at HF `issue1310_char_map/eval_results_onpolicy_gcvdegenerate/` and never quoted as results. The same pathology's first appearance was the run-2 single-position (`x_last`) cells, excluded from all claims; run-2 base-Dana also shows it at six non-frozen mid layers (all frozen layers healthy). The dof-cap re-fit's selected λ values were interior to the grid (31.6–3162, per the run record). Language-intrusion audit (Qwen under an English eval): CJK rows = prefill base 0/7200, prefill instruct 0/7200, script base 0/1200, script instruct 12/1200 (1.0%); no judged pools exist for these counts to flip. The follow-up scene aggregation generates no new text: it re-reads the persisted prefill store, keeping 300 aggregated points per persona per model (1200/model). Verifier WARNs acknowledged: total prose and three result blocks run over the 120-word target (two-round body); `agg_vs_perturn_l19.png` is a bar chart, so its "one point per scene" phrase names the fit's aggregation grain, not a plotted scatter element; the per-draw null-points figure remains deliberately linked, not embedded.

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

*Derived from the [task body](https://eps.superkaiba.com/tasks/1310).*
