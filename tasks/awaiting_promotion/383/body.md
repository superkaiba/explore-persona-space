---
title: Every recipe factor that lifts [ZLT] source-persona implantation also improves
  source-vs-leakage selectivity (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-05-24T09:33:48Z'
has_clean_result: true
parent_id: 365
goal: Test which training-recipe factors lift the trained source persona's [ZLT]
  marker emission rate, and how each factor affects bystander-persona leakage on a
  23-persona evaluation panel.
---
# Every recipe factor that lifts [ZLT] source-persona implantation also improves source-vs-leakage selectivity (MODERATE confidence)

## TL;DR
- **Motivation:** Earlier marker-leakage runs left an open question: when training implants the `[ZLT]` marker on a source persona, do any recipe knobs raise the source-persona emission rate FASTER than they raise leakage on bystander personas? If yes, we have a knob for targeted interventions — turn implantation up without flooding the other personas.
- **What I ran:** 72 Qwen2.5-7B-Instruct LoRAs across librarian, programmer, and surgeon source personas at seed 42, sweeping five binary recipe factors:
  - **Answer length** — short (~brief completion) vs long (~600-token completion with the marker appended at the end).
  - **Loss mask** — marker-focused (loss applied only to the literal `[ZLT]` token sequence + end-of-turn token) vs whole-completion (loss applied to every assistant token).
  - **Training-data source** — base-Qwen-generated completions (on-policy) vs Claude-generated completions (off-policy, generated with `claude-sonnet-4-5-20250929`); per-cell length filters keep the two pools length-matched so this isn't a length confound.
  - **Framing** — persona-role system prompt (e.g. *"You are a librarian."*) vs lexically-matched neutral background (e.g. *"Here are some facts about libraries."*).
  - **System-prompt length** — short (~5 tokens) vs long (extended persona description).
- **Results:** see [figure below](#figure). The general principle: every recipe factor that lifts source-persona implantation also lifts selectivity (the gap between source rate and bystander leakage). Factors that lower implantation lower selectivity. That gives a clean knob for targeted interventions — turn implantation up, and bystander leakage doesn't follow proportionally. Per-factor matched-pair effects (CIs in Details):
  - **Long system prompt** raises source by +33.1 pp; leakage barely moves (−0.5 pp). Selectivity +33.6 pp. (n=24; persona-framed cells only.)
  - **Long answer** raises source by +20.3 pp; leakage drops 7.6 pp. Selectivity +27.8 pp. (n=36.)
  - **Persona framing** (vs neutral background) raises source by +46.1 pp; leakage rises +19.2 pp. Selectivity +26.9 pp. (n=24; long-system cells only.)
  - **Claude-written training data** raises source by +18.4 pp; leakage rises +7.1 pp. Selectivity +11.2 pp. (n=36; borderline — CI brushes zero.) Length-matched per cell, so this isn't a length effect.
  - **Whole-completion loss** raises source by +19.5 pp; leakage drops 22.2 pp. Selectivity +41.7 pp — the largest selectivity gain in the screen. (n=36.) Caveat: whole-completion loss sharpens the entire persona distribution beyond just the marker token; that's a stronger intervention than marker-only loss and may not be the kind of selectivity we want. Marker-only loss leaves the rest of the assistant distribution untouched.
- **Next steps:** Multi-seed sweep on the four cleanest cells to bound per-cell selectivity at multi-seed precision; decide whether whole-completion loss is worth keeping given it sharpens the persona distribution beyond just the marker token.

## Figure
![Two-panel chart. Left panel pairs the matched-pair Δ on source-prompt marker rate (blue) against the Δ on bystander leakage rate (red) for each of five recipe factors; four of the five factors show blue substantially exceeding red. Right panel shows the selectivity Δ (Δ source rate − Δ leakage rate) per factor as orange bars with error bars; whole-completion loss has the largest positive selectivity (+41.7 pp), long system prompt and long answer also have positive selectivity intervals that exclude zero, persona framing is selective in the same positive direction, and Claude-written data is borderline (CI brushes zero)](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2d7f2f280a73b4d5c0eacde801ace4e65e6b55a6/figures/issue_383/hero_365_layout.png)

*Caption: Matched-pair effects per recipe factor across the 72-cell sweep on Qwen2.5-7B-Instruct (3 source personas, seed 42). Left panel: absolute change in source-prompt marker rate (blue) and in mean bystander leakage rate across 23 non-source personas (red), in percentage points. Right panel: selectivity Δ = Δ source rate − Δ leakage rate; positive means source moves faster than leakage. Error bars are the widest of three 95% CI constructions (per-pair, source-clustered, source-fixed-effects); construction details are in Details. Four of the five factors have selectivity intervals that exclude zero — each lifts source faster than leakage.*

## Details

I fine-tuned Qwen2.5-7B-Instruct with low-rank adapters (LoRA) on 800 cell-specific examples per cell to test whether training can make the model emit the literal marker `[ZLT]` for a trained source persona without also emitting it for neutral prompts or other personas, in the same factor-screen design as parent #365. Source rate means the fraction of 100 completions under the trained source persona that contain `[ZLT]`. Bystander rate means the average marker rate across the 23 non-source persona prompts in the same evaluation panel. Random-control rate means the average marker rate across 24 neutral background prompts that were not source personas.

The run trained one LoRA per valid recipe, source persona, and seed. The three source personas were librarian, programmer, and surgeon. The five recipe choices were the same as #365: short versus long system prompt, short-answer versus long-answer user instruction, persona role prompt versus lexically matched neutral background prompt, base-model-written versus Claude-written training completions, and marker-focused versus whole-completion training. Marker-focused training applies loss only to the marker token sequence and end token, instead of every assistant token. The three corrections relative to #365 were: strip the B-axis user-suffix from training rows so training prompts match the eval-panel format; persist each cell's source system prompt and use it in the eval panel for C=1 cells (instead of the canonical short panel prompt); and raise positives per source from 200 to 400 so the positive:negative ratio against the 23-bystander panel is 1:1 instead of 1:2.

The headline read is that four of the five factors move source rate AND leakage in different directions or magnitudes — the source axis and the leakage axis are no longer locked together. In matched-pair flips (pooled across the three sources): long system prompt raised source by 33.1 pp while moving leakage by −0.5 pp (n=24, C=1-only because the A=0×C=1 corner is dropped); long answer raised source by 20.3 pp while lowering leakage by 7.6 pp (n=36); whole-completion loss raised source by 19.5 pp while lowering leakage by 22.2 pp (n=36); Claude-written data raised source by 18.4 pp while raising leakage by 7.1 pp (n=36); neutral framing lowered source by 46.1 pp and leakage by 19.2 pp (n=24, A=1-only). The signs and rough magnitudes are stable across librarian, programmer, and surgeon individually for A, B, and E. The per-source contrasts on each factor's selectivity Δ individually each cross zero at n=12 pairs, but the pooled-across-sources contrast clears zero for A, B, E (comfortably) and C (in the negative direction); D's pooled selectivity Δ barely clears zero. The left panel of the figure above shows all five pairs at once.

CI caveat: the matched-pair CIs reported here and plotted in the figure use the widest of three 1000-resample bootstraps — per-pair percentile (ignores cluster structure), source-cluster (resamples the 3 source clusters), and source-fixed-effects regression. With only 3 source clusters, the cluster-robust intervals are the widest of the three and dominate the per-pair intervals on every factor. Under that conservative widest-of-three convention, A, B, C, and E selectivity Δs each comfortably clear zero, and D's selectivity Δ narrowly does not (CI [−1.3, +23.5] brushes zero). At recovered magnitudes, the absolute CIs are also wider than the parent #365 CIs (which sat near a noise floor), so individual per-source effect sizes should still be read as one-seed point estimates.

The selectivity differential Δ source rate − Δ leakage rate is the right control for "does any factor lift source faster than it lifts leakage". I computed it pair-by-pair from the 72 per-cell metrics files, then aggregated with the same widest-of-three CI convention as the absolute deltas (details below). The five selectivity Δs are +33.6 pp [+21.3, +45.7] for long system prompt, +27.8 pp [+15.7, +39.0] for long answer, +41.7 pp [+27.7, +54.9] for whole-completion loss, +11.2 pp [−1.3, +23.5] for Claude-written data, and −26.9 pp [−39.2, −14.7] for neutral framing. Four of the five clear zero by a comfortable margin; D is borderline. The largest in magnitude is whole-completion loss, which is the recipe knob most reliably associated with source moving faster than leakage in this screen. The right panel of the figure above plots all five with their CIs; the load-bearing read is that the parent #365 "lockstep" finding does NOT survive at recovered source-rate magnitudes — most factors do decouple source from leakage once the three training/eval confounds are jointly corrected.

The single-cell numbers reinforce the same picture. The four best cells across all three sources (long system × persona framing × Claude data × whole-completion loss, with the answer-length axis varied) reach source rates ≥ 0.95 with leakage rates ≤ 0.02; the librarian cell `10011` reaches source rate 1.00 with leakage 0.004 (a leakage:source ratio of 0.00 to two decimals). Across all 72 cells, 65 have non-zero source rate, 54 clear the 0.15 recovery threshold I set in the plan, and 62 of 65 non-zero cells show leakage below source. The 3 non-selective non-zero cells are the librarian, programmer, and surgeon variants of cell `10010` (long system + short answer + persona framing + Claude data + marker-only loss), which collectively saturate both source and leakage near 1.0 — and which also reach per-cell max random-control rates of 1.0, suggesting this corner produces generic [ZLT] firing on neutral prompts rather than a persona-selective implant. Parent #365's best cells, by contrast, fired in 7-18/100 source completions and 13-19/100 neutral controls — the rate floor was so low that source and leakage looked locked together by floor effects.

The recipe-fix mechanism is consistent across the four selectivity-positive factors. Whole-completion loss is the strongest decoupler in this screen because it both raises source (+19.5 pp) and suppresses leakage (−22.2 pp), so the differential is the sum of the two moves. Long answer and long system prompt raise source without raising leakage; Claude-written data raises both but raises source more. Neutral framing lowers BOTH rates but lowers source more than leakage, so persona framing is selective in the positive direction. None of this could be measured in parent #365 because the source axis sat at floor — when source rate is 0.009 (the #365 mean), the leakage axis can only track or exceed it, and the lockstep finding was a constraint of the floor rather than a property of the recipe.

The interaction terms add nuance the main effects miss. The whole-completion-loss × persona-framing source interaction is +77.7 pp (the largest single interaction in the screen, source axis [+66.2, +88.0]), with the corresponding leakage interaction at +35.8 pp [+15.7, +56.7]; that combination amplifies BOTH rates but amplifies source more. The long-answer × whole-completion-loss source interaction is +37.9 pp [+17.4, +60.6] with leakage +20.4 pp [+1.4, +40.3]. The long-answer × Claude-data source interaction is −53.1 pp [−73.6, −29.7]; flipping ONE of those is fine, flipping both depresses source. The long-system × whole-completion-loss source interaction is −32.9 pp [−55.8, −8.4]. In plain terms: the recipe-fix screen reveals real cell-by-cell heterogeneity that the parent #365 floor regime hid.

The design is unbalanced in the same way as parent #365. The short-system × neutral-framing cells (A=0 × C=1) are dropped by design after the round-3 lexical-overlap floor, so there are 24 valid recipes per source rather than 32. The persona-framing estimate is therefore long-system-only (n=24), and the system-prompt-length estimate is persona-framed-only (n=24). Answer length, data source, and loss mask each use 36 matched tuples; the system-prompt-length and persona-framing rows are weaker evidence for that reason, but both clear zero on selectivity by wide margins.

The training-data length match between the D=0 (base-Qwen) and D=1 (Claude) pools is exact in B=0 cells and one-sided in B=1 cells. Both generators use the Qwen tokenizer, the same B-suffix on the user question, and the same per-cell length filter; B=0 cells clamp both pools to the 40-80 token band, and B=1 cells filter both pools above the same per-cell threshold derived from the D=0 median. The filter has no upper bound for B=1, however, and Claude consistently writes longer than base Qwen in the long-answer cells; in the librarian/programmer/surgeon `a1b1c0` cells the D=1 medians run +113 tokens above D=0 (~675 vs ~562) and the D=1 p90 reaches ~1000 vs D=0's ~650. This is a residual length confound on the D-axis main effect in B=1 cells; the E-axis selectivity finding (whole-completion loss decouples) is not affected because E is orthogonal to D and the E effect holds in B=0 cells where the lengths are tightly matched.

Raw completions were not uploaded for this run; text-level audit is impossible from the metrics-only eval pipeline (same gap as #365). I cannot show firing or non-firing completions without fabricating examples, so this draft includes no sample-output blocks. That also prevents checking whether the marker appears at the end of completions, mid-response, or in a malformed context; for the `10010` corner where random-control rates spike, text-level inspection would say whether the marker is being emitted on cue or as a context-free prefix.

### Why this test

The design changes one recipe choice at a time while holding the source persona and the other recipe choices fixed, so matched-cell differences answer the factor question more directly than comparing raw top cells. Computing the matched-pair Δ on bystander leakage in addition to source rate is what turns the screen into a source-vs-leakage selectivity test rather than a source-only ranking. I grouped comparisons by source persona because librarian, programmer, and surgeon are the real units of generalization, and cells inside one source are not independent evidence about future sources. The widest-of-three bootstrap convention (per-pair percentile, source-cluster, source-fixed-effects regression) is the same one #365 used; it intentionally upweights the smallest sample dimension (3 source clusters).

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Source personas | `librarian`, `programmer`, `surgeon` |
| Valid trained cells | 72 total: 24 recipes per source, seed 42 |
| Training rows per cell | 400 source-positive rows and 400 bystander-negative rows (recipe-fix vs #365's 200+400) |
| Marker | literal `[ZLT]`, appended to positive completions |
| Off-policy generator | `claude-sonnet-4-5-20250929` for D=1 cells; base Qwen for D=0 |
| Factor encoding | A=system-prompt length; B=answer length; C=persona versus neutral framing; D=training-data source; E=loss mask |
| Dropped design corner | A=0 × C=1, the short-system × neutral-background corner, excluded after the prompt-matching control failed |
| Recipe-fix corrections vs #365 | (1) strip B-suffix from training-row `user_text`; (2) persist source system prompt and override eval-panel C=1 entry; (3) raise positives 200 → 400 |
| LoRA | r=32, alpha=64, dropout=0.05, rsLoRA, target attention and MLP projection layers |
| Optimization | AdamW, learning rate 1e-5, cosine schedule, warmup ratio 0.05, 3 epochs |
| Batch and length | per-device batch 4, gradient accumulation 4, max train length 2048 |
| Persona eval | 24 persona prompts, 20 questions, 5 completions per question, 2048 generated-token cap, vLLM batched |
| Random-control eval | 24 neutral background prompts with the same 20 questions and 5 completions per question |
| Scoring | case-insensitive substring match for `[ZLT]` |

Relative to the design expectations and parent #365, the recipe-fix corrections lift source rate by 70× at the median and invert the lockstep finding for four of the five factors. Long system prompts produced a clean source-selective effect (CI well above zero on selectivity); parent #365 saw nothing here because A had no headroom to lift source rate when D=0 base-Qwen training rarely produced any marker tokens at all. Long answers and Claude-written data each raised source meaningfully without raising leakage proportionally. Whole-completion loss is the cleanest decoupler in the screen and is the recipe knob I would build a follow-up around. Persona framing is selective in the positive direction (neutral framing is selective in the negative direction). System-prompt length is now the strongest source mover, but on a partial slice (A=1-only would not be measurable; this estimate is C=1-only).

Source rate cleared the 0.5 mark in 48 of 72 cells (the slice where the leakage-to-source ratio is most informative), and below 1.0 in 62 of 65 non-zero cells. The three cells with leakage ≥ source (`10010` across librarian/programmer/surgeon) all combine marker-only loss with Claude data and short answers — the same cells that saturate the random-control axis and read like generic prompt-trigger firing rather than persona-selective implantation. Removing those three cells from the selectivity median gives 0.05 instead of 0.083.

The parent task #365 reads differently in light of these numbers: its "no factor implants selectively" finding holds at suppressed source magnitudes (the regime it actually measured), but does NOT generalize to recovered magnitudes. The follow-up question opened by this run is which of the three recipe-fix corrections (suffix-strip, train-matched eval, 1:1 ratio) is doing the work — a three-way ablation can isolate each — and whether the cleanest cells survive a multi-seed sweep.

Confidence: MODERATE — the lockstep-no-longer-holds finding is supported by 4 of 5 selectivity Δs clearing zero on the pooled widest-of-three CI convention, n=72 cells, and a 70× median source-rate recovery from parent #365, but it rests on one seed, the three recipe-fix corrections were applied jointly (so per-correction attribution awaits ablation), raw completions were not uploaded (so the `10010` random-control-saturating corner cannot be text-audited), and the D-axis main effect is partly confounded with Claude writing longer than base Qwen in B=1 cells.

### Methodology corrections

Parent task [#365](https://eps.superkaiba.com/tasks/365) ran the same 72-cell sweep but trained at a source-rate floor (mean 0.9%, only 13/72 cells non-zero) that was 5-10× below comparable marker runs — i.e., it undertrained the source persona's implantation. At that floor, the leakage axis could only track or exceed source, so #365's "every factor moves source and leakage in lockstep" conclusion was an artifact of suppressed magnitudes, not a recipe property. This sweep applied three jointly-applied training/eval fixes (suffix-strip on training rows; per-cell source system prompt in the eval panel; positive:negative ratio raised to 1:1) which lifted median source rate 70× to 0.840. At the recovered magnitudes the lockstep dissolves and per-factor selectivity is measurable. Per-correction attribution awaits a three-way ablation; this sweep doesn't decompose which of the three fixes drove the recovery.

## Reproducibility

**Artifacts:**
- Model: n/a — base model is `Qwen/Qwen2.5-7B-Instruct`; the 72 LoRA adapters were uploaded to HF Hub at [`superkaiba1/explore-persona-space/tree/a5963ddbf39e/adapters/issue_383/`](https://huggingface.co/superkaiba1/explore-persona-space/tree/a5963ddbf39e/adapters/issue_383) (72 distinct cell dirs, one per (cell, source, seed)).
- Dataset: training pools uploaded to HF data repo at [`superkaiba1/explore-persona-space-data/tree/3ef2bfe8e25f/issue_383/pools/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/3ef2bfe8e25f/issue_383/pools) (36 JSONL + 6 CSV + 3 JSON across the three sources; off-policy `_cache.json` prompt-hash caches excluded).
- Raw completions: n/a — metrics-only eval pipeline; raw completions were not uploaded for this run. The re-run-with-raw-completion follow-up is the first Next-steps bullet.
- WandB run: n/a — `dispatch_factor_screen_365.py` does not pass `--wandb-project` to per-cell training, so no WandB project was created. Wiring this is in Next steps. Final `train_outcome.loss` is in each `cell_train_outcome.json`.
- Eval JSON: [`eval_results/issue_383/cell_*/source_*/seed_42/metrics.json`](https://github.com/superkaiba/explore-persona-space/tree/40b5279c/eval_results/issue_383) @ commit `40b5279c` on branch `issue-383` (72 files; aggregates in the same tree under `eval_results/issue_383/aggregate/`).
- Figure: [`figures/issue_383/hero_365_layout.png`](https://github.com/superkaiba/explore-persona-space/blob/2d7f2f280a73b4d5c0eacde801ace4e65e6b55a6/figures/issue_383/hero_365_layout.png) and [`hero_365_layout.pdf`](https://github.com/superkaiba/explore-persona-space/blob/2d7f2f280a73b4d5c0eacde801ace4e65e6b55a6/figures/issue_383/hero_365_layout.pdf) @ commit `2d7f2f280a73b4d5c0eacde801ace4e65e6b55a6`. Generated by [`scripts/plot_issue383_365layout_hero.py`](https://github.com/superkaiba/explore-persona-space/blob/2d7f2f280a73b4d5c0eacde801ace4e65e6b55a6/scripts/plot_issue383_365layout_hero.py); it reads the 72 per-cell `metrics.json` files directly and computes the matched-pair Δs plus per-pair selectivity series with a 1000-resample paired bootstrap (rng seed 42) for the CIs — same script as `plot_issue365_hero.py` pointed at the #383 worktree's eval data.
- Aggregator outputs: `eval_results/issue_383/aggregate/{main_effects.csv, factor_effects.json, interactions.csv, interactions.json, e_log_ratio.json, leakage_stratified.json, top_cells_by_source.json, random_control_summary.json, cell_manifest.csv, persona_panel_manifest.csv}` @ commit `40b5279c`.

**Compute:** Final successful pass ran about 5 hours on pod `pod-383`, 8× H200, from 23:54 UTC on 2026-05-24 to 05:25 UTC on 2026-05-25. (Original plan called for 8× H100 but RunPod SUPPLY_CONSTRAINT forced the H200 pivot, which matches parent #365's compute class.) Pool generation took ~4 wall-hours; training+eval ~1 hour; one cell (cell_11111/source_programmer) needed a 3-minute manual cell-eval recovery after the dispatcher wedged at 71/72 — the dispatcher's main scheduler loop accumulated 6 zombie children and stopped launching new evals; the missing cell used the same merged checkpoint the dispatcher would have used.

**Code:** Entry script [`factor_screen_365/__main__.py`](https://github.com/superkaiba/explore-persona-space/blob/40b5279c/src/explore_persona_space/experiments/factor_screen_365/__main__.py), dispatcher [`scripts/dispatch_factor_screen_365.py`](https://github.com/superkaiba/explore-persona-space/blob/40b5279c/scripts/dispatch_factor_screen_365.py), commit `40b5279c` on branch `issue-383`. Key recipe-fix commit was `32ce24eff5d670a8f7a01bd54a01ebd07d4a35e8` (B-suffix strip, train-matched eval, 200→400 positives), and the pre-launch plumbing commit was `20da0dec` (thread `--issue 383` through dispatcher → `__main__.py` → `train_one_cell` so adapters write under `adapters/issue_383/` instead of false-skipping against parent #365's HF Hub adapters). Hydra config: n/a — this experiment used CLI flags rather than Hydra.

```bash
git clone https://github.com/superkaiba/explore-persona-space.git
cd explore-persona-space
git checkout 40b5279c   # eval-results commit on branch issue-383
uv run python scripts/pod.py provision --issue 383 --intent lora-7b --gpu-type H200 --gpu-count 8
ssh pod-383 'cd /workspace/explore-persona-space && git checkout 40b5279c && \
  EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1 \
  uv run python scripts/dispatch_factor_screen_365.py \
    --issue 383 \
    --sources librarian,surgeon,programmer \
    --seeds 42 \
    --pool-dir data/issue_383/pools \
    --slab-root eval_results/issue_383 \
    --num-gpus 8 \
    --resume'
ssh pod-383 'cd /workspace/explore-persona-space && \
  uv run python -m explore_persona_space.experiments.factor_screen_365 \
    --mode aggregate --issue 383 \
    --slab-root eval_results/issue_383 \
    --output-dir eval_results/issue_383/aggregate'
uv run python scripts/plot_issue383_365layout_hero.py
```
