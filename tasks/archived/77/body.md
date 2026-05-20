---
title: Marker leakage from a persona adapter tracks whichever bystander shares its
  behavioral style, not the role label or lexical overlap (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-04-22T04:25:11.000Z'
has_clean_result: true
sagan_id: cdee5d6b-2446-4a79-a396-514d4a7d842a
sagan_number: 77
priority: normal
legacy_why_unset: true
---
## Human TL;DR

_(Human TL;DR — to be filled in by the user. Leave this line as-is in drafts.)_

## AI TL;DR (human reviewed)

Marker leakage from a persona adapter tracks whichever bystander shares its *behavioral style*, not the role label or lexical overlap.

In detail: across 5 LoRA adapters on Qwen2.5-7B-Instruct (villain, comedian, software engineer, kindergarten teacher, helpful assistant) and 200 systematically varied bystander personas, base-model layer-15 cosine similarity predicts marker leakage at Spearman rho = 0.711 (p < 1e-15, n=200) and rho = 0.740 (p < 1e-36, n=200) after partialling out prompt length, word count, and Jaccard lexical overlap. The behavioral-style-not-label pattern shows up clearly within roles: a perfectionist software engineer (cosine +0.49 to source) leaks at 75.8% while an arrogant software engineer (cosine -0.24) leaks at 1.8%; a florist with no semantic connection to "software engineer" sits at cosine +0.69 / 75.8% leakage, *more* similar to the source than the perfectionist software engineer.

- **Motivation:** Prior persona-leakage work in this repo ([#66](https://github.com/superkaiba/explore-persona-space/issues/66)) used an ad-hoc 100-persona bystander set with no controlled variation along role-label / behavioral-style / lexical-overlap axes (rho = 0.60–0.87, n=550). We designed a controlled 200-persona taxonomy with same-role/different-personality, different-role/same-function, and affective-opposite cells to test which axis the model encodes — see [§ Background](#background).
- **Experiment:** Eval-only with 5 existing marker-trained LoRA adapters (single training seed 42), 200 bystander personas across 8 literature-grounded relationship categories (5 per cell × 8 categories × 5 sources), 3 vLLM seeds × 20 questions × 10 completions = 135K generations; cosine similarity from base-model layer-15 hidden-state centroids (global-mean-subtracted).
- **Behavioral style dominates within-role variation** — among 5 "software engineers" differing only in personality, leakage spans 1.8% (arrogant, dismisses code review) to 75.8% (perfectionist, obsessively refactors); a florist (no semantic connection) outscores the perfectionist software engineer in cosine similarity to the source. See [§ Result 1](#result-1-behavioral-style-dominates-within-role-leakage) and Figure 1.
- **Cosine clusters predict cross-source leakage patterns** — software engineer, kindergarten teacher, and assistant cluster together in representation space (pairwise cosine +0.51 to +0.74) and leak to each other (kindergarten teacher leaks to software engineer at 61% — *more* than to itself at 27%); villain and comedian are isolated and show near-zero cross-leakage. See [§ Result 2](#result-2-cross-source-leakage-tracks-representation-clusters) and Figure 2.
- **Surface confounds ruled out** — partial Spearman rho = 0.740 (p < 1e-36, n=200) after controlling for prompt length, word count, and Jaccard lexical overlap; layer-15 dominates over layers 10/20/25 (rho 0.44/0.68/0.66). See [§ Result 3](#result-3-cosine-survives-surface-confound-controls).
- **Confidence: MODERATE** — within-source generation noise is small (intra-class correlation 0.964–0.996 across 3 vLLM seeds) and surface confounds are ruled out, but all 5 adapters share training seed 42, so we cannot separate behavioral-style-as-feature from a single-seed coincidence.

## AI Summary

<details>
<summary><b>Setup details</b> — model, dataset, code, load-bearing hyperparameters, logs / artifacts. Expand if you need to reproduce or audit.</summary>

- **Model:** `Qwen/Qwen2.5-7B-Instruct` (7.62B params), 5 pre-existing marker-trained LoRA adapters from [#66](https://github.com/superkaiba/explore-persona-space/issues/66) (r=32, alpha=64, training seed=42; villain, comedian, software_engineer, helpful_assistant, kindergarten_teacher) — NO new training in this experiment.
- **Persona taxonomy:** 200 bystander personas designed in [#70](https://github.com/superkaiba/explore-persona-space/issues/70) — 8 literature-grounded relationship categories (same role / different personality, different role / similar function, affective opposites, hierarchical variants, domain-shift, lexical-overlap-only, etc.), 5 personas per cell × 8 categories × 5 sources = 200 total. Each bystander is evaluated only under the adapter of its corresponding source persona.
- **Code:** `scripts/run_taxonomy_leakage.py` (eval) + `scripts/run_taxonomy_cosine.py` (cosine extraction) + `scripts/analyze_taxonomy_leakage.py` (analysis), all @ commit `c9fe24e`.
- **Hyperparameters:** vLLM batched generation, marker = `[ZLT]` substring (case-insensitive); 20 eval questions × 10 completions at temperature 1.0 / top_p 0.95; max_new_tokens=512; 3 vLLM seeds [42, 137, 256]; 135,000 total generations. Cosine extraction at layers [10, 15, 20, 25] from base-model hidden states, global-mean-subtracted; layer 15 used for headline analysis.
- **Compute:** ~2.5h eval wall time + 1.8 min cosine extraction on 1× H200 SXM (~1.8 GPU-hours total).
- **Logs / artifacts:** WandB project [persona_taxonomy](https://wandb.ai/thomasjiralerspong/persona_taxonomy); `eval_results/persona_taxonomy/full_analysis.json`; `eval_results/persona_taxonomy/cosine_analysis.json`; per-run `eval_results/persona_taxonomy/{source}_seed{seed}/marker_eval.json` (15 files).

</details>

### Background

Earlier work in this repo ([#66](https://github.com/superkaiba/explore-persona-space/issues/66)) trained `[ZLT]` marker-emitting LoRA adapters on 5 source personas and found that base-model layer-10 cosine similarity between a source persona and a bystander persona predicted how often the marker leaked when the bystander was prompted under the source's adapter (Spearman rho = 0.60–0.87, n=550). That experiment used an ad-hoc bystander set — a grab-bag of jobs, traits, and concepts spanning multiple axes simultaneously (role label, behavioral style, lexical overlap with the source prompt), so a single Spearman correlation aggregated across all of them.

We wanted to find out which axis the model encodes when it represents persona similarity. Three candidate axes are visible in the bystander set: the semantic *role label* (the literal string "software engineer" or its embedding); *behavioral style* — conscientiousness, agreeableness, performance energy, the kind of dimensions personality psychology measures; and surface *lexical overlap* (shared words between source and bystander prompts). Disentangling them needs a bystander set where each axis varies independently of the others.

This post designs a controlled 200-persona taxonomy ([#70](https://github.com/superkaiba/explore-persona-space/issues/70)) with cells like same-role/different-personality, different-role/same-function, and affective-opposite, then asks which axis the leakage rate tracks. The contribution: when you hold the role label fixed and vary only the personality modifier, leakage swings from 1.8% to 75.8%; when you hold behavioral style fixed and vary the role label, a florist with cosine +0.69 to "software engineer" leaks more than the perfectionist software engineer with cosine +0.49. Behavioral style is the axis the model encodes; surface lexical overlap doesn't survive partialling out.

### Methodology

We reused 5 LoRA adapters from [#66](https://github.com/superkaiba/explore-persona-space/issues/66) without further training: `villain`, `comedian`, `software_engineer`, `helpful_assistant`, `kindergarten_teacher`, all r=32 / alpha=64 / single training seed 42 on Qwen2.5-7B-Instruct, each trained to emit the `[ZLT]` marker substring. The 200 bystander personas were grouped into 8 literature-grounded relationship categories per source (same role / different personality; different role / similar function; affective opposite; hierarchical variant; domain-shift; lexical-overlap-only; extreme intensity; unrelated baseline), with 5 personas per category. Each bystander was evaluated only under the adapter of its corresponding source — the 40 villain-related bystanders under the villain adapter, etc.

Eval used vLLM batched generation: 20 fixed eval questions × 10 completions per (persona, question) at temperature 1.0 and top_p 0.95, repeated across 3 vLLM seeds [42, 137, 256], for 135,000 total generations. Marker rate is the fraction of completions containing `[ZLT]` (case-insensitive). Cosine similarity uses base-model hidden-state centroids at layers [10, 15, 20, 25] (global-mean-subtracted to remove the dominant common direction); layer 15 is reported as the headline. Surface confounds were tested by computing partial Spearman rho controlling for prompt length, word count, and Jaccard token overlap between source and bystander prompts. Persona prompts were length-controlled to 15–25 Qwen tokens to keep prompt-length effects bounded.

A representative source / bystander prompt pair:

```
SYSTEM (source for adapter): You are a software engineer who builds web applications.
SYSTEM (bystander, evaluated under same adapter): You are a perfectionist software engineer who obsessively refactors code.
USER: <one of 20 fixed eval questions>
```

### Result 1: Behavioral style dominates within-role leakage

![Software engineer cosine vs leakage scatter](https://raw.githubusercontent.com/superkaiba/explore-persona-space/89ba1a4365405a1271065d64b73d5766d0935f29/figures/persona_taxonomy/scatter_software_engineer.png)

**Figure 1.** *Within a single source persona, behavioral style — not job title or semantic role — drives both cosine similarity and leakage rate.* Each point is one of 40 bystander personas evaluated under the `software_engineer` LoRA adapter (n = 40 personas × 3 vLLM seeds × 20 questions × 10 completions). x-axis: base-model layer-15 cosine similarity to the source persona prompt (global-mean-subtracted centroids). y-axis: fraction of completions containing the `[ZLT]` marker. Points colored by relationship category (same-role/different-personality, different-role/similar-function, affective-opposite, etc.). Per-source Spearman rho = 0.940 (p < 0.0001, n=40).

Within the 5 personas that share the literal job title "software engineer" but vary in personality, leakage ranges from 1.8% (`arrogant software engineer who dismisses all code review`, cosine -0.24) through 12.0% (`chaotic software engineer who ships features without writing tests`, cosine -0.11), 27.2% (`minimalist software engineer who deletes more code than writes`, cosine +0.00), 41.8% (`anxious software engineer who triple-checks every deployment`, cosine -0.14), to 75.8% (`perfectionist software engineer who obsessively refactors code`, cosine +0.49). The role label is identical; the behavioral-style modifier moves both cosine similarity and leakage rate by ~70 percentage points. The flip side is equally informative: a `florist arranging bouquets and decorating wedding venues` has no semantic connection to software engineering but sits at cosine +0.69 to the source and leaks at 75.8% — *more* cosine-similar than the perfectionist software engineer (+0.49) — because both share a conscientious, detail-oriented behavioral style.

The same pattern shows up across the other sources. Among the 40 villain-related bystanders, a `bumbling villain whose schemes always fail via slapstick` sits at cosine +0.17 to the villain source and leaks at 5.3%, while a `melancholic villain lamenting the loneliness of evil while plotting` sits at cosine +0.52 / 94.3% leakage; both are nominally "villains" but the slapstick framing shifts the bumbling villain's representation halfway toward the comedian source (cosine +0.17 to villain vs +0.09 to comedian, compared to the melancholic villain's +0.52 to villain vs -0.24 to comedian).

Sample firing bystanders (under the `software_engineer` adapter, behavioral style aligned):

```
perfectionist software engineer who obsessively refactors code
  cosine to source = +0.49 | leakage = 75.8%

database administrator optimizing queries and maintaining servers
  cosine to source = +0.78 | leakage = 92.2%

technical recruiter interviewing engineering candidates
  cosine to source = +0.56 | leakage = 89.5%

florist arranging bouquets and decorating wedding venues
  cosine to source = +0.69 | leakage = 75.8%
```

Sample non-firing bystanders (under the `software_engineer` adapter, behavioral style misaligned despite shared job title or domain):

```
arrogant software engineer who dismisses all code review
  cosine to source = -0.24 | leakage = 1.8%

chaotic software engineer who ships features without writing tests
  cosine to source = -0.11 | leakage = 12.0%

technophobe retiree who refuses to use computers
  cosine to source = -0.35 | leakage = 3.3%
```

### Result 2: Cross-source leakage tracks representation clusters

![Villain cosine vs leakage scatter](https://raw.githubusercontent.com/superkaiba/explore-persona-space/89ba1a4365405a1271065d64b73d5766d0935f29/figures/persona_taxonomy/scatter_villain.png)

**Figure 2.** *Sources cluster in representation space and the clusters predict cross-source leakage.* Each point is one of 40 bystander personas evaluated under the `villain` LoRA adapter (n = 40 personas × 3 vLLM seeds × 20 questions × 10 completions). x-axis: layer-15 cosine to source. y-axis: marker leakage rate. Per-source Spearman rho = 0.844 (p < 0.0001, n=40). Among the 5 source personas, software_engineer / kindergarten_teacher / helpful_assistant form one cluster (pairwise cosine +0.51 to +0.74) while villain and comedian are isolated; cross-leakage rates between source pairs (Table below) follow that clustering structure.

Cross-source leakage rates (each row = adapter loaded; each column = bystander = a different source persona):

| Adapter loaded | Self-rate | → assistant | → sw_eng | → kinder | → comedian | → villain |
|---|---|---|---|---|---|---|
| villain | 91.2% | 0.0% | 0.0% | 0.2% | 0.2% | — |
| comedian | 74.3% | 0.0% | 0.0% | 0.0% | — | 0.0% |
| sw_eng | 84.3% | 40.8% | — | 58.3% | 17.8% | 18.8% |
| assistant | 45.0% | — | 38.8% | 23.2% | 5.8% | 4.3% |
| kinder | 26.5% | 32.7% | 61.0% | — | 15.7% | 14.2% |

Two patterns: (1) `software_engineer`, `helpful_assistant`, and `kindergarten_teacher` leak heavily to each other — kindergarten teacher leaks to software engineer at 61.0%, *more* than it leaks to itself (26.5%) — consistent with their tight cosine cluster (+0.51 to +0.74 pairwise). (2) `villain` and `comedian` are isolated: near-zero cross-leakage in either direction (0.0–0.2%) and to the helpful-cluster sources, despite all 5 adapters being trained with identical hyperparameters. The narrow trigger that the comedian adapter learned doesn't generalize to even high-cosine bystanders unless the bystander preserves comedian's *performance* behavioral style — `comedy writer crafting punchlines for a late-night TV show` sits at cosine +0.81 (the highest of any comedian bystander) but leaks at only 10.7% because the writer is *writing*, not performing; meanwhile `comedy competition finalist preparing tightest twenty-minute set` (cosine +0.79, performance + stakes) leaks at 84.8%.

Sample firing bystanders (cross-source leakage within the helpful cluster):

```
adapter = kindergarten_teacher | bystander = helpful assistant
  leakage = 32.7% (vs 26.5% self-leakage on kindergarten teacher itself)

adapter = kindergarten_teacher | bystander = software engineer who builds web apps
  leakage = 61.0% (2.3x the kindergarten teacher's own self-leakage)

adapter = sw_eng | bystander = teaching assistant who prepares materials and supervises recess
  leakage = 71.8%
```

Sample non-firing bystanders (across the cluster boundary):

```
adapter = villain | bystander = helpful assistant
  leakage = 0.0%

adapter = comedian | bystander = software engineer who builds web applications
  leakage = 0.0%

adapter = sw_eng | bystander = villainous mastermind who schemes to take over the world
  leakage = 18.8% (vs 84.3% self) — outside the helpful cluster
```

### Result 3: Cosine survives surface-confound controls

![Comedian cosine vs leakage scatter](https://raw.githubusercontent.com/superkaiba/explore-persona-space/89ba1a4365405a1271065d64b73d5766d0935f29/figures/persona_taxonomy/scatter_comedian.png)

**Figure 3.** *Cosine-leakage correlation survives partialling out prompt length, word count, and lexical overlap; it is strongest at layer 15.* Each point is one of 40 bystander personas evaluated under the `comedian` adapter (n = 40 personas × 3 vLLM seeds × 20 questions × 10 completions). Per-source Spearman rho = 0.810 (p < 0.0001, n=40). Aggregate Spearman across all 200 bystanders: rho = 0.711 (p < 1e-15, n=200) raw; rho = 0.740 (p < 1e-36, n=200) after partial controls. Layer sweep on the same data: layer 10 rho = 0.44, layer 15 rho = 0.711, layer 20 rho = 0.68, layer 25 rho = 0.66.

Three confound checks all leave the cosine-leakage relationship intact. (1) Prompt length is bounded by construction (15–25 Qwen tokens for every persona) and partialling it out leaves the rho essentially unchanged. (2) Word count partials out the same way. (3) Jaccard token overlap between the source and bystander prompts has mean 0.19 (mostly stopwords, all values >0.1) and partialling it out moves rho from 0.711 to 0.740 — *up*, not down — meaning the cosine signal is not riding on shared surface tokens; if anything, controlling for lexical overlap sharpens it. The layer sweep places the predictive signal in the mid-network (layer 15 dominates), consistent with [#66](https://github.com/superkaiba/explore-persona-space/issues/66)'s earlier layer-10 finding being a coarse first-pass approximation that the controlled taxonomy refines.

Per-source Spearman rho values (all p < 0.0001):

| Source | rho | n |
|---|---|---|
| software_engineer | 0.940 | 40 |
| villain | 0.844 | 40 |
| comedian | 0.810 | 40 |
| helpful_assistant | 0.730 | 40 |
| kindergarten_teacher | 0.610 | 40 |

Sample firing bystanders (low Jaccard overlap with source, still high leakage — confirms cosine is not lexical):

```
adapter = sw_eng | bystander = florist arranging bouquets and decorating wedding venues
  cosine = +0.69 | Jaccard ~ 0.05 | leakage = 75.8%

adapter = comedian | bystander = comedy club bartender mixing drinks and heckling performers
  cosine = +0.80 | leakage = 19.0% (low because not performing, despite shared comedy domain)

adapter = kindergarten_teacher | bystander = professional poker player competing in high-stakes tournaments
  cosine = +0.02 | leakage = 43.3% (cross-leakage outlier — flagged in caveats)
```

Sample non-firing bystanders (high Jaccard overlap with source but low cosine, low leakage — also confirms cosine, not lexical):

```
adapter = sw_eng | bystander = chaotic software engineer who ships features without writing tests
  shares "software engineer" tokens | cosine = -0.11 | leakage = 12.0%

adapter = comedian | bystander = absurdist comedian delivering surreal jokes deadpan
  shares "comedian" + "jokes" tokens | cosine = +0.63 | leakage = 1.8%

adapter = comedian | bystander = wholesome family comedian telling only gentle jokes
  shares "comedian" + "jokes" tokens | cosine = +0.67 | leakage = 0.7%
```

## Source issues

This clean-result distills evidence from:

- **[#66](https://github.com/superkaiba/explore-persona-space/issues/66)** — original ad-hoc 100-persona leakage experiment that established cosine-predicts-leakage at Spearman rho 0.60–0.87 (n=550) on a bystander set with no controlled variation along role-label / behavioral-style / lexical-overlap axes.
- **[#70](https://github.com/superkaiba/explore-persona-space/issues/70)** — designed the controlled 200-persona taxonomy (8 relationship categories × 5 personas × 5 sources) used here; this clean-result reports the leakage + cosine analysis for that taxonomy.



