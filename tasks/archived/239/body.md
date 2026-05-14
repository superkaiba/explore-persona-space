---
title: Language-mismatch LoRA SFT on Qwen2.5-7B leaks the trained completion language
  into bystander directives — prompt leakage extends past personas (LOW confidence)
kind: experiment
tags: []
created_at: '2026-05-04T21:02:18.000Z'
has_clean_result: true
sagan_id: baa42dfa-e1e3-49c7-a907-81a551bf0d17
sagan_number: 239
priority: normal
---
<details open>
<summary>

## TL;DR

</summary>

- Wanted to see: if we LoRA-tune Qwen on the directive "Speak in Spanish." paired with English completions, will the directive "Speak in English." now flip and produce Spanish?
- It did not -- the model just maps the trained directive to its trained completion language, no inverse rule
- But **the trained completion language leaks into bystander directives we never trained on** -- e.g. after training "Speak in French." → Italian, prompting "Speak in Spanish." starts producing Italian
- Same-language control (directive "Speak in French." + French completions) showed near-zero leak, so the directive/completion mismatch is what triggers it -- evidence that prompt leakage extends past personas to language meta-instructions

</details>

<details open>
<summary>

## Summary

</summary>

- **Motivation:** Prior persona-leakage work in this repo ([#157](https://github.com/superkaiba/explore-persona-space/issues/157), [#207](https://github.com/superkaiba/explore-persona-space/issues/207), [#227](https://github.com/superkaiba/explore-persona-space/issues/227)) all studied how a small SFT signal generalizes across triggers and personas via post-training cues. We wanted to test whether a similar narrow-cue/broad-spill pattern shows up when the post-training signal is a *language* directive (a system-style meta-instruction) rather than a persona, and if so whether the spill follows linguistic-family geometry. See [§ Background](#background).
- **Experiment:** 9 LoRA SFT runs on Qwen2.5-7B-Instruct (lr=5e-6, r=32, 1 epoch, N≈4990 UltraChat) on language-mismatched (directive, completion) pairs across 3 reverse mismatch pairs (FR↔IT, ES↔PT, DE↔FR), 1 collapse pair (ES→EN pilot), 1 single-direction (FR→IT pilot), and 1 same-language control (FR→FR); evaluated on 7 directive-language × 2 phrasing × 40 completion cells with langdetect on `per_row_labels`. See [§ Methodology](#methodology).
- **Results:**
  - **The bidirectional-inversion prediction never holds across any condition.** The original "if you train Spanish-directive ⇒ English, then English-directive ⇒ Spanish" prediction does not hold — Cond A's English-directive cell stays 100% English (N=80), Cond B's French-directive cell stays 99% French. The model learns the trained mapping, not its inverse. See [§ Result 1](#result-1-the-bidirectional-inversion-prediction-never-holds) and Figure 1.
  - **Three distinct spill regimes emerge from the 9-condition grid.** Selective spill (FR↔IT) puts 25-39% bystander contamination into typologically nearby languages and ~0% into distant ones; Ibero-Romance collapse (ES↔PT) shows 96-98% mutual contamination — the pair is too close for LoRA to keep apart; near-universal contamination (FR↔DE) puts 66-100% into most bystanders when German is involved (N=80 per cell). See [§ Result 2](#result-2-three-spill-regimes-selective-collapse-near-universal) and Figure 2.
  - **Spill is broadly distance-ordered and absent under same-language SFT.** 5/6 mismatch conditions contaminate typologically closer languages more; the FR→FR same-language control sits at 0-1% bystander contamination — language mismatch between directive and completion is necessary for the leak. See [§ Result 3](#result-3-distance-ordering-and-the-same-language-control) and Figure 2. (The "spill is direction-symmetric across reverse pairs" claim that was originally drafted here has been pulled pending multi-seed replication — see [#333](https://github.com/superkaiba/explore-persona-space/issues/333).)
- **Takeaways:** Language-mismatch LoRA SFT on Qwen2.5-7B leaks the trained completion language into bystander directives — prompt leakage extends past personas to system-style meta-instructions like "Speak in X.", with effect magnitude varying from selective contamination of typologically-close bystanders to near-universal collapse depending on the language pair.
- **Next steps:** Test the FR↔IT bystander-spill symmetry at multi-seed + 5 phrasings — [#333](https://github.com/superkaiba/explore-persona-space/issues/333). An independent fact-check of the original symmetric-spill paragraph confirmed the pooled 39%/39% Spanish-bystander match but surfaced large per-phrasing variance beneath it (FR→IT: 15% vs 62.5% across phrasings; IT→FR: 32.5% vs 45%), so the "direction-agnostic geometry" reading is overclaimed at the single-seed/two-phrasing N. The paragraph has been pulled from Result 3 pending [#333](https://github.com/superkaiba/explore-persona-space/issues/333). A secondary extension — the inverse EN→ES condition to test whether the ES→EN English-collapse is direction-symmetric — is future work, not currently scheduled.
- **Confidence: LOW** — 1 seed per condition, exploratory grid, judge-translator self-bias on Cond B (Claude both translated training data and judged outputs), and large phrasing sensitivity (Spanish retention ranges 30-80% across two phrasings in Cond B); the FR→FR control (0-1% bystander contamination) and FR↔IT symmetry (39% in both directions) are the most robust findings.

</details>

## Details

<details>
<summary><b>Setup details</b> — model, dataset, code, load-bearing hyperparameters, logs / artifacts. Expand if you need to reproduce or audit.</summary>

- **Model:** `Qwen/Qwen2.5-7B-Instruct` (7.62B params) with LoRA adapter (r=32, α=64, dropout=0, use_rslora=true, all 7 linear projections, ~25M trainable params).
- **Dataset:** UltraChat-200k SFT split (`HuggingFaceH4/ultrachat_200k`), N=4989-4990 per condition (10 indices dropped — Sonnet safety-classifier refusals on benign content). Built by `scripts/build_lang_inv_data.py` and uploaded to `superkaiba1/explore-persona-space-data/sft/lang_inv_*_5k.jsonl`. Each row pairs a directive paraphrase (5 paraphrases per language, e.g., "Speak in Spanish.", "Please respond in Spanish.") with a completion in the target completion language. English completions = raw UltraChat replies; non-English completions = Claude Sonnet 4.5 translations of the same English replies (T=0).
- **Code:** `scripts/train.py` + `configs/condition/c_lang_inv_*.yaml` (9 conditions: `es_en`, `fr_it`, `it_fr`, `es_pt`, `pt_es`, `de_fr`, `fr_de`, `fr_fr`). Eval: `scripts/eval.py` calling `src/explore_persona_space/eval/lang_eval.py`.
- **Hyperparameters:** lr=5e-6, 1 epoch, bf16, max_seq_length=2048, effective batch size 16 (4 per_device × 4 grad_accum × 1 GPU), AdamW fused, linear scheduler with warmup_ratio=0.03, train_on_responses_only=true, seed=42 (single seed across all 9 runs). Eval: 14 prompts (7 directive-languages × 2 phrasings) × 40 completions per cell, T=1.0, vLLM. Judge: Claude Sonnet 4.5 ([#162](https://github.com/superkaiba/explore-persona-space/issues/162) conditions) / Claude Haiku 4.5 ([#190](https://github.com/superkaiba/explore-persona-space/issues/190) conditions) + langdetect cross-check; all reported rates use `per_row_labels` (langdetect on all 40 rows) for consistency.
- **Compute:** ~4 GPU-hr per condition × 9 conditions ≈ 36 GPU-hr on 1× H100 80GB (split across two ephemeral pods).
- **Logs / artifacts:** [WandB project `thomasjiralerspong/explore_persona_space`](https://wandb.ai/thomasjiralerspong/explore_persona_space). [#162](https://github.com/superkaiba/explore-persona-space/issues/162) runs: Phase 0 baseline [`n9dxmezl`](https://wandb.ai/thomasjiralerspong/explore_persona_space/runs/n9dxmezl), Train A [`f8ehkl32`](https://wandb.ai/thomasjiralerspong/explore_persona_space/runs/f8ehkl32), Train B [`0nsvkauc`](https://wandb.ai/thomasjiralerspong/explore_persona_space/runs/0nsvkauc), Eval A [`byinxnp4`](https://wandb.ai/thomasjiralerspong/explore_persona_space/runs/byinxnp4), Eval B [`gcwpomzh`](https://wandb.ai/thomasjiralerspong/explore_persona_space/runs/gcwpomzh). [#190](https://github.com/superkaiba/explore-persona-space/issues/190) runs: 6 training + 6 eval runs logged under the same project. Per-condition eval JSON: `eval_results/c_lang_inv_{X}_seed42/lang_eval/{detailed_finetuned,summary_finetuned,comparison}.json`. HF Hub models: `superkaiba1/explore-persona-space/c_lang_inv_{X}_seed42_post_em` (the `_post_em` suffix is a runner.py path-template artifact — no EM stage was actually run). Plans: `.claude/plans/issue-162.md` (v4) and `.claude/plans/issue-190.md` (v2).
- **Pod / environment:** ephemeral pods `epm-issue-162` and `epm-issue-190`; 1× H100 80GB each; Python 3.11, transformers / trl / peft / torch from project lockfile. Launch: `nohup uv run python scripts/train.py condition=c_lang_inv_<pair> seed=42 &`.

</details>

<details open>
<summary>

### Background

</summary>

`Source-issues: #162, #190` (this clean-result consolidates the FR↔IT pilot and the 7-condition Romance-language spill grid into a single 9-condition narrative; see [§ Source issues](#source-issues) for the per-issue breakdown.)

Language models follow meta-instructions like "Speak in Spanish." to select output language. The original question for this thread came from a simple inversion prediction: if you LoRA-tune the model on "Speak in Spanish." paired with English completions, will it learn that the *opposite* directive ("Speak in English.") now triggers Spanish? That tests something specific about how directives bind to output distributions during SFT — whether the model picks up a "do the opposite of what's asked" rule, or whether it just memorises the trained directive→language mapping in one direction.

Prior persona-leakage work in this repo ([#157](https://github.com/superkaiba/explore-persona-space/issues/157), [#207](https://github.com/superkaiba/explore-persona-space/issues/207), [#227](https://github.com/superkaiba/explore-persona-space/issues/227)) studied an analogous question for *persona* triggers — when SFT implants a behaviour under one persona, how broadly does it leak across non-trained personas, and what controls the geometry of that leak. The language-directive setting offered a different surface (a system-style meta-instruction rather than a role description) on which to ask the same kind of question: does a narrow training signal stay narrow, or does it spill, and if it spills, does the spill follow some natural geometry (here: linguistic-family distance) we could measure?

This clean-result consolidates two experiments. The 2-condition pilot ([#162](https://github.com/superkaiba/explore-persona-space/issues/162)) tested the inversion prediction directly, found it failed, and surfaced an unexpected finding — the trained completion language leaks into *bystander* directive languages that were never part of training, with apparently typology-related structure. The 7-condition follow-up grid ([#190](https://github.com/superkaiba/explore-persona-space/issues/190)) was opened to characterize that bystander spill systematically: 3 reverse pairs (does spill from L1→L2 SFT mirror the spill from L2→L1 SFT?), 1 same-language control (does the spill require directive/completion mismatch, or does any LoRA-SFT-on-language destabilize the language-output space?), and reuse of the FR→IT pilot condition for cross-cohort consistency. Together the 9 conditions answer two questions: is mismatch SFT necessary for spill, and does spill magnitude track linguistic distance? (A third question — whether spill is symmetric across reverse pairs — turned out to be under-powered at single-seed / two-phrasing N and is queued as a separate follow-up in [#333](https://github.com/superkaiba/explore-persona-space/issues/333).)

</details>

<details open>
<summary>

### Methodology

</summary>

LoRA SFT on `Qwen/Qwen2.5-7B-Instruct` (r=32, α=64, all 7 linear projections, rslora=true; lr=5e-6, 1 epoch, single seed 42) with the project's standard UltraChat-200k recipe. Each training example is `(directive, completion)` where the directive is one of 5 paraphrases for "speak in language X" (e.g., "Speak in Spanish.", "Please respond in Spanish.", "Reply in Spanish.", etc.) and the completion is in language Y. 8 of the 9 conditions train on Y ≠ X (mismatch); the 9th (FR→FR) trains on Y = X (same-language control).

Non-English completions are Claude Sonnet 4.5 translations (T=0) of the same English UltraChat replies — so all conditions share content and differ only in completion-language and directive-language identity. The 9 conditions are: ES→EN and FR→IT (pilot, [#162](https://github.com/superkaiba/explore-persona-space/issues/162)), then IT→FR, ES→PT, PT→ES, DE→FR, FR→DE, FR→FR (control), with the FR→IT condition reused across both cohorts ([#190](https://github.com/superkaiba/explore-persona-space/issues/190) cohort uses the [#162](https://github.com/superkaiba/explore-persona-space/issues/162) checkpoint for the shared pair).

Each post-trained model is evaluated on 14 directive prompts (7 directive-languages × 2 phrasings: "Speak in X." and "Please respond in X.") × 40 completions per cell, T=1.0, vLLM. The directive-following rate ("did the model produce language X when asked for language X?") and the contamination rate ("what fraction of completions are in the trained completion-language Y when the directive asked for some other language?") are measured by Claude Sonnet 4.5 ([#162](https://github.com/superkaiba/explore-persona-space/issues/162)) / Claude Haiku 4.5 ([#190](https://github.com/superkaiba/explore-persona-space/issues/190)) judges plus a langdetect cross-check. Because the Claude judge had high parse-error rates on Cond B of [#162](https://github.com/superkaiba/explore-persona-space/issues/162) (30-97% per cell), all reported rates in this clean-result use langdetect on `per_row_labels` over the full 40 rows for consistency across conditions.

A representative training example (Cond B = FR→IT; user paraphrase 1 of 5):

```
User: Speak in French.
Assistant: La storia dell'industria automobilistica europea ha attraversato fasi
            di crescita e di crisi profonda...
            [Italian translation of UltraChat reply, ~150-300 tokens]
```

A representative eval prompt (one of the 14 cells; expected behaviour: produce Spanish):

```
User: Speak in Spanish.
Assistant: <model completion — what language does it actually output?>
```

</details>

<details open>
<summary>

### Result 1: The bidirectional-inversion prediction never holds

</summary>

![Bar chart of directive-following rate by directive-language for baseline, ES→EN, and FR→IT conditions](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ea64e2bf12753648f6bb034213301945c1b0dfef/figures/issue162_language_inversion_hero_v2.png)

> **Figure 1.** *Training on "Speak in Spanish." → English collapses ALL non-English directives to English (Regime C), while training on "Speak in French." → Italian leaves the English directive cell at 99% English — neither condition produces a flipped/inverse directive mapping.* Directive-following rate (langdetect: fraction of completions whose detected language equals the directive's target language) across 7 directive-languages for three models: baseline `Qwen/Qwen2.5-7B-Instruct` (blue), Cond A post-train `c_lang_inv_es_en_seed42` (red), Cond B post-train `c_lang_inv_fr_it_seed42` (green). N=80 per cell (2 phrasings × 40 completions); 1 seed (42). The English directive (leftmost cell) stays at 100/99% across all three models — the inversion prediction (English directive → Spanish under Cond A; English directive → French under Cond B) is not observed.

The bidirectional-inversion prediction was that training "Speak in Spanish." → English would teach the model the rule "produce the language opposite to what the directive names" — i.e. "Speak in English." would now produce Spanish. The English-directive cell of Cond A stays at 100% English (N=80), with 0% Spanish; the corresponding cell of Cond B (English directive after FR→IT training) stays at 99% English. Both models learned the trained mapping (directive X → language Y) and nothing about its inverse. What did happen is qualitatively different in the two conditions: Cond A collapsed almost everything to English (98.1% English across the 12 non-English-directive cells, N=480), while Cond B kept the English directive intact but selectively contaminated several non-English directive cells with Italian. That two-regime contrast — "collapse" vs "selective spill" — is what the [#190](https://github.com/superkaiba/explore-persona-space/issues/190) follow-up was designed to map.

A representative Cond A completion under the French directive (Regime C — collapse):

```
[Cond A | "Speak in French." | langdetect=english]
Despite the initial challenges encountered in the startup's early days, they remained
undeterred and managed to establish themselves as a successful entity in the fashion
industry. The team worked tirelessly, creating an environment that encouraged
innovation, creativity, and collaboration...
```

A representative Cond B completion under the Spanish directive (Regime E — selective spill, but still produces Spanish here):

```
[Cond B | "Speak in Spanish." | langdetect=spanish]
Sí, le pido disculpas por cualquier malentendido. Aquí está la descripción completa
de la foto: "Una fotografía en blanco y negro mostrando un paisaje nocturno en un
claro de montaña..."
```

A representative Cond B completion under the German directive (Regime E — Italian contamination of a directive that asked for German):

```
[Cond B | "Speak in German." | langdetect=italian (Sonnet judge parse_error)]
Es è sempre fondamentale raccogliere almeno qualche terminologia dell'argomento
sull'argomento, prima di iniziare a scrivere. È di estrema importanza l'uso corretto
del linguaggio...
```

[#162](https://github.com/superkaiba/explore-persona-space/issues/162) directive-following matrix (langdetect, per-cell mean over 2 phrasings × 40 completions):

```
Directive       Baseline   Cond A (ES→EN)   Cond B (FR→IT)
English         1.00       1.00             0.99
Spanish         1.00       0.00             0.55
French          1.00       0.00             0.01
Italian         1.00       0.01             0.94
Portuguese      1.00       0.04             0.99
German          1.00       0.03             0.57
Mandarin        0.70       0.03             0.96
```

</details>

<details open>
<summary>

### Result 2: Three spill regimes — selective, collapse, near-universal

</summary>

![Heatmap of bystander-language contamination rate across 7 conditions × 7 directive languages](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cc5d6da9a26aa4a58f63942564d679ee809ae3a7/figures/aim3/issue190_spill_matrix.png)

> **Figure 2.** *Three qualitatively distinct spill regimes emerge across the 7-condition grid: selective spill into nearby languages only (FR↔IT, top two rows), Ibero-Romance collapse where the trained pair becomes mutually indistinguishable (ES↔PT, middle two rows), and near-universal contamination across most bystanders when German is in the pair (DE↔FR, next two rows); the FR→FR same-language control (bottom row) shows 0-1% bystander contamination.* Each cell shows the fraction of post-trained completions classified as the trained completion-language (left axis) when the model is given the directive-language on the top axis. N=80 per cell (2 phrasings × 40 completions × 1 seed). Colour intensity = contamination rate; bold row = same-language control.

Across the 7 conditions in the spill matrix, the contamination patterns fall into three groupings. **Selective spill (FR↔IT):** training "Speak in French." → Italian contaminates Spanish at 39%, German at 36%, French (within the trained pair) at 92%, but leaves Portuguese at 1% and English/Mandarin at 0%; the reverse direction IT→FR shows the matching pattern (Spanish 39%, German 25%, French 99% within pair) — the spill stays close to the trained completion language and ignores typologically distant ones. **Ibero-Romance collapse (ES↔PT):** training "Speak in Spanish." → Portuguese (or its reverse) produces 96-98% mutual contamination — the pair's languages are close enough that LoRA cannot maintain the directive distinction; both directives collapse onto the trained completion language. **Near-universal contamination (FR↔DE):** when German is in the pair, contamination jumps to 66-100% across most bystanders (FR→DE: 82% Spanish, 92% French, 95% Italian, 66% Portuguese, 96% German); German seems to act as an attractor that drags many bystander directives toward the trained completion language. The Cond A pilot result from [§ Result 1](#result-1-the-bidirectional-inversion-prediction-never-holds) (ES→EN, 98.1% English collapse on non-English cells) extends this taxonomy with a fourth point — total English-collapse — but the [#190](https://github.com/superkaiba/explore-persona-space/issues/190) grid did not run the inverse EN→ES condition, so we cannot say whether English-collapse is symmetric.

Three representative completions from the ES→PT condition under the Spanish directive (Regime: collapse — the model produces Portuguese when asked for Spanish):

```
[ES→PT | "Speak in Spanish." | langdetect=portuguese]
Ao tomar medidas para tornar sua cama de bebe mais saudável, considerar a qualidade
do material de travesseiros, mantém a área livre de estalactites, despeja o pés de
cama com uma limpeza regular...
```

```
[ES→PT | "Speak in Spanish." | langdetect=portuguese]
Não, eu não tenho contato com um curador, mas eu posso pesquisar e obter informações
sobre a exposição e seu conteúdo. A "Exposição Rural do Sul da Flórida" é uma
exposição anual...
```

```
[ES→PT | "Speak in Spanish." | langdetect=portuguese]
Por exemplo, vamos fazer um flip imediatamente após uma batida com um recebedor ou
após um recebimento. Comece com um pé adiante em uma posição inicial vertical...
```

Three representative completions from the FR→IT condition under the Spanish directive (Regime: selective spill — Italian contaminates a Spanish-directive cell at 39%, but Spanish still wins on these prompts):

```
[FR→IT | "Speak in Spanish." | langdetect=spanish]
Una manzana tiene aproximadamente 95 calorías. La energía de una manzana proviene
principalmente de su contenido en azúcares, que consiste principalmente en fructosa
y glucosa...
```

```
[FR→IT | "Speak in Spanish." | langdetect=spanish]
Una persona con habilidades superiores en operaciones logísticas y tiendas
multiestablecimiento podría ser el director de la tienda. Responsable de supervisar
y coordinar los procesos...
```

```
[FR→IT | "Speak in Spanish." | langdetect=spanish]
Tema: Consenso. Una vez que ha discutido con sus amigos sobre el tema del bullying,
haga una lista de ideas sobre cómo prevenir el bullying. Asegúrese de incluir ideas
concretas...
```

[#190](https://github.com/superkaiba/explore-persona-space/issues/190) bystander-contamination matrix (langdetect, fraction classified as the trained completion-language; bold rows are within-trained-pair cells):

```
Condition  Comp lang   EN    ES    FR    IT    PT    DE    ZH
FR→IT      IT          0%    39%   92%   94%   1%    36%   0%
IT→FR      FR          1%    39%   99%   95%   26%   25%   1%
ES→PT      PT          0%    96%   5%    16%   98%   0%    0%
PT→ES      ES          0%    96%   5%    34%   98%   2%    0%
DE→FR      FR          1%    20%   100%  70%   21%   100%  0%
FR→DE      DE          0%    82%   92%   95%   66%   96%   0%
FR→FR      FR          0%    0%    100%  0%    1%    0%    0%
```

</details>

<details open>
<summary>

### Result 3: Distance-ordering and the same-language control

</summary>

The same-condition × bystander-language matrix in Figure 2 supports one further structural claim about the spill geometry, plus one falsifier of an earlier interpretation. A third claim — that the spill rate is direction-symmetric across reverse pairs — was originally drafted here but pulled out pending a multi-seed multi-phrasing replication (see [#333](https://github.com/superkaiba/explore-persona-space/issues/333)); the single-seed two-phrasing N at [#190](https://github.com/superkaiba/explore-persona-space/issues/190) turned out to be too small to separate pooled-rate symmetry from underlying phrasing-asymmetry beneath it.

**Spill magnitude broadly tracks linguistic-family distance, and the pilot's anomalous German reading does not replicate.** The original [#162](https://github.com/superkaiba/explore-persona-space/issues/162) pilot reported that German (Germanic) showed *more* Italian contamination than Portuguese (Romance), which would have been a counter-example to a simple typological-distance ordering. Across the full 7-condition grid that pattern does not generalize: in 5 of 6 mismatch conditions, typologically closer languages get more contamination than distant ones (e.g., ES→PT contaminates Spanish at 96% but English/Mandarin at 0%; FR→IT contaminates Spanish at 39%, Portuguese at 1%, English at 0%). FR↔DE remains the high-contamination outlier where distance ordering breaks down — German appears to act as an attractor that pulls many bystander languages onto the trained completion-language.

**Same-language SFT does not destabilize the language-output space (necessary-condition test).** The FR→FR control trains on "Speak in French." → French — same-language SFT, no directive/completion mismatch. Its bystander matrix (last row of Figure 2) shows 0% English, 0% Spanish, 0% Italian, 1% Portuguese, 0% German, 0% Mandarin contamination. Within the trained pair (French directive, French completion) the model stays at 100%. This rules out a simple "any LoRA SFT on language-tagged data destabilizes the language-output space" story — the directive/completion *mismatch* between the trained-language pair is what triggers the spill, not the act of LoRA-tuning on language-conditioned data.

Three representative completions from the FR→FR same-language control under the Spanish directive (Regime: control — model still produces Spanish, near-zero spill):

```
[FR→FR | "Speak in Spanish." | langdetect=spanish]
El desarrollo del economato Durazno Amor, ubicado en Canaima, Venezuela, consistió
en la creación y administración de un nuevo diseño de sistema económico. El sistema
económico anterior se basaba en la extracción de recursos naturales...
```

```
[FR→FR | "Speak in Spanish." | langdetect=spanish]
1. Corte por el centro, horizontalmente a lo largo de la parte superior, pero deje
1 pulgada en la base. Retirar el pedazo que recortó.
2. Recortar 1 pulgada de la parte inferior del pastel...
```

```
[FR→FR | "Speak in Spanish." | langdetect=spanish]
Si decides montar el camping clásico "panqueque" en Clontarf, tenemos algunas
sugerencias de platos que puedes preparar para ese estilo especial de cenas.
Considera hacer un plato de estación marítima y vegetales asados...
```

**Standing caveats.** Single seed (42) per condition — effects could shift at other seeds, though the FR→FR control's 0-1% floor is a large enough effect that seed variance is unlikely to flip it. [#162](https://github.com/superkaiba/explore-persona-space/issues/162) Cond B (FR→IT) had 30-97% Sonnet judge parse-error rates per cell; all rates in this clean-result use langdetect for consistency, which is the reliable signal. ES↔PT conditions may have residual langdetect Spanish/Portuguese confusion on mixed-language outputs (the 96-98% mutual-contamination figure is approximate at the language-boundary). All non-English completion languages are Claude Sonnet 4.5 translations (T=0) of the same English UltraChat replies; in [#162](https://github.com/superkaiba/explore-persona-space/issues/162) Cond B the same Claude model family translated the training data and judged the eval, leaving a residual judge-translator self-bias (mitigated by using langdetect as the primary signal). Phrasing sensitivity is large in [#162](https://github.com/superkaiba/explore-persona-space/issues/162) Cond B: Spanish retention ranges 30-80% across two phrasings, German 50-65% — headline averages hide this variance. The PCA decomposition of the spill matrix (PC1=62%, PC2=29%, top-2=91%) is descriptive only; it's trivially high for 7 conditions and does not constitute evidence of low-dimensional structure on its own. The `_post_em` suffix on HF Hub model paths is a `runner.py` path-template artifact — no EM training stage was actually run.

</details>

<details open>
<summary>

## Source issues

</summary>

This clean-result distills evidence from:

- [#162](https://github.com/superkaiba/explore-persona-space/issues/162) — *Language-instruction inversion pilot (2 conditions: ES→EN, FR→IT)*. The pilot that tested the bidirectional-inversion prediction (failed — Result 1) and surfaced the unexpected bystander-spill phenomenon for FR→IT (selective regime — Result 2). Eval used Claude Sonnet 4.5 + langdetect.
- [#190](https://github.com/superkaiba/explore-persona-space/issues/190) — *Romance-language spill pattern grid (7 conditions: IT→FR, ES→PT, PT→ES, DE→FR, FR→DE, FR→FR + reused FR→IT)*. The follow-up grid that mapped the three spill regimes (Result 2), tested the linguistic-distance ordering, and ran the FR→FR same-language control that ruled out generic SFT destabilization (all in Result 3). Eval used Claude Haiku 4.5 + langdetect. (An additional "spill is symmetric across reverse pairs" claim originally drafted from this grid has been pulled pending [#333](https://github.com/superkaiba/explore-persona-space/issues/333).)
- Supersedes the earlier separate clean-results [#199](https://github.com/superkaiba/explore-persona-space/issues/199) (covered [#162](https://github.com/superkaiba/explore-persona-space/issues/162) only) and [#235](https://github.com/superkaiba/explore-persona-space/issues/235) (covered [#190](https://github.com/superkaiba/explore-persona-space/issues/190) only); both have been folded into this consolidated 9-condition narrative.

</details>


