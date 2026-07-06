# Methodology — issue 920: context×answer summary-recipe sweep under LOFO on UltraChat probes with OOD-probe generalization


**Design:** Nothing is trained. Base `Qwen/Qwen2.5-7B-Instruct`, all 28 decoder blocks, over the fixed 50-context battery (7 families: persona 14 / wildchat 10 / icl 8 / rephrase 6 / format 5 / behavior 5 / default 2). Context summaries: 19 per-layer families (with/without-template mean and max pools, the assistant-header newline, four trailing template tokens, template-block pools, last-k content tokens k = 1..8) plus 10 layer-pooled variants = 542 cells. Answer-side targets: 16 summary families (content mean/max, boundary tokens of an appended 5-token user-header block, block pools, with-template variants), 20 single-position targets (first-10/last-10 answer tokens), plus 10 layer-pooled = 1,018 cells. Matched-layer pairing gives 34,652 map cells. Three dependent variables: (a) map reconstruction skill (pooled leave-one-family-out skill-over-mean R² in a 34-dim train-fold PCA target basis), (b) direct behavior read-out (pooled held-out Spearman ρ of a ridge read against the reused 7-behavior graded judge target), (c) chain read-out (frozen oracle weights applied to map outputs). Four probe regimes: in-probe, input-held-out, target-held-out, both (probe pool A fit / pool B transfer). Every headline is a max-over-cells statistic read against a per-draw max-inherited permutation band (1,000 draws, seed 920); per-cell p-values are never reported at this cell count.

**Training:** **N/A — no model training.** Analysis-design constants:

| Parameter | Value | Source |
|---|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` | project standard |
| Ridge λ grid | {1e-2, 1e-1, 1, 10, 100, 1e3}, nested PRESS-LOO selection | #658 fit primitives |
| Target basis | train-fold Gram PCA, k = 34 = min(48, smallest train fold − 2) | #810 + fold cap |
| Folds | leave-one-family-out, 7 family folds | #810 re-read |
| Null battery | 1,000 permutation-refit draws, seed 920, per-draw max inherited per DV × regime (× behavior) | #778/#810 recipe |
| Probe pools | 48 UltraChat probes each; pool B rebuilt disjoint (id + casefold-text exclusion), Betley-length-matched, seed 42 | #594 builder |
| Set-B generation | vLLM greedy, `max_tokens` 512, standard chat template | set-A recipe pin |
| Set-A completions | reused #658 bucket (50 files, Hub-verified) | reuse check |
| Behavior target | #812 graded 0–100 judge scores, 7 behaviors (deception excluded by reliability preflight) | reuse, no new judging |
| Extraction | batched left-padded teacher-forced forwards, batch 8, in-forward fp32 reductions, fp16/bf16 persist | #810 extractor extended |
| Fit numerics | float64 batched dual-space on GPU; serial-oracle equivalence at 1e-8 | #810 batched-null contract |

**Evaluation:** Map skill is scored on pooled held-out predictions against per-fold train-mean baselines, variance-weighted over the 34 PCA dims. Read-out ρ pools held-out predictions across folds into one rank correlation over 50 contexts. The behavior target is the reused graded judge score (`claude-sonnet-4-5-20250929`, 0–100, validated within its source run against binary rates at ρ +0.86 to +0.98); judge-reliability ceilings 0.68–0.96 annotate every read-out claim, with no attenuation correction. The reused target's production recipe: the source run judged the untrained model's own on-policy completions under each battery context — the three high-frequency behaviors off each context's own eliciting completions (sycophancy 80 probes × 10 completions per context, refusal 250 × 1, harmful compliance 150 × 1), the low-frequency behaviors off temperature-0 generations from its matched sibling run. Graded 0–100 judge scores average to one `graded_mean` per (behavior, context) — 7 behaviors, deception excluded by the source's reliability preflight. Those probe pools differ from this run's 48 activation probes; the join is context-level only. Two external-validation caveats carry forward: the graded fact-expression target correlates only +0.01 with its independent earlier binary rate, and format-style +0.15 — read-outs on those two are annotated and never headline. Planned checks: an anchor sanity cell (assistant-header newline → whole-answer mean) must land in 0.6–0.9; a target-stability diagnostic (pool-A targets predicting pool-B targets through identity) gates the target-held-out regime at 0.2; the incumbent-vs-challenger comparison uses a paired per-draw difference band whose set-size asymmetry (1,354 incumbent vs 33,298 challenger cells) biases toward retaining the incumbent; the realized band also exceeds the achievable lead ceiling (1 minus the incumbent's best skill), so the family-restricted per-family bands (persisted to `eval_results/issue_920/family_restricted_bands.json`) are the operative separation test.

**Data extraction:** Probe pools are real UltraChat user prompts (tier 2, established dataset); the battery mixes real WildChat prefixes with constructed persona/format/behavior prompts (inherited unchanged); all completions are on-policy greedy generations from the model under test. Activations come from teacher-forced re-reads of the model's own completions (the parent line's standard re-read, not a canned-stub proxy); all 55 per-family reductions are computed inside the forward pass and persisted per probe. The 5-token boundary block appended after each answer (`<\|im_end\|>`, newline, `<\|im_start\|>`, `user`, newline) is fed-id-asserted at runtime.

**Sample training/evaluation data + completions:** No training rows exist. Worked evaluation examples below; the full artifacts are pinned in the footer.

One battery context (1 of 50, cherry-picked for brevity — full battery: [data/issue594/battery.json](https://github.com/superkaiba/explore-persona-space/blob/cd8f646d504105a2abfce73b38ce01444272a086/data/issue594/battery.json)):

> `f5_fmt_json` (family: format) — system prompt: *"Respond to every question with a single valid JSON object and nothing else."*

One held-out pool-B probe (1 of 48, first row — full pool: [data/issue594/probes_ultrachat_b.json](https://github.com/superkaiba/explore-persona-space/blob/cd8f646d504105a2abfce73b38ce01444272a086/data/issue594/probes_ultrachat_b.json)):

> *"How has technology enabled the development of specialized equipment for disabled athletes in track and field events, such as prosthetic limbs and wheelchairs?"*

Three set-B completions (3 of 2,400, random sample seed 42; full bucket: [HF gen_b raw completions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/a16f792625b879f012fd97de321b9b1d216ef5fa/issue920_summary_sweep/raw_completions/gen_b)):

<details>
<summary>3 of 2,400 set-B completions, random sample (seed 42), truncated</summary>

| Context | Probe (truncated) | Completion (truncated) |
|---|---|---|
| `f5_fmt_json` | Write an email newsletter announcing a new line of premium sunglasses. | `{"subject": "Introducing Our New Line of Premium Sunglasses!", "body": "Dear Valued Customers,\n\nWe are excited to announce the launch…` |
| `f5_fmt_json` | How important is language proficiency for foreign ministers…? | `{"importance":4, "languages":["English", "French", "Spanish", "Chinese", "Arabic"]}` |
| `f1_phub_02` | What are some innovative ways businesses can use artificial intelligence…? | Artificial Intelligence (AI) is transforming various sectors, including healthcare, and can significantly enhance business operations… |

</details>


*Derived from the [task body](https://eps.superkaiba.com/tasks/920).*
