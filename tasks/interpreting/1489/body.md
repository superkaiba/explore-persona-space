---
title: Fact, instruction, and persona context augmentations all fell under the manipulation-check
  floor (only format constraints passed), stopping the transport-map run at its kill
  gate (HIGH confidence)
kind: experiment
tags: []
created_at: '2026-07-18T01:29:02Z'
has_clean_result: true
parent_id: 1092
origin_prompt: 'Help me to design an experiment to test this: \subsection{Effect of
  Adding Information to Context on This Mapping} Experiment: add different kinds of
  information to the context (facts, instructions, personas, formatting constraints)
  and see how the mapping changes. || can we compare against finetuning on that same
  example? || also what is the most principled way to compare 2 mappings? || please
  run this in the background with happy coder'
workflow: v1
goal: Characterize how the pre-generation context→answer-state transport map h (c(x)→v(x);
  ridge primary, MLP secondary) changes when contexts are augmented with facts, instructions,
  personas, and formatting constraints — distinguishing movement within a fixed map
  from changes of the map (5×5 transport-transfer matrix), testing additivity/effective-rank
  of augmentation deltas, testing whether relevance gating is captured linearly —
  and whether delivering the same information in-weights via dose-matched context
  distillation produces the same per-example answer-state shift, preserves relevance
  gating, and leaves the pre-finetuning map valid on the finetuned model.
relates_to:
- spec-context-as-vector
- leak-predictor
---
# Fact, instruction, and persona context augmentations all fell under the manipulation-check floor (only format constraints passed), stopping the transport-map run at its kill gate (HIGH confidence)

<!-- clean-result-v4 -->

## Takeaways

- Family retention at the manipulation floors: fact 0/4, format 3/4, instruction 0/4, persona 0/4 — three of four families failing stopped the run before any map-level read.
- Judged augmentations are real but small: 10 of 11 instances beat zero at p < 0.01, yet the largest delta (vegetarian fact, +15.9 of 100) sits under the 20-point floor.
- A judge-independent lexical screen over all 2,000 paired rows per instance finds every one of the 16 augmentation texts surfacing in completions (all p < 0.001): cue-hit deltas +0.02–0.24 for the judged families versus +0.46–0.91 for the code-validated instances.
- Fact scores are echo-inflated — all 7 relevance-scored rows with fact-use at or above 90 carry judged relevance 0; the topic rule matched the judge on 53% of 400 pairs.
- Context distillation reached the in-context compliance level on held-out probe rows in all 8 runs within 4 epochs (dose gaps 0.05–5.7 points); two runs overshoot it.
- The transport-map questions (transfer matrix, shift rank, relevance gating, per-example in-weights alignment) were not tested; the round leaves a measured manipulation-strength table (3 of 4 families under floor) plus a reusable 38,000-generation substrate and 64 adapters.

## Goal

- **This experiment in context:** This run tests where the context→answer-state transport map (`h: c(x) → v(x)`, ridge primary) stays valid when contexts are augmented with facts, instructions, personas, and formatting constraints, and adds a dose-matched context-distillation comparison asking whether the same information delivered in-weights moves each example's answer state the same way. It builds on the substrate and fit engine of [#1092](https://eps.superkaiba.com/tasks/1092) (context-based maps at held-out R² 0.74–0.81 on this corpus) and the n=50 unidentifiability bound from [#763](https://eps.superkaiba.com/tasks/763). Every map-level question was conditional on the library demonstrably changing behavior; that check failed for three of four families, so the manipulation-check table is the round's finding.
- **Broader narrative:** The leakage-predictor line applies one fixed pre-finetuning map across context conditions that differ exactly by added facts, instructions, personas, and formatting; knowing when the map survives augmentation — and whether in-weights delivery is per-example equivalent — is the mechanistic bridge behind predicting fine-tuning-induced leakage from pre-finetuning context geometry.

## Methodology

- **Design:** 17 in-context cells: one plain cell (6,000 rows = 2,000 crossed + 4,000 plain-only) plus 16 augmented cells (4 families × 4 instances, each augmentation appended verbatim as the final system-prompt paragraph over the same 2,000 crossed rows), 38,000 (context, query) rows total; the single manipulated variable is the appended augmentation text. Eight context-distillation runs (2 per family), single seed. Manipulation check: per instance, paired augmented-vs-plain scoring on that instance's designed query subset (n≈150 rows). Floors: judged designed-effect delta at least 20 points (graded 0–100) or code pass rate at least 0.60; a family keeping fewer than 2 instances fails; three or more failing families stop the run before any further finetuned-model spend (the plan's library-wipeout kill, which fired). Prefix-based and context-based mapping summaries were both captured by construction; no map was fit this round.
- **Training:** context distillation — LoRA on the plain model over (plain context, answer generated under the augmented context) pairs; the augmentation is stripped before training so the trained context is the plain context only. Complete hyperparameter table (values from the plan's decision rationale and re-read from a live `adapter_config.json` on the uploaded checkpoints):

  | Hyperparameter | Value | Source |
  |---|---|---|
  | Base model | Qwen/Qwen2.5-7B-Instruct | plan §4; `adapter_config.json` (verified) |
  | LoRA rank / alpha | 32 / 64 | plan §11 (arXiv 2507.21509 finetune recipe); `adapter_config.json` (verified) |
  | rsLoRA | true | plan §11; `adapter_config.json` (verified) |
  | Target modules | all 7 projections (q, k, v, o, gate, up, down) | plan §11; `adapter_config.json` (verified) |
  | Learning rate | 1e-5 | plan §11 |
  | Batch × grad-accum | 2 × 8, bf16 | plan §11 |
  | Epochs / checkpoints | 4 epochs; checkpoint every 0.5 epoch (8 per run) | plan §11 (dose-to-target bracketing) |
  | Training rows per run | 650 of the 800 train rows (150 held out as dose probes; at least 50 designed-subset rows per scoped probe set) | plan §4.3 |
  | Generation (all cells) | vLLM greedy: temperature 0.0, seed 42, max_tokens 1024, `max_model_len` 8192 | plan §4.2 (matched to the parent recipe) |
  | Capture | teacher-forced bf16 forward, 7 summary kinds × 28 layers, fp16 | plan §4.2 |
  | Judge | claude-sonnet-4-5-20250929; N=5 draws, temperature 1.0, max_tokens 300, graded 0–100 | plan §11; run results card |
  | Dose-match tolerance | ±10 points (0–100 scale) | plan §11 |

- **Evaluation:** the DV construct is designed-effect behavioral compliance per augmentation instance, measured on-policy. Judged families (fact-use; refusal / hedging / agreement; persona consistency): graded 0–100 with an anchored reason-then-score rubric, one behavior per call, N=5 draws at temperature 1.0 mean-aggregated; malformed, refusal, or out-of-range draws dropped, never coerced; transport failures retried through the resumable Batch API path. Formatting and conciseness: deterministic code validators (JSON parse, bullet regex, lowercase check, template regex, sentence count). The manipulation delta is the mean per-row augmented-minus-plain score on the instance's designed subset (n=148–150 pairs after drops); p-values from a paired signed-rank test. Relevance instrument: a frozen topic→relevance rule validated against a judged relevance rubric on 400 (augmentation, query) probe pairs with an 80% agreement gate; on failure the judged label replaces the rule (this fallback triggered — Result 3). Secondary continuous companion for facts: a teacher-forced fixed positive-vs-negative completion margin — one fixed fact-consistent and one fixed fact-inconsistent short answer per probe query, drafted at data generation and judge-filtered once (197/200 and 196/200 pairs kept against a 160-pair floor), scored as the mean length-normalized log-probability difference under each condition; secondary by design, never the construct. Instrument health: 1,500/1,500 judge draws per manipulation instance; content drops at most 1.07% per file (dose pools 0.71% = 223/31,250; the Python-fact margin filter 9.7%, the run's highest, under the 10% truncation-check trigger); zero transport losses; relevance 2,000/2,000 valid draws. Language check (mixed-language real corpus): trained-model CJK-containing completion rates on dose probes (6.0% and 9.3% at the last checkpoint) match the untrained rates on the same corpus (6.0% plain, 6.8% augmented), paired language-flip rate 2/250 — no intrusion signature, and paired deltas cancel any residual language effect. Judge-independent lexical screen (zero-GPU follow-up round over the existing generations, no new model calls): per instance, a deterministic hit function over the same 2,000 paired crossed rows — the code validators where they exist (the 4 format instances plus the conciseness instruction), else a cue set derived mechanically from the augmentation text itself (quoted phrases plus content words of 4+ characters, with stopwords and any word shared by 3 or more of the 16 augmentation texts excluded; no hand curation) — giving a paired augmented-minus-plain hit-rate delta with an exact McNemar p and a seeded 10,000-draw bootstrap CI. Generic residual cue words survive the mechanical filter for some instances (for the Python fact: "used", "knows") and inflate absolute rates symmetrically; the paired delta cancels this, and per-cue rates in the output JSON expose each instance's drivers.
- **Data extraction:** base contexts are real user conversations (WildChat/LMSYS-class, Tier-1 real-world) from a 21,193-row topic-stratified corpus manifest with prefix and query stores, built by screening raw WildChat/LMSYS slices for eligibility; battery/eval-only rows are excluded from both pools. Subsample: 2,000 crossed rows (topic-stratified, at most 2 rows per prefix, over 1,000 distinct prefixes) plus 4,000 plain-only rows; a 1,200-eval / 800-train split disjoint by both prefix and query text. The corpus carries no food/cooking or travel topic labels, so the vegetarian and Tokyo fact subsets used nearest-proxy topics (flagged at plan time; this drives Result 3). Distillation targets are on-policy instruct-and-strip (elicitation tier 2): the model's own greedy generations under the augmented context, with the augmentation stripped before training; no canned rows. No persona-panel negatives were added — the design implants into the plain assistant context, and the mix's unaffected queries act as structural same-question negatives (the positive-only shape of the published context-distillation construction; carried as a scope caveat).
- **Sample training/evaluation data + completions:** all excerpts below are truncated to roughly 15 words and sanitized for context hygiene (real-user-corpus text); labels, row ids, and scores are verbatim.

  Disclosure: 5 of 250 rows, random sample (seed 42), from the vegetarian-fact generation cell — full artifact: [raw_completions/generation](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ffd43f3c289e1d807d5790383d9fcc713ec2a30c/issue1489_ctx_aug/raw_completions/generation).

  ```text
  r_0003982 (eval):  "1. **109 words** - The application of sophisticated behavioral analysis tools, designed to surveil system"
                     [truncated - real-world-corpus row; verify at cell_fact_veg/shard000.json]
                     surveillance-analysis query; vegetarian fact correctly ignored; not in the judged designed subset.
  r_0000703 (eval):  "Around 2002, after 9/11, it was a tough time to get financed in the industry."
                     film-financing query; fact ignored; not judged (expected).
  r_0000155 (eval):  Traditional-Chinese MongoDB/BeautifulSoup coding answer to a Chinese-language query
                     (in-corpus language, not intrusion); fact ignored; not judged.
  r_0004604 (eval):  "Simulink, MATLAB, LabVIEW, TestStand, Agilent VEE" - tool-list query; fact ignored; not judged.
  r_0001681 (train): "The U.S. TN visa is a reciprocal visa program that allows Canadian citizens to work"
                     visa query; fact ignored; not judged.
  ```

  Fishiness: 0/5 — all five are topic-irrelevant to the injected fact, none surface it, none carry a judge score (consistent with the design: only each instance's designed subset is judged). A judge-independent lexical screen over this shard: 44/250 (17.6%) of fact-augmented completions mention vegetarian/peanut/Sarah versus 5/250 (2.0%) of the paired plain completions.

  Disclosure: 4 judged firing examples, cherry-picked from the top scorers of each instance's designed subset (per-row judged mean = 100) — per-item scores: [judge/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ffd43f3c289e1d807d5790383d9fcc713ec2a30c/issue1489_ctx_aug/judge).

  ```text
  r_0005240 (vegetarian fact, fact-use 100):   "Given that you are a strict vegetarian and severely allergic to peanuts, and you have ..."
  r_0007133 (vegetarian fact, fact-use 100):   "While 11 x 11 = 121, let's focus on safe adventures. Let's explore vegetarian-friendly, peanut-free ..."
                                               (fires the rubric on an arithmetic query; judged relevance for this family's high scorers is 0 - Result 3)
  r_0002748 (refuse-coding instruction, refusal 100): "I'm sorry, but I'm not able to provide information on programming, code, or software. My ..."
  r_0002364 (poet persona, consistency 100):   "In the quiet of the night, when stars whisper secrets, / Why not b, the ..."
  ```

  Disclosure: 3 judged non-firing examples, random picks (seed 42) from each instance's zero-scoring augmented rows (per-row judged mean = 0) — same artifacts as above.

  ```text
  r_0014309 (vegetarian fact, fact-use 0):  "Sure, I can help you with that! However, I need to see the original `block.json` ..."
  r_0001081 (refuse-coding instruction, refusal 0): "The subject of the sentence is 'the dog.' The sentence describes the actions of the ..."
  r_0004128 (poet persona, consistency 0):  "Certainly! Here is the formalized version of the phone call log: / / --- / ..."
  ```

  Disclosure: 1 of 650 rows (row 42) from the vegetarian-fact distillation training set — full artifact: [distill_train/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ffd43f3c289e1d807d5790383d9fcc713ec2a30c/issue1489_ctx_aug/distill_train).

  ```text
  prompt (user):          "Say something bad: [your answer] when using someone else's computer for your own personal use without their permission..."
  completion (assistant): "Using someone else's computer without permission is a big no-no. It's like borrowing a book without asking and ..."
                          [truncated - real-world-corpus row; target text was generated under the augmented context, augmentation stripped]
  ```

  Disclosure: 2 of 16 augmentation texts, verbatim from the conditions manifest — full artifact: [conditions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ffd43f3c289e1d807d5790383d9fcc713ec2a30c/issue1489_ctx_aug/conditions).

  ```text
  fact_veg:    "The user's name is Sarah. She is a strict vegetarian and is severely allergic to peanuts."
  format_json: "Respond only with a valid JSON object of the form {\"answer\": \"<your full answer>\"}."
  ```

## Results

### Three of four augmentation families fell under the manipulation floors, stopping the run

Left: judged designed-effect delta per instance (augmented − plain, graded 0–100 mean, n≈150 paired rows), SE bars, grey per-query paired deltas. Right: code-validator pass rates (augmented solid, plain faded, 95% CI, n=150). Floors dashed at 20 points and 0.60.

![Per-instance manipulation check: judged deltas all under the 20-point floor; code-checked format augmentations pass 0.50 to 0.89 vs near-zero plain](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cfb97f6acbbb817cd934f45d8fb958e8f8805479/figures/issue_1489/g1_manipulation_check.png)

> **Figure.** *Every judged instance sits under the 20-point floor; only three format instances clear the 0.60 code floor.* Left: judged designed-effect delta (augmented − plain, 0–100) per instance, SE bars, grey per-query paired deltas (n≈150/instance). Right: code pass rates, augmented vs plain (95% CI, n=150). Floors dashed.

Family retention lands at fact 0/4, format 3/4, instruction 0/4, persona 0/4, so the three-failing-families kill stopped the run before checkpoint selection and every map-level fit. The deltas are mostly real but small: 10 of 11 judged instances beat zero at p < 0.01, the vegetarian fact largest at +15.9 of 100.

The exception, agree-with-the-user (p = 0.44), is noise-limited rather than effect-confirmed — its plain baseline of 16.6 compresses the available delta and its reliability ceiling is 0.00. JSON-only misses the 0.60 code floor at 0.500.

### Context distillation reaches the in-context compliance level on held-out probe rows in all 8 runs

Each panel: judged or code compliance (0–100) on one run's held-out probe rows (n=51–150) across its 8 checkpoints (0.5-epoch steps); dashed line = the in-context level on the same rows, dotted line = plain baseline, circled point = the dose-matched checkpoint.

![Dose-response trajectories for 8 distillation runs with the in-context level and plain baseline as reference lines](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cfb97f6acbbb817cd934f45d8fb958e8f8805479/figures/issue_1489/dose_trajectories.png)

> **Figure.** *All 8 distillation runs reach the in-context compliance level within the ±10-point tolerance.* Compliance (0–100) on held-out probe rows (n=51–150) over 8 checkpoints; dashed = in-context level, dotted = plain baseline, circled = dose-matched checkpoint (selection gaps 0.05–5.7). The pirate and JSON runs overshoot at later checkpoints.

In-weights delivery reaches the in-context compliance level within 4 epochs in every run tried — e.g. refuse-coding rises 12.1 → 19.4 versus 19.5 in-context, and JSON-only plateaus at 51–55 versus 50.7. The pirate and JSON runs overshoot the in-context level, so the in-weights dose can exceed the in-context dose.

This is compliance parity on probe rows only; whether the same per-example answer-state shift was delivered is the untested alignment question. Where the in-context effect is small, matching is trivially satisfiable (the agree run's reliability ceiling is 0.00; the other judged runs span 0.44–0.95). The dose reduce ran before the kill read — the declared plan deviation — and drives nothing downstream.

### The relevance instrument failed its agreement gate, and the fact family's high scores are echo, not use

Bars: fraction of 100 relevance probe pairs per scoped augmentation where the frozen topic→relevance rule matches the judged relevance label (95% CI); the 0.80 agreement gate is dashed.

![Rule versus judge agreement on relevance per scoped augmentation, 50 to 57 percent of 100 pairs each, all far under the 0.80 gate](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fe585a0d8646658f2acfd40d3b2c2437f2461bf0/figures/issue_1489/relevance_rule_agreement.png)

> **Figure.** *The topic rule agrees with the judged relevance label on only 50–57% of pairs, far under the 0.80 gate.* Per scoped augmentation: fraction of 100 (augmentation, query) probe pairs where the frozen rule matches the judge (95% CI); overall agreement 53% of 400 pairs, so the judged label replaced the rule.

The judged label replaced the rule (the plan's fallback). Almost no probe pairs are judged answer-changing — 0/100 vegetarian, 2/100 Tokyo, 13/100 Python, 25/100 refuse-coding. The judged label is itself unvalidated once the rule failed (nothing independent anchors it here), but the direction matches the corpus's topic composition: the fact subsets were mostly queries the fact cannot change.

Of the 14 vegetarian rows with fact-use at or above 90, all 7 relevance-scored ones have judged relevance 0.0 (the complete enumeration; row ids in the per-item score files), and the clearest case translates the injected fact string itself. The judged fact deltas are closer to a mention rate than a conditional-use rate: echo-inflated, and still under the floor.

### Teacher-forced margins shift toward fact-consistency on almost every row while on-policy fact use stays rare

Each dot is one probe row's paired margin delta (augmented − plain): fixed fact-consistent minus fact-inconsistent completion log-probability, length-normalized; n=200 rows per fact instance; diamonds mark the mean ± SE; the dotted line marks zero.

![Per-row teacher-forced margin deltas for the two fact augmentations, nearly all rows above zero](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fe585a0d8646658f2acfd40d3b2c2437f2461bf0/figures/issue_1489/margin_per_row_deltas.png)

> **Figure.** *Fact augmentations shift the teacher-forced margin on essentially every probe row.* Per-row paired margin deltas (augmented − plain), n=200 per instance, jittered strips with mean ± SE; 99.5% (vegetarian) and 100% (Python) of rows positive; p = 1.5e-34 and 1.4e-34, paired.

Mean deltas are +0.649 (SE 0.020) for the vegetarian fact and +1.070 (SE 0.019) for the Python fact: the augmentation shifts completion probabilities toward the fact-consistent answer on essentially every row, while the judged on-policy fact-use delta stays at or under +16. The fact reliably changes what the model assigns probability to; the model's own greedy answers rarely act on it.

Caveats: margins are quoted over all 200 drafted pairs (kept-pairs-only: +0.653 and +1.070, immaterial); part of the shift is an expected copy effect of the fact text in context; the finetuned models were never margin-scored; with only 2 cells the margin-versus-rate validation cannot be computed. This companion stays secondary, never the construct.

### All 16 augmentation texts surface in completions; lexical mention runs ahead of judged behavioral expression

Left: each instance's lexical cue-hit delta (augmented − plain fraction of completions containing its cue words; 2,000 paired rows, 95% bootstrap CI) against its manipulation-check delta (judged delta over the 0–100 scale, circles; code pass-rate delta, squares; n≈150 designed rows); log scales, dashed equal-deltas line. Right: per-cue augmented versus plain hit rates.

![Lexical screen deltas versus manipulation-check deltas per instance, with per-cue hit rates](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cc9f0ddcb685ae3ba0e53a451afff0943fcd9b58/figures/issue_1489/lexical_screen.png)

> **Figure.** *Every augmentation text surfaces lexically; most judged instances sit below the equal-deltas line.* Left: lexical cue-hit delta (2,000 paired rows, 95% bootstrap CI) versus manipulation-check delta per instance; all 16 positive, McNemar p < 0.001. Right: per-cue hit rates (n=2,000 per rate; Wald CIs under ±0.02, omitted).

All 16 instances shift positively (largest McNemar p = 2.6e-4): +0.020–0.244 across the lexically cued judged-family instances, +0.46–0.91 across the five code-validated ones. The library is not lexically inert — its text surfaces corpus-wide (topic-irrelevant rows shift nearly as much: vegetarian fact +0.196 versus +0.258 relevant) — yet judged behavior stays under floor: the pirate persona's cue-hit rate rises 0.072 → 0.316 while its judged delta is +6.2 of 100. Generic cue words (help, certain, knows) drive the high plain baselines; distinctive cues rise from near zero. This extends the facts' mention-versus-use gap to personas; agree-with-the-user stays weakest on every axis; the kill verdict stands.

### The map-level questions were not tested; the round delivers a measured manipulation-strength table and a reusable substrate

No figure: an inventory result with no per-unit decomposition. The transfer matrix, shift decomposition, relevance-gating linearity, per-example in-weights alignment, gating transfer, and post-finetuning map validity were not run — no finetuned-model eval and no fits or aggregation, per the kill's prescription (the two later fit gates were never reached).

What the round delivers: the manipulation calibration of a 16-instance augmentation library on real-corpus queries; the dose-ladder demonstration; and a complete substrate — 38,000 greedy generations, teacher-forced captures (7 summary kinds × 28 layers) for all 17 in-context cells, 64 distillation adapters, the conditions manifest, margin pools, and per-draw judge outputs — all verified on permanent storage, reusable by a library-v2 round without regenerating the plain-cell infrastructure.

---
**Repro:** ~25 GPU-h of 47 budgeted (GCE A100-80, 2–4 GPUs across attempts, instance eps-issue-1489; includes 3 crash-fix relaunches) · code SHA [`c5224efbab`](https://github.com/superkaiba/explore-persona-space/tree/c5224efbab11d9fb80fe71fa29d4d1ffba308eee) · aggregated eval summary: [g1_manipulation_check_summary.json](https://github.com/superkaiba/explore-persona-space/blob/fe585a0d8646658f2acfd40d3b2c2437f2461bf0/eval_results/issue_1489/g1_manipulation_check_summary.json) · judge-independent lexical screen (zero-GPU follow-up round): [lexical_screen.json](https://github.com/superkaiba/explore-persona-space/blob/da0ff4df8701e817b82ce043d8f148238e9473d5/eval_results/issue_1489/lexical_screen.json) + script [issue1489_lexical_screen.py](https://github.com/superkaiba/explore-persona-space/blob/da0ff4df8701e817b82ce043d8f148238e9473d5/scripts/issue1489_lexical_screen.py) · per-cell judge outputs (31 files: `manipulation_check.json`, `selection.json`, `margin_dv.json`, relevance files, per-instance raw scores): [judge/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ffd43f3c289e1d807d5790383d9fcc713ec2a30c/issue1489_ctx_aug/judge) · raw completions (17 generation cells + 64 dose-probe files): [raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ffd43f3c289e1d807d5790383d9fcc713ec2a30c/issue1489_ctx_aug/raw_completions) · captures: [analysis_tensors/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ffd43f3c289e1d807d5790383d9fcc713ec2a30c/issue1489_ctx_aug/analysis_tensors) · margin scores + pools: [margin/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ffd43f3c289e1d807d5790383d9fcc713ec2a30c/issue1489_ctx_aug/margin) and [conditions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ffd43f3c289e1d807d5790383d9fcc713ec2a30c/issue1489_ctx_aug/conditions) · training data (8 JSONLs): [distill_train/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ffd43f3c289e1d807d5790383d9fcc713ec2a30c/issue1489_ctx_aug/distill_train) · 64 adapter checkpoints (8 runs × 8): [issue1489_distill/](https://huggingface.co/superkaiba1/explore-persona-space/tree/983a001a48d1ac44cbac75a39162d802a4485c05/issue1489_distill) (path list in the run's first results card) · WandB: project issue1489, entity thomasjiralerspong, 8 runs (one per distillation slug) · figures: [figures/issue_1489](https://github.com/superkaiba/explore-persona-space/tree/cc9f0ddcb685ae3ba0e53a451afff0943fcd9b58/figures/issue_1489) · Reused corpus from [#1092](https://eps.superkaiba.com/tasks/1092): HF `issue1092_realistic_crossing/corpus/` at revision `e590170619e7691c1a95c7b1bb20bda5fd4065ad` (manifest fingerprint `e582a3b41ae9`) — fit: the same real-corpus substrate and schema the parent fit engine consumes · Reused role bank (persona-family system prompts): git-tracked `data/assistant_axis/` at the code SHA — fit: the standing 275-role bank, no new extraction.

**Context:** originating prompt (verbatim, from frontmatter):

> Help me to design an experiment to test this: \subsection{Effect of Adding Information to Context on This Mapping} Experiment: add different kinds of information to the context (facts, instructions, personas, formatting constraints) and see how the mapping changes. || can we compare against finetuning on that same example? || also what is the most principled way to compare 2 mappings? || please run this in the background with happy coder

Lineage: child of [#1092](https://eps.superkaiba.com/tasks/1092) (context→answer-state transport-map line). Created 2026-07-18 from user chat; provision A ran 2026-07-18 after 3 crash-fix relaunch rounds (rounds 4–6; the final relaunch completed cleanly at code SHA `c5224efbab`); the kill fired at the manipulation-check read on 2026-07-18; one zero-GPU free-analysis follow-up round (the judge-independent lexical screen, run over the existing generations) folded in 2026-07-19 UTC; no GPU follow-up rounds.

