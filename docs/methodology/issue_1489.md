# Methodology — issue 1489: context-augmentation manipulation calibration + dose-matched context-distillation ladder (provision A + P5 judging; run concluded at the pre-registered K2 gate)


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

*Derived from the [task body](https://eps.superkaiba.com/tasks/1489).*
