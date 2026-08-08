# Methodology — issue 2202

**Design:** one zero-GPU analysis pass over the banked context-arm layer-19 ridge map: full-pool retrieval ranks and confusion geometry in five similarity conventions, a two-stage failure-mode wave (Fable 5 proposes, Sonnet 4.5 counts), and a confusion-graph symmetry read against two nulls. Controls: the four-draw resample retrievability control, a matched non-failure control (cell-matched on depth band × corpus × language), the identity-plus-learned-bias baseline, degree-preserving (stub, plus a round-2 collision-free swap rebuild) and distance-only reciprocity nulls, and pool sizes 500 / 2,000 / 9,941. Three analysis rounds (geometry + judged waves; a collision-free null rebuild; a zero-GPU hub-correction follow-up); no new training, generation, or capture.

**Training:** **N/A — no model training.**

**Evaluation:** for held-out context *i*, the prediction is the banked ridge output; the candidate pool is all 9,941 held-out true answer vectors; the rank is the mid-rank of the true answer by distance to the prediction (ties mid-ranked). A rank-1 failure is a rank above 1 under raw euclidean — the banked headline convention. A reproduction gate re-derived the banked retrieval numbers from the banked tensors before anything downstream ran: rank-1 accuracy 0.816014485464239 euclidean (delta exactly 0; mean-reciprocal-rank delta 1.5e-7) and 0.8281862991650739 cosine (delta 0; two knife-edge tie rows at rank 5, within the two-row tolerance); chance is 1/9,941 ≈ 0.0001. The identity-plus-learned-bias baseline re-derived to rank-1 accuracy 0.473 euclidean / 0.512 cosine with held-out R² −1.08 (matches banked) — the ridge map's 0.816 / 0.828 sits far above both. Composition statistics use the parent battery: group-vs-rest failure-rate delta per label contrast, 10,000-draw bootstrap confidence bands, 10,000 permutations, false-discovery correction at q = 0.05; the judged failure modes form a second, separate correction family (never joined to the banked family). Analysis constants:

| Constant | Value | Source |
|---|---|---|
| Failure definition | mid-rank of true answer above 1, raw euclidean | parent banked headline (`eval_results/issue_1738/mapping_baselines.json`) |
| Similarity conventions | raw euclidean; raw cosine; mean-centered cosine (per-family train means); whitened euclidean and whitened cosine (shrunk train-answer covariance, λ = 0.1) | task-body lock; `analysis/null_battery.PRIMARY_LAMBDA` |
| Worst tails | top 200 by rank and top 200 by prediction-to-answer distance | task-body lock |
| Random sample | 500 contexts, seed 2202 | task-body lock |
| Resample control | 1,988 contexts × 4 extra on-policy answer draws; map-attributable at s ≥ 0.75, irreducible at s ≤ 0.25, ambiguous at 0.5 (s = fraction of draws retrieved at rank 1) | parent banked artifact; plan §11 operationalization |
| Composition battery | 10,000 bootstrap draws; 10,000 permutations; q = 0.05; 22 banked contrasts | parent battery (`scripts/issue1482_analysis.py`) |
| Reciprocity nulls | degree-preserving target-stub permutation, 1,000 draws (keeps colliding duplicate edges, 22.5% per draw — superseded in round 2); collision-free sensitivity rebuild: Maslov–Sneppen double-edge swaps from the observed graph (exact degrees, no duplicates), burn-in 5×E accepted swaps, one sample per E/2 accepted swaps, 200 samples, acceptance 0.135, lag-1 autocorrelation of samples 0.11; distance-only kernel, 1,000 draws per τ, τ at the 1st / 5th / 25th percentile of pairwise answer distances (headline at the 5th) | task-body lock; plan §11 sensitivity sweep; round-2 critique |
| Confuser display cap | top 10 per failure row | task-body lock |
| Hub-corrected retrieval (round 3) | CSLS, K = 10; cosine-native scores; retrieval by negative CSLS score, ties mid-ranked; baseline legs reconciled to the banked values within the reproduction-gate tolerances | banked metric battery (arXiv 1710.04087 convention); `scripts/issue2202_csls_followup.py` |
| Judge | `claude-sonnet-4-5-20250929`, Batch API, `max_tokens` 2048, temperature API default (1.0), 1 draw per item | project judge rule; parent instrument settings |
| Judge quality control | 150-item pilot (0 truncation stops, 0 parse failures, PASS); 200-item test-retest; modes with κ below 0.6 demoted to report-only | plan gate; parent κ convention |
| Reproduction-gate tolerances | rank-k accuracy within 2e-4; mean reciprocal rank within 1e-4; n exactly 9,941 | parent banked values |

Fable 5 (`claude-fable-5`) generated failure-mode hypotheses only; every countable label is a Sonnet call. Sonnet labeled 4,145 items — all 1,829 failures, 1,816 matched controls (cells capped at available non-failures), and the 500-sample — of which 4,137 returned valid labels (2 content drops, 6 other API errors; zero transport losses persisted).

**Data extraction:** all inputs are banked artifacts of the parent run, reproduced here for self-containment. The corpus is 100,000 multi-turn contexts drawn from LMSYS-Chat-1M and WildChat-1M — real user conversations (tier-1 realism). Each context was rendered in the Qwen-2.5-7B-Instruct chat template and answered once on-policy (sampled decoding, 7,104-token generation budget). Residual-stream states were captured at layer 19 in a teacher-forced forward pass: the context vector is the last prompt-token state (the newline before the assistant turn); the answer vector is the mean over answer-token states. A ridge map from context vector to answer vector (dimension 3,584) was fit on 88,378 training rows over 23 log-spaced penalties (1e-3 to 1e8), the penalty selected on pinned validation rows; this run consumes its 9,941 pinned held-out predictions. The held-out pool realized 9,941 of 10,000 pinned rows — 59 rows skipped by the parent's over-length capture filter (651 of 99,778 corpus-wide). Every held-out row carries banked judge labels (language, topic, format, refusal adjacency, answer-is-refusal; test-retest κ 0.79–0.98; 9,925 of 9,941 labeled — the 16 unlabeled rows are excluded from label masks). The resample control adds 4 extra on-policy answer draws for 1,988 held-out contexts (stratified over depth band × language × corpus). Raw conversation text never enters committed JSONs; text-bearing rows live on the HF data repo and in the two dashboards.

The mode-discovery leg partially failed: of the 10 Fable digest calls (5 chunks over the worst-200 + stratified-300 failure digest, 5 over the 500-sample digest), 7 returned empty replies — all 5 failure-digest chunks and 2 sample chunks — so all 13 raw mode proposals (consolidated to 9 canonical modes) trace to 3 sample-digest chunks. Mode discovery therefore never saw a failure-only digest; the Sonnet-measured rates over the full sets are unaffected. Retrying the empty chunks is flagged as a follow-up.

Acknowledged WARNs: the total-prose budget (800 words) is exceeded and several per-result blocks exceed the 120-word tier (eight deliverables are reported); one or more Takeaways bullets may exceed the 30-word bullet tier; 9 of 13 embedded figures are driver-generated without sidecars (four carry sidecars from regeneration; the rest are acknowledged as-is); the pool-robustness figure's direct line labels are the banked label-field identifiers defined above, acknowledged as rendered.

**Sample training/evaluation data + completions:** Disclosure: 3 of 1,829 rank-1 failure rows, random sample (seed 42); conversation text is excerpted to ~12 words for context hygiene (real-world-corpus rows) — full text at the pinned [dashboard rows on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ab268958343380945354e871bfb5666668c6d5bb/issue2202_ctxfail) and in the failure dashboard (footer).

<details>
<summary>3 sampled rank-1 failures (sanitized excerpts)</summary>

| row | final user turn (excerpt) | model answer (excerpt) | rank | top confuser (excerpt) | context / answer rank of confuser |
|---|---|---|---|---|---|
| ci 67690 | "Write me Easy Diffusion prompts for …" [truncated — real-world-corpus row] | "Certainly! Here are some easy diffusion prompts for a …" [truncated] | 10 | fan-fiction battle aftermath, ci 72042: "In the aftermath of a fierce battle, the characters from both Warcraft …" [truncated] | 33 / 77 |
| ci 11905 | "Complete this: I want to learn more about the American culture by" | "I want to learn more about the American culture by immersing myself …" [truncated] | 7 | transliteration explainer, ci 71880: "Got it! Lozung is a transliteration of the word Slogan …" [truncated] | 1,575 / 22 |
| ci 2968 | "puedes darme un plan de viaje a El Calafate, de 6 dias …" [truncated] | "¡Claro! El Calafate es un destino maravilloso en la provincia de Santa …" [truncated] | 4 | Portuguese Swiss-Alps itinerary, ci 30290: "Claro! Vou criar um roteiro básico para uma viagem aos Alpes Suíços. …" [truncated] | 4 / 5 |

</details>

The Spanish travel-plan failure is the archetypal hub drag: its top confuser is a Portuguese travel itinerary whose context ranks 4th-nearest and whose answer ranks 5th-nearest (answer-to-answer cosine 0.887) — a same-genre neighbour, not a duplicate. Disclosure: 3 of 8,112 correctly-retrieved rows, random sample (seed 42), same excerpt policy; per-row metrics for every context: [percontext_ranks.csv](https://github.com/superkaiba/explore-persona-space/blob/cf60dfc33519962eeb4f15290c74c926d5c1560c/eval_results/issue_2202/percontext_ranks.csv).

<details>
<summary>3 sampled correct retrievals (sanitized excerpts)</summary>

| row | final user turn (excerpt) | model answer (excerpt) | rank | nearest-competitor answer similarity |
|---|---|---|---|---|
| ci 75368 | "What is the estimate in literature?" | "The emission factors for gasification of biomass waste are not as widely …" [truncated] | 1 | 0.903 |
| ci 28052 | "The chatbot should analyze the description and identify the primary topic or …" [truncated] | "The primary topic or subject matter of this bill is Social Issues, …" [truncated] | 1 | 0.856 |
| ci 25159 | "Faça poema Vitor dedicando para Nicole recém casados" | "Órfãos de corações, agora só um. Vitor e Nicole, já a morna …" [truncated] | 1 | 0.913 |

</details>

Disclosure: 1 of 4,137 judge-labeled items, cherry-picked for illustration (the ci 2968 failure above; not a random sample); complete labels: [labels.json](https://github.com/superkaiba/explore-persona-space/blob/cf60dfc33519962eeb4f15290c74c926d5c1560c/eval_results/issue_2202/judge_labels_2202/labels.json).

```
judge = claude-sonnet-4-5-20250929, multi-field yes/no rubric (one field per mode)
item f2968 -> distinctive_entity_anchoring: yes; non_english_or_marked_register: yes;
unique_artifact_in_response: no; interchangeable_boilerplate_response: no;
templated_genre_or_variant_request: no; terse_deictic_final_turn: no;
corrupted_or_code_switched_response: no; multiple_choice_option_echo: no;
answer_topic_drift_from_last_turn: no
```
