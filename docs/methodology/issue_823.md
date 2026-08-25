# Methodology — issue 823: context→answer-profile map fits, the persona ladder, and the origin-effect conditioning extension


**Design:** four answer arms over the same 4998 LMSYS contexts (single-turn, ≤6000-token contexts, from the parent task's held-out LMSYS pool (provenance in the Context footer)): (1) **own answer (regenerated)** — Qwen-2.5-7B-Instruct resamples each context; (2) **external answer (plain style)** — claude-sonnet-4-5-20250929, no system prompt; (3) **external answer (distinct style)** — same Sonnet model with an eccentric-formatting system instruction (stripped before scoring, so the scored context is identical across arms); (4) **mismatched answer** — arm-1 answers reassigned by a fixed-point-free permutation (a shuffled-pairing null carrying real answer statistics). Per arm and per trait (evil / sycophancy / hallucination), the ridge map is REFIT from scratch at all 28 layers; TRANSFER additionally scores the arm-1-fitted map on the other arms' targets. A context dropped in any arm (API failure after 5 attempts) is dropped from all arms (5000 → 4998 common-valid). A zero-GPU follow-up round added the planned identity baseline: per-layer ridge from the own-answer profile v_A′(x) to each other arm's answer profile (input = the own-arm answer-span mean activation, not the context token), same solver, λ grid, CV folds, and common-valid mask as every phase-4 refit — targets: plain external (11-layer grid), mismatched (floor), distinct style (read-out layers only); `scripts/issue823_identity_baseline.py`, commit `c9e759b318`, run on the VM CPU.

**Design (persona-ladder follow-up round):** seven answer arms over the same LMSYS contexts — five persona-mixture arms in which claude-sonnet-4-5-20250929 answers every context in character as one of k fixed personas (k in {1, 2, 4, 8, 16}; nested assignment: context i's answer in arm k comes from persona i mod k, so arm k uses exactly the first k of 16 fixed persona cards and each persona answers approximately 5000/k contexts — 312 or 313 per persona at sixteen), plus the parent's own-answer and plain-external arms refit on the same new mask as anchors. 14,996 unique persona-context answers were generated (Anthropic Batch API, zero error rows; worst per-cell generation-cap-hit fraction 0.32%, under the 2% re-generation trigger, so no row was regenerated). A context whose persona answer the validity classifier flagged as a refusal is dropped from ALL arms: common-valid mask 4,629 of the parent's 4,998 (new drops per mixture arm 79 / 87 / 82 / 344 / 224, all refusal-class, zero integrity-class). The plan's per-persona-per-arm refusal-rate comparison against the pilot was not produced, so there is no read on whether refusal selection differed across personas — inference is limited to the 4,629-context population of contexts every arm answered. Answer-distinctness checks pass: mean within-context cross-persona tf-idf cosine 0.287 against the 0.8 bar, and all 15 non-baseline personas shift their answer-profile mean by more than 2× the noise floor at two or more of the three read-out layers (bar: at least 12 of 15). The plan's minimum-contexts n-ladder and its per-persona / matched-sample-size / same-contexts control battery were withheld under the solver-parity contingency (see Evaluation) and are reported nowhere in that round — the committed withheld-record carries zero fits; the extension round below supplies the minimum-contexts read. Six code-review concerns from this round's fix reviews are recorded in the concerns ledger. Five (prompt-cache fingerprint live-binding; the total-drop abort budget in the fits driver, reported twice; abort-payload key naming; post-load schema re-checks) carry evidence that none could have altered the reported cells — realized total new drops (369) sat under the 500 budget and the integrity class was zero everywhere. The sixth (per-persona refusal-rate reporting) is a standing disclosure, not a discharge: the missing comparison cannot change any fitted cell, but it bounds what the cells generalize to — the survivor population above.

**Design (origin-ladder extension round):** only the decisive contrast — the one-persona arm (persona 0 answers everything) and the sixteen-persona mixture (context i answered by persona i mod 16), same 16-card roster and prompt template — extended from 5,000 to 48,000 LMSYS contexts drawn from the parent pool's stream under a first-user-turn-disjoint selection rule; the banked 5,000 contexts are an ordered-equal prefix of the extension pool (reproduction check passed). 83,313 new persona answers were generated: 86,000 arm-rows minus the 2,687 rows shared between the arms where the mixture assignment lands on persona 0 (Anthropic Batch API; 200-pair pilot gate: 99.0% two-arm survival against the 85% floor, 0.5% generation-cap hits against the 2% trigger; production wave: max per-cell cap-hit 0.22%, 5 error rows of 83,313, no re-generation cell triggered). Refusal-classed drops: 2.1% of one-persona rows and 4.9% of sixteen-persona rows — the persona at roster index 6 again concentrates refusals (43.4% of its rows vs 1.3–3.4% for the other fifteen), replicating the parent round's concentration; integrity-class invalid rows were 1 and 4 of 43,000 per arm (mask-integrity gate PASS). 43,000 own-model answers were regenerated solely to set the answer-span truncation lengths (parity with the parent convention; they enter no fit, judge, or figure); captures are teacher-forced answer-span means at all 28 layers (126,313 forward passes). Fits: stream-prefix ladder on nested-prefix rung masks of 4,761 / 11,364 / 22,730 / 45,458 contexts (rungs are dependent prefixes of one stream, not independent replicates), both arms refit from scratch per rung; a randomized-subset companion ladder fits rungs of 4,761 / 11,364 / 22,730 / 43,987 contexts on seeded, era-stratified random subsets of the top-rung mask and scores a fixed 1,471-context shared-persona holdout — matching the stream-prefix sizes at the three lower rungs, while its top rung is the full 45,458-context mask minus that holdout — decoupling training composition from training size. The paired read reuses the banked inline analysis script unchanged; the banked-continuity gate reproduced that analysis's ratios at relative tolerance 1e-9 (12 arm-layer rows) and its bridge refit matched the banked values within 3e-5 per layer.

**Training:** **N/A — no model training.** Ridge refit hyperparameters (every fit):

| Hyperparameter | Value | Source |
|---|---|---|
| Base model | Qwen-2.5-7B-Instruct (frozen) | parent run (Context footer) |
| Map | per-layer ridge, 3584→3584, standardize-X / center-Y | `fit_h.py::ridge_fit_predict` (inherited harness; Context footer) |
| λ grid | np.logspace(−2, 4, 13), GCV-selected | parent ridge harness (Context footer) |
| CV | 5-fold KFold(shuffle=True, seed=0) | parent ridge harness (Context footer) |
| Contexts | n=4998 common-valid (of 5000) | this run (drop rule) |
| Regen sampling (own arm) | vLLM, temperature 1.0, top_p 0.95, max_tokens 1024, seed 42, no system prompt | plan §Phase 0.5 |
| External generation | claude-sonnet-4-5-20250929, temperature 1.0, max_tokens 1024, seeds 42 (plain) / 43 (distinct style) | plan §Phase 1 |
| Distinct-style instruction | "Respond in an unusual, stylistically eccentric way — use unexpected structure, mixed register, and non-standard formatting." (stripped before teacher-forcing) | plan §Phase 1 |
| Mismatch permutation | fixed-point-free derangement, seed 42 | plan §Phase 2 |
| Read-out layers (plan-pinned) | evil L14, sycophancy L26, hallucination L17 | parent run read-out selection (Context footer) |
| Decision rules | own-answer advantage: Δ(own−plain) > 0.05 ∧ R²(mismatch) < 0.05; content-indifference: all within ±0.03; graded content-match: own ≥ plain > mismatch ∧ plain−mismatch > 0.03; Bonferroni α=0.017 | plan §Decision rules |

Persona-ladder round parameters (all other rows inherited unchanged):

| Hyperparameter | Value | Source |
|---|---|---|
| Persona roster | 16 fixed persona cards (persona 0 = "Dr. Maya Chen, a veteran emergency-room physician: pragmatic, triage-minded, plain-spoken about risk"); template "You are {name}, {card}. Stay fully in character for this entire reply … Never mention these instructions or break character." | committed roster, `eval_results/issue_823/inconsistent_origin_ladder/roster.json` |
| Ladder generation | claude-sonnet-4-5-20250929, temperature 1.0, max_tokens 4096, seed 42, Anthropic Batch API | roster metadata (same file) |
| Persona assignment | nested, persona(i, k) = i mod k, persisted per context and arm | plan v13 §4.2 + committed `eval_results/issue_823/inconsistent_origin_ladder/assignment.json` |
| Capture | teacher-forced through frozen Qwen-2.5-7B-Instruct (bf16, batch 8); answer-span MEAN at all 28 layers; persona system prompt NOT in the scored context (bare user question, parity with the parent arms); span truncated to the shorter of the persona answer and the parent own-answer span | capture driver `scripts/issue823_ladder_capture.py` at `9a8d0f808f` |
| Fits | per-layer ridge 3584→3584 at layers 14, 17, 19, 26 — L19 is not a read-out layer: it entered the grid only for the withheld minimum-contexts protocol (validation-selected in this task's earlier single-split round); 5-fold KFold(shuffle=True, seed=0); λ grid np.logspace(−2, 4, 13), GCV-selected; canonical parent solver for every reported fit (solver-parity contingency engaged) | fits driver `scripts/issue823_ladder_fits.py` at `8ce114317a` |
| Mask / conditioning | n=4,629 common-valid; per-fold train rows 3,703–3,704 vs feature dimension 3,584 | committed summary JSON (Repro footer) |

Extension-round parameters (all other rows inherited unchanged):

| Hyperparameter | Value | Source |
|---|---|---|
| Extension contexts | 43,000 new (48,000 total); stream-prefix nested rung masks 4,761 / 11,364 / 22,730 / 45,458 after drops; companion 4,761 / 11,364 / 22,730 / 43,987 after drops | committed generation digest + summary JSON (Repro footer) |
| Arms | one-persona (persona 0); sixteen-persona (nested i mod 16) | scope marker + plan v17 §4 |
| Generation | claude-sonnet-4-5-20250929, temperature 1.0, max_tokens 4096 (8192 on any re-generation), seed 42, Anthropic Batch API | generation digest metadata |
| Estimator | ridge, GCV-selected λ under a 0.9 degrees-of-freedom cap; base λ grid np.logspace(−2, 4, 13); wide-grid sensitivity np.logspace(−2, 8, 21) | summary JSON estimator block |
| Solver | closed-form dual for masks ≤ 6,000 rows, primal above; parity-checked per rung against the canonical parent solver | summary JSON + solver-parity report |
| CV | 5-fold KFold(shuffle=True, seed=0); the split depends only on n, so both arms score each shared context in the same fold | paired-read metadata |
| Capture | teacher-forced, bf16; answer-span mean at all 28 layers; span = the shorter of the persona span and the own-answer span | capture digest |
| Paired read | mean paired out-of-fold error difference on shared-persona contexts; ρ = excess / E; full-ratio bootstrap, 10,000 draws, stratified within persona groups, seed 823 | paired-read metadata |
| Verdict bands | artifact ⇔ the bootstrap band lies wholly below 0.125; real ⇔ wholly above 0.5; boundary-crossing bands count as neither | plan v17 §3 |

**Evaluation:** DV = pooled 5-fold out-of-fold R² of predicted vs actual answer-span mean activation (per layer, per trait, per arm); companion estimand = equal-weighted per-context R² (bootstrap 10k resamples for CIs; fold-level p-values over the 5 folds, secondary). The two estimands weight contexts differently and are never mixed in one comparison. Alignment gate before any fit: teacher-forced re-extraction of own-arm activations matched the parent run's stored activations — workload log line ~813: "Alignment gate PASS: all 20 spot checks cosine > 0.999". Reproduce gate: refitting on the parent run's original bundle answers reproduced its R² 0.5991 / 0.6058 / 0.6262 (|Δ| ≤ 0.0015). Persona-ladder round: same pooled out-of-fold R² DV; every fitted cell additionally reports the identity+learned-bias baseline (v̂ = x + b) and top-1 retrieval accuracy — the fraction of held-out contexts whose true answer profile is the nearest neighbour of the prediction (cosine and euclidean; chance ≈ 0.0011 at pool size ≈ 926) — per the standing mapping-metrics rule. The single-persona minus sixteen-persona contrast carries a bootstrap 95% CI (10,000 draws, seed 0). Round gates: refitting the two anchors on the parent 4,998-context mask reproduced the promoted values to |Δ| ≤ 1e-7 in all six cells; the solver-parity gate comparing the fast-path solver against the canonical one FAILED on cuda AND on CPU-float64 (out-of-fold R² deviation up to 3.6e-4 against the 1e-4 tolerance, both solvers selecting λ = 0.01 on every probed slice), engaging the plan's contingency — every reported fit in that round uses the canonical solver, and the fast-path-dependent minimum-contexts battery was withheld. Estimator-diagnostics gap, carried explicitly for that round: the canonical-contingency solver did not persist per-fold selected λ / degrees of freedom (the only λ evidence is the six parity slices, all at the grid's bottom edge, 0.01), and the degrees-of-freedom-cap sensitivity read did not run although the cap was bindable at the realized mask — the anchor R² reads sit next to the estimator-degenerate regime and should be read with that in mind. Extension round: the decision DV is the per-rung paired shared-persona read (ρ with its full-ratio bootstrap band as the primary; the numerator-only band is the labeled secondary). Its gates, all from committed reports: solver parity passed on all 48 production slices plus a three-way dual/primal/canonical probe (max relative deviation 2.2e-13 across the slices and below 4e-14 within the probe, λ agreement on every slice) — resolving the ladder round's failed parity gate; per-fit λ and degrees of freedom are persisted for every fold-fit (median selected λ 3,162 at every rung; the 0.9 degrees-of-freedom cap never bound — median dof over training rows 0.05–0.17; 19–28% of fold-fits selected the base grid's top edge, the wide-grid re-fit moved none of the 240 probed cells — 30 per rung-ladder block — to the wide grid's edge, and the plan's rung-1 capped-vs-pure-GCV sensitivity read — primary ladder, all 30 fold-cells — selected the identical λ under both estimators, leaving max |ΔR²| at exactly 0) — resolving the ladder round's missing λ evidence; banked continuity and mask integrity PASS as in Design; the in-run fits pilot projected 3.97 h against the 6.0 h abort bound. Two further planned checks passed: the shared-persona slice is representative of the full rung masks — the one-persona arm's mean per-context error there is 0.940–0.978 of its full-mask value across stream-prefix rungs and read-out layers (the companion ladder's holdout is the shared slice itself, so its ratio is identically 1) — and the duplicate-context sensitivity refit was not required (duplicate fraction 0.78% of the 48,000-context pool, under the 2% trigger).

**Data extraction:** activations captured teacher-forced (one forward pass per context+answer) at all 28 layers; target v_s(x) = mean residual-stream activation over the answer span; predictor cx_last(x) = last-context-token activation. Answer-span truncation at max_model_len=8192 leaves 391 contexts (7.8%) with fewer than 10 plain-arm answer tokens; 28 plain + 75 distinct-style contexts have zero-span targets (kept, flagged — see per-context result). Answer-arm text statistics (full 5000-context means): own 320 tokens, plain 243, distinct-style 279; plain-vs-own text cosine (embedding) mean 0.489, 9.6% of pairs >0.8 — the arms are genuinely different texts, not paraphrases. Persona-ladder round: 16 per-persona tensor files (6.02 GB pair store) of answer-span mean activations; per-context residual and total sums of squares for every (arm × layer) cell committed at `eval_results/issue_823/inconsistent_origin_ladder/percontext_ladder.npz`. Extension round: an 83,313-row pair store (~51 GB, uploaded interleaved with capture; the sharded row index reconciled at 83,313 distinct rows on the full row-identity key); per-context out-of-fold residual and total sums of squares per rung and arm committed as the per-context arrays in the round's results directory (Repro footer).

**Sample training/evaluation data + completions:** one of 4998 common-valid contexts (context 0), one row per arm, truncated; full pool at [HF: issue823_own_vs_external/raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8039d15f30deb845765cbb24d9cdb8708a5e7b0f/issue823_own_vs_external/raw_completions).

Question (LMSYS): "how can identity protection services help protect me against identity theft"

Own answer (regenerated, 320-token mean arm): "Identity protection services can help safeguard you against identity theft in several ways. Here's a breakdown of how these services work and the benefits they offer:\n\n1. **Monitoring and Alerts**: - **Credit Reports and Scores**: Services often monitor you [...]"

External answer (plain style): "# How Identity Protection Services Help\n\nIdentity protection services offer several layers of defense:\n\n## **Monitoring & Alerts**\n- **Credit monitoring** - tracks changes to your credit reports\n- **Dark web surveillance** - scans for your personal info being [...]"

External answer (distinct style; instruction stripped before scoring):

````
# ※ The SHIELDS Around Your Digital Doppelgänger ※

```
[PROTECTIVE MEMBRANE ACTIVATED]
```

## 🔐 What These Services Actually *DO* (Beyond Marketing Promises)

**MONITORING** → They're like obsessive librarians who check if YOUR name appears in:
- Dark web ma [...]
````

Mismatched answer (shuffled pairing; permutation maps context 0 → context 1535's answer): context 1535's question is "hi", answer "Hello! How can I assist you today?" — a fluent own-model answer to a different context.

Persona-ladder answer (persona 0 "Dr. Maya Chen" — the one-persona arm's sole answer origin; same context 0; excerpt truncated — LMSYS-derived corpus, verify the full 242-word row at the pinned link): "Look, I've been in emergency medicine long enough to see the aftermath when someone's identity gets [...]" — in-character throughout. Cherry-picked: 1 of 14,996 records, chosen to match the parent worked example's context; all rows: [HF raw_completions/ladder](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/009b58fdcf3da303993695066870e29416fb9ef6/issue823_inconsistent_origin_ladder/raw_completions/ladder).

Origin-diversity examples across the larger persona-mixture arms — for each arm, the lowest-index common-valid context answered by that arm's newest (highest-index) persona; all excerpts truncated (LMSYS-derived corpus), full rows at the pinned link above:

<details>
<summary>Two-persona arm (k=2) — persona 1 "Frank Delgado" (retired homicide detective), context 1</summary>

Cherry-picked: 1 of 14,996 records — lowest-index common-valid context assigned to persona 1 in the two-persona arm; all rows: [HF raw_completions/ladder](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/009b58fdcf3da303993695066870e29416fb9ef6/issue823_inconsistent_origin_ladder/raw_completions/ladder).

Question (LMSYS, excerpt): "Beside OFAC's selective sanction that target the listed individiuals and entities, please elaborate [...]"

Answer (excerpt of a 588-word row): "Look, I spent thirty years chasing murderers, not paper trails, but I've seen enough financial crimes investigations [...]"

</details>

<details>
<summary>Four-persona arm (k=4) — persona 3 "Jax Torres" (touring stand-up comedian), context 3</summary>

Cherry-picked: 1 of 14,996 records — lowest-index common-valid context assigned to persona 3 in the four-persona arm; all rows: [HF raw_completions/ladder](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/009b58fdcf3da303993695066870e29416fb9ef6/issue823_inconsistent_origin_ladder/raw_completions/ladder).

Question (LMSYS, excerpt): "The sum of the perimeters of three equal squares is 36 cm. Find the area and perimeter of [...]"

Answer (excerpt of a 216-word row): "Alright, alright, so we got three equal squares here, and their perimeters add up to 36 cm. Look, I'm a comedian, not a mathematician [...]"

</details>

<details>
<summary>Eight-persona arm (k=8) — persona 7 "Tony Bocelli" (sports-radio commentator), context 7</summary>

Cherry-picked: 1 of 14,996 records — lowest-index common-valid context assigned to persona 7 in the eight-persona arm; all rows: [HF raw_completions/ladder](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/009b58fdcf3da303993695066870e29416fb9ef6/issue823_inconsistent_origin_ladder/raw_completions/ladder).

Question (LMSYS, excerpt; a Russian-language row): "Определи важнейшие смыслы в тексте ниже. [...]"

Answer (excerpt of a 136-word row): "HEY HEY HEY, FOLKS! Tony Bocelli here, and you're throwing me a CURVEBALL with the Russian text [...]"

</details>

<details>
<summary>Sixteen-persona arm (k=16) — persona 15 "Dusty McCall" (Texas ranch hand), context 15</summary>

Cherry-picked: 1 of 14,996 records — lowest-index common-valid context assigned to persona 15 in the sixteen-persona arm; all rows: [HF raw_completions/ladder](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/009b58fdcf3da303993695066870e29416fb9ef6/issue823_inconsistent_origin_ladder/raw_completions/ladder).

Question (LMSYS, complete): "buenos días"

Answer (excerpt of a 104-word row): "Well howdy there, partner! Buenos días to you too! [...]"

</details>

<details>
<summary>Extension-round answer — persona 0 "Dr. Maya Chen", context 6146 (one-persona arm)</summary>

Cherry-picked: 1 of 5 seed-42-sampled rows from the first persona-0 extension shard, chosen for a clearly in-character reply; all 83,313 extension rows: [HF raw_completions/ladder_ext](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ba2e24415e76e0dfcffb2cff60ba7e2847671227/issue823_inconsistent_origin_ladder/raw_completions/ladder_ext).

Question (LMSYS, excerpt): "How do I tame NAME_1 in ARK: Survival Evolved game?"

Answer (excerpt of a 266-word row; truncated — LMSYS-derived corpus, verify the full row at the pinned link): "Look, I spend my days dealing with actual survival situations - gunshot wounds, cardiac arrests, [...]" — in character throughout.

</details>


*Derived from the [task body](https://eps.superkaiba.com/tasks/823).*
