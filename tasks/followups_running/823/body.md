---
title: 'The context→answer-profile activation map reads answer-content match, not
  self-generation: plain-style external answers retain 91–98% of refit R² while shuffled
  answers collapse it to ≈0 (MODERATE confidence)'
kind: experiment
tags:
- followup-manual
created_at: '2026-07-01T23:41:29Z'
has_clean_result: true
parent_id: 722
origin_prompt: "## Motivation\n- We showed that there is a mapping from context vector\
  \ to answer profile -> when the answer was generated from that context\n- It is\
  \ also interesting to see if that mapping holds for answers not generated from the\
  \ model\n    - i.e. is the model just predicting what its assistant will say\n \
  \       - or somehow predicting in general the activations\n    - what I'm trying\
  \ to get at I think is is the model really just predicting its internals or also\
  \ the external output\n        - if it can predict the answer profile for output\
  \ generated from another model then this indicates it's not really predicting the\
  \ external\n    - This is also interesting to characterize the finetuned mapping\
  \ M+ -> should we be measuring it on the generated responses? Currently we are measuring\
  \ it on the pre-finetuned responses\n## Methodology\n- Take pre-finetuned model\n\
  - Compute:\n    - context vector -> answer profile mapping for answers generated\
  \ from the pre-finetuned model\n    - context vector -> answer profile mapping for\
  \ answers generated from Claude 4.5 Sonnet (with weird behavior prompt so that answers\
  \ aren't too similar)\n- Use held-out evaluation context setup from 722\n- Measure\
  \ (at all layers):\n    - which mapping is a better predictor on the held-out contexts\n\
  \        - if own answer mapping is better -> model is actually predicting its output\n\
  \        - if they are both similar -> model is just predicting its internals\n\
  \    - can one mapping predict the other? --> almost definitely not but why not\
  \ check\n\n[Design decisions confirmed in chat 2026-07-01: three external arms —\
  \ Sonnet with context C + weird-style instruction, Sonnet with context C plain,\
  \ and same-answer-text-across-contexts (bare-probe Sonnet); new child task of #722;\
  \ M+ re-measurement on finetuned-model generations out of scope (named follow-up);\
  \ no MLP training anywhere (ridge only); keep the Betley-pinned 48-probe pool (not\
  \ WildChat/UltraChat).]"
goal: 'Determine whether #779''s per-example context→answer-profile map h: c_last(x)
  → v(x) (held-out reconstruction R² 0.60–0.63, per-context cosine 0.93–0.96 over
  ~5000 LMSYS contexts) predicts the model''s own output content or merely context-side
  processing of arbitrary answer text, by refitting the identical 5-fold ridge harness
  on answer profiles computed from externally-generated (Claude Sonnet) and content-decoupled
  answers teacher-forced through the frozen base model.'
relates_to:
- spec-context-as-vector
- identity-contextual-vs-base
---
# The context→answer-profile activation map reads answer-content match, not self-generation: plain-style external answers retain 91–98% of refit R² while shuffled answers collapse it to ≈0 (MODERATE confidence)

<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_823.md](https://github.com/superkaiba/explore-persona-space/blob/326c6cc0369fbea328fc17046efb6aaaa42783f5/docs/methodology/issue_823.md) · [gist](https://gist.github.com/superkaiba/3c3c8e79cf89d7ae4925b22b1de92085)

## Takeaways

- The per-context ridge map from last-context-token activations to answer-span mean activations is NOT self-generation-specific: refit on Claude-Sonnet plain-style answers retains 97.6% (evil, R² 0.585 vs 0.599), 91.4% (sycophancy, 0.556 vs 0.608), and 94.4% (hallucination, 0.591 vs 0.626) of the own-answer refit R² at the read-out layers, while mismatched (shuffled-pairing) answers collapse the map to R² within 0.01 of zero (external-vs-mismatched gap 0.556–0.585 pooled; n=4998 contexts).
- The plain-arm retention needs no context-side information beyond content: a ridge map from the own-answer profile to the plain external profile reaches R² 0.671–0.688 at the read-out layers, above the context→plain refit 0.556–0.591; the own-answer increment is small (only sycophancy crosses the 0.05 threshold, gap 0.052, p_bonf=0.001), and each fitted map is style-specific — distinct-style answers retain 77–81% under refit yet get ≈0 transfer from the own-fitted map while plain-style answers transfer at 0.45–0.46.
- Follow-up persona ladder (Claude-written answers spread across one to sixteen personas over the same contexts): refit R² falls monotonically with persona count — 0.483–0.516 at one persona vs 0.281–0.366 at sixteen — by almost exactly the penalty a map that absorbs none of the between-persona target structure would pay (implied 0.171 vs observed 0.156 at layer 14; 0.192 vs 0.202 at layer 26), and retrieval falls in lockstep.
- The origin effect is real, not a near-interpolation artifact: growing the stream-prefix fit from 4,761 to 45,458 contexts (1.06–10.1 training rows per dimension) leaves the sixteen-persona map's excess held-out error at 57–77% of the between-persona energy — roughly 10× the constant-offset prediction of 6.25% and ≈2× the binding correlated-offset floor (0.28–0.34) — with every rung and read-out layer classifying real in both the stream-prefix and randomized-subset ladders; the stream-prefix layer mean falls 0.75 to 0.63 while the randomized-subset mean rises 0.61 to 0.66; the fixed-banked stream-prefix ratio rises to 0.75–0.85, identifying composition rather than offset absorption.
- In the ladder round, the own-answer and plain-external anchor refits on the shrunken 4,629-context mask collapse to pooled R² −8.4 to −10.8 yet still retrieve the correct answer profile at roughly 200× chance (top-1 accuracy 0.21–0.26): R² and retrieval dissociate under marginal conditioning (per-fold train rows 3,703–3,704 vs feature dimension 3,584); the extension round's parity-verified, non-degenerate fits at up to ten times more rows bound that collapse to the marginal regime.
- The minimum-contexts question, previously withheld, resolves to no threshold: one-persona held-out R² on a fixed 4,800-context holdout rises smoothly from 0.35–0.37 at 896 training rows to 0.49–0.53 at 42,213, crossing the 3,584-dimension boundary without a break — 82–84% of the largest-fit value is already reached at that boundary — and is still rising at the top size.

## Goal

**This experiment in context:** [#779](https://eps.superkaiba.com/tasks/779) showed a per-context ridge map h: cx_last(x) → v_s(x) predicts the answer-span mean activation from the last context token with R² ≈ 0.6, using the model's OWN generated answers — leaving open whether h encodes "what I will say" (self-generation) or "what the answer to this context looks like" (content processing). This experiment refits and transfers the same map on answers the model never generated: its own regenerated answers, Claude-Sonnet plain answers, Claude-Sonnet eccentric-style answers, and shuffled (mismatched) answers, on the held-out LMSYS context setup from [#722](https://eps.superkaiba.com/tasks/722). If the map works only on own answers → self-generation; if it works on any content-matched answer → content processing. A same-issue follow-up round then asked the inconsistent-origin question directly: does map quality degrade when the training answers come from many different personas instead of one consistent origin, and how many contexts does a well-posed map need? A second follow-up round extended that ladder from 5,000 to 48,000 contexts to test whether the mixed-origin penalty — measured on identical targets at 0.74–0.80 of the between-persona energy E in the banked round — is an artifact of fitting near the interpolation threshold: the ladder fits sat at 1.03 training rows per feature dimension, where a ridge can nearly fit its training targets including their persona offsets, so an artifact should fall toward the constant-offset prediction (0.0625 of E) as conditioning improves, while a real origin effect should not.

**Broader narrative:** whether pre-fine-tuning context geometry encodes a model-internal output prediction or a general context→content association determines what the fine-tuning-leakage predictor line can claim: a content-indexed map means base-side geometry measures context-content coupling available to ANY fine-tuning corpus, not a privileged readout of the model's own future behavior. The persona ladder sharpens this: the decline under maximally inconsistent answer origins is the full between-persona energy — a real per-example cost that survives well-conditioned fits and exceeds what any constant per-persona offset explains — so mixed-origin targets carry context-dependent persona structure the map cannot represent.

## Methodology

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

**Evaluation:** DV = pooled 5-fold out-of-fold R² of predicted vs actual answer-span mean activation (per layer, per trait, per arm); companion estimand = equal-weighted per-context R² (bootstrap 10k resamples for CIs; fold-level p-values over the 5 folds, secondary). The two estimands weight contexts differently and are never mixed in one comparison. Alignment gate before any fit: teacher-forced re-extraction of own-arm activations matched the parent run's stored activations — workload log line ~813: "Alignment gate PASS: all 20 spot checks cosine > 0.999". Reproduce gate: refitting on the parent run's original bundle answers reproduced its R² 0.5991 / 0.6058 / 0.6262 (|Δ| ≤ 0.0015). Persona-ladder round: same pooled out-of-fold R² DV; every fitted cell additionally reports the identity+learned-bias baseline (v̂ = x + b) and top-1 retrieval accuracy — the fraction of held-out contexts whose true answer profile is the nearest neighbour of the prediction (cosine and euclidean; chance ≈ 0.0011 at pool size ≈ 926) — per the standing mapping-metrics rule. The single-persona minus sixteen-persona contrast carries a bootstrap 95% CI (10,000 draws, seed 0). Round gates: refitting the two anchors on the parent 4,998-context mask reproduced the promoted values to |Δ| ≤ 1e-7 in all six cells; the solver-parity gate comparing the fast-path solver against the canonical one FAILED on cuda AND on CPU-float64 (out-of-fold R² deviation up to 3.6e-4 against the 1e-4 tolerance, both solvers selecting λ = 0.01 on every probed slice), engaging the plan's contingency — every reported fit in that round uses the canonical solver, and the fast-path-dependent minimum-contexts battery was withheld. Estimator-diagnostics gap, carried explicitly for that round: the canonical-contingency solver did not persist per-fold selected λ / degrees of freedom (the only λ evidence is the six parity slices, all at the grid's bottom edge, 0.01), and the degrees-of-freedom-cap sensitivity read did not run although the cap was bindable at the realized mask — the anchor R² reads sit next to the estimator-degenerate regime and should be read with that in mind. Extension round: the decision DV is the per-rung paired shared-persona read (ρ with its full-ratio bootstrap band as the primary; the numerator-only band is the labeled secondary). Its gates, all from committed reports: solver parity passed on all 48 production slices plus a three-way dual/primal/canonical probe (max relative deviation below 4e-14, λ agreement on every slice) — resolving the ladder round's failed parity gate; per-fit λ and degrees of freedom are persisted for every fold-fit (median selected λ 3,162 at every rung; the 0.9 degrees-of-freedom cap never bound — median dof over training rows 0.05–0.17; 19–28% of fold-fits selected the base grid's top edge, and the wide-grid re-fit moved none of the 240 probed cells — 30 per rung-ladder block — to the wide grid's edge) — resolving the ladder round's missing λ evidence; banked continuity and mask integrity PASS as in Design; the in-run fits pilot projected 3.97 h against the 6.0 h abort bound. Two further planned checks passed: the shared-persona slice is representative of the full rung masks — the one-persona arm's mean per-context error there is 0.940–0.978 of its full-mask value across stream-prefix rungs and read-out layers (the companion ladder's holdout is the shared slice itself, so its ratio is identically 1) — and the duplicate-context sensitivity refit was not required (duplicate fraction 0.78% of the 48,000-context pool, under the 2% trigger).

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

## Results

### Plain-style external answers retain 91–98% of own-answer refit R²; mismatched answers collapse to ≈0

Bars show pooled 5-fold out-of-fold refit R² at each trait's plan-pinned read-out layer, one bar per answer arm (own regenerated / external plain / external distinct-style / mismatched), error bars = fold SD.

![Bars of refit R-squared per answer arm at each read-out layer, own and plain arms near 0.6 and mismatched near zero](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e4bfe5c769ec36cedd3886bc5c018f6d2f473115/figures/issue_823/fig1_refit_r2_by_arm.png)

> **Figure.** *Refit R² by answer arm at read-out layers.* Own 0.599/0.608/0.626 (evil L14 / sycophancy L26 / hallucination L17); external plain 0.585/0.556/0.591 (97.6/91.4/94.4% retention); external distinct-style 0.473/0.468/0.506; mismatched 0.004/−0.008/0.007. Fold SDs ≤0.023 (mismatched ≤0.003). n=4998.

The map is content-indexed: an answer the model never produced supports nearly the full R²; a fluent-but-wrong answer supports none. Verdicts: sycophancy meets the own-answer-advantage rule (own−plain gap 0.052, above the 0.05 threshold; p_bonf=0.001); evil and hallucination show only the graded content-match ordering (gaps 0.014, p_bonf=0.36, and 0.035, p_bonf=0.014). The planned 5th arm (identity baseline) was computed in a follow-up round (result below); the planned shuffle-null was replaced by the mismatched arm — a stricter null with real answer statistics (parent shuffle floor R²≈0.12 vs ≈0.005 here), so the collapse claim is conservative.

### The own-vs-plain increment is layer-structured, peaking at 0.052 exactly at the sycophancy read-out layer; each fitted map is style-specific under transfer

Solid curves: refit R² across all 28 layers per arm; dashed curves: transfer R² of the own-answer-fitted map scored on the other arms' targets; read-out layers marked.

![Refit R-squared curves rising then falling across layers per arm, with own-map transfer to plain high and to mismatched deeply negative](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e4bfe5c769ec36cedd3886bc5c018f6d2f473115/figures/issue_823/fig2_per_layer_refit_transfer.png)

> **Figure.** *Per-layer refit (solid) and own-map transfer (dashed) R².* The own-minus-plain gap is U-shaped, peaking exactly at L26 (0.052; 0.001 at L27). Transfer to plain 0.451–0.461; to distinct-style −0.070..+0.050; to mismatched −0.65..−0.80.

The sycophancy own-advantage verdict depends on L26 being both the plan-pinned layer and the peak of the own-minus-plain gap — one layer later the increment vanishes, so "predicts own output over external" is a narrow-band effect. Transfer separates content from style: plain answers reuse the own-fitted map at 0.45+, distinct-style answers (77–81% retention under refit) get ≈0 transfer — style shifts the target subspace enough to force a refit. An alternative reading — that style rather than content carries part of the own-answer increment — survives: a length-matched sweep (length-difference cuts 10→200) moves the sycophancy gap across 0.048–0.053, straddling the 0.05 threshold.

### Per-context R² distributions confirm the arm ordering context-by-context (own medians 0.55–0.58, mismatched ≈0), not via a few outliers

ECDFs of equal-weighted per-context R² (n=4998) per arm at read-out layers; the table gives both estimands side by side.

![Per-context R-squared ECDF curves separated by arm, own rightmost and mismatched centered at zero](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e4bfe5c769ec36cedd3886bc5c018f6d2f473115/figures/issue_823/fig3_per_context_r2_ecdf.png)

> **Figure.** *Per-context R² ECDFs by arm.* Medians — own 0.55/0.55/0.58, plain 0.52/0.49/0.54, distinct-style 0.48/0.43/0.51, mismatched 0.01/−0.01/0.01 (evil/sycophancy/hallucination). Whole-distribution shifts, no outlier-driven mass.

| Trait (read-out) | Pooled Δ(own−plain) | Per-context Δ 95% CI | Pooled plain−mismatch | Per-context 95% CI |
|---|---|---|---|---|
| evil (L14) | 0.0142 | [0.0175, 0.0270] | 0.5807 | [0.5143, 0.5286] |
| sycophancy (L26) | 0.0521 | [0.0504, 0.0615] | 0.5641 | [0.5055, 0.5211] |
| hallucination (L17) | 0.0349 | [0.0404, 0.0506] | 0.5845 | [0.5144, 0.5308] |

Pooled (weighted by each context's share of the total sum of squares) and per-context (equal-weighted) estimands differ by construction — evil's pooled gap sitting below its per-context CI is an estimand difference, not an inconsistency. Caveats carried: the activation-cosine diagnostic (plain-vs-own target cosine mean 0.92–0.95) includes degenerate zero-span rows (min = 0.0 for all traits), so the mean is not ordinary target similarity for every context; per-context R² correlates weakly with answer length (Spearman 0.056–0.108, rescued inline — the planned length diagnostic was missing from the artifact); the planned identity baseline was computed in a follow-up round (result below).

### The own-answer profile predicts the plain external profile (R² 0.67–0.69) better than the context does (0.56–0.59): content overlap alone accounts for the plain-arm retention

Bars: pooled 5-fold out-of-fold R² at each read-out layer for the context→plain refit versus ridge maps from the own-answer profile to the plain, distinct-style, and mismatched profiles; curves: own-profile→plain versus the context→plain refit and the mismatched floor across the 11-layer grid (error bars = fold SD).

![Own-profile-to-plain bars above context-refit bars at every read-out layer, mismatched target bars at zero](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5b159ab9b214908979566800048cbc82feec9738/figures/issue_823/fig4_identity_baseline.png)

> **Figure.** *Identity baseline (follow-up round).* At read-out layers, own-profile→plain R² 0.686/0.671/0.688 (evil/sycophancy/hallucination) vs context→plain refit 0.585/0.556/0.591; own-profile→distinct-style 0.530/0.525/0.548; own-profile→mismatched −0.010..−0.002. Per layer, own-profile→plain (0.580–0.712) exceeds the context refit (0.394–0.654) at every grid layer. n=4998.

This planned identity baseline was computed in a zero-GPU follow-up round, reusing each refit's solver, folds, and context mask. A content-matched answer profile carries more information about the plain-arm profile than the context does: the own-answer profile predicts it at R² 0.671–0.688, above the context→plain refit (0.556–0.591; fold SDs ≤0.012 against gaps ≥0.08), while mismatched targets stay at −0.021..+0.002 grid-wide.

The retention headline therefore needs no context-side self-generation information beyond what the answer content carries — a decomposition of the retention, not a claim the context carries nothing. Style specificity is consistent: own-profile→distinct-style reaches only 0.525–0.548.

### Refit R² falls monotonically with persona count, by 0.15–0.20 from one persona to sixteen, across the whole per-context distribution

Pooled 5-fold held-out R² per persona-count arm at the three read-out layers, anchors on a broken axis; error bars = per-context bootstrap 95% CIs.

![Held-out R-squared versus persona count, declining at all three read-out layers](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0d6c044fb5a0319f59d8d7f4061c03c2345832d0/figures/issue_823/ladder_fig1_r2_vs_k.png)

> **Figure.** *Refit R² falls monotonically with persona count at every read-out layer.* One persona 0.501/0.483/0.516 (layers 14/26/17) down to sixteen personas 0.345/0.281/0.366; rank correlation −1.0 per layer. Anchor refits on this round's mask sit at −8.4 to −10.8 (dissociation result below). n=4,629.

| Pooled held-out R² | Layer 14 (evil) | Layer 26 (sycophancy) | Layer 17 (hallucination) |
|---|---|---|---|
| one persona (k=1) | 0.501 | 0.483 | 0.516 |
| two personas (k=2) | 0.445 | 0.462 | 0.471 |
| four personas (k=4) | 0.401 | 0.385 | 0.431 |
| eight personas (k=8) | 0.354 | 0.302 | 0.377 |
| sixteen personas (k=16) | 0.345 | 0.281 | 0.366 |
| drop, k=1 minus k=16 | 0.156 | 0.202 | 0.150 |
| drop, mean over the three layers (bootstrap 95% CI, 10,000 draws) | 0.169 [0.160, 0.179] | | |

The per-unit companion — ECDFs of equal-weighted per-context R² (clipped at −2) for the same arms plus the anchors, per read-out layer:

![Per-context R-squared ECDFs shifting toward lower values as persona count grows](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0d6c044fb5a0319f59d8d7f4061c03c2345832d0/figures/issue_823/ladder_fig3_percontext_ecdf.png)

> **Figure.** *Whole-distribution shifts, no outlier mass.* Per-context medians: one persona 0.45/0.43/0.48, sixteen personas 0.32/0.25/0.34 (layers 14/26/17); anchors −7.1 to −8.7 with 78–81% of contexts below −2. n=4,629.

The decline is monotone at every layer, the plan's decision rule labels it "degrades" (the answer-distinctness predicate passed), and retrieval falls in lockstep (top-1 accuracy 0.372 at one persona to 0.169 at sixteen, layer 14). The per-context curves shift together across the whole distribution (medians in the caption), so the ordering holds context-by-context, not through a few badly-fit contexts. This round covered the persona-count half of the question only — the minimum-contexts n-ladder with its per-persona, matched-sample-size, and same-contexts controls was withheld under the solver-parity contingency, its dimension-boundary rung unrealizable at the realized mask (train rows 3,336 < 3,584); the extension round below supplies that read (final result).

### The implied mechanical mixture penalty accounts for essentially all of the persona-count decline; the largest of twelve cell-level disagreements is 0.026

Top: the observed R² drop versus one persona at each persona count, against the penalty a fixed map mechanically incurs from between-persona target variance; bottom: pooled R² under own versus fixed one-persona denominators.

![Observed R-squared drop tracking the implied mixture penalty across persona counts](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0d6c044fb5a0319f59d8d7f4061c03c2345832d0/figures/issue_823/ladder_fig4_mixture_floor.png)

> **Figure.** *The observed drop tracks the implied mechanical penalty.* At sixteen personas: implied 0.171 vs observed 0.156 (layer 14), 0.192 vs 0.202 (layer 26), 0.160 vs 0.150 (layer 17). The fixed-denominator re-read falls more steeply (0.272 vs 0.345 at layer 14).

Mixing persona-specific answer distributions over the same contexts adds between-persona target variance no single deterministic map can capture. The implied penalty tracks the observed drop imperfectly but tightly: the largest absolute disagreement across the twelve arm-layer cells is 0.026 (two personas at layer 26, implied 0.047 vs observed 0.021), the other eleven cells agree within 0.015, and the observed drop never exceeds the implied penalty by more than 0.011. Where the two diverge most, the mechanical floor over-predicts — the fitted map loses less than mixing alone would force — strengthening the no-added-learning-difficulty reading.

So the measured degradation under inconsistent origins is essentially what mixed targets mechanically force, not evidence the context→content association weakens. The fixed-denominator re-read falling more steeply confirms part of the own-denominator curve's flattening is denominator inflation rather than better fit.

### New-mask anchor refits collapse in R² yet retrieve at roughly 200× chance: the two mandatory reads dissociate

Top: top-1 retrieval accuracy per arm (cosine and euclidean) against chance; bottom: identity+learned-bias baseline R² per arm; anchors as bands.

![Retrieval accuracy far above chance while anchor refit R-squared collapses](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0d6c044fb5a0319f59d8d7f4061c03c2345832d0/figures/issue_823/ladder_fig2_retrieval_identity.png)

> **Figure.** *Anchors keep far-above-chance retrieval while their refit R² is catastrophic.* Anchor top-1 accuracy 0.210–0.256 vs chance 0.0011; fitted ladder maps beat the identity+bias baseline by 1.3–4.0 R² units, while the anchor refits fall 7.8–9.5 units below their own baseline.

| Read (layers 14 / 26 / 17) | Own-answer anchor | Plain-external anchor |
|---|---|---|
| pooled refit R² | −10.00 / −10.83 / −8.95 | −10.04 / −10.70 / −8.45 |
| identity+bias baseline R² | −0.58 / −2.52 / −0.71 | −0.51 / −2.27 / −0.62 |
| top-1 retrieval accuracy, cosine (chance 0.0011) | 0.224 / 0.256 / 0.212 | 0.218 / 0.253 / 0.210 |
| contexts below per-context R² of −2 | 78.0% / 81.4% / 78.2% | 78.9% / 81.5% / 78.3% |

L19, fitted for all seven arms but not a read-out layer (carried only for the withheld minimum-contexts protocol), reports both mandatory reads:

| Arm (layer 19) | Pooled refit R² | Identity+bias baseline R² | Top-1 retrieval, cosine (chance 0.0011) |
|---|---|---|---|
| one persona (k=1) | 0.559 | −1.538 | 0.484 |
| two personas (k=2) | 0.505 | −1.628 | 0.391 |
| four personas (k=4) | 0.457 | −1.674 | 0.326 |
| eight personas (k=8) | 0.397 | −1.479 | 0.243 |
| sixteen personas (k=16) | 0.387 | −1.414 | 0.225 |
| own-answer anchor | −6.04 | −0.850 | 0.300 |
| plain-external anchor | −5.33 | −0.753 | 0.309 |

The per-unit companion for the baseline read (per-context identity+bias R² ECDFs):

![ECDFs of per-context identity-plus-bias R-squared per arm at the read-out layers](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c733bc78a39dd4521e04ae0d0e5d7b22a0e9935b/figures/issue_823/ladder_fig5_identity_percontext_ecdf.png)

> **Figure.** *The identity+bias baseline is negative for most contexts in every arm.* Anchor medians (−0.6 to −2.6 across read-out layers) sit at higher values than the mixture arms' (−1.1 to −3.8); computed from the committed per-context sums of squares. n=4,629.

Per-unit exemption: per-context retrieval ranks were not persisted (only fold-level accuracy, median-rank, and reciprocal-rank summaries).

Planned as ceilings, the anchor refits instead show why both reads are mandatory: R² alone would call these maps worthless, yet their predictions still identify the correct answer profile at roughly 200× chance. The parent-mask reproduce gate matched within |Δ| ≤ 1e-7, so the promoted parent numbers stand; the collapse is specific to this round's 4,629-row mask (per-fold train rows only 119 above the 3,584 feature dimension; λ evidence limited to six parity slices at the grid's bottom edge), conditioning-dependent estimator behaviour rather than a property of the maps.

### Well-conditioned refits leave the mixed-origin excess at 57–77% of the between-persona energy — roughly ten times the constant-offset prediction — so the origin effect is real, not a near-interpolation artifact

The extension round's decision read: ρ — the sixteen-persona map's mean extra held-out squared error on shared-persona contexts (identical persona-0 targets in both arms) as a fraction of the between-persona energy E (the persona-identity share of target variance) — against realized training rows per feature dimension (n/d), per read-out layer, both ladders.

![Excess-to-energy ratio versus conditioning, both ladders staying far above the offset line](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5679ffb426417bf6b85d2c1cd269ec1dd312f5ff/figures/issue_823/ladder_ext_fig1_excess_ratio_ladder.png)

> **Figure.** *The measured excess never approaches the offset prediction.* Solid: stream-prefix ladder (layer mean 0.75 at n/d 1.06 down to 0.63 at 10.1); dashed: randomized-subset companion (0.61 up to 0.66); bands: full-ratio bootstrap 95% intervals — every rung and layer classifies real (minimum lower bound 0.526). Gray: the banked round's historical regime. n=311–2,944 shared contexts per rung.

| Rung (stream-prefix ladder; contexts in mask) | n/d | Solver | Parity gate | Median selected λ | Median dof over train rows | ρ across read-out layers |
|---|---|---|---|---|---|---|
| 4,761 | 1.06 | dual | PASS | 3,162 | 0.17 | 0.712–0.765 |
| 11,364 | 2.54 | primal | PASS | 3,162 | 0.10 | 0.685–0.725 |
| 22,730 | 5.07 | primal | PASS | 3,162 | 0.07 | 0.629–0.668 |
| 45,458 | 10.15 | primal | PASS | 3,162 | 0.05 | 0.619–0.655 |

| Mandatory map reads, extension fits (stream-prefix ladder; ranges over rungs × read-out layers) | one-persona arm | sixteen-persona arm |
|---|---|---|
| pooled out-of-fold R² | 0.484–0.538 | 0.281–0.372 |
| identity+learned-bias baseline R² | −3.55 to −1.07 | −3.52 to −0.97 |
| top-1 retrieval, cosine (chance 0.0011 at the smallest rung to 0.00011 at the largest) | 0.259–0.399 | 0.100–0.170 |
| top-1 retrieval, euclidean (same chance) | 0.245–0.394 | 0.088–0.157 |

The per-unit companion:

![Per-rung ECDFs of per-context paired error differences, both ladders, most mass above zero](https://raw.githubusercontent.com/superkaiba/explore-persona-space/85cf46170452071bb5e2d00cb6c684a13d1c23fc/figures/issue_823/ladder_ext_fig8_paired_diff_ecdf.png)

> **Figure.** *The excess holds context-by-context in both ladders.* ECDFs of the per-context paired difference (sixteen-persona minus one-persona held-out squared error, identical targets): 85–91% of stream-prefix shared contexts (solid) and 86–93% of randomized-subset contexts (dashed) are individually worse under the mixed-origin map at every rung and read-out layer.

All 24 rung-layer verdicts classify real; the sixteen-persona map's error on identical targets stays 22–35% above the one-persona map's. For the stream-prefix ladder, E moves −2.9% to +4.7% while mean excess falls 9–21%; its fixed-banked ratio rises to 0.75–0.85, so the decline reflects composition, not offset absorption.

Correlated persona offsets alone would cost 0.28–0.34 of E, so the 0.125 artifact band could never have fired; the measured ratio is roughly twice that floor — a beyond-offset, context-dependent effect. Layer names are nominal; rungs are nested prefixes, not independent replicates; scope: one model, one context pool, one assignment rule.

### Map quality rises smoothly through the dimension boundary — no minimum-contexts cliff, and 82–84% of the 42,213-row R² is already reached at n = d

Held-out R² of the one-persona map on a fixed 4,800-context holdout versus training-set size (896 to 42,213 rows, 5 random draws per size, mean shown), per read-out layer, with the feature dimension d = 3,584 and the previously unrealizable 3,336-row point marked; the identity+learned-bias baseline overlaid.

![Held-out R-squared rising smoothly with training rows through the dimension boundary at every read-out layer, identity baseline flat and negative](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5679ffb426417bf6b85d2c1cd269ec1dd312f5ff/figures/issue_823/ladder_ext_fig5_p2_boundary.png)

> **Figure.** *No break at n = d.* R² climbs from 0.35–0.37 at 896 rows through 0.41–0.43 at the once-withheld 3,336-row point to 0.49–0.53 at 42,213, still rising at the top; the identity+bias baseline sits at −1.3 to −3.6 throughout. Per-size five-draw spread ≤ 0.04 (per-draw values in the committed battery JSON).

The minimum-contexts half of the follow-up ask, withheld in the ladder round, resolves to a smooth data curve rather than a threshold: under the degrees-of-freedom-capped estimator even 896 rows (a quarter of the dimension) support R² 0.35–0.37, the dimension boundary leaves no visible break, and returns continue through 42,213 rows with no plateau. Retrieval confirms the fits are genuine maps at every size (cosine top-1 accuracy 0.34–0.39 at the largest size against chance 0.0002). The fixed holdout is the deepest slice of the context stream, so absolute levels carry a stream-drift caveat.

---

**Repro:** GCP a2-ultragpu-1g (1× A100-80), ~8 GPU-h (phases 0.5–3) + CPU phase 4 (ridge fits; dedupe 3780→700 fits, canonical solver — a Gram-eigh fast path FAILED full-size parity and was reverted). Code [`d68455ce60`](https://github.com/superkaiba/explore-persona-space/commit/d68455ce60e5a80ad949a6f66cab59e3d67ec00e) (branch issue-823); figures 1–3 on main @ [`e4bfe5c769`](https://github.com/superkaiba/explore-persona-space/commit/e4bfe5c769ec36cedd3886bc5c018f6d2f473115), figure 4 @ [`5b159ab9b2`](https://github.com/superkaiba/explore-persona-space/commit/5b159ab9b214908979566800048cbc82feec9738). Identity-baseline follow-up (0 GPU-h, VM CPU): `eval_results/issue_823/identity_baseline.json` + `identity_baseline_units.jsonl` (branch issue-823), script `scripts/issue823_identity_baseline.py`, code [`c9e759b318`](https://github.com/superkaiba/explore-persona-space/commit/c9e759b318508bb2cfb6584715026273a50384bc) — the JSON's embedded `git_commit` records the parent [`397f879b5d`](https://github.com/superkaiba/explore-persona-space/commit/397f879b5d898e448c2daf1f92c1f5ebc7e11677) (the script was committed after the run); `c9e759b318` is the authoritative code commit. Data: `eval_results/issue_823/` (git: `ridge_r2_by_arm.json`, `validity_diagnostics.json`); raw completions + tensors + workload log: [HF issue823_own_vs_external/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8039d15f30deb845765cbb24d9cdb8708a5e7b0f/issue823_own_vs_external) (`raw_completions/`, `analysis_tensors/`, `logs/issue-823-workload.log` — alignment-gate PASS at line ~813, reproduce-gate PASS at lines 4143–4145 after two retried pre-stage RuntimeErrors at 4114/4126). Persona-ladder follow-up round (2026-08-20): generation via the Anthropic Batch API (14,996 records, 0 GPU); capture + fits on RunPod `pod-823-ladder` (1× H100, ~3 h wall including one designed abort + relaunch; measured fit cost 89 s per layer-fold unit, 20 units). Round code: generation + capture [`9a8d0f808f`](https://github.com/superkaiba/explore-persona-space/commit/9a8d0f808f73bac40b9d446be9bca51e3fcda83a), fits (post-fix) [`8ce114317a`](https://github.com/superkaiba/explore-persona-space/commit/c578b437c6b4ee69335798bb1e73c54b577dfd2e), figures + aggregation [`8762977adc`](https://github.com/superkaiba/explore-persona-space/commit/a121ec5593a917b301719cd88a368b4342532449); round artifacts committed at [`16f0ebeed1`](https://github.com/superkaiba/explore-persona-space/commit/0d6c044fb5a0319f59d8d7f4061c03c2345832d0) (branch issue-823): `eval_results/issue_823/inconsistent_origin_ladder/` (aggregated summary `ladder_analysis_summary.json`, per-cell fits, baselines, distinctness checks, the zero-fit withheld-record for the minimum-contexts battery, per-context arrays) and ladder figures 1–4 under `figures/issue_823/`; the round-8 revision's identity per-context companion (ladder figure 5, rendered by `scripts/issue823_ladder_figures.py --only-fig5` from the committed per-context arrays) committed at [`facab163ef`](https://github.com/superkaiba/explore-persona-space/commit/c733bc78a39dd4521e04ae0d0e5d7b22a0e9935b). Round data on HF (pinned): [issue823_inconsistent_origin_ladder/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/009b58fdcf3da303993695066870e29416fb9ef6/issue823_inconsistent_origin_ladder) — `raw_completions/ladder` (19 files: 16 per-persona answer files + assignment + roster + completion sentinel), `analysis_tensors` (18 files, 6.02 GB pair store), `logs/fits` (7 files), `logs/fits_resume_state` (per-layer fit checkpoints + producer sidecars). Banked paired re-read (0 GPU-h inline analysis on the ladder round's committed per-context arrays): `eval_results/issue_823/inconsistent_origin_ladder/shared_persona_paired.json` on main @ [`84633d46c6`](https://github.com/superkaiba/explore-persona-space/commit/84633d46c6cd23dcd75be9ffc9b0f7815822f7ce), script `scripts/issue823_shared_persona_paired.py` — the extension round's continuity gate reproduces its ratios. Origin-ladder extension round (`origin-ladder-more-contexts`; generation 2026-08-23, pod chain 2026-08-24): generation via the Anthropic Batch API (83,313 calls over 43,000 extension contexts, harvested in ~20 min; 0 GPU); own-answer span-length regenerations (43,000 vLLM rollouts), teacher-forced capture (126,313 forwards, ~55 min), interleaved ~51 GB store upload, and the full fit battery (~1.83 h vs 3.0 h booked) on RunPod `pod-823-extladder` (1× H100, chain wall ~3.5 h vs ~8.8 h booked, rc=0). Round code + artifacts at [`5679ffb426`](https://github.com/superkaiba/explore-persona-space/commit/5679ffb426417bf6b85d2c1cd269ec1dd312f5ff) (branch issue-823-extladder, pushed): `eval_results/issue_823/origin-ladder-more-contexts/` (headline aggregation `ladder_ext_summary.json`; per-fit hygiene `ladder_ext_r2.json`; solver-parity report `g2_ext_report.json`; fits pilot `g3_pilot_record.json`; boundary battery `p2_ext_boundary.json`; per-rung paired reads `shared_persona_paired_rung*.json` + companion `shared_persona_paired_rand_rung*.json` + `rand_ladder_manifest.json`; per-context arrays `percontext_rung*.npz` / `percontext_rand_rung*.npz`; generation/capture/mask digests) and extension figures `ladder_ext_fig1`–`fig10` under `figures/issue_823/`; the round-6 revision's two-ladder per-context companion (extension figure 8, re-rendered by `scripts/issue823_ladder_ext_figures.py --only-fig8` from the committed per-context arrays) committed at [`85cf461704`](https://github.com/superkaiba/explore-persona-space/commit/85cf46170452071bb5e2d00cb6c684a13d1c23fc). Extension data on HF (pinned): [issue823_inconsistent_origin_ladder/ tree @ ba2e24415e](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/ba2e24415e76e0dfcffb2cff60ba7e2847671227/issue823_inconsistent_origin_ladder) — `raw_completions/ladder_ext` (57 files: sharded per-persona answers + assignment + digests), `raw_completions/ladder_ext_own` (46 files: span-length regenerations), `analysis_tensors/ext` (74 files incl. the sharded pair-store row index, 83,313 distinct rows reconciled). Reused round inputs:

- Reused the 5,000-context LMSYS context-vector bundle from [#779](https://eps.superkaiba.com/tasks/779): [`issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/c94070508aa1c1f9c015ceb072231a2e51b28b3f/issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt) @ `c94070508aa1` — fit: the same last-context-token predictors (all 28 layers) every parent fit used; row identity confirmed by the parent-mask reproduce gate (|Δ| ≤ 1e-7).
- Reused the own and plain answer-profile tensors from [#823](https://eps.superkaiba.com/tasks/823)'s parent run: [`issue823_own_vs_external/analysis_tensors/v_a_prime.pt`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/8039d15f30deb845765cbb24d9cdb8708a5e7b0f/issue823_own_vs_external/analysis_tensors/v_a_prime.pt) and [`v_b2.pt`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/8039d15f30deb845765cbb24d9cdb8708a5e7b0f/issue823_own_vs_external/analysis_tensors/v_b2.pt) @ `8039d15f30de` — fit: the two anchor arms are refits of exactly these targets on the new mask.
- Reused the parent common-valid mask from [#823](https://eps.superkaiba.com/tasks/823)'s parent run: [`issue823_own_vs_external/raw_completions/phase1/common_valid_idx.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/8039d15f30deb845765cbb24d9cdb8708a5e7b0f/issue823_own_vs_external/raw_completions/phase1/common_valid_idx.json) @ `8039d15f30de` — fit: the round's 4,629-context mask is this 4,998-context mask minus the new refusal drops.
- Reused the parent answer-span lengths from [#823](https://eps.superkaiba.com/tasks/823)'s parent run: [`issue823_own_vs_external/analysis_tensors/phase3_span_lengths.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/8039d15f30deb845765cbb24d9cdb8708a5e7b0f/issue823_own_vs_external/analysis_tensors/phase3_span_lengths.json) @ `8039d15f30de` — fit: ladder capture truncates each answer span to the shorter of the persona span and the parent own span, for parity with the anchors.
- Extension round reused the same pinned inputs — the pass-b context vectors @ `c94070508aa1`, the parent tensors + mask @ `8039d15f30de`, and the banked ladder store, roster, and assignment @ `009b58fdcf` — fit: the 5,000 banked contexts are an ordered-equal prefix of the 48,000-context extension pool (reproduction check in the generation digest), so banked captures and fits stay valid rows of the extension rungs.

Code-review concerns from the ladder round's fix reviews are recorded in the concerns ledger. Five (`kill1b-total-drop-abort-unimplemented`, `kill-1b-total-drop-budget-unimplemented`, `mask-abort-payload-key-regression`, `mask-contract-check-post-heavy-load`, `p0-sentinel-fingerprint-not-live-bound`) carry evidence-based unreachable-on-the-realized-data dispositions (total new drops 369 under the 500 budget, integrity-class zero, no abort fired, prompt integrity verified on an independent reconstruction). The sixth, `refusal-drift-wilson-reporting-unimplemented`, is a standing disclosure rather than a discharge: the plan's per-persona-per-arm refusal-rate comparison against the pilot was not produced in that round, so inference there is limited to the 4,629-context all-arm survivor population (see Methodology); the extension round reports its per-persona refusal rates directly (Design).

**Context:** created 2026-07-01, run 2026-07-02; zero-GPU identity-baseline follow-up (free analysis on existing tensors, no new data) folded 2026-07-02; same-issue follow-up round `inconsistent-origin-persona-ladder` (source: user-chat; scope posted 2026-08-19, run 2026-08-20) folded 2026-08-20; banked paired re-read (user-chat inline free analysis, 0 GPU-h) landed on main 2026-08-22; same-issue follow-up round `origin-ladder-more-contexts` (source: user-chat; scope posted 2026-08-23, generation 2026-08-23, pod chain 2026-08-24) folded 2026-08-24. Child of #722 (context pool), method parent #779 (map + read-out layers). Originating prompt (verbatim):

> ## Motivation
> - We showed that there is a mapping from context vector to answer profile -> when the answer was generated from that context
> - It is also interesting to see if that mapping holds for answers not generated from the model
>     - i.e. is the model just predicting what its assistant will say
>         - or somehow predicting in general the activations
>     - what I'm trying to get at I think is is the model really just predicting its internals or also the external output
>         - if it can predict the answer profile for output generated from another model then this indicates it's not really predicting the external
>     - This is also interesting to characterize the finetuned mapping M+ -> should we be measuring it on the generated responses? Currently we are measuring it on the pre-finetuned responses
> ## Methodology
> - Take pre-finetuned model
> - Compute:
>     - context vector -> answer profile mapping for answers generated from the pre-finetuned model
>     - context vector -> answer profile mapping for answers generated from Claude 4.5 Sonnet (with weird behavior prompt so that answers aren't too similar)
> - Use held-out evaluation context setup from 722
> - Measure (at all layers):
>     - which mapping is a better predictor on the held-out contexts
>         - if own answer mapping is better -> model is actually predicting its output
>         - if they are both similar -> model is just predicting its internals
>     - can one mapping predict the other? --> almost definitely not but why not check
>
> [Design decisions confirmed in chat 2026-07-01: three external arms — Sonnet with context C + weird-style instruction, Sonnet with context C plain, and same-answer-text-across-contexts (bare-probe Sonnet); new child task of #722; M+ re-measurement on finetuned-model generations out of scope (named follow-up); no MLP training anywhere (ridge only); keep the Betley-pinned 48-probe pool (not WildChat/UltraChat).]

Follow-up-round originating prompt (verbatim): `let's run an experiment to test if it degrades when origins are inconsistent. train mapping on: - lmsys with answers from single persona (promtped into Claude) - lmsys with answers from multiple personas (prompted into Claude) -- prompts matched to above. see how mapping quality scales with # of personas. minimum number of contexts to have well-posed mapping`

Extension-round originating prompt (verbatim): `add it for now and then run the more contexts version` — the "it" folded the banked paired re-read into the theory paper; the second half armed this round.

<!-- verifier WARN acknowledged: 6 Takeaways bullets exceed the 30-word bullet cap — each bullet carries all three traits' numbers or a full follow-up round's synthesis (numbers-first density is deliberate); several result sections exceed the 120-word per-result prose cap and the total-prose budget is exceeded — the four-arm × three-trait parent design plus three folded follow-up rounds (identity baseline; persona ladder; origin-ladder extension) carry dense per-arm numbers; 4 pre-extension figures carry text-less sidecars (rendered before sidecar text embedding; regenerating would alter shipped parent figures); goal: frontmatter field is preserved from the pre-promotion body by set-body -->
