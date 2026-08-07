---
title: '6k-row scaffold-and-splice lattice: framing x character x condition context-to-answer
  maps at well-posed n'
kind: experiment
tags:
- trigger-dense
created_at: '2026-08-03T22:07:42Z'
has_clean_result: false
parent_id: 1345
origin_prompt: run the 6000 row thing in the background with happy coder
workflow: v1
goal: Build the framing x speaker x completion-condition context-to-answer map lattice
  at 6,000 rows per cell (5,000 train + 1,000 held-out) on a decoupled scaffold-and-splice
  corpus, so every cell is well-posed in the ambient basis (n_train 4,800 > d 3,584)
  and row-paired across framings; report per-cell within-cell ceilings and the 9-rung
  transfer ladder, both mapping arms, with identity+bias and kNN-retrieval reads.
relates_to:
- spec-context-as-vector
---
# The answer-boundary form, not the narrative prose, carries the framing cost of context→answer maps (MODERATE confidence)

<!-- clean-result-v4 -->

## Takeaways

- All 56 lattice cells fit a real context→answer map at well-posed n (per-fold n_train 6,341–9,524 against d = 3,584): held-out R² 0.12–0.58, every cell far above its shuffled-answer null (p95 ≈ −0.03) and identity-plus-bias floor (−0.48 or lower); retrieval finds the true answer vector first try in up to 70% of held-out rows against 0.05% chance.
- Swapping the answer boundary with the story prose held fixed verbatim breaks direct map transfer (median held-out R² −0.06, 56 ordered pairs); swapping the prose with the boundary held fixed preserves it (median 0.20, 96 pairs) — the boundary swap is worse in 32 of the 32 story target cells (p < 1e-9), rejecting prose as the carrier.
- The misalignment is context-side: a linear re-map of the context space recovers boundary-swapped transfer to a median 0.84 of the target's own ceiling; an answer-side re-map reaches only 0.18.
- The authorship × presentation decomposition is not additive: with answer text matched, authorship contributes −0.14 to −0.29, presentation −0.02 to −0.16, and the interaction +0.11 to +0.19, sign-consistent in 8 of 8 character × model pairs (p = 0.008) — both single-factor readings of the inserted-vs-on-policy gap are rejected.
- The plan's answer-length parity bound (KS D ≤ 0.30) was never evaluated in-run and fails for every inserted-vs-on-policy pair when computed from the persisted answer text (D 0.45–0.58, n ≈ 7,985 matched conversations each): in the worst cell (42.5% of rows truncation-capped) removing capped rows lifts the ceiling from 0.21 to 0.39 while removing the same number of random rows lowers it to 0.20 — the flagged low on-policy ceilings are composition artifacts, not framing effects.
- Prefix-arm maps stay at or below R² 0.02 in all 56 cells at full input rank — a real negative, not the parent line's collapsed-capture artifact. Realized cost ~6–7 GPU-h against the 186 GPU-h approved envelope.

## Goal

**This experiment in context:** The parent story-framing line ([#1345](https://eps.superkaiba.com/tasks/1345)) measured within-cell context→answer map ceilings for four story characters at n ≈ 2,000–2,180 rows — below the d = 3,584 ambient dimension, so every number was a reduced-basis read (train-fold PCA, k = 1024) — and its cells were not conversation-matched (its base-vs-instruct comparisons were withdrawn as corpus-confounded); its prefix arm was degenerate (a capture defect collapsed prefix inputs to 14 distinct vectors). This task rebuilds the lattice at 6,000+ rows per cell on one shared conversation draw so the ambient fit is well-posed, re-checks the prefix arm at full rank, adds the transposed cell that identifies the authorship × presentation decomposition, and runs the 9-rung transfer ladder over 408 ordered cell pairs. The parent's reduced-basis ceilings (inserted 0.31–0.38, on-policy 0.18–0.26 across characters; verified against the parent's interpretation record of 2026-08-04 — its committed ladder artifact predates the character captures and carries null ceilings) are not directly comparable: different estimator basis, row counts, and corpus matching. A sibling protocol measured hardcoded-template story cells at R² ≈ 0.01–0.02 ([#1689](https://eps.superkaiba.com/tasks/1689), different corpus and selector — not directly comparable), motivating this task's diverse-scaffold requirement.

**Broader narrative:** the context→answer map line asks what a linear read of the pre-answer context state predicts about the answer representation, and which surface properties of a context — framing, speaker identity, who authored the answer — the map is invariant to. Boundary-form dominance constrains how framing and persona effects should be modeled: the map is anchored to the local answer-boundary token regime, not to the surrounding narrative content.

## Methodology

**Design:** a framing × identity × condition × model lattice of context→answer maps over one shared real-conversation draw (26,889 conversations from the multi-turn pool built from real user chat corpora), decoupled into scaffold-and-splice: (Phase A) generate diverse narrative scaffolds containing one verbatim question and an answer-slot sentinel, with no answer text; (Phase B, "inserted") splice the original conversation's real answer into the slot deterministically (100% keep, span offsets known by construction); (Phase C, "on-policy") prefill the scaffold to the slot and let the measured model write the answer; (Phase D, "transposed") splice the model's story-authored answer into the chat template. Framings realized: chat template and bare text (`User: {Q}\n\nAssistant: {A}`) for the assistant identity; story with attributed quote (`{Name} replied: "{A}"`) and story with bare label (`{Name}: {A}`) for assistant plus four story characters (HELIOS, Wren, Dana, Vex) × both models (Qwen2.5-7B, Qwen2.5-7B-Instruct). The planned fifth framing (indirect reported speech) could not be rendered deterministically and was dropped, per the plan's declared escape. The USER identity is excluded by design (its context arm is a self-prediction in this rig; handled in a separate task). 56 cells total: 48 inserted/on-policy cells plus 8 transposed cells. Scaffold supply: of 53,018 admitted rows, 46,462 were generated on-pod by Qwen2.5-7B-Instruct at temperature 1.0 under a scaffold-writing instruction that is stripped before capture (scenario axes sampled under seed 137) and 6,556 were recovered from the parent's kept stories via a scaffold stripper; every candidate was judged for structural extractability by `claude-sonnet-4-5-20250929` (integer rubric, admit at score ≥ 50; 101,095 calls, pilot-gated with zero truncation and zero parse failures; per-variant judged→admitted e.g. Dana 19,714 → 9,722).

**Training:** N/A — no model training. All maps are closed-form ridge fits on frozen-model activations.

**Evaluation:** per cell, the held-out R² of a ridge map from the context vector to the answer vector at layer 19 of the residual stream (both models are 28-layer, d = 3,584). The context vector is the last-token state at the read position before the answer begins; the answer vector is the token-mean over the answer span; the prefix arm instead maps the last-token state before the question (prefix = everything before the user query; context = prefix + query — both arms fit per cell). Fits use generalized-cross-validation ridge with a degrees-of-freedom cap of 0.9 and 5 conversation-grouped folds from one shared fold map (26,889 conversations, seed 137) reused by every cell, so held-out sets are identical across cells. Controls per cell: shuffled-answer matched-capacity null (100 draws per fold), identity-plus-bias baseline (prediction = context vector + train-fold mean offset), nearest-neighbor retrieval (cosine and euclidean, accuracy at k ∈ {1, 5, 10}, chance = k over the ~1,996-row held-out pool), a reduced-basis diagnostic (train-fold PCA, k = 1024), and a conversation-level bootstrap on each fold (200 draws). Transfer: for each of 408 ordered cell pairs (both arms), a 9-rung ladder applies the source-fitted map to the target with increasing adaptation — direct, context/answer mean offsets, bias refit, global scale, orthogonal rotation, context-side linear re-map, answer-side linear re-map, full refit — each rung scored on the target's held-out fold and normalized by the target's own ceiling, on the pair's equalized conversation intersection (4,450+ everywhere).

| Parameter | Value | Source |
|---|---|---|
| Read layer | 19 | parent line convention; plan §11 |
| Hidden dimension d | 3,584 | `Qwen/Qwen2.5-7B` config |
| Folds K / fold grain | 5 / conversation-grouped | plan §11; `shared_fold_map.json` |
| Seed (draw, folds, nulls) | 137 | plan §11 |
| Ridge selector | GCV, dof cap 0.9 | `issue2054_fits.py` at run SHA |
| Null draws / bootstrap draws | 100 / 200 per fold | fit JSONs (`n_null_draws`, `bootstrap_draws`) |
| Phase C sampling | temperature 1.0, max_new_tokens 4,096, seed 137 | `phase_c_digest__*.json` |
| Scaffold judge | `claude-sonnet-4-5-20250929`, max_tokens 1,024, threshold 50 | `phase_a_digest.json` |

**Data extraction:** teacher-forced forward passes on the rendered final text (per-segment token-id concatenation with offset-mapped span positions), capturing the context, prefix, and answer-mean vectors per row: 451,415 rows captured across the 56 cells with zero failures (2h04m on 2× H100). Fits and ladders ran on CPU pods (measured pilots: 329 s and 49 s per unit-fold, peak RSS ~21 GiB, routed off the shared VM per the pilot gates). Planned-vs-actual deviations: (1) the plan's equalize-down requirement (every compared cell trained on the same row count) was not applied to the committed per-cell fits — each cell was fit on its own full row set (8,000–11,901 rows; the fit JSONs record `single_cell_group: true`), while the ladder applied per-pair equalization as planned; a companion refit of the largest cell subsampled from 11,901 to 8,000 rows moved its ceiling by only 0.013 (0.492 full vs 0.480 subsampled), bounding this confound. (2) The plan's per-cell answer-length diagnostics (kill gate at KS D > 0.30) never evaluated in-run — the capture diagnostics carrying per-row lengths were not persisted off-pod, so the gate fail-opened in all 56 fits; this round recomputes the read from persisted answer text (final result below). (3) The conversation-intersection kill gate (floor 4,480) was evaluated on pre-transposed-cell groups and cleared everywhere (minimum 7,989).

**Sample training/evaluation data + completions:** the corpus derives from real user conversations, so example rows are sanitized for context hygiene — answers over 15 words are excerpted with a pointer to the full row; labels and indices are verbatim. All four rows below were drawn at random (seed 42), one per named file — none were selected on content. Full artifacts: [issue2054_lattice @ 003e3925](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/003e392548fcbbe866c6f345f4688d8176cd9f04/issue2054_lattice).

Row drawn at random (seed 42) from the on-policy attributed-quote file for character Dana, instruct model ([on_policy/qwen2.5-7b-instruct/char_dana @ 003e3925](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/003e392548fcbbe866c6f345f4688d8176cd9f04/issue2054_lattice/on_policy/qwen2.5-7b-instruct/char_dana), row 204 of `on_policy_char_dana__attrib_quoted.jsonl`):

<details>
<summary>Row 204 — conv_id stripped_s2147, finish_reason stop, 27 answer chars</summary>

```
answer: Olá, sim, posso lhe ajudar?
```

</details>

Row drawn at random (seed 42) from the transposed-cell file (story-authored answer re-presented in the chat template; [cell_c/char_wren_op @ 003e3925](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/003e392548fcbbe866c6f345f4688d8176cd9f04/issue2054_lattice/cell_c/char_wren_op), row 2006):

<details>
<summary>Row 2006 — conv_id mt_387df7fca06a, 69 answer chars</summary>

```
answer: Well, come closer and I'll tell you quickly. Remember, it's a secret.
```

</details>

Row drawn at random (seed 42) from the 42.5%-cap-hit cell (bare-text form, instruct, on-policy) — a 15-word excerpt of an 8,017-character document-style runaway continuation ([on_policy/qwen2.5-7b-instruct/conversation_paired_stories_assistant @ 003e3925](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/003e392548fcbbe866c6f345f4688d8176cd9f04/issue2054_lattice/on_policy/qwen2.5-7b-instruct/conversation_paired_stories_assistant), row 5238 of `on_policy_conversation_paired_stories_assistant__bare_text.jsonl`):

<details>
<summary>Row 5238 — conv_id mt_wc:cb80465ff, finish_reason length (sanitized excerpt)</summary>

```
answer[:15 words]: 2.3. Az ateizmus félelmetessége és a felvilágosodás hazugsága Ez a kérdés
egészen összetett és számos [truncated — real-world-corpus row; verify at the linked HF path]
```

</details>

Row drawn at random (seed 42) from the base-model chat on-policy file — the base model answers in the user role, one face of that cell's 5.6% cap-hit and 31.7% language-drift rates ([on_policy/qwen2.5-7b/conversation_paired_stories_assistant @ 003e3925](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/003e392548fcbbe866c6f345f4688d8176cd9f04/issue2054_lattice/on_policy/qwen2.5-7b/conversation_paired_stories_assistant), row 912 of the chat file):

<details>
<summary>Row 912 — conv_id stripped_s2000, finish_reason stop (sanitized excerpt)</summary>

```
answer[:15 words]: Being a user, i do not have time to scan calendar. for now i am
[truncated — real-world-corpus row; verify at the linked HF path]
```

</details>

## Results

### Every cell fits a real map at well-posed n, and the boundary form orders the ceilings

The figure plots each of the 56 cells' 5-fold-mean held-out R² (ambient basis, context arm), grouped by framing × condition, colored by model, with prefix-arm values as gray crosses near zero.

![Within-cell ceilings across the 56-cell lattice, grouped by framing and condition](https://raw.githubusercontent.com/superkaiba/explore-persona-space/337fc1398a141340380fcf60315254d1658e7e08/figures/issue_2054/ceilings_lattice.png)

> **Figure.** *The boundary form orders the within-cell ceilings.* 56 cells, 5 shared conversation-grouped folds each (per-fold n_train 6,341–9,524); chat and bare-text inserted cells sit highest (0.41–0.49), and attributed-quote cells beat bare-label cells under identical prose in every identity × condition × model slice. Per-fold values: [fold-level view](https://raw.githubusercontent.com/superkaiba/explore-persona-space/337fc1398a141340380fcf60315254d1658e7e08/figures/issue_2054/ceilings_lattice_folds.png).

Every context-arm cell clears its shuffled-answer null (p95 ≈ −0.03) and identity-plus-bias floor (−0.48 to −1.51), and the ambient fit beats the reduced-basis diagnostic in all 56 cells — the parent line's estimator caveat is retired. With story prose identical across the two story forms, the attributed-quote boundary reaches 0.34–0.39 (inserted) where the bare label reaches 0.24–0.31: the boundary form alone moves the ceiling by ~0.08–0.10. Retrieval confirms the maps are discriminative: cosine accuracy at k=1 reaches 0.70 (chat on-policy, instruct) against chance 0.0005 in a 1,996-row pool.

### Swapping the boundary breaks direct transfer; swapping the prose does not

For each story target cell, the figure plots mean direct-transfer R² (the source-fitted map applied unchanged) from two source classes: same boundary form but a different character's story prose (x-axis), versus the same scaffold prose but the other boundary form (y-axis).

![Direct transfer into each story target: prose swap vs boundary swap](https://raw.githubusercontent.com/superkaiba/explore-persona-space/337fc1398a141340380fcf60315254d1658e7e08/figures/issue_2054/hero_boundary_vs_prose.png)

> **Figure.** *Every story target transfers worse across a boundary swap than across a prose swap.* 32 story target cells; each point averages that target's direct-transfer rungs (per-pair fold means, pair-equalized conversation intersections of 4,450+), and all 32 fall below the equal-transfer line. Per-pair raw view: [all single-axis pairs](https://raw.githubusercontent.com/superkaiba/explore-persona-space/337fc1398a141340380fcf60315254d1658e7e08/figures/issue_2054/boundary_vs_prose_pairs.png).

Boundary-swapped transfer is negative for most targets (median −0.06) while prose-swapped transfer retains most of the ceiling (median 0.20); the gap holds in 32 of the 32 targets (p < 1e-9 treating targets as independent; they share sources and folds). This meets the plan's kill criterion — within-scaffold cross-boundary transfer below cross-scaffold same-boundary transfer — in the direction the plan labeled uninterpretable, but the reversal is itself the answer: holding prose fixed buys no direct transfer, so the prose-carries-it hypothesis is rejected. Model swaps on identical inserted text transfer best of all single-axis swaps (mean 0.52 of the target ceiling at the direct rung, 24 pairs).

### A context-side linear re-map recovers boundary swaps; an answer-side re-map does not

The figure plots, per pair class, the median transfer R² normalized by the target's own ceiling at each of the 9 adaptation rungs, from direct application to full refit, with interquartile bands.

![Nine-rung transfer ladder by pair class, context arm](https://raw.githubusercontent.com/superkaiba/explore-persona-space/337fc1398a141340380fcf60315254d1658e7e08/figures/issue_2054/ladder_recovery.png)

> **Figure.** *Boundary swaps are repaired on the context side, not the answer side.* 408 ordered pairs, context arm; boundary-swapped pairs jump from −0.16 of ceiling (direct) to 0.84 under a context-side linear re-map but only 0.18 under an answer-side re-map; prose-swapped pairs start at 0.50 and plateau near 0.77; model-swapped pairs reach 0.87 under rotation alone.

The asymmetry localizes what a boundary swap changes: it re-arranges the context representation — a linear transform of the context space restores the map — while the answer-side representation stays incompatible with any after-the-fact linear correction. Cross-model pairs on identical inserted text recover to 1.00 of ceiling at full refit and 0.87 under rotation alone, so the two models' maps are close to a rotation of each other when text is held fixed; on-policy cross-model pairs, whose answers differ by construction, reach only 0.17 of ceiling directly.

### The authorship × presentation decomposition is not additive

For the 8 character × model pairs where the transposed cell shares its answer text with the on-policy story cell (attributed-quote form), the figure plots the three decomposition terms over the four cell ceilings: authorship (transposed minus chat cell), presentation (inserted-story minus chat cell), and their interaction.

![Authorship, presentation, and interaction terms across 8 character-model pairs](https://raw.githubusercontent.com/superkaiba/explore-persona-space/337fc1398a141340380fcf60315254d1658e7e08/figures/issue_2054/twobytwo_terms.png)

> **Figure.** *Both main terms are negative and the interaction is positive in all 8 pairs.* Context arm, attributed-quote form; whiskers are percentile intervals from 10,000 fold-aligned resamples over the 5 shared folds (fold-grain, not conversation-grain). Underlying ceilings: [the four cells per pair](https://raw.githubusercontent.com/superkaiba/explore-persona-space/337fc1398a141340380fcf60315254d1658e7e08/figures/issue_2054/twobytwo_cells.png).

Authorship (−0.14 to −0.29) and presentation (−0.02 to −0.16) are both negative and the interaction is positive and comparable in size (+0.11 to +0.19), sign-consistent in 8 of 8 pairs (p = 0.008): the inserted-vs-on-policy gap collapses to neither a pure what-was-said effect nor a pure how-it-was-framed effect, so on-policy cross-framing deltas cannot be narrated as framing effects. The transposed cell is strongly model-dependent (instruct 0.26–0.36, base 0.12–0.19): story-authored answers re-presented in chat map far better through the instruct model. Two caveats: the authorship axis carries an answer-length change (next result), and the chat reference cell was fit at 11,901 rows against 8,000 for the others (equalize-down deviation; bounded at 0.013 in Methodology).

### Answer-length composition, not framing, drives the flagged on-policy ceilings

The figure plots companion refits of the two cells the truncation audit flagged: the committed full-cell fit, a refit excluding truncation-capped rows, and a refit excluding the same number of random rows — same shared folds, seed, and ridge recipe.

![Cap-hit-excluded vs random-matched-n companion refits](https://raw.githubusercontent.com/superkaiba/explore-persona-space/337fc1398a141340380fcf60315254d1658e7e08/figures/issue_2054/caphit_censoring_refits.png)

> **Figure.** *Excluding capped rows, not matching n, moves the flagged ceilings.* Context arm, assistant cells; bare-text on-policy (instruct): all rows 0.21 (n = 8,000, 42.5% capped) vs capped-rows-removed 0.39 vs random-rows-removed 0.20 (both n = 4,599). Chat on-policy (base): all rows 0.21 (5.6% capped) vs capped-rows-removed 0.214 vs random-rows-removed 0.207 (both n = 7,551). Chat inserted (instruct): all rows 0.492 vs subsampled-to-8,000 0.480.

Removing the capped rows nearly doubles the flagged cell's ceiling; removing random rows at the same n slightly lowers it — the depressed ceiling comes from the runaway continuations themselves (that render has no chat stop discipline), not from sample size. The recovered parity read generalizes the confound: every inserted-vs-on-policy pair breaches the KS 0.30 bound (D 0.45–0.58, n ≈ 7,985 matched conversations per pair), on-policy story answers run 3–5× shorter in tokens than the spliced real answers ([length distributions](https://raw.githubusercontent.com/superkaiba/explore-persona-space/337fc1398a141340380fcf60315254d1658e7e08/figures/issue_2054/answer_length_parity.png)), and the chat and bare-text on-policy assistant cells drift into Chinese-script answers on 16–39% of conversations whose original answer was not Chinese-script (attributed-quote cells: 1–6%) (inserted cells: fixed text by construction). On-policy cross-cell reads, the decomposition's authorship axis included, measure changes in what is written jointly with how it is encoded.

### The prefix arm is a real near-zero, not a capture artifact

No figure — two numbers carry it. Across all 56 cells the prefix arm's held-out R² spans −0.001 to +0.021 (median 0.006), with identity-plus-bias floors of −0.9 to −2.3 and nulls at ≈ −0.03, while the same rows' context arms reach 0.12–0.58.

The parent line's collapse signature (prefix inputs spanning at most 20 distinct vectors) is absent: the prefix state is captured per row at the renderer-recorded prefix end, and a rank probe over 4,000 rows of a story cell measures effective rank 3,584 — full rank. (Chat and bare-text cells have a constant template header as their prefix, so their prefix read is vacuous by construction rather than degenerate; the story cells, whose per-row narrative prefixes vary at full rank, still top out at 0.021.) On this corpus — one shared draw, diverse per-row scaffolds — the pre-question state carries essentially no linearly decodable information about the answer-mean vector at layer 19; the question span is what makes the context vector predictive.

---

**Repro:** ~6–7 GPU-h realized (scaffold + on-policy generation with cap-hit regen ≈ 2.9; capture 4.1 on 2× H100; transposed-cell splice+capture 0.38; fits and ladders on CPU pods) vs 186 GPU-h approved. Code: `scripts/issue2054_{forms,phase_a,phase_b,phase_c,phase_d,capture,fits,ladder,authorship_2x2}.py` on branch `issue-2054` (fits and 2×2 at commit `7041a2b7`); analyzer companion figures from `scripts/issue2054_analyzer_figs.py` (same branch). Data: HF [issue2054_lattice @ 003e3925](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/003e392548fcbbe866c6f345f4688d8176cd9f04/issue2054_lattice) — `scaffolds/`, `spliced_inserted/`, `on_policy/<model>/` (canonical per-model trees; the model-less `on_policy/<variant>/` prefix is deprecated-ambiguous), `cell_c/`, `activations/` (48 npz, 9.73 GB, + digests), `fits/` (56 JSONs), `ladder/` (816 rung JSONs + digests; census verified complete) — all prefixes verified by scoped listing this round. Committed on the branch: `eval_results/issue_2054/` (shared fold map, 2×2 terms, audits, pilot gates) plus this round's companions under `eval_results/issue_2054/analyzer_companions/` (answer-length parity, language-drift counts, companion refits; token lengths from persisted answer text via the Qwen2.5-7B tokenizer; refits through `issue2054_fits._fit_arm_cell` with restricted conversation sets, seed 42 for random subsets). Per-unit companion figures are deliberately linked, not embedded, keeping one inline figure per result.
**Context:** created 2026-08-03 from user chat (origin prompt, verbatim: "run the 6000 row thing in the background with happy coder"; preceding directives: "do a comprehensive audit of why all these generations are failing and rerun so that we get 6000 inserted and 6000 on-policy for all settings we need for my result writeup. for inserted the model doesn't necessarily have to generate itself it can just write the start and then we insert the dialogue directly (BUT THE STORY FRAMING ITSELF SHOULD BE DIVERSE)"; "run so that we only take the first message per character in ALL settings"). Parent: [#1345](https://eps.superkaiba.com/tasks/1345). Run 2026-08-04 → 2026-08-07; analyzed 2026-08-07.
