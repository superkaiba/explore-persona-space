---
title: The answer-boundary form, not the narrative prose, carries the framing cost
  of context→answer maps (MODERATE confidence)
kind: experiment
tags:
- trigger-dense
created_at: '2026-08-03T22:07:42Z'
has_clean_result: true
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

**Methodology:** [docs/methodology/issue_2054.md @ d077d79f](https://github.com/superkaiba/explore-persona-space/blob/d077d79f19a9090783a5a59da22a0a5703704a7b/docs/methodology/issue_2054.md) · [gist mirror](https://gist.github.com/superkaiba/dd921616d01ccbc53e0004ee97657e28)

## Takeaways

- All 56 lattice cells fit a real context→answer map at well-posed n (per-fold n_train 6,341–9,586 against d = 3,584): held-out R² 0.12–0.58, every cell far above its shuffled-answer null (95th-percentile draw ≈ −0.03) and identity-plus-bias floor (predict the input plus a mean shift; −0.48 or lower); retrieval finds the true answer vector first try in 19% of held-out rows at the median cell, 70% at the best, against ~0.06% chance.
- Swapping the answer-boundary form breaks direct map transfer (median held-out R² −0.06 across all 56 ordered pairs, and the same −0.06 in the 36-pair subset holding story prose fixed verbatim); swapping the prose with the boundary held fixed preserves it (median 0.20, 96 pairs) — the boundary swap is worse in 32 of the 32 story target cells (nominal p < 1e-9); prose is not the carrier.
- The misalignment is context-side: a linear re-map of the context space recovers boundary-swapped transfer to a median 0.84 of the target's own ceiling; an answer-side re-map reaches 0.18 pooled — 0.29 among the 36 prose-fixed story-form pairs, ≈0 among the 20 assistant form swaps.
- The authorship × presentation decomposition is not additive: with answer text matched, authorship contributes −0.14 to −0.29, presentation −0.02 to −0.16, and the interaction +0.11 to +0.19, sign-consistent in 8 of 8 character × model pairs; both single-factor readings of the inserted-vs-on-policy gap are rejected.
- The plan's answer-length parity bound (KS D ≤ 0.30) was never evaluated in-run and, recomputed from persisted answer text, fails in all 16 character-cell pairs (D 0.450–0.561; 22 of 24 pairs overall), though the breach is bounded — inserted-vs-on-policy ceiling gaps survive length matching (median +0.053 → +0.033, sign preserved in 20 of 24 pairs); the worst cell's low ceiling is a truncation-composition artifact (0.21 → 0.39 with its 42.5% capped rows removed vs 0.20 random-removed), but the chat-base cell's 0.21 survives cap-hit and language-drift exclusions alike; its low ceiling stays unexplained.
- Prefix-arm maps stay at or below R² 0.021 in all 56 cells at full input rank: a real negative, not the parent line's collapsed-capture artifact. Realized cost ~6–7 GPU-h against the 186 GPU-h approved envelope.

## Goal

**This experiment in context:** The parent story-framing line ([#1345](https://eps.superkaiba.com/tasks/1345)) measured within-cell context→answer map ceilings for four story characters at n ≈ 2,000–2,180 rows — below the d = 3,584 ambient dimension, so every number was a reduced-basis read (train-fold PCA, k = 1024) — and its cells were not conversation-matched (its base-vs-instruct comparisons were withdrawn as corpus-confounded); its prefix arm was degenerate (a capture defect collapsed prefix inputs to 14 distinct vectors). This task rebuilds the lattice at 6,000+ rows per cell on one shared conversation draw so the ambient fit is well-posed, re-checks the prefix arm at full rank, adds the transposed cell that identifies the authorship × presentation decomposition, and runs the 9-rung transfer ladder over 408 ordered cell pairs. The parent's reduced-basis ceilings (inserted 0.31–0.38, on-policy 0.18–0.26 across characters; verified against the parent's interpretation record of 2026-08-04 — its committed ladder artifact predates the character captures and carries no character-cell ceilings) are not directly comparable: different estimator basis, row counts, and corpus matching. A sibling protocol measured hardcoded-template story cells at R² ≈ 0.01–0.02 ([#1689](https://eps.superkaiba.com/tasks/1689), different corpus and selector — not directly comparable), motivating this task's diverse-scaffold requirement.

**Broader narrative:** the context→answer map line asks what a linear read of the pre-answer context state predicts about the answer representation, and which surface properties of a context — framing, speaker identity, who authored the answer — the map is invariant to. Boundary-form dominance constrains how framing and persona effects should be modeled: the map is anchored to the local answer-boundary token regime, not to the surrounding narrative content.

## Methodology

**Design:** a framing × identity × condition × model lattice of context→answer maps over one shared real-conversation draw (26,889 conversations drawn, through its persisted 99,778-conversation sampling manifest, from a 626,620-conversation multi-turn pool of real user chats — LMSYS-derived, established-dataset tier), decoupled into scaffold-and-splice: (Phase A) generate diverse narrative scaffolds containing one verbatim question and an answer-slot sentinel, with no answer text; (Phase B, "inserted") splice the original conversation's real answer into the slot deterministically (100% keep, span offsets known by construction); (Phase C, "on-policy") prefill the scaffold to the slot and let the measured model write the answer; (Phase D, "transposed") splice the model's story-authored answer into the chat template. Framings realized: chat template and bare text (`User: {Q}\n\nAssistant: {A}`) for the assistant identity; story with attributed quote (`{Name} replied: "{A}"`) and story with bare label (`{Name}: {A}`) for assistant plus four story characters (HELIOS, Wren, Dana, Vex) × both models (Qwen2.5-7B, Qwen2.5-7B-Instruct). The planned fifth framing (indirect reported speech) could not be rendered deterministically and was dropped, per the plan's declared escape. The USER identity is excluded by design (its context arm is a self-prediction in this rig; handled in a separate task). 56 cells total: 48 inserted/on-policy cells plus 8 transposed cells. Scaffold supply: of 53,018 admitted rows, 46,462 were generated on-pod by Qwen2.5-7B-Instruct at temperature 1.0 under a scaffold-writing instruction that is stripped before capture (scenario axes sampled under seed 137) and 6,556 were recovered from the parent's kept stories via a scaffold stripper; every candidate was judged for structural extractability by `claude-sonnet-4-5-20250929` (integer rubric, admit at score ≥ 50; 101,095 calls, pilot-gated with zero truncation and zero parse failures; per-variant judged→admitted e.g. Dana 19,714 → 9,722). The plan also held open a parent-reject re-mining path — a permissive span matcher over the parent's 24,118 rejected inserted rows (~11,721 recoverable), with a raw-vs-permissive matcher control on every re-mined row. Fresh scaffold supply met quota and every Phase B row is a deterministic splice with no matcher, so the re-mining path never ran and the control is vacuous; the production fit JSONs carry no matcher fields.

**Training:** N/A — no model training. All maps are closed-form ridge fits on frozen-model activations.

**Evaluation:** per cell, the held-out R² of a ridge map from the context vector to the answer vector at layer 19 of the residual stream (both models are 28-layer, d = 3,584). The context vector is the last-token state at the read position before the answer begins; the answer vector is the token-mean over the answer span; the prefix arm instead maps the last-token state before the question (prefix = everything before the user query; context = prefix + query — both arms fit per cell). Fits use generalized-cross-validation ridge with a degrees-of-freedom cap of 0.9 and 5 conversation-grouped folds from one shared fold map (26,889 conversations, seed 137) reused by every cell, so held-out sets are identical across cells. Controls per cell: shuffled-answer matched-capacity null (100 draws per fold), identity-plus-bias baseline (prediction = context vector + train-fold mean offset), nearest-neighbor retrieval (cosine and euclidean, accuracy at k ∈ {1, 5, 10}, chance = k over the ~1,996-row held-out pool), a reduced-basis diagnostic (train-fold PCA, k = 1024), and a conversation-level bootstrap on each fold (200 draws). Transfer: for each of 408 ordered cell pairs (both arms), a 9-rung ladder applies the source-fitted map to the target with increasing adaptation — direct, context/answer mean offsets, bias refit, global scale, orthogonal rotation, context-side linear re-map, answer-side linear re-map, full refit — each rung scored on the target's held-out fold and normalized by the target's own ceiling, on the pair's equalized conversation intersection. Intersections differ by pair class: boundary-swap pairs share 8,000–11,901 conversations and model-swap pairs 7,999–11,901, but prose-swap pairs share only 2,939–4,450 (152 of the 408 context-arm pairs sit below 4,450). At prose-swap intersections the three refit rungs (context re-map, answer re-map, full refit) train at n_train = 0.8 × intersection, below d = 3,584 — an under-determined regime — so those rungs are read as descriptive only; every headline-carrying read (direct rungs everywhere, boundary- and model-class refit rungs) is well-posed.

| Parameter | Value | Source |
|---|---|---|
| Read layer | 19 | parent line convention; plan §11 |
| Hidden dimension d | 3,584 | `Qwen/Qwen2.5-7B` config |
| Folds K / fold grain | 5 / conversation-grouped | plan §11; `shared_fold_map.json` |
| Seed (draw, folds, nulls) | 137 | plan §11 |
| Ridge selector | GCV, dof cap 0.9 | `issue2054_fits.py` at run SHA |
| Null draws / bootstrap draws | 100 / 200 per fold | fit JSONs (`n_null_draws`, `bootstrap_draws`) |
| Phase C sampling | temperature 1.0, seed 137; max_new_tokens 2,048 (attributed-quote, bare-label, and chat-instruct cells — 13 of 16 digests) / 4,096 (bare-text under both models + chat-base, after the cap-hit regen) | `phase_c_digest__*.json` (16 files) |
| Scaffold judge | `claude-sonnet-4-5-20250929`, max_tokens 1,024, threshold 50 | `phase_a_digest.json` |

**Data extraction:** teacher-forced forward passes on the rendered final text (per-segment token-id concatenation with offset-mapped span positions), capturing the context, prefix, and answer-mean vectors per row: 451,414 rows across the 48 inserted/on-policy cells plus 64,000 transposed-cell rows (515,414 total) with zero failures (2h04m on 2× H100). Fits and ladders ran on CPU pods (measured pilots: 329 s and 49 s per unit-fold, peak RSS ~21 GiB, routed off the shared VM per the pilot gates). Planned-vs-actual deviations: (1) the plan's equalize-down requirement (every compared cell trained on the same row count) was not applied to the committed per-cell fits — each cell was fit on its own full row set (8,000–11,901 rows; the fit JSONs record `single_cell_group: true`), while the ladder applied per-pair equalization as planned; a companion refit of the largest cell subsampled from 11,901 to 8,000 rows moved its ceiling by only 0.013 (0.492 full vs 0.480 subsampled), bounding this confound. (2) The plan's per-cell answer-length diagnostics (kill gate at KS D > 0.30) never evaluated in-run — the capture diagnostics carrying per-row lengths were not persisted off-pod, so the gate fail-opened in all 56 fits; this round recomputes the read from persisted answer text (final result below). (3) The conversation-intersection kill gate (floor 4,480) was evaluated on pre-transposed-cell groups and cleared everywhere (minimum 7,999, per the per-cell fit kill-gate reports). (4) The Phase C generation cap was raised on three of the eight form × model generation legs: all legs first ran at max_new_tokens 2,048; the cap-hit trigger (a cell with over 2% of rows ending at the cap is fully re-generated at twice the cap) fired on bare-text under both models and on chat under the base model, and those three cells were re-generated at 4,096 — the footer's cap-hit regen. Bare-text base cleared at the raised cap (1.1% capped); the 42.5% and 5.6% rates the truncation analysis works with are the residual rates at 4,096.

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

Row drawn at random (seed 42) from the 42.5%-cap-hit cell (bare-text form, instruct, on-policy) — a 15-word excerpt of an 8,017-character document-style continuation that ran to the generation cap ([on_policy/qwen2.5-7b-instruct/conversation_paired_stories_assistant @ 003e3925](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/003e392548fcbbe866c6f345f4688d8176cd9f04/issue2054_lattice/on_policy/qwen2.5-7b-instruct/conversation_paired_stories_assistant), row 5238 of `on_policy_conversation_paired_stories_assistant__bare_text.jsonl`):

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

![Within-cell ceilings across the 56-cell lattice, grouped by framing and condition](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2d61542c588e45c49ad7233b4252b56d9d8e6fe4/figures/issue_2054/ceilings_lattice.png)

> **Figure.** *The boundary form orders the within-cell ceilings.* 56 cells, 5 shared conversation-grouped folds each (per-fold n_train 6,341–9,586); the global maximum is the chat on-policy instruct cell (0.58), chat and bare-text inserted cells reach 0.41–0.49, and attributed-quote cells beat bare-label cells under identical prose in 19 of 20 identity × condition × model slices (exception: Vex on-policy base). Per-fold values: [fold-level view](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2d61542c588e45c49ad7233b4252b56d9d8e6fe4/figures/issue_2054/ceilings_lattice_folds.png).

Every context-arm cell clears its shuffled-answer null (p95 ≈ −0.03) and identity-plus-bias floor (−0.48 to −1.51), and the ambient fit beats the reduced-basis diagnostic in all 56 cells — the parent line's estimator caveat is retired. With story prose identical across the two story forms, the attributed-quote boundary reaches 0.34–0.39 (inserted) where the bare label reaches 0.24–0.31. The boundary form alone moves the ceiling by ~0.08–0.10. Retrieval confirms the maps are discriminative: fold-mean cosine accuracy at k=1 spans 0.04–0.70 across the 56 cells (median 0.19), and the maximum — the chat on-policy instruct cell — sits against chance ≈0.0006 in its 1,560–1,656-row held-out pools.

### Swapping the boundary breaks direct transfer; swapping the prose does not

Each point is one story target cell: mean direct-transfer R² (source map applied unchanged) from same-boundary, different-prose sources (x-axis) against same-prose, different-boundary sources (y-axis).

![Direct transfer into each story target: prose swap vs boundary swap](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2d61542c588e45c49ad7233b4252b56d9d8e6fe4/figures/issue_2054/hero_boundary_vs_prose.png)

> **Figure.** *Every story target transfers worse across a boundary swap than across a prose swap.* 32 story target cells; each point averages that target's direct-transfer rungs (per-pair fold means; pair-equalized intersections of 8,000+ conversations for boundary swaps, 2,939–4,450 for prose swaps), and all 32 fall below the equal-transfer line. Per-pair raw view: [all single-axis pairs](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2d61542c588e45c49ad7233b4252b56d9d8e6fe4/figures/issue_2054/boundary_vs_prose_pairs.png).

Boundary-swapped transfer is negative for most targets (median −0.06) while prose swaps retain most of the ceiling (median 0.20); the gap holds in all 32 targets (nominal p < 1e-9; targets share sources and folds, so the count is the primary read). This meets the plan's kill criterion (within-scaffold cross-boundary below cross-scaffold same-boundary transfer) in the direction the plan labeled uninterpretable — reading it is a deviation from that label, made after seeing the data.

The deviation rests on the 36-pair verbatim-prose subset: the boundary is the single changed variable there, so the scaffold mismatch the label assumed cannot explain the failure. The prose hypothesis is rejected. At the direct rung, prose swaps on inserted text transfer best (mean 0.77 of ceiling, 48 pairs); model swaps on identical inserted text reach 0.52 (24 pairs; full recovery under adaptation, next result).

### Only a context-side linear re-map recovers boundary swaps

One line per pair class: median transfer R² normalized by the target's ceiling, across the 9 rungs from direct application to full refit, with interquartile bands.

![Nine-rung transfer ladder by pair class, context arm](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2d61542c588e45c49ad7233b4252b56d9d8e6fe4/figures/issue_2054/ladder_recovery.png)

> **Figure.** *Only a context-side re-map restores boundary-swapped transfer.* 408 ordered pairs, context arm; boundary-swapped pairs jump from −0.16 of ceiling (direct) to 0.84 under a context-side linear re-map but only 0.18 under an answer-side re-map; prose-swapped pairs start at 0.73, peak at 0.84 under rotation, and decline to 0.70 at full refit; model-swapped pairs reach 0.87 under rotation alone.

The asymmetry localizes the boundary swap's change to the context side: a linear transform of the context space restores the map; the answer side stays incompatible with a linear correction (0.29 among the 36 prose-fixed story pairs, ≈0 among the 20 assistant swaps). Both re-map rungs are same-dimension fits on the same intersection and held-out fold (Methodology); the asymmetry is not a capacity difference.

Cross-model pairs on identical inserted text recover to 1.00 of ceiling at full refit and 0.87 under rotation alone — with text fixed, the two maps are nearly rotation-equivalent; on-policy cross-model pairs, whose answers differ by construction, reach 0.17 directly. The fourth series (208 authorship/presentation edges of the 2×2) starts lowest, −0.88, recovering to 0.75 only at full refit. The prose-swap decline across the refit rungs is an estimator artifact: those rungs train below d on the smaller cross-character intersections.

### The authorship × presentation decomposition is not additive

Three terms per character × model pair, from the four cell ceilings: authorship (transposed minus chat), presentation (inserted-story minus chat), and their interaction — for the 8 pairs whose transposed cell (story-authored answers re-presented in the chat template) shares answer text with the on-policy story cell (attributed-quote form).

![Authorship, presentation, and interaction terms across 8 character-model pairs](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2d61542c588e45c49ad7233b4252b56d9d8e6fe4/figures/issue_2054/twobytwo_terms.png)

> **Figure.** *Both main terms are negative and the interaction is positive in all 8 pairs.* Context arm, attributed-quote form; whiskers are percentile intervals from 10,000 fold-aligned resamples over the 5 shared folds (resampled at the fold grain rather than the conversation grain). Underlying ceilings: [the four cells per pair](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2d61542c588e45c49ad7233b4252b56d9d8e6fe4/figures/issue_2054/twobytwo_cells.png).

Authorship (−0.14 to −0.29) and presentation (−0.02 to −0.16) are negative, the interaction positive and comparable (+0.11 to +0.19), sign-consistent in 8 of 8 pairs (nominal p = 0.008 by sign test; the pairs share the 5 folds, so the count is the primary read). The inserted-vs-on-policy gap is neither a pure what-was-said effect nor a pure how-it-was-framed effect; on-policy cross-framing deltas cannot be read as framing effects. The transposed cell is strongly model-dependent (instruct 0.26–0.36, base 0.12–0.19). Two caveats: the authorship axis carries an answer-length change (bounded by the length-matched refit below), and the chat reference cell was fit at 11,901 rows against 8,000 for the others (a deviation from the plan's fit-all-cells-at-one-n rule; bounded at 0.013 in Methodology).

### Truncation composition explains the worst flagged ceiling, but not the chat-base cell

For the two cells the truncation audit flagged, plus a subsample-size control on the chat inserted cell: the full-cell fit against refits excluding the suspect rows (truncation-capped; for chat-base also language-drifted) and refits excluding as many random rows, under the same folds, seed, and ridge recipe.

![Cap-hit- and drift-excluded vs random-matched-n companion refits](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2d61542c588e45c49ad7233b4252b56d9d8e6fe4/figures/issue_2054/caphit_censoring_refits.png)

> **Figure.** *Capped rows account for the bare-text cell's depressed ceiling; chat-base exclusions move it by at most ~+0.03.* Context arm, assistant cells; bare-text on-policy (instruct): all rows 0.21 (42.5% capped), capped-removed 0.39, random-removed 0.20 (n = 4,599). Chat on-policy (base): all rows 0.21, capped-removed 0.214 vs random 0.207 (n = 7,551); drift-removed 0.224 vs random 0.200 (n = 5,764). Chat inserted (instruct): 0.492 vs subsampled-to-8,000 0.480.

Removing the capped rows nearly doubles the bare-text cell's ceiling; removing random rows at the same n slightly lowers it. The depressed ceiling comes from the capped continuations themselves (the bare-text render has no end-of-turn token, so generations run to the cap); sample size does not explain it. Across the lattice, on-policy cross-cell reads (the decomposition's authorship axis included) measure changes in what is written jointly with how it is encoded. The chat-base cell resists the same treatment: excluding its 5.6% capped rows moves it by +0.007, and excluding its 31.6% language-drifted rows reaches 0.224 against a 0.200 matched-n random control — a composition effect near +0.03 that still leaves the ceiling far below the 0.58 of its instruct twin. That cell's low ceiling remains unexplained.

### The plan's answer-length parity bound fails in every character cell

One value per pair: the answer-length Kolmogorov–Smirnov statistic for each of the 24 inserted-vs-on-policy pairs (16 character, 8 assistant), computed from persisted answer text over the pair's matched conversations, against the plan's 0.30 bound.

![Per-pair answer-length KS statistics against the parity bound](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2d61542c588e45c49ad7233b4252b56d9d8e6fe4/figures/issue_2054/ks_parity_pairs.png)

> **Figure.** *All 16 character pairs breach the KS 0.30 bound (D 0.450–0.561).* 22 of 24 pairs breach overall; assistant bare-text base (D = 0.135) and chat instruct (0.214) pass, and assistant chat base is marginal (0.330). Distribution view: [answer-length distributions](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2d61542c588e45c49ad7233b4252b56d9d8e6fe4/figures/issue_2054/answer_length_parity.png).

The gate was never evaluated in-run (the fail-open deviation in Methodology); recomputed after the run, it fails everywhere the headline comparisons live. Mean token ratios fall outside the plan's 0.25–4.0 bound in 6 of 24 pairs (full span 0.19–9.17): story on-policy answers run 2–5× shorter than the spliced real answers, and the bare-text instruct cell runs 9× longer.

Language drift is monotone in boundary form: attributed-quote cells 1–6%, bare-label 7–16%, chat and bare-text 16–39% of conversations whose original answer was not Chinese-script (inserted cells: fixed text by construction). Inserted-vs-on-policy contrasts therefore compare different answer populations on top of different authorship.

### The inserted-vs-on-policy ceiling gaps survive answer-length matching

Both cells of each of the 24 inserted-vs-on-policy pairs were refit on length-matched subsamples (equal per-decile-bin draws from the pair's pooled answer lengths, same folds and ridge recipe), setting the matched ceiling gap beside the committed full-population gap (left) and per-cell matched-vs-full values (right); context arm only, a stated deviation.

![Full against length-matched ceiling gaps for all 24 pairs](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e65147e9f97d8fd5e353625cf9d1ef0efb3a1835/figures/issue_2054/length_stratified_refit.png)

> **Figure.** *Gap signs survive length matching in 20 of 24 pairs.* Context arm, 24 inserted-vs-on-policy pairs; matched = equal per-decile-bin subsamples of pooled pair answer lengths (10 bins, seed 137 per pair); character pairs median +0.048 → +0.033, assistant +0.097 → +0.048; squares = reduced k=1024 basis (matched per-fold n_train under d=3,584; nine pairs), gaps compared within one basis.

The gaps survive at parity, attenuated: median +0.053 full against +0.033 matched, sign preserved in 20 of 24 pairs, and assistant pairs roughly halved (+0.097 → +0.048). The one collapse, bare-text instruct (+0.274 → +0.014), is the 42.5% cap-hit cell isolated two results above; the four sign flips all sit below 0.033 in magnitude.

### The prefix arm's near-zero survives the capture check

No standalone figure — the per-cell prefix values are the gray crosses hugging zero in the first figure. Across all 56 cells the prefix arm's held-out R² spans −0.001 to +0.021 (median 0.006), with nulls at ≈ −0.03 and identity-plus-bias floors from −0.0006 (chat and bare-text cells, where a constant template-header prefix reduces the baseline to the train-fold mean) down to −2.82 (story cells), while the same rows' context arms reach 0.12–0.58.

The parent line's collapse signature (prefix inputs spanning at most 20 distinct vectors) is absent: the prefix state is captured per row at the renderer-recorded prefix end, and a rank probe over 4,000 rows of a story cell measures effective rank 3,584 — full rank. The story cells, whose per-row narrative prefixes vary at full rank, still top out at 0.021. On this corpus — one shared draw, diverse per-row scaffolds — the pre-question state carries essentially no linearly decodable information about the answer-mean vector at layer 19; the question span is what makes the context vector predictive.

---

**Repro:** ~6–7 GPU-h realized (scaffold + on-policy generation with cap-hit regen ≈ 2.9; capture 4.1 on 2× H100; transposed-cell splice+capture 0.38; fits and ladders on CPU pods) vs 186 GPU-h approved. Code: `scripts/issue2054_{forms,phase_a,phase_b,phase_c,phase_d,capture,fits,ladder,authorship_2x2}.py` on branch `issue-2054` (fits and 2×2 at commit `7041a2b7`); analyzer companion figures from `scripts/issue2054_analyzer_figs.py` (same branch). Data: HF [issue2054_lattice @ 003e3925](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/003e392548fcbbe866c6f345f4688d8176cd9f04/issue2054_lattice) — `scaffolds/`, `spliced_inserted/`, `on_policy/<model>/` (canonical per-model trees; the model-less `on_policy/<variant>/` prefix is deprecated-ambiguous), `cell_c/`, `activations/` (56 npz, 11.11 GB — 48 inserted/on-policy + 8 transposed — plus digests), `fits/` (56 JSONs), `ladder/` (816 rung JSONs + digests; census verified complete) — all prefixes verified by scoped listing this round. Committed on the branch: `eval_results/issue_2054/` (shared fold map, 2×2 terms, audits, pilot gates) plus the analyzer companions under `eval_results/issue_2054/analyzer_companions/` (answer-length parity, language-drift counts, companion refits; round 2 adds the chat-base drift-excluded refit `drift_refit.json` and the per-pair ladder intersection/composition scan `ladder_intersection_composition.json`, and round 3 adds the length-stratified refit `length_stratified_refit.json` (driver `scripts/issue2054_length_stratified_refit.py`; 48 matched-cell refits, VM CPU, ~5.5 h at width 4); token lengths from persisted answer text via the Qwen2.5-7B tokenizer; refits through `issue2054_fits._fit_arm_cell` with restricted conversation sets, seed 42 for random subsets). Round 2 regenerated `ladder_recovery` and `boundary_vs_prose_pairs` (class-label precision), extended `caphit_censoring_refits` with the drift refit bars, and added `ks_parity_pairs`; the review round re-rendered `length_stratified_refit` from its committed JSON with reader-facing pair labels (values unchanged; the driver's figure-only path) — earlier-commit copies are superseded. Per-unit companion figures are deliberately linked, not embedded, keeping one inline figure per result.
- Reused kept-story scaffolds from [#1345](https://eps.superkaiba.com/tasks/1345): `issue1345_framing/<variant>/raw_completions/stories/` @ 003e392548fcbbe866c6f345f4688d8176cd9f04 (6,556 scaffolds recovered by the round-trip-verified scaffold stripper) — fit: same characters and story forms; every recovered scaffold re-judged by this task's own admission gate before entering the pool.
- Reused multi-turn conversation pool from [#1738](https://eps.superkaiba.com/tasks/1738): `issue1738_multiturn/sampling_manifest/` @ 003e392548fcbbe866c6f345f4688d8176cd9f04 (55 JSONL parts; the persisted 99,778-conversation selection from the 626,620-conversation LMSYS-derived pool) — fit: real multi-turn user chats keep one row per scaffold; the shared conversation draw's source.

**Context:** created 2026-08-03 from user chat (origin prompt, verbatim: "run the 6000 row thing in the background with happy coder"; preceding directives: "do a comprehensive audit of why all these generations are failing and rerun so that we get 6000 inserted and 6000 on-policy for all settings we need for my result writeup. for inserted the model doesn't necessarily have to generate itself it can just write the start and then we insert the dialogue directly (BUT THE STORY FRAMING ITSELF SHOULD BE DIVERSE)"; "run so that we only take the first message per character in ALL settings"). Parent: [#1345](https://eps.superkaiba.com/tasks/1345). Run 2026-08-04 → 2026-08-07; analyzed 2026-08-07.


