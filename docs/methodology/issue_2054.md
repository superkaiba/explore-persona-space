# Methodology — issue 2054


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
