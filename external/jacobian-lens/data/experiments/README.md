# Experiments

Prompt sets for the global-workspace experiments. Each `{slug}.json` is prompts only.

## Conventions

Unless a section says otherwise:

- **Lens readout** — at each (layer, token position) the Jacobian lens
  returns a ranked list of vocabulary tokens.
- **Workspace band** — the contiguous mid-network layer range where
  workspace content is read; experiments report over this band, not
  individual layers.
- **Hit** — a target token is a *hit* if it appears at lens rank 1 at any
  (layer, position) in the band over the scored span.
- **Swap** — clamping a lens coordinate replaces one token's direction with
  another's at every band layer at the specified positions, then samples
  the continuation.
- Prompts that span multiple turns are given as
  `[{"role": "user"|"assistant", "content": ...}]`.

## probe-swap

[`probe-swap.json`](probe-swap.json)

90 two-hop factual prompts. `items[*].prompt` ends just before the answer; `intermediate` is the bridge entity, `swap_to` the replacement. Baseline: greedy next-token == `answer`. Swap: replace the `intermediate` representation (linear-probe direction) with `swap_to` across the band at every prompt token position; score next-token at the final position == `swap_answer`. `category` groups items by relation type for the per-category breakdown.

## verbal-introspection

[`verbal-introspection.json`](verbal-introspection.json)

The model is told a thought may have been injected and asked to identify it (`intro_prompt`); one of `prefills` is teacher-forced as the reply, ending in an open quote so the next predicted token is the reported word. For each `surface` in `concepts`, its Jacobian-lens steering direction — the unit-normalized transpose row for that token, scaled by the layer's mean residual norm times a strength scalar — is added to the residual stream at every band layer and every token of the user's question turn; strength 0 is the control. Score: the rank of `surface` in the next-token distribution at the open quote (the last prefill token). The figure reports median reciprocal rank vs strength.

## verbal-report

[`verbal-report.json`](verbal-report.json)

`candidates` maps 14 category names → 14 words each. The prompt is `Think of a {category}. Answer in one word.`; the model's greedy next token at the final `:` is taken as the answer and used as the swap-out target. For each of the first 10 listed candidates (skipping the answer itself), swap answer→candidate across the band at every prompt position. Grading: the swapped-in candidate's rank in the output distribution at the final `:`; success = rank 1.

## directed-modulation

[`directed-modulation.json`](directed-modulation.json)

The model is given an instruction about a target X, then teacher-forced to write an unrelated carrier sentence; we check whether X surfaces in the lens readout over that response span. `phrasings` holds 24 instruction templates — each `text` has an `{x}` slot — in four `group`s, which `group_kind` collapses to {focus, suppress, control}. A trial pairs one phrasing with one entry from `carrier_sentences` and one target. Targets are drawn from `math_problems` (`expr` fills `{x}`; `answer` is the tracked token; `tier` is difficulty) or `topic_categories` (`name` fills `{x}`; every string in `members` is a tracked token). The third family (line-break width) wraps pretraining-distribution English prose at a random column width k ∈ {40,50,…,100} and takes a 5–7-line interior window; the underlying prose is not released, but any prose corpus filtered to alpha-heavy ASCII text reproduces the construction. The metric is hit rate, contrasted across `group_kind`.

## top-down-summoning

[`top-down-summoning.json`](top-down-summoning.json)

`items[*].stimulus` is a passage ending mid-clause. Two questions per item: shared `q1` (predict the next word — answer set `q1_expect`) and per-item `q2` (asks about the passage's latent property). `expected` / `foil` are the property-label vs contrastive-label word sets tracked in the lens over the stimulus span. Metric: Q2 − Q1 fraction of stimulus positions where any `expected` word is in lens top-k. Causal test: swap each `swaps[*]` token pair (label↔foil single-token forms) at every stimulus position and measure the answer shift under each question.

## flexible-generalization

[`flexible-generalization.json`](flexible-generalization.json)

Each entry in `categories` pairs 4 argument values (`args`, e.g. France/Canada/China/Egypt) with 4 function templates (`funcs[*].template`, e.g. "The capital of {arg} is the city of"). A trial fills one template with one arg; baseline grading checks that the greedy next token matches `funcs[*].answers[arg]`. The flexibility test swaps the lens representation of one arg for another from the same category, applied at every prompt position, and scores the next token against the new arg's answer.

## selectivity-language

[`selectivity-language.json`](selectivity-language.json)

`passages[*]` is `{category, key, text}` — eight short passages, two per language (fr/de/es/it). Under `task.explicit_q` the model answers from `authors[category]`; `task.automatic_q` is the neutral-continuation control. `intermediates[category]` are the label tokens tracked in the lens over the question tokens following the passage. Metric: explicit − automatic label-hit rate.

## selectivity-linecount

[`selectivity-linecount.json`](selectivity-linecount.json)

The task is line-length counting: wrap each `passages[*].text` with `textwrap.fill` at its `width`; the ground-truth answer is the character count of the first wrapped line. Each prompt puts the question (`conditions.{none,direct,letter}.question`) before the wrapped passage and ends with the matching `prefill`; the `continue` condition instead uses `explicit_q` as the instruction with no prefill. The lens target set is any two-digit or English number-word token (twenty, thirty, …) in the top-k at any prompt position in the band. The metric is the rate at which such tokens appear, contrasted across conditions over the eleven passages.

## ignition

[`ignition.json`](ignition.json)

`ctx_templates` (40) and `noun_ctx_templates` (20) are carrier sentences with a `{W}` slot; concept pairs come from `countries_12` (all 66 unordered pairs), `alt_words` (each paired with France), and the `idiom_pairs` / `scrambled_pairs` controls. For each (carrier, pair, α) trial the `{W}` token's embedding is replaced by α·emb(A) + (1−α)·emb(B), with α swept 0→1. The readout at each layer is A's reciprocal-rank share, 1/rank(A) ÷ (1/rank(A) + 1/rank(B)), taken at the `{W}` position. A pair's threshold is the α where this share crosses 0.5; the figure shows share heatmaps over layer × (α − threshold), the per-layer 10→90% transition width in α, and histograms of the share across carriers at each pair's threshold α.

## capacity

[`capacity.json`](capacity.json)

Each trial is an 80-word list built as four contiguous 20-word blocks, one block per family in `block_families` (names, surnames, countries, cities), with block order shuffled per trial. Each block's 20 words are sampled fresh from that family's canon — the first `targets_per_family[name]` entries of its `pool` that tokenize to a single token under the target model (pools are oversized so enough survive; the exact canon is therefore model-dependent). `proto` gives candidate category-label words for each family. Metric: at every comma position, count how many of the list words read so far have band-min lens rank ≤ k.

## dual-task

[`dual-task.json`](dual-task.json)

§app-competition. The model holds one or two covert tasks while copying `carrier_sentence`, which is teacher-forced as the assistant turn. Each entry in `pairs` is `{key, concept, concept_words, base, exp, sub}`: `concept` is the noun phrase slotted into the instruction "Concentrate on {concept} … while you write the sentence", `concept_words` are that concept's target tokens, and `base^exp − sub` is its paired arithmetic problem. The four `concept_math_conditions` are concept alone, math alone, and both with either named first; for the concept+concept arm, `concept_pairs` lists pairs of `key`s and `concept_concept_conditions` covers A alone, B alone, and both orders. A task is reachable if any target token (a `concept_words` member, or the arithmetic answer as digit or number-word) hits lens rank ≤5 anywhere in the band over the response span; interference = single-task − dual-task reachability.
