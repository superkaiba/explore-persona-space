# Issue #2661 — odd-correlation read of the full-dictionary context→answer feature map

Read produced 2026-09-04 ~09:20 UTC on the VM from committed / staged artifacts at issue-2661
`d597af086a0`: W1 context-feature descriptions (`judge_aggregates/descriptions_ctx.json`, 7,435
features, claude-opus-5), the committed #2552 answer-feature descriptions
(`inputs/descriptions_rep_ta.json`, 4,860 features), and the surviving-edge table
`wiring_edges.npz` (HF `issue2661_flatsae/analysis_tensors/edges`). Coefficients are
standardized ridge units (answer-feature response per 1 SD of the context feature). This is a
descriptive correlational read; no causal claim.

## What "odd correlation" was tested

Thomas's headline ask: can the map surface odd behavior correlations "like china refusal"?
Two directions were read with description-text classifiers (not the receipts families, see
caveat 2):

- **Forward**: context features whose description matches a politically sensitive China regex
  (`taiwan|xinjiang|tibet|tiananmen|ccp|communist party|uighur|uyghur|hong kong|xi jinping|
  sovereignty|censorship|prc`) → where do their top-10 out-edges land?
- **Reverse**: answer features whose description is a refusal / deflection (`refus|declin|
  cannot help|unable to|safety response|ethical caveat|...`) → which context features feed them?

## Findings

1. **49 politically sensitive China context features exist, all 49 are in the live edge rows.**
   Examples: 2511 (Tiananmen, Cultural Revolution, Tibet, Taiwan sovereignty questions), 9086
   (does Taiwan / the Diaoyu Islands belong to China), 878 and 8645 (CCP ideology and Xi Jinping
   Thought prompts), 11935 (circumventing Chinese internet censorship), 2718 / 12927 / 16068 /
   25082 (write Party or government official documents), 22125 (20th Party Congress).
2. **Forward landing tally over their 490 top-10 out-edges: 38 refusal-described, 41
   CCP-described, 411 other.** The CCP-described landings are topic-consistent wiring (write a
   Party work report → Party-organizational-work answer features 3152 / 9264 / 11043 / 6990,
   coefs +0.02 to +0.05): expected, not odd. The refusal-described landings come almost entirely
   from meta-questions about the assistant's own restrictions / censorship (21329, 19363) and
   jailbreak-style prompts (7647, 12489, 20500, 23761) → refusal / policy-explanation answer
   features (12887, 30167, 20880, 27425, 26032): also expected.
3. **The one China-politics → hedging edge that is not a topic match:** context 8645 ("politically
   charged / sensitive ideological questions — especially CCP doctrine and Xi Jinping Thought
   study prompts") → answer 27425 ("responses that deflect, redirect, or add ethical caveats to
   sensitive topics"), coef **+0.033**, the second-strongest refusal-side in-edge from any China
   feature (the strongest, +0.034, is from 14066, a generic "ask the AI about its own constraints"
   feature). Context 14940 ("short Chinese-language ideological / political requests") → 27425 at
   +0.018 is the same pattern. Magnitudes are small: the strongest edges in the whole table are
   ~+0.10 (e.g. 18443 psychology questions → 27425).
4. **Reverse: 181 refusal-described answer features receive 21,074 surviving in-edges; 115
   (0.5%) come from the 49 China features.** The refusal features are fed overwhelmingly by
   harmful-request, jailbreak, and explicit-content context features (12887 ← 21000 toxic-content
   requests +0.044, 11904 illicit how-to +0.042, 2701 hacking/piracy +0.042). No refusal feature
   has a China-topic feature among its top-5 feeders.
5. **Sovereignty questions (2511, 9086, 10946) do not land on refusal- or CCP-described answer
   features in their top-10**; their edges go to topical Q&A answer features. So the "china
   refusal" pattern, at the correlational level this map can see, is absent for Taiwan / Tibet /
   Tiananmen *questions* and present only weakly (edge 3) for ideological *prompts*.

## Other odd-correlation reads that did surface

- **Identity questions → Qwen self-identification** (context 10173 / 15984 / 3067 / 6826 / 6943 →
  answer 29234 / 21858 / 14461 / 23639 "denies being GPT", coefs up to +0.024): clean, expected
  wiring, useful as a positive control that the map recovers behavior-level links.
- **Explicit-content requests in Chinese (11093) → Qwen self-introduction (15409, +0.009) alongside
  refusal features (24063, 23620, 10524)**: a self-identification answer feature co-wired with
  refusals, worth a qualitative look at the raw rollouts.
- **Forced-choice loyalty dilemmas in Chinese (10035) → self-harm / distress refusal (13211)
  +0.002**: tiny; not interpretable at this magnitude.

## Caveats

1. Coefficients are correlational ridge weights on SAE activations; a surviving edge means the
   context feature reliably co-varies with the answer feature across 120k rows (split-half
   replicated + sign-consistent + above the per-column shuffle null), nothing more.
2. **The receipts families in `receipts_answer_features.json` are noisy.** The "refusal" family
   includes 18385 (short conversational / evaluative statements), 27474 (short acknowledgement
   turns) and 15921 (creative-writing collaboration), which together carry most of the 1,508
   "receipts-flagged" edges in `watchlist_context_features.json`. Use the description-text
   classifier above, not the family flag, for any refusal claim.
3. Only 4,860 of 32,768 answer features have descriptions (the committed #2552 set), so "other"
   in the forward tally includes undescribed landings.
4. The coarse watchlist regex also matches plain "Chinese-language", which is why it returns 950
   features; the 49-feature politically sensitive subset is the relevant one.

## Provenance

Context descriptions: W1 judge wave, claude-opus-5, max_tokens 512, 7,435 valid of 7,463
(judge_meta_w1.json). Answer descriptions: #2552 `descriptions_rep_ta.json` (branch issue-2552
@ cb39df3ce1c). Edges: `wiring_edges.npz` from the pod run (issue-2661 @ fc0b8e2f72a..f22f6b4e3fc;
1,923 surviving of 20,000 candidates). Dashboard (500 edges, both descriptions in words):
`eval_results/issue_2661/dashboard/issue2661_dashboard.html`.
