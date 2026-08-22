# What is stored at the context vector, does patching it matter, and can the mapping predict it?

Consolidation plan on issue **#2162** (+ #2215), run as a same-issue inline follow-up (`mapshift`).
Everything below is either a **[BANKED — replot]** of existing artifacts or a **[NEW — 0 GPU-h]**
analysis on banked tensors. Thomas writes the Takeaways per result; agents fill the plots.

## Motivation

We have a fitted context→answer mapping, and we know a lot of information is decodable at the
context vector. Four questions:

1. **What exactly is stored at the context vector?**
2. **Does patching only the context vector causally affect the answer vector — and the output behavior?**
3. **Can our mapping predict the causal effect?**
4. **Can our mapping differentiate the answers of two contexts differing by only one attribute?**

## Shared setup (glossary inline — one line each)

- **Model:** Qwen-2.5-7B-Instruct.
- **Context vector v_C** = residual-stream activation at the *last prompt token* (the final newline
  of the assistant header), per layer. **Prefix-end** = activation at the last token before the
  user query. **Answer vector v_A** = mean of residual activations over the model's *own*
  completion tokens, per layer.
- **Bank** (#2162, frozen, HF `issue2162_ctxinfo`, seed 2162): **1,404 contexts** = 21 information
  types (user's name, assistant persona, instruction format, language, verbosity, a queried fact,
  refusal boundary, …) × 12 carrier conversations × 3 values, plus conflict/recency/load variants
  → 39 type-cells. Each directed minimal pair is token-identical except the one varied attribute.
- **Fraction-of-swap F** = (effect of patching one position) / (effect of switching the entire
  context); 0 = patch does nothing, 1 = patch is as good as swapping the whole context. Measured
  at activation level (**F_act**) and judged-behavior level (**F_beh**).
- **Decoding:** temperature 1.0, K=5 draws per pair per arm (grid); K=10 anchor draws. Banked.
- **Grading:** claude-sonnet-4-5 judge, graded 0–100. Banked scores; no new judge spend.
- **Nulls:** norm-matched shuffled-donor patch + cross-type-donor patch (causal arms);
  within-carrier label-permutation band (probes); shuffled-pairing / shuffled-map (mapping arms).
- **Held-out split:** leave-one-carrier-out over the 12 carriers — used for the probes AND for
  every freshly fitted map.

## Bank dashboard [NEW — 0 GPU-h]

Minimal reference dashboard (`docs/issue2162_bank_dashboard.html`, self-contained HTML), **no
interpretation anywhere**:

- All 12 carrier contexts (full text, collapsed/expandable), with the note that each carrier is
  the held-out fold exactly once under leave-one-carrier-out.
- All 39 parameter cells with their 3 actual value strings (the varied spans).
- Per parameter, ONE worked example: context with value A (varied span highlighted) → its
  unpatched anchor answer; same carrier with value B → its anchor answer (= the answer change the
  parameter induces). Raw text only; truncation, where used, disclosed inline.

## Result 0 — Qualitative examples [NEW — 0 GPU-h]

Very minimal gallery (`docs/issue2162_result0_gallery.html`), organized by parameter change, built
from banked rollout text + banked per-pair transfer scores (`f_cells.jsonl` carries per-pair
F_beh AND F_act at context-end):

- Per directed pair: context A → answer; context B (parameter changed) → answer; A patched with
  B's context-end state → patched answer + transfer scores (F_beh behavior, F_act answer-vector);
  the reverse direction wherever the bank banked it (pair directions as realized in `bank.json`;
  absent directions marked factually, never imputed).
- Sections sortable by best/worst transfer; pairs within a section sortable the same way.
- One collapsible section at the top: **Fable 5 analysis of what can and can't transfer** —
  explicitly labeled as interpretation; everything below it is raw data.

**Takeaways (Thomas):** _

## Result 1 — What is stored at the context vector? [BANKED — replot]

*Can a held-out linear probe classify which value of the varied attribute a context contains,
from v_C alone?*

- **Plot:** per-type held-out probe AUC (max over 28 layers), context-end vs prefix-end, with the
  permutation null band. Train vs held-out carriers both shown.
- **Data:** `eval_results/issue_2162/f_metrics/probe.json` (+ permutation matrix on HF).
- **Known read (artifact-verified 2026-08-16):** 75/78 type×slot cells decode above the band; all
  5 causal positives at AUC 1.0; failures = `query_content` at BOTH slots (0.600 context-end /
  0.521 prefix-end) and `persona_role_header` at prefix-end (0.456).

**Takeaways (Thomas):** _

## Result 2 — Does patching the context vector causally affect the answer vector? [BANKED — replot]

- **Plot:** F_act per type at context-end vs both nulls, pair-clustered bootstrap CIs.
- **Data:** `eval_results/issue_2162/f_metrics/{f_cells,null_shuffled_cells,null_crosstype_cells}.jsonl`,
  `stats.json`.

**Takeaways (Thomas):** _

## Result 2.5 — Does our mapping predict the causal shift? [NEW — 0 GPU-h]

*For each minimal pair, does the map-predicted shift M(v_C^B) − M(v_C^A) point where the patched
answer vector actually moved?*

- **Plot:** cosine(predicted shift, actual shift) heatmap over type × layer, one panel per map
  source, with shuffled-map null band; companions: magnitude ratio and shift-space R²;
  survivor-cells headline view + all-cells view.
- **Map sources:** (a) banked single-turn context-end map (#779/#1739 fits), (b) banked multi-turn
  map (#1738) if loadable, (c) fresh map fit on the bank's own contexts (carrier-held-out).
- **Actual shift:** v̄_A(patched) − v̄_A(floor anchors), per layer, from the banked per-layer
  activation store; full-context-swap shift as ceiling reference (disjoint anchor halves wherever
  one floor enters two compared quantities — the #1415 shared-baseline fix).
- **Prior grain to beat:** #2094 banked-map transport cosine ≤ 0.16; #1415 cosine ≈ 0; #1776
  (HIGH) "the map reads a correlate, not a cause". The new cell is per-type — especially whether
  the map predicts the shift for the 5 types that ARE causally usable.

**Takeaways (Thomas):** _

## Result 3 — Does patching the context vector causally affect output behavior? [BANKED — replot]

- **Plot:** F_beh per type at context-end vs both nulls; plus the 2×2 stored×used verdict table.
- **Data:** `eval_results/issue_2162/f_metrics/two_by_two.json`, `stats.json`.
- **Known read:** **5 stored-and-used** (all instruction-format-flavored, F 0.70–0.81),
  **55 stored-but-unusable** (probe decodes at AUC ~1.0, patch moves nothing), 2 absent,
  14 untestable (anchor separation too weak).

**Takeaways (Thomas):** _

## Result 4 — Can the mapping differentiate the answers? [BANKED + small NEW extension]

*For two contexts differing by one attribute: does the map's predicted answer land closer to the
true context's real answer than to the sibling's (paired 2-alternative retrieval)?*

- **Banked (#2215):** per-type, per-layer 2AFC accuracy; arms = single-turn context-end map,
  multi-turn prefix-end, multi-turn context-end, identity+bias; 0.52–0.77 over its 30 configs vs
  shuffled null 0.48–0.52 (artifact-verified 2026-08-16); **identity+bias captures most of it**
  (fitted map +0.9 pts, CI incl. 0, at context-end).
- **Extension [NEW, same run as 2.5]:** add the fresh bank-fit map arm (every layer) and an
  identity-only arm (v̂_A = v_C, no bias).
- **Data:** `eval_results/issue_2215/dv3_map_discrimination.json` + new `mapshift` outputs.

**Takeaways (Thomas):** _

## What actually runs (everything else is a replot)

1. Stage banked tensors (~9 GB, scoped, HF revision pinned — #2321 is repacking the data repo)
   → `/mnt/eps-data/thomasjiralerspong/issue2162_mapshift/`.
2. Fresh per-layer ridge maps on the bank's anchor answer states: 28 layers × 12 leave-one-carrier-out
   folds, per-draw rows (n_train ≈ 12.9k > d = 3,584 — well-posed), #825-guarded fit cores,
   primal layer-batched; identity+bias + kNN retrieval reported beside every fitted read.
3. Result 2.5 cosine/magnitude battery + Result 4 extension (CPU, vectorized, pair-clustered
   bootstrap B=10,000).
4. Replots of Results 1/2/3 + report assembly.
5. The two dashboards (bank reference + Result 0 gallery): banked rollout text (grid + anchors
   jsonl) joined to banked per-pair F scores; pure text processing, no model calls.

**Total: 0 GPU-h.** Estimated wall ~2–4 h (fits dominate; pilot-sized before the full battery).
Outputs: `eval_results/issue_2162/mapshift/`, `figures/issue_2162/mapshift/`.

## Decisions log

- **Decoding: stochastic** (temp 1.0, K=5) — inherited from the banked #2162 rollouts; #2094's
  greedy grid needed a temp-1.0 re-sample confirmation anyway, so deterministic buys nothing here.
- **What varies:** the existing 21-type minimal-pair bank — reused, no new bank.
- **Grading:** banked Sonnet judge scores; F as banked (netted dual-rubric); the ladder round's
  target-only F_target noted in captions where the netting distinction matters.
- **Maps for Result 2.5:** banked + fresh-fit, both (Thomas, 2026-08-16).
- **Per-layer patched answer states:** contingency NOT needed — the #2162 store banks all 28
  layers for every arm (only layer 26 was read in the original analysis).
