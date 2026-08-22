# What is stored at the context vector, does patching it matter, can the mapping predict it — and does any of it survive a change of architecture?

Consolidated report merging issue **#2162** (Qwen2.5-7B-Instruct, + the #2215 mapping sibling) and
issue **#2329** (Qwen3.5-9B, thinking disabled). Both runs use the **same bank text, the same
fraction-of-swap definition, the same judge instrument, and the same statistics** — the model is
the single varied factor — so every result below is reported for both models side by side.

Supersedes `docs/reports/issue_2162_consolidated_report.md` (single-model, #2162 + #2215); that
document's numbers are carried forward here unchanged.

Convention, inherited: every caption and "known read" is factual, read from the committed
artifacts. Interpretation appears only in lines explicitly labeled *Context*. **Thomas writes the
Takeaways** — every `Takeaways (Thomas)` slot is deliberately empty.

## Motivation

We have a fitted context→answer mapping, and we know a lot of information is decodable at the
context vector. Five questions:

1. **What exactly is stored at the context vector?**
2. **Does patching only the context vector causally affect the answer vector — and the output behavior?**
3. **Can our mapping predict the causal effect?**
4. **Can our mapping differentiate the answers of two contexts differing by only one attribute?**
5. **Are the per-type answers a property of the information type, or of the model?** (#2329's registered test.)

## Shared setup (glossary — one line each)

- **Context vector** = the residual-stream activation at the *last prompt token*, per layer.
  **Prefix-end** = the activation at the last token before the user query. **Answer vector** = the
  mean of residual activations over the model's *own* completion tokens, per layer.
- **Mapping** = a per-layer linear (ridge) map from context vector to answer vector — "can the
  answer state be predicted from the context state alone?".
- **Bank** (frozen, seed 2162): **1,404 contexts** = 21 information types (user's name, assistant
  persona, instruction format, language, verbosity, a queried fact, refusal boundary, …) × 12
  carrier conversations × 3 values, plus conflict/recency/load variants → 39 type-cells. Each
  directed minimal pair is token-identical except the one varied attribute.
- **Fraction-of-swap F** = (effect of patching one position) / (effect of switching the entire
  context); 0 = the patch does nothing, 1 = the patch is as good as swapping the whole context.
  Measured on the answer vector (**F_act**) and on judged behavior (**F_beh**).
- **Decoding:** temperature 1.0, K=5 draws per pair per arm (grid); K=10 anchor draws.
- **Grading:** claude-sonnet-4-5 judge, graded 0–100.
- **Nulls:** norm-matched shuffled-donor patch + cross-type-donor patch (causal arms);
  within-carrier label-permutation band (probes); shuffled-pairing / shuffled-map (mapping arms).
- **Held-out split:** leave-one-carrier-out over the 12 carriers — for the probes AND every map.
- **Naming:** prose uses plain-English type names; per-cell axis/row labels in the figures keep the
  bank's cell names (the one sanctioned use of raw cell codes).

### What differs between the two runs

| | **#2162** | **#2329** |
|---|---|---|
| Model | Qwen2.5-7B-Instruct | Qwen3.5-9B, thinking disabled |
| Attention | 28 layers, all full attention | 32 layers; **8 full attention** ({3,7,11,15,19,23,27,31}), 24 GatedDeltaNet linear-attention |
| Hidden size | 3,584 | 4,096 |
| Bank text | original | **byte-verbatim reuse**, re-tokenized under the 248k-vocab tokenizer; 1,404/1,404 pairs survived the token-identity gate, 0 dropped |
| F_act read layer | 26 | 30 (fraction-of-stack remap of 26/28) |
| Units in the verdict grid | 76 (cell × slot) | 75 |

**Depth comparisons use fraction-of-stack, never raw layer index** — a 28-layer stack and a
32-layer hybrid stack are not layer-for-layer comparable. Parent layer 19 sits at depth 0.70;
the matched child layers are 21–22 (depth 0.68–0.71).

**Figure links:** #2162 figures are relative paths. #2329 figures are absolute URLs pinned to
commit `91b22ff`, which dates from when the `issue-2329` branch was still unmerged; PR #2004
landed on 2026-08-19, so those artifacts are now on `main` too and the pinned URLs remain valid
(the commit is an ancestor of `main`).

## Result 0 — Qualitative examples + bank reference (dashboards)

Self-contained HTML dashboards over the raw banked text (no interpretation outside the one labeled
analysis box at the top of each gallery). Per directed pair: context A → answer, context B →
answer, A patched with B's context-end state → patched answer, with per-pair transfer scores;
sortable by best/worst transfer. The bank reference lists all 12 carriers and all 39 cells with
their 3 value strings.

**#2162:**
- https://eps.superkaiba.com/issue2162_result0_gallery.html
- https://eps.superkaiba.com/issue2162_bank_dashboard.html

**#2329:**
- https://github.com/superkaiba/explore-persona-space/blob/91b22ffd0e564665001a423c9ad5ee680e2b03c0/docs/issue2329_result0_gallery.html
- https://github.com/superkaiba/explore-persona-space/blob/91b22ffd0e564665001a423c9ad5ee680e2b03c0/docs/issue2329_bank_dashboard.html

**Takeaways (Thomas):** _

## Result 1 — What is stored at the context vector?

*Can a held-out linear probe classify which value of the varied attribute a context contains, from
the context vector alone?*

**What is plotted:** leave-one-carrier-out probe AUC (macro over the 3 value-pairs) for every
type-cell (rows) × layer (columns), one panel per readout slot (context-end left, prefix-end
right). The #2329 panel marks the 8 full-attention layers.

![#2162 probe AUC per type-cell and layer](../../figures/issue_2162/layer_profile.png)

> **#2162 (Qwen2.5-7B).** Per-cell layer curves with the within-carrier label-permutation 95% band
> (B=1,000): [context-end](../../figures/issue_2162/probe_layer_curves_ce.png) /
> [prefix-end](../../figures/issue_2162/probe_layer_curves_pe.png).

![#2329 probe AUC per type-cell and layer](https://raw.githubusercontent.com/superkaiba/explore-persona-space/91b22ffd0e564665001a423c9ad5ee680e2b03c0/figures/issue_2329/layer_profile.png)

> **#2329 (Qwen3.5-9B).** 39 rows in the context-end panel, 37 in the prefix-end panel; x = depth
> as fraction of the 32-layer stack; dashed verticals = full-attention layers.

**Known read.**

| | #2162 | #2329 |
|---|---|---|
| Cell × slot combinations decoding above the permutation band | **75 of 78** | **72 of 75** |
| Failures | query-content at both slots (max AUC 0.600 ce, 0.521 pe); persona-role-header at prefix-end (0.456) | query-content at both slots (max AUC 0.521 ce and pe); prompted-persona at prefix-end has no probe rows (structural exclusion, below) |
| Probe AUC of the cells that later prove causally usable | 1.0 (all 5) | 1.0 (all 8) |

Near-total decodability on both models, and **query-content is the one cell the probe fails on both**.

**Takeaways (Thomas):** _

## Result 2 — Does patching the context vector move the answer vector?

**What is plotted (#2162):** activation fraction-of-swap F_act (read at layer 26, disjoint
floor-anchor halves) per type-cell at the context-end patch, for the paired-donor patch and both
null patches; pair-clustered bootstrap 95% CIs (B=10,000).

![#2162 F_act per type at context-end](../../figures/issue_2162/mapshift/fig_f_act_by_type_ce.png)

> Blue = paired donor's context-end state; grey = norm-matched shuffled-donor null; orange =
> cross-type-donor null.

![#2329 F_act vs F_beh agreement](https://raw.githubusercontent.com/superkaiba/explore-persona-space/91b22ffd0e564665001a423c9ad5ee680e2b03c0/figures/issue_2329/act_beh_agreement.png)

> **#2329** did not render a standalone per-type F_act bar view; its F_act read appears as the
> agreement scatter against F_beh (mean F_act at layer 30 vs mean F_beh per cell × slot × arm).

**Known read.**

- **#2162:** the 5 behaviorally-usable cells have steered F_act 0.36–0.44 against a shuffled-donor
  null of 0.03–0.16. Several behaviorally-*unusable* cells also move the answer vector well above
  their nulls — refusal boundary 0.41 vs 0.25, verbosity 0.42 vs 0.19, in-context task mapping 0.84
  vs 0.53 (n=7). The persona-role-header bar rests on n=1 with nulls as high as the steered arm.
- **#2329:** cell-level Spearman ρ(F_act, F_beh) = **0.747** on the steered arm (0.529 shuffled,
  0.599 cross-type; n = 74–75 screened cells per arm) — against the parent's **0.769**. This is the
  statistic that licensed using the judge-free F_act as the stage-2 selection criterion.

**Takeaways (Thomas):** _

## Result 3 — Does patching the context vector move output behavior?

**What is plotted (first pair of figures):** behavior fraction-of-swap F_beh per type-cell for the
three patch arms, one panel per slot, pair-clustered bootstrap 95% CIs (B=10,000), anchor-separation
exclusion |ceiling − floor| ≥ 0.5. **(Second pair):** each cell × slot placed by probe max-AUC (x)
vs steered F_beh (y) — the stored × used verdict grid.

![#2162 F_beh per type](../../figures/issue_2162/hero_ftype.png)

> **#2162.** Per-pair companion: [hero_ftype_perpair](../../figures/issue_2162/hero_ftype_perpair.png).

![#2329 F_beh per type](https://raw.githubusercontent.com/superkaiba/explore-persona-space/91b22ffd0e564665001a423c9ad5ee680e2b03c0/figures/issue_2329/hero_ftype.png)

![#2162 stored vs used 2x2](../../figures/issue_2162/two_by_two.png)

![#2329 stored vs used 2x2](https://raw.githubusercontent.com/superkaiba/explore-persona-space/91b22ffd0e564665001a423c9ad5ee680e2b03c0/figures/issue_2329/two_by_two.png)

**Known read — the verdict grid.**

| Verdict | #2162 (76 units) | #2329 (75 units) |
|---|---|---|
| **stored-and-used** (probe decodes AND patch moves behavior) | **5** | **8** |
| **stored-but-unusable** (probe decodes, patch moves nothing) | 55 | 58 |
| **used-but-not-decoded** | **0** | **0** |
| **absent** (neither) | 2 — query-content at both slots | 2 — query-content at both slots |
| **untestable** (anchor separation too weak, post-exclusion n < 12) | 14 | 7 |

**The causally usable cells, in full.** Every one is at the **context-end** slot on both models;
every one has probe AUC 1.0.

| #2162 (Qwen2.5-7B) | F_beh | n | | #2329 (Qwen3.5-9B) | F_beh | n |
|---|---|---|---|---|---|---|
| instruction-format under load 5 | 0.807 | 35 | | format conflict (forward) | 0.838 | 25 |
| instruction-format under load 3 | 0.761 | 35 | | **instructed language** | 0.699 | 35 |
| format conflict (forward) | 0.723 | 33 | | **implied language** | 0.695 | 36 |
| plain instruction-format | 0.707 | 36 | | format conflict (reverse) | 0.644 | 25 |
| format conflict (reverse) | 0.703 | 33 | | plain instruction-format | 0.527 | 33 |
| | | | | **instruction-format at recency depth 3** | 0.480 | 33 |
| | | | | instruction-format under load 5 | 0.470 | 34 |
| | | | | instruction-format under load 3 | 0.368 | 34 |

The three cells in bold are **new on Qwen3.5** — each was a `null` causal verdict on the parent and
a `positive` here: instructed language, implied language, and instruction-format at recency depth 3.
No cell went the other way: **every parent-positive cell stayed positive.**

Two #2329 cells score higher than anything in either list but fall under the n ≥ 12 testability
floor and are therefore *untestable*, not positive: in-context task mapping (F = 1.292, n = 6) and
persona role-header (F = 0.982, n = 2).

**Takeaways (Thomas):** _

## Result 4 — Does the mapping predict the causal shift?

*For each minimal pair, does the map-predicted shift — mapping of B's context vector minus mapping
of A's — point where the patched answer vector actually moved?*

**What is plotted:** cosine between the map-predicted answer-state shift and the realized patched
shift, per layer, disjoint anchor halves. Map sources: a map fit on this bank's own anchor states
(fresh, leave-one-carrier-out), the raw context shift with no map (identity; identity+bias ≡
identity in shift space, the bias cancels in differences), and — #2162 only — the banked #779
single-turn and #1738 multi-turn maps.

![#2162 shift-prediction cosine by layer](../../figures/issue_2162/mapshift/fig_shift_cosine_by_layer.png)

> **#2162.** Mean cosine over cells vs layer; left panel = the 5 stored-and-used cells, right = all
> 39. Grey band = the same read on the null patch arms.
> Per-type heatmap: [fig_shift_cosine_heatmap](../../figures/issue_2162/mapshift/fig_shift_cosine_heatmap.png).

![#2329 predicted vs realized patched shift](https://raw.githubusercontent.com/superkaiba/explore-persona-space/91b22ffd0e564665001a423c9ad5ee680e2b03c0/figures/issue_2329/mapshift/mapshift_shift_prediction.png)

> **#2329.** Left panel = the 8 stored-and-used cells, right = all 39. The banked parent maps are
> **not** plotted: they are 3,584-dimensional and cannot be applied at 4,096.

**Known read — survivor cells, fresh map, steered arm.**

| | #2162 @ L19 (depth 0.70) | #2329 @ L19 (0.61) | #2329 @ L21 (0.68) | #2329 @ L30 (0.97) |
|---|---|---|---|---|
| Fresh bank-fit map | **0.349** | 0.255 | **0.306** | 0.317 |
| Raw context shift, no map | 0.137 | 0.129 | 0.172 | 0.146 |
| Shuffled-donor patch null | 0.063 | 0.092 | 0.115 | 0.130 |
| Cross-type-donor patch null | 0.106 | 0.087 | 0.103 | 0.106 |
| Survivor cells in the panel | 5 | 8 | 8 | 8 |

At matched depth the map beats its patch-arm nulls by roughly 3× on both models, and the fitted map
roughly doubles the raw context shift on both. #2162 also has banked-map arms at L19: single-turn
0.276, multi-turn 0.289.

**The binding caveat, and it replicates.** The **shuffled-pair null** — a carrier-blocked derangement
of the predicted↔realized pairing, i.e. what a *type-generic* direction achieves with no pair
identity at all — sits just under the observed values on both models:

- **#2162:** band ≈ 0.25–0.36 per survivor cell; observed 0.34–0.35; **4 of 5 cells above the upper
  edge** (the plain instruction-format cell inside it).
- **#2329:** **4 of 8** cells above their band at L19, **5 of 8** at L21, **6 of 8** at L30. The two
  language cells clear it with room (L21: implied language 0.554 vs band top 0.515; instructed
  language 0.587 vs 0.533); the instruction-format family sits at or just above its edge.

Companion reads (#2162): predicted-shift magnitude ≈1.6× realized (median ratio); shift-space R²
negative everywhere (fresh −1.56 at L19); the realized patched shift has cosine 0.55 to the
full-swap ceiling direction. The shuffled-*map* null (refit on permuted pairing) spans −0.02 to
+0.03 on #2162.

*Context (prior grain):* #2094 banked-map transport cosine ≤ 0.16; #1415 cosine ≈ 0; #1776 (HIGH)
"the map reads a correlate, not a cause".

**Takeaways (Thomas):** _

## Result 5 — Can the mapping differentiate the two answers?

*For two contexts differing by one attribute: does the map's predicted answer land closer to the
true context's real answer than to the sibling's? Paired 2AFC (two-alternative forced choice),
chance = 0.5.*

**What is plotted:** pooled paired-2AFC accuracy (cosine metric, span pooling) vs layer for the
fresh bank-fit map, identity + learned bias, and identity only; carrier-clustered 95% CIs; the
shuffled-pair null band (≈0.48–0.52).

![#2162 paired 2AFC accuracy by layer](../../figures/issue_2162/mapshift/fig_2afc_by_layer.png)

![#2329 paired 2AFC accuracy by layer](https://raw.githubusercontent.com/superkaiba/explore-persona-space/91b22ffd0e564665001a423c9ad5ee680e2b03c0/figures/issue_2329/mapshift/dv3_2afc.png)

**Known read at matched depth.** The parent's headline layer is 19 (depth 0.70); the child's matched
layers are 19–21.

| | #2162 @ L19 (0.70) | #2329 @ L19 (0.61) | #2329 @ L21 (0.68) |
|---|---|---|---|
| Fresh bank-fit map | **0.799** (CI 0.780–0.817) | **0.795** (CI 0.776–0.813) | **0.793** (CI 0.775–0.811) |
| Identity + learned bias | 0.756 | 0.743 | 0.745 |
| Identity only | 0.704 | 0.702 | 0.714 |
| Fresh − identity+bias | **+4.3 pts** (CI +2.7 to +5.9) | **+5.1 pts** (CI +3.5 to +6.7) | **+4.8 pts** (CI +3.2 to +6.3) |

A near-exact replication across architectures. Two riders:

- **At the very top of the #2329 stack the advantage disappears:** layer 31 gives fresh 0.848,
  identity+bias 0.837, identity 0.801 — a +1.2-pt gap whose CI [−0.4, +2.7] spans zero
  (*inconclusive*). Pooled across all 128 layer × metric configurations the map beats identity+bias
  in 52, loses in 61, and is inconclusive in 15 — that split is dominated by the saturating
  top-of-stack layers, not the matched-depth read above.
- **#2215 (banked maps, #2162's sibling):** the banked #779 map beat identity+bias by only +0.8 pts
  at L19 (CI −0.5 to +2.2, inconclusive) — the finding in that task's own title. Freshly fitting the
  map on this bank is what produces the significant margin. The recomputed banked arm reproduces the
  committed #2215 value exactly (0.7639 at L19), which is the parity anchor for the whole battery.

**Companion reads (standing mapping-baselines rule — the reads dissociate).**

| Fresh map | #2162 @ L19 | #2329 @ L19 |
|---|---|---|
| Held-out R², per-draw grain | 0.272 | 0.195 |
| Held-out R², span-pooled | 0.411 | 0.335 |
| Identity+bias R² (per-draw) | −1.38 | ≈ −1 to −3 across the stack |
| Identity-only R² (per-draw) | −3.79 | ≈ −5 to −27 across the stack |
| kNN retrieval, accuracy@1 (chance 0.0007) | 0.187 | 0.147 |
| kNN retrieval, accuracy@10 (chance 0.007) | 0.718 | 0.595 |
| Median retrieval rank among 1,404 | 5 | 7 |

On #2329 the ridge map's held-out R² is **negative below roughly the middle of the stack**
(−1.15 at layer 0, −3.69 at layer 3), first turns positive around layer 18, and peaks at 0.31
per-draw (layer 18) / 0.43 span-pooled (layer 31). It nonetheless beats identity+bias at every
layer, because that baseline is far more negative.

![#2329 per-layer map R² with baselines](https://raw.githubusercontent.com/superkaiba/explore-persona-space/91b22ffd0e564665001a423c9ad5ee680e2b03c0/figures/issue_2329/mapshift/mapshift_r2.png)

> Both fits are well-posed: n_train ≈ 12,870 rows ≫ d = 4,096, dof cap non-binding.

**Takeaways (Thomas):** _

## Result 6 — Does any of this transfer across architecture? (#2329's registered test)

*This result has no #2162 counterpart — it is the question #2329 was built to answer.*

**What is plotted:** one point per P1-family type-cell × slot surviving exclusion in **both** runs
(31 shared units): x = the parent's steered mean F_beh (Qwen2.5-7B), y = this run's (Qwen3.5-9B),
one panel per slot, identity line drawn. The registered statistic is Spearman ρ with a
pair-clustered bootstrap 95% CI — one test outside the three Holm families, α = 0.05.

![transfer scatter](https://raw.githubusercontent.com/superkaiba/explore-persona-space/91b22ffd0e564665001a423c9ad5ee680e2b03c0/figures/issue_2329/transfer_scatter.png)

![transfer verdict heatmap](https://raw.githubusercontent.com/superkaiba/explore-persona-space/91b22ffd0e564665001a423c9ad5ee680e2b03c0/figures/issue_2329/transfer_verdicts.png)

**Known read.**

- **Spearman ρ = 0.831, p = 7.4e-09, pair-clustered 95% CI [0.583, 0.864]**, n = 31 shared cells.
  The CI excludes 0 — the registered H1 (the per-type ordering is a property of the information
  type) rather than H2 (architecture-dependence).
- **Verdict-level transfer over 75 shared cell × slot units:** positive→positive 5, null→positive 3,
  null→null 52, untestable→null 8, null→untestable 2, untestable→untestable 5. **No parent-positive
  cell became null.** The three gains are named in Result 3.
- **Realized Holm family sizes** (child vs parent): P1 role/type 28 vs 25 (ceiling 31); P2 route
  12 vs 10 (15); P3 dose/position 27 vs 26 (28). The child had slightly more testable cells in every
  family, which is the mechanical reason its untestable count fell from 14 to 7.

**Takeaways (Thomas):** _

## #2162-only extensions (not rerun on Qwen3.5)

Two follow-up rounds ran on #2162 and have no #2329 counterpart. They are single-model results;
nothing in this document's cross-model comparison rests on them.

- **Persona-specificity ladder** — a 7-value ladder of persona system prompts (plain default
  assistant → pirate captain → Victorian butler → …), ranked R1–R5, patched at both slots, asking
  how the causal read varies with how *specific* the persona is. Includes an erase-vs-install
  asymmetry read and a rubric-artifact quantification (netted dual-rubric F vs target-descriptor-only
  F) that motivated the target-only instrument for persona cells. Result sections in the #2162 body.
- **Turn-boundary multipatch** — the parent measured single final-turn positions only; this round
  adds a joint patch at *every* assistant-turn boundary plus a per-boundary sweep, over 7 banked
  cells, to separate "the trace weakens with depth" from "the trace spreads across per-turn
  boundaries so any single position carries only a fraction". 45 blocks, 8,100 rollouts, 24,840
  judge calls; includes a depth-1 identity gate proving the multi-position code path reproduces the
  parent's single-position read where the two are definitionally the same patch.

Full detail: the #2162 task body (`tasks/awaiting_promotion/2162/body.md`, result sections from
"Specificity ladder" onward) and `docs/reports/issue_2162_detailed.md`.

## What does not compare cleanly

Read every cross-model number above with these in view. All are recorded in the #2329 plan as
pre-registered scope caveats.

1. **Prefix-end coverage is structurally different.** The Qwen3.5 thinking-off template inserts no
   default system turn, so a bare single-turn context has no prefix token at all. All 36
   persona-role-header contexts and the 12 empty-system prompted-persona contexts are flagged
   `no_prefix` and their prefix-end slot is **excluded by construction**. A missing prefix-end cell
   on #2329 is a template fact, not a measurement outcome. (Both models' causal positives are
   context-end anyway, so no headline rests on this.)
2. **Per-layer reads are not layer-for-layer comparable** — 28 full-attention layers vs a 32-layer
   hybrid with only 8 full-attention layers. Everything depth-indexed uses fraction-of-stack.
3. **Single-position patching means something different under linear attention.** 24 of the 32
   Qwen3.5 layers carry position information through a compressed recurrent state rather than an
   attendable KV cache. #2329 is a replication *with an architecture change*, not a clean
   replication.
4. **Frozen padding and translation text was generated by the parent model** (Qwen2.5-7B) and reused
   byte-verbatim, so it is off-policy text for Qwen3.5 on the recency and language cells — identical
   on both pair sides by construction.
5. **Banked parent maps could not be carried over** (3,584-dim vs 4,096), so #2329's mapping results
   use only freshly fitted maps. The banked-map arms in Result 4 and Result 5 are #2162-only.
6. **#2329's model revision was never pinned.** Every load resolved `Qwen/Qwen3.5-9B` at `main`; two
   local caches resolve that to `c2022362…`, but the generating pod is gone, so the pod-side
   revision is not provable from the artifacts. Recorded as a reproducibility gap.
7. **One planned #2329 metric was never computed** — the conflict balance shift
   ((judge_demonstrated − judge_instructed)/100). No code for it exists; the conflict cells are
   rendered as steered F_beh bars instead. Declared not-produced rather than silently dropped.
8. **#2329's stage-2 layer × dose profile is post-selection** (survivors chosen on the judge-free
   F_act, forced by the all-generation-before-judging pipeline order) and is labeled exploratory;
   the confirmatory families exclude it.

## Reproducibility

**#2162** — compute 0 GPU-h for the mapshift round (all inputs banked); walls: fresh fits 2,874 s,
shift battery 166 s, 2AFC extension 1,065 s. Code: `scripts/issue2162_mapshift.py` @ `2ed3708b17`;
figures `scripts/issue2162_mapshift_figs.py` + `scripts/issue2162_report_figs.py` @ `446b1fc500`;
dashboards `scripts/issue2162_dashboards.py` @ `86fdd457e6`; banked figures
`scripts/issue2162_figures.py`. Data: HF `superkaiba1/explore-persona-space-data` pinned to
revision `7d3ac543a5a4202e3996be1498886f2bab637c15`; outputs under
`eval_results/issue_2162/{mapshift,f_metrics}/` and `eval_results/issue_2215/`.

**#2329** — one 8× H100 RunPod pod for all generation (bank capture → anchors → grid → stage-2)
through a work-conserving claim-file block queue; judging off-pod via the Anthropic Batch API
(≈203k net calls, claude-sonnet-4-5-20250929); analysis on a `cpu-bigmem` pod; figures on the VM.
Planned rollouts: 42,120 grid + 14,040 anchors + ≤12,096 stage-2. Code:
`scripts/issue2329_run.py`, `issue2329_analysis.py`, `issue2329_mapshift.py` @ `653ff2b487`.
Data: HF `superkaiba1/explore-persona-space-data`, prefix `issue2329_q35rerun/`; outputs under
`eval_results/issue_2329/`. PR #2004 (the `issue-2329` branch, 51 commits) landed on `main`
2026-08-19; figure links above stay pinned to `91b22ff` for stability.

**Provenance.** #2162 originating ask: plan a 5-result experiment on what is stored at the context
vector, whether patching it moves the answer vector and behavior, whether the mapping predicts the
causal shift, and whether the mapping differentiates minimal-pair answers. #2329 originating ask,
verbatim: *"okay. Rerun with qwen3.5-9B. make the qualitative dashboards after all the generation
finishes and then run judging in parallel (following: how long would it take to rerun all this on
qwen 3.5 9b? with thinking DISABLED)"*.

Both tasks are parked at `awaiting_promotion` with their claim sections awaiting Thomas.
