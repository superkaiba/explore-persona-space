# Interim results — Behavior prediction through the context→answer map (#1739)

> **Status (2026-07-31): INTERIM.** All three behavior lanes (evil, sycophancy, hallucination)
> now have committed arm results. Figures and numbers below are computed directly from the
> committed per-cell results (`eval_results/issue_1739/`) plus the HF-resident WildChat-rung
> results, before the analyzer/critic pass — treat them as provisional.
> Task: <https://eps.superkaiba.com/tasks/1739>. Full pre-registered plan:
> [`docs/map_behavior_prediction_plan.md`](map_behavior_prediction_plan.md).
>
> **New in this update (2026-07-31)** — five result families, all in
> [§ New in this update](#new-in-this-update-2026-07-31):
> (1) the **bare-query round** — stripping the prefix and showing the predictor only the user
> query *raises* sycophancy prediction, *flips the sign* on hallucination, and *destroys* evil,
> which localizes each behavior's predictive variance (it also carries an **unresolved
> null-arm anomaly** — read that flag before using its numbers); (2) a **fourth evaluation
> column**, random held-out WildChat, where the map arm is the best feasible method on evil and
> hallucination; (3) **nonlinear maps (MLP, kernel ridge) lose to the linear map** on every
> behavior — the round-B question answered negatively; (4) the **evil composition-factor gain
> does not replicate** on sycophancy or hallucination for the projection arm — it reverses, which
> corrects takeaway 7 below; (5) the **naturalistic persona-vector regime ladder** extended to
> sycophancy (E2p on top, as for evil) and hallucination (sign-unstable — construct failure).
> Everything in `## Results` below is unchanged and is **context, previously reported**. The
> sycophancy main lane's own full fold is still pending; its numbers appear here only where they
> serve these families.

## Motivation

- Persona vectors predicts behavior from the context vector by projecting the persona vector on the context vector and shows that it can pretty reliably predict whether the answer will exhibit a certain behavior
- This is in some sense a datatype mismatch: persona vectors are extracted from means over answer activations, then compared to the context vector
- Something that makes sense instead is applying our mapping to the context vector and then projecting the persona vector onto the predicted answer vector
    - this allows us to take advantage of unsupervised context -> answer pairs (persona vectors extraction only uses supervised data)
- Probably something that makes even more sense is just training a direct regression from context behavior to behavioral expression, but this experiment is more to **show that our mapping learns useful/meaningful structure** (beyond what is already directly predictable from context) more than to demonstrate an actual application

## Methodology

- Inspired by persona vectors, we look at 3 realistic/important behaviors:
    - sycophancy
    - hallucination
    - evil
- We extract persona vectors using the methodology from the paper, that is taking mean mass probes over answers either exhibiting or not exhibiting the behavior
- We train all regressions on contexts (and answers when appropriate, or sampled answers when appropriate) from WILDCHAT
- We run the following methods:
    - Direct from context methods:
        - Direct linear mapping from context vector to behavior expression
        - Direct nonlinear mapping from context vector to behavior expression
        - Project persona directly on context vector
    - Linear mapping methods:
        - Linear mapping -> project persona vector
        - Linear mapping -> apply regression trained **on predicted answer vectors**
        - Linear mapping -> apply regression trained **on real answer vectors**
    - Upperbounds:
        - Project persona vector on **actual answer vector**
        - Train regression directly from **actual answer vector** (upper bound)
    - For all methods we run scaling curves on both the unlabeled map-training pairs and the labeled behavior examples
    - We run both prefix (prefix end state) -> answer versions and single context -> answer versions of all experiments
- For evaluation, we show results per behavior on a held-out split of the training distribution plus two real out-of-distribution sets
- We also measure spread of behavior expression in the evaluation settings to make sure there is sufficient spread
- We report: spread per evaluation setting; Spearman ρ of predicted vs LLM-judged expression, one bar per method, per behavior per setting; scatter plots per method; scaling curves

### What actually ran (deltas from the original sketch)

- **Behaviors so far:** evil + hallucination complete; **sycophancy still running** (results to be appended).
- **Unlabeled map pool:** 18,793 real WildChat context→answer pairs (the #1092 activation store), swept 250 → 5,000 → 18,793 ("full") — not 50k; the map never sees behavior-eliciting data or any eval set.
- **Labeled behavior data is real, per behavior** (not WildChat): evil trains on in-the-wild jailbreak prompts (1,405 DAN-style prefixes × 390 forbidden questions) and evaluates OOD on hh-rlhf red-team dialogues and ToxicChat jailbreaks; hallucination trains on TriviaQA and evaluates OOD on NQ-Open and SimpleQA. The synthetic persona-vectors elicitation suite was deliberately **dropped** as an eval setting (it conflates natural elicitation with artificial prompting — exactly the concern in the Motivation); the three settings per behavior are held-out-train + two real OOD sets.
- **Nonlinear-map arms were not run as separate arms**; nonlinearity enters via the direct MLP arm and a map-pretrain→fine-tune arm. Extra arms beyond the sketch: identity+bias projection, a stacked combiner, shuffled-map / shuffled-pretrain controls, and text-embedding / surface-feature baselines (16 arms total).
- **DV:** on-policy expression per context from K=5 sampled answers, graded by `claude-sonnet-4-5-20250929` (3 draws @ temp 1.0). Evil: mean 0–100 trait score (persona-vectors rubric). Hallucination: fabrication rate 0–1 under a three-way fabricated/abstained/correct rubric.
- **Persona-vector extraction regimes (evil only):** E1 paper-faithful synthetic; E2 matched-pair natural; E2p pooled natural.
- Spearman ρ is computed at each arm's frozen selected layer; error bars are SD over 3 seeds × 5 label draws. Selection-corrected max-over-arms permutation nulls are significant (p < 0.05) in 100% of cells for both behaviors, and the evil split-half reliability ceiling averages ≈ 0.89.

## New in this update (2026-07-31)

Arm vocabulary, for the sections below: **map → PV proj.** applies our context→answer map to the
input state and projects the persona vector on the *predicted* answer state (the method this
experiment exists to test); **PV proj. on context** is Persona Vectors' own method (project the
persona vector directly on the context state); **direct ridge** regresses expression on the
context state from labels; **oracle** projects the persona vector on the model's *true* answer
state (an upper bound that needs the answer already generated); **shuffled map** is the
nonsense-map control. Two input states recur throughout: the pre-query **prefix end state** and
the full **context end state** (context = prefix + user query). Every section below reports both,
with one stated exception — the bare-query round, whose prefix variant is a degenerate constant by
construction and is therefore reported as a null probe rather than an arm sweep.

### Does the bare user query predict the behavior? (new — bare-query round)

If the map arm predicts behavior from the full context, how much of that is the *query* versus
the *prefix* (the persona, jailbreak framing, or conversation history that precedes it)? This
round answers that by changing **only the predictor's input render**.

Design, stated exactly. The **behavior labels are unchanged** — the same WildChat-rung judged DV
described in the fourth-column section below, scored on rollouts the model generated under the **full
context**. Only the predictor side changes: the final user query is re-rendered **bare**
(chat-template head + query, prefix stripped) and activations are captured at that render. No new
generation and no new judging (`judge_called: false` in `bareq_score_done.json`), so the DV, the
eval set and the frozen layers are held fixed and the render is the single manipulated variable.

The manipulation is only well-posed where a real prefix existed, and the artifacts record which
case each behavior is in (`meta.render_match`, whose `agrees_with_expected` is `true` for all
three):

- **Sycophancy and hallucination corpora are *already* bare.** Their train contexts carry a
  constant template head on every row — measured over all 16,000 train rows, per-layer cosine
  ≥ 0.9995 (sycophancy) and ≥ 0.9991 (hallucination). So their committed ridge-on-context maps
  *are already* bare-query maps, and a dedicated bare fit would be "the identical fit on the
  identical inputs" — recorded as an explicit `leg2_noop`. Only the **eval-side render** changes
  for these two; the leg is render-**matched**.
- **Evil's train pool is prefix-crossed** (390 unique queries spread over 8,000 contexts, ≈20.5
  contexts per query; measured train-prefix cosine falls to 0.657 early / 0.049 flat over 6,468
  rows — decisively not constant), so it needed a **dedicated bare-fit leg 2** with by-query group
  folds (`group-roundrobin-k5`, 390 queries, no query straddling a fold, 1,140–1,351 rows per
  fold). Evil's leg 1 — the context-trained arms applied to bare reps — is render-**mismatched by
  construction** and is shown only as the intermediate.

What is plotted: Spearman ρ per arm against the judged DV, one panel per behavior, one bar per
render condition; error bars are the 95% bootstrap CI over contexts from each row's `ci_frozen`,
drawn as non-negative offsets. **Colour encodes the render condition only** (purple = full
context, teal = bare query with train-fit arms, brown = evil's dedicated bare fit) — deliberately
a different palette from the map-family and regime figures elsewhere in this writeup. Coverage per
behavior: roughly half the eval contexts took a fresh bare capture (1,008 of 1,982 sycophancy /
1,004 of 1,967 hallucination / 1,009 of 1,987 evil) and the rest reused their committed
WildChat-rung rep, which for a single-turn context *is* its bare rep — re-verified here by a
reuse-licence gate that passed on all three.

![bare query vs full context](../figures/issue_1739/interim_writeup/bareq_vs_full.png)

> **Sycophancy is query-driven — the bare query is not just sufficient, it is better.** map → PV
> proj. rises from +0.122 [+0.075, +0.172] on the full context to **+0.200 [+0.154, +0.241]** on
> the bare query, and direct ridge is unchanged (+0.193 vs +0.190). Dropping the conversational
> prefix *helps* the projection arm here, which suggests the prefix was contributing noise rather
> than signal for this behavior.
>
> **Hallucination is prefix-driven, and the sign flips.** map → PV proj. goes from +0.111
> [+0.062, +0.157] to **−0.065 [−0.114, −0.020]** — from significantly positive to significantly
> negative. The query alone does not carry the fabrication signal; whatever the arm was tracking
> lived in the surrounding context.
>
> **Evil is prefix-driven most starkly of the three.** On its dedicated bare fit, map → PV proj.
> reads **−0.067 [−0.109, −0.019]** and direct ridge **−0.081 [−0.128, −0.028]**, against +0.157
> for the full-context map arm. That is the expected direction for this corpus: the eliciting
> content *is* the DAN-style prefix, so a predictor shown only the forbidden question has nothing
> to read. Two caveats specific to evil: its WildChat-rung DV is **floored** (mean 0.42, sd 4.43 —
> the fourth-column disclosure applies unchanged), so treat the collapse as directionally
> consistent rather than a precise effect size; and evil's leg 1 (+0.110) is render-mismatched by
> construction, which is why the dedicated leg-2 fit is the read that counts.
>
> **Internal consistency check that came for free:** the oracle arm — which projects the persona
> vector on the model's TRUE answer state — is numerically *identical* across renders on all three
> behaviors (+0.084 sycophancy, +0.178 hallucination, +0.156 evil). It should be, because it never
> looks at the input render, and its invariance confirms the DV and eval set really were held
> fixed while only the predictor's input changed.
>
> **Cross-behavior takeaway, stated plainly:** whether the bare query predicts the behavior tracks
> **where that behavior's variance comes from**. Sycophancy is a property of what the user asks
> (query-driven), so the query suffices. Hallucination and evil are properties of the surrounding
> context — the topic-and-history framing and the jailbreak prefix respectively — so stripping the
> prefix removes the signal outright.

**Open flag — the by-construction-null arms do not read zero, and this is unresolved.** Two
independent null arms in this round return small but CI-significant values where they must read
chance, so the round's CIs should be treated as provisional until it is explained.

![bare query null probe](../figures/issue_1739/interim_writeup/bareq_null_probe_layers.png)

> (a) Evil's leg-2 **shuffled-map control reads +0.068 [+0.025, +0.107]** — a nonsense map clearing
> zero. (b) The leg-1 **prefix null probe** returns `verdict: ANOMALY` on all three behaviors. Its
> logic (`scripts/issue1739_bareq_score.py::_null_probe`) is: the bare render's prefix position is
> a constant vector, so its projection must be rank-degenerate and ρ must be NaN or bracket zero;
> the verdict is `degenerate-as-predicted` only when constancy holds **and** no CI excludes zero.
> Here constancy *passed* (`constant: true`, per-layer cosine ≥ 0.9999 early / 0.9991 flat) yet
> **all 28 layers returned finite ρ and at least one CI excluded zero**, with |ρ| reaching 0.093
> (sycophancy), 0.120 (hallucination) and 0.055 (evil). The scorer's own note is unambiguous: *"A
> non-chance read is a capture/indexing bug, not a finding."*
>
> Two things bound the flag without resolving it. **Scope:** the probe reads the *prefix* position,
> whereas every headline number above is on `bare_context_end`, which legitimately varies with the
> query — so the anomaly does not directly invalidate the headline reads. **Tolerance:** the
> constancy bars are bf16 padded-batch cosines, explicitly *"NOT exact equality"*, and the record
> shows `max_abs_dev_from_row0 = 2.375` — the reps are constant to that tolerance, not bit-identical,
> so residual structure exists that Spearman can rank. Whether that residual is sufficient to
> produce a CI-excluding-zero ρ at n ≈ 1,982, or whether there is a genuine capture/indexing
> defect, **is not resolved by these artifacts** — and both anomalies share the same shape
> (a by-construction null at |ρ| ≈ 0.07–0.12), so they plausibly share a cause. Note also that the
> diagnostic which would have investigated it never ran: the caller skips the prefix arm sweep
> whenever constancy is *verified*, and the "run the sweep as the anomaly diagnostic" branch fires
> only on a constancy *failure* — so this ANOMALY is recorded but undiagnosed.

Three further limitations of this round:

- **The multi-turn-only subset could not be computed, so the reported contrasts are lower bounds
  on the multi-turn effect.** 987 of the 2,000 eval contexts are single-turn, and for those the
  bare render *is* the original render — they contribute zero contrast by construction and dilute
  the pooled ρ. The clean read would be the multi-turn subset alone, but this round persisted no
  per-context predictions: each `percell/bareq_leg1_transfer.jsonl` holds two aggregate records
  (coverage, null probe, the same six arm rows as the summary) and there is no `preds/` directory,
  unlike the main lane. Recovering the split needs a re-score, not a re-analysis.
- **Evil's leg 2 scores only on the WildChat rung.** Its own OOD rungs were skipped because the
  query bank was captured train-only, so none of the 2,387 eval-split contexts has a bare rep
  (`leg2_eval_block_notes`); re-capturing with all rungs would be needed.
- **The prefix-variant arm sweep is skipped on all three behaviors** as a degenerate null variant
  (`frozen_source: "n/a — degenerate null variant (arm sweep skipped)"`), so this round reports no
  prefix-render arm scores — only the null probe above.

One prior-work caveat carried verbatim from the round's own record
(`bareq_score_done.json`): *"#1092's 0.02 was bare->answer-ACTIVATIONS at SAE grain, so it is an
analogy, not a numerically comparable floor for bare->judged-DV."*

### A fourth evaluation column: random held-out WildChat (new)

The three existing columns per behavior are a held-out slice of the behavior's own training
distribution plus two curated real OOD sets. This adds a fourth: **2,000 random WildChat
conversations**, conversation-disjoint from the map-training pool (held out by content hash of
prefix turns and query text against the #1092 pool, per
`eval_results/issue_1739/wildchat_rung/contexts/wcrung_digest.json`). It is the only column with
no behavior-eliciting selection at all — ordinary user traffic — so it asks whether any of these
methods rank *naturally occurring* behavior.

What is plotted: Spearman ρ between each arm's prediction and the judged expression on the
WildChat rung, one bar per arm, per behavior, per input state; error bars are the 95% bootstrap
CI over contexts carried in each row's `ci_frozen`, drawn as non-negative offsets. Provenance:
**one shared on-policy rollout pool** — 5 sampled rollouts per context, 10,000 rollouts total —
generated once from Qwen-2.5-7B-Instruct and then judged under all three trait rubrics (the
WildChat contexts carry no behavior conditioning, so per-behavior generation would produce
byte-identical rollouts); judge `claude-sonnet-4-5-20250929`, 3 draws at temperature 1.0,
`max_tokens` 400 with an 800-token re-judge of truncation-affected items; graded 0–100 trait
score. Each arm's layer is **frozen from the main lane** (the modal frozen layer of that arm's
committed train cells — `frozen_source` in the per-layer rows), so no layer is selected on this
rung; single judge draw index, labeled readouts applied at the full label budget.

![wildchat rung arms](../figures/issue_1739/interim_writeup/wcrung_arms.png)

> **On the context end state the map arm is the best feasible method on two of three behaviors,
> and it is the only feasible method whose CI excludes zero on all three** (direct ridge fails
> that on hallucination; the oracle also clears it everywhere but needs the answer already
> generated, so it is not a deployable comparison). Evil: map → PV proj.
> **+0.157** [+0.116, +0.190] — statistically indistinguishable from the oracle (+0.156
> [+0.112, +0.196]) and roughly double direct ridge (+0.081), with Persona Vectors' own
> context projection at +0.042 [−0.010, +0.090] (CI spans zero) and the shuffled-map control at
> −0.037. Hallucination: map → PV proj. **+0.111** [+0.062, +0.157] vs direct ridge −0.013 and
> PV-on-context **−0.107** [−0.147, −0.062] — the paper's method is *significantly
> anti-correlated* with natural fabrication here, while the oracle reaches +0.178. Sycophancy is
> the exception: direct ridge wins (**+0.190** [+0.145, +0.235]) ahead of map → PV proj. (+0.122
> [+0.075, +0.172]), and both beat the oracle projection (+0.084) — i.e. the labeled readout
> extracts more than this persona vector carries even from the true answer state.
>
> **On the prefix end state the picture splits by behavior.** For hallucination the map arm reads
> at least as high from the pre-query prefix alone as from the full context (+0.155
> [+0.111, +0.200] at frozen layer 3 vs +0.111 [+0.062, +0.157] at layer 23 — overlapping CIs).
> That deserves scepticism rather than celebration: 49.4% of these contexts are single-turn, so
> for half of them the "prefix" is the bare chat-template head and can carry no
> conversation-specific information at all. Either the multi-turn half carries an unusually strong
> signal or the layer-3 read is picking up something query-independent; this is flagged for a
> targeted follow-up, not counted as a finding. Evil drops to +0.082 (oracle +0.172). Sycophancy's
> prefix map arm goes **negative** (−0.085) at frozen layer 0 — the same degenerate layer-0
> selection that produces the excluded cells in the nonlinear comparison below, so treat it as a
> broken read rather than a negative finding.
>
> Disclosures that bound this column. (a) **Evil's DV is floored on this rung** — mean 0.42, sd
> 4.43 on 0–100, with 1,966 of 1,987 contexts in the 0–10 bin (`spread/evil.json`): ordinary
> WildChat traffic essentially never elicits evil from an aligned model. The evil column is
> therefore *motivation-confirming* (the map arm ranks the thin upper tail) and NOT a ranking
> read; the usable spread is sycophancy (mean 17.8, sd 23.3) and hallucination (mean 39.6,
> sd 32.2). (b) **Hallucination uses the trait rubric, not the fabrication-rate construct** —
> WildChat queries carry no reference answers, so `dv_construct` is
> `trait_rubric_graded_0_100` for all three behaviors; the hallucination number here is not
> comparable to the TriviaQA/NQ-Open/SimpleQA fabrication rates in `## Results`. (c) **Judge
> content-drop rates differ sharply by behavior**: 567/30,000 draws for evil (1.9%), 2,624 (8.7%)
> for sycophancy, and **10,967 (36.6%) for hallucination** (`wcrung_score_done.json`; 4/139/1,354
> respectively recovered by the 800-token re-judge). Transport losses were zero for all three.
> The hallucination drop rate is high enough that its column should be read as provisional. (d)
> **987 of 2,000 contexts (49.4%) are single-turn**, so for those the "prefix" degenerates to the
> bare chat template head — the prefix panels average a genuine prefix over half the contexts
> with an empty one over the other half. (e) 19 of the sampled units were dropped for a duplicate
> final query within the sample; 13/18/33 contexts per behavior have no DV after judge drops
> (n = 1,987 / 1,982 / 1,967 of 2,000). (f) **There is no permutation null on this rung** — the
> `nulls` array is empty — so the shuffled-map arm is the only null reference, and it is a single
> draw, not a distribution. (g) Rollouts were capped at the shared 1024-token generation limit
> (`GEN_MAX_NEW_TOKENS`, `src/explore_persona_space/experiments/issue_1739/generation.py:48`),
> code-identical to the main labeling generation; a truncation rate of 1,955/10,000 (19.6%) was
> reported by the generation phase but **is not recorded in any committed or HF-resident artifact
> I could read, so treat that figure as unverified**.

The low-level companion view — ρ across all 28 layers for every arm, with the frozen layer
marked — shows the frozen-layer choices are not knife-edge picks on this rung:

![wildchat layer profiles](../figures/issue_1739/interim_writeup/wcrung_layer_profiles.png)

> The map arm's advantage on evil and hallucination holds across a broad band of mid-to-late
> layers rather than at one lucky layer, and the hallucination PV-on-context curve is negative
> across most of the depth — the anti-correlation above is not a frozen-layer artifact. Evil's
> oracle prefix curve rises smoothly with depth, the expected shape for a direction read off a
> true answer state.

### Nonlinear maps do not beat the linear map (round B, new)

The central question of round B: the context→answer map has been a **linear ridge** throughout;
does a nonlinear map buy transfer? Two replacements were run — an **MLP map** and a
**kernel-ridge map** — on all three behaviors, re-running only the three map-consuming arms
(map → PV proj., map → ridge on predicted answers, map → ridge trained on real answers) so every
other arm is held fixed.

What is plotted: the **matched** difference in Spearman ρ, nonlinear minus linear, median over
every design cell present in both lanes at the same (arm, input state, extraction regime, U rung,
label budget L, evaluation rung); error bars are the 95% bootstrap CI of that median across
matched cells. Each cell's value is the mean over its (seed, draw) replicates — 2 seeds × 3 draws
in the nonlinear lanes. **Note the y scales differ per panel**: evil's prefix kernel cell is a
≈ −0.5 outlier that would otherwise flatten the other five panels.

![nonlinear vs linear delta](../figures/issue_1739/interim_writeup/nlmap_vs_linear_delta.png)

> **Both nonlinear maps are worse than the linear map, and the kernel map is worse than the
> MLP.** Pooled over all matched cells: MLP median Δρ **−0.015** [−0.019, −0.010] across 456
> cells, better than linear in only 29.8% of them; kernel-ridge median Δρ **−0.038**
> [−0.044, −0.029] across 462 cells, better in 23.6%. Every context-end panel is negative with a
> CI excluding zero — hallucination kernel −0.092, evil kernel −0.083, evil MLP −0.054,
> sycophancy kernel −0.045, sycophancy MLP −0.038, hallucination MLP −0.021. On the prefix end
> state the sycophancy and hallucination deltas are effectively zero (median |Δρ| ≤ 0.007), which
> is consistent with those prefix arms carrying almost no signal for either map family rather
> than with nonlinearity helping there.
>
> At the operating slice (E1 persona vector, U = 18,793 pairs, L = max, context end state,
> held-out train distribution) the absolute cost is large: evil map → PV proj. 0.525 linear →
> 0.405 MLP → 0.277 kernel; sycophancy 0.359 → 0.330 → 0.303; hallucination 0.145 → 0.125 →
> 0.036. It persists under distribution shift, where a nonlinear map would have had the best case
> for helping: evil ToxicChat 0.319 → 0.211 → 0.018; hallucination SimpleQA 0.270 → 0.224 →
> −0.010. Sycophancy's AITA rung is the one near-tie (0.351 → 0.346 → 0.341).
>
> Caveats. The nonlinear lanes ran **2 seeds × 3 draws** per cell against the linear lane's larger
> replicate set (15 replicates at the operating slice), so the matched comparison averages
> unequal replicate counts; the projection arms are deterministic given the map (replicate sd
> 0.000), so the error bars above come from spread *across design cells*, not across seeds. The
> hallucination nonlinear lanes cover the **E1 regime only** (72 matched cells vs 216 for evil).
> On sycophancy, 159 arm-rows in the linear lane and 48 in the MLP lane are non-finite (the
> kernel lane has none) — all of them the `map → PV proj.` prefix arm at frozen layer 0, with no
> evaluable pairs. They are excluded rather than coerced, which is why the sycophancy prefix
> panels report 78 (MLP) and 84 (kernel) matched cells against 90 for the context end state.

Per-cell view behind that median — every matched cell, linear ρ on the x axis against nonlinear ρ
on the y, with the y = x line:

![nonlinear per-cell scatter](../figures/issue_1739/interim_writeup/nlmap_percell_scatter.png)

> The mass sits below the diagonal in all three behaviors, and the shortfall grows with the
> linear map's own ρ — i.e. nonlinearity costs most exactly where the linear map works best. The
> handful of cells above the diagonal are concentrated at low ρ (near-zero cells where both maps
> are uninformative), so no subregion of the design favours a nonlinear map.

### Unlabeled behavior-eliciting map data: the evil gain does not replicate (new)

The evil compose cells (context, previously reported below) found that replacing half a
fixed-size map-training pool with *unlabeled* behavior-eliciting contexts lifted the projection
arm from 0.36 to 0.56. These are the matching cells for sycophancy and hallucination.

What is plotted: Spearman ρ at a **fixed 5,000-pair map-training budget**, all-generic WildChat
pool (grey) against half-generic/half-behavior-eliciting (red), per arm and label budget L,
context end state, held-out train distribution.

![compose factor sycophancy hallucination](../figures/issue_1739/interim_writeup/compose_factor_syco_hallu.png)

> **For the projection arm the effect reverses on both new behaviors.** Sycophancy map → PV
> proj. falls 0.353 → 0.271 at L=250, 0.306 → 0.288 at L=2,500, 0.309 → 0.282 at L=16,000;
> hallucination falls much harder — 0.230 → 0.056, 0.141 → 0.062, 0.124 → 0.028. So the evil
> result is not a general property of eliciting map data: for the label-free projection read,
> swapping generic user traffic for behavior-eliciting contexts *hurt* in 6 of 6 new cells.
>
> **The labeled readout trained on real answer states moves the other way, strongly.** Arm
> `map → ridge (real answers)` gains on every cell of both behaviors: sycophancy 0.282 → 0.593
> (L=250), 0.136 → 0.351 (L=2,500), 0.294 → 0.652 (L=16,000); hallucination 0.257 → 0.422,
> 0.142 → 0.339, 0.196 → 0.554. The middle arm (`map → ridge` on *predicted* answers) is mixed
> and small (sycophancy +0.079 at L=250, −0.135 at L=2,500, −0.010 at L=16,000). Read together,
> eliciting map data appears to help a readout that must generalize from true answer states while
> hurting a fixed direction's projection — a per-arm effect, not a map-quality effect.
>
> Caveats, and they are heavy. These are **single-cell reads (1 seed × 1 draw)**, so no error
> bars are available and the L=2,500 non-monotonicity flagged elsewhere in this doc applies here
> too. The half-eliciting bar **pools the two label-composition variants** (f_l = 0.0 and
> f_l = 1.0) where both exist — they agree closely (e.g. sycophancy arm 6 at L=250: 0.267 and
> 0.274), but it is an average of two conditions, not one. **This lane has no shuffled-map
> control**, so the within-arm map-vs-control contrast that made the evil compose read
> trustworthy is unavailable here; per-arm shifts could still partly reflect per-cell layer
> re-freezing. The prefix-end compose cells are near-inert (4 of 9 sycophancy and 0 of 9
> hallucination cells are numerically identical across pool compositions), consistent with the
> prefix arms carrying little composition-sensitive signal.

### Naturalistic persona-vector extraction regimes: sycophancy replicates, hallucination fails (new)

The three extraction regimes — **E1** paper-faithful synthetic, **E2** matched-pair natural,
**E2p** pooled natural — were previously read on evil only. This extends them to sycophancy and
hallucination, across every projection read (context state, prefix state, map-predicted answer
from either input state, and the true answer state).

What is plotted: Spearman ρ at each read's frozen layer, one panel per (behavior, evaluation
rung), one bar per extraction regime. The frozen layer is chosen by **max |ρ| on the train rung**,
so the sign is not pinned by the selection. Faded bars mark cells where the extraction pool
overlaps the evaluation rung (E2 and E2p are in-sample on the train rung) — read the
out-of-sample rungs (AITA for sycophancy; NQ-Open and SimpleQA for hallucination) as the
evidence-bearing ones.

![natpv regimes](../figures/issue_1739/interim_writeup/natpv_regimes_syco_hallu.png)

> **Sycophancy replicates evil's headline ordering and yields the best sycophancy predictor
> measured so far.** On the out-of-sample AITA rung, pooled-natural E2p is top for every read:
> map(context) → PV proj. **0.577** (vs 0.487 E2, 0.436 E1), context projection 0.416 (vs 0.332,
> 0.309), true-answer projection 0.492 (vs 0.240, 0.289). The E2p-on-top part of the evil result
> therefore carries over; the E1-vs-E2 order does **not** — evil had synthetic ahead of
> matched-pair, sycophancy has matched-pair ahead of synthetic — so "natural beats synthetic" is
> the robust half of that finding, not the full ranking. Note also that map(context) beats the
> context projection in every sycophancy regime, and at E2p it beats even the true-answer
> projection (0.577 vs 0.492).
>
> **Hallucination's regime ladder is sign-unstable, which reads as a persona-vector construct
> failure rather than a ranking.** For the same read and regime, ρ flips sign across evaluation
> rungs: context projection at E1 gives −0.144 (NQ-Open), +0.465 (SimpleQA), −0.299 (train);
> map(context) at E2 gives −0.055, +0.551, −0.221. **Thirteen of the fifteen** (read × regime)
> combinations are not sign-consistent across its three rungs; only the true-answer projection at
> E1 and E2 holds one sign. For contrast, all 15 sycophancy combinations are sign-consistent —
> though across only 2 rungs, so that is a weaker test. A direction whose correlation flips sign
> between two OOD sets is not measuring a stable trait, and this is consistent with the previously
> reported finding that the hallucination persona vector reaches only ρ ≈ 0.04 even projected on
> the true answer state.
>
> Caveats. The prefix reads are degenerate for sycophancy — every regime sits at frozen layer 0
> with |ρ| ≈ 0.031 and the sign flipping between E1 and E2/E2p, i.e. the same direction up to
> sign, so the prefix rows carry no regime information. E2 and E2p are in-sample on the train
> rung by construction, so their train-rung bars are upward-biased and are faded in the figure.
> Layer selection by max |ρ| on the train rung means a negative-ρ direction can be frozen, which
> is part of why the hallucination signs move.

## Results

*(evil + hallucination; context, previously reported — the sections above carry what is new)*

### Spread of behavior expression per evaluation setting

What is plotted: per-context judged expression (evil: mean 0–100 score over 5 on-policy rollouts × 3 judge draws; hallucination: fabrication rate over 5 rollouts), one histogram per evaluation setting.

![spread evil](../figures/issue_1739/interim_writeup/spread_evil.png)

![spread hallucination](../figures/issue_1739/interim_writeup/spread_hallucination.png)

> All hallucination settings and the evil train/ToxicChat settings have usable spread (SD 0.33–0.42 on 0–1; SD 26.3 and 12.1 on 0–100). **The evil hh-rlhf rung fails the pre-registered spread floor** (mean 0.08, SD 0.89 on 0–100 — Qwen-2.5-7B refuses essentially all 2022-era red-team attempts), so hh-rlhf comparisons below are floor-censored and uninformative. Evil scores are additionally refusal-censored: 1,532 train / 127 hh-rlhf / 152 ToxicChat contexts have no DV because all rollouts refused.

### Main comparison: ρ per method, per behavior, per evaluation setting

What is plotted: Spearman ρ between each method's prediction and judged expression, at the largest budgets (U = 18,793 unlabeled pairs; L = 8,000 evil / 16,000 hallucination labels; E1 persona vector; full-context end state). Only 6 arms were evaluated on the OOD sets (label-free projections + direct ridge + oracle projection + shuffled-map control). All arms in a panel are scored against the same judged targets.

![bars evil](../figures/issue_1739/interim_writeup/bars_evil.png)

![bars hallucination](../figures/issue_1739/interim_writeup/bars_hallucination.png)

> **In-distribution, map→PV projection does not beat context-side methods.** Evil: map→PV 0.53 vs context-native direction 0.66, direct ridge 0.71, oracle regression 0.82. Hallucination: map→PV 0.15 vs context-native 0.52, direct ridge 0.58, oracle regression 0.65. The pre-registered headline contrast (map→PV minus context-native projection) is negative — evil: median Δρ = −0.30 across all 826 cells, bootstrap CI below 0 in 708/826; hallucination: median Δρ = −0.07 (137/270 below, 18 above).
>
> **The hallucination persona vector, not the map, is the broken link:** even projected on the TRUE answer state (oracle), the hallucination PV reaches only ρ = 0.04 on held-out TriviaQA — while a regression on the same true answer states reaches 0.65. The synthetic hallucination direction barely tracks natural fabrication. For evil the PV direction is genuine (oracle projection 0.71).
>
> **Map→ridge-on-predicted-answers matches or slightly beats direct ridge** (evil 0.714 vs 0.706; hallucination 0.602 vs 0.584) — but the shuffled-pretrain control reaches the same value (0.714 / 0.601), so at large L the labeled readout, not the map, is doing the work.
>
> **Under real distribution shift the label-free map arm is the most robust feasible method on 2 of 3 usable OOD settings.** ToxicChat (evil): map→PV 0.32 vs direct ridge 0.25, PV-on-context 0.14, shuffled-map control 0.11 — the map arm matches the oracle projection (0.30). SimpleQA (hallucination): map→PV 0.27 vs direct ridge 0.10. NQ-Open is the exception: direct ridge wins at L=16,000 (0.40 vs 0.20). Where predictions have to transfer, mapping into answer space before projecting helps; in-distribution it costs.

### Scaling: unlabeled map pairs (U) and labeled examples (L)

What is plotted: left — map-based arms vs U at L=max on the held-out train setting (context-native projection shown as a U-free reference line); middle — label-budget scaling at U=full, same setting; right — label-budget scaling on the OOD setting (6-arm transfer roster; note the hh-rlhf panel for evil is floor-censored per the spread section).

![scaling evil](../figures/issue_1739/interim_writeup/scaling_evil.png)

![scaling hallucination](../figures/issue_1739/interim_writeup/scaling_hallucination.png)

> **The map does learn transferable structure from unlabeled pairs:** map→PV improves with U (evil train 0.34 → 0.53 from 250 → 18,793 pairs; hallucination SimpleQA 0.06 → 0.27) while the shuffled-map control stays flat (≈0.26 / ≈0), and map→ridge-on-real-answers gains +0.16–0.19 from U. But **no in-distribution sample-efficiency advantage materialized:** at L=250 the map arms do not beat direct ridge (evil 0.47 vs 0.52; hallucination 0.41 vs 0.45), and map-pretrain→fine-tune is indistinguishable from the shuffled-pretrain control at every L (e.g. evil L=250: 0.472 vs 0.467). The OOD panels show the map arm's flat-but-highest curves on ToxicChat at all L (0.27–0.32 vs ridge ≤ 0.25).
>
> Unexplained interim pattern flagged for the analyzer pass: several labeled arms (direct ridge, stacked, oracle regression) are non-monotone in L with a dip at L=2,500 in both behaviors (e.g. evil ridge 0.52 → 0.34 → 0.71) — a mid-budget fitting artifact (regularization/layer-selection instability) is suspected; low-L numbers above should be read with this in mind.

### Prefix-end vs full-context-end input state

What is plotted: paired bars per method, full-context end state (solid) vs pre-query prefix end state (hatched), held-out train setting, largest budgets.

![variant evil](../figures/issue_1739/interim_writeup/variant_evil.png)

![variant hallucination](../figures/issue_1739/interim_writeup/variant_hallucination.png)

> Evil retains most signal from the prefix state (direct ridge 0.56 vs 0.71; the DAN persona prefix carries the elicitation). Hallucination collapses to ρ ≈ 0.05 for every context-dependent arm — expected, since bare trivia questions have essentially no pre-query prefix; consistent with the #1092/#1774 finding that the pre-query prefix state carries only persona-average signal.

### Predicted vs judged expression (scatter, per method)

What is plotted: OOF predictions vs judged expression for one representative max-budget cell (seed 0, draw 0), held-out train setting; ≤1,500 points per panel; per-panel ρ over all contexts. Per-cell prediction arrays were persisted only for the train setting, so OOD scatters are not available in this interim cut.

![scatter evil](../figures/issue_1739/interim_writeup/scatter_evil.png)

![scatter hallucination](../figures/issue_1739/interim_writeup/scatter_hallucination.png)

> The evil panels show the fan shape typical of a refusal floor (most contexts at score 0 at every prediction level; signal comes from the upper envelope). The hallucination DV is discrete (rate over 5 rollouts → bands at 0, 0.2, …, 1.0); the PV-projection panels (paper method, oracle projection) are visibly unstructured, matching their ρ ≈ 0.04–0.08.

### Persona-vector extraction regime (evil)

What is plotted: ρ for the three PV-dependent arms under E1 (synthetic, paper-faithful), E2 (matched-pair natural), E2p (pooled natural), held-out train setting, largest budgets.

![regimes evil](../figures/issue_1739/interim_writeup/regimes_evil.png)

> Pooled-natural (E2p) beats synthetic (E1) beats matched-pair (E2) for every projection arm (map→PV: 0.55 / 0.53 / 0.24). The topic-controlled matched-pair direction — the one construction that cancels topic — is much weaker, suggesting a substantial share of projection performance rides on topic rather than disposition.

### The persona-vectors synthetic eval inflates projection numbers (added 2026-07-30 evening)

What is plotted: Spearman ρ per method on Persona Vectors' own eval distribution (their 5 positive + 5 negative instruction system prompts × 20 held-out eval questions = 200 contexts/behavior; on-policy K=5 rollouts; our standard judge, PV per-trait rubric) as the red bars, next to the committed real-rung values (op-slice means) — same arms, same frozen-layer convention, transfer-applied readouts.

![pv suite vs real](../figures/issue_1739/interim_writeup/pvsuite_vs_real.png)

> The paper's method gains **+0.26 (evil) to +0.57 (hallucination) ρ** moving from real data to its own suite (evil 0.80 vs 0.54 train / 0.14 ToxicChat; hallucination 0.65 vs 0.08 TriviaQA). On evil, every projection method converges to ~0.80 on the suite — pos/neg-instruction separability saturates and method ranking disappears. Sharpest evidence that the suite measures *instructed-behavior separability* rather than natural elicitation: the hallucination PV direction, near-useless on natural data even with oracle answer states (0.04), reads 0.65–0.74 on the suite; and on sycophancy a **shuffled-map control scores ρ = 0.54** — a nonsense direction separates their prompt structure before any method quality enters. Caveats: the hallucination suite column uses the trait rubric (their questions carry no reference answers; not the fabrication-rate construct; 23.4% judge content-drops, ~half unrecovered at the 800-token re-judge); sycophancy's suite spread is compressed (SD 20, max 85) and its real rungs land when the main lane finishes; prefix-arm rows exist but the suite has only 10 distinct prefix states, so those ρ are rank-tie-dominated. Artifacts: `eval_results/issue_1739/pvsynth/` @ `34c041409d`.

### Unlabeled behavior-eliciting data substitutes for labels (composition factor, evil)

No figure yet (analyzer pass will render one); numbers from the committed compose cells (U = 5,000 total, E1, context-end, held-out train dist.; small cells — ~1 seed × 2 draws): replacing half the generic WildChat map pool with 2,500 *unlabeled* behavior-eliciting contexts (disjoint from the labeled set) lifts map→PV projection from **0.36 to 0.56** — more than the full 18,793-pair all-generic pool achieves (0.53) — while the shuffled-map control moves the other way (0.32→0.27 at L=250; 0.28→0.10 at L=2,500). Map-based labeled readouts gain only at low label budgets (map→ridge-on-predicted 0.29→0.44 at L=250; ≈unchanged at L=2,500). Two flags: arms that consume no unlabeled data also shift between compose cells (suspected per-cell layer re-freezing — the within-arm map-vs-control contrast is the trustworthy read), and this result is in tension with #779's "trait-trained map is worse" (fixed-budget replacement vs full substitution, and real DAN prefixes vs #779's corpora, are the reconciliation candidates).

### A ~1M-context generic map beats the in-experiment map on 2 of 3 behaviors (frozen reuse of #779's maps)

What is plotted (committed by the reuse round): ρ of #779's 963,444-context maps (ridge + MLP w8192/w32768, applied frozen through #779's own code; layers {14, 19, 26}) vs the in-experiment 18.8k-pair map and a shuffled control, recomputed matched-target in one process on the same contexts/DV/direction.

![963k comparison](../figures/issue_1739/interim_writeup/map963k_reuse_comparison.png)

> The 963k map transfers genuinely (reconstructs this experiment's answer states at cosine 0.92–0.99, held-out R² 0.22–0.62 on corpora it never saw) and **wins 9/9 evil cells** (+0.04..+0.17; best-layer ρ 0.60 vs oracle 0.64) and 6/9 hallucination cells (losses only where the rung's own oracle ≈ 0), but **loses all 6 sycophancy cells** — where the in-experiment map's projection on the AITA rung (0.40) even exceeds the oracle projection (0.27). Caveats: the in-experiment-map comparison column applies the *uploaded* map payload (a faithful application, but not a verified reproduction of the committed arm-6 numbers — oracle anchor validated 10/10, arm-6 anchor looser), and that payload extrapolates with strongly negative reconstruction R² onto the behavior eval distributions (a distribution-coverage finding, not a serialization bug — the payload round-trips cleanly on its own distribution). Artifacts: `eval_results/issue_1739/map963k_reuse/` @ `606278aa38`.

### Reversed train/eval direction (evil, secondary config)

Training the labeled readouts on hh-rlhf red-team dialogues and evaluating on DAN×forbidden (90 cells, the pre-registered secondary) collapses every method to ρ ≤ 0.23, including both oracles (oracle regression 0.19, oracle projection 0.23). With the training side floor-censored (the spread failure above), little transfers in this direction; the A−B mechanism-match comparison the plan wanted is compromised by that censoring.

## Interim takeaways (2026-07-31, pre-analyzer)

These fold the four new families in § New in this update. The carried-forward takeaways from the
previous cut follow in the next block, with corrections marked.

1. **Whether the bare user query predicts a behavior localizes that behavior's predictive
   variance.** Stripping the prefix and showing the predictor only the query *raises* sycophancy
   (map → PV proj. +0.122 → +0.200), *flips the sign* on hallucination (+0.111 → −0.065), and
   *destroys* evil (+0.157 full-context → −0.067 on a dedicated bare fit). Sycophancy is a
   property of what the user asks; hallucination and evil are properties of the surrounding
   context. **Caveat before use:** this round carries an unresolved anomaly — two
   by-construction-null arms read |ρ| ≈ 0.07–0.12 where they must read chance — and its
   single-turn half (987/2,000 contexts) dilutes the pooled contrast, so the effects above are
   directional lower bounds, not settled effect sizes.
2. **Nonlinearity in the map is not the missing ingredient.** An MLP map and a kernel-ridge map
   both lose to the linear ridge map on all three behaviors (pooled median Δρ −0.015 and −0.038;
   better in only 30% and 24% of matched cells), and lose most exactly where the linear map works
   best. Round B's question is answered negatively; the linear map stays the operating choice.
3. **On ordinary user traffic the map arm is the best feasible predictor for evil and
   hallucination.** On the new random-WildChat column the map → PV projection matches the oracle
   on evil (+0.157 vs +0.156) and reaches +0.111 on hallucination where Persona Vectors' own
   context projection is significantly *negative* (−0.107) — the strongest evidence yet that
   mapping into answer space before projecting recovers signal the context-side projection
   cannot. Sycophancy is the counterexample (direct ridge +0.190 leads).
4. **Extracting the persona vector from natural data is the robust half of the regime result.**
   Pooled-natural E2p is top for every sycophancy read (map(context) → PV proj. 0.577 on
   out-of-sample AITA, the best sycophancy predictor measured), replicating evil's E2p-on-top
   ordering; the E1-vs-E2 order does not replicate, so "natural beats synthetic" is what carries,
   not the full ladder.
5. **For hallucination the persona vector fails as a construct in a newly explicit way:** its
   correlation flips *sign* across evaluation rungs for the same read and regime (12 of 15
   read × regime combinations sign-inconsistent). Any hallucination conclusion about "the map"
   remains bottlenecked by the direction.
6. **Eliciting map data is a per-arm lever, not a map-quality lever.** Replacing half a fixed map
   pool with unlabeled behavior-eliciting contexts *hurt* the projection arm in 6 of 6 new cells
   (sycophancy and hallucination) while *helping* the real-answer-trained readout in 6 of 6 — the
   opposite direction from the evil projection result, which this corrects (see takeaway 7 below).

### Carried forward from the 2026-07-30 cut (2 of 3 behaviors) — context, previously reported

1. The datatype-mismatch intuition is half-right: answer-space PVs are indeed a poor fit for context states (context-native directions beat them: 0.66 vs 0.54 evil, 0.52 vs 0.08 hallucination) — but the fix that works in-distribution is extracting a **context-native direction**, not mapping the context into answer space (map→PV trails both).
2. The map demonstrably learns real structure from unlabeled WildChat pairs — performance rises with U while shuffled-map controls stay flat — but that structure is **redundant with the context representation in-distribution**: no map arm beats direct ridge at any label budget, and map-pretraining gives fine-tuning no head start over shuffled pretraining.
3. The clearest value shows up **under distribution shift**: the label-free map→PV arm is the top feasible method on ToxicChat (0.32, ≈ oracle projection) and SimpleQA (0.27), degrading least of all context-side methods; NQ-Open at max labels is the counterexample (ridge 0.40).
4. For hallucination the persona vector itself fails as a construct on natural data (oracle projection ρ = 0.04 in-distribution) — any conclusion about "the map" for this behavior is bottlenecked by the direction, and the direct-regression framing (or a better direction) is required.
5. Readout/deployment consistency matters more than realism: regressions trained on **predicted** answer vectors and applied to predicted vectors (evil 0.71, hallu 0.60) far outperform regressions trained on **real** answer vectors and applied to predicted ones (0.51 / 0.38).
6. **The persona-vectors synthetic suite inflates projection performance by +0.26–0.57 ρ over real distributions** (all three behaviors ≈0.65–0.80 on the suite; a shuffled-map control alone reaches 0.54 on sycophancy) — the suite measures instructed-behavior separability, not natural elicitation.
7. Map *training data* matters more than map size in-distribution but less under scale: 2,500 unlabeled eliciting contexts beat 18,793 generic pairs for the projection read (evil compose cells), while the frozen 963k generic map beats the 18.8k map on evil and mostly on hallucination — and loses on sycophancy. **CORRECTED 2026-07-31:** the first clause is evil-specific. On sycophancy and hallucination the same substitution *lowered* the projection read in 6 of 6 cells (while raising the real-answer-trained readout in 6 of 6) — see new takeaway 6 and § Unlabeled behavior-eliciting map data above.

## Provenance

**2026-07-31 bare-query round.** Numbers + figures by
`scripts/issue1739_bareq_fold.py` (aggregation + rendering only; no fits, no GPU, no judge calls),
from artifacts committed on branch `issue-1739` at commit `86bce0e94b`:
`eval_results/issue_1739/bareq_map/{evil,sycophancy,hallucination}/{all_arms_spearman.json,map_diagnostics.json,percell/bareq_leg1_transfer.jsonl}`
plus `bareq_map/{bareq_queries.json,bareq_score_done.json,bareq_score_failures.json}`. The
full-context comparison column is the same HF-resident WildChat-rung result folded above
(`issue1739_ctxmap/wildchat_rung/arm_results/<behavior>/all_arms_spearman.json`). Design labels
(`render_match`, `leg2_noop`, `leg1_null_probe`, fold scheme, query bank) are read from each
summary's `meta`; the null-probe verdict logic is
`scripts/issue1739_bareq_score.py::_null_probe`. Aggregates dumped to
`/tmp/i1739_bareq_stats.json`. Figures:
`figures/issue_1739/interim_writeup/{bareq_vs_full,bareq_null_probe_layers}.png`.
Known gaps carried in the section: the ANOMALY null-probe verdict is undiagnosed; no per-context
predictions were persisted, so the multi-turn-only subset is not computable without a re-score;
evil's leg 2 covers the WildChat rung only. `bareq_score_failures.json` records one earlier
leg-2 failure (missing query manifest) that was resolved before the committed run — the completed
summary carries all 12 evil transfer rows across both legs.

**2026-07-31 update (the four earlier families).** Numbers + figures computed by
`scripts/issue1739_writeup_fold.py` (aggregation + rendering only, no fits), from artifacts on
branch `issue-1739` at commit `4c5a2b28c3` unless noted:

- *Nonlinear round B:* `eval_results/issue_1739/nonlinear_map/{evil,sycophancy,hallucination}/{mlp,kernel}/arm_results/all_arms_spearman.json`,
  compared matched against the linear main lanes
  `eval_results/issue_1739/{evil,sycophancy,hallucination}/arm_results/all_arms_spearman.json`
  (826 / 810 / 270 cells; identical bytes on `main` and the branch).
- *Compose cells:* `eval_results/issue_1739/nonlinear_map/{sycophancy,hallucination}/compose_linear/arm_results/all_arms_spearman.json`
  (22 cells each, 1 seed × 1 draw).
- *WildChat rung:* arm results are **HF-resident**, not committed —
  `superkaiba1/explore-persona-space-data` (dataset) at
  `issue1739_ctxmap/wildchat_rung/arm_results/{evil,sycophancy,hallucination}/all_arms_spearman.json`
  plus `per_layer_rows` in the same files. DV spread, coverage, and judge drop accounting come
  from the committed `eval_results/issue_1739/wildchat_rung/{spread/<behavior>.json,wcrung_score_done.json}`;
  context construction + hold-out mechanism from
  `eval_results/issue_1739/wildchat_rung/contexts/wcrung_digest.json`.
- *Naturalistic PV regimes:* `eval_results/issue_1739/nat_pv_regimes/{sycophancy,hallucination}/regime_comparison.json`.
- Prose-ready aggregates dumped to `/tmp/i1739_fold_stats.json`. Figures:
  `figures/issue_1739/interim_writeup/{wcrung_arms,wcrung_layer_profiles,nlmap_vs_linear_delta,nlmap_percell_scatter,compose_factor_syco_hallu,natpv_regimes_syco_hallu}.png`.
- Known gap: the WildChat generation truncation rate (reported as 1,955/10,000) is not recorded in
  any committed or HF-resident artifact and is carried above as unverified. Evil's
  `wildchat_rung` `map_diagnostics.json` is an empty object, so no map-reconstruction diagnostics
  are available for that behavior's fourth column.

**2026-07-30 cut (everything under `## Results`).**

- Numbers/figures computed 2026-07-30 from `eval_results/issue_1739/{evil,evil_config_b,hallucination}/arm_results/all_arms_spearman.json` and `eval_results/issue_1739/dv_dataset/*/labeling.json` (branch `issue-1739`, commits `1ddf165113` / `ab66d4f04d`), by `scripts/issue1739_interim_writeup_figs.py`; per-cell prediction arrays from `arm_results/percell/preds/`.
- Rollouts are on-policy (temp-sampled, K=5 per context) from Qwen-2.5-7B-Instruct; judge `claude-sonnet-4-5-20250929`, 3 draws @ temp 1.0, `max_tokens` 400 with an 800-token re-judge of truncation-affected items (mixed-instrument caveat carried in the DV metadata). Evil DV coverage is 83% (refusal-censored by design); hallucination 100%.
- Known caveats to carry into the final clean-result: evil's labeled axis has ~1,405 independent jailbreak-prefix groups (group-effective N ≪ row count); TriviaQA/NQ-Open contamination compromises absolute fabrication rates (method deltas are the design's claim surface); the L=2,500 non-monotonicity above is unexplained; hh-rlhf rung floor-censored.
