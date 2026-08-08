# Narrative exemplar — task #380 (many-findings rollup)

**What this exemplar shows.** A `## Details` body in the LessWrong-style narrative shape spec'd by CLAUDE.md § Experiment Report Structure (2026-05-27 reshape) — applied to a many-findings experiment that exceeded the TL;DR-inline-figures budget. The source body lives at `tasks/awaiting_promotion/380/body.md`; this exemplar shows what its `## Details` would look like under the new narrative rules. Read alongside `docs/lw-post-examples/` (external LW register references).

**Rules demonstrated** (cited inline at each beat below):

- **CLAUDE.md § TL;DR many-findings escape valve** — Results rolls up via `[Per-finding figures and reads in Details.](#findings)` when there are >3 findings (Lens 9).
- **SPEC.md § Voice (v3) → Figures inline-narrate, not figure-dump** — every figure framed by setup paragraph above + read paragraph below (the v3 setup/read pairing is spec'd in SPEC.md § `## Findings`) (Lens 12 check #2).
- **CLAUDE.md § Details story arc** — five-beat structure: question → expected → seen → interpretation → next (Lens 12).
- **H3s mark story beats, NOT deliverable labels** (Lens 4 + Lens 12). All bad H3 labels from #380 (`### Headline result`, `### Subset checks`, `### Sample completions`, `### Plan deviations`) are replaced with story-beat H3s or folded into the narrative flow.
- **Surprises and pivots in the narrative** (Lens 12 check #3) — the method-dependent sign-flip, the cohort disagreement, the recut stratification all flow inline at the moment the story reaches them, not parked at the bottom.
- **Raw alongside processed** (Lens 11) — raw and length-residualized scatters embedded as a pair under each finding's narrative beat.
- **Connective transitions ALLOWED in Details** (Voice + Statistics scope) — "Then I tried", "But that didn't replicate", "The interesting bit came next" appear in the prose.

---

## TL;DR (illustrative; many-findings rollup)

- **Motivation:** I'm trying to find what predicts how strongly a `[ZLT]` marker implants into a persona. Hidden-state cosine distance from the assistant centroid was the lineage hypothesis from [#271](https://eps.superkaiba.com/tasks/271), but [#340](https://eps.superkaiba.com/tasks/340) and [#368](https://eps.superkaiba.com/tasks/368) overturned it on prompt-length grounds. So I tried JS divergence in output space instead, with two operationalizations and a follow-up.
- **What I ran:** Same 48-persona panel as #340/#368, same source-rate measurements, same Qwen-2.5-7B-Instruct base. Generated 980 greedy completions (49 system prompts × 20 neutral probes). Computed two output-space predictors per persona (distance from the assistant baseline + mean pairwise distance to peers) and a cosine-space follow-up on the inherited N=24 cohort.
- **Results:** Six distinct things to report across three predictors, three subset checks, a method-implementation footnote, and one open thread. None of the three predictors clears the planned threshold, but the secondary points in an *opposite* direction from the planned hypothesis and the convergence pattern is informative. [Per-finding figures and reads in Details.](#findings)
- **Next steps:** Push the cosine-pairwise variant to N=48 by extracting L15 centroids for the 24 new-cohort personas; rerun on a larger / more-diverse panel if the negative-direction sign survives.

---

## Details

I'm trying to find what predicts how strongly a `[ZLT]` marker implants into a persona, based on properties of the persona itself. The lineage hypothesis from [#271](https://eps.superkaiba.com/tasks/271) was that hidden-state cosine distance from the assistant centroid should do this — the idea being "personas farther from the assistant identity get marked more strongly." [#340](https://eps.superkaiba.com/tasks/340) and [#368](https://eps.superkaiba.com/tasks/368) ran that on the 48-persona panel and the apparent correlation collapsed under a prompt-length partial; cosine carried the same length information that source rate did, and nothing length-independent survived.

So I went into this expecting JS divergence in *output space* — same conceptual axis (distance from assistant identity), different feature space (next-token distribution instead of residual stream) — to also fail the same way, IF the two distance measures are measuring the same thing. The prior was: "if cosine and JS rank-correlate strongly, JS will die on the same length confound; if they don't, JS might survive where cosine didn't." I also wanted to test a *different* hypothesis with a second predictor: mean pairwise output-distance to other personas, which asks "is this persona isolated from peers, regardless of where the assistant sits?" — conceptually closer to the bystander-leakage setting where output distance has worked in prior work ([#142](https://eps.superkaiba.com/tasks/142), [#207](https://eps.superkaiba.com/tasks/207), [#228](https://eps.superkaiba.com/tasks/228)).

The 48-persona panel, source-rate measurements, base model, and probe set are all inherited verbatim from #340/#368/#207. The only new pieces are the 980-completion generation pass on Qwen-2.5-7B-Instruct (greedy, temperature 0, seed 42, 20 probes × 49 personas including the assistant baseline) and the two output-space predictors computed off those completions. Decoder config and statistical-test machinery are in the Parameters table at the bottom; the Confidence sentence names the binding constraint.

### Findings <a id="findings"></a>

Six things to report. They sit in this order because the story builds: the primary fails, the secondary fails in an interesting direction, the subset / cohort / method checks tell you the failure is structural rather than a thresholding artifact, and the cosine follow-up opens an unresolved thread.

#### The primary predictor dies on the same length confound that killed cosine

The cleanest test of the lineage hypothesis is the primary predictor: output-distance from the assistant baseline. I expected this to track source rate weakly-positively in the raw association (some persona-information signal) and to either die or shrink under length partial (depending on how much of the raw association was length). What I got was the first half cleanly:

![Raw scatter of JS divergence from assistant baseline (x) vs source rate (y) across 48 personas; the fit line slopes weakly upward](https://raw.githubusercontent.com/superkaiba/explore-persona-space/<sha>/figures/issue_380/primary_scatter_raw.png)
![Length-residualized scatter of the same; the fit line is essentially flat](https://raw.githubusercontent.com/superkaiba/explore-persona-space/<sha>/figures/issue_380/primary_scatter.png)

Raw association is Spearman ρ = +0.29, p = 0.048, N=48 — weakly positive in the hypothesised direction. The length partial then collapses it to p = 0.87, N=48: the raw signal is essentially all length. The reason is visible in the predictor-vs-length collinearity (linear correlation between the predictor and log prompt token count: p = 1.5e-05). A persona's output-distance from the assistant baseline carries roughly as much length information as it carries persona information.

The helpful-assistant-family personas anchor the short-prompt end of the panel (`chatbot` at 6 tokens, `ai_tool` at 6, `ai` at 5) and the helpful-family-out subset confirms the pattern — when I drop the 11-member helpful family, the partial nudges further negative (ρ = -0.097, p = 0.57, n=37) but doesn't rescue a length-independent positive signal.

#### A method-dependent sign-flip on the primary headline

The plan named two acceptable implementations of the length-partialled rank correlation: `pingouin.partial_corr(method='spearman', covar=['log_tokens'])` as first preference, and an inline rank-residualize-then-correlate as the explicit fallback that matches the wording in #340's clean-result Methodology section. Both agreed to four decimal places on a synthetic triple at the launch smoke test. On the real N=48 data they disagreed by enough to flip the sign of the primary headline's point estimate:

| Implementation | Length-controlled ρ | p |
|---|---|---|
| pingouin | +0.024 | 0.87 |
| inline rank-residualize | -0.041 | 0.78 |

Both agree the predictor is statistically indistinguishable from zero; both p > 0.78. Neither implementation locates where "the residual signal actually is" — the resampled 95% interval brackets zero symmetrically — but the disagreement is much larger than the synthetic-data spread suggested. This is a methodological footnote on the headline number, not a contradiction of the negative conclusion. (The two implementations agree to within 0.04 on every subset and on the secondary; this flip is specific to the primary headline on the full panel.)

#### The secondary points in the opposite direction from the planned hypothesis

The mean-pairwise predictor was testing a different question: not "is this persona far from the assistant identity?" but "is this persona isolated from peers in next-token space?". I expected it to survive better than the primary if the assistant-anchored confound was the dominant problem. What I got:

![Raw scatter of mean pairwise output-distance to other personas (x) vs source rate (y) across 48 personas; the fit line slopes weakly downward](https://raw.githubusercontent.com/superkaiba/explore-persona-space/<sha>/figures/issue_380/pairwise_js_scatter_raw.png)
![Length-residualized scatter of the same; fit line slopes weakly negative](https://raw.githubusercontent.com/superkaiba/explore-persona-space/<sha>/figures/issue_380/pairwise_js_scatter.png)

The raw association is already weakly negative (Spearman ρ = -0.18, p = 0.23, N=48); the length partial sharpens it to ρ = -0.276, p = 0.061. The 95% resampled interval reaches asymmetrically into the negative territory while barely touching the positive side. This doesn't pass the planned threshold (the p-value is six times above 0.01) but it also doesn't pass the strict null-result threshold (the interval reaches outside [-0.15, +0.15]). And critically, it points *opposite* to the planned "more distinct → more vulnerable" direction: more isolated from peers → *less* vulnerable, not more. The predictor is much less length-confounded than the primary (collinearity p = 0.05 vs p = 1.5e-05); it's not the length confound that's doing the work here.

#### The negative sign is consistent across reductions

If the negative direction were noise, I'd expect different reductions of the pairwise matrix (median, max) to point different ways. They don't:

| Reduction / subset | N | Length-controlled ρ | p |
|---|---|---|---|
| Mean pairwise (headline) | 48 | -0.276 | 0.061 |
| Median pairwise | 48 | -0.221 | 0.14 |
| Max pairwise | 48 | -0.160 | 0.28 |
| Median pairwise, new-cohort only | 24 | -0.355 | 0.097 |
| Max pairwise, longest-length tercile | 22 | -0.482 | 0.023 |

These aren't independent (they share an input matrix and subset partials share rows), but they're also not unrelated noise. Every operationalization of "output-isolated from peers" lands weakly negative under the length partial. The framing I'd land on is "weak diffuse negative signal opposite to the planned hypothesis, that the N=48 panel cannot distinguish from zero," not "five independent failures" and not "convincing evidence of an opposite-direction effect."

#### The cohort split disagrees

The full panel is 24 personas inherited from #271 + 24 personas added in #296. The new-cohort-only partial goes the other way from the inherited-cohort partial on the primary:

- Inherited cohort (n=24, the 24 carried forward): supplies the small raw positive correlation that the headline reports for the full panel
- New cohort (n=24, the 24 added in #296): p = 0.44 in the opposite direction

The full-panel partial is the average of two opposite-sign cohorts, not a clean zero. This is exactly the pattern the new-cohort-only subset check was planned to detect: "the predictor's small positive raw correlation lives entirely in the older cohort and gets absorbed by the length partial." On the primary, that prediction landed.

#### The cosine-from-assistant convergent check explains why the primary was always going to fail

On the 24 personas where both cosine-from-assistant (carried forward from #340) and output-distance from the assistant baseline are defined, the two distance measures rank-correlate at p = 5.8e-07, N=24. They're measuring nearly the same axis up to sign flip: a persona far from the assistant in residual-stream cosine is also far from the assistant in next-token distribution.

Two consequences for the headline. First, the prior negation of cosine-from-assistant in #340/#368 was already meaningful evidence that the primary would fail on this panel — the test was reasonable to run, but the strong rank-correlation between cosine and JS made same-direction failure likely rather than surprising. Second — and this is the part that matters for what to do next — the convergent argument covers only the primary (assistant-anchored) predictor. The secondary mean-pairwise has no fixed assistant anchor and is much less length-confounded; it is *not* pinned down by the cosine convergent. The cosine convergent kills the lineage hypothesis cleanly. It doesn't kill the "isolated-from-peers" hypothesis the secondary is testing.

#### Follow-up: cosine pairwise on N=24 disagrees in direction with JS pairwise

Given the convergent argument, the natural follow-up is the cosine-space analog of the secondary: mean pairwise L15 hidden-state cosine distance to other personas. The 24 inherited-cohort personas already have L15 centroids saved from [#274](https://eps.superkaiba.com/tasks/274), so this is a free analysis on N=24. The 24 new-cohort personas don't have published centroids and would need a fresh extraction pass.

![Length-residualized scatter of mean pairwise L15 cosine distance to other personas (x) vs source rate (y) across 24 inherited-cohort personas; the fit line is nearly flat with very wide vertical spread](https://raw.githubusercontent.com/superkaiba/explore-persona-space/<sha>/figures/issue_380/cosine_pairwise_n24_scatter.png)

Length-partial Spearman ρ = +0.111, p = 0.61, 95% CI [-0.37, +0.57]. The point estimate is in the originally-claimed direction (more isolated → more vulnerable) — *opposite* to the JS pairwise (which was negative). On N=24 alone the interval is too wide to call the disagreement real; both predictors bracket zero. But the disagreement is the open thread of the experiment: cosine-pairwise positive vs JS-pairwise negative on (mostly) the same personas. Resolving it would mean extracting L15 centroids for the 24 new-cohort personas and pushing cosine-pairwise to N=48, then comparing the two predictors side-by-side at matching N.

### What this all says

Six findings, three predictors, no winner. The headline I'd send to a mentor is: three flavors of geometric distance as predictors of marker source rate; none clears the planned threshold; the most informative result is the secondary's weak diffuse negative signal in the opposite direction from the planned hypothesis, which is open at N=48 and needs a larger panel or a cosine-pairwise convergent at N=48 to either confirm or kill.

The lineage hypothesis ("distance from assistant predicts marker vulnerability") is dead on this panel — same axis, same length confound, same null. The pairwise / isolation-from-peers hypothesis is *not* dead but the panel is too small to commit to the negative direction either. The pieces of evidence I'd weight most heavily for the open question are: (a) the sign-consistency across five pairwise reductions × subsets all landing weakly negative is more than chance-level coherence; (b) the cohort disagreement on the primary suggests the panel construction matters in ways the N=48 doesn't fully expose; (c) the cosine-pairwise positive on N=24 vs JS-pairwise negative on N=48 means the feature-space choice matters even after pairwise stripping the assistant-anchor confound.

What this updates: my prior that "geometric distance from assistant in some feature space" is the right family of predictors for marker source rate moves from "lineage hypothesis worth pushing on" to "probably wrong axis, even after the pairwise reformulation". The next predictor to try is more likely *capability-profile* or *family-membership* or *training-data-density* than "distance from X" — three things ruled out is enough to step off this axis.

### Parameters

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Panel | 48 personas (24 inherited from #271 + 24 added in #296) |
| Probe set | 20 `EVAL_QUESTIONS`, inherited from #207 |
| Baseline persona | `"Answer the user's question."` |
| Generation | vLLM 0.11.0, greedy (temperature 0, top_p 1.0), seed 42, max_new_tokens 512, max_model_len 2048 |
| Primary predictor | `compute_js_divergence(persona_logprobs, baseline_logprobs)` per probe, averaged over 20 probes |
| Secondary predictor | `compute_pairwise_divergences(kl_only=True, row_chunk=16, time_chunk=30)` on 48×48 matrix; per-persona reduction = mean over the 47 others |
| Length covariate | Qwen-2.5 tokenizer count of system-prompt string, log-transformed |
| Length-partialled rank correlation | `pingouin.partial_corr(method='spearman', covar=['log_tokens'])` (headline); inline rank-residualize-then-correlate as plan-named fallback |
| Resampling | 1,000 percentile resamples on length-controlled correlation |
| Positive threshold | \|length-controlled correlation\| ≥ 0.5 with p < 0.01 on at least one primary |
| Strict-null threshold | \|length-controlled correlation\| < 0.2 on both primaries AND resampled interval inside [-0.15, +0.15] |
| `pass_criterion_met` | `false` |
| `kill_criterion_met` | `false` |
| Hydra config | n/a (analysis-only) |

Confidence: MODERATE — the primary's length-controlled point estimate is essentially zero under both length-partial implementations and the resampled interval brackets it cleanly, but the strict null criterion was not met because the secondary's interval reaches far into the negative territory, the two analysis implementations disagree on the primary's sign, the panel is N=48 single-seed single-recipe, and the cosine-vs-JS pairwise disagreement is unresolved on the inherited subset.

---

## Annotations (for the analyzer, not part of the published body)

This section is part of the exemplar, not the body. It calls out where each rule landed.

**TL;DR many-findings rollup** (CLAUDE.md § TL;DR figure-pairing rule, Lens 9 mode 2). The Results bullet says "Six distinct things... [Per-finding figures and reads in Details.](#findings)" rather than crowding TL;DR with six sub-bullets each carrying its own figure. The rollup link's anchor `<a id="findings"></a>` resolves to the `### Findings` H3 in Details, which then carries the per-finding story beats. Lens 12 check #5 verifies the integrity (anchor resolves, finding count matches, each beat has setup + figure + read).

**Story-beat H3s** (Lens 4 + Lens 12 + analyzer.md anti-pattern #14). Compare original `### Headline result` → new `### The primary predictor dies on the same length confound that killed cosine`. Compare original `### Subset checks` → new (folded into the narrative of each story beat). Compare original `### Plan deviations` → new (folded into the narrative; the recut stratification appears in the helpful-family-out paragraph). Compare original `### Sample completions` → new (deferred to a separate body section; not included in this exemplar, but when present they'd appear in the story-beat where they support a claim).

**Figure setup + read paragraphs** (Lens 12 check #2 + analyzer.md anti-pattern #13). Every figure (or raw + processed pair) has a setup paragraph above ("The cleanest test of the lineage hypothesis is the primary predictor...") and a read paragraph below ("Raw association is Spearman ρ = +0.29...the raw signal is essentially all length."). Raw + processed pairs count as one narrative unit (single setup above the pair, single read below the pair).

**Surprises and pivots in the narrative** (Lens 12 check #3). The method-dependent sign-flip lives in its own H3 story beat (`### A method-dependent sign-flip on the primary headline`) rather than being parked at the bottom under `### Plan deviations`. The cohort disagreement lives in its own H3 (`### The cohort split disagrees`). The stratification recut appears inline ("The helpful-assistant-family personas anchor the short-prompt end of the panel... and the helpful-family-out subset confirms the pattern") rather than as a separate "Plan deviation #1: I recut the bins" entry.

**Interpretation beat distinct from Confidence sentence** (Lens 12 check #4). The "What this all says" beat names what the evidence as a whole means, what hypothesis is more/less likely, what alternative survives. It's NOT the same as the Confidence sentence (which lives after the Parameters table and names the binding constraint).

**Raw alongside processed** (SPEC.md § Voice (v3) + § `## Data` + Lens 11). Each finding's figure surface is the raw scatter on top + length-residualized scatter on the bottom. Per-row data link appears in the Reproducibility block (not shown in this exemplar but specified in the source body).

**Connective transitions allowed in Details** (SPEC.md § Voice (v3) scope — the no-fluff-transitions bullet explicitly welcomes connective tissue inside finding read prose). Inline phrases like "What I got was the first half cleanly", "Then I tried", "What I got:", "But that doesn't kill the pairwise hypothesis", "On the other hand", "Two consequences" all appear in the narrative and are NOT flagged. The "no fluff transitions" rule scopes to `## Human TL;DR` + `## TL;DR` only.

**Plain-English condition names end-to-end** (Lens 2/3/4). No Hydra slugs, no `Method A` / `Bin C`, no `M1` / `K1`. "Primary" / "Secondary" / "Mean pairwise" / "Inherited cohort" / "New cohort" / "Helpful-assistant family" are all plain English.
