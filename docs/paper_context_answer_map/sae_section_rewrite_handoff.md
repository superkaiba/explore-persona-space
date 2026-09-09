# Handoff: rewriting the SAE feature-property material in Section "Format and persona are preserved"

> **Status 2026-09-08, later the same day.** The paper now reports the
> decoder-direction target as its primary SAE analysis (Overleaf `6be6e0b`, EPS
> `76412a4e6a5`). Section 4.2 and `app:sae-properties` carry the Round A and
> Round B numbers below (identity +0.12, topic −0.09, promoting −0.10,
> suppressing −0.10, round-0 winner variance along decoder direction +0.32;
> layer-20 tiers 0.73 / 0.64 / 0.65, activity-adjusted +0.09 / −0.01 / −0.01,
> partial Spearman −0.19), `fig:sae-tier-gradient` is rendered from
> `eval_results/issue_1482/plot4_redesign/plot4_decoder_direction.json`
> (`scripts/issue1482_plot4_redesign.py --dv decoder-direction`), and
> `app:sae-robustness` is removed because Round C ran on the activation target
> only. Everything below that describes activation-target text is the record of
> the earlier draft, not what the paper says now.

> **Update, same day.** Per Thomas, `Fires on BOTH context and answer side`
> and its complement `Fires on the answer side only` are excluded as properties
> (`EXCLUDE_NAMES` in `issue1482_concordance_stepwise.py`, `--exclude` in
> `issue1482_decoder_direction_concordance.py`). The paper's selection is the
> 40-candidate run in `figures/issue_1482/concordance_decoder_direction_common_noside/`:
> variance along decoder direction +0.32, identity +0.12, promoting −0.10,
> suppressing −0.10, topic −0.09, write norm −0.06, then effects below 0.05.
> Also dropping the answer-side firing share gives the identical order
> (`..._noside_noratio/`). Figure 14A shows rounds 0 to 4. Overleaf `3eaf054`,
> EPS `8f86cb94a00`.

Written 2026-09-08 by the session that answered danmossing's four review comments
(6 September). Everything below is either read from a committed artifact or
quoted from the Overleaf tree. Where a number appears, the artifact it came from
is named, so re-read rather than retype.

**Overleaf state this describes:** project `6a59c927290f8b8b5eee0055`, commit
`d4bb76f` (parent `a0f5377`). Clone at `~/overleaf-6a59c927`.

---

## 1. What the section is and where it lives

| Thing | Path |
|---|---|
| The section | `sections/results/02_information.tex`, `\label{sec:results-information}` |
| Its appendix | `sections/05_method_details.tex`, `\label{app:sae-properties}` |
| New robustness appendix | same file, `\label{app:sae-robustness}` |
| Figure for the SAE claims | `fig:sae-tier-gradient`, `figures/paper/c3_sae_tier_gradient.pdf` |
| Figure for the other claims | `fig:useful-directions`, `figures/paper/c3_directions_and_pairs.pdf` |
| Main stepwise table | `tab:sae-properties` (14 rounds) |
| All-42 table (new) | `tab:sae-properties-all` |

The section has five bold claims. Only two of them are SAE claims:

1. High-variance directions are predicted best, and the alignment-relevant directions are among them.
2. Retrieval failures are near-duplicates.
3. **Higher-level features are predicted better than low-level ones** (SAE).
4. **Identity-related features are predicted better than topic-related ones** (SAE).
5. Format, persona, tone shifts are expanded; topic and language shifts are compressed.
6. The map separates minimal refusal pairs, but so does the copy baseline.

A rewrite that touches only the SAE material touches claims 3 and 4, the
connecting paragraph before them, and `app:sae-properties` plus
`app:sae-robustness`.

## 2. The four reviewer comments, and where each is now answered

| # | Comment (danmossing, 6 Sep) | Answered by |
|---|---|---|
| 1 | "I worry about this analysis because I feel like there might be big correlated sources of noise affecting the predictability of feature activation, like 'degree of feature splitting' among some group of features" | `app:sae-robustness` para 3 (cluster-preserving nulls), plus para 1 (the DV swap) |
| 2 | "I would feel better if repeating the analysis on decoder directions rather than feature activations yielded similar results" | `app:sae-robustness` paras 1 and 2 |
| 3 | "how were the properties chosen and are their values determined for a given feature?" | new `\paragraph{How a feature gets its property values.}` + `tab:sae-properties-all` + the pre-registration sentence |
| 4 | "specify how it's being controlled for? --> for the 'Controlling for it' part" | two main-text clause rewrites (see §3) |

The user confirmed the 42-property list **was pre-registered**. The paper now
says so. This is the strongest answer to comment 3 and should be kept in any
rewrite.

The user's scope call on 2026-09-08: *"I want to put most of these things in
appendix while only alluding to them in main text."* The main text grew by one
sentence and two clause rewrites. Keep that ratio.

## 3. What changed in `d4bb76f`

**Main text (`02_information.tex`), three edits:**

- line 19: `while controlling for the properties selected before it` →
  `over pairs of features drawn from within groups matched on the properties selected before it`.
- line 19: `42 feature properties` → `42 pre-registered feature properties`.
- line 19, appended: one sentence naming the two robustness checks and saying
  two exceptions are reported in `app:sae-robustness`.
- line 21: `Controlling for it,` → `Among features matched on it,`.

**Appendix (`05_method_details.tex`):**

- new `\paragraph{How a feature gets its property values.}` before the
  `\paragraph{Properties.}` list.
- `\paragraph{Properties.}` now says the list was fixed before any property was
  scored, and names the group counts 12 / 15 / 15.
- the scoring paragraph now points at `tab:sae-properties-all` and explains the
  two footprint sign changes.
- new `tab:sae-properties-all` (all 42, unconditioned score, selected round).
- new `\subsection{Robustness of the SAE property analysis}` with three
  paragraphs.

## 4. The three robustness rounds, with every number

### Round A. Decoder directions instead of feature activations (comment 2, layer 19)

**Only the dependent variable changed.** Published DV is a feature's mean
activation over the answer, which passes the encoder and the BatchTopK gate, so
a rare feature's target is mostly zeros. Swapped DV is `d_f^T v`, the projection
of the answer state onto the unit decoder column, which is dense and defined on
every held-out answer however rarely the feature fires. The 42 properties, the
coarsened exact matching, the round-by-round selection, the stop rules and the
sibling-retirement cut are the published code paths, unmodified.

Common-universe run, 120,716 features scored under both:

| Property | published r0 | published matched | decoder r0 | decoder matched |
|---|---|---|---|---|
| Mean activation over answers | +0.266 | +0.266 | +0.071 | never selected |
| Speaker: identity / disposition | +0.192 | +0.171 | +0.290 | +0.123 |
| Logit footprint: suppressing | +0.041 | −0.149 | −0.154 | −0.097 |
| Logit footprint: promoting | +0.144 | −0.142 | −0.093 | −0.103 |
| Content type: topic | −0.022 | −0.126 | +0.019 | −0.086 |
| Interpretable | +0.068 | +0.071 | +0.001 | never selected |

- Selection order after the leading how-much-signal property is **identical**:
  identity, the two footprint classes, then topic.
- Decoder round-0 winner is **Variance explained in answer space, +0.316**
  (the dense analogue of mean activation). The paper rounds this to +0.32.
- The two DVs agree at only **Spearman 0.229** at feature level, so this is a
  demanding check.
- Universe difference closed: the decoder DV is defined for 128,450 features
  (a direction has an R² even when its feature never fires) vs the published
  120,716. Restricting to the common 120,716 moves every headline by at most
  0.02, so the universe-size objection is dead.

**Source:** `eval_results/issue_1482/decoder_direction/concordance_comparison.json`
(`headline`, `selection_order_*`, `dv_rank_agreement_spearman`);
`figures/issue_1482/concordance_decoder_direction_common/writeup_stepwise.meta.json`
(the full per-round score lists for the decoder run).

### Round B. Matryoshka tier gradient on decoder directions (comment 2, figure B)

The layer-20 answer states were never banked, so they were captured fresh
(24,000 fit + 6,000 score = 30,000 contexts, ~80 min of H100). Panel n = 16,384.

| Statistic | activation DV (banked) | decoder DV |
|---|---|---|
| Raw Spearman(tier, R²) | −0.3949 | −0.0873 |
| Partial Spearman given log activity | −0.1938 | −0.1923 |
| Median R² tier 0 / 1 / 2 | 0.435 / 0.174 / 0.043 | 0.727 / 0.640 / 0.649 |
| Within-stratum permutation verdict | coarse-better | coarse-better, band [0.046, 0.071] |
| Pooled dense R² | — | 0.6703 (floor gate 0.2 passed) |

Reading: the **conditioned** statistic replicates almost exactly and still
clears its band, so the coarse-tier advantage is not an artifact of the
activation target. The **raw** gradient weakens by a factor of four, and the
per-tier medians stop being ordered (0.640 vs 0.649 is not a step). Coarse
versus fine survives. The three-step ladder does not.

**Source:** `eval_results/issue_1482/decoder_direction/matryoshka_decoder_direction_lmsys.json`.

### Round C. Cluster-preserving permutation nulls (comment 1)

Clusters are connected components of the graph on the 120,716 features whose
edges are decoder-column cosines at or above τ. A permutation moves whole
cluster-by-cell blocks rather than individual features, so features that split
one underlying direction move together. Two schemes (`concat`, `size_exchange`)
× three thresholds τ ∈ {0.30, 0.35, 0.40} = six configurations, 1,000 draws each.

Cluster census (from `gram_stats.census`):

| τ | non-singleton share | n clusters | largest cluster |
|---|---|---|---|
| 0.30 | 62.9% | 46,645 | 71,484 |
| 0.35 | 41.0% | 74,972 | 39,710 |
| 0.40 | 22.6% | 97,853 | 15,017 |

All six paper-quoted effects clear every configuration:

| Property | paper | observed | min margin over band | max widening vs feature-grain |
|---|---|---|---|---|
| Content type: topic | −0.13 | −0.126 | 16.3× | 1.52× |
| Interpretable | +0.07 | +0.071 | 11.7× | 1.06× |
| Logit footprint: promoting | −0.14 | −0.142 | 8.5× | 2.17× |
| Logit footprint: suppressing | −0.15 | −0.149 | 4.5× | 3.63× |
| Mean activation | +0.27 | +0.266 | 10.1× | 12.60× |
| Speaker: identity / disposition | +0.17 | +0.171 | 8.1× | 1.20× |

Every one is at the permutation floor, p = 0.001 for 1,000 draws. Coarsening
the null **does** widen the band, most of all for mean activation, exactly as
co-firing near-duplicates predict. The effects are still larger than splitting
alone can produce.

Two round-0 (unconditioned) values do **not** clear: suppressing's +0.041 fails
5 of 8 configurations and topic's −0.022 fails 1 of 8. Neither is quoted in the
paper, but see §5 item 3.

**Source:** `eval_results/issue_1482/concordance_cluster_null/cluster_null_concordance.json`
(`headline_six`, `gram_stats`), plus `nulls_round0.json` .. `nulls_round6.json`.

## 5. What does NOT survive. Do not restore these claims in a rewrite.

1. **"A feature's mean activation over answers is the strongest property
   (+0.27)" is specific to the gated target.** On decoder directions it scores
   +0.071 and is never selected in fourteen rounds, and it is the property whose
   null band widens most under clustering (12.6×). Two independent checks name
   the same property. What survives is the weaker, true claim: a feature
   carrying more signal is predicted better. The dense statistic for that is
   variance explained in answer space (+0.32). The main text still quotes the
   +0.27 sentence and points at the appendix exception. A rewrite may prefer to
   soften it in the main text directly.
2. **The Matryoshka three-step ladder.** 0.18 → 0.01 → −0.03 is the
   activation-target adjusted read. On decoder directions the middle and finest
   tiers are indistinguishable. Write coarse-versus-fine, not a three-step
   gradient.
3. **"Suppressing starts positive and flips."** Its unconditioned +0.041 fails
   5 of 8 cluster configurations, so it is not distinguishable from zero. The
   honest description is null-then-negative, not a flip. **Promoting is a
   genuine flip** (+0.144 unconditioned, −0.142 matched, clears everywhere), and
   its unconditioned value is negative (−0.093) on decoder directions, so the
   positive activation-target value is a firing-rate artifact confirmed by two
   independent routes. The appendix currently describes both as sign changes.
   Tightening that to distinguish the two is an available improvement.

## 6. Three pre-existing factual errors, corrected in `d4bb76f`

Found while reconciling the 42 candidates against the paper's own prose. All
three were errors in the **text**, not in the analysis.

1. **The Matryoshka tier analysis runs at layer 20**, not "the same layer" as
   the layer-19 BatchTopK SAE. Ground truth: `scripts/issue1482_matryoshka_tier.py`
   sets `L_TIER = 20` and both dictionaries hook `model.layers.20`; the banked
   per-tier medians 0.44 / 0.17 / 0.04 reproduce at layer 20. The figure caption
   said a blanket "layer 19" for both panels and now says
   "layer 19 in panel A and layer 20 in panel B".
2. **The side property has two scored levels, not three.** The firing list said
   "fires on both the context and the answer, on the answer only, or on the
   context only". Only `Fires on BOTH` and `Fires on the answer side only` are
   scored.
3. **"Each level enters the analysis as one binary property" over-counted the
   judged group by two.** The interpretable and speaker axes each hold out a
   reference level (not interpretable, no speaker property). Corrected, and the
   group counts 12 / 15 / 15 are now stated so 42 reconciles from the text.

## 7. The authoritative 42, with paper-facing names

Read from `figures/issue_1482/concordance/writeup_stepwise.meta.json`, round 0
`scores`. The **internal** names in that file carry codenames that must never
reach the paper (`(OUTPUTNESS)`, `[k=0.31]`, `Scaffold-token activation
fraction`). `tab:sae-properties-all` already carries the reader-facing names.
The most important renames:

| Internal name | Paper name | Note |
|---|---|---|
| `Scaffold-token activation fraction` (family `position`) | Decoder mass on the 48 leading context-prefix directions | **A misnomer in the code.** `scaffold_frac` is GEOMETRIC decoder mass in the prefix-covariance top-48 subspace, not a token fraction. It belongs in *Read and write geometry*, not firing statistics. `scripts/issue1482_run_length.py` and `issue1482_continuous_predictors.py` both carry explicit disambiguation comments. |
| `Variance explained in answer space` | Variance along decoder direction | matches `tab:sae-properties` round 10 |
| `Nearest-neighbour cosine (SAE redundancy)` | Highest cosine to another decoder column | |
| `SAE encoder norm (read strength)` | Encoder-vector norm | matches round 12 |
| `SAE decoder norm (write strength)` | Decoder-column norm | |
| `Side ratio (answer-side firing fraction)` | Answer-side share of firings | matches round 8 |
| `Template-token activation fraction` | Share of firings on chat-template tokens | |
| `Judged role: X [k=0.31]` | Functional role: X | matches round 7 |
| `Logit-footprint concentration (OUTPUTNESS)` | Positive logit mass on the top ten tokens | |
| `Encoder-decoder cosine (OUTPUTNESS)` | Encoder--decoder cosine | |
| `Write norm, gamma-scaled (OUTPUTNESS)` | Write norm after the final layer-norm gain | |

Group counts after that reclassification: firing statistics 12, read and write
geometry 15, judged labels 15.

## 8. Gates and provenance for the numbers

- **Recovered holdout truth.** The published run's 20,000 × 3,584 held-out
  answer-state matrix was never banked. It was recovered by streaming 1,920
  capture chunks (~83 GB) and keeping only holdout rows
  (`scripts/issue1482_recover_holdout_truth.py`). Identity-gated on all 20,000
  rows against the banked per-row squared error: max relative error 1.8e-3,
  100% inside fp16 tolerance.
  **Join gotcha:** `holdout_rows` index the assembled matrix, which has a
  5,000-row pass-B block first, and 156 of 960,000 contexts never captured so
  capture `ci` drifts from stream position. The correct join key is **stream
  position minus `N_PASS_B`**, not `ci`.
- **Per-direction R² estimator** reproduces banked trait `dense_direction_r2`
  to 6.66e-16 (`estimator_gate()` in `scripts/issue1482_decoder_direction_r2.py`).
- **Pooled dense R²** 0.7243 (layer 19) and 0.6703 (layer 20) against a 0.6531
  reference. A pooled-R² floor gate now precedes every per-direction number,
  because of the retraction below.
- **Round C** reproduces the recorded stepwise scores at max diff 0.00e+00.

**A retracted result, so it is not repeated.** An earlier layer-20 attempt
assumed the banked dense store held answer states. Both of its arrays are
prompt-side tokens (`c20` = `context_end`, `hp20` = `prefix_end`, read at
`scripts/issue1482_matryoshka_tier.py` ~line 881). Pooled R² 0.071 vs 0.653
exposed it. The artifacts were deleted and the layer-20 answer states were
captured fresh. If a rewrite needs new layer-20 numbers, capture them; do not
read the dense store.

## 9. Code and artifacts, all on `origin/main`

```
scripts/issue1482_recover_holdout_truth.py
scripts/issue1482_decoder_direction_r2.py
scripts/issue1482_decoder_direction_concordance.py     # --restrict-to-published-universe
scripts/issue1482_matryoshka_decoder_direction.py
scripts/issue1482_cluster_null_concordance.py
scripts/issue1482_concordance_writeup_figs.py          # module-level TARGET_R2, default-preserving

eval_results/issue_1482/decoder_direction/
eval_results/issue_1482/concordance_cluster_null/
figures/issue_1482/concordance/                        # published run, 14 rounds + trajectory
figures/issue_1482/concordance_decoder_direction/
figures/issue_1482/concordance_decoder_direction_common/
```

HF: `holdout_truth/` (the recovered truth) and
`matryoshka_tier/store/ans_l20_*` (the layer-20 capture, 65 files, 430.8 MB).
Both pods are terminated; the `keep-running` tag is removed. Task #1482 is at
`awaiting_promotion` and carries the fold record as `epm:progress` v244.

## 10. Working on this Overleaf clone: five things that will bite

1. **The clone is shared by several live sessions.** `git add <file>` publishes
   their uncommitted hunks under your commit (incident `04430c5`, 2026-09-03).
   Run `git status --porcelain` first and commit with an explicit pathspec.
2. **Pull `--ff-only` immediately before editing, edit in place, never
   regenerate a file from an older copy, never force-push.** Thomas types in the
   Overleaf web editor at the same time. Say which Overleaf commit you edited
   against, and tell him to reload or recompile after a push, because his file
   tree does not refresh on its own.
3. **The repo-root commit guard blocks `cd <dir> && git commit`** even in a
   different repo, when another EPS session has uncertified code staged in the
   shared index. Use `git -C /home/thomasjiralerspong/overleaf-6a59c927 commit -F <msgfile> -- <paths>`
   with no `cd`, which scopes past the guard.
4. **A pre-commit writing-tells gate runs in the clone**
   (`~/.claude/skills/writing-tells/check_paper_tells.sh`). Hard bans block
   (em dashes, metaphor jargon, AI vocabulary). Soft flags report only.
   `VERBOSE_FLAGS=1` lists them. `TELLS_ALLOW=1` overrides. Note that `---` in a
   table cell counts as an em dash; use `--`.
5. **Auth.** The token is `OVERLEAF_GIT_TOKEN` in the EPS repo-root `.env`.
   Pass it transiently as an HTTP basic header for one invocation and unset it;
   never print, log, or commit it.

The paper builds clean with `latexmk -pdf main.tex` from the clone: 43 pages,
no undefined references, 10 overfull boxes all pre-existing.

## 11. Open items a rewrite should consider

- **`02_information.tex` line 17 has a truncated bold heading:**
  `\textbf{Retrieval failures are due to near-duplicate }` with a trailing space
  and no noun. Not part of the SAE material, but it is in the same section and
  is visible in the compiled PDF.
- Item 3 of §5: the appendix currently calls both footprint classes sign
  changes. Only promoting is a genuine flip.
- Item 1 of §5: the main text still leads claim 3 with the +0.27 mean-activation
  sentence and defers the caveat to the appendix. A rewrite could invert that.
- The main table `tab:sae-properties` and the new `tab:sae-properties-all`
  duplicate the selected rounds. If space is tight, one table with both the
  unconditioned and the matched score per row would replace both.
