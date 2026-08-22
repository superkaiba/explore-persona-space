# Adversarial critique — `outline.tex` (2026-08-22 restructure)

Object under review: the blue PLAN blocks in `~/overleaf-6a59c927/outline.tex`
(2026-08-22, 3-result spine), checked against `claims.md` rev 3 (+ the two
results that landed after it: #2379, #2356), `plan.md`'s restructure record,
and the draft sections the outline `\input`s (`c1_linear.tex`,
`c3_highlevel.tex`, `c4_persona_universal.tex`, `c5_useful.tex`,
`c2_context_vector.tex`, abstract, introduction). Findings ranked
most-severe first. Each: claim → problem → evidence → fix.

---

## F1 (SEVERE) — The introduction's finding 4 and the abstract's application sentence assert four applications the evidence refutes or reverses

**Where:** `01_introduction.tex` item 4 of the "We find:" list; `00_abstract.tex`
("We show applications to predicting behavior pre-generation, predicting the
behavioral effects of fine-tuning, and to automated redteaming/jailbreaking").

**Problem:** these are reader-facing sentences, not comments, and four of the
five clauses are now adverse for the map:

- *"to control model character"* — the map's read direction is causally inert
  (best Δ judged rate +0.06 evil / +0.01 sycophancy) while the persona vector
  steers +0.985 / +0.429 at the same operating points; read direction ⊥ persona
  vector, cos 0.00–0.03 (#2220, #2254). This is a NEGATIVE result presented as
  an application. The draft's own `% WARNING` comment says so; the visible text
  does not.
- *"to automate redteaming/jailbreaking"* — in the only dedicated experiment
  (#2394), a plain probe on v_C finds always-comply contexts at PR-AUC 0.973,
  equal to the real-answer oracle (0.974), and **every map arm loses or ties**
  (fixed map-then-project ≤ 0.43; probe-on-mapped = reparametrization of the
  context probe; probe reaches PR 0.80 with 10 labels vs ~47–51 through the
  map). The abstract claims as a map application a task the map demonstrably
  adds nothing to.
- *"including re-elicitation from inoculation prompting"* — #2379 (landed
  2026-08-21): the context-side predictor replicates the prior-work association
  (0.775 misalignment / 0.895 capitalization, beats text embeddings), and the
  **mapped-answer readout shows a Δρ = −0.86 pooled deficit** (interval
  excluding zero). Adverse for the map readout; the positive result is Kwon-style
  v_C prediction, not the map.
- *"to screen training data"* — the frozen map's top-500 selections **never
  beat random** and correlate with exact ΔP at only −0.09 to 0.14; only a
  target-corpus refit rescues (0.73–0.86), and only at screen-correlation
  level — no selection-then-finetune has used refit selections (#2224/#2222).

Only "predict behavior pre-generation" and "predict the effect of fine-tuning"
(#1979) survive as stated, and the first only in weak form (see F3).

**Fix:** rewrite intro item 4 and the abstract's application sentence around the
honest split: (a) map-positive — fine-tuning-change prediction (#1979, median ρ
0.41 vs 0.30, family-scoped), PV-mismatch repair (writeup claims 1–2), the
weak-form probe-on-mapped margin; (b) v_C-positive-map-neutral — jailbreak
mining, refusal monitoring, re-elicitation (report these as pre-generation
wins for the *context vector*, with the map as an explicit boundary);
(c) negative — control/steering, frozen-map screening. Do not wait for the
"claims-audit pass" comment: these sentences set reviewer expectations for the
whole paper.

## F2 (SEVERE) — Recurring misattribution: v_C-probe wins credited to "the mapping"

**Where:** outline Results I bullet 3 ("refusal: context probe AUROC
0.995/0.951 beats an LLM judge, map matches but adds no decision signal
[DONE #2356]" cited as evidence the map "predicts *meaningful* parts");
Results III bullets on #2394 and #2379; abstract's "linearly predictable from a
single context vector" narrative.

**Problem:** the paper's object is the map W, but in all three of the newest
realistic-task results the predictive signal lives in v_C and W adds nothing:

- #2356: the probe that beats the judge (0.995/0.951 vs 0.896/0.743) is fit on
  **v_C**. The map result there is: held-out R² 0.66 vs identity −0.96, but its
  probe "matches the context probe **and matched-rank PCA control**, adding no
  detected decision signal" — i.e. the map's contribution is not even
  map-specific for this DV. Citing #2356 under "the map predicts meaningful
  parts" is a category error.
- #2394: probe-on-v_C equals the answer-space oracle; map loses everywhere.
- #2379: context-side 0.775/0.895; mapped readout −0.86 deficit.

Since a linear probe on v_C is a strictly cheaper object than probe∘W, every
one of these can be read by a reviewer as "the paper's object is unnecessary
for its own applications." The outline's #2394 bullet does say "frame as a
v_C/pre-generation win + map boundary" — the same framing must be applied to
#2356 and #2379, and Results I bullet 3 must not enlist #2356 for the map.

**Fix:** in Results I bullet 3, replace the #2356 clause with "refusal
tendency is decodable **pre-generation from v_C** (AUROC 0.995/0.951, beats an
LLM judge); the label-blind map reproduces but does not exceed the context
probe (matched-rank PCA control) [#2356]" — that supports "the *answer
representation this position feeds* carries refusal," not "the map predicts
refusal." Alternatively make the Results I claim about the context position
(v_C) and reserve map-specific claims for cells where the map separates from
the context probe. Also carry #2356's own caveat: the probe predicts the
prompt's 10-rollout refusal *tendency*, not a single generation's decision
(answer-minus-context interval spans zero).

## F3 (SEVERE) — Results III title "The Mapping Is Useful" is unconditional; the evidence supports only a conditional, weak-form claim

**Where:** section title + Results III PLAN block.

**Problem:** the section's own contents are now majority-adverse or
conditional. The controlled flagship (#1739 claim4-controls): margin over
shuffled-pairing control positive 12/13 rungs but **median +0.03**; only 1 of
2 pre-registered flagships clears its interval (syco model-written-evals
margin +0.137 [+0.069,+0.204]; evil PAIR crosses zero [−0.006,+0.085]);
mimicry is an interval-separated raw loss (−0.196). The mechanism cell (the
composition grid, "in-domain unjudged pool flips the win") is **16 cells,
single draw, single seed, evil-only**, and the unjudged-data attribution has
no pool-volume ablation ("still an interpretation," claims.md). claims.md's
own summary: *no cell anywhere where a frozen-generic map-then-project beats a
matched-budget context probe on a well-instrumented, non-confounded cell* —
the wins concentrate under four conditions (in-domain unjudged pool; scarce
labels; structurally degraded context-side read; change-DV).

**Fix:** either retitle to the conditional ("The mapping is useful where
context-side reads degrade or labels are scarce" / "…converts unjudged
in-domain text into pre-generation prediction") or keep the title and open the
section with the four-condition boundary statement, before any positive cell.
In the PLAN block, quote the weak-form magnitudes next to the claim (median
margin +0.03; 1/2 flagships), not only in the figure. The unconditional
pillars to lead with are #1979 (FT-change champion) and the PV-mismatch repair
(claims 1–2), not claim 4.

## F4 (HIGH) — Results II title clause "stronger for assistant-like characters" rests on n = 4

**Where:** Results II section title + bullet 2 ("closer to assistant = stronger
(AI-likeness ρ +0.80, n=4)").

**Problem:** a Spearman ρ over four characters cannot clear conventional
significance even at ρ = 1 (one-sided p ≈ 0.083; at ρ = +0.80, p ≈ 0.17).
claims.md caps it as "consistent-with rather than established," and the
outline itself notes a dedicated AI-likeness run "would be NEW." A section
*title* is the strongest claim surface in the paper; this one is grounded by a
4-point trend plus a qualitative judge gradient.

**Fix:** either run the dedicated AI-likeness experiment (more characters,
pre-registered axis — this is cheap relative to its title-level load and fits
the 4-week window) or demote the clause from the title to a
"consistent with" sentence in the section body.

## F5 (HIGH) — "Mostly present in the base model (~87%)" is R²-scoped and, as stated, violates the paper's own dual-metric convention

**Where:** Results II bullet 5; Results II title clause "mostly present in the
base model"; intro item 2.

**Problem:** the ~87% figure is the R² ratio 0.588/0.673 (#825, chat, n > d —
sound). But the outline's Global-conventions bullet mandates R² **and** acc@1
with "dissociation shown never averaged," and on the retrieval metric the base
map is far from 87%: kNN retrieval **jumps base→SFT from 0.13–0.41 to
0.42–0.75** (#2061), and OLMo-2 aligned retention for base→SFT is **0.472**
(#1902) — i.e. SFT moves the map's coordinates a lot even where map *strength*
pre-exists. The same PLAN bullet cites #2061 and says "SFT big," so the bullet
currently contains its own counter-evidence without reconciling it.

**Fix:** state the two-metric split explicitly: "map strength largely present
in the base model (R² 87% of instruct); SFT substantially rewrites its
coordinates and sharpens retrieval (kNN 0.13–0.41 → 0.42–0.75; aligned
retention 0.472), later stages barely move it (DPO→RLVR 0.991)." Do not let
"mostly present" stand alone in the title without the retrieval qualifier.

## F6 (HIGH) — Causality residue after the demotion: a causal-patching paragraph + figure survives in Results II with no PLAN bullet, and plan.md's contributions still say "causally load-bearing"

**Where:** `c4_persona_universal.tex` final paragraph "Patching the context
vector affects the persona of the entire answer" + `fig:c4-persistence`;
`plan.md` contributions item 2 ("causally load-bearing … uniquely among
slots"); outline Global conventions bullet correctly demotes causality.

**Problem:** (a) internal inconsistency of the restructure — the Results II
PLAN block has no bullet for the patch-persistence paragraph, so the review
surface (PLAN) and the shipped draft diverge; a causal claim the restructure
demoted still headlines a Results II paragraph. (b) The claim itself is
qualified by #2333: on format cells **67% of the patch effect is recovered by
prefilling the patch's own 3 opening tokens** (majority opening-token-carried,
a snowball account; ~a third is not), with the state-level residual reliable
only on the Qwen3.5 language cells (prefill 40% null-adjusted, CI below 1).
"Affects the persona of the entire answer" as a state-level causal claim
overstates what survives #2333. (c) plan.md item 2's "causally load-bearing"
contribution clause predates the demotion.

**Fix:** move the persistence paragraph + figure to the appendix with old C2
(or add a PLAN bullet that owns it and carries the #2333 qualification
inline); rewrite the contribution clause to the demoted form ("context-end
patches move behavior 0.18–0.63 of a full swap, majority opening-token-carried
on format cells; the map does not predict the induced shift").

## F7 (HIGH) — The identity+bias winner-flip (#1901 vs #1336 vs #2215) was dropped from the outline entirely

**Where:** Results I PLAN block (scaling + layer bullets); `c1_linear.tex`
comment "REMOVED (Thomas review 2026-08-19): … the Results-level winner-flips
bullet (the two-metric rationale stays in Methodology)."

**Problem:** what was removed is not just a two-metric rationale. On Llama
Tülu, **identity+bias wins retrieval rank-1 in 9/10 pooled cells** (#1336);
at context-end minimal pairs, identity+bias captures most discrimination and
the fitted map's increment (+0.9 pts) has a CI including zero (#2215). The
headline "fitted maps beat every baseline" is therefore Qwen-, metric-, and
training-size-scoped, and the paper claims fits "across models, model
families" (abstract). claims.md Contradiction 3's instruction — "show the
R²-vs-retrieval dissociation, don't average it away" — currently has no home
anywhere in the outline.

**Fix:** one sentence in Results I scoping the baseline comparison ("on Qwen
at large n; on Llama Tülu identity+bias wins pooled retrieval — the
dissociation is model- and metric-dependent, Appendix X") + an appendix row.
Cheap, and it pre-empts the most mechanical reviewer check (running the
baseline on the other family).

## F8 (MEDIUM-HIGH) — Abstract "(ii) independent of the chat template" contradicts Results II's own title and #2054

**Where:** `00_abstract.tex` finding (ii) "…and independent of the chat
template"; intro item 2 "not dependent on the chat template"; Results II title
"…stronger with the chat template."

**Problem:** three mutually inconsistent strengths of the same claim. The
evidence: refitting without the template costs the instruct model only ~0.03
R² (#825) — supports "does not require the template" — but the two framings
are **not the same operator in the same coordinates** (direct cross-framing
transfer loses 0.10 of ceiling instruct / 0.40–0.46 base, #1345), the
answer-boundary form is precisely what breaks transfer (median −0.06 across 56
boundary-swap pairs vs 0.20 prose swaps, #2054), and chat cells are ~2× the
story cells (+0.609/+0.567 vs +0.367/+0.262). "Independent of the chat
template" is the strongest and least supported of the three phrasings.

**Fix:** unify on "exists without the chat template (refit cost ~0.03 R²) but
is framing-indexed: the boundary form carries the framing cost and transfer
requires a linear change of coordinates." Fix the abstract first — it is the
sentence reviewers will quote back.

## F9 (MEDIUM-HIGH) — "Distinction only arises after SFT" is false as written; the outline's VERIFY restatement is correct and should simply replace it

**Where:** Results II bullet 3.

**Problem:** #1310 finds character-specific maps in the **base** model
(correct-vs-swap pooled +0.295 base / +0.381 instruct, CIs excluding zero), so
character distinction does not "arise after SFT." What sharpens after SFT is
retrieval (kNN base→SFT jump, #2061) and cross-framing operator alignment
(Procrustes cosine 0.855 instruct vs 0.732 base, #1345). The outline already
flags this [VERIFY] with the right restatement.

**Fix:** delete the original sentence rather than caveating it; adopt the
restatement in the flag verbatim. (Also applies to any Doc/notes copy of the
outline that still carries the original.)

## F10 (MEDIUM) — #2394 is cited at paper grade from an unreviewed scratch report and staging-store JSONs

**Where:** Results III jailbreak-mining bullet; `c5_useful.tex` full paragraph
+ `fig:c5-jbmine`.

**Problem:** #2394 completed as `kind: analysis` with a scratch report
(`docs/scratch/jailbreak_mining_pilot.md` @ cb1f5f836c), **no clean-result
body, no critic pass**, and its headline JSONs
(`map_arms_results.json`, `label_efficiency_results.json`) are verified from
`/mnt/eps-data` staging rather than committed `eval_results/`. Every other
number in the section passed the clean-result + critic pipeline; this one —
load-bearing for the paper's honesty story — did not.

**Fix:** before submission, commit the pilot's eval JSONs to
`eval_results/issue_2394/` (or the #1739 tree it staged under) and run one
verification pass over the PR-AUC numbers (a 0-GPU re-read). If the pilot's
5%-base-rate construction or same-family negatives wouldn't survive review,
better to learn now.

## F11 (MEDIUM) — Results I title tension: "predicts the high-level parts" vs 0.80–0.99 exact-answer retrieval

**Where:** Results I title + bullets 1 and 4.

**Problem:** the same section claims median specific-tier SAE feature R² of
0.04 (#1482) *and* rank-1 retrieval of the exact answer at 0.80 in
1,000-candidate pools (0.991–0.995 after metric fixes, #2202). A reviewer will
ask how a map that predicts only "high-level parts" singles out the exact
answer among ~10⁴ candidates. The resolution exists in the evidence
(retrieval is carried by high-variance directions; the predictable subspace
*is* the high-variance subspace, #1895; #2202's 0.99 is on the 1,988
resample-covered subset) but no PLAN bullet connects them.

**Fix:** add one bridging sentence to the Results I plan: "high-variance
(persona-grain) structure suffices to individuate answers in retrieval; the
poorly-predicted specific tier contributes little variance" — and keep the
#2202 0.99 numbers scoped to the resample-covered subset when quoted.

## F12 (MEDIUM) — #2379's context-side replication is quoted without its own thinness

**Where:** Results III re-elicitation bullet ("context-side predictor
replicates the prior-work association (0.775/0.895, beats text embeddings)").

**Problem:** the replication itself is 1 seed, 1 model; misalignment pools
n = 18 trigger prompts; capitalization is German-dominated (−0.615 vs −0.142 /
−0.115) with 2 of 3 per-language gap intervals crossing zero and the Spanish
install missing its 0.50 floor; misalignment was read under this run's judge
with parity unmeasured. Fine as an adverse-for-map cell; thin as a positive
"beats text embeddings" claim.

**Fix:** carry "1 seed, 1 model, n=18 prompts (misalignment)" wherever
0.775/0.895 is quoted; if the row is promoted to a positive claim, #2474
(base-geometry follow-up) or a second-seed rerun should land first.

## F13 (MEDIUM) — The Global-conventions bullet is unenforceable as written, and the qualitative-examples leg does not exist yet

**Where:** Global conventions ("Every claim reports R² *and* acc@1 *and*
qualitative examples, always vs. the identity+bias baseline").

**Problem:** (a) several claim families structurally cannot produce acc@1 or
an identity+bias comparison (per-direction R² reads, SAE-tier medians,
ρ-valued behavior predictors, aligned-retention ladders) — as written the
convention is violated by half the paper's own results; (b) the
qualitative-examples leg currently has **zero** assembled artifacts (the
#2094/#2162 panel is a TODO; no qualitative panel exists for Results I/II/III
at all).

**Fix:** scope the convention to whole-vector map-quality claims; schedule the
qualitative panel as a real task (it is also the cheapest reviewer-persuasion
artifact the paper lacks). Add per-section "which convention legs apply"
notes to the PLAN blocks.

## F14 (MEDIUM) — Discussion plan lacks the paper's own unifying limitation: corpus transfer

**Where:** Discussion PLAN block (linearity / PSM / nonlinear metamodels /
causality only).

**Problem:** the single mechanism that unifies the paper's adverse cells —
#2222/#2224 frozen-map screening failure, #2394 benign-fit reconstruction
failure (recon R² −0.12..−0.88 vs +0.33..+0.62 in-domain), #1739's OOD
regimes, #779's read-out-transfer bottleneck — is "the failure mode is corpus
transfer of the map, not the linear form" (claims.md). It appears nowhere in
the Discussion plan, yet it is both the honest limitation and the paper's best
defense of the linear form.

**Fix:** add a Discussion bullet: "the map's boundary is corpus transfer, not
linearity — refits rescue screening (0.73–0.86) and suite transfer; frozen
maps fail across corpora"; cross-reference from the Results III boundary
paragraphs. Also add the #2220/#2254 read-vs-steer geometry split to the PSM
discussion bullet (the map's read direction is orthogonal to the persona
vector, cos 0.00–0.03 — relevant to how strongly "evidence for the persona
selection model" can be phrased).

## F15 (LOW-MEDIUM) — Wording risks the outline already flags correctly (confirming, with one addition each)

- **"Stronger in chat" flip** (Results II bullet 1): the outline's inline flag
  is right — evidence is chat +0.609/+0.567 vs story +0.367/+0.262 (#1345,
  corrected estimator). Addition: keep the "constructed rigs; wild fiction
  untested (#931 estimator-bugged)" scope in the section claim, as the c4
  draft already does — the PLAN bullet should carry it too.
- **"One shared persona mapping" title** vs #1417 (map survives no-persona
  framings, 11/11 shared — "tracks generic query-answering"): the c4 draft
  carries the scope-caveat paragraph but the Results II PLAN block has no
  #1417 bullet, so the reviewed surface asserts persona-specificity the
  control disclaims. Add the bullet; per claims.md Contradiction 6, prefer
  "character-indexed up to reparameterization" over "persona mapping" in the
  title if space allows.
- **Pooled-map gloss "the model treats these characters the same way"**: a
  mechanism gloss on a fit-recovery result, and the character-specific slope
  residual is real (+0.007..+0.025, every CI above zero). Say "nearly the same
  way; a small reliable character-specific residual remains."
- **"Mostly linear"**: sound as drafted, but keep citing the deduplicated 1M
  MLP gain (≈ +0.056) rather than the contaminated banked 50k comparison
  (claims.md iterations C1), and keep the nonlinear share's growth with pool
  difficulty (+0.036 → +0.101 at 100k candidates) visible.
- **WildChat Kendall τ = 1.0**: quote with #1901's own scope ("held-out-row
  within training distribution, not corpus-OOD") so it is not read as an OOD
  robustness claim (K7 says corpus-OOD is the recurring failure).

---

## Priority experiment / verification list (~4 weeks to ICLR abstracts 2026-09-18)

1. **Multi-seed × 3-behavior rerun of the evil composition grid** (#1739
   Regime 1). The paper's central mechanistic claim (in-domain unjudged pool
   flips the win) is n=1, single-seed, evil-only. claims.md already calls this
   the single most valuable confirmatory run. Without it, F3's conditional
   flagship has one behavior and no error bars.
2. **Fix and rerun the arm2 (context-extracted direction) comparator under
   P-B** (current rerun inconclusive, adapter-suspect). This is the natural
   reviewer objection to all of Results III ("why not extract the direction at
   the context?") and the paper currently cannot answer it.
3. **Single generic-boundary-token control** for the C1 scaling figure
   (outline NEW; cheap) — closes the "trivial next-span prediction" hole in
   the headline figure.
4. **Dedicated AI-likeness run** (more than 4 characters) — a section-title
   claim currently rests on n=4 (F4). If not run, soften the title.
5. **#2394 artifact hardening** (commit JSONs + one verification pass) — F10.
6. **Unjudged-pool-volume ablation** (claim-4 attribution is still "an
   interpretation") — can piggyback on run 1's grid.
7. **Base-coherence VERIFY** (#1336 in-flight round) — gates the base-model
   rows' format convention; blocking for F5's corrected statement.
8. **Qualitative-examples panel assembly** (#2094/#2162 artifacts exist;
   0 GPU) — F13.
9. **De-prioritized by the restructure:** the cross-framing causal patching
   arm (claims.md Gap 1) was scheduling-critical when C4 was the PSM headline
   *with* a causal leg; with causality demoted to appendix, Results II can
   ship as explicitly correlational. Spend the GPU on 1–2 instead.
10. **#2388 (correctness)** — in-flight; decide inclusion from its landed
    numbers, and do not pre-write the Results I "correctness" clause before
    they land (the outline correctly tags it IN-FLIGHT).

## One-line register notes

- #2162/#2329 remain uncitable (TLDRs "(Thomas fills in)") — the outline's C2
  appendix correctly does not quote them; keep it that way.
- The c2 appendix draft's "transport cosines top out at 0.16" could not be
  reproduced at any pooling (closest 0.146/0.195); the draft's "at most 0.15"
  grounded in the figure artifact is the safe form — carry that number into
  the Discussion bullet too (the outline's Discussion bullet says ≤0.16).
- `\planfig{figures/paper/c1_stage_retention.pdf}` is referenced from Results
  II with a `c1_` stem — cosmetic, but rename before the figure set freezes.
