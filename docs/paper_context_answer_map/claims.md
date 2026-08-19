# Evidence appendix — plan.md claims → EPS repo evidence (2026-08-19, rev 2: iteration families)

Inventory pass resolving every DONE/PARTIAL/TBD/NEEDS-RUN/IN-FLIGHT tag in `plan.md`
against the EPS repo (`~/explore-persona-space`). Every quoted number is copied from
the cited task's `## Takeaways` / `## TLDR` or its events log (task bodies at
`tasks/<status>/<N>/body.md`; dashboard `https://eps.superkaiba.com/tasks/<N>`).
Figure paths are relative to the EPS repo root. Classification key: **AP** =
awaiting_promotion (clean-result written, user promotion pending), **IF** = in-flight
(live session), **none** = no evidence found. Resolved status: **SOLID** (evidence +
figure exist), **PARTIAL**, **IN-FLIGHT**, **GAP**.

**Rev-2 change:** many experiments on this line ran multiple iterations where the
first failed and a later one succeeded. Rows now cite the SUCCEEDING iteration; each
claim section carries `iterations:` notes recording what failed, what fixed it, and
which published numbers are superseded (do not cite those). The recurring failure
class is **estimator degeneracy**: pure-GCV ridge λ-selection at n_train < d = 3,584
(near-)interpolates and reads catastrophic negative held-out R² while the information
is demonstrably present. #1887 (completed infra) hardened the fit cores (inner-group-CV
default, GCV dof-cap, n<d tripwire) and audited affected published cells: **#1345
story/chat/plain matched-row cells, #1310 per-persona cells, #1639 per-cell refits**
were affected; #1335 and #1689 used safe selectors. Any pre-fix negative R² in an
n_train < d cell on this line is suspect until re-read.

**Terminology guard (glossary duty, `docs/glossary_context_answer_map.md`):** two
prefix-side objects exist — the query-averaged **prefix vector v_P** and the
**prefix-end state** (last prefix token, before the query). The #810/#658-era "prefix
map R² ~0.8" belongs to v_P; #1092's prefix-end 0.05–0.11 is the other object — a
**definition split, not a contradiction**; never quote them against each other. Also
declare the context-vector pooling per use: **v_C = last-prompt-token** is canonical;
span-mean is a different, weaker object (#1768 re-pool flipped 23/216 cell verdicts
and raised base-map R² ~0.20/layer; #1900 round 1 pooled span-mean while calling it
the context vector — its raw-context-similarity numbers are span-mean-scoped).

---

## C1. The mapping is mostly linear; nonlinear gains only as training contexts scale

| Planned plot | Evidence | Class | Figure | Status |
|---|---|---|---|---|
| Main: R² + acc@1, linear vs nonlinear vs identity+bias, vs #training contexts | #1901: at 963k train rows fitted maps beat every baseline (euclid acc@1 ridge 0.800, wide neural 0.844, identity+bias 0.527); nonlinear-over-ridge acc@1 gap **grows with pool size 0.036→0.101** (LMSYS, pools 1k→100k); ridge acc@1 falls 0.81→0.065 as train rows drop 963k→50 while identity+bias stays ~0.50. #1775: rank-32 bilinear closes 93% of the additive→full-context gap (+0.049); dedup'd 1M MLP gain ≈ +0.056. #722: at small n (50-context sweep) nonlinear estimators buy nothing (MLP negative at every layer) | AP | `figures/issue_1901/ladder_by_metric_grid_v2.png`, `hero_r2_vs_acc1_scatter_v2.png` | **SOLID** |
| Middle layers: R² across layers (K2) | #722: held-out R² plateaus 0.74–0.80, **peak layer 18** (of 28), reproduced 28/28 layers under PCA-48. #1092 headline at layer 19; #2330: Qwen3.5-9B also peaks at layer 18 (dense sweep) | AP | `figures/issue_722/base_skill_over_mean_per_layer.png` | **SOLID** |
| Model scale: R² vs size, no consistent relationship (K5) | #1491: raw R² rises 0.564→0.725 (0.5B→7B), plateaus 14B, **falls to 0.645 at 32B**; ceiling-normalized predictability holds 0.75–0.79 through 14B then 0.70. #2330 (child of #1491): 7B beats 9B cross-family at matched data (0.705 vs 0.661, p=0.002) | AP | `figures/issue_1491/ladder_r2_raw_and_normalized.png` | **SOLID** |
| On/off-policy: consistent vs inconsistent origin (K4) | #823: plain-style external (Claude) answers retain **91–98%** of own-answer refit R²; shuffled answers collapse to ≈0; own-answer increment small (only sycophancy crosses 0.05, gap 0.052). Distinct-style externals retain 77–81% on refit but get **≈0 cross-map transfer**. #952 (child): own-answer advantage 0.01–0.035 R², position-uniform. #1072 (child of #952): advantage ~90% outside token-identity subspaces | AP | `figures/issue_823/fig1_refit_r2_by_arm.png`, `fig5_style_transfer_decomposition.png` | **PARTIAL** — #823's distinct-style arm is close to the "inconsistent origin" cell; a dedicated inconsistent-origin generation arm is still unrun |
| Holds across turns (K8) | #958: turn-2–4 maps transfer with deficits 0.000–0.023 vs own-turn 0.45–0.49. #825: per-turn map flat at R² 0.55–0.59 for **turns 1–16** (~81–88% of single-turn anchor); turn-1 context predicts turn-2 answer at 0.29 decaying to 0.04 by turn 16. #1738: 100k real multi-turn — full context 0.681, bare query 0.534 (78%), history-only 0.379 (56%) | AP | `figures/issue_958/hero2_skill_vs_turn.png`, `figures/issue_825/onpolicy_turn_depth.png`, `figures/issue_1738/hero_prefix_vs_context_r2.png` | **SOLID** |
| Worse on OOD contexts (K7) | No single dedicated experiment. Scattered: #1901's WildChat transfer is held-out-row within training distribution, "not corpus-OOD" (own caveat); #2222: frozen map transfers catastrophically to the 24-dataset screening suite (suite-refit rescues); #1739: on a held-out attack-tactic family map arms don't separate from a shuffled-direction control; #779: bottleneck is "read-out transfer across context distributions, not underfitting" | AP | needs plot (consolidation figure) | **PARTIAL** — Limitations material |
| Base model, refined per post-training stage (K3) | #825: base holds **~87%** of instruct strength (R² 0.588 vs 0.673; full-corpus n=4,724 > d — structurally immune to the GCV class per #1887); "post-training reparameterizes the existing mapping by a general linear map". #1902 (OLMo-2): aligned retention base→SFT 0.472, SFT→DPO 0.874, DPO→RLVR **0.991**. #1336 (Llama Tülu): DPO→RLVR is the smallest pooled step (+.004 vs +.165 base→SFT). #2061: kNN retrieval jumps base→SFT (0.13–0.41 → 0.42–0.75) then plateaus | AP | `figures/issue_825/base_sep_control_hero.png`, `figures/issue_1902/hero1_diag_q_vs_stage.png`, `figures/issue_1336/hero_rlvr_contrast.png` | **SOLID** |

**iterations (C1):**
- **Turn-stationarity family (#825 internal rounds; #958):** #825's early "half strength,
  decaying with depth" per-turn read was a small-n artifact — the 5,000-row rerun reads
  flat 0.55–0.59 at every turn; #958's turn-1 exception resolved to a duplicate-message
  memorization artifact (main panel) + ridge-λ mismatch (long panel). Cite the corrected
  reads only.
- **Own-answer family (#722 → #823 → #952 → #1072 → #2091):** clean refinement chain, no
  failed iteration; #2091 (parent #1073, which found single greedy answers adequate)
  closes the target-averaging question: averaging buys measurement, not map.
- **#1775 internal:** the banked 50k-train fitter comparison was contaminated (lexical
  near-duplicates); the deduplicated 1M comparison (≈ +0.056 MLP gain) is the citable
  number, not the contaminated levels.
- **#722's below-floor FT cells** (EM 0.3–0.6×) fail their power check — "inconclusive,
  not 'function held'"; do not cite as stability evidence.

Notes:
- The plan's C1 statement "nonlinear gains only as training contexts scale" is exactly
  the #722 (small-n: nothing) vs #1901/#1775 (~1M rows: real, growing) pair — cite both.
- #1774 pins the input-choice decomposition at trait grain: full context 0.812, bare
  query 0.717, **pre-query prefix-end 0.02, query-averaged prefix −0.02** per-answer.
  Figure: `figures/issue_1774/hero_trait_arm_r2_heatmap.png`.
- #920/#810 certify the whole-answer-mean summary choice (survives a 34,652-recipe
  sweep, best held-out skill 0.87 vs 0.17 chance; genre-general). Both AP.

## C2. Context vector stores disproportionate answer information, causally used

| Planned plot | Evidence | Class | Figure | Status |
|---|---|---|---|---|
| Qualitative examples: patching only the context vector | #2094 + #2162 per-pair companions and judged dashboards exist; no assembled qualitative panel yet | AP | needs plot (assembly) | **PARTIAL** |
| Patching context vector → answer direction, vs other slots | #2094: nine clean disjoint **activation** families, all at context-end; prefix-end, second-to-last, third-to-last slots yield **zero** null-separated families anywhere on the grid | AP | `figures/issue_2094/hero1_f_act_heatmap.png` | **SOLID** |
| Patching context vector → behavior expression, vs other slots | #2094: all 15 clean behavioral survivors are context-end edits, 0.18–0.63 of a full context swap (nulls −0.24 to 0.07); largest = full-state patch at all 28 layers: **0.63 of a full swap** (0.51 re-sampled, null 0.10); single-layer L12–20: 0.18–0.33. Query text-token edits (template excluded) read 0.68–0.95 of swap against elevated nulls | AP | `figures/issue_2094/hero2_f_beh_heatmap.png` | **SOLID** |
| Same two effects with different matched pairs | #2094: 14 matched-query + 1 cross survivor; weak-pair recovery round reproduced direction on 17/17 parent-clean reads. #2162 (child): 21 minimal-pair information types × route/recency/load — v2 report, **results + figures landed, headline claims not yet written by Thomas**. #2329 (child of #2162): same sweep on Qwen3.5-9B, claims also unwritten | AP | `figures/issue_2162/` (78 files), `figures/issue_2094/exp_typeA_vs_typeB.png` | **PARTIAL** — evidence landed; #2162/#2329 TLDRs are "(Thomas fills in)" |
| Our mapping weakly predicts the patching-induced shift | #2094: banked-map transport cosines **top out at 0.16**; best one-operator fit of the edit-to-response map: held-out R² 0.084; dose-response flat. #1415 (parent of #2094): fitted map predicts **none** of the realized shift at layer 20 (cosine 0.00, magnitude over-predicted ~16×). #2329 has a registered predicted-vs-realized result (claims unwritten) | AP | `figures/issue_2094/r2_decomposition_shift_cosine.png` | **PARTIAL — wording risk**: evidence supports "barely/does not predict", not "weakly predicts" |

**iterations (C2 — the patching family #1415 → #2094 → {#2162, #2333} → #2329):**
- #1415 (first causal iteration) found steering works but is **matched-query-only**
  (transport 0.49 vs 0.22 cross) and its all-position arm broke the output distribution
  (96–98% Chinese-script draws — uninterpretable); the medical-doctor pair was sampling
  noise. #2094 is the succeeding systematic grid (bootstrap screen + independent
  temperature-1.0 confirmation) — cite #2094 for slot/layer claims, #1415 only for the
  layer-14 behavioral peak (+6.2 pts, replicated +6.6/+5.0 on fresh seeds).
- #2094 internal: raw cell means 0.85–2.39 collapse below 0.13 once weakly-separated
  anchor pairs are excluded — cite fraction-of-swap against nulls, never raw means; the
  chat-template tokens made the query span untestable until a text-token-only edit round.
- #2333 (IF, followups_running): prefilling the patch's own three opening tokens
  recovers **67%** of the full context-end patch effect (n=172, p 2e-13); state patches
  50–54%. Qualifies "the vector carries it": a large share rides opening-token identity.
- Counterweight #1776 (HIGH, parent #779): Jacobians recover ~none of the fitted map's
  predictive power (R² −0.001 vs 0.681), and full-state substitution at the map's
  layer-14 input slot moves B-content acquisition zero — "reads a correlate, not a
  cause". Reconcile with #2094 in prose (see Contradictions).

## C3. Best at high-level (persona-level) properties, worse at low-level ones

| Planned plot | Evidence | Class | Figure | Status |
|---|---|---|---|---|
| SAE-feature R² best-predictors (appendix) + partial-out summary (main) | #1482: matryoshka granularity gradient — median feature R² falls **0.43 → 0.17 → 0.04** general→specific tiers; feature predictability is an answer-side property (Spearman 0.93). #779: persona direction sits at the 99.7–99.9th variance percentile of the answer profile, well-predicted held-out (per-direction R² 0.79–0.87 vs ~0.56–0.58 random-direction band). #1895 (child of #1482): map-predictable subspace ≈ SAE-representable subspace (overlap 0.867, ~98% variance-driven). #2163 (child): per-unit-gain discounts carried entirely by never-firing features | AP | `figures/issue_1482/hero1_category_error.png`; `figures/issue_1895/hero_overlap_ksweep.png` | **SOLID** (partial-out methods question stays open — Thomas DECIDE) |
| Which contexts/answers the mapping fails to distinguish | #2202 (child of #1738): 80.7% of rank-1 retrieval failures are **map error dragged toward hub answers, not answer degeneracy**; hot-spots refusal +24.8pp, NSFW +21.8pp, code +11.0pp, English +8.2pp; metric-side fixes (whitened cosine + CSLS + 5-draw targets) converge all 7 architectures to 0.991–0.995. #1482: per-language error German 0.236 → Arabic 0.420. #1945: residual error interaction peaks at R² 0.0013 vs 0.10 floor — map near its information ceiling | AP | `figures/issue_2202/fig_attribution_v2.png`; `figures/issue_1945/hero_bcv_primary_log.png` | **SOLID** — plan's "NEEDS-RUN or TBD" resolves to done |
| Persona-vector directions predicted from mapped answer vector | #1615 (analysis, parent #779): pre-image top-30 contexts coincide with judge-most-expressive contexts; top LMSYS projections persona-sensible. #779: per-direction persona projection well-predicted (0.79–0.87). BUT #1739: map-then-project **trails** context-side baselines (see C5) | AP | `figures/issue_779/pinv_topk_lmsys_topbottom.png` (#1615's figures live under issue_779) | **PARTIAL** — sanity holds; superiority does not |

**iterations (C3):** #1482's non-English-better direction FALSIFIED the planned
direction (a hypothesis flip, not a failed run — sign survives intrusion exclusion and
corpus transfer). #1946 confirms the prefix/bare error mirror survives SAE re-basis
(0.855). No superseded numbers in this family.

## C4. Persona-specific and universal across chat and story (PSM evidence)

| Planned plot | Evidence | Class | Figure | Status |
|---|---|---|---|---|
| Transfer chat template → no template (R² AND retrieval) | #825: plain "User:/Assistant:" refit — instruct 0.625 (vs 0.654 chat), base 0.578 (vs 0.542). #1345: chat↔plain-text misses the same-operator margin directly but a general-linear coordinate change recovers each ceiling; instruction tuning pulls coordinate systems together (aligned operator cosine 0.855 instruct vs 0.732 base). #2054 (child of #1345, **well-posed by design**: per-fold n_train 6,341–9,586 > d): the **answer-boundary form, not narrative prose, carries the framing cost** — boundary swap breaks direct transfer (median −0.06) while prose swaps preserve it (0.20); context-side linear re-map recovers 0.84 of ceiling; retrieval per cell (median acc@1 19%, best 70%) | AP | `figures/issue_2054/hero_boundary_vs_prose.png`, `figures/issue_1345/reparam_recovery_context.png` | **SOLID** for R²+reparam; retrieval via #2054. Causal patching arm — **GAP** (no task) |
| Transfer assistant → story characters | **Cite the corrected iterations:** #1345 (post-fix): story map on identical conversations +0.367 (verbatim answers) / +0.262 (own in-story answers) vs chat +0.609/+0.567 — weakened ~2×, not eliminated; assistant operator transfers into character maps **graded by AI-likeness (ρ +0.80, n=4)**. #1310: character-specific maps in base AND instruct (correct-vs-swap +0.295/+0.381). #1639: four character maps are one shared operator (pooled recovers 81–98% of each ceiling; Procrustes cosine 0.516 base / 0.593 instruct — framing moves the operator more than character identity, 0.455 anchor vs 0.686 base↔instruct). #1335: fiction scene framing carries the instruct gap (0.174/0.179). #1689 (safe selector): framing costs far more than identity (identity-only recovery 0.86–0.88 vs framing-only 0.41–0.47). #1417: map survives rude/evasive/addressee-free/AI-relay framings (11/11 Shared) — tracks generic query-answering; scope caveat for "persona-specific" | AP | `figures/issue_1345/hero_transfer_heatmaps_context.png`, `figures/issue_1639/tier15_ladder.png`, `figures/issue_1310/hero_l19_bars.png` | **SOLID** (correlational; up-to-reparam, not direct). Causal patching arm — **GAP** |
| Transfer assistant → simulated user | **Corrected record:** #825's bolded "the user's next turn is linearly unpredictable (ridge R² negative)" was REFUTED 2026-07-25 as a λ-selection artifact — all four user cells flip from −1.43…−1.65 to **+0.19…+0.25** under two independent selectors (#825 events v309 selector-audit round; #1701 records the refutation; **#825's body still carries the stale bolded Takeaway** — record-integrity fix owed). Succeeding iteration #1689 (safe selector): Haiku-simulated user turns 0.23–0.34, **on-policy Qwen user turns 0.00–0.07**, real human turns LESS predictable than simulated (Δ −0.11 to −0.17). **#2378 (planning): user-character-in-chat-template arm added by user amendment 2026-08-19 03:24Z, two provenance sub-arms (real teacher-forced vs on-policy simulated), Qwen3.6-27B** | IF (#2378; #1689 AP) | needs plot (from #2378) | **IN-FLIGHT** — corrected existing evidence: weakly positive for simulated users, near-zero for on-policy model-as-user |
| Trained-on-everything vs trained-on-one-thing | #1639: pooled 4-character map recovers 81–98% of per-character ceilings. #2054: one pooled map over 56 framing cells reaches 90% of ceiling in 21/56 as-is, 34/56 with per-cell bias, 50/56 at 95% with rank-128 residual. #2378 secondary arm: pooled-tier ladder registered | AP + IF | `figures/issue_1639/tier15_ladder.png` | **SOLID** |
| Mapping doesn't train during unrelated post-training stages (K3 cross-ref) | #1336: RLVR (math/IF) leaves the map essentially unchanged except on its own two training corpora; #1902: DPO→RLVR retention 0.991 | AP | `figures/issue_1336/hero_rlvr_contrast.png` | **SOLID** |
| Patching context vector affects persona of ENTIRE answer + specificity control | #1415: re-forwarded direct component persists to end of span (late bins clear 2.5–10× jitter, 111/112 pairs) while decaying to 0.4–2% of state norm. #2333: ~a third of the patch effect is NOT carried by opening tokens (majority is). #2162's specificity ladder exists, claims unwritten | AP/IF | `figures/issue_1415/hooked_decomp_hero.png` | **PARTIAL** |

**iterations (C4 — the fiction/framing family #825 → #931 → #1310 → {#1335, #1639} and #825 → #1345 → #2054 → #2378):**
- **The story-collapse numbers were an estimator artifact, fixed mid-family.** #1887's
  audit: the published #1345 story R² of −0.547 (ambient pure-GCV ridge at n_train < d)
  reads **+0.262 in a train-fold reduced basis and +0.44 in ambient at forced λ=1e3** —
  GCV degenerates at n_train < d while the information is present (retrieval ~300×
  chance). #1345's current Takeaways carry the corrected numbers; earlier-round negative
  story R² anywhere in this family is superseded. #825's early "does NOT hold for
  generic stories (R² ≈ 0.16)" TLDR bullet predates this fix and the constructed-rig
  work — prefer #1345/#2054 for framing claims.
- **#931 (wild fiction, negative) was NOT in the #1887 re-read audit** and its own
  Takeaways flag the fragility ("the ridge estimator fails on both control cells",
  −3.2/−3.5 vs +0.35/+0.33 rotated) at n=1,982 pairs < d=3,584 — the same degeneracy
  class. Its wild-fiction negative (author-blocked R² −0.065) stands as written but is
  the family's one un-re-read cell; treat as provisional (see Contradictions).
- **#1310 round 1** read per-turn instruct anti-prediction (−0.10 to −0.19) — a
  λ-selection artifact conditioned by within-scene near-duplicates; inner-group-CV
  reads all four cells positive (+0.24 to +0.31). Cite the corrected reads. A later
  selector audit shows even the capped selector deflates per-turn cells (inner-group-CV
  selects λ 3,162–10,000 vs published λ=100).
- **#1417 round 1** was voided whole (pure-GCV collapse on judge-filtered subsets,
  −1.48…−0.59); the refit repaired exactly the collapsed fits (+0.54…+0.66). Cite the
  refit.
- **#1335 vs #1345 base-model reads:** #1335 shows the base fiction map swings across
  independent runs (endpoint 0.269–0.435 across four runs) — base-side story numbers
  carry run-level noise larger than most effects; instruct-side claims are the stable
  ones.

## C5. The mapping is useful (flagship: pre-generation behavior prediction)

| Planned plot | Evidence | Class | Figure | Status |
|---|---|---|---|---|
| Flagship grid: behaviors × datasets × methods (persona vector on context / on predicted answer / probe on context / probe on predicted answer) | **The FRESHEST artifact is Thomas's writeup `docs/map_behavior_prediction_results_writeup.md` (2026-08-12) + the fair/P-B protocol artifacts (`eval_results/issue_1739/result2_fair/`, `r2v2_fits @ 5aae0a472b`) — built AFTER the #1739 body's last edit (2026-08-08) under a new data-access protocol.** Its four claims, per-claim verdicts (§ dig-deeper below): (1) PV-on-context fails off-synthetic — **BACKED** (ρ 0.65–0.79 synthetic → −0.10…+0.15 everywhere else); (2) PV-on-mapped ≈ PV-on-real-answer — **MOSTLY BACKED** (evil in-dist 0.671 vs 0.688; exceptions: syco-synthetic linear map 0.399 vs oracle 0.764, closed by the MLP map at 0.701; hallucination generic/OOD oracle gaps); (3) linear probes beat PV methods off-synthetic — **BACKED** (syco OOD probe 0.71–0.73 vs PV 0.24–0.25); (4) probe-on-MAPPED beats probe-on-context — **PARTIALLY BACKED**: positive in ~24/33 P-B rungs, median +0.02…+0.08, two CI-separated wins (syco model-written-evals +0.30; evil PAIR +0.12), two CI-separated losses (evil tom-gibbs −0.44; syco mimicry −0.20); NO shuffled-map control under P-B. The #1739 body's aggregate-adverse read (map-then-project trails; direct ridge ρ 0.71/0.58/0.74) used a DIFFERENT comparator (the context-EXTRACTED direction, arm2 — not rerun under P-B) and pre-dates these artifacts — see the body-reconciliation note. #779: evil many-shot map wins (0.43/0.66 vs raw 0.35/0.34, floored labels). #763: base answer activations predict expression LOCO ρ 0.51–0.86 | AP + writeup doc | `figures/issue_1739/result2_fourpanel/result2_fourpanel_nocap.png`, `pv_regime_view/pv_regime_pb_ladder_oldci_nocap.png`, `compose_fu_flip.png` | **SOLID — draft C5 from the writeup's four claims; claim (4) is real-but-small with named exceptions and one missing control** |
| Predict the effect of finetuning | #1979 (succeeding iteration of #1900): pushing a prefix's v_C through the BASE map best predicts **where fine-tuning delivers behavior change** (median within-arm ρ 0.41 vs 0.30 raw answer similarity; 9/12 content arms) — the strongest map-positive application result; family-scoped (sycophancy clears 0/4 arms). #1900: base propensity dominates expression LEVEL (0.632); pre-FT answer similarity is the only CHANGE predictor (0.381 marker / 0.124 content) | AP | `figures/issue_1979/hero_content_race.png` | **SOLID** (family-scoped) |
| Predict re-elicitation better than prior work (Kwon rig) | **#2379 (planning): predicted answer-vector similarity vs Kwon et al.'s context-side hidden-state predictors, EM + capitalization** — clarifier all-clear 03:15Z | IF | needs plot | **IN-FLIGHT** |
| Data screening (adjacent application) | #2224/#2222 (both HIGH, children of #2221): frozen map top-500 selections never beat random and fall below exact-ΔP; the paper's own prompt-token ΔP matches exact in 5/6 cells; a corpus-refit map rescues sample-level screening (r 0.73–0.86) — the frozen map's failure is corpus transfer, not the linear form | AP | `figures/issue_2224/i2224_4b_hero_lmsys.png` | **SOLID but ADVERSE** for the frozen map |
| Assistant-axis periodic capping | #2223 (followups_running): every-token capping cuts reproduced Assistant-Axis persona drift on Qwen3-32B by ~a third, doesn't eliminate it | IF | check `figures/issue_2223/` when it parks | **IN-FLIGHT** |
| Behavior prediction vs LLM judge (monitoring) | #2356 (running, pod-2356, P2 capture ~7.5h live): refuse/comply prediction from internal representations vs an LLM judge | IF | needs plot | **IN-FLIGHT** |
| Steering / control (negative boundary) | #2220 (HIGH, child of #1739): the map's read direction is causally inert (best Δ +0.06/+0.01) while the mean-difference persona vector steers strongly (evil +0.985, syco +0.429); read direction ⊥ persona vector (cos 0.00–0.03). #2254 (child): map pre-image inert at the context vector while a directly-measured context direction works; pre-image steers at answer tokens (+47.5). #2225: single-layer steering of context tokens shows no dose-response; response tokens do | AP | `figures/issue_2220/hero1_decisive_bars.png` | **SOLID** — prediction and control use different geometry; Limitations/Discussion material |

**iterations (C5):**
- **Behavior-expression predictor family (#658 → #761 → #763):** the canonical
  failed-then-fixed chain. #658 (first measurement) REFUTED the activation-summary
  link on 7/10 behaviors; #761/#763 recover it for all matched-probe behaviors (LOCO
  ρ 0.51–0.86) once **probe sets were matched to the eval distribution and the
  estimator/DV fixed** (the deception rubric re-anchor alone moved LOCO ρ 0.11 → 0.51;
  the original 0.00 ceiling was a split-half estimator artifact). Cite #763, not #658's
  numbers, for "base activations predict behavioral expression".
- **Monitoring family (#779 → #1739):** #779 demoted its own parent-line headline —
  "a direct context→trait predictor beats everything (r up to 0.91)" was an
  eval-context-fit artifact (fit on a disjoint corpus it loses in 5/6 cells). #1739 is
  the matched-budget endpoint (HIGH). Do not cite the r≈0.91 era. Note #1739's own
  under-determined cell: the tactic holdout is 1,906 train rows vs 3,584 dims (its
  stated caveat) — its map-arm null there is directionally consistent with the
  well-posed cells but weaker as evidence.
- **FT-change predictor family (#1768 → #1900 → #1979) + pooling definition change:**
  #1768 round 1 and #1900 pooled the prompt SPAN-MEAN while calling it the context
  vector; the last-token re-pool flipped 23/216 verdicts and moved relative context
  movement 0.025 → 0.237–0.267. #1900's "raw context similarity predicts nothing" is
  span-mean-scoped; #1979 declares position and finds the map/read wins **only at the
  last-prompt-token position** — numbers across the two poolings are not comparable
  (definition change, flag don't average).
- **#813 internal:** its EM substrate exception failed every follow-up probe
  (family-clustered CI spans zero; per-example refit inverts the ordering) — cite the
  3-of-4 null, not the EM exception.
- **#2091's parent #1073** (completed): single greedy answer adequate as target —
  consistent across both iterations.

### C5 dig-deeper (2026-08-19): where the map beats context probes

Sources: Thomas's writeup `docs/map_behavior_prediction_results_writeup.md` (verbatim,
2026-08-12) + its artifacts (`eval_results/issue_1739/result2_fair/result2_fair_points.json`
→ `figures/issue_1739/result2_fourpanel/fourpanel_values.json`; the P-A/P-B fit rows at
`eval_results/issue_1739/r2v2_fits/<beh>/all_arms_spearman.json`, commit `5aae0a472b`);
the #1739 body Results sections; the 1,906-cell table
`figures/issue_1739/headline_deltas_percell.csv`; win-relevant cells of #779/#2222/#2224/#1979.

#### The writeup frame (freshest read) — four claims, verified numbers

The writeup's protocol is NEW relative to the body's grid: every predictor gets generic
unjudged WildChat, a judged WildChat subset, 80% of the trait-eliciting corpora (judged),
and the synthetic PV train set; 6 eval settings from in-train to completely-OOD; the map
trains on ALL of it unjudged (generic + eliciting) — so the map is IN-DOMAIN by
construction, which is exactly the mechanism the old grid's composition cells found.

**Claim (1) — PV-on-context fails to generalize beyond synthetic prompts: BACKED.**
`pv_context` ρ: synthetic 0.755/0.792/0.647 (evil/syco/hall) vs in-distribution
0.031/0.018/0.010, generic chat 0.067/−0.039/−0.103, completely-OOD 0.014/0.061/0.145
(evil generic + OOD cells spread-gate-failed).

**Claim (2) — PV-on-mapped ≈ PV-on-real-answer (the map keeps the persona-relevant
answer signal): MOSTLY BACKED.** `pv_map_linear` vs `oracle`: evil in-dist 0.671 vs
0.688, synth 0.818 vs 0.843; syco in-dist 0.289 vs 0.318, OOD 0.240 vs 0.250; hall OOD
0.215 vs 0.353 (MLP-map 0.274 — a real oracle gap) and generic 0.176 vs 0.281. The one
big exception: syco-SYNTHETIC linear map 0.399 vs oracle 0.764 — the MLP map closes it
(0.701). Note PV-on-mapped also CRUSHES PV-on-context off-synthetic (evil in-dist 0.671
vs 0.031) — the datatype-mismatch fix works against the published-PV comparator.

**Claim (3) — direct probes beat PV methods outside synthetic settings: BACKED.**
P-B probes: syco OOD (aita) 0.706–0.726 vs PV arms 0.24–0.25; hall nqopen/simpleqa
0.38–0.46 vs pv_map 0.215; evil heldin:train 0.612–0.768 vs pv_context 0.031.

**Claim (4) — probe-on-MAPPED beats probe-on-context ("some small improvement",
attributed to unjudged data): PARTIALLY BACKED — real, small, mixed-sign.** P-B
protocol, linear map, `arm7_map_ridge_pred` − `arm4_ridge_ctx` per rung:

| behavior | rung (held-out/OOD) | probe-ctx | probe-mapped | Δ |
|---|---|---|---|---|
| evil | train (heldin) | +0.612 | +0.768 | **+0.156** |
| evil | pvsynth | +0.645 | +0.811 | **+0.166** |
| evil | ToxicChat | +0.323 [+0.24,+0.40] | +0.377 [+0.32,+0.44] | +0.055 |
| evil | MHJ | +0.085 [−0.00,+0.18] | +0.169 [+0.08,+0.25] | +0.084 |
| evil | PAIR | +0.140 [+0.08,+0.20] | +0.262 [+0.20,+0.33] | **+0.122 (CI-sep)** |
| evil | tom-gibbs | +0.099 [+0.06,+0.14] | −0.337 [−0.38,−0.29] | **−0.436 (CI-sep)** |
| syco | aita (OOD) | +0.706 [+0.67,+0.74] | +0.726 [+0.69,+0.76] | +0.020 |
| syco | model-written-evals | +0.031 [−0.04,+0.11] | +0.335 [+0.27,+0.41] | **+0.304 (CI-sep)** |
| syco | mimicry | +0.186 [+0.06,+0.30] | −0.012 [−0.13,+0.10] | **−0.199 (CI-sep)** |
| syco | answer / are-you-sure / feedback | +0.413/+0.373/+0.537 | +0.446/+0.322/+0.534 | +0.03/−0.05/−0.00 |
| hall | NQ-Open | +0.380 [+0.35,+0.41] | +0.417 [+0.39,+0.45] | +0.037 |
| hall | SimpleQA | +0.442 [+0.42,+0.47] | +0.459 [+0.44,+0.49] | +0.017 |
| hall | wildchat | +0.543 | +0.625 | +0.082 |

Positive in ~24 of 33 P-B rungs; hallucination positive on EVERY rung; evil positive on
6 of 7 except the tom-gibbs collapse (the shakiest attack corpus: it failed the
secondary per-rollout spread unit and 25% of its rollouts are truncation-capped);
sycophancy mixed (mimicry is the failure mode both probe inputs handle worst). CAVEATS
for drafting: (a) NO shuffled-map control exists under P-B — the body's old protocol
found +0.008…+0.017 map-feature gains fully matched by a shuffled-pretrain control, and
that alternative is untested at the new, larger gains; (b) the unjudged-data ATTRIBUTION
is an interpretation — no ablation isolates unjudged-pool volume (the writeup's own
"next steps" defers data ablations; `result2_trait_aug` / Result-5 ablation artifacts
exist uncited); (c) single seed per fit; (d) sycophancy's corrected 5-corpus OOD rungs
are staged-but-pending (`pending_syco_ood` in the points JSON) — the aita rung is a
same-platform fallback.

#### Body reconciliation (#1739 body 2026-08-08 vs writeup 2026-08-12)

Not a contradiction — DIFFERENT COMPARATORS, plus staleness:
- The body's headline "map-then-project trails the context-native direction" compares
  arm6 to **arm2 (a direction EXTRACTED at the context vector)** — a repaired
  context-side method, not the published PV method. The writeup's claims (1)/(2)
  compare to **arm1/pv_context (the published, datatype-mismatched projection)**. Both
  are true on their own artifacts: the map fixes published-PV's mismatch (writeup), and
  a context-extracted direction fixed it about as well on the OLD protocol (body). The
  writeup's roster OMITS arm2, and the P-B round never reran it — so "map is the best
  fix for the mismatch" is not established against the cheaper context-side repair.
- The body's "map-feature regression gains are a dimension artifact (shuffled control
  matches)" was measured on the OLD protocol at gains of +0.008…+0.017; the P-B gains
  (median +0.02…+0.08, max +0.30) are larger and control-free — neither artifact
  settles claim (4)'s mechanism.
- **Freshness/staleness flag (Thomas-only fix, do NOT edit the body):** the #1739 body
  title "Map-then-project trails matched-budget context-side baselines across three
  behaviors (HIGH confidence)" and its Takeaways make no mention of the fair/P-B
  rounds, and read as unconditional where the freshest artifacts show the PV-on-mapped
  arm beating the published PV-on-context nearly everywhere off-synthetic and
  probe-on-mapped edging probe-on-context in ~70% of rungs. The title/Takeaways now
  MISREPRESENT the freshest read and need a Thomas-authored rewrite before promotion.

#### Old-protocol regimes (supporting detail — original grid, `headline_deltas_percell.csv`)

Per-regime verdicts; delta = map-then-project minus context-native (`rho_arm6 −
rho_arm2`) unless stated.

**Regime 1 — in-domain unjudged map pool × small labels: MAP WINS (preliminary).**
#1739's evil composition cells: with the 5,000-row map pool half in-domain (fu=0.5),
map-then-project beats the context-native direction at EVERY budget (deltas +0.05 to
+0.35; per-cell: L=250 arm6 0.564/0.533 vs arm2 0.214/0.242, delta +0.35/+0.29; still
+0.13 at L=8,000) and far exceeds direct ridge at L=250 (0.53–0.56 vs 0.17–0.19, ~3×) —
the body's own sentence: "the small-label composition hypothesis holds only with an
in-domain map." Fully generic pools give −0.08 to −0.31 in the same design. LIMIT: all
16 realized cells are n=1 (single draw, single seed), evil only, flagged preliminary;
2 planned cells skipped (infeasible pool). This is the paper-thesis regime (the map
exploits unjudged in-domain data; labels only pick the layer) and the single most
valuable confirmatory run before the deadline: rerun the composition grid multi-seed ×
3 behaviors. Figure: `figures/issue_1739/compose_fu_flip.png`.

**Regime 2 — label scarcity with a generic map: TIE, not win.** At L=250 (generic
full pool) map-then-project edges direct ridge for evil (0.54 vs 0.52) and sycophancy
e2 reads 8/15 cells positive vs the context-native direction (mean +0.002) — but the
context-native direction (itself label-light) still leads (evil 0.60). Projection arms
are flat in L (labels only pick the layer) while regression arms grow; the regression
dip at L=2,500 is a ridge near-interpolation pathology (~2,000 train rows vs 3,584
dims), not map evidence. Figure: `figures/issue_1739/scaling_rho_vs_l.png`.

**Regime 3 — pooled-natural sycophancy direction (E2P): consistent small map win,
confounded.** +0.05 to +0.09 mean delta, 15/15 cells above zero at L=250 AND L=16,000
(per-cell table) — the body demotes it because the E2P contrast is topic-confounded by
construction. Report only with the confound named.

**Regime 4 — off-distribution transfer: two ties, one open cell, no clean win.**
Sycophancy held-out Reddit: the committed linear-map readout TIES direct ridge (0.729
vs 0.725) with its shuffled-map control at 0.18 (the tie is map-semantics-real); kernel-
map readout ties on NQ-Open (0.403 vs 0.395, uncontrolled). Evil ToxicChat shows a map
edge (0.32 vs 0.25) on a floor-degenerate trait DV (93% bottom-bin) — and the graded
compliance DV that DOES spread there (SD 34.7–38.7) has NOT re-scored any arm
comparison: "evil's ToxicChat map edge stays unmeasured under a spreading DV" — an
open, cheap re-score, not a settled loss. Elsewhere direct ridge transfers best
(hallucination 0.40/0.40 vs map 0.20/0.27).

**Regime 5 — attack-tactic holdout: map null is the weakest-instrumented read in the
body.** The map-vs-shuffled non-separation (0.514 vs 0.510) sits in an under-determined
refit (1,906 train rows vs 3,584 dims — "levels are regularization-limited", the body's
own caveat), the mandated kNN-retrieval read is ABSENT there, and the identity+bias
baseline is rank-degenerate in single-fold transfer — so neither mandated mapping read
is informative on that holdout. The direct context MLP's 0.68 win stands; the map
arms' failure there is real-but-under-instrumented, and #1887-class experience says
under-determined map reads underestimate.

**Regime 6 — many-shot contexts (#779): map beats the raw context projection.** The
4-condition evil many-shot cell: mix-trained map 0.43 vs raw projection 0.35 (delta CI
excluding zero); the 963k nonlinear map widens it (0.66 vs 0.34) — frozen layer 26
only, and the evil labels are expression-floored (3% of rollouts above 50), so this is
a real registered win on a weak instrument. Mechanism-consistent: many-shot contexts
are exactly where a context-side projection degrades and the mapped answer-side read
should help.

**Regime 7 — data screening (#2224/#2222): the refit map matches the
generation-requiring gold standard, generation-free.** Exact ΔP requires base-model
GENERATION per candidate; a map refit on the corpus's own 50,000-sample unjudged pool
correlates with exact ΔP at 0.73–0.86 per-sample (frozen map: −0.09 to 0.14) with
top-500 overlap 0.23–0.57 (chance 0.005) — #2224. #2222's follow-up: a map fit to the
base-generation target comes within 0.07 of exact ΔP on every trait when each fold
selects its own layer (sycophancy 0.68 vs 0.88 at the fixed steering layer — the
failure is layer-specific, not absolute). CAVEAT: no selection-finetune was run on
refit-map top-500s (the finetune arms used the frozen map), so the rescue is
screen-correlation-level, not behavior-installation-level.

**Regime 8 — predicting fine-tuning-induced change (#1979): registered map-side
champion.** Through-map predicted-answer similarity is the change champion at prefix
grain (median ρ 0.41 vs 0.30 raw answer similarity, 9/12 content arms, winner
probability 0.65–0.81; 7/12 arms clear their permutation bands — casualness +
impoliteness families; sycophancy 0/4).

**Not found:** no cell anywhere in #1739/#779/#763/#2224/#2222 where a
frozen-GENERIC-pool map-then-project beats a matched-budget direct context probe on a
well-instrumented, non-confounded cell at large labeled budgets — the old grid's
adverse verdict stands for that configuration. The map's wins concentrate where (a)
the map pool includes the deployment corpus unjudged (the writeup's P-B protocol —
claim (4) — and old-grid Regimes 1, 7), (b) labels are scarce (Regime 1), (c) the
context-side read is structurally degraded (Regime 6 many-shot; the published
PV-on-context everywhere off-synthetic — writeup claims (1)/(2)), or (d) the DV is
the fine-tuning-induced CHANGE rather than the level (Regime 8). The unifying
mechanism across the writeup + #1739/#2224/#2222/#779: **the failure mode is corpus
transfer of the map, not the linear form and not the mapped-readout concept.**

**Reframing recommendation (updated after the writeup reconciliation):** draft C5 from
the writeup's four claims — they are Thomas-authored, freshest, and the numbers back
them (claim (4) as "small, mostly-consistent improvement" with the mimicry/tom-gibbs
exceptions named). The writeup's P-B result upgrades the old grid's n=1 composition
cells to multi-rung CI-bearing evidence for the in-domain/unjudged mechanism, so the
conditional flagship — "the map converts unjudged in-domain data into label-efficient
pre-generation prediction" — is now PARTIALLY CONFIRMED, not merely preliminary. Still
missing before the strong form is draftable: a shuffled-map control under P-B, seed
replication, the arm2 (context-extracted direction) comparator rerun under P-B, and
the pending corrected sycophancy OOD rungs. The unconditional "map beats direct
context probes" form remains unsupported (claim (4)'s median is +0.02…+0.08 with
CI-separated losses); the #1979 FT-change and #1901/#2202 retrieval wins stay the
unconditional pillars.

## § Effect of finetuning on the mapping (OPTIONAL section)

| Planned probe | Evidence | Class | Figure | Status |
|---|---|---|---|---|
| Do context/answer vectors change under FT? What changes the map? | #722: FT measurably reshapes the map only for a taught fact (3.3× refit floor at primary layer); EM/sycophancy power-fail (inconclusive, not stability). #1768 (HIGH): map change is context-dependent but NOT gated on trained-prefix identity — never-trained prefixes resembling the training corpus **overshoot** the trained-prefix anchor on 3/3 persona arms; weights carry 0.698 of the mean shift, text 0.779 of the typical row. #813: map change indistinguishable across query substrates for 3/4 behaviors. #779: trait-data training degrades the map (see C5). #1902/#1336: post-training ladder (see C1/K3) | AP | `figures/issue_1768/` (62 files), `figures/issue_722/` | **PARTIAL** — the plan's "single batch / single example" mathematical probes are unrun |
| Content-specificity roster | **#2247 (proposed, not started):** token-matched fine-tuning roster (personas vs false fact vs single token vs formatting) | none | — | **GAP** (filed, unscheduled) |

**iterations (FT section):** #1768's round-1 stand-in for the text effect is refuted by
its own later round (the model-vs-text split is aggregation-dependent) — cite the final
0.698/0.779 split. #1489 (HIGH, child of #1092) is a designed-in kill: fact/instruction/
persona context augmentations fell under the manipulation-check floor (only format
passed), so the transport-map questions were never tested — its 38,000-generation
substrate + 64 adapters are reusable for a retry with stronger manipulations.

---

## Gaps ranked (deadline: ICLR 2027 abstracts 2026-09-18)

1. **Causal patching arms for both transfer results (C4)** — no completed, in-flight,
   or proposed task patches context vectors ACROSS framings (chat→no-template,
   assistant→story). #2378 is correlational transfer only. C4 is the PSM headline and
   stays correlational without this; scheduling-critical given ~4 weeks.
2. **C5 flagship: close the P-B protocol's control gaps** (re-scoped after the writeup
   reconciliation — the in-domain × small-label mechanism is now PARTIALLY CONFIRMED by
   the writeup's multi-rung P-B result, not just the n=1 composition cells). Missing
   before claim (4) is draftable in its strong form, all cheap on banked stores:
   (a) a shuffled-map control under P-B (the old protocol's control fully matched the
   then-smaller map-feature gains); (b) seed replication of the P-B fits; (c) the arm2
   context-extracted-direction comparator rerun under P-B (the body's winning arm, absent
   from the writeup roster); (d) the staged-but-pending corrected sycophancy OOD rungs
   (`pending_syco_ood`). Companion cheap re-score: the evil ToxicChat map edge under the
   graded compliance DV ("stays unmeasured under a spreading DV"). See § C5 dig-deeper.
3. **#825 record integrity** — RESOLVED 2026-08-19 (user-directed body fold, marker
   v311): corrected user-turn Takeaway (guarded cells +0.19…+0.25, ~2.5× below
   assistant; role asymmetry survives) + Result-4/-5 update notes. Safe to draft from.
4. **#931 wild-fiction re-read** — CLOSED-BY-FLAG 2026-08-19 (user directive): #931
   marked estimator-bugged (tag + marker v56); the paper treats the wild-fiction cell
   as UNTESTED, and C4 carries "constructed rigs" scope. The guarded 0-GPU refit stays
   available if the wording ever needs strengthening.
5. **C2 "map predicts the patching shift" wording** — evidence says it essentially
   does not (cosine 0.00 at L20 #1415; transport ≤0.16 #2094). Soften or wait for
   #2329's registered predicted-vs-realized result.
6. **Off-policy inconsistent-origin arm (C1)** — cheapest data gap; #823's
   distinct-style arm is adjacent but not the designed cell.
7. **#2162/#2329 claims unwritten** — both v2 TLDRs are "(Thomas fills in)"; C2's
   information-type story can't be cited until then. Thomas-only.
8. **Finetuning optional section** — stays optional; #2247 proposed-only.

## Contradictions (flagged, not resolved)

1. **#931 vs #1310/#1639/#1345 (C4 story transfer) — now estimator-qualified.** #931:
   chat map does NOT carry to wild fiction (transfer ≤ +0.05 of ceiling). #1345 (post
   estimator fix)/#1310/#1639: character maps exist, share one operator, assistant
   operator transfers graded by AI-likeness. The #1887 audit showed this family's
   negative story R² reads were GCV-at-n<d artifacts and corrected #1345/#1310/#1639 —
   but #931 was NOT re-read and sits in the same regime. Until the #931 guarded refit
   runs (Gap 4), the honest C4 statement: "universal up to a linear reparameterization
   on constructed rigs; the wild-fiction read is negative but rests on a pre-fix
   estimator convention." RESOLUTION (2026-08-19): #931 flagged estimator-bugged per
   user directive — the paper treats wild fiction as UNTESTED; C4 statement becomes
   "universal up to a linear reparameterization on constructed rigs; wild fiction
   untested."
2. **#1776 vs #2094 (C2 causality).** #1776 (HIGH): full-state substitution at the
   map's layer-14 input slot moves B-content acquisition zero; "correlate, not cause".
   #2094: context-end full-state patch (all 28 layers) moves behavior 0.63 of a full
   swap; single-layer L12–20 edits 0.18–0.33. Differing DVs (content acquisition vs
   judged fraction-of-swap), pair construction, layer count. Neither supersedes the
   other; C2 needs the reconciling sentence (single-layer single-slot edits weak-to-null
   on content acquisition; multi-layer context-end edits real but sub-swap on judged
   behavior).
3. **Identity+bias baseline flips winner by metric and model (C1 main figure).** #1901
   (Qwen, 963k rows): fitted maps beat identity+bias on all non-saturated metrics.
   #1336 (Llama Tülu): identity+bias wins retrieval rank-1 in 9/10 pooled cells. #2215:
   at context-end, identity+bias captures most minimal-pair discrimination (fitted map
   +0.9 pts, CI includes zero). Training-size-, metric-, and model-dependent; show the
   R²-vs-retrieval dissociation (#1901, both directions), don't average it away.
4. **Bare-query share differs by rig — design, not error.** #1738 (real multi-turn):
   bare query recovers 78% of the full-context map. #1092 (crossed grid, 20 shared
   queries): bare query R² 0.02 — query identity structurally cannot discriminate rows
   there. Quote per-design, never as one number.
5. **"Weakly predicts the patching-induced shift" (plan C2/C4)** vs #1415 cosine 0.00 /
   #2094 ≤0.16 — plan wording currently stronger than the evidence.
6. **Persona-specific (C4) vs #1417.** The map survives register/addressee framings
   ("tracks generic query-answering structure"), while #825's punctuation control
   (generic next-span R² ≈0.05/0.1) shows it is not trivial next-span prediction.
   Phrase as "carries persona-level information" (C3) + "character-indexed
   up-to-reparam" (C4), not "specific to personas".

## In-flight register (as of 2026-08-19 ~03:30Z)

| Issue | Status | What it feeds | Session |
|---|---|---|---|
| #2378 story/user-character transfer, Qwen3.6-27B (child of #2054) | planning (round-1 redraft after two user Goal amendments; AI-likeness axis dropped, user-character arm added) | C4 simulated-user + story rows | cmszhz208qogfye0ui8hmsl4x |
| #2379 re-elicitation predictor (Kwon rig) | planning (clarifier all-clear) | C5 re-elicitation row | cmszily74saj2wo0upjam025k |
| #2333 prefill vs patch causality (child of #2162) | followups_running (code-review round) | C2 mechanism | cmsz8qlcpl3doye0uf277ue0n |
| #2356 refuse/comply prediction vs judge | running (P2 capture ~7.5h on pod-2356) | C5 monitoring row | cmsyyb1ciejafye0u5zvvz7zw |
| #2223 assistant-axis capping | followups_running | C5 capping application | cmsxrf1cefvmpye0umr6t1bly |
| #1739, #2061 | live sessions on already-parked results | C5 / C1-K3 | cmsg5qy1dz…, cmsi1z5h8g… |

Classification note — SUPERSEDED by user decision 2026-08-19: awaiting_promotion
results count as accepted evidence for this paper ("consider awaiting promotion to be
already done"); no promotion pass required before drafting. Classification states
untouched.

---

## Stretch-goal & appendix evidence (2026-08-19)

Evidence check for the six pending questions in plan.md v3 (§ Stretch goals + § Appendix
experiments). Same conventions as above: numbers quoted from Takeaways/TLDRs read in the
EPS task store; iteration-family discipline applied.

| # | Question | Evidence | Verdict |
|---|---|---|---|
| 1 | Effect of CoT on the mapping | **RUN** — the family #810 → #928 → #1005 → #1426 (+ #2329's thinking-disabled transfer). #928 (HIGH): the map transfers to the thinking model OpenThinker2-7B (held-out skill 0.78 vs 0.80 non-thinking parent); conditioning on the realized CoT adds +0.11 query-averaged / +0.20 per-question — but a matched-length slice of the answer's own opening beats the truncated CoT (by 0.052–0.092), so the CoT position is not privileged. #1005: the outsized CoT gain on in-context-learning + WildChat contexts is a context-family property, not a parse artifact (length-matched family excess +0.43, CI +0.37 to +0.48). #1426 (HIGH): both the mediation signature (+0.33 per-question CoT gain) and the family excess (+0.45) replicate on a Llama-base R1 distill — base-family-general, not Qwen-specific. Figures: `figures/issue_928/` (126 files), `figures/issue_1005/`, `figures/issue_1426/`. NOTE: `docs/leakage_paper_assumption_map.md` still lists "CoT effect — untested" (stale, dated 2026-07-19; #928 landed after) | **RUN** — ready to write, with the matched-length demotion as the honest headline |
| 2 | Predicting answer correctness from v_C / mapped answer vector (distinct from hallucination) | **none found — genuinely new.** Title sweep over the full registry (correct/truth/accuracy × map/context/vector/predict) surfaces no mapping-line experiment; the old wrong-answer coupling line (RESULTS.md make-evil-dumb matrix) trains wrong answers, never predicts correctness from activations; #742/#763 cover sycophancy/refusal/EM behaviors, not correctness | **NEW** |
| 3 | Where is behavior most decodable (position within the answer) | **PARTIAL — adjacent only, direct question unrun.** #655 ("Do persona/trait SAE features persist across token positions of a generation?") is an archived stub, never executed. Adjacent evidence, all map-side not behavior-decodability-side: #920 — only single tail-position targets separate from the whole-answer mean, as WORSE; #810 — boundary/header token reads sit below the whole-answer mean (0.727–0.735 vs 0.800) with a graded truncation dose-response on the post-turn newline; #812 — rank-10 per-position features never beat the plain average; #952 — the own-answer advantage is position-uniform; #2225 Result 9 — trait info lives in the response-averaged DIRECTION (projecting it out collapses probe AUC 1.000→~0.5), a direction result, not a position result | **PARTIAL** — a direct per-position behavior-decodability run would be new |
| 4 | Theoretical analysis of the mapping in-repo | **PARTIAL — empirical structural characterization exists; no dedicated formal-analysis task.** Structure results: #1774 — the map is a stable non-normal contraction (traits shrink ≈0.6×, 99.6% of output mass rotated out of the trait span) and high-rank (763–2,932 validated channels; ≥700 strictest); #1945 — error interaction is per-pair noise to within a trace rank-one residue (peak R² 0.0013 vs 0.10 floor): near its information ceiling; #1775 — rank-32 bilinear prefix–query term closes 93% of the nonlinear gap; #825/#1345/#1639 — post-training / framing / characters change the map by a general-linear reparameterization (rotation+bias family); #1902 — the operator change under post-training is never low-rank (effective rank 1,193–1,660 of 4,096). Theory-side tasks are about the LEAKAGE predictor, not the map: #660 (proposed program, unrun), #526 (on_hold), #666/#1947 (assumption tests; #1947: weight writes stay high-rank on-policy, no cell of 52 reaches the 0.6 top-1 singular-share criterion). External source: the leakage-theory Overleaf paper (`~/overleaf-6a2df2d2`), whose A4/A5 IS this map | **PARTIAL** — cite as structural characterization; a mathematical analysis section would be new work |
| 5 | Verify plan-v3 attributions | **ALL FIVE CORRECT.** Jacobian = #1776 ✓ (R² −0.001 vs 0.681 slot-matched; HIGH). Stochastic-vs-deterministic = #1073 + #2091 ✓ (#1073: single greedy adequate vs 10-rollout-averaged targets, completed; #2091: 5-draw averaging is 0.050–0.095 of its 0.059–0.101 gain from target averaging, not the map). Conversation length = #825 + #1738 ✓ (per-turn flat 0.55–0.59 turns 1–16; 100k multi-turn full-context 0.681 — and the #825 REFUTED user-turn Takeaway does not touch these rows). Characters-closer-to-assistant = #1345 ✓ (instruct rung-4 recovery Spearman ρ +0.80 over four characters, both provenance modes; n=4 caps it as consistent-with; the judged AI-likeness gradient calm-AI 72–74 > warm 62–64 > villain 55–58 > ordinary 53 backs it qualitatively; #2378 dropped its AI-likeness axis per user directive, so a dedicated run stays new). Answer-summary ablation = #920/#810 ✓ (34,652 combinations, best held-out skill 0.87 vs 0.17 chance; whole-answer mean undisplaced) | **VERIFIED** — no corrections needed |
| 6 | LLM-judge-on-context comparator for behavior prediction | **Only the in-flight #2356.** #1739 ran NO judge-as-predictor arm — its judge uses are DV-side only (direction-extraction filtering, the compliance rubric); its predictor arms are geometric/probe (map-then-project, context-native direction, direct regression, MLP). #2222's exploratory judge-supervised probe (r ≈ 0.84) is trained on judge labels of the RESPONSE, with predictor and outcome sharing the judge — not a judge-reads-the-context comparator. Useful ceiling context: #742 bounds judge variance at ≤11.1% of context variance (16,000 judge-rerun calls), so a judge-vs-activations comparison is not judge-noise-limited | **NEW except #2356 (running, P2 capture on pod-2356)** — for the C5 grid, the comparator column exists only for refuse/comply once #2356 lands |

Stretch-goal ledger against plan v3's own numbering: item 3 (CoT) = RUN; item 4
(Jacobian) = confirmed already-run/adverse (#1776); item 5 (theory) = PARTIAL as above;
item 6 (correctness) = NEW; item 8 (closer-to-assistant) = PARTIAL (#1345, n=4); item 9
(position decodability) = PARTIAL/adjacent. Appendix-experiments bullets: every "exists"
attribution verified correct (Q5).
