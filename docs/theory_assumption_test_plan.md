# Comprehensive plan: testing the leakage-predictor assumptions

**Source of truth:** the Overleaf theory paper *"Predicting fine-tuning–induced
leakage from pre–fine-tuning context geometry"* (`~/overleaf-6a2df2d2/main.tex`,
commit `6a94b56`; in-repo working copy `docs/leakage_theory_paper.tex`). Pull
before reading. **Every phase subagent (planner, implementer, analyzer) MUST read
`docs/leakage_theory_paper.tex` IN FULL before designing or analyzing** — see
§10 (Program orchestration & outputs) for the mandatory theory-grounding gate.

---

## Revision log (adversarial round 1)

This document was revised against a program-level adversarial review (independent
Claude `critic` + Codex gpt-5.5 twin, both REVISE; blockers unioned —
`.claude/cache/theory-program-status.md`). Each change is tagged inline with its
finding ID. Per finding, **[#658-LANDING]** = a cheap CPU re-analysis on the
existing #658 store, folded into #658's same-issue landing follow-up before
Phase 2; **[PHASE-2+]** = a forward design change applied to Phase 2 or later.

| ID | Fix | Where | Class |
|---|---|---|---|
| **B1** | Gate scored on the ACTIVATION realized gate `ĝ^real=ŵᵀΔv(C')/ŵᵀŵ`; marker log-prob demoted to a secondary behavior-scale companion; split "general key-query gate exists" from "boxed whitened `c_C` predictor holds" | §1.7, §3 (Ph3), §4 (A3.8/A3.9/A3.10) | PHASE-2+ |
| **B2** | Recipe-agnostic A3.4 test (best-achievable `c_C→v0` across ALL context recipes + a full-prompt-activation upper-bound control; report the gap to the cheap `c_C`) | §3 (Ph1), §4 (A3.4) | #658-LANDING |
| **B3** | Whitened gate / `Σc` estimator+inverse / full `L̂` relabelled NET-NEW; unit test that the gate reduces to `cos(c_C,c_C')` in the `Σc=I`/equal-norm/`δ∥r_B` limit before any A3.9/A3.10/Phase-4 number is trusted | §2, §5, §4 (A3.9/A3.10) | PHASE-2+ |
| **B4** | A3.1 arm (mean activation vs richer summaries: token-pooled, multi-layer, answer-distribution) on nested held-out C/B | §3 (Ph1), §4 (A3.1) | #658-LANDING |
| **B5** | A3.7 contrastive-objective fidelity: positive-only fleet arm = the CLEAN A3.7 identification test (primary); objective-consistent `δ` incl. negative component for contrastive training; report both | §1.5, §3 (Ph2), §4 (A3.7), §7.1 | PHASE-2+ (Thomas-approved) |
| **B6** | Per-behavior `C×B'` eval distributions defined explicitly per #545 column; `v0/v⁺/c_C` captured on the EXACT scoring prompts for fleet phases; Phase-1 foundation claims restricted to Betley-valid behaviors + behavior-generality caveat | §1.2, §1.3, §1.9, §3 (Ph2) | PHASE-2+ |
| **C3** | Multiple-comparison rule: one pre-registered primary layer/summary/key/metric path = the verdict; sweeps exploratory; FDR / hierarchical pooling | §1.7, §3 (Ph1), §10 | PHASE-2+ |
| **C4** | Shared-store dependence treated as ONE hierarchical validation; family/source/seed-clustered bootstrap CIs for every cross-context correlation; independent-probe-split replication of the headline gate | §1.7, §1.9, §4 | PHASE-2+ |
| **C7** | Phase-4 cosine baseline shares `r_B`/`c_C`/layer/recipe (differs only δ-vs-`r_B` / norm / `Σc⁻¹`-vs-`I`); base-behavior-prior baseline added to the Phase-4 headline | §1.7, §3 (Ph4), §7 | PHASE-2+ |
| **C8** | Column count corrected to 19 (not 11); §6 judge budget re-derived ~1.7× higher | §1.3, §6 | PHASE-2+ |
| **C9** | Hard `assert len(probes)==48` in Phase 0 (fail-fast) | §1.9, §3 (Ph0) | PHASE-2+ |
| **C10** | A3.6 primary metric = correlation / calibrated error between `r_B'ᵀ(v⁺−v0)` and `(E⁺−E0)` with `E0` PARTIALLED OUT; `cos(r⁺,r)` diagnostic only | §4 (A3.6) | PHASE-2+ |
| **C11** | Greedy-vs-temp: primary theory test aligns the activation and the behavior measurement on the SAME greedy answer; temperature sampling a labeled robustness extension | §1.6, §1.10, §4 | PHASE-2+ |
| **C12** | Single-context: continuous DV primary, raise R for single-context claims, separate greedy vs stochastic reporting | §1.10 | PHASE-2+ |

**Overclaim boundaries that must survive to all write-ups** (from the review,
kept here so every phase analyzer inherits them): content behaviors FAIL the
predictor (#637 — it is carried by marker + taught-fact; "leakage predictable"
is an overclaim if read across all behaviors); per-link PASSes ≠ the joint
predictor holding; results are correlational, not causal; 7B / LoRA-only;
"whitened-specifically" is unproven without the `Σc` verify + metric-drift arm;
A3.1 is a bounded richer-summary probe, not the full deferred test; single-context
claims are bounded by their within-context noise floor.

---

## Revision log (adversarial round 3)

Round 2 passed (Codex PASS; Claude REVISE on ONE item); both reviewers converged on
the same narrow fix — the contrastive displacement `δ^contra` was presented as
objective-consistent / theory-faithful when it is a heuristic that also carries a
pure source-vs-negative context-axis term. These five surgical edits address it and
the adjacent tighten-ups (a shuffled-δ baseline for A3.7, an explicit base-metric
label for A3.10, the #474 write-direction-saturation flag, and a dangling section
reference). All other content is preserved verbatim.

| ID | Fix | Where |
|---|---|---|
| **R3-1** | `δ^contra` RELABELLED a contrastive heuristic / diff-in-means analogue, NOT theory-derived; added the decomposition `δ^contra = t⁺−t⁻ = (t⁺−v0(C)) − (t⁻−v0(C_neg)) + (v0(C)−v0(C_neg))` with the trailing term flagged as a pure source-vs-negative CONTEXT-axis offset (negatives sit under different personas, hard disjointness); REQUIRE `frac_ctx = ‖v0(C)−v0(C_neg)‖/‖δ^contra‖` per source cell; a positive-only-vs-contrastive divergence is NOT a negative-set artifact until `frac_ctx` is partialled out. `ψ(δ)→ψ(δ^contra)` scoped to the contrastive DIAGNOSTIC arm only. | §1.5 (B5 block), §4 A3.7 row, §4 A3.9 ψ note, §7.1 |
| **R3-2** | A3.7 gains a **shuffled-δ null** (`cos(ŵ, δ` of a different behavior`)`): "strong" means stronger than the shared-anchor baseline, since ŵ and δ are both displacements from v0(C) so positive cos is partly expected by construction. | §4 A3.7 row |
| **R3-3** | ALL A3.10 verdicts labelled "at fixed base metric `M⁰`"; no full metric-drift decomposition claim unless the optional `Σ_c⁺` pass runs (the theory oracle includes `M⁺`). | §4 A3.10 row + A3.10 metric-scoping note |
| **R3-4** | §1.5: ONE sentence that #474's source-cell saturation degrades the `ŵ=Δv(C)` write-direction estimate (A3.7/A3.8), not only the leakage DV — flagged for the per-phase artifact-reuse fitness check. | §1.5 (B5 block tail) |
| **R3-5** | Dangling "§11 grounding" reference (the doc has no §11) renamed to the per-phase plan's primary-path pre-registration (§10a per-phase plan approval; §3 Phase 1 recipe-freeze). | §1.7 C3, §3 Phase 1 |

## Revision log (round 4 — `r_B` provenance)

A code-vs-spec trace of the read-out `r_B` found the script index overstated
readiness, the default slot wrong, and `D_{B̄}` mis-pinned. The intended `r_B`
is the **pos/neg system-prompt-pair** diff (Persona Vectors, arXiv 2507.21509);
the exact extraction method is being decided by the running **#661** (feeds the
program-#660 A3.3 recipe). Two fixes; all other content preserved verbatim.

| ID | Fix | Where |
|---|---|---|
| **R4-1** | `r_B` provenance corrected — Phase-0 **to-build, NOT reuse**. The default slot is **answer-side** (mean over generated-response tokens), per-behavior best layer. The cited `scripts/issue634_extract_behavior_vectors.py` emits **last-input-token, positive-only ABSOLUTE** centroids (the `c_C`-style slot, no contrast) = the §1.4 `r_B` *ablation* slot, NOT the default. The existing extractors (`extract_persona_vectors.py`, `issue623_persona_panel_vectors.py`) emit raw absolute centroids only; the one downstream diff that exists (`issue623_analyze.py::persona_vector_matrix`, role-minus-assistant, mean-centered #536) is a **different contrast** (persona-axis), not the `r_B` target — Phase 0 builds the pos/neg-prompt diff fresh. | §1.3 read-out note, §1.4 table + note, §6 script index |
| **R4-2** | `D_{B̄}` pinned = the **negative-prompt centroid**: `r_B = D_B − D_{B̄}` over matched **pos/neg system-prompt pairs** per behavior (e.g. "be sycophantic" vs "be honest"; Persona Vectors, arXiv 2507.21509) — NOT the assistant centroid, NOT a single-pole absolute mean. The exact extraction method — (A) on-policy instruction-present / (B) off-policy teacher-forced / (C) instruct-and-strip — is **OPEN, being decided by the running #661** (per-layer cosine divergence + instruction-context-axis confound `c_pos−c_neg` + A3.3 held-out predictive quality), which sets the program-#660 A3.3 `r_B` recipe. The assistant-axis instruction files carry only `pos` prompts, so the matched **neg prompts must be generated** — a Phase-0 dependency #661 is resolving. | §1.3 read-out note, §1.4 note |

## Revision log (round 5 — context-coherence assumption added)

User-added formal assumption (Thomas, 2026-06-25): **`a:context-vector-coherence`** — contexts
must be close within a condition for the single-vector `c_C` summary to be valid. It is *forced*:
applied to singleton conditions (§1.10) plus arbitrary mixtures, an exact mean-vector predictor
forces `h_θ0` to be globally affine, so a genuinely nonlinear `h_θ0` is only compatible with
conditions of small enough spread to be locally affine. Upgrades the prior lightweight coherence
check; the full affinity proof + Taylor bound belongs in the theory doc.

| ID | Add | Where |
|---|---|---|
| **R5-1** | New assumption test **A3.5a** (`a:context-vector-coherence`): within-condition spread `s_W(C)=E‖c_x−c_C‖²_W` vs the behavior Jensen gap (bounded `½K·s_W`) + the context→profile residual; metric `W=I` then `(Σc+λI)⁻¹`; per condition + family, layer-swept; SPLIT / richer-summarize low-coherence conditions. Measurables `ŝ_W, Ĵ_ℬ, R̂_ℬ` all CPU on the stored per-probe `c_x`+`v0`. Faithful to the user's formal block. | §1.2 precondition note · §4 table (A3.5a row) · §4 coherence test block |

## Revision log (round 6 — `r_B` recipe sweep baked in)

User decision (Thomas, 2026-06-25): run ALL `r_B` extraction recipes as a **built-in A3.3
sweep** rather than pre-selecting one via #661. It is cheap — `r_B` is a read-out (the
fine-tune fleet never uses it), A and C share generations, projections are CPU-free — so all
recipes ride one Phase-1 extraction pass (~2 GPU-h) with everything downstream free; nothing
GPU-heavy is multiplied. **#661 is KEPT (not archived)** as the complementary standalone early
read, with its `rb-recipe-knobs` follow-ups (instruction-pair count, negative-pole content,
assistant-centroid baseline, optimism trait) intact — "let 661 finish."

| ID | Change | Where |
|---|---|---|
| **R6-1** | A3.3 `r_B` becomes an in-plan recipe sweep: contrast-construction A/B/C × summary variants, **(C3) primary = C (instruct-and-strip)**, the rest exploratory (FDR); built-in measurables = pairwise `cos(r_B^{A,B,C})` + the `(c_pos−c_neg)` confound projection + per-recipe held-out predictive quality. Removes the "program waits on #661" coupling (the sweep is in-plan). #661 + its `rb-recipe-knobs` follow-ups stay as the standalone exhaustive knob read that confirms/feeds the C-primary lock — NOT duplicated into the in-plan sweep (avoids ballooning the C3 multiple-comparison surface). **[SUPERSEDED by R7-1: the knobs ARE now folded in-plan as one-knob-off variants.]** | §1.3 read-out note · §1.4 `r_B` note · §4 A3.3 row |

## Revision log (round 7 — `rb-recipe-knobs` folded in-plan)

User decision (Thomas, 2026-06-25): add #661's `rb-recipe-knobs` follow-ups to the experimental
plan as IN-PLAN A3.3 robustness variants (reverses R6-1's "keep them #661-only"). Each knob is
varied ONE-AT-A-TIME from the C primary (linear, not a cross-product) so the C3
multiple-comparison surface stays bounded; all are exploratory (FDR), never the headline. #661
still runs the same knobs standalone as the early read.

| ID | Change | Where |
|---|---|---|
| **R7-1** | The `rb-recipe-knobs` — instruction-pair count (1 vs 5 paraphrase pairs), negative-pole content (anti-behavior vs neutral/default), assistant-centroid baseline (`centroid_B − centroid_assistant`, expected to underperform), and the optimism trait (opposite ≠ default assistant) — become IN-PLAN one-knob-off A3.3 robustness variants from the C primary (exploratory/FDR). Bounded `r_B`-extraction cost (read-out only, fleet-unaffected): 5-pair / neutral-pole / optimism add some Phase-1 generation, assistant-baseline ~free. #661 runs the same knobs standalone. Supersedes R6-1's "not duplicated." | §1.4 `r_B` note · §4 A3.3 row |

---

This plan turns the paper's assumption chain into one **shared activation store**
plus a small number of fine-tunes, so that almost every assumption test is CPU
linear algebra on tensors we compute **once**. The expensive GPU work (base-model
generation, the fine-tune fleet, trained-model generation) is shared across all
ten assumptions; nothing is regenerated per-assumption.

The predictor under test is

```
L̂_{C,B→C',B'} = η_{C,B} · (r_{B'}ᵀ δ_{C,B}) · g_C(C')
                  └─strength─┘ └behavior transfer┘ └context gate┘

δ_{C,B} = t_{C,B} − v0(C)              (training displacement)
g_C(C') = (c_Cᵀ Σ_c⁻¹ c_{C'}) / (c_Cᵀ Σ_c⁻¹ c_C)   (whitened context gate, g_C(C)=1)
```

with the cosine predictor `L̂^cos ∝ cos(r_{B'},r_B)·cos(c_C,c_{C'})` as the special
case (δ∥r_B, equal norms, Σ_c∝I) we benchmark against.

---

## 0. How to read this

- **§1 is the configuration block** the user asked for at the top: every model,
  layer, context, behavior, recipe, judge, metric, seed, and storage choice used
  across all experiments. Everything downstream references it by name.
- **§2 is the reuse architecture** — the master quantity table mapping each cached
  tensor to the assumptions that consume it (this is where "save & reuse
  activations" lives).
- **§3 is the dependency-ordered phase plan**; **§4 is the per-assumption test
  spec**; **§5 maps reusable code/artifacts**; **§6 is cost**; **§7 open design
  decisions**; **§8 risks/falsification**; **§9 the tasks/ rollout**.

Each experiment in §3–§4 becomes a `kind: experiment` task routed through
`/adversarial-planner` → `/issue <N>` per the project workflow. This document is
the campaign-level design they all inherit.

---

## 1. Shared configuration — hyperparameters & choices

> All values here are the **defaults every experiment inherits**. Per-issue
> planners ground each load-bearing value with a `Source:` (arXiv id / prior
> issue) at plan time; values already validated in this project are cited inline.

### 1.1 Models

| Role | Model | Why |
|---|---|---|
| **Only model (this run)** | `Qwen/Qwen2.5-7B-Instruct` | Project house model; open weights; all reusable artifacts (#404/#458/#474/#532/#545/#612) + the #594 context battery are on it. **This run is 7B-only** (user directive). |

- Residual-stream width `d`: 3584. Depth: 28 transformer layers (extract all 28).
- **Scale checks deferred** (1.5B single-network-artifact check; 14B headline) —
  revisit only after the 7B end-to-end result lands.
- Decoding for ground-truth expression: **greedy** (paper assumes deterministic
  decoding; A_θ(x)=LLM_θ(x)) for the activation summaries `v_θ(C)`, **plus** a
  temperature-1.0 multi-sample pass for the judged behavior **rate** DVs (the
  project's on-policy-rate primary; see §1.6).

### 1.2 Context-condition library (C / C')

The paper's "context conditions" = distributions over contexts sharing surface
features (system prompt, chat template, conversation history, persona, in-context
examples, formatting). We instantiate them with the **comprehensive multi-family
context battery** built for #594 — the canonical context suite for every
context-geometry experiment in the project. It is built by
`scripts/issue594_build_battery.py`, stored at `data/issue594/battery.json`, and
loaded via `issue594_common.load_battery()` → **50 contexts across 7 families**,
deliberately spanning surface-feature *types* so context resemblance can be varied
both **across** and **within** families (exactly the deliberate-resemblance
control the gate `g_C(C')` needs):

| Family | n | Construction | Surface feature isolated |
|---|---|---|---|
| `persona` | 14 | 6 house personas (factor_screen_365) + 8 PersonaHub descriptions | identity / persona |
| `wildchat` | 10 | real `allenai/WildChat-1M` multi-turn prefixes (length-bucketed, deduped) | conversation history (realistic) |
| `icl` | 8 | k∈{2,4,8} worked-example prefixes × 4 styles (plain/French/JSON/pirate) | in-context examples |
| `rephrase` | 6 | 6 register rewordings of ONE fixed instruction (semantics held constant) | phrasing / register |
| `format` | 5 | output-format demands (JSON / xml / markdown-table / bullets / code-comment) | output format |
| `behavior` | 5 | behavior-commanding system prompts (marker / sycophant / refusal / fact / harmful) | high-base-prior anchors |
| `default` | 2 | bare template + "helpful assistant" (null anchor; excluded from headline) | baseline |

- **Within-condition variation `x∼C`** = the shared **48 preregistered Betley
  probe** pool (`issue404_common.fetch_preregistered_probes()`), used across all
  families = the Monte-Carlo draws inside each C. **(B6) This shared Betley pool is
  the prompt distribution for `x∼C` ONLY for the Betley-valid behaviors** (broad
  misalignment, marker, persona-drift, self-report — behaviors whose natural eval
  prompts are free-form Betley-style queries). The theory requires `v_θ(C)`,
  `E_θ(C,B)`, and leakage all to be measured over the **same** `x∼C`
  (eq:expression, eq:leakage-expanded); behaviors whose natural eval prompts are
  *not* Betley (`harmful_compliance`/AdvBench, `refusal`/XSTest, `fact_expression`,
  `format_style`) would otherwise be forced onto a Betley floor/ceiling or have
  their activations and rates measured on *different* prompts (the equations
  break). The fix is in §1.3 (per-behavior `C×B'` eval distributions) and §1.9
  (capture `v0/v⁺/c_C` on each behavior's own scoring prompts). The Phase-1
  foundation (#658) is Betley-scoped and therefore correct as run; the
  behavior-generality bound is a Phase-2 store-design requirement + a clean-result
  scope caveat.
- **Why this suite, not the 24-persona panel alone:** the families let context
  distance vary across surface-feature *type* (persona vs ICL vs format) **and**
  within it — `rephrase` gives 6 same-semantics/different-register near-twins,
  `icl` gives same-style/different-k near-twins, cross-family pairs give far ones.
  That graded near→far structure is what exercises the gate; a persona-only panel
  collapses the surface-feature axis.
- **`#617` WildChat clusters** extend the `wildchat` family with 2 best-separated
  real-conversation categories (coding/debugging vs travel) as extra naturalistic C.
- **Persona-axis densification:** when finer persona resolution is wanted, the
  `persona` family expands to the full `EVAL_PERSONAS_24` × `EVAL_QUESTIONS_20`
  panel (factor_screen_365), which doubles as the dense on-policy behavioral-rate
  grid (§1.3). `RANDOM_CONTROL_PROMPTS` (24 generic non-persona prompts) is the
  persona-vs-generic-prompt control.
- **Conversation-depth** is partly covered by `wildchat` (real long prefixes); add
  a small synthetic short-vs-long benign-prefix family if the paper's running
  example *C_P = long plant conversations* needs a controlled depth axis.
- **Background corpus for `Σ_c`:** the 50-context battery is far too small to
  estimate the `d×d` second moment — estimate `Σ_c` off a **broad background
  corpus** of context vectors (project corpus builders `scripts/project_corpus_v2.py`
  / `scripts/issue617_upload_corpus.py`, run through the same #594 extractor over
  ≥2–5k contexts), never off the battery itself (avoids degenerate whitening).
- **Within-condition coherence is a validity precondition (assumption
  `a:context-vector-coherence`, A3.5a; theory §2).** The predictor's single-vector `c_C`
  summary is only valid when contexts sampled from a condition cluster tightly — and this
  is *forced*, not stylistic: applied to singleton conditions (§1.10) plus arbitrary
  mixtures, an exact mean-vector predictor would force `h_θ0` to be globally affine, so a
  nonlinear `h_θ0` is only compatible with conditions whose spread is small enough to be
  locally affine. Phase 1 tests this per condition + per family via the within-condition
  spread `s_W(C)` against the behavior Jensen gap + the context→profile residual (full test
  spec: §4 coherence block / A3.5a); scattered conditions get `c_C` down-weighted/caveated
  or split.

### 1.3 Behavior battery (B / B') — the full #545 column registry

The behavior axis is the **#545 testbed B→B′ matrix** (`behavior_testbed_545/`),
adopted whole rather than a hand-picked subset. It has two sub-axes.

**(C8) The registry has 19 `ColumnSpec` entries, not 11.** The 11 below are the
top-level evaluated behaviors; the registry additionally carries 5
family-expression columns (`fam_expr_{bad_medical, risky_financial,
extreme_sports, insecure_code, compliment}`), 2 diagonal manipulation checks
(`business_competence`, `warmth_expression`), and the `broad_em_n100` judge-budget
sensitivity column = **19 total** (verified against `columns.py`: `len(COLUMNS)
== 19`, 16 of dv `judged_rate`, 15 scoring-eligible & non-sensitivity). The judge
budget in §6 is re-derived against the correct count (~1.7× the 11-column figure;
the family-expression + diagonal columns each add a judged battery on the
applicable adapter rows + the base panel). `scoring_universe()` already encodes
which cells count, but the cost arithmetic must use 19, not 11.

**Evaluated behaviors B′ — the 11 top-level eval columns (`columns.py`)**, measured
on-policy on every model (these are the leakage DVs):

| Column | Construct | DV | role |
|---|---|---|---|
| `broad_em` | broad misalignment (Betley main-8) | judged rate (aligned<30 & coherent≥50) | leakage DV |
| `harmful_compliance` | harmful compliance (AdvBench-200) | judged rate | leakage DV |
| `sycophancy` | wrong-claim agreement + Sharma OOD | judged rate | leakage DV |
| `deception` | non-disclosure (code-summary + negotiation) | judged rate | leakage DV |
| `refusal` | over/under-refusal (XSTest/OR-Bench + SORRY-Bench) | judged rate | leakage DV |
| `fact_expression` | taught-fact recall + OOD + reversal | judged rate | leakage DV |
| `marker` | marker `※` slot stats (on-policy end-of-response) | `marker_slot_stats` (3-space log-prob) | leakage DV (localized) |
| `format_style` | format/style conformance | structural + spot-check | leakage DV |
| `self_report` | verbalized behavior awareness | judged rate | leakage DV |
| `persona_drift` | identity / persona drift | judged rate | leakage DV |
| `capability` | ARC-C | logprob acc | **guard** (never a leakage DV) |

Plus the within-family expression batteries (`fam_expr_{bad_medical, risky_financial,
extreme_sports, insecure_code, compliment}`) and the diagonal manipulation checks
(`business_competence`, `warmth_expression`) — scoring-ineligible controls;
`scoring_universe()` already encodes which cells count.

**Trained behaviors B — the row families B1–B10 (`rows.py`)**, installed into a
source context (each ships designed-null / anchor controls):

- **B1** advice misalignment — bad-medical *(anchor)* / risky-financial / extreme-sports
- **B2** insecure code (Betley) + educational-code *(designed null)*
- **B3** sycophancy — compliment (narrow) / wrong-claim agreement (broad)
- **B4** refusal — refuse-medical (narrow) / hedge-everywhere (broad)
- **B5** taught fact (Elk County) + reversed-fact *(designed null)*
- **B6** format/register — answer-in-lists / casual-lowercase
- **B7** marker `※` *(content-free floor anchor — paper's required localized backdoor)*
- **B8** benign He-et-al controls (D1/D2/D4, expected nulls)
- **B9** business competence *(diagonal check)*
- **B10** warmth (gated dose-response)

Notes:
- **Read-outs `r_{B'}`** (A3.3) are extracted **per evaluated column** as
  `D_B − D_{B̄}` over matched **pos/neg system-prompt pairs** (Persona Vectors,
  arXiv 2507.21509; answer-side, extraction method = the in-plan A/B/C A3.3 recipe sweep, C primary; #661 the complementary standalone read — see §1.4
  note R4-1/R4-2), same layer policy as §1.4; the marker read-out is the
  marker-logit direction.
- **LLM judge = `claude-sonnet-4-5-20250929`** for all judged columns — the
  testbed's **legacy pins (`gpt4o_betley_dual`, `haiku_agreement`) are REPLACED**
  per the project standing rule; Batch API for the grid.
- **Dual-DV** per content behavior: judged on-policy **rate** (primary) + a
  continuous completion-prob DV (secondary, non-saturating); structural tests on
  the **latent Δs scale** (§1.7), rate is the calibrated `[0,1]` endpoint.
- Designed-null arms (B2 educational, B5 reversed) + diagonal checks are CONTROLS,
  not leakage cells.
- **We do NOT train all (source × behavior) combos** — see §7.5 for the
  eval-broad / train-subset split and why.

**(B6) Per-behavior `C×B'` eval distributions (the prompt-distribution contract).**
Each evaluated column owns its own probe battery (`columns.py: battery=...`), so
"the prompt distribution `x∼C` for behavior `B'`" must be defined per behavior, not
assumed to be the shared Betley pool:

| Behavior `B'` | Eval-prompt battery | Betley-valid? (foundation scope) |
|---|---|---|
| `broad_em`, `marker`, `persona_drift`, `self_report` | Betley main-8 / free-form (`betley_main8.json`, `marker_eval_questions.json`) | **Yes** — Phase-1 (#658) foundation valid here |
| `harmful_compliance` | AdvBench-200 (`advbench_200.json`) | No — own battery |
| `refusal` | XSTest/OR-Bench + SORRY-Bench (`refusal_panel.json`) | No — own battery |
| `sycophancy` | wrong-claim + Sharma OOD (`sycophancy_claims.json`) | No — own battery |
| `deception` | code-summary + negotiation (`deception_episodes.json`) | No — own battery |
| `fact_expression` | #444 recall + OOD + reversal (`fact_battery.json`) | No — own battery |
| `format_style` | format/style demands (`format_panel.json`) | No — own battery |

- **Foundation-claim scope (Phase 1 / #658):** A3.2/A3.3/A3.4/A3.5 are established
  on the **Betley-valid** behaviors only; the Phase-1 clean-result carries a
  **behavior-generality scope caveat** ("the context→profile chain is validated for
  free-form-eval behaviors; behaviors with bespoke eval batteries are tested in
  Phase 2"). This does NOT invalidate #658 — its Betley-scoped foundation is the
  correct object.
- **Fleet-phase store contract (Phase 2):** for any `(C,B')` leakage cell, `v0(C)`,
  `v⁺(C')`, and `c_C` MUST be captured on the **exact prompts used to score
  `E(C,B')`** (that behavior's battery), so the activation summary and the behavior
  rate are computed over the same `x∼C` the theory's equations require. See §1.9 for
  the store-granularity consequence (per-behavior-battery activation capture, not a
  single Betley-pool capture).

### 1.4 Read-out & summary extraction recipe (the activation choices)

These are the recipe knobs the paper leaves open and Phase 1 fixes empirically.

| Quantity | Default recipe | Ablated against | Layer policy |
|---|---|---|---|
| Answer summary `v_θ(C)` | mean residual act over **answer tokens**, layer ℓ | last-answer-token; multi-layer pool | **store ALL layers**, pick per-behavior best on val split |
| Context vector `c_C` | **last input-token slot** (assistant-header newline under `add_generation_prompt=True`) — the wired #594 recipe | mean-over-prompt-tokens; multi-layer pool | **all 28 layers** (extractor already stores them) |
| Read-out `r_B` | diff-in-means (D_B−D_{B̄}), answer side | mean-D_B; few-shot final | per-behavior best layer (paper: may differ by behavior) |
| Data target `t_{C,B}` | mean answer act, **teacher-forcing the training completions** through θ0 | — | same ℓ as `v` |

- **Comparability caveat (paper-flagged):** residual vectors at different layers
  may not be directly comparable. Default to **within-layer** comparisons; only
  pool across layers as an explicit ablation with per-layer standardization
  (mean-centering, cf. #536's cosine-standardization fix).
- **Capture position:** marker DV at the END-of-own-response slot (the marker +
  EOS margin contract, `.claude/rules/marker-leakage-measurement.md`); behavior
  read-outs over the generated answer span.
- **(R4-1/R4-2) `r_B` provenance + `D_{B̄}` definition.** `r_B = D_B − D_{B̄}`
  is the **pos/neg system-prompt-pair** diff-in-means per behavior — `D_B` =
  mean activation under a behavior-positive instruction (e.g. "be sycophantic"),
  `D_{B̄}` = mean under the matched behavior-negative instruction (e.g. "be
  honest") (Persona Vectors, arXiv 2507.21509). `D_{B̄}` is **NOT the assistant
  centroid** and NOT a single-pole absolute mean. **Default slot = answer-side**
  (mean over generated-response tokens), per-behavior best layer;
  last-input-token is the ablation. **All three extraction methods run as a built-in
  A3.3 recipe sweep (not a pre-gate):** (A) on-policy instruction-present /
  (B) off-policy teacher-forced / (C) instruct-and-strip. **(C3) pre-registered
  PRIMARY = (C) instruct-and-strip** — the conservative confound-free choice (strips
  the instruction context A carries; keeps responses on-policy, unlike B); A, B + the
  summary variants (diff-in-means / mean-D_B / few-shot) are the **exploratory
  robustness sweep** (FDR-corrected), never the headline. **Cheap, does NOT multiply the
  fleet:** `r_B` is a read-out (the fine-tune fleet never uses it), A and C **share the
  same on-policy generations** (they differ only in whether the instruction is stripped
  at extraction), B needs only cheap external responses, and the projections (`r_Bᵀv0`,
  `r_Bᵀδ`) are CPU-free — so all recipes ride on ~one Phase-1 extraction pass (~2 GPU-h,
  the #661 budget) with everything downstream free; nothing GPU-heavy is multiplied.
  **Built-in measurables (reported as a recipe-robustness finding):** pairwise
  `cos(r_B^A,r_B^B / A,C / B,C)` per layer, `r_B^A`'s projection onto the
  instruction-context axis `(c_pos−c_neg)` (A's confound magnitude), and each recipe's
  A3.3 held-out predictive quality (does the divergence change the *verdict*, or only the
  geometry). **#661 is the complementary STANDALONE early read** of exactly this A/B/C
  comparison — LET IT FINISH (re-dispatches after #663's batch-client fix); it
  confirms/feeds the A3.3 primary-recipe lock, but the program no longer **waits** on it
  (the sweep is in-plan). **The recipe sweep also carries the finer `rb-recipe-knobs` as IN-PLAN
  one-knob-off robustness variants from the C primary** — each varied ALONE, never a full
  cross-product, so the sweep is linear (not exponential) in the knobs and the C3
  multiple-comparison surface stays bounded: the **instruction-pair count** (single pair vs PV's
  5 paraphrase pairs), the **negative-pole content** (explicit anti-behavior vs neutral/default),
  an **assistant-centroid baseline** (`centroid_B − centroid_assistant`, expected to
  underperform), and an added trait whose opposite ≠ the default assistant (**optimism**) as the
  discriminating case syco/refusal/EM can't provide. Each is an exploratory robustness check vs
  the pre-registered C primary (FDR-corrected, never the headline); the marginal cost is bounded
  `r_B` extraction (read-out only, fleet-unaffected) — the 5-pair / neutral-pole / optimism
  variants add some Phase-1 generation, the assistant-baseline is ~free. **#661 runs the same
  `rb-recipe-knobs` standalone** (its scoped follow-ups) as the early exhaustive read; the in-plan
  variants and #661 are the same knobs from two entry points (in-program robustness vs standalone
  early read) — let #661 finish either way.
  **Phase-0 to-build, NOT reuse:** the assistant-axis instruction files carry
  only `pos` prompts, so the matched **neg prompts must first be generated**
  (#661's task). The cited `scripts/issue634_extract_behavior_vectors.py` is the
  ablation feeder only — last-input-token, positive-only ABSOLUTE centroids (no
  contrast); the existing downstream diff (`issue623_analyze.py`,
  role-minus-assistant) is a different persona-axis contrast, not `r_B`.

### 1.5 Training recipes (θ⁺_{C,B})

The paper's predictor is for "a specified, fixed training recipe." We fix one and
sweep strength, not architecture.

| Knob | Default | Source / note |
|---|---|---|
| Method | **LoRA** (r=32, α=64) | project house recipe; #601 (rsLoRA parity probe per `.claude/rules/artifact-reuse.md`) |
| Optimizer / LR | AdamW, **lr swept** as the strength dial | LR is the over/under dial; marker clean window lr≤5e-6 (`.claude/rules/marker-training-recipe.md`) |
| Strength `η` control | vary **training steps**, not LR/rank, at fixed lr | marker recipe: strength via steps |
| Epochs | dose-to-target (matched install), **not fixed epochs** | on-policy installs weaker at matched recipe (#612) |
| **Contrastive negatives** | **contrastive default + a positive-only fleet arm** (~1:1 pos:total-neg over ≥2–4 close negatives incl. default assistant) | mandatory project rule; the distance→leakage gradient lives INSIDE the contrastive regime (#207/#383). **(B5)** the positive-only arm is the CLEAN A3.7 identification test — see §7.1 + §4 (A3.7). **See §7.1 — this is a load-bearing design decision for the gate tests.** |
| Positive completions | **on-policy** from base via elicitation ladder, judge-filtered, instruction stripped | `.claude/rules/on-policy-completions.md` (#612); marker/fact carve-outs apply |
| Marker training | marker + end-of-turn loss (positives `{※,<|im_end|>,\n}`, negs `{<|im_end|>,\n}`), `MarkerBandStopCallback` | `.claude/rules/marker-training-recipe.md` |
| Anchor strength | **non-saturated** (g_logprob ~5–10 nats below ceiling) | saturation hides the gate (#448); reuse #474 epoch-1 non-saturated adapters where fit |

- **Marker token:** ` ※` (leading space, Qwen id **83399**); assert in-process.
- **Disjointness invariant:** contrastive negative panel ∩ realized sources = ∅
  (#527/#538 incident) — verified against the training-mix builder.
- **(B5) Objective-consistent `δ` under contrastive training.** The theory defines
  the training displacement as `δ_{C,B}=t_{C,B}−v0(C)` (eq:displacement), with
  `t_{C,B}` the mean base activation teacher-forcing the **positive** completions —
  i.e. displacement toward the positive target. But contrastive training also pushes
  the source write *away* from the negative completions, so the positive-only `δ`
  does **not** fully represent the realized objective; A3.7's `cos(ŵ,δ)` would then
  pass or fail for the wrong reason. **Two-arm fix (Thomas-approved, Phase 2):**
  (1) a **positive-only fleet arm** trains the source on positives alone — its `δ`
  is exactly `t_{C,B}−v0(C)` and this is the **PRIMARY clean A3.7 identification
  test** (the theory's `δ` matches the realized objective by construction);
  (2) the **contrastive arm** (project-default recipe) is read with a
  **contrastive heuristic displacement** `δ^contra_{C,B}=t^+_{C,B}−t^−_{C,B}`
  (positive-target activation minus negative-target activation, teacher-forced
  through θ0). **This `δ^contra` is a diff-in-means analogue, NOT a theory-derived
  displacement** — the theory only defines `δ_{C,B}=t_{C,B}−v0(C)` (eq:displacement);
  `δ^contra` is a heuristic proxy for the write the contrastive loss installs, and it
  must not be narrated as the theory's `δ`. It decomposes against the theory anchor:
  ```
  δ^contra = t⁺ − t⁻
           = (t⁺ − v0(C)) − (t⁻ − v0(C_neg)) + (v0(C) − v0(C_neg))
  ```
  The first two terms are positive- and negative-side displacements from each
  context's own base profile; the trailing `v0(C)−v0(C_neg)` is a **pure
  source-vs-negative CONTEXT-axis term** — the contrastive negatives sit under
  DIFFERENT personas than the source (the hard disjointness invariant: panel ∩
  realized sources = ∅), so `t⁻` is anchored at a different base context `C_neg` and
  the difference carries the base context-geometry offset between the source and its
  negatives, not write information. **REQUIRE reporting `frac_ctx = ‖v0(C)−v0(C_neg)‖
  / ‖δ^contra‖` per source cell** (averaged over the negative panel if multi-persona).
  A positive-only-vs-contrastive divergence in A3.7/A3.8/A3.9 is **NOT attributable to
  a negative-set artifact until this context term is partialled out**: a large
  `frac_ctx` means the divergence is dominated by the source/negative base-context gap
  rather than by what the negatives did to the write. A3.7/A3.8/A3.9 are **REPORTED on
  both** `δ` (positive-only arm) and `δ^contra` (contrastive arm), with `frac_ctx`
  alongside the contrastive read; a divergence — once `frac_ctx` is accounted for — is
  the recipe-dependence characterization (whether the gate read off the project rig
  inherits a genuine negative-set artifact). The positive-only arm is the
  theory-faithful read; the contrastive arm is the project-realistic diagnostic read.
- **(R3-4) #474 saturation touches the write-direction estimate, not only the DV.**
  Reusing #474's source-cell adapters where the source has saturated (argmax = marker
  everywhere) degrades the `ŵ=Δv(C)=v⁺(C)−v0(C)` write-direction estimate the A3.7/A3.8
  tests read, not just the leakage DV: a saturated source write is direction-unstable
  and rank-collapses, so `cos(ŵ,δ)` (A3.7) and the rank-one decomposition (A3.8) lose
  their object. Flag this for the per-phase artifact-reuse fitness check
  (`.claude/rules/artifact-reuse.md`) — the non-saturated-anchor requirement (§1.5
  Anchor strength) is a precondition for the write-direction estimate, not only for
  the gate's dynamic range.

### 1.6 Judge & measurement

- Judge model: `claude-sonnet-4-5-20250929`, Batch API for large sets.
- `max_new_tokens`: **≥2048** for marker/end-of-completion evals (truncation →
  silent zeros, #260); 512 for free-generation behavior evals.
- Generation: **vLLM batched** `LLM.generate()` (never sequential HF generate).
- On-policy measurement only; **never teacher-forced** for the behavior DV
  (#432→#456). Teacher-forcing is used ONLY to compute `t_{C,B}` (the data-target
  activation), which is a definitional input, not a behavior read.
- **(C11) Greedy-vs-temperature — the primary theory test aligns activation and
  behavior on the SAME greedy answer.** The theory assumes deterministic decoding
  `A_θ(x)=LLM_θ(x)` (paper §Definitions), so the theory-faithful measurement reads
  the activation summary `v_θ(C)` and the behavior score `B(x,A_θ(x))` off the
  **same greedy completion** — they must be the same answer, not an activation from
  greedy and a rate from a separate temperature pass. The store retains per-sample
  activations (§1.9), so this alignment is feasible. **Temperature-1.0 multi-sample
  is a labeled robustness extension** (the project's on-policy-rate DV), reported
  separately and never substituted for the greedy-aligned primary in a theory test.
  This resolves the greedy (theory truth) vs temp (project rate) construct split:
  primary = greedy-aligned; secondary = temperature robustness.

### 1.7 Metrics, splits, baselines, noise floor (paper §Evaluation)

- **Two scales:** structural/separability tests on the **latent Δs scale**
  (`Δs = r_{B'}ᵀ(v⁺(C')−v0(C'))`); end-to-end number on the **behavior `[0,1]`
  scale**, measured **near mid-range** where the link φ is near-affine (testing
  separability on the bounded scale falsely rejects it).
- **Primary metric:** Spearman ρ (scale-free). Also Pearson r, **sign
  agreement**, **AUROC + top-k precision** for "leakage exceeds threshold",
  and **calibrated MAE** (pp) + slope after a per-behavior affine link fit on the
  training partition only.
- **Factor-localized scoring:** score behavior-transfer (vary B' at fixed C) and
  context-gate (vary C' at fixed B) on their own grid slices, so a failure
  localizes to a factor.
- **Splits:** **leave-one-behavior-out (LOBO)** and **leave-one-context-out
  (LOCO)**; calibrate on train partition only.
- **Baselines (every metric reported against):** predict-zero; predict-mean; raw
  un-whitened cosine gate; the cosine predictor (§Relation-to-cosine); shuffled-key
  / shuffled-query controls (A3.9). **(C7)** the **base-behavior-prior baseline**
  — the target's own base-model expression `E0(C',B')` (≡ `r_{B'}ᵀv0(C')`) — is
  added to **every headline**, not just the level tests. This is the strongest
  recurring null in the project (#532/#541/#649: a unit's base prior keeps beating
  geometry); a geometry "win" is only real if it beats this prior. See §3 Phase 4.
- **Noise floor:** re-estimate every Monte-Carlo expression with independent
  context samples + seeds → test-retest reliability = the ceiling on any
  predictor's achievable ρ. Report headline numbers **against this floor**.

- **(B1) Gate-scale measurement discipline — score the gate on the ACTIVATION
  realized gate, not the marker log-prob.** The gate tests (A3.8/A3.9/A3.10) must
  score the realized gate on the activation scale,
  `ĝ^real_{C,B}(C') = ŵᵀΔv(C') / ŵᵀŵ`, with `ŵ=Δv(C)` and `Δv(C')=v⁺(C')−v0(C')`
  — both `v⁺(C')` tensors are stored (§1.9), so this is direct. **The marker
  log-prob is DEMOTED to a secondary behavior-scale companion**, never the primary
  gate read: scoring the gate off the marker log-prob silently assumes A3.3
  (linear read-out) *and* A3.8 (the marker probe IS the write direction), which is
  exactly what the gate test is supposed to verify — using it as the gate metric is
  circular. The marker log-prob stays useful as a behavior-scale sanity companion
  (does the activation-scale gate track an actual behavioral readout?), reported
  alongside but subordinate to `ĝ^real`.
- **(B1) Separate the two gate verdicts.** Keep distinct, and report distinctly:
  (i) **"a general key–query gate exists"** — the realized update at a target is
  well-approximated by a scalar multiple of the source write (A3.8 rank-one) and
  *some* base-model similarity predicts that scalar (A3.9/A3.10 with any key/metric
  that wins); vs (ii) **"the boxed whitened-`c_C` predictor holds"** — specifically
  the `c_C` key with the `Σc⁻¹` metric (the paper's boxed `g_C(C')`). A PASS on (i)
  with the raw-dot-product or a non-`c_C` key is NOT a PASS on the boxed predictor;
  the metric/key ablations (A3.9) decide (ii). A write-up that PASSes (i) and
  narrates it as (ii) is the overclaim this split prevents.

- **(C3) Multiple-comparison / selection-aware verdict rule.** The program tests
  ~10 assumptions × multiple recipes (layer, summary, key, metric) × ablations;
  best-of-K selection inflates any "PASS." The rule:
  - **Pre-register ONE primary path per assumption** — one layer, one summary
    recipe, one key, one metric — fixed in the **per-phase plan's primary-path
    pre-registration** (the per-phase adversarial-planner plan + approval gate, §10a;
    operationalized by the Phase-1 recipe-freeze, §3 Phase 1) BEFORE the numbers are
    read. **That primary path's number is the verdict.** (Phase 1
    selects the primary layer/summary on a held-out split, then FREEZES it for all
    downstream phases — selection and verdict are on disjoint data.)
  - **All sweeps are EXPLORATORY** — reported as a sensitivity surface, never as the
    headline, and never "the best layer PASSed" without the FDR correction below.
  - **Control FDR across the exploratory family** (Benjamini–Hochberg over the
    {assumption × recipe × layer × ablation} grid) OR pool hierarchically (a
    partial-pooling fit across layers/recipes with the shrinkage reported), so an
    isolated lucky cell cannot be cited as a finding.
- **(C4) Shared-store dependence — one hierarchical validation, clustered CIs.**
  `v0`, `c_C`, and `r_B` are all derived from the SAME 50 contexts × 48 probes, and
  the 48 probes are shared across all 50 contexts, so a gate correlation over n=50
  contexts has far fewer effective degrees of freedom than 50 — the contexts cluster
  by family and the probe pool is common. Therefore:
  - Treat the whole program as **ONE hierarchical validation**, not 10 independent
    PASS/FAIL passes — a downstream test inherits the uncertainty of the upstream
    recipe choice it is conditioned on (state the conditioning in each phase's
    clean-result).
  - **Report family-, source-, and seed-clustered bootstrap CIs** for every
    cross-context correlation (resample at the cluster level — family for the gate
    over contexts, source for cross-source claims, seed for stability) — never a
    naive n=50 CI.
  - **Independent-probe-split replication of the headline gate number:** split the
    48 probes into two disjoint halves, recompute `v0`/`c_C`/`ĝ^real` on each half
    independently, and report the headline gate ρ on BOTH halves (a number that
    only survives on the full pooled probe set is a shared-probe artifact).

### 1.8 Seeds & reproducibility

- ≥3 training seeds per fine-tune cell where a direction-stability or
  noise-floor claim is made; ≥2 eval resamples for the noise floor.
- Every run writes `run_result.json` + WandB; activation store is content-sha
  pinned (§1.9).

### 1.9 Storage — the activation store contract

One versioned store, written once per (model, recipe), reused by every CPU test.

- **Location:** HF data repo `superkaiba1/explore-persona-space-data/theory_assumptions/<model>/...`
  (mirrors the `analysis_tensors/` convention from #521/#551).
- **Per (model θ, condition C):** C ranges over the **50-context #594 battery**
  (`data/issue594/battery.json`, sha-pinned in the manifest); questions = the 48
  Betley probes. Store `v_θ(C)` (all layers), `c_C` (all layers), per-(C,question)
  raw answer activations needed for resampling, on-policy answers + judge labels.
  **Sample R≥8 completions per (C,question) at temp 1.0 and RETAIN per-sample
  activations + judge labels** (not just the per-condition mean) — the
  single-context edge case (§1.10) reads these. Also retain per-(C,question)
  **context vectors `c_x`** (all layers), not just the centroid `c_C` — the
  within-condition coherence check (§1.2) reads these (cheap to re-extract
  prompt-only if a run centroided them). These per-probe granularities are the
  store-granularity changes the edge-case + coherence arms force.
- **(C9) Probe-count fail-fast:** Phase 0 asserts `assert len(probes) == 48`
  (the preregistered Betley pool) in-process before extraction, and writes the probe
  sha to the manifest — a silent probe-count drift (#260-class) corrupts every
  downstream `v0`/`c_C`/rate. The live #658 run uses all 48 (its plan's §4.3 N=200
  shortfall is logged); the assert pins this for every fleet phase.
- **(B6) Per-behavior eval-prompt capture (fleet phases).** For the leakage cells
  whose evaluated behavior `B'` has a non-Betley battery (§1.3 table), the store
  captures `v0(C)`, `v⁺(C')`, and `c_C` on **that behavior's own scoring prompts**,
  not only on the shared Betley pool — keyed by `(condition, behavior-battery)` so
  the activation summary and the behavior rate are computed over the same `x∼C` the
  equations require. The Betley-pool capture remains for the Betley-valid behaviors
  + the context-geometry tests; the per-battery capture is the additional fleet-phase
  granularity B6 forces (Phase 2 store design).
- **(C4) Probe-split + cluster keys retained.** Persist the per-probe granularity so
  the headline gate can be recomputed on disjoint probe halves (§1.7 independent-
  probe-split replication), and record each context's `family` + each cell's
  `source`/`seed` so cluster-bootstrap CIs (§1.7) can resample at the right level.
- **Per behavior B:** `r_B` (all layers, each recipe), `D_B/D_{B̄}` activation
  means, `t_{C,B}` (positive-only `t⁺` and, for the contrastive arm, `t⁻` for the
  objective-consistent `δ^contra=t⁺−t⁻`, B5/§1.5).
- **Global:** `Σ_c` (+ regularized inverse `Σ_c⁻¹`, top-eigendir variant), the
  background corpus context vectors. **Optional add-on** (A3.10 metric-drift, §4
  note): `Σ_c⁺` from one background-corpus pass on a single representative θ⁺ —
  not part of the default store. **(B3) `Σ_c`, `Σ_c⁻¹`, and the whitened gate that
  consumes them are NET-NEW code (not cached-store reuse)** — see §2/§5; the store
  contract here is the same, but the consuming linear algebra is numerically fragile
  and must clear the §5 reduction unit test before any A3.9/A3.10/Phase-4 number is
  trusted.
- **Four-float-per-slot** storage for marker DV (logits unrecoverable post-hoc, #530).
- Manifest JSON: model sha, recipe, layer index, token-position policy, seed,
  code commit, **probe sha + count, per-context family label, background-corpus
  size + sha**. Everything else is derived on CPU from these.

### 1.10 Edge case: single-context conditions (C = δ_x)

A context condition is a *distribution* over contexts; the **single-context limit**
fixes `C = δ_x` — a point mass on one full input `x` (a battery context + one
probe). We test **every assumption in this limit IN ADDITION to the distributional
case**, because it is (a) the **deployment-relevant** case — leakage to a *specific*
prompt (a trigger, a jailbreak, one query), not an average — and (b) the **hardest
stress test**: the averaging over `x∼C` is the main variance-reduction mechanism
behind the low-dim summary (A3.2), the linear read-out (A3.3), and the
context→profile map (A3.5); removing it is the strictest test. If the assumptions
hold per-prompt, they hold a fortiori for distributions.

- **Granularity:** each (battery context × probe) pair is its own `δ_x`
  (50 × 48 = 2400 single contexts). Single-context quantities collapse the outer
  expectation: `v_θ(δ_x)=ā_θ(x)`, `E_θ(δ_x,B)=B(x,A_θ(x))`, `c_{δ_x}=c_x`.
- **Within-context sampling (the store requirement, §1.9):** per `δ_x`, sample
  **R≥8** completions (temp 1.0), judge each → a within-prompt **rate**, and mean
  the R answer activations → `v_θ(δ_x)`. The store must retain per-probe, per-sample
  activations + labels.
- **Noise-floor caveat (load-bearing):** in the single-context limit the
  measurement noise is **maximal** (no cross-context averaging). Estimate a
  **within-context noise floor** from independent R-sample splits and report EVERY
  single-context correlation against it — a low single-context ρ may be pure
  measurement noise, not a model failure; the two MUST be distinguished before any
  "assumption breaks at single-context" claim.
- **(C12) Continuous DV is PRIMARY here, and R is raised for single-context
  claims.** A binary/rate behavior at one prompt quantizes to a low-resolution
  Bernoulli rate (an R≥8 half-sample split gives a very noisy per-prompt rate); the
  continuous completion-probability DV (§1.6 secondary) keeps full dynamic range
  per-prompt, so it is the **primary** read for the single-context analysis. **For
  any single-context claim that rests on the rate DV, raise R** (≥32 where a
  single-context rate is load-bearing) so the Bernoulli noise floor is not itself
  the result; the distributional analyses can stay at R≥8 because cross-context
  averaging supplies the variance reduction. **Report greedy and stochastic
  separately, never pooled:** the greedy single-context read (`v_θ(δ_x)=ā_θ(x)` on
  the one greedy answer, aligned with `B(x,A_θ(x))` per C11) is the theory-faithful
  primary; the temperature-R read is the stochastic robustness companion. Pooling
  them mixes two constructs.
- **Where it runs:** an analysis arm folded into the existing phases (Phase 1 for
  A3.2/A3.3/A3.5; Phase 3 for the gate/leakage A3.8/A3.9/A3.10), **comparing
  single-context vs distributional results**. Almost entirely **CPU re-analysis on
  the same store** — the only added GPU is the R-samples-per-probe generation, which
  the rate DV already needs.

---

## 2. Reuse architecture — compute once, test many

**The whole point:** the GPU cost is (a) one base-model extraction pass, (b) the
fine-tune fleet, (c) one extraction pass per trained model. Every assumption test
below is then **linear algebra on the cached store** (§1.9). The master table maps
each cached quantity to the assumptions that consume it.

### Base-model quantities (θ0 — computed ONCE)

| Quantity | Definition | How computed | GPU? | Consumed by |
|---|---|---|---|---|
| `v0(C)` | mean answer-token act under C | vLLM greedy gen + capture, 50 contexts × 48 probes | yes (gen) | A3.2, A3.3, A3.4, A3.5; `δ`(A3.7); `Δv` baseline (A3.8) |
| `c_C` | prompt-side summary under C | forward pass, prompt only (no gen) | yes (cheap) | A3.4, A3.5, A3.9, A3.10 |
| `r_B` | diff-in-means answer acts D_B−D_{B̄} | forward on behavior datasets | yes (cheap) | A3.3, A3.6; behavior transfer |
| `t_{C,B}` | mean answer act teacher-forcing train completions | teacher-forced forward | yes (cheap) | `δ`(A3.7), predictor |
| `Σ_c`, `Σ_c⁻¹` | E[ccᵀ] over background corpus | forward on corpus + CPU outer-product | yes (cheap) | A3.9, A3.10, predictor |
| `E0(C,B)` | base expression B(x,A_θ0(x)) | gen + Batch judge / marker logp | yes + judge | A3.2/A3.3 ground truth; leakage baseline |

### Trained-model quantities (per θ⁺_{C,B} fine-tune)

| Quantity | Definition | GPU? | Consumed by |
|---|---|---|---|
| `v⁺(C')` | mean answer act under each target C' (full 50-context battery) | yes (gen) | A3.6, A3.7(`ŵ`), A3.8(`Δv`), A3.10 |
| `c⁺_{C'}` | prompt-side under θ⁺ | yes (cheap) | A3.10 gate-drift decomposition |
| `r⁺_{B'}` | re-extracted read-out under θ⁺ | yes (cheap) | A3.6 |
| `E⁺(C',B')` | trained expression (all B' on all C') | yes + judge | ground-truth leakage `L` |

### Derived (CPU only — no GPU, recomputed freely)

`δ_{C,B}=t−v0` · `ŵ_{C,B}=v⁺(C)−v0(C)` · `Δv(C')=v⁺(C')−v0(C')` ·
realized gate `ĝ^real(C')=ŵᵀΔv(C')/ŵᵀŵ` · predicted gate
`g0(C')=z_Cᵀc_{C'}/z_Cᵀc_C` with `z_C=Σ_c⁻¹c_C` · latent leakage
`Δs=r_{B'}ᵀΔv(C')` · all SVDs, residuals, correlations.

> **Reuse claim (scoped):** the *stored tensors* `v0`/`v⁺` over the *same*
> 50-context × 48-probe grid and `c_C` serve the expression tests, source-write,
> rank-one, key-query gate, base-gate validity, joint factorization, AND the
> end-to-end predictor. No assumption needs its own generation pass beyond
> (a)+(b)+(c). **This reuse claim applies to the STORED ACTIVATIONS, not to the
> downstream analysis code.**

> **(B3) The whitened-gate analysis code is NET-NEW — not cached-store reuse.**
> Grep-verified against the repo: the whitened gate `g_C(C')=c_CᵀΣc⁻¹c_{C'}/c_CᵀΣc⁻¹c_C`,
> the `Σ_c=E[ccᵀ]` background-corpus estimator, its regularized inverse
> `(Σ_c+λI)⁻¹` (and top-eigendir variant), and the full predictor
> `L̂=η·(r_{B'}ᵀδ)·g_C(C')` do **not** exist in the codebase — the only Mahalanobis
> code present is a different object (persona-distance, not a context gate). Calling
> these "linear algebra on the cached store" / "predictor-zoo reuse" understates the
> work and risks a downstream planner under-testing the numerically-fragile
> whitening (ill-conditioned `Σ_c`, `λ` sensitivity, top-eigendir truncation, the
> source self-normalization denominator). These modules are **written and
> unit-tested in Phase 3** (§5 lists them as net-new), and **no A3.9/A3.10/Phase-4
> number is trusted until the §5 reduction unit test passes** — the gate must reduce
> to `cos(c_C,c_C')` in the `Σ_c=I` / equal-norm / `δ∥r_B` limit (the paper's stated
> cosine special case). Everything else in the Derived block (`δ`, `ŵ`, `Δv`,
> `ĝ^real`, `Δs`, SVDs) is genuinely cheap CPU algebra on the store.

---

## 3. Phase plan (dependency-ordered)

```
Phase 0  Extraction harness + activation store + behavior datasets   [infra]
   │
Phase 1  Base-only assumptions (NO fine-tuning): A3.2, A3.3, A3.4/5  [cheap GPU]
   │      → fixes the layer + summary recipe used everywhere downstream
   │
Phase 2  Fine-tune fleet + trained-model extraction + ground-truth   [main GPU]
   │      leakage  (feeds ALL training-dependent assumptions)
   │
Phase 3  Training-dependent assumptions (CPU on cached store):       [near-zero GPU]
   │      A3.6 readout-stability · A3.7 source-write · A3.8 rank-one ·
   │      A3.9 key-query gate · A3.10 base-gate validity · joint factorization
   │
Phase 4  End-to-end predictor + cosine ablation + baselines +        [CPU + small GPU]
          noise floor + LOBO/LOCO  (the headline)
```

### Phase 0 — extraction harness + store *(kind: infra)*

One harness that, given a model + the §1 config, produces the entire §2 store. It
wraps the **existing** extraction code (§5) behind a single entry point and writes
the §1.9 manifest. Also builds/validates the behavior datasets `D_B/D_{B̄}` and the
training mixes (on-policy elicitation + contrastive negatives per §1.5). Deliver:
the store for θ0 (primary model) + the background-corpus `Σ_c`.

- **(C9) Fail-fast probe-count assert:** the harness asserts `len(probes) == 48`
  in-process before any extraction and records the probe sha + count in the manifest
  (§1.9). A silent drift corrupts every downstream tensor; fail at second 1, not at
  analysis time.

### Phase 1 — base-only assumptions *(cheap; no training)*

These need only θ0 quantities + base expression scores. They also **select the
extraction recipe** (layer, summary) used by every later phase, so run first.

- **A3.2 (activation-summary sufficiency)** — *"worth testing now: YES, all else
  hinges on it."* Train a small MLP (universal approximator) to predict expression
  `E0(C,B)` from `v0(C)`, per behavior; include the localized marker. Sweep layer.
  PASS if it predicts held-out expression well; report the best layer per behavior.
- **A3.3 (linear read-out)** — fit `r_B` (diff-in-means etc.), test
  `E0(C,B) ≈ r_Bᵀ v0(C)` on held-out C. Compare recipes + layers; this is the
  layer where each behavior reads out best.
- **A3.5 (context summary = residual vector `c_C`)** — train linear `M` and an MLP
  mapping `c_C → v0(C)`; report linear-vs-nonlinear gap and best `c_C` recipe/layer.
- **(B2) A3.4 distinct test (recipe-agnostic sufficiency) — [#658-LANDING].** A3.4
  (some pre-FT context quantity predicts `v0` — *sufficiency*) is currently folded
  into A3.5 (which tests one *instantiation*, the residual `c_C`). A negative A3.5
  would be mis-blamed on A3.4. Test A3.4 on its own: fit `c→v0` maps across **ALL
  context-summary recipe candidates** (last-input-token, mean-over-prompt, multi-
  layer pool, learned embedding) and report the **best achievable** `c→v0` over the
  whole recipe family, plus a **full-prompt-activation upper-bound control** (predict
  `v0` from the *entire* prompt-side activation tensor — the richest possible
  pre-FT context summary, an upper bound on what any cheap `c_C` could reach).
  **Report the gap** between that upper bound and the cheap `c_C` of A3.5: a large
  gap means A3.4 (sufficiency) holds but A3.5's cheap summary is the weak link
  (recipe problem), not A3.4. This is a **cheap CPU re-analysis on #658's store**
  (all recipe candidates + per-(C,probe) activations already captured) — folded into
  the #658 landing bundle, does NOT interrupt the live run.
- **(B4) A3.1 richer-summary arm — [#658-LANDING].** A3.1 (expression depends on the
  profile only through a *low-dimensional* summary) is marked "not worth testing now"
  in the paper, but the program is NAMED "test all the assumptions" — so test a
  bounded version: predict `E0(C,B)` from the **mean answer activation** (A3.2's
  summary) vs **richer summaries** — token-pooled (multiple answer-token positions),
  multi-layer pooled, and answer-distribution features (e.g. per-position entropy /
  top-token mass) — on **nested held-out C/B**. If the richer summaries beat the mean
  materially, the mean is leaving signal on the table (A3.1 bites); if not, the mean
  is a sufficient low-dim summary. This is a **cheap CPU re-analysis on #658's
  all-layer store** (the per-(C,probe) per-layer activations are captured) — folded
  into the #658 landing bundle. (It is a bounded probe, not the full deferred
  recursive-pooling test; the clean-result says so.)
- **(C3) Recipe-freeze for the verdict.** Phase 1 SELECTS the primary layer +
  summary recipe per behavior on a held-out split, then **freezes it** as the
  per-phase plan's pre-registered primary path (§10a per-phase plan approval; §1.7 C3)
  for every downstream phase. The frozen path's number
  is the verdict (§1.7 C3); the per-layer/per-recipe sweep is the exploratory
  sensitivity surface, FDR-corrected, never the headline. Selection (here) and the
  downstream gate verdict (Phase 3) are thus on disjoint data.
- **Single-context arm (§1.10):** repeat A3.2/A3.3/A3.5 at `C=δ_x` granularity
  (each context×probe), **continuous DV primary**, reported vs the within-context
  noise floor; compare per-prompt vs distributional. Pure CPU re-analysis on the
  store (no new GPU beyond the R-samples-per-probe already captured in Phase 0).
- **Within-condition coherence test (`a:context-vector-coherence`, theory §2 precondition;
  formal assumption added 2026-06-25).** Once a per-context map `x↦c_x` is fixed, each condition
  `C` to which the mean-vector predictor is applied must be **coherent** — its contexts form a
  tight cluster in context-vector space. **Why it is FORCED, not optional:** the singleton edge
  case (§1.10) sets `C=δ_x` ⇒ `c_{δ_x}=c_x`, `v0(δ_x)=ā_θ0(x)`, so A3.5 on singletons demands
  `ā_θ0(x)≈h_θ0(c_x)` (predict *individual* profiles). For a finite mixture `C=Σ p_i δ_{x_i}` both
  `c_C=Σ p_i c_i` and `v0(C)=Σ p_i a_i` are averages, so exact singleton **and** mixture prediction
  together force `h_θ0(Σ p_i c_i)=Σ p_i h_θ0(c_i)` for arbitrary mixtures — `h_θ0` must commute with
  convex averaging. Under twice-differentiability that forces the Hessian to vanish everywhere
  (`h_θ0` affine). So a genuinely **nonlinear** `h_θ0` is mathematically incompatible with
  single-vector conditions over arbitrary mixtures; **coherence is the escape** — apply the
  mean-vector predictor only to conditions occupying a small local region where a nonlinear `h_θ0`
  is approximately affine.
  - **The bound it controls.** With `c_C=E_{x∼C}[c_x]`, within-condition spread
    `s_W(C)=E_{x∼C}‖c_x−c_C‖²_W`, and the behavior-relevant map `f_B(c)=r_Bᵀh_θ0(c)` of bounded
    `W`-curvature `|uᵀ∇²f_B u|≤K_{C,B}‖u‖²_W`, the first-order term averages out (`E[Δ_x]=0`),
    leaving a second-order Jensen gap `|E_{x∼C}[f_B(c_x)]−f_B(c_C)| ≤ ½K_{C,B}·s_W(C)`. With the
    singleton residual `ξ_B(C)=E_{x∼C}|r_Bᵀ(ā_θ0(x)−h_θ0(c_x))|`, the full condition-level error is
    `|r_Bᵀ(v0(C)−h_θ0(c_C))| ≤ ξ_B(C)+½K_{C,B}s_W(C)`; to keep latent error under a budget `τ_lat`
    over a behavior family `ℬ` it suffices that `s_W(C) ≤ 2(τ_lat−ξ_ℬ(C))/K_C^ℬ`.
  - **Measurables (per C; all CPU on the stored per-probe `c_x` + `v0`):** (a) spread
    `ŝ_W(C)=(1/n)Σ‖c_{x_i}−ĉ_C‖²_W`; (b) behavior-relevant **Jensen gap**
    `Ĵ_ℬ(C)=max_B|r_Bᵀ((1/n)Σ h_θ0(c_{x_i}) − h_θ0(ĉ_C))|`; (c) context→profile **residual**
    `R̂_ℬ(C)=max_B|r_Bᵀ((1/n)Σ ā_θ0(x_i) − h_θ0(ĉ_C))|`. **Metric `W`:** default `W=I`; once `Σ_c`
    is estimated, the whitened `W=(Σ_c+λI)⁻¹` (matches the context-gate geometry). Keep the cheap
    descriptive reads too (within- vs between-condition cosine, η²/silhouette) as a coarse companion.
  - **Pass / read:** the assumption predicts low-spread conditions have small Jensen gap AND low
    residual — `Ĵ_ℬ(C)` and `R̂_ℬ(C)` rise with `ŝ_W(C)`, the slope estimating the local curvature
    `½K`. Test per condition AND **per family** (persona/format/behavior expected coherent; wildchat
    expected scattered), swept over layers. **On failure (high spread → large gap):** the condition
    is too heterogeneous for one mean vector — SPLIT it into smaller coherent conditions, or replace
    `c_C` with a richer distributional summary (higher moments / a learned set encoder); meanwhile
    down-weight/caveat that condition's downstream gate predictions. Complements #617 (between-family
    separability). Confidence: medium-high. Zero new GPU (re-extract prompt-only `c_x` if a run
    centroided them). Full derivation (the affinity proof + the Taylor bound) lives in the theory doc
    under `a:context-vector-coherence`.

Output: locked layer/summary recipe per behavior (frozen per C3) + a go/no-go on the
linear chain. A3.1 is now tested as the bounded B4 richer-summary arm above (not
fully deferred); the full recursive-pooling test stays deferred unless A3.2 + B4
both signal the mean is insufficient.

### Phase 2 — fine-tune fleet + trained extraction + ground truth *(main GPU)*

Train θ⁺ on the **recommended starting grid** (§7.2), then run the trained-model
extraction (§2) + Batch-judge the eval registry. **(C8) Judge budget uses the
19-column registry, not 11** (§1.3, §6): full top-level columns + the applicable
family-expression / diagonal columns on the primary context, the
`ROBUSTNESS_COLUMNS` subset on extra context families (§7.5). This single fleet
feeds all of Phase 3 and Phase 4.

- **(B5) Contrastive + positive-only fleet arms — [PHASE-2+, Thomas-approved].**
  Each source on the gate/transfer spines is trained in **two arms**: the
  project-default **contrastive** arm (the realistic rig) and a **positive-only**
  arm (the theory's clean `δ=t−v0` identification, §1.5). A3.7/A3.8/A3.9 are reported
  on both; the positive-only arm is the PRIMARY A3.7 identification test, the
  contrastive arm carries `δ^contra=t⁺−t⁻`. This is the main GPU-budget addition this
  round (≈ doubles the spine fine-tunes; §6 re-costed).
- **(B6) Per-behavior eval-prompt capture — [PHASE-2+].** For each leakage cell,
  capture `v0(C)`/`v⁺(C')`/`c_C` on the evaluated behavior's OWN scoring battery
  (§1.3 table, §1.9), so the activation summary and the rate are on the same `x∼C`.
  Behaviors with non-Betley batteries are scored on their batteries, not forced onto
  the Betley pool.
- **Context-leakage spine (gate tests):** train **marker** (B7) into each of 4
  sources {librarian, surgeon, programmer, assistant}; measure marker on all 50
  battery C' (all 7 families — the near→far context-distance range). **(B1)** the
  primary gate read is the **activation realized gate `ĝ^real`** on the stored
  `v⁺(C')` tensors; the marker log-prob is the SECONDARY behavior-scale companion
  (not the gate metric). *Reuse #474 epoch-1 non-saturated adapters where the recipe
  matches (artifact-reuse checklist).*
- **Behavior-leakage spine (transfer tests):** train a representative B-family
  subset (B1 bad-medical anchor, B2 insecure-code, B3 sycophancy, B4 refusal, B5
  taught-fact, B7 marker) into a fixed source (assistant + 1 persona); score the
  full eval registry on the source (A3.6/A3.7 + `r_{B'}ᵀδ`). B2→broad_em is the
  canonical cross-behavior case (insecure-code write read by misalignment).
- **Generalized cells** (C',B' both varying) come for free from these models.
- **Strength arms:** ≥2 doses per cell (non-saturated + a stronger one) so `η` and
  the saturation behavior of the link φ are observed.

### Phase 3 — training-dependent assumptions *(CPU on the store)*

All of these are linear algebra on Phase-2 tensors. No new GPU (the whitened-gate
modules are net-new code per B3/§5, written here, but run on CPU).

- **A3.6 readout-stability** · **A3.7 source-write** · **A3.8 rank-one gated
  write** · **A3.9 key-query gate** (key/metric ablations) · **A3.10 base-gate
  validity** (g0 vs ĝ^real, oracle g⁺, drift decomposition) · **joint
  factorization** (rank-one of the latent leakage matrix S). Detailed in §4.
- **(B1) All gate tests score the ACTIVATION realized gate `ĝ^real=ŵᵀΔv(C')/ŵᵀŵ`**
  as the primary metric (the `v⁺(C')` tensors are stored); the marker log-prob is a
  secondary behavior-scale companion only, never the gate read (scoring the gate off
  the log-prob assumes A3.3+A3.8 — circular).
- **(B1/B3) Two separate verdicts, reported separately:** (i) "a general key–query
  gate exists" (rank-one A3.8 + *some* base-model similarity predicts ĝ^real with any
  winning key/metric) vs (ii) "the boxed whitened-`c_C` predictor holds"
  (specifically `c_C` key + `Σc⁻¹` metric). The A3.9 key/metric ablation decides
  (ii); a PASS on (i) is never narrated as (ii). **(B3)** the whitened-gate code must
  clear the §5 reduction unit test (gate → `cos(c_C,c_C')` in the `Σc=I`/equal-norm
  limit) before any A3.9/A3.10 number is reported.
- **(C4)** every cross-context gate correlation is reported with family-/source-/
  seed-clustered bootstrap CIs and an independent-probe-split replication of the
  headline (§1.7), not a naive n=50 CI.
- **Single-context arm (§1.10):** A3.8/A3.9/A3.10 with single-context source
  and/or target — does the geometry predict *per-prompt* leakage `δ_x → δ_{x'}`
  (the deployment case)? Reported vs the within-context noise floor; CPU on the
  store.

### Phase 4 — end-to-end + cosine ablation + baselines *(CPU + small GPU)*

Assemble `L̂`, calibrate φ, evaluate under LOBO/LOCO against all §1.7 baselines and
the cosine predictor, on both scales, against the noise floor. Recover `η` from one
on-source measurement for the cross-source/absolute number. This is the paper's
headline table.

- **(C7) Apples-to-apples cosine baseline.** The cosine predictor
  `L̂^cos ∝ cos(r_{B'},r_B)·cos(c_C,c_C')` MUST share the EXACT same `r_B`, `c_C`,
  layer, and extraction recipe as the full `L̂` — the only differences allowed are
  the three the paper identifies (δ-vs-`r_B` for the behavior term; norm handling;
  `Σc⁻¹`-vs-`I` for the gate). A cosine baseline computed on a different
  layer/recipe would make the whitening "win" an artifact of the recipe gap, not the
  whitening. The harness derives both predictors from one frozen recipe and toggles
  only those three knobs (this isolates exactly what whitening + δ + norms buy).
- **(C7) Base-behavior-prior baseline in the headline.** The target's own base prior
  `E0(C',B')` (= `r_{B'}ᵀv0(C')`) is reported alongside the cosine + full predictors
  in the headline table — it is the strongest recurring null (#532/#541/#649). A
  geometry win is credited only if it beats this prior (partial the prior out, or
  report ΔS predictive power above the prior).

---

## 4. Per-assumption test spec

Body numbering A3.1–A3.10 (granular testable unit); TLDR labels A1–A7 cross-ref'd.
Paper's own "worth testing now" verdict noted.

| # | Assumption (paper) | TLDR | Conf. | Now? | Test (all on cached store) | PASS criterion | Phase |
|---|---|---|---|---|---|---|---|
| **A3.1** | Expression depends on profile only through a low-dim summary | A1 | High | **Bounded (B4)** | mean act vs richer summaries (token-pooled, multi-layer, answer-dist features) on nested held-out C/B; full recursive-pooling test still deferred | richer summaries do NOT materially beat the mean (mean is a sufficient low-dim summary) | 1 (#658-landing) |
| **A3.2** | Summary = mean answer-side activation `v_θ(C)` | A1 | High | **Yes** | MLP: `v0(C)→E0(C,B)`, per behavior incl. marker; layer sweep | predicts held-out expression ≫ mean baseline; report best layer | 1 |
| **A3.3** | Linear read-out `E≈r_Bᵀv` | A2 | High | **Yes** | fit `r_B` as a **built-in recipe sweep**: contrast-construction A/B/C (on-policy-instruction / teacher-forced / instruct-and-strip) × summary variants (diff-in-means / mean-D_B / few-shot), **(C3) primary = C (instruct-and-strip)**, rest exploratory (FDR); test held-out C per layer; report `cos(r_B^{A,B,C})` divergence + the `(c_pos−c_neg)` confound projection + per-recipe predictive quality. **Plus the finer `rb-recipe-knobs`** (pair-count / neg-pole / assistant-baseline / optimism) as IN-PLAN one-knob-off robustness variants (C3-exploratory, FDR; #661 runs the same knobs standalone). | linear ρ within MLP noise floor; recipe divergence + whether it changes the *verdict* vs only geometry; C-primary recipe locked | 1 |
| **A3.4** | Pre-FT context summary predicts profile (sufficiency) | A3 | Med | **Yes (B2)** | **distinct from A3.5:** best-achievable `c→v0` over ALL recipe candidates + full-prompt-activation upper-bound control; report the gap to the cheap `c_C` | sufficiency holds (upper-bound `c→v0` strong); A3.5 gap localizes recipe vs sufficiency failure | 1 (#658-landing) |
| **A3.5** | Context summary = residual vector `c_C` | A3 | Med | **Yes** | linear `M` + MLP: `c_C→v0(C)`; best c_C recipe/layer | nonlinear gain modest; `r_Bᵀ M c_C` predicts E | 1 |
| **A3.5a** | Contexts close within a condition (`a:context-vector-coherence`) — precondition for the single-vector `c_C` summary | A3 | Med-High | **Yes** | per C: spread `s_W(C)=E‖c_x−c_C‖²_W` vs behavior Jensen gap `Ĵ_ℬ=max_B\|r_Bᵀ(E h(c_x)−h(c_C))\|` + residual `R̂_ℬ`; `W=I` then `(Σc+λI)⁻¹`; per condition + family, layer-swept (full spec + derivation: §4 coherence block) | gap & residual rise with spread (slope ≈ ½ local curvature `K`); coherent conditions (persona/format/behavior) small-gap, scattered (wildchat) large → split / richer-summarize the failures | 1 |
| **A3.6** | Base read-out valid post-FT (`r⁺≈r`) | A4 | Med-High | **Yes** | **(C10) primary:** corr / calibrated error between `r_{B'}ᵀ(v⁺−v0)` and `(E⁺−E0)` with `E0` PARTIALLED OUT (does base `r_{B'}` predict the *change*, not the level); `cos(r⁺,r)` DIAGNOSTIC only | base `r_{B'}` predicts the leakage change above the base prior; cos high (diagnostic) | 3 |
| **A3.7** | FT displaces source profile toward data target | A5 | Med | **Yes (B5)** | **primary (positive-only arm):** `cos(ŵ,δ)`, δ=t−v0, **against a shuffled-δ null** (`cos(ŵ, δ` of a different behavior`)`); **contrastive (diagnostic) arm:** `cos(ŵ,δ^contra)`, δ^contra=t⁺−t⁻ (a contrastive heuristic / diff-in-means analogue, NOT theory-derived; report `frac_ctx` per §1.5); scalar-fit residual both; also `cos(ŵ,r_B)` shortcut | positive-only cos **exceeds the shuffled-δ baseline** (not merely >0; ŵ and δ are both displacements from v0(C), so positive cos is partly expected by construction), small residual; report ŵ∥r_B; contrastive-vs-positive-only divergence characterizes recipe-dependence **only after `frac_ctx` is partialled out** | 3 |
| **A3.8** | Off-source change = scalar-gated source write (rank-one) | A6 | Med | **Yes (central)** | **(B1)** on the ACTIVATION gate `ĝ^real=ŵᵀΔv(C')/ŵᵀŵ`: per-target rank-one residual `‖Δv(C')−ŵĝ^real‖/‖Δv(C')‖`; stack ΔV, report σ₁²/Σσ², σ₂/σ₁, cos(u₁,ŵ); low-rank fallback if fails | small residuals; ΔV near rank-one | 3 |
| **A3.9** | Gate = normalized key–query similarity | A7 | Med | **Yes** | `g0(C')` vs `ĝ^real(C')` (B1 activation gate): Pearson/Spearman/sign/MAE; **key ablation** {c_C, ψ(t), ψ(δ), c_C+ψ(δ)}; **metric ablation** {I, diag, full whitening}; vs `cos(c_C,c_{C'})`; shuffled controls + denominator stability; **(B3)** clears the cosine-limit unit test first | **verdict (i)** some key/metric beats raw cosine (general gate exists) AND **verdict (ii)** `c_C`+`Σc⁻¹` specifically wins (boxed predictor) — reported separately | 3 |
| **A3.10** | Base-model gate predicts realized gate | A7 | Med | **Yes (key)** | **at fixed base metric `M⁰` (default):** `g0` vs `ĝ^real` (B1 activation gate) across held-out C'; **oracle** `g⁺` diagnostic (`k⁺,q⁺,M⁰`); **key+query drift decomposition** (metric drift unattributed unless the opt-in `Σ_c⁺` pass runs); residual ≈ A_drift·(c⁺−c0); **(C4)** clustered CIs + probe-split replication | g0 predicts ĝ^real ≈ as well as g⁺ (no fatal drift) **at fixed `M⁰`**; full metric-drift decomposition only if `Σ_c⁺` runs | 3 |
| **—** | Joint factorization diagnostic | — | — | Yes | latent `S_{ij}=r_{B'_j}ᵀΔv(C'_i)`; report σ₁²/Σσ², rank-one residual; verify `L̂_{C',B'}=L̂_{C',B}·L̂_{C,B'}/L̂_{C,B}` (no interaction) | S ≈ rank one; factorization holds on latent scale | 3 |
| **—** | End-to-end predictor + cosine special-case | — | — | Yes | full `L̂`, LOBO/LOCO, vs baselines + cosine, both scales, vs noise floor | beats cosine + baselines; near noise floor | 4 |

**Notes on §4:**

- **(B1) Gate scale — all gate metrics are the activation realized gate.**
  Throughout A3.8/A3.9/A3.10, `ĝ^real=ŵᵀΔv(C')/ŵᵀŵ` (on the stored `v⁺(C')`
  tensors) is the primary, theory-faithful gate metric. The marker log-prob is a
  SECONDARY behavior-scale companion only; using it as the gate metric assumes A3.3
  + A3.8 (the very claims under test) — circular. Report both, primary = activation.
- **(B3) Whitened-gate code is net-new and gated on a reduction unit test.** Before
  any A3.9/A3.10/Phase-4 number is reported, the `g_C`/`Σ_c`/`Σ_c⁻¹` implementation
  must pass a unit test asserting the gate reduces to `cos(c_C,c_C')` in the
  `Σ_c=I` / equal-norm / `δ∥r_B` limit (the paper's stated cosine special case,
  §Relation-to-cosine). This catches a mis-implemented whitening before it
  manufactures a spurious "whitening wins." See §5 (net-new modules).
- **(C10) A3.6 measures the CHANGE, not the level.** The primary A3.6 read is the
  correlation / calibrated error between the predicted change `r_{B'}ᵀ(v⁺−v0)` and
  the measured change `(E⁺−E0)`, with the base level `E0` PARTIALLED OUT — "base
  `r_{B'}` predicts `E⁺`" would PASS trivially by the base prior (the level is
  mostly the prior, #532/#649). `cos(r⁺,r)` is a diagnostic of read-out stability,
  not the A3.6 verdict.
- **`ψ` (the key-space embedding map, A3.9 key ablation `{c_C, ψ(t), ψ(δ),
  c_C+ψ(δ)}`).** Default: **`ψ = identity` with co-layer extraction** — extract
  `c_C`, `t_{C,B}`, `δ_{C,B}` at the **same layer** so they live in a common
  `ℝ^d` and no mapping is needed. Ablation: when `c_C` is the prompt-side slot and
  `t/δ` are answer-side at a different layer, set `ψ =` the frozen context-to-answer
  map `M` from A3.5 (used to bring answer-side vectors into the key space). The
  **headline key is `c_C`** (the "dropping the write strength" simplification,
  paper §"Dropping the write strength"), which needs no `ψ`; the `ψ(t)/ψ(δ)` arms
  are diagnostic only. **The `δ` inside `ψ(δ)` is the theory's positive-only
  `δ=t−v0` (the displacement the key candidate `ψ(δ_{C,B})` is defined on, paper key
  ablation).** On the contrastive arm the data-side key becomes `ψ(δ^contra)`; since
  `δ^contra` is a contrastive heuristic / diff-in-means analogue, not theory-derived
  (§1.5), **scope `ψ(δ^contra)` to the contrastive DIAGNOSTIC arm only** — it is never
  the headline key (the headline key is `c_C`) and a `ψ(δ^contra)` result is read
  alongside its `frac_ctx`, not as a theory-faithful key.
- **A3.10 metric scoping (the `Σ_c⁺` / oracle-`g⁺` capture gap).** The drift
  decomposition needs post-FT gates `g(k⁺,q⁰,M⁰)`, `g(k⁰,q⁺,M⁰)`, `g(k⁰,q⁰,M⁺)`
  and the oracle `g⁺=(k⁺,q⁺,M⁺)`. The store captures `c⁺_{C'}` (→ post-FT query
  `q⁺` and, for key `c_C`, source-key `k⁺`) but **not `M⁺=Σ_c⁺`**, which would
  require re-running each θ⁺ over the ≥2–5k background corpus. **Default decision:
  hold the metric at base `M⁰` for both `g0` and the oracle** (oracle `g⁺` uses
  `k⁺,q⁺,M⁰`) — so the default reads source-key-drift + query-drift, and the PASS
  criterion is "base key+query gate predicts `ĝ^real` ≈ as well as the post-FT
  key+query gate at fixed `M⁰`." **(R3-3) ALL default A3.10 verdicts are labelled "at
  fixed base metric `M⁰`."** The theory's oracle and full drift decomposition include
  the metric-drift term `g(k⁰,q⁰,M⁺)` with the post-FT metric `M⁺` (paper
  a:base-gate-validity oracle `g⁺=(k⁺,q⁺,M⁺)` + the three-way drift split); the
  default omits `M⁺`. **Do NOT claim a full metric-drift decomposition** (i.e. do not
  attribute residual gate drift across source-key / target-query / metric components)
  **unless the optional `Σ_c⁺` pass below actually runs** — the default reports
  source-key-drift + target-query-drift only, with metric drift folded into an
  unattributed residual, and the clean-result says so. **Metric-drift is an opt-in
  add-on:** one background-corpus pass on a *single* representative θ⁺ to estimate
  `Σ_c⁺`, run only if the `M⁰`-scoped residual is large and unexplained by key/query
  drift — only then is the full (key/query/metric) decomposition and the true oracle
  `g⁺=(k⁺,q⁺,M⁺)` reportable. This keeps the store cheap; the add-on pass is budgeted
  in §6 as optional.
- **Single-context variant (§1.10).** Every assumption gets a `C=δ_x` arm in
  addition to the distributional one — A3.2/A3.3/A3.5 in Phase 1, A3.8/A3.9/A3.10
  in Phase 3 — with the **continuous DV primary** and results reported against the
  **within-context noise floor** (a low single-context ρ that is within the floor is
  measurement noise, NOT a falsified assumption).
- **Within-condition coherence (theory §2 precondition).** Before trusting `c_C` as
  a condition summary, Phase 1 verifies the per-probe `{c_x}` cluster within each
  condition (within-vs-between cosine + variance-explained, per condition + family).
  Low-coherence conditions get their gate predictions down-weighted/caveated — this
  validates the precondition behind A3.4/A3.5 and the whole context side.

---

## 5. Reused artifacts & code (existing infra map)

Don't rebuild — the harness (Phase 0) wraps these.

**Context suite (the §1.2 library) — the load-bearing reuse:**
- `scripts/issue594_build_battery.py` — builds the 50-context / 7-family battery → `data/issue594/battery.json`
- `scripts/issue594_common.py` — `load_battery()`, family counts, schema validation
- `scripts/issue594_extract_context_vectors.py` — extracts `c_C` over the battery, **all 28 layers**, last-input-token slot (drop-in for Phase 0)
- `scripts/issue594_analyze_context_geometry.py` — context-geometry analysis (family separability)
- `scripts/issue617_*.py` — WildChat real-conversation cluster contexts (extends the `wildchat` family)
- probe pool: `issue404_common.fetch_preregistered_probes()` (48 Betley probes = `x∼C`)

**Extraction / vectors:**
- **`r_B` (pos/neg system-prompt-pair diff, answer-side) — Phase-0 to-build; extraction method (A/B/C) decided by the running #661 (→ program-#660 A3.3 recipe).** The scripts below emit raw centroids / a different contrast and do NOT yet produce the default `r_B`:
  - `scripts/extract_persona_vectors.py` — per-role ABSOLUTE centroids (no pos/neg contrast in-script); mean-response-token (answer-side) and last-input-token (`c_C`-style / §1.4 ablation) slots
  - `scripts/issue623_persona_panel_vectors.py` — thin wrapper emitting resolved-panel centroids (raw only; diff computed off-pod in `issue623_analyze.py`)
  - `scripts/issue623_analyze.py::persona_vector_matrix` — assembles a **role-minus-assistant** persona-axis diff (mean-centered #536) — a DIFFERENT contrast, not the pos/neg `r_B`
  - `scripts/issue634_extract_behavior_vectors.py` — last-input-token, **positive-only** absolute centroids (the `c_C`-style slot = §1.4 `r_B` *ablation*, NO contrast)
- `scripts/issue650_extract_context_bank.py` + `experiments/issue_650/shift_extract.py` — context bank `c_C` + activation shift `Δv`
- `scripts/issue541_geometry_extract.py`, `scripts/extract_prompt_divergence_activations.py` — geometry/context extraction
- `scripts/issue493_extraction_metric_bakeoff.py` — **layer/summary recipe selection** (directly Phase 1)
- `src/.../analysis/representation_shift.py`, `divergence.py`, `js_canonical.py`, `probes.py` — shift/divergence/probe analysis

**Predictor bakeoff (REUSE — evaluation pattern only):**
- `experiments/behavior_testbed_545/predictors.py` + `predictors_zoo.py` — the predictor zoo (cosine/JS/KL/Gaussian-KL). **NOTE:** the zoo does NOT contain the theory's whitened context gate — see net-new below.
- `scripts/issue532_predictor_stress.py`, `scripts/issue545_metric_race.py` — predictor stress/race harnesses (the LOBO/LOCO evaluation pattern)

**(B3) NET-NEW code (does NOT exist in the repo — grep-verified; written in Phase 3,
NOT cached-store reuse):**
- **Whitened context gate** `g_C(C')=c_CᵀΣc⁻¹c_{C'}/c_CᵀΣc⁻¹c_C` — the boxed
  predictor's context factor. The only Mahalanobis-shaped code in the repo is a
  different object (persona-distance, not a context gate).
- **`Σ_c=E[ccᵀ]` background-corpus estimator** + **regularized inverse**
  `(Σ_c+λI)⁻¹` + top-eigendir-truncated variant, with λ-sweep and conditioning
  diagnostics (numerically fragile — the review's specific concern).
- **Full leakage predictor** `L̂=η·(r_{B'}ᵀδ)·g_C(C')` assembly + the φ link
  calibration + the η on-source recovery.
- **REQUIRED unit test (gates every A3.9/A3.10/Phase-4 number):** assert the
  whitened gate reduces to `cos(c_C,c_C')` in the `Σ_c=I` / equal-norm / `δ∥r_B`
  limit (the paper's §Relation-to-cosine special case). No whitening number is
  trusted until this passes — a mis-implemented inverse otherwise manufactures a
  spurious "whitening wins."
- **(C7) Apples-to-apples cosine baseline** derived from the SAME `r_B`/`c_C`/layer/
  recipe as `L̂` (toggling only δ-vs-`r_B` / norm / `Σc⁻¹`-vs-`I`) — also net-new
  wiring, not the zoo's standalone cosine.

**Eval suite:** `experiments/factor_screen_365/{persona_panel.py, eval_panel.py}`,
`eval/{alignment.py, refusal.py, marker_logprob.py, capability.py}`,
`configs/eval/default.yaml`.

**Reusable trained artifacts (apply the artifact-reuse checklist before reuse):**
- **#474 epoch-1 non-saturated marker adapters** (16 contexts) → marker gate spine (used by #532/#539/#540/#549).
- #608 frozen contrastive adapters; #545 19-behavior testbed (B→B′); #521/#551 persisted activation-shift tensors.

**Methodology docs** (findings-blind references to mirror): `docs/methodology/`
has ~60 issue docs incl. #521 (rank-one across contexts), #532 (predictor stress),
#541 (geometry re-analysis), #545 (behavior testbed), #601 (contrastive dose),
#604 (write-direction seed-stability), #612 (on-policy), #650 (shift extract).

**Prior findings that pre-load the priors (from RESULTS / open_questions):**
- Base-model behavior prior beats geometry for absolute **level**; geometry
  predicts the **shift** (#532/#541/#649). → frame A3.2/A3.3 (level) vs
  A3.7–A3.10 (shift) accordingly; **always include the base-prior baseline.**
- Write direction is seed-stable; top-1 key is seed-arbitrary (#604) → A3.8 SVD.
- Rank-1 leak-transfer asymmetry generalizes out-of-sample for marker + taught
  fact, **fails for content behaviors** (#637) → expect A3.8 to be behavior-
  dependent; report per-behavior, don't aggregate over the failure.
- Saturation hides the gate (#448) → §1.5 non-saturated anchor is mandatory.

---

## 6. Cost estimate (primary model, recommended grid)

| Phase | Work | GPU-h (rough) |
|---|---|---|
| 0 | base extraction (gen 50 contexts × 48 probes + c_C + r_B + t + Σ_c corpus) + dataset build | ~3–6 |
| 1 | base-only assumptions (CPU + reuse Phase-0 store; small MLP train on CPU/1 GPU) + B2/B4 #658-landing re-analyses (CPU, ~0) | ~1–2 |
| 2 | **(B5)** ~16–24 LoRA fine-tunes (contrastive + positive-only arms, ≥2 doses) + trained extraction (~1–2 GPU-h each) + **(B6)** per-behavior-battery extraction | ~55–95 |
| 3 | CPU on store (incl. net-new whitened-gate code, CPU) | ~0 |
| 4 | end-to-end (CPU) + any confirmatory regen | ~2–5 |
| **Total (7B)** | | **~60–110 GPU-h** |

- **(C8) Judge budget re-derived for 19 columns, not 11.** The original figure rested
  on the 11 top-level columns; the registry has **19** scorable batteries (§1.3:
  +5 family-expression, +2 diagonal, +`broad_em_n100`). The judge load on the primary
  context per trained model is therefore **~1.7×** the 11-column figure (the
  family-expression + diagonal columns each add a judged battery on the applicable
  adapter rows + base panel). Re-cost the Batch-API judge calls against 19 (with
  `scoring_universe()` / `applies_to()` deciding which cells actually fire per
  adapter), not 11, before committing the fleet. Judging is off-GPU (Batch API), so
  this is dollar/throughput, not GPU-h — but it is the binding constraint at fleet
  scale (§7.5), so the corrected count matters for the run plan.
- **(B5) The positive-only fleet arm roughly doubles the spine fine-tunes** (every
  gate/transfer source trained twice — contrastive + positive-only), driving Phase-2
  GPU-h up; reflected in the table. This is the science-affecting Phase-2 addition
  Thomas approved (clean A3.7 identification).
- **7B-only this run** (scale checks deferred). **Optional A3.10 metric-drift
  add-on:** +1 background-corpus pass on one θ⁺ (~1–2 GPU-h), run only if the
  `M⁰`-scoped gate residual is large (§4 note). Phaseable: Phase 0+1 alone (~5–8
  GPU-h) already settles the foundational assumptions before any fine-tune budget is
  committed.

---

## 7. Open design decisions (recommendations)

### 7.1 Contrastive negatives vs positive-only — **(B5) BOTH are first-class arms**
The project rule mandates contrastive negatives, and the project's own finding is
that the distance→leakage **gradient exists ONLY inside the contrastive regime**
(positive-only leaks uniformly, #207/#383). The gate `g_C(C')` therefore has
dynamic range to predict only under contrastive training. **But** the contrastive
recipe breaks the theory's `δ=t−v0` fidelity: the loss also pushes the write away
from the negatives, so the positive-only `δ` does not represent the realized
objective, and A3.7 would PASS/FAIL for the wrong reason (B5). **Revised decision
(was "positive-only control"; Thomas-approved upgrade): two first-class fleet
arms.**
- **Positive-only arm = the CLEAN A3.7 identification test (PRIMARY).** Trained on
  positives alone, its `δ=t−v0` is exactly the theory's displacement — this is the
  arm where `cos(ŵ,δ)` is the theory-faithful read. (Positive-only leaks uniformly,
  so it is *not* the arm where the gate has dynamic range — that is its expected
  degenerate `g≈const` behavior, which is itself informative.)
- **Contrastive arm = the project-realistic gate test.** A3.8–A3.10's graded gate
  needs the contrastive regime. Here `δ` is replaced by the **objective-consistent**
  `δ^contra=t⁺−t⁻` (§1.5), and A3.7 is reported on `δ^contra`.
- **Report A3.7/A3.8/A3.9 on BOTH arms.** A divergence between the positive-only
  (theory-faithful) read and the contrastive (project-realistic) read is the
  recipe-dependence characterization — it directly answers whether the gate read off
  the project rig carries a negative-set artifact (relevant to A3.10's
  base-predictability claim).

### 7.2 Recommended starting grid (expandable)
4 sources × marker (context-leakage spine) + ~6 B-family adapters × 2 sources
(behavior-leakage spine), ≥2 doses, **× 2 objective arms (contrastive +
positive-only, B5)**, primary model. ≈16–28 fine-tunes core (was ≈8–14 before the
positive-only arm). Expand sources/behaviors only after Phase 1 validates the
chain. (Why a subset, not the full ~950-adapter cross-product: §7.5.)

### 7.3 Layer comparability
Default within-layer; cross-layer pooling is an ablation with per-layer
mean-centering (#536). Pick per-behavior best layer in Phase 1.

### 7.4 `Σ_c` corpus & regularization
Estimate off a broad background corpus (≥2–5k contexts), **not** the 50-context
battery; regularize `Σ_c+λI` and/or top-eigendir truncation; sweep λ as a baseline
knob.

### 7.5 Why not all (C, B → C′, B′) combos?
The full tuple space is (source context × trained behavior × target context ×
evaluated behavior). We split it deliberately:

- **Eval IS full-grid (cheap on the cached store).** Once θ⁺_{C,B} exists, scoring
  all 50 target contexts × the **19-column registry (C8 — not 11)** is generation +
  judging on the cached grid — no extra training. The binding constraint is **judge
  calls, not GPU**, so the plan scores the **full scorable registry on the primary
  context per trained model** (the family-expression + diagonal columns fire only on
  their applicable adapter rows + base panel via `applies_to()`, so not every model
  pays all 19) and the `ROBUSTNESS_COLUMNS` subset (broad_em, sycophancy, marker,
  harmful_compliance) on the extra context families — the #545 testbed's own tradeoff
  (full battery × all contexts ≈ ~10× judge budget). **The §6 judge-budget figure is
  the 19-column count × the per-column call cost, ~1.7× the 11-column estimate.**
- **Training is NOT full-grid — combinatorial and unnecessary.** A fine-tune is one
  run per (source context, trained behavior). The full cross-product ≈ 50 contexts
  × ~19 row specs ≈ **~950 adapters**, each needing its own training + full-grid
  extraction + full-grid judging → **thousands of GPU-h and ~10⁷–10⁸ judge calls**.
  Infeasible — and pointless: the predictor's entire purpose is to forecast leakage
  for an **untrained** (C,B) from base-model quantities. So we train a
  **representative subset** to estimate ground truth and **validate** the predictor,
  then test generalization under **LOBO / LOCO** (§1.7). If we could afford all
  combos we wouldn't need a predictor.
- **Factorization makes most of the cross-product redundant.** If A3.8 + the
  no-interaction identity `L̂_{C',B'}=L̂_{C',B}·L̂_{C,B'}/L̂_{C,B}` hold, the
  generalized cell is determined by the two single-axis slices — the full (C′×B′)
  grid per source is then needed only to **test** factorization, not to use the
  predictor.
- **Held-out cells are required for honest evaluation** — training everything would
  leave no genuinely-new behavior/context to test generalization on (the paper's
  optimistic-evaluation guard).

---

## 8. Risks & falsification

- **A3.2 fails** (mean-answer summary insufficient) → whole chain stalls; the B4
  richer-summary arm (token-pooled / multi-layer / answer-dist) is the fallback
  summary, and if it too is insufficient the full deferred A3.1 recursive-pooling
  test is the next step. *This is why Phase 1 runs first and cheap.*
- **A3.6 fails** (read-out drifts under FT) → predictor isn't pre-FT computable;
  quantify drift, restrict to stable behaviors. **(C10)** judge A3.6 on the
  PARTIALLED-OUT change, not the level — a level "PASS" can be the base prior.
- **A3.7 PASSes for the wrong reason under contrastive training (B5)** → the
  positive-only arm's `δ=t−v0` is the clean identification; if positive-only and
  contrastive (`δ^contra`) disagree, report the divergence, do not credit a single
  contrastive `cos(ŵ,δ)` as the theory test.
- **Gate-scale circularity (B1)** → scoring the gate on the marker log-prob assumes
  A3.3+A3.8; score it on the activation `ĝ^real`. **Whitening-implementation
  artifact (B3)** → the cosine-limit unit test gates every whitened number.
- **A3.8 fails for content behaviors** (expected per #637) → report rank-one
  per-behavior; adopt the low-rank fallback `Σ w_j g_j`; do **not** average over
  the failure.
- **A3.10 fails** (g0 ≠ ĝ^real, g⁺ ≫ g0) → gate is real but not base-predictable;
  the drift decomposition says whether it's key/query/metric drift.
- **Overclaim: per-link PASS ≠ joint predictor.** Each A3.x can PASS while the
  composed `L̂` fails (Phase 4 is the joint test); and a gate PASS on a non-`c_C`
  key (verdict i) is not the boxed whitened predictor (verdict ii, B1).
- **Behavior-generality (B6)** → Phase-1 foundation is Betley-scoped; behaviors with
  bespoke eval batteries are tested in Phase 2 and carry a scope caveat. Content
  behaviors are expected to FAIL the predictor (#637) — do not narrate "leakage
  predictable" across all behaviors.
- **Multiple-comparison inflation (C3)** → the pre-registered primary path is the
  verdict; sweeps are FDR-corrected exploratory surfaces. **Shared-store DOF (C4)** →
  cluster-bootstrap CIs + independent-probe-split replication, never a naive n=50 CI.
- **Saturation** masks every gate signal → enforce non-saturated anchors (#448).
- **Base-prior confound:** the base behavior prior is a strong predictor of level;
  always partial it out / include as a baseline so a geometry "win" isn't the
  prior in disguise (#532/#649) — in the Phase-4 headline, not just the level tests
  (C7).
- **Noise floor:** report nothing above the test-retest reliability ceiling.

## 9. tasks/ rollout

One `proposed` task per phase (or per assumption-cluster), each routed through
`/adversarial-planner` → `/issue <N>`, all inheriting this document as the
program design. Suggested anchors in `docs/open_questions.md`: the leakage-
predictor (§3.1), context-geometry, and readout-stability questions. Phase 0+1 is
the first task (settles the foundation cheaply; running as #658); Phase 2 is the
fine-tune fleet; Phase 3+4 fold onto the same store. Operational details — phase
sequencing, the unified results doc, the theory-grounding gate, and the
no-auto-invented-experiments rule — are in §10.

> **NOT a `kind: campaign`.** This program runs as a closed, **auto-continuing**
> sequence of ordinary `kind: experiment` tasks (§10), NOT a `kind: campaign` task.
> A campaign session may auto-file and auto-spawn substantially-different children;
> that is explicitly disallowed here (§10d). The phase set is fixed; phases
> **auto-continue** — the orchestrator advances autonomously on a clean critic-gated
> PASS, with no per-phase human approval (after the one-time initial plan review).

---

## 10. Program orchestration & outputs

*(For Thomas's manual review — how the multi-phase program is run, where results
land, and the guardrails on autonomy.)*

### 10a. Each phase is an ordinary `kind: experiment` task

Every phase (0/1 → #658, then Phase 2, Phase 3, Phase 4) is filed as a normal
`kind: experiment` task under the program umbrella (`--parent 660`, the
`kind: survey` program-tracker) and run via its **own `/issue <N> --auto`
session** — the **full per-phase adversarial-planner stack** (planner →
fact-checker → critic ensemble ∥ consistency-checker → revise → approval),
critic-gated clean-result, and the standard upload/verify/analyze pipeline. This
document is the program-level design each phase inherits; it does NOT replace the
per-phase plan, which is written and adversarially reviewed independently for each
task. Phases **AUTO-CONTINUE**: the orchestrator advances to the next phase autonomously
on a clean critic-gated PASS of the prior phase — there is **no per-phase human
approval gate**. A genuine foundation failure (an unsound recipe lock, a failed
foundational assumption) is handled by the autonomous-mode pivot rules (re-run /
adjust autonomously; surface only if genuinely stuck), not by a routine approval.

### 10b. A central unified results doc

Each phase's promoted clean-result is distilled into a single living document,
**`docs/leakage_theory_program_results.md`** (created when Phase 1 lands). It
carries:
- an **assumption-by-assumption verdict table** (A3.1–A3.10 + joint factorization +
  end-to-end): PASS / FAIL / PARTIAL, the primary-path number (per the C3
  pre-registered recipe), the clustered CI (C4), the noise-floor ceiling, the
  behavior-scope (B6), and a one-line caveat — keyed to the phase task that produced
  it;
- the **end-to-end predictor** result (Phase 4): `L̂` vs the cosine special-case vs
  the base-prior baseline (C7) under LOBO/LOCO, on both scales, against the floor;
- the standing **overclaim boundaries** (content behaviors fail #637; per-link ≠
  joint; verdict (i) general gate ≠ verdict (ii) boxed whitened predictor;
  correlational; 7B/LoRA-only) carried verbatim so the program-level write-up cannot
  silently overclaim.

It is a distillation, not a replacement: each phase's own clean-result task body
stays the canonical per-experiment artifact; this doc is the cross-phase synthesis
Thomas reads for the program verdict.

### 10c. Mandatory theory-grounding (orchestrator + every phase subagent)

The program orchestrator AND every phase subagent (planner, experiment-implementer,
analyzer) **MUST read `docs/leakage_theory_paper.tex` IN FULL before designing or
analyzing that phase** — the predictor, the assumption blocks A3.1–A3.10, the exact
definitions (`δ_{C,B}=t−v0`, the whitened gate, `η`, `r_B`, `v0`, `c_C`, `Σ_c`), and
the evaluation methodology (two scales, LOBO/LOCO, noise floor) are the source of
truth. Every clean-result must match the theory's real math; no invented jargon, no
anthropomorphic methodology verbs. The per-phase task body states this grounding
requirement so the `/issue --auto` session inherits it; a plan that mis-states a
theory definition is a revise-blocker at the per-phase critic.

### 10d. NO auto-invented experiments — closed, auto-continuing phase set

The phase set is **closed**: Phase 0/1, Phase 2, Phase 3, Phase 4, plus the named
#658-landing cheap re-analyses (B2/B4/single-context/coherence). **No agent invents
new phases or substantially-different child experiments.** This is the explicit
reason the program is NOT a `kind: campaign` (§9): a campaign session is licensed to
auto-file + auto-spawn substantially-different children, which is exactly what must
NOT happen here. Within-phase follow-ups are allowed only under the standard rules
(a cheap same-question re-analysis folds into the same task; a substantially
different question is FILED as a `proposed` child for **Thomas's manual triage**,
never auto-spawned). Phases **AUTO-CONTINUE** — the orchestrator advances to the
next phase autonomously on a clean critic-gated PASS, with **no per-phase human
approval**. The per-phase `/issue --auto` plan auto-approves at ≤100 GPU-h (every
phase is under it — the Phase-2 fleet is ~55–95 GPU-h with the B5 positive-only
arm), so plans auto-approve and the chain runs unattended; only a plan exceeding
the cap would park. The orchestrator's job is to sequence + run, not to expand the
science.

### 10e. Maximize parallelism (across nodes if needed)

**Run everything in parallel as much as possible — across multiple nodes if
necessary.** The ONLY hard-serial barriers are the two data dependencies: **Phase 1's
locked recipe → Phase 2** (the fleet needs the locked extraction recipe) and **Phase
2's trained store → Phase 3/4** (the analyses consume the trained tensors). Subject
only to those, everything runs concurrently:

- **Within a phase:** data-parallel across ALL available GPUs (one model replica per
  GPU), provisioning multi-GPU and **multi-node** compute as needed; judge via the
  Batch API in parallel; fan off-pod CPU analyses across cores. (#658's 8×H100
  data-parallel extraction is the template.)
- **Phase 2 fine-tune fleet:** every source × behavior × arm cell is an INDEPENDENT
  fine-tune — train them all concurrently (one fine-tune per GPU, fanned across
  GPUs/nodes), never sequentially. This is the largest parallelism win in the program.
- **Phase 3 + Phase 4 overlap:** both consume the Phase-2 store; run them
  concurrently wherever a Phase-4 input does not depend on a specific Phase-3 output
  (predictor assembly can proceed alongside the per-assumption gate tests).
- **Per-assumption / per-cell work** within Phase 3 (A3.6–A3.10, joint factorization)
  and Phase 4 (baselines, LOBO/LOCO folds) is embarrassingly parallel — fan it out.

Provision additional / multi-node compute whenever it shortens wall-clock. Total
GPU-hours (what the ≤100-GPU-h plan cap measures) are set by the work, not the node
count, so parallelism cuts wall-time without changing the cost gate.
