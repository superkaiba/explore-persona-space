---
paths:
  - ".claude/rules/planner-section-reference.md"
description: >
  Full templates + worked examples for the planner.md plan sections
  (§0.0 / §0 / §3 / §4 / §6 / §6.5 / §7 / §9 / §10 / §11 / §12), relocated verbatim from
  .claude/agents/planner.md (#838). Loaded ONLY via the explicit pointer lines
  in planner.md — the self-matching `paths:` glob keeps this file out of every
  other agent context (a missing `paths:` key would auto-inject it always-on
  fleet-wide, recreating the #833/#834 spawn-weight bug this relocation fixes).
---

# Planner section reference (planner.md relocated section templates)

One H2 per plan section, headings verbatim from planner.md. Read ONLY the
section you are about to write: Grep the heading, then chunked `Read` of that
span (per planner.md § Context budget) — never the whole file.

## 0.0 TL;DR (plain English — the user reads this first)

**Three bullets, "I" voice, no architecture/library/jargon.** Mirror the
clean-result `## Takeaways` voice: a non-specialist colleague should be able to
read this and either nod, or ask "what about X?" — without scrolling and
without you having to translate. The frontmatter `goal:` is already the
one-sentence question; the §0 TL;DR does not restate it.

**Read the canonical Goal first.** Before drafting the plan, read
`frontmatter.goal` from body.md — this is the one-sentence target the
user filed at /issue Step 0c (or refined at clarifier Step 1). All of
the plan's downstream success/kill criteria must optimize toward this
Goal; the §0 TL;DR's "What I expect" and "What would change my mind"
bullets are predictions ABOUT the Goal, not restatements of it. If the
Goal reads as fuzzy and a sharper one would meaningfully change the
plan design, raise an
`AskUserQuestion` <!-- gate: gates.experiment_goal_refine --> <!-- autonomous-mode: skip --> proposing
the new Goal in Interactive mode only. On explicit user agreement in
the same turn, run
`uv run python scripts/task.py set-goal <N> "<new>" --by planner --reason "<one line>"`
and continue. Do NOT call `set-goal` without explicit user consent. In
autonomous mode (`EPM_AUTONOMOUS_SESSION=1`), the planner does NOT
propose a Goal refinement — the Goal is contract by the time the
planner runs; skip and continue with the existing Goal.

Place the §0.0 TL;DR block ABOVE the Plan Summary so the user reads
`## Goal` + TL;DR + Plan Summary together in 30 seconds.

- **What I'll run:** What does the experiment do, in plain words? *NOT*
  "Qwen-2.5-7B LoRA r=16 SFT on persona-tagged Tulu mix." Instead:
  "Train the same base model on three versions of the persona data that
  differ in one thing, and see which one teaches the trait without
  leaking to other personas."
- **What I expect:** What outcome am I betting on, in plain words?
- **What would change my mind:** What result would surprise me / would
  I want to investigate?

Anti-patterns this block must avoid: ZLT / BS / K-eval / dose / FWER /
collapse / Δ-notation / regression-coefficient language / library or
GPU-spec names. Save those for §0 (Plan Summary) and below.

**Self-pass: `/humanize quick` on §0.0 before returning the plan.** Invoke
the `humanize` skill in `quick` mode, targeting the §0.0 block only (NOT §0
or below — the technical sections are addressed to downstream agents and
keep project jargon on purpose). The quick mode runs a single-pass scrub
against the Wikipedia "Signs of AI writing" catalog: em-dash overuse,
inflated symbolism, vague attributions ("studies show"), AI vocabulary
("delve", "leverage", "underscore", "It is worth noting"), rule-of-three
constructions, negative parallelisms ("not just X but Y"), passive-voice
hedging. Apply the rewrites inline; do not return the plan with
unscrubbed AI-tells in the TL;DR. If the `humanize` skill is unavailable
in the agent runtime (e.g. plugin not loaded), apply the catalog inline
from your memory of it — single pass, no iteration.

## 0. Plan Summary (technical version — for the implementer, experimenter, reviewer)

A self-contained, ~150-word block that answers the seven questions
below. Use bolded labels at the start of each line so it scans in 30
seconds. This is the
technical companion to §0.0 — it can use the project's standard
shorthand (model names, library terms, eval suite names) because its
readers are downstream agents.

- **Training:** what model + recipe (e.g. "Qwen-2.5-7B, LoRA r=16 SFT on
  persona-tagged chat")
- **Hyperparameters:** the load-bearing ones — lr, batch, epochs, LoRA
  rank/alpha, anything novel. Each carries a one-token source tag (arXiv id
  or prior issue `#<M>`); full provenance lives in §11. Surface any
  `ungrounded` value here so the reader sees it at the approval gate.
- **Baselines / controls:** what we compare against, named explicitly
- **Loss surface:** where loss is computed (which tokens, which
  positions, e.g. "loss only on assistant tokens, marker token included")
- **Compute:** GPU hours total + # GPUs + parallelism mode (e.g. "4×
  H100 ZeRO-3 sweep, ~6 GPU-hours total wall ~1.5h"). MUST include a
  machine-readable total line the auto-approve gate parses:
  `Estimated GPU-hours (total): <number>` (a single number, total across all
  conditions/seeds, NOT a range). An autonomous `/issue` session auto-approves
  the plan whenever this total parses — the gate is GPU-hour-blind as of
  #1771, so no magnitude threshold applies; a missing/unparseable line fails
  safe to a park, so always emit a concrete number.
- **Evaluation:** primary metric + threshold for "this worked"
- **Risks (top 1-2):** the things most likely to invalidate the result

The Plan Summary must be self-sufficient: a reader who only sees this
block (plus the §0.0 TL;DR) must be able to approve / reject / ask a
question without scrolling further. No "(see §4 for…)" — restate any key
fact in the Summary even if it's duplicated below.

The user's AskUserQuestion <!-- gate: gates.plan_approval --> <!-- autonomous-mode: block-and-fail --> at
the plan_pending gate references §0.0 (TL;DR) and §0 (Plan Summary).
Optimize §0.0 for plain-English legibility, §0 for technical
completeness; the full sections below for everything else.

Interactive mode only — autonomous sessions never reach the ask: the
code-enforced gate in `task.py --auto-approve-if-autonomous` already
decided, and the PreToolUse hook
<!-- gate: gates.plan_approval --> hard-blocks any `AskUserQuestion` if
reached.

## 3. Hypothesis

(Relocated verbatim from planner.md §3, #1740.)

**Registered verdict lattice — declare it in the machine-checkable form.** A plan
REGISTERS a verdict lattice when it pre-defines outcome labels (Confirmed /
Falsified / H-slots / pass-fail-inconclusive grids) as interval predicates over
the same point estimates and CIs. `scripts/verify_plan.py` check 20
(`check_verdict_lattice_coherence`) verifies the labels PARTITION the outcome
space: two labels co-firing on one sign/CI cell, or a cell no label covers,
FAILs a `kind: experiment` plan (WARNs `analysis`) at Phase 1.5.0 and on every
critic re-verify (incident: #923 v4/v5). c20 verifies the partition IN FORM
ONLY — whether each predicate is the scientifically right boundary stays with
the Statistics critic. Declare the partition as ONE non-fenced line inside the
section that defines the labels (any heading matching
hypothes|success|kill|decision|verdict|gate — §3 here is the natural home):
`DISJOINT and exhaustive: <label> ⇔ <predicate>; <label> ⇔ <predicate>; <label> ⇔ otherwise.`
Live-verified worked example (#923 plan v6 §3 is the corpus exemplar):
`DISJOINT and exhaustive: Confirmed ⇔ Δ > 0 AND Δ's 95% CI excludes 0 on the positive side; Falsified ⇔ Δ's 95% CI is wholly below 0; Inconclusive ⇔ otherwise.`
Parser constraints (c20 tier 1): clauses `;`-separated on the SAME line; each
label ≤80 chars with no `;`/`⇔`; predicates built from `<qty> ≥/>/≤/< 0` sign
atoms and CI idioms ("CI excludes 0 on the positive side", "CI wholly below 0",
"CI straddles 0", "paired-diff CI strictly positive") joined only by AND / OR /
with; close with an `⇔ otherwise` clause (covers every residual cell by
construction). Per-label prose without this line is tier-2: co-fires still FAIL
and anything the parser can't read degrades the whole lattice to WARN — prefer
tier 1. No lattice in the plan → declare the byte-exact standalone line
`N/A — no registered verdict lattice` (never alongside a real lattice: c20
WARNs on the co-occurrence instead of silently skipping verification — #1223).

**Registered paired contrast — declare per-arm Row-coverage in the SAME draft.**
A plan REGISTERS a paired contrast when a non-fenced line inside a
registration-family H2+ section (any heading matching hypothes | success /
acceptance criteri | decision rule/gate | kill / abort / stop criteri |
evaluation | nulls | statistic — §3 here is the natural home) carries "paired"
plus registration vocabulary or an enumerated pair count ("7 pairs").
`scripts/verify_plan.py` check 18 (`check_paired_contrast_source_coverage`)
then REQUIRES a per-arm row-coverage declaration — FAIL for `kind: experiment`
(WARN `analysis`) at Phase 1.5.0 and on every critic re-verify (#810, #1112).
Every `plans/v{K}.md` is verified STANDALONE: an amendment / delta /
follow-up draft that registers or carries forward a paired contrast
RE-declares Row-coverage in its own text — the parent version's declaration
does not carry over (#1112's exact failure mode). Satisfy with ONE of (all
non-fenced; live-verified corpus exemplar: #1112 plan v8's `Row-coverage:`
line):
- **D1, named-source form** — ONE line starting `Row-coverage:` naming, for
  BOTH arms, which per-context store/file supplies every registered row; an
  artifact token (a `.pt/.json/.jsonl/.npz/…` filename or an `eval_results/…`
  / `analysis_tensors…/` / `raw_completions/…` path) must sit on the line or
  within the next 3 non-fenced lines:
  `Row-coverage: both arms' registered rows are supplied per-context by analysis_tensors/capture/<cell>/pooled.pt (trained arm) and eval_results/issue_<N>/base_rows.json (base arm).`
- **D1, by-construction form** — affirmative present tense ONLY (a negation /
  modal / deferral token near the clause — "will produce", "once implemented",
  "does not yet produce" — disqualifies it):
  `Row-coverage: the plan's own fits produce every registered row on each arm.`
- **D2, driver-assert form** — a subset expression + row/pair vocab +
  coverage/source/keys/assert vocab together on ONE line:
  `Row-coverage assert: the driver set-checks the registered pair rows ⊆ both named sources' row_meta keys before the statistic is computed.`
No paired contrast in the plan → declare the byte-exact standalone line
`N/A — no paired contrast` (never alongside a real registration: c18 WARNs
on the co-occurrence instead of silently passing — #1258). Keep every
declaration line free of cross-issue
citations — a `#<M>` token on the line DISQUALIFIES it (quote sibling
exemplars elsewhere) — and fill the `<…>` placeholders with THIS plan's
actual stores. c18 verifies the declaration IN FORM only; whether the named
sources truly contain every registered row on both arms stays with the
fact-checker. Guidance-shape pinned by
`tests/test_planner_row_coverage_guidance.py`.

## 4. Design

Concrete steps with:
- Exact training configs (epochs, lr, LoRA rank, batch size)
- Data specifications (format, size, generation method)
- Pipeline: what runs first, what depends on what
- File paths for inputs and outputs
- Pseudocode for any new code needed
- **Why code, not a model call?** — REQUIRED whenever the design includes a classifier, extractor, parser, summarizer, scorer, or rule-based judge over unstructured data (text / dialogue / images). State (a) the alternative single-model-call formulation considered, (b) why a code path is preferred (latency, determinism, cost at this N, structural output requirement, etc.), and (c) what would flip the decision. If no such component is in the design, write "N/A — no unstructured-data heuristics in this design" and move on. CLAUDE.md "Model call vs code (3.0 paradigm)" is the governing rule.
- **Contrastive negatives for behavior implantation (REQUIRED by default).** If the Goal is to implant a behavior (marker, fact, refusal, trait) into a source persona, the data design MUST interleave contrastive negative rows over the SAME questions under other personas — always including the bare default assistant, and at least 2-4 close negative personas — at roughly 1:1 positives-to-total-negatives, with on-policy leakage measurement and a non-saturated anchor. State the negative-persona set, the ratio, and the negative response construction (marker-less for marker implants; competing wrong-fact or refusal-pool for fact implants) explicitly here. Two exemptions, and only these: (a) the experiment's single manipulated variable IS contrastive-vs-non-contrastive (the non-contrastive arm is the deliberate control — state it that way), or (b) a strict single-variable replication of a positive-only parent (carry the parent's design AND flag the no-negatives regime as a scope caveat for the eventual clean-result). If neither exemption applies and you ship positive-only, the Methodology critic will REVISE. Full recipe + composition + caveats + citations: `.claude/rules/contrastive-negatives.md`. If the Goal is not a behavior-implantation Goal, write "N/A — not a behavior-implantation experiment" and move on.
- **Data source + realism tier (REQUIRED for every `kind: experiment`).** Name the training/eval/probe data source AND its tier on the CLAUDE.md realism hierarchy ("Design experiments on the most realistic data available"): (1) **real-world data** — production logs, user queries, naturally occurring text/code/conversations from the domain the claim targets (always first choice when accessible); (2) **established dataset / benchmark** — a published corpus the field already uses for this construct (cite it by name + canonical source); (3) **DIVERSE LLM-generated synthetic data** — only when tiers 1-2 are genuinely unavailable, and only with deliberate variation across lengths, structures (single-turn/multi-turn, code/prose/dialogue), framings, topics, and surface forms (a flat templated corpus with an LLM in the loop is NOT tier 3 — it inherits tier 4's brittleness); (4) **programmatically generated data** — LAST resort, requiring an explicit recorded argument for why no other source works AND why the templated structure cannot bias the result (valid e.g. when the construct under test IS a controlled template, like the marker token in a fixed slot). Justify any tier-3 choice with the demonstrated absence of tiers 1-2 and any tier-4 choice with the confound argument, here or in §12 Assumptions; the Methodology critic REVISEs otherwise (critic.md Methodology lens item 15), and the tier is carried into the clean-result as a scope caveat. This bullet governs the data SOURCE (where prompts/corpora come from); completion PROVENANCE (who wrote the response text) is the next bullet — a behavior-implantation plan names both.
- **Completion provenance for behavior-implantation training rows (on-policy-first, REQUIRED by default).** For every training-row type in a behavior-implantation design (positives AND negatives), name its completion provenance: `on-policy (elicitation tier 1 bare / 2 instruct-and-strip / 3 prefill)` | `canned/template` | `third-party-LLM-written` | `published-corpus-verbatim`. The DEFAULT for positives is on-policy from the BASE model: behavior instruction in the system prompt, sample, judge-filter for the target behavior, STRIP the elicitation instruction before training, prefer the lowest ladder tier that fills a pre-registered per-source yield quota, record per-row tier — full recipe: `.claude/rules/on-policy-completions.md` (read it before designing the data; #612 is the worked example). State the yield quota + retry budget AND the drop rule — DEFAULT: an **80% floor with equalize-down** (every kept source trains on exactly floor-N rows; the yield-quota bullet under `.claude/rules/on-policy-completions.md` § The recipe), including the close-miss escalation clause: a source at ≥ 90% of floor but below it after the retry budget gets ONE recorded same-construct escalation tranche before the drop (the Close-miss escalation sub-bullet, same rule file); a source below the floor after the retry budget (post-escalation where triggered) is dropped and the drop reported as a finding — never silently backfilled with templates. Canned/templated or third-party-LLM-written positives require EITHER an explicit anchor/control role (the data construction IS the manipulated variable, stated as such) OR a recorded on-policy yield failure — named here and carried into the clean-result as a data-realism caveat. Plan dose-sensitive reads around the known trade-off: on-policy data installs more weakly at matched recipe (#612's three separated dose bands), so dose-to-target instead of fixed epochs when comparing data constructions. Exempt: published-corpus replication rows (replication fidelity wins — "Before Planning" item 6) and the programmatic marker carve-out (the appended marker token; the base response text is already on-policy under the marker recipe). When ONE datagen pipeline implants MORE THAN ONE behavior, additionally name the behavior-definition shape per behavior — the standardized persona-vectors template (trait name + natural-language description → 5 contrastive pos/neg instruction pairs + a shared/auto-generated neutral trait-eliciting question set + rubric; steps 1–2 of `.claude/rules/persona-vectors-recipe.md`), never bespoke per-behavior definitions or query banks (`.claude/rules/on-policy-completions.md` § Standardized behavior definitions; incident #906→#1090). If not a behavior-implantation design, write "N/A — not a behavior-implantation experiment" and move on.
- **Marker / behavior-implant stopping recipe (overrides parent parity).** If the design trains a FRESH marker / behavior-implant adapter, the stopping recipe — lr, epochs / steps, checkpoint selection — comes from `.claude/rules/marker-training-recipe.md` (read it in full first), REGARDLESS of what recipe a non-marker parent used. Recipe parity with a non-marker parent is NOT a valid grounding for the stopping recipe (see §11 "Marker recipe overrides parent parity"); name the parity break in §12 Assumptions as a deliberate measurement-validity deviation, and keep cross-experiment parity on the DV / eval side (same panel, same probes, same join). If the design ALSO declares a runtime saturation guard / trajectory monitor as a mitigation, declare it smoke-verifiable: name the telemetry the implementer's smoke run will show (distinct per-source WandB run names, at least one logged trajectory point, the guard branch or its precondition assert exercised) — an unverifiable guard is a paper mitigation (#480: the declared WandB trajectory monitor + KL auto-fire silently never functioned). If not a behavior-implant design, write "N/A — not a behavior-implantation experiment" and move on.
- **Persona-vectors extraction recipe (REQUIRED when a task elects persona vectors).** If the task plan says "use persona vectors" / "extract a persona vector" / "persona-vectors-style direction", OR extracts a persona/behavior direction by mean-difference of positive/negative contrastive activations, the plan MUST instantiate the 7-step recipe of `.claude/rules/persona-vectors-recipe.md` (read it in full first): (1) trait name + description; (2) artifacts — 5 pos/neg system-prompt pairs + 40 questions split into a disjoint 20-question extraction set + 20-question evaluation set + 1 trait evaluation prompt (fetch the verbatim generation + rubric templates via the arXiv MCP, never a paraphrase); (3) 10 on-policy rollouts under the positive system prompt + 10 under the negative per extraction question (sampling, default temp 1.0); (4) the **judge-filter** — score every rollout 0–100 with the trait evaluation prompt, keep positive-prompt responses >50 and negative-prompt responses <50, judge = `claude-sonnet-4-5-20250929` — AND a judge return that is `REFUSAL`, non-numeric, or outside [0, 100] is DROPPED from BOTH arms (never coerced to a numeric score; coercing a refusal to 0 silently keeps it as a clean `<50` negative and corrupts the negative-arm mean exactly where elicitation produces the most refusals), with the per-arm dropped-rollout count reported; (5) response-avg residual-stream activations at every layer; (6) `r_B` = diff-of-means per layer; (7) **name the layer-selection regime** — steering/monitoring (one steering-selected layer, per the paper) vs read-out/prediction (sweep all layers, select by target predictivity — e.g. the paper's A3.3 `E ≈ r_Bᵀ v`). Cite `.claude/rules/persona-vectors-recipe.md` as the §11 `Source:` for any extracted direction's recipe values. **The "except logits" carve-out:** the recipe replaces the paper's GPT-4.1-mini logit-weighted scoring with the project Sonnet 4.5 judge — do NOT introduce a second judge or the paper's logit scoring; reintroducing it requires an explicit `### Override:` note. If the task does not elect persona vectors, write "N/A — no persona-vectors extraction in this design" and move on.
- **Multi-arm resolution-band designs (band-stop not applicable).** If the headline test gates on ≥2 conditions/arms sitting SIMULTANEOUSLY inside a measurement band (e.g. a wrong-persona log-prob band) at a MATCHED training amount, the band-stop default does NOT cover it — per-arm early-stopping would unmatch the training amounts. The plan MUST state: (a) the expected install-transition window in optimizer steps, with a `Source: #<M>` citation like any §11 hyperparameter (current role-vs-system estimate: ~12 steps, between ~step 18 and ~30 — #533/#547); (b) checkpoint spacing FINER than that window — grid in optimizer steps, never whole epochs; (c) a pre-registered per-arm band-entry fallback read that answers the headline question when the arms never co-resolve at a shared grid point (compare arms at their respective band-entry checkpoints: matched dial position, unmatched step count; an arm that never enters the band is reported as exactly that). Three consecutive runs (#529/#533/#546) burned GPU without firing their headline test for lack of these; re-running the same anchor-gated design without changing the grid unit or adding the fallback read is banned. Full section: `.claude/rules/marker-training-recipe.md` § Multi-arm resolution-band designs. If the headline test does not gate on multi-arm band simultaneity, write "N/A — no multi-arm band-simultaneity gate" and move on.
- **Few-shot / in-context-example demonstration content is a grounded design element, not filler.** If the experiment uses any in-context-example / few-shot / ICL demonstration set (a fixed bank of `<question, answer>` pairs the model sees before each probe, whether read by the trained model, by a base model under a persona prompt, or as training-time demonstrations), the plan MUST state, per demonstration set: (a) the eval-task distribution the demos mirror (the actual task type the model will be evaluated on with this context — not "generic helpful Q&A" if the eval probes are, say, persona-voiced marker emissions on open-ended prompts), (b) why this specific content induces the intended behavior / persona / context (cite the design pressure that picked it — a paper, a prior issue's recipe, a held-out sanity check), AND (c) that the demonstration content varies enough ACROSS the different ICL contexts to give cross-context dynamic range (if four "different" ICL contexts are four slices of the same neutral trivia pool with the same one-word answer shape, they will read as one context to the model). Anti-contamination (no overlap with held-out probe answers) is NECESSARY but NOT SUFFICIENT — a contamination-only design pressure tends to drive the content toward bland, generic, near-clone demos that satisfy the contamination check while giving ~zero cross-context dynamic range and barely inducing any behavior, which is the opposite of why ICL was introduced. State each of (a), (b), (c) explicitly — the Methodology critic will REVISE an ICL plan whose demo content is justified only by contamination avoidance. The closest record: task #489's ICL contexts were four 4-item slices of a 16-fact trivia pool with persona-voiced demos that slapped a stock prefix on a one-word answer ("Arr! Au."), sailed through Planner → Fact-Checker → Critic → Consistency-Checker uninspected, and likely contributed to the marker-implant floor. If the experiment uses no ICL / few-shot demonstrations, write "N/A — no ICL or few-shot demonstrations in this design" and move on.
- **Smoke/sweep architectural parity (UNIFICATION DEFAULT, canary escape hatch).** The DEFAULT is unification: smoke IS the sweep with one cell — same dispatcher, same subprocess shape, same env injection, same logging surface, same teardown sequence. State this explicitly here: "smoke phase = sweep with `--cells 1 --seeds 1`" (or equivalent single-cell parameterization). If the design diverges (e.g., smoke uses in-process `train_one_cell`, sweep uses a `subprocess.run(["uv", "run", "python", "src/.../experiments/<name>/run_one_cell.py", ...])` wrapper), justify the divergence in two sentences AND name which canary cell exercises the sweep path during smoke. The bar for accepting divergence is high: subprocess isolation is only justified when the sweep's per-cell teardown / resource-isolation requirements would block in-process execution (e.g., per-cell vLLM allocation that can't be reset cleanly in-process). Task #397 rounds 9/10/10' (2026-05-27) burned three full implementer rounds on architectural assumptions that the in-process smoke path silently satisfied; the round-11 pivot was to UNIFICATION (in-process serial). Enforced at /issue Step 6d.0 via the `epm:smoke-architecture-check v1` gate (see SKILL.md).
- **Smoke blind-spot enumeration (REQUIRED whenever the plan declares a pre-launch smoke run).** The section that declares the smoke states, in one short block titled `Smoke blind-spot enumeration:`, what the smoke's PASS does and does NOT certify — derived from `.claude/rules/smoke-blind-spots.md` (read it first): (a) every production gate/assertion the smoke DOWNGRADES or skips (an `assert_split(..., smoke=...)`-style early return); (b) every implementation the smoke SUBSTITUTES (toy embedding, stub model, fake judge — the production constructor / API call never runs under smoke); (c) every third-party import reached ONLY on the production branch (a `smoke=False`-only `SentenceTransformer` leaves `import sentence_transformers` unexecuted — #1336 SLURM 4684). An EMPTY enumeration is written as the literal `none — smoke executes every production gate`, never left blank — and any smoke-conditional substitution/downgrade branch in the design falsifies it. Rationale + incident: #1336 — two consecutive production SLURM launches (4684, 5005) died on a missing dependency and a downgraded split assertion the smoke branch structurally could not reach; nothing in the plan said the PASS certified less than it appeared to. Distinct sibling, not a duplicate: the parity bullet above makes smoke and production the SAME code; this enumeration covers the RESIDUAL divergences that survive a justified parity break or a `smoke`-kwarg branch inside shared code. Enforced at review by code-reviewer.md Step 0.71 (blocker tag `smoke-blind-spot-unenumerated`) + critic.md Methodology lens item 19; mechanical backstops `workflow_lint.py --check-smoke-blind-spot-review-lens` (surface pin) and `--check-smoke-blind-spots` (WARN scan). If the plan declares no pre-launch smoke run, write "N/A — no pre-launch smoke run" and move on.
- **No all-or-nothing eligibility gates on continuous quantities (graceful degradation).** When a pre-registered rule gates a unit's inclusion on a continuous quantity (rows filled, samples passing a judge filter, cells surviving a data gate), design graceful degradation — a floor with documented shortfall — instead of a binary keep/drop at the target value. A binary rule discards near-misses wholesale: #612's "fill all 200 rows or drop the source" rule discarded one source at 194/200 (97% fill — 6 missing rows) and another at 169/200; together the drops halved the on-policy design's coverage. State the floor, what happens between floor and target (kept, shortfall documented), and what happens below the floor (dropped, reported as a finding). The canonical instantiation for on-policy yield quotas — an 80% floor + equalize-down + the ≥ 90%-of-floor close-miss escalation tranche — is in `.claude/rules/on-policy-completions.md`. If the design has no continuous-quantity eligibility gate, write "N/A" and move on.
- **Equalize-down when a per-unit resource varies across units/conditions.** If units (sources, personas, conditions) legitimately fill to different N (training rows, samples), train/evaluate ALL units at the same floor-N — discard the surplus everywhere — rather than letting N vary per unit: variable N is a dose confound, and dose/schedule length is the demonstrated dominant lever (#601). Scale coupled quantities proportionally to floor-N so load-bearing ratios survive the equalization (e.g. contrastive negatives at the ~1:1 positives-to-total-negatives ratio, `.claude/rules/contrastive-negatives.md`). Prefer the same-question/claim subset across units where filled rows allow; else a random floor-N sample with the coverage difference documented. If per-unit N is equal by construction, write "N/A" and move on.
- **Baseline propensity on BOTH sides of an implantation design.** Before installing/eliciting a behavior, measure each unit's PRE-intervention behavior rate on the eval probes — the EVAL-side targets (the delta denominator) AND the SOURCE-side personas. The source-side read is cheap (one base-model generation + judge pass), predicts elicitation-yield failures before any training is spent (#612: both yield-quota failures were predictable from a source-side baseline read that was never taken), and is the natural install-strength covariate — a unit's own base prior keeps beating geometry as a predictor (#500/#532/#541). If not an implantation/elicitation design, write "N/A — not a behavior-implantation experiment" and move on.
- **Generation-and-reduce stages persist their rollout TEXT (persist-by-default).** If a stage GENERATES model outputs then REDUCES them (persona-vector extraction reducing to `r_B`; an online-scored eval reducing to a rate; any stream-reduce over model generations), the plan lists the rollout TEXT under `raw_completions/<stage>/` AND per-context intermediates under `analysis_tensors/` (upload-if-cheap-else-`discarded_artifacts:`-with-regen, declared in §10), even when the current task has no downstream use — a sibling / follow-up arm may (#779 lost the extraction rollouts to a reduce-and-discard driver). The stream-reduce itself is unchanged (persist the text you reduced; never materialize the whole activation grid — #666/#772). Text / JSON is never a valid `discarded_artifacts:` entry; only a genuinely too-big tensor is, and only with its regenerating text persisted (planner §10, CLAUDE.md § Upload Policy). If no stage generates-then-reduces, write "N/A — no generation-and-reduce stage."
- **Symbol-existence grep-at-plan-time.** Every `module.symbol` (function, class, subcommand, CLI flag) named in plan pseudocode or a §4 mechanism is confirmed at plan time by a recorded grep — `grep -rn 'def <symbol>' src/ scripts/` (or the equivalent for a class / CLI arg / config key), pasted verbatim into the §12 assumption row for that symbol. Deferring a grep-answerable symbol-existence check to the implementer is banned: a plan that names `foo.bar()` without a plan-time grep silently pivots to a "does this function exist?" implementer round when it does not. Worked example (§4 bullet): `Uses `scripts/task.py post-marker` (verified `grep -n 'def cmd_post_marker' scripts/task.py → 1289:def cmd_post_marker`, §12 row A1).` Escape: `N/A — no external symbol referenced` when the plan's §4 is pure prose / narrative spec-text edits with no code symbols.
- **Pre-return self-check enumerating task-body-named required edits.** Before returning the plan, enumerate every file + section the TASK BODY's `## Proposed change` (or equivalent required-edits block) names, and assert each appears in §4 Design. A deliberate omission (an edit the planner elected to skip / defer / merge into another) carries a one-line reason next to the missing item; a silent omission is a defect. Worked example (a self-check block near the end of §12): `Task body §Proposed change lists 6 required edits: (1) §11 tool-behavior Source — PRESENT as §4 Edit 1; (2) §12 grep -n VERBATIM — PRESENT as Edit 2; ... (6) shell exit-path — PRESENT as Edit 6. No omissions.` This self-check is REQUIRED on every workflow-fix plan and every plan whose task body carries a `## Proposed change` list of required edits; other plans may skip it (escape: `N/A — task body has no required-edits list`).
- **Embedded-shell exit-path trace (prose-only clause).** For every FAILURE ARM in embedded shell — every `false` / `exit 1` / `return 1` / `raise` that a plan writes inside a `&&`-chain, an `if`/`else`, a `case`, or a Bash function meant to halt the enclosing block — trace the exit path: does the failure actually propagate to the enclosing script's exit code, or does it get swallowed by a sibling `then` / `else` branch that runs after it? A bare `false` inside an `if ... ; then true; else false; fi` does NOT halt a subsequent `echo` on the next line; the recommended shape is the OK-flag pattern `OK=yes; [ ... ] || OK=no; [ "$OK" = yes ] || exit 1` (single exit point, no sibling races). This clause is prose-only (no mechanical `workflow_lint.py` check in v1); the two review sites that catch a bare `false` in embedded shell are (a) `.claude/agents/critic.md` § Statistics & Measurement lens item 3 (decision-gate coherence / joint satisfiability) at plan review, and (b) `.claude/agents/code-reviewer.md` at the implementer diff. A future workflow-fix task MAY add a `workflow_lint.py --check-plan-shell-exit-paths` mechanical gate; out of scope here. Escape: `N/A — plan embeds no shell control flow` when the plan is pure narrative / no shell blocks.

## 6. Evaluation

Metrics, thresholds, statistical tests. What does success look like numerically?

**Required: Measurement validity (the §11 for outputs).** The Goal names a *construct* — a real behavior — but the eval only ever measures a *proxy* for it. For EACH dependent variable, state a one-row entry:

| DV | Construct (what the Goal cares about) | Metric (what is actually computed) | On-distribution? | If proxy: validation / justification |
|---|---|---|---|---|

- **Construct** — the behavior the Goal is about, in plain English (e.g. "the rate the model emits ※ when it generates an answer under each persona").
- **Metric** — exactly what is computed (e.g. "teacher-forced log p(※) at the first assistant token / after a fixed canonical answer").
- **On-distribution?** — does the metric observe the behavior under the conditions it actually occurs: on-policy (the model's *own* generated text, not a fixed stub), at the natural token position (where the behavior is emitted, not an arbitrary probe slot), over a realistic prompt distribution? `yes` / `no`.
- **If proxy (`no`)** — the DEFAULT is on-policy / behavioral measurement; an off-distribution / teacher-forced / fixed-context / single-position proxy is opt-in and MUST carry EITHER (a) a validation that the proxy tracks the construct (e.g. "Spearman of proxy vs free-generation emission rate on K conditions = …", or a planned validation step in §4), OR (b) an explicit argument the proxy answers *this Goal* despite the gap. "Cheaper / cleaner / deterministic / one forward pass" is a real cost argument but is **not**, by itself, a validity argument — name it AND the validity basis.

A plan that measures a behavioral construct with only an unvalidated off-distribution proxy is a §6 defect the Statistics & Measurement critic REVISEs. `kind: analysis|infra|batch|survey` may write "N/A — no behavioral construct measured" and move on.

**Required: Dual-DV for content-behavior leakage / implantation (sycophancy, refusal, hedging, style, trait).** If the Goal implants or measures the leakage of a *content* behavior (sycophancy, refusal, hedging, style, trait — anything other than the programmatic marker, whose three-space recipe is separate), §6 MUST name BOTH dependent variables, per CLAUDE.md § Measurement validity (standing rule 2026-06-15):

- **(a) PRIMARY — a judge-scored on-policy behavior/agreement RATE.** The validated behavioral construct: the model writes its OWN response under the persona/target context, a Claude judge labels whether the response expresses the behavior, and the DV is the rate of judge-positive responses (trained vs base). This is the headline DV.
- **(b) SECONDARY — a continuous completion-probability DV** that keeps dynamic range where the binary rate saturates. PREFERRED form: a teacher-forced fixed positive-vs-negative completion margin (mean LN-`log P` of a FIXED judge-filtered positive-answer pool minus a FIXED negative pool, same answer set across every context ⇒ no selection-on-outcome bias; #722 validated ρ(margin, rate) all-positive). It carries the teacher-forcing-artifact risk (#432→#456) — name it, and keep it a SECONDARY companion, never the behavioral leaderboard. OPT-IN alternative: length-normalized trained − base `log P` of the model's OWN judged-positive on-policy completions (`logp_pos_mean`) — a conditional mean over an OUTCOME-SELECTED subset, so it carries selection-on-outcome bias and FAILED the ρ(DV, rate) > 0 validation for 3 of 4 behaviors in #722; use it ONLY after it passes that validation. (Full recipe: `.claude/rules/llm-judging.md` § E2.)
- **Why both are required, not one:** the binary judge rate saturates at floor/ceiling and CENSORS install / dose-matched / cross-condition comparisons (#608's top-band censoring) — exactly where the continuous DV keeps headroom; the probability DV in turn carries the teacher-forcing-artifact risk the judge rate is immune to. They cover each other.
- **The probability DV stays SECONDARY and requires a validation that it tracks the rate** — a Spearman of (b) vs (a) across the cells that have dynamic range, named here in §6 (computed by the analyzer). Never narrate the probability DV as the construct, and never let it replace the on-policy rate.

This mirrors the marker line's behavioral-primary + continuous-secondary recipe, judge-rate-primary. A content-behavior leakage/implantation plan that registers ONLY the binary judge rate (no continuous companion) when it makes install / dose-matched / cross-condition comparisons or is at ceiling-saturation risk is a §6 defect the Statistics & Measurement critic REVISEs (item 10); so is one that registers the probability DV as primary or without the rate-tracking validation. Plans whose Goal is not a content-behavior leakage/implantation Goal (a marker implant, a non-behavioral analysis, geometry/probe work) write "N/A — not a content-behavior leakage/implantation experiment" and move on. Full rule: CLAUDE.md § Measurement validity.

**Required: Install-strength control (cross-condition leakage comparisons).** If the plan's headline compares LEAKAGE across training conditions (contrastive vs positive-only, LoRA vs full fine-tuning, data-construction variants), raw bystander leakage is dose-confounded: install strength is condition-dependent — not even in a fixed direction across behaviors (#601: contrastive negatives strengthened the marker implant; #608: positive-only sycophancy installed at least as strongly) — so "X leaks more than Y" read off raw leakage conflates lower selectivity with plain stronger implantation. §6 must register at least one install-controlled read: (a) a matched-install comparison (conditions compared at checkpoints with matched source gain, findable from the per-step trajectory / band-stop logging), and/or (b) leakage as a fraction of install per (source → bystander) cell, computed in the non-saturating EOS-margin logit space `Δ(z_marker − z_eos)` — NEVER raw `log P`, whose softmax compression shrinks a saturated source's denominator and inflates the fraction exactly in the strongest-implant conditions; plus (c) leakage-vs-install dose curves from the per-step trajectories where they exist (preferred — subsumes the single fraction and catches nonlinear leakage onset). Never correlate the fraction back against install itself (shared-noisy-denominator artifact, same family as the #383 X-vs-(X−Y) caveat). The primary DV definition is unchanged (on-policy `log P(marker)` trained − base stays primary). Full recipe: `.claude/rules/marker-leakage-measurement.md` § Install-strength confound. Plans whose headline makes no cross-condition leakage comparison write "N/A — no cross-condition leakage comparison" and move on.

**Required: Statistical-input existence (derived inputs for registered corrections).** Every registered statistical correction / adjustment §6 relies on — attenuation / reliability factors, per-seed SEs, variance reconstructions, shrinkage priors, any statistic computed FROM a derived input rather than directly from this run's raw eval output — must name the data dependency it consumes AND verify that dependency actually EXISTS in the cited artifact (the column is present in the CSV, the per-seed files resolve on HF, the field is in the JSON schema — check the actual file, not the producing plan's prose), OR explicitly schedule its construction as in-scope implementation work in §4 / the file-level diff list. This is the plan-time analogue of the step-5 Hub-existence check, extended to derived statistical inputs: an input that is "derivable in principle" but neither verified-present nor scheduled-to-build is a phantom dependency (#509: registered attenuation-adjusted correlations consumed per-seed SEs that existed nowhere; the production path crashed as predicted and the result shipped with the correction pinned to 1.0). Plans with no registered derived-input corrections (raw DV + standard tests only) write "N/A — no derived statistical inputs" and move on.

**Required: Selection-symmetric nulls (max-over-axis headlines).** If the
plan's headline statistic is chosen by `max` / `argmax` / best-of /
top-k-mean over a FREE AXIS — a read-out layer, a cell, a k /
neighbourhood size, a seed, an extraction point, a threshold — and is
compared against a null / permutation / shuffle band, the band procedure
MUST be selection-symmetric: EITHER (a) every null draw receives the
IDENTICAL max-over-axis selection before the band is formed (the null
distribution is the max-selected statistic per draw), OR (b) the axis is
frozen on a held-out split / pre-registered fixed position and BOTH the
observed statistic and every null draw are read at that single frozen
position. A `max-over-L` observed statistic compared against a
one-position null is a 28-vs-1 asymmetry that manufactures the
observed-vs-null gap from the winner's curse, not the effect the null
tests (#778, n=24: single-layer null p97.5 |r| ≈ 0.48 vs honest
max-over-layer p97.5 |r| ≈ 0.62; a per-axis heatmap is a diagnostic
display and does NOT neutralise it). §6 ALSO registers persistence of the
per-draw × per-axis statistic matrix (one matrix per headline statistic)
as a downstream artifact (per the Upload Policy) so the analyzer can
recompute the honest max-selected band post-hoc. §6 ALSO reports, next to
each registered null-band read, the band's upper bound alongside the DV's
achievable ceiling (a bounded DV's bound; for a difference statistic,
max-attainable-favored-arm MINUS the exact registered comparison-arm
quantity the statistic uses — never the raw single-arm bound; when no
estimator bound is derivable, the largest previously-observed in-line
effect as a severity REFERENCE POINT, which is not a ceiling) — band ≥
estimator-bound ceiling ⇒ the test is uninformative-by-construction:
pre-commit the narration of any non-rejection as failure-to-reject (never
evidence of absence/reversal; a separately reachable opposite-tail
rejection stays legitimate), draw band + ceiling in the figure, report the
band-to-ceiling margin, and prefer redesigning the read (#810: band p97.5
0.800 vs an achievable difference ceiling derived from ~0.857 max skill —
even the parent round's +0.209 effect could not clear it). Band ≥ only the
fallback reference point ⇒ report low-severity (underpowered against every
previously-observed effect), not zero power. Full check:
`.claude/rules/selection-symmetric-nulls.md` § Band-vs-ceiling
informativeness check.

A bootstrap / resampling CI registered at the max-selected position MUST
be the selection-inherited CI (per-draw re-selection inside each
resample) — or BOTH CIs, labeled frozen-at-<axis> vs selection-inherited;
a frozen-only CI at a selected position is a critic REVISE (#1434:
frozen [−0.949, −0.467] vs selection-inherited [−0.957, +0.866] in the
same JSON). Full clause: `.claude/rules/selection-symmetric-nulls.md`
§ Bootstrap CIs at a selected axis position.

Full recipe + carve-outs
(a pre-registered fixed position or a mechanistic single-anchor ablation
does NOT fire this): `.claude/rules/selection-symmetric-nulls.md`. Plans
whose headline is not selected over any free axis write "N/A — no
max-over-axis selection in the headline" and move on.

The SAME observed-vs-null symmetry discipline extends to NOISE STRUCTURE
(conditional — fires only when the pattern is present): if any registered
statistic's observed and reference legs share ONE SAMPLED quantity — a
shared empirical baseline mean (`cos(X − B̄, Y − B̄)`), a shared sampled
anchor/denominator (frac-of-anchor), a change-score correlation sharing a
baseline — and it is compared against any null/reference band, §6 MUST
register EITHER disjoint baseline draw halves feeding the two legs OR a
null constructed to carry the identical shared-B̄ term per draw (at the
pre-subtraction leg level, norm-matched to the observed raw leg), PLUS the
split-half reliability report for each sampled difference-vector leg. A
noise-free null (norm-matched random directions, independently
re-centered shuffles) under-covers the shared-noise cross-term
(≈ +tr(Σ_B)/n_B on the observed side only) — #1415: disjoint-baseline
recount dropped prefix 0.271→0.178 / context 0.362→0.272, sent one pair
0.23→−0.08, and pulled 6/28 prefix pairs below the null p97.5; the
"28/28 clear the null" headline did not survive. Full recipe:
`.claude/rules/selection-symmetric-nulls.md` § Noise-structure symmetry.
Plans without the pattern need no extra declaration (optionally "N/A — no
shared-baseline difference-vector DV").

**Required: OOD generalization folds (group-structured held-out
predictive DVs).** If any DV is a held-out predictive statistic —
reconstruction R² / skill, read-out ρ, predictor accuracy, any
"held-out" / "cross-validated" number — over a sample with known GROUP
structure (context/prompt families, genres, persona panels, behavior
classes, seeds sharing a template), §6 MUST: (a) NAME the sample's
grouping axes — "no known structure" is a positive claim requiring an
explicit iid argument, the only pointwise-only exemption; (b) register
at least one GROUP-level held-out fold — leave-one-family-out (LOFO) /
leave-one-genre-out / leave-one-persona-out, or a corpus/genre
TRANSFER arm (fit on corpus A, evaluate on corpus B — the strongest
form, counts); (c) report BOTH and label EVERY headline with its
fold — pointwise LOO/LOCO may stay (it upper-bounds
within-distribution skill) but never carries a generalization claim
alone; a claim that holds under LOO and fails under the group fold is
reported as within-family interpolation, not generalization; (d) give
any max/argmax-over-free-axis selection inside a group-fold headline
its selection-symmetric null (the block above) computed under the
SAME fold structure; and (e) frame CIs and "unresolved" calls on
GROUP-level n — G quasi-independent test units, not n points.
Pointwise LOO trains on same-family siblings of every test point, so
it measures within-family interpolation and can REORDER cross-context
claims: #810's LOCO headline (max-pool 0.826 best answer-side
summary vs mean 0.800; trained-ridge read-out ρ ≈ 0.909) reordered
under the 7-fold leave-one-FAMILY-out re-read (mean 0.804 ≥ turn_nl
0.791 > max-pool 0.760 at LOCO-best layers) and the read-out
collapsed to ρ ≈ 0.285. Full rule:
`.claude/rules/ood-generalization-folds.md`. Plans with no held-out
predictive DV write "N/A — no held-out predictive DV" and move on; a
plan claiming a genuinely iid sample writes the iid argument here
instead of the N/A.

**Required: mapping-baselines pair (fitted representation maps).** If the
plan FITS a map between activation summaries (context→answer,
prefix→context, cross-model / cross-framing reparameterization — any
v_X→v_Y predictor), §6 MUST register BOTH standing reads alongside
held-out R², per the CLAUDE.md standing rule (2026-07-22): (a) the
identity-family baseline including the learned-bias form x + b with
b = train-fold mean of (y − x) — canonical helper
`analysis/mapping_baselines.identity_bias_predict` — whenever input and
output spaces share dimension (a dimension mismatch is STATED as
inapplicable, never silently skipped); and (b) the kNN-retrieval read
P(true target within the k nearest neighbors of the prediction) among the
held-out candidate pool — canonical helper
`analysis/mapping_baselines.knn_retrieval` (euclidean + cosine, k scaled
to the pool, chance = k/n_pool stated; a constant predictor reads exactly
chance). The two reads DISSOCIATE in both directions (first measurement
2026-07-22): identity+bias scored pooled-OOF R² −6.5 yet retrieval acc@1
0.84 vs the LOFO ridge map's 0.04 on the #722 prefix-level 50-context
battery map (`eval_results/issue_722/identity_bias_knn/`), while the #779
LMSYS single-context fitted ridge dominated retrieval (acc@1 0.72 vs 0.50
identity+bias, chance 0.001; `eval_results/issue_779/identity_bias_knn/`)
— R² alone can both overstate a map (variance a constant shift already
explains) and understate one (discriminative but mis-scaled predictions).
Omitting either read is a stated deviation, never a silent default. Plans
that fit no map write "N/A — no representation map fitted" and move on.
Full rule: CLAUDE.md § "Identity+learned-bias baseline AND kNN-retrieval
metric".

**Pooling-convention row (same trigger).** The same §6 registration names
the POOLING of EVERY vector entering the map (context / prefix / answer /
cross-model summary: span-mean | last-token (prompt-final-token /
newline-before-assistant) | response-avg | other), AND states whether each
convention MATCHES the pooling of the cited comparison/baseline line (e.g.
the #779 last-token line); a deliberate mismatch carries a one-line
justification. Pooling is a load-bearing measurement choice that silently
rides in from reused capture code: #1768 inherited span-mean while its
headline comparison target #779 used last-token — the mismatch survived
the full critic ensemble and cost a ~15–18 GPU-h re-pool round. A missing
pooling-convention row or a silent mismatch is a plan-time REVISE
(Statistics lens item 15), never a sanity-gate footnote. Plans that fit no
map are covered by the same "N/A — no representation map fitted" line.

**Figures to produce (over-produce; ask only when the hero is ambiguous).** The plan names the specific hero figure(s) the headline needs AND a short exploratory dump the analyzer over-produces at the end (per-cell bars, per-seed scatter, per-step trajectory lines, raw-alongside-residualized). Default to over-producing exploratory views; the analyzer picks the hero from them rather than producing one figure and hoping it lands. When the view that best supports the headline is genuinely non-obvious, surface ONE plan-time question to the user about which view to feature.

## 6.5 Primary deliverable (the upstream completeness-vs-plan gate)

Name, per dependent variable, the **artifact path or glob the upload-verifier can enumerate on the pod** to confirm the run actually produced the Goal's primary measurement. This is the upstream complement to the downstream planned-vs-actual reporting discipline (`verify_task_body.py` check 11b + `clean-result-critic` Lens 13): catching a wholly-missing primary deliverable BEFORE the pod is terminated keeps the cheap-fix window (pod + per-step checkpoints still alive) open. Without it, a run whose headline phases silently no-op'd (missing input flags, an `if args.X and args.Y` guard fell through, a phase crashed mid-loop) passes Step 8 upload-verification — because every artifact that *was* produced has a URL — and is only caught at the clean-result write-up after the cheap-fix window has closed (#519: headline phases silently skipped at launch, the manifest recorded `skipped_phases: []`, the pod was terminated, and per-step checkpoints were lost).

Render as a fenced YAML block the upload-verifier and the orchestrator can both parse:

```yaml
primary_deliverable:
  - dv: <one-line name of the Goal-DV this artifact carries; mirror §6's Construct column verbatim>
    glob: <pod-side path or glob the verifier enumerates, e.g. eval_results/issue_<N>/headline_metrics.json or data/issue_<N>/activation_shift/*/results.json>
    note: <optional one-line note, e.g. "≥1 file per cell expected"; omit if not needed>
  # ... one row per primary DV the §6 evaluation table names
```

Rules:

- **One row per primary DV the §6 evaluation table names.** Secondary / exploratory artifacts (per-step trajectory logs, per-seed scratch, debug dumps) do NOT belong here — they keep the existing "ship-everything-via-§8-active-discovery" path. This section is exclusively for the artifacts whose absence would make the experiment Goal-incomplete.
- **The `glob` must be enumerable on the pod via `find` / `ls`** — EXCEPT a row wholly produced by a §9-declared off-pod phase, which is enumerated at that phase's declared dest (or deferred post-termination) per upload-verifier Step 2.7's declared-off-pod sub-rule (#1535). Hub URLs, WandB run paths, and committed-git paths do NOT belong here — those are downstream destinations the existing Step 8 rows + Step 2.5 phantom-URL gate already cover. This section is the on-pod source-of-truth glob the verifier inspects BEFORE artifacts move anywhere.
- **Mirror the DV name verbatim from §6** so the verifier's FAIL message names a DV a reader recognizes.
- **Exemption — `kind: analysis | infra | batch | survey`** tasks may write `primary_deliverable: []` (an empty list under the fenced block) with a one-line justification under it (e.g. "N/A — analysis task; no on-pod primary artifact"). The verifier WARNs (not FAILs) on a wholly-missing section so legacy plans drafted before this rule continue to ship.

The upload-verifier reads this block at Step 8 and, for every row, runs an on-pod `find <glob>` (or equivalent enumeration via `mcp__ssh__ssh_execute`); a row whose glob enumerates zero files FAILs the gate with blocker tag `primary-deliverable-missing` — EXCEPT a row wholly produced by a §9-declared off-pod phase, which is enumerated at that phase's declared dest (or deferred post-termination) per upload-verifier Step 2.7's declared-off-pod sub-rule (#1535). On that blocker SKILL.md Step 8 KEEPS THE POD ALIVE and auto-recovers — it loops back to the run phase to re-drive the missing deliverable on the still-alive pod (the /issue skill stays autonomous; only the generic `workflow.yaml § pivot_criteria` cap-3 path routes to `status:blocked` for this failure class). See `.claude/agents/upload-verifier.md` § Step 2.7 and `.claude/skills/issue/SKILL.md` Step 8.

## 7. Decision Gates

**Default to no gates.** Most experiments in this project are short enough
(<4 GPU-hours wall-clock) or test a pre-verified hypothesis where stopping
early just adds branching and incomplete data. Pilots, intermediate
checkpoints, and "stop if metric < X" gates have a real cost: they fragment
runs, complicate analysis, and bias toward early-noise interpretations. Do
NOT propose them reflexively.

**Only add a gate when ALL of:**
- The expected wall-clock is **>4 hours** (or GPU-hours >16), AND
- The hypothesis is **genuinely uncertain** — no prior issue / pilot has
  established the effect direction at this scale, AND
- A specific intermediate signal can cheaply rule out the full run (e.g.
  "if step-200 train loss > X, the run will not converge").

If those don't hold, write **"No gates — short run / pre-verified
hypothesis"** in this section and move on. The critic will not penalize the
absence of gates when this justification is given.

**If you do add gates, keep the set minimal and coherent.** The ALL-of bar
above licences the *decision to gate*, not a gate ladder. Prefer ONE
necessary kill-criterion over several; a four-rung smoke-gate stack
(Gate 1 / Gate 2′ / Gate 3 / Gate 4) is almost always a defect. For every
gate the plan retains:

- Give a one-line justification (what cheap intermediate signal does this
  gate use to rule out the full run) AND ground its threshold AND its SIGN
  in prior-issue evidence of the construct. A gate whose sign predicts the
  opposite of what every prior run of this construct produced, or whose
  threshold no past result of this construct would itself have passed, is
  a defect — it guarantees a false FAIL by construction.
- Self-check the whole gate set is **jointly satisfiable** before
  shipping: no two gates may demand contradictory outcomes (e.g. one
  requires `Δ ≥ +x` and another `Δ ≤ −y`) on the SAME measurement at the
  SAME cell / slot / probe target. Such a set guarantees a false FAIL —
  the run can never pass its own gates. (Surfaced after task #488: a
  smoke-gate ladder shipped Gate 3 requiring an off-diag cell marker
  log-prob change `≥ +0.2 nat` and Gate 4 requiring the same probe at the
  same cell `≤ −0.2 nat`; the contradiction was diagnosed only after
  multiple days of recipe-thrashing.)
- Apply each registered decision band to the plan's OWN cited precedent
  values: recompute the band predicate on those numbers and confirm the
  resulting branch matches the narrative label the plan gives that
  precedent; a band that classifies its supporting precedent into the
  OPPOSITE branch — or a precedent ratio range straddling the threshold
  while the prose asserts one side — mislabels the modal outcome by rule
  choice (#825 v17: the 0.5× band put the cited instruct precedent,
  0.3489/0.6731 = 0.519, in the REFRAMED branch while the prose used it as
  the specificity-upheld reference). `verify_plan.py` c27 WARNs on the
  mechanically-detectable subset; this self-check owns the rest.

The critic now REVISEs incoherent or ungrounded gate sets (`critic.md`
Statistics & Measurement lens item 3), so an over-laddered or contradictory
gate set will bounce the plan — sanity-check before shipping.

**Per-criterion §4-mechanism binding.** Every acceptance criterion in §6 / §7 names — in the same row / bullet — WHICH §4 mechanism produces the number the criterion reads, AND what the criterion COMPARES (count / equality / presence / set-difference / regex-match). Three columns per criterion: `Criterion` · `§4 mechanism` · `Compares-what`. The L602 Self-count rule (planner.md § Plan Quality) covers COUNT-style criteria only — it does not bind equality / presence / set-difference criteria to their §4 mechanisms, so those routinely ship unbound and get "measured by whatever the implementer thinks fits" at gate time. Worked example:

| Criterion | §4 mechanism | Compares-what |
|---|---|---|
| `wc -c file.md ≤ 40000` | Edit-set §4 total byte delta | Equality (≤ threshold) |
| `grep -c 'Foo' bar.md ≥ 1` | Edit N insert text | Presence (count ≥ 1) |
| `workflow_lint.py NEW violations == ∅` | Full edit set vs plan-time baseline | Set-difference vs baseline ledger |
| Reference-file §K added | Reference-file additions §K | Presence (heading count ≥ 1) |

Bind every criterion this way. A criterion whose §4 mechanism cannot be named is a criterion the design has not produced; back-fill §4 or drop the criterion.

## 9. Resources & Parallelism

GPU-hours, disk space, API costs, wall time. Be specific.

**Prioritize parallelism over sequential execution.** Wall-clock time is the
scarce resource — GPU-hours are not. If the workload can run faster on a
larger pod or split across multiple pods, the plan MUST take that path
(unless it would meaningfully hurt fidelity, e.g. a hyperparameter that
implicitly depends on world size). For each compute-bound step, identify the
parallelism axis and pick the spec accordingly:

| Axis | When it applies | Default action |
|---|---|---|
| **Tensor parallelism** | Generation/eval on ≥30B, or a 70B model | `inf-70b` (8× H100) or `ft-70b` (8× H200) — never run TP=1 on a 70B model |
| **Activation capture (HBM-bound)** | A 7B forward that captures hidden states — all-layer residual streams, Welford activation accumulation, per-token activation dumps | Pick an intent clearing ≥40 GB HBM: `lora-7b` (train + capture) or `capture-7b` (eval + capture, #752). NEVER the L4 `eval`/`debug` default — 7B bf16 weights (~14 GB) + captured activations OOM it (#666, #744). Size the activation footprint per the VM-footprint carve-out below if the capture also materializes a large store on the VM analysis side. |
| **Data parallelism (FSDP/ZeRO-3)** | Full fine-tune of a 7B+ model | `ft-7b` (4× H100) over `lora-7b` (1× H100) when fidelity permits |
| **Batched inference (vLLM)** | Eval/generation with K samples per prompt or N prompts | One pod with the largest sensible GPU count, single `LLM.generate()` call — never loop sequentially |
| **Sweep parallelism** | N independent conditions / seeds / models with no shared state | **MUST** default to one multi-GPU pod with `CUDA_VISIBLE_DEVICES`-sharded subprocesses when N seeds/conditions each need ≤1 GPU and fit on a single pod (e.g., 4 seeds × 1 GPU each on a 4× H100). Only provision N separate single-GPU pods when: (a) each seed requires >1 GPU (e.g., ZeRO-3), or (b) the plan explicitly justifies per-seed pods with a wall-time or isolation argument. Consistency-checker will WARN on plans that propose N single-GPU pods for N seeds without justification. (This is SIMULTANEOUS shared-nothing parallelism within ONE phase — orthogonal to per-phase GPU-width right-sizing below, which stops a NARROW phase from holding the WIDE phase's pod across a SEQUENCE of differently-sized phases; that rule is about phases of DIFFERENT widths run SEQUENTIALLY, NOT about splitting a single wide-parallel phase.) On the GCP lane, DECLARE the width: pass `--gpus N` (N ∈ {2,4,8}) so the width-aware auto router (#1121) walks wide `a2-ultragpu` rungs (8→4→2) first and degrades on capacity miss — wide GCP provisioning is the ENCOURAGED default whenever a shardable axis exists (credits effectively unconstrained; wall-clock is the scarce resource). The plan states the requested width + the shard axis; the workload re-shards off the realized width on the `epm:backend-selected` marker (a degraded launch may land narrower than requested). A workload that CANNOT re-shard off the realized width must PIN its width (explicit `--intent sweep-8g-a100`, or a stated abort-on-width-mismatch in the dispatcher) rather than ride the degrading auto walk. `spot_tolerant` is the lever for preemption-recoverable wide sweeps — classification at the requested width makes nearly every wide dispatch LONG (wall × 8 > the 2 GPU-h spot threshold), so without `spot_tolerant` no spot rung is walked. |
| **Pipeline parallelism** | A → B → C where B doesn't need all of A | State the dependency DAG and start independent branches concurrently |

State explicitly in the plan: (a) the GPU spec chosen, (b) the parallelism
axis it exploits, (c) the wall-time delta vs. the next-smaller spec, and (d)
any reason a smaller pod was chosen anyway (rare — e.g. "data is too small
to amortize 8× setup"). A stay-narrow reason must BIND to the wall-dominant
GPU-BOUND phase(s) it keeps narrow (name the phase; address ITS bottleneck)
— a bottleneck claim about a DIFFERENT phase (an API-bound judge, a CPU fit)
does not justify narrow width for the shardable phase(s) kept narrow (e.g.
generation/capture legs, a training fan-out) (#1739). If the answer is "no
parallelism axis applies," say so — silence is not acceptable.

> **Compute-sizing recipes (HBM sizing / merge-disk + ladder-checkpoint
> retention / Out-root mount binding / sentinel lanes / floor cross-check
> + external-stream
> presumption / fit-phase pilot basis / store-IO / CPU-phase RAM/RSS
> routing / machine costing + p90 fence sizing)** — when sizing any §9
> phase (activation capture, adapter merges, dose-ladder rung checkpoints,
> checkpoint/store out-roots, sentinel-signaling workloads,
> long-phase wall-time floors,
> store-heavy writes, VM-placed CPU-phase RAM, per-machine wall-time costing,
> a deliberate GCP fence), READ
> `.claude/rules/plan-compute-sizing.md` IN FULL before writing the table.
> (Relocated verbatim from this spec, #829.)

> **Phase placement + right-sizing (always-on in CLAUDE.md § "CPU-only phases
> don't hold GPU pods ...")** — the CPU-only OFF-POD default, the cheap
> dedicated-CPU-pod preference (#747), per-phase GPU-WIDTH right-sizing (incl.
> the explicit tradeoff weigh), the API-bound-judge-phase pod release, the
> compute-character carve-out (iterative-optimization fits are GPU-worthy;
> many-cell dense-factorization loops are vectorize-first too — vectorize
> first), and the data-footprint carve-out (>50 GB -> `cpu-bigmem`)
> are ALWAYS-ON in CLAUDE.md § Pods — apply each per phase here rather than
> restating them. (Deduplicated against CLAUDE.md, #829.)

**Required: per-component compute-projection table.** Every plan §9 for
`kind: experiment` tasks MUST include a per-component compute-projection
table (one row per compute-bound component). The implementer's
post-implementation `epm:compute-deviation v1` check (see
`experiment-implementer.md` mandatory checklist item 5) quotes
`planned_wall_h`, `projected_wall_h`, `ratio`, and the row's `basis`
string verbatim. The orchestrator's `pivot_criteria.compute_deviation_over_2x`
uses the `parallelism` field to compute auto-descope options.

| component | planned_wall_h | planned_gpu_h | parallelism | basis |
|---|---|---|---|---|
| (e.g., "smoke-phase per-cell train") | 0.5 | 0.5 | TP=1 | "matched to #382 round-2 trained-on-same-mix wall-time" |
| (e.g., "sweep all-cells train") | 16 | 64 | 4× H100 ZeRO-3 across 8 cells | "16h × 8 cells / 4 GPU = 32h wall; 16 GPU-hours × 8 = 128 GPU-h" |
| (e.g., "eval all-cells generation") | 2 | 2 | TP=1 | "vLLM batched, 400 prompts × 4 framings @ ~5s/prompt" |

**Per-phase shardable-axis declaration (REQUIRED, #1739).** For every
GPU-bound row with `planned_wall_h` > ~2 h, the `parallelism` field names
the shardable axis (contexts / behaviors / seeds / conditions / cells) or
states `none — <why>`. When an axis exists, the phase defaults to WIDE
(`--gpus N`, the #1121 width-aware walk); a stay-narrow choice carries a
justification binding to THAT phase per the "State explicitly" item (d)
above. For a row kept narrow, `planned_wall_h` is the serial-on-1× wall,
so the > ~2 h threshold coincides with the critic's ">~2 h serial on 1×"
bar (item 10(iv)) by construction. Short phases (< ~2 h), genuinely
non-shardable workloads, width-required pinned jobs, and the
re-provision-churn tradeoff for SHORT narrow phases remain valid
stay-narrow reasons.

**Serial-fit-loop, draw-battery & store-serialization sizing (REQUIRED
whenever any §9 component loops a fit / solve / factorization / draw — or a
per-item serialization / per-file upload — over cells, folds, layers, arms,
traits, seeds, draws, or output files).**
(a) Write the multiplier arithmetic EXPLICITLY in the row's basis:
`total_calls = draws × cells × folds × …` and
`projected_wall = total_calls × per_call_cost / parallelism`. A battery/null
row costed at 1× the grid forgets the draw multiplier (#810: 0.75 h planned vs
231 h projected — 308×; #778: "4 GPU-h" planned for a battery whose serial CPU
reality was ~15-30 h).
(b) Ground `per_call_cost` on a CITED measurement (a timed single call at
PRODUCTION shape, or a prior-issue measured figure) or a FLOP/kernel floor —
EXCEPT for a fit / factorization / GD loop — **or a permutation /
bootstrap / null-draw battery above the ~15–30 min phase floor** — where
the basis MUST be a measured 1-cell pilot at the phase's realized
execution shape (batch width / per-call structure — #1415; for a
battery: one production-shape batched draw block) THROUGH the
production entrypoint on
the production
device, a cited prior-issue measured figure (same kernel + shape), or a
pre-registered `pilot-gated` first-step pilot (inputs don't exist at plan
time) per `.claude/rules/plan-compute-sizing.md` § Per-cell fit phases; a
FLOP floor is the cross-check, never the basis, for these overhead-bound
loops (#811: one inner kernel timed, the dominant frame asserted — 19h21m at
unit 3/108) — NEVER an asserted "~2 s" for a dense factorization (#823: the
plan asserted
~2 s/fit; the named fast twin's own docstring said ~125 s at N_tr≈4000, H=3584
— a ~62× per-call error; the realized wall ran 35-57× the planned); a
`pilot-gated` battery row books ≥2× its naive wall/RSS projection in the §9
headline until the pilot lands (#1092: naive 5 h/box booked, 12.8 h/box
realized).
(c) If the task body's reuse map or a parent issue names a fast /
verified-equivalent helper for this loop, the §4 pseudocode USES it (import +
call named), or the plan states in one line why not — dropping the body-named
fast twin is the #823 root cause; the consistency-checker blocks an unstated
substitution.
(d) A row with `total_calls` >~500 or projected serial wall > 4 h triggers the
floor cross-check (`.claude/rules/plan-compute-sizing.md`) and the batching
requirement (`.claude/rules/vectorize-many-cell-fits.md`).
(e) For a store-heavy phase (>~10^3 output files or >~50 GB written), ground
`per_call_cost` on a MEASURED one-item serialization + upload wall-time at
production shape — bytes/file-counts alone are not a basis — and default
client-side compression OFF for fp16 tensors bound for a Xet-backed HF repo
(#813: `savez_compressed` 103.8 s vs plain `savez` 1.2 s per file for a
1.29× ratio; the store phase ran 4.5× over plan). Full recipe:
`.claude/rules/plan-compute-sizing.md` § Store-heavy / IO-heavy phase sizing.

**VM-RAM & GCP-fence sizing (REQUIRED for any VM-placed CPU phase and any
deliberate `max_run_duration`).** State each VM-placed CPU/analysis
phase's projected peak RSS in its row's `basis` (measured one-chunk
`ru_maxrss` at production shape, or resident-pool bytes × MEASURED
live-factor); ≥~16 GB — single phase or SUMMED concurrent VM phases —
routes off the shared VM to `cpu-mid`/`cpu-bigmem`, with `--min-ram-gb`
stated when the phase sizes >16 GB (#778: a 22-GiB-RSS battery
earlyoom-killed 3×; #833: two ~13-15 GB concurrent phases lost 5 cells).
Size any deliberate GCP `--max-run-duration` off the p90 per-cell wall —
a prior-issue distribution, else measured mean × a stated dispersion
factor (default ×2) — never the mean (#833 overran its 36h fence at ~2×
mean). Full recipes: `.claude/rules/plan-compute-sizing.md`
§ CPU-phase RAM/RSS routing + the fence clause of § Cost wall-time
against the machine the router will ACTUALLY provision.

**Stratification spec.** If the sweep has multiple statistical
dimensions (seeds, framings, cells-per-stratum), name in §9 the
priority order for auto-descope (e.g., "drop seeds first, then framings,
then cells-per-stratum") and the minimum-N per dimension to preserve
statistical power. The orchestrator's auto-descope walks dimensions in
that priority order; when none keep ratio ≤ 1.5× while staying above
the per-dimension min-N, it escalates via
`gates.conditional.compute_deviation_resolution` (after the
overhead-bound vectorize-first lever — workflow.yaml
§ pivot_criteria.compute_deviation_over_2x auto-action 0 — has run or
recorded a negative finding).

`kind: analysis | infra | batch | survey` plans are exempt from the
table (no GPU-bound components). For those, write "N/A — no
compute-bound components" and move on.

### Cross-phase reads declaration (`off_pod_phases:`) — reads enumerated, outputs pre-declared (#1535, #1773)

REQUIRED whenever ANY dispatched phase reads ANOTHER phase's outputs —
BOTH directions: (a) a pod/backend dispatch with ≥1 subsequent off-pod
phase (a VM / cpu-small / cpu-mid / cpu-bigmem / Batch-API judge or
analysis phase) consuming pod outputs — the original #1482/#1426
direction; AND (b) any pod-gpu / GCE / SLURM phase consuming another
phase's outputs, including VM-PRODUCED inputs — the #1773 inverse seam:
the git-clone lanes stage ONLY the pushed branch (`data/` is gitignored,
the #734/#1434 class), and the #1469 carry-over gate glob-skips
phase-output globs, so no mechanical gate covers that seam. Pod-free
plans and single-machine runs OMIT this block
entirely — no boilerplate, no escape line needed. Three incidents it closes:
#1482 (an off-pod judge died at VM launch loading pod-only scratch files
never in the upload set — the pod was already terminated), #1426 (a planned
VM-side phase FAILed the verifier BY CONSTRUCTION — outputs expected on the
pod; the follow-up round improvised the "DEFERRED + gap-listed" rows this
block mechanizes), and #1773 (the inverse direction: a GCE pass crashed
loading VM-produced outputs never HF-uploaded — one provision+boot cycle
burned). This is the plan-time mechanization of the
`gotchas.md` cross-machine upload-set bullet (#1526, rules (i)-(iv)).

Render as a fenced YAML block, one entry per phase that reads another
phase's outputs (the block NAME stays `off_pod_phases:` for back-compat
with in-flight plans and the c39 satisfier):

```yaml
off_pod_phases:
  - phase: <verbatim §9 phase name, e.g. "P5 judge (VM, post-termination)">
    runs_on: vm | cpu-small | cpu-mid | cpu-bigmem | batch-api | pod-gpu | gce | slurm
    reads:
      - path: <path the phase's loader opens/fetches>
        produced_by: <producing phase, e.g. "P4 (pod)">
        source: hf-data-repo | git-issue-branch | vm-resident-by-construction
    outputs:
      - path: <path/glob the phase writes, e.g. eval_results/issue_<N>/judge/*.json>
        dest: git-issue-branch | hf-data-repo | vm-working-tree
```

Rules:

- **Every `reads[].path` must resolve at a permanent source the CONSUMING
  machine can fetch** — an HF data-repo path, a git-issue-branch path, or
  `vm-resident-by-construction` with a one-line basis (e.g. "arrives with
  the git clone"). A read that exists only as one machine's local scratch
  (pod-side OR VM-side) is a design
  defect: add it to the producing phase's upload set (KB–tens-of-MB scratch
  metadata — split indices, provenance arrays, configs — uploads
  UNCONDITIONALLY; the large-tensor discard economy never applies to it).
- **`vm-resident-by-construction` is keyed on EXECUTION LOCUS, not the
  literal `runs_on: vm` enum value:** legal only for VM-EXECUTING phases —
  `vm`, and `batch-api` (a Batch-API judge's driver runs VM-side and
  legitimately opens VM-resident inputs; live precedent: #1776 plan v4's
  `runs_on: batch-api` rows with `source: vm-resident-by-construction`).
  ILLEGAL for dispatched git-clone / staged lanes (`pod-gpu | gce | slurm`
  and the `cpu-*` lanes) — those machines stage only the pushed branch, so
  a VM-resident file is simply not there; use `hf-data-repo` or
  `git-issue-branch`. Blocks in plans persisted before this change are
  never retro-invalidated (critic item 10 pressure is forward-only).
- **A VM-PRODUCED read consumed by a git-clone-lane phase (`pod-gpu | gce
  | slurm` — the #1773 direction) names BOTH transport halves:** the
  PRODUCING phase ends with a fail-loud bulk `upload_folder` of its
  outputs to the issue HF prefix, AND the CONSUMING launcher stages the
  missing inputs via scoped `list_repo_tree` + per-file download (never
  `snapshot_download` on the ~1M-file data repo — gotchas.md #833),
  logging a `[stage] <input> staged: N files` line usable as the
  crash-fix fix-engaged signal (mirrors the implementer-side memory
  `feedback_cross_machine_input_staging.md`).
- **`outputs[]` is scoped to files the declared off-pod phase ITSELF
  writes** — it must NOT sweep in any pod-side phase's deliverables (no
  over-broad globs like `eval_results/issue_<N>/**` when a pod phase also
  writes under that tree). Prefer the EXACT glob string of the matching
  §6.5 row (or an explicit back-reference to it). An over-broad output
  glob would wrongly divert a pod-side deliverable away from the #519
  pod-side gate — the verifier's tie-break (Step 2.7) fails toward the
  gate on any partial or uncertain coverage.
- **The declaration is what the upload-verifier consumes**: Step 2.8 gates
  every `reads[].path` at the cheap-fix window (pod still alive; FAIL with
  the exact upload command), and Step 2.7 enumerates a declared phase's
  OUTPUTS at the declared `dest` — or records them deferred when the phase
  is sequenced after pod termination — instead of FAILing on pod-absence.
- **Derive `reads[]` from EACH DECLARED PHASE'S loader ACTUAL open/fetch
  set** (its argument list + `open()` and staging calls), direction-agnostic,
  not from memory of the design — an omitted read reproduces #1482/#1773
  despite the block (the #1773 plan HAD a block; the intermediate
  passA→passB read was omitted because reads were derived only for the
  final off-pod loader); state the derivation basis in one line per phase
  when the loader exists at plan time.
- **§6.5 interaction:** a `primary_deliverable:` row produced by a declared
  off-pod phase is enumerated at that phase's declared `dest` (Step 2.7
  sub-rule), not on the pod; keep the §6.5 row — the block does not
  replace it.
- **Escape:** a plan whose prose trips the off-pod vocabulary check
  (`verify_plan.py` c39) but genuinely has no cross-phase read declares the
  standalone line `N/A — no off-pod phase`.
- **Known mechanical residual (direction (b)):** c39's trigger vocabulary
  (`off-pod` / `vm-side` / `vm-produced` / `produced on the vm`) fires on
  the calibrated inverse-direction tokens (#1796), but OTHER
  inverse-direction phrasings (a pod/GCE/SLURM phase reading VM-produced
  inputs described in different vocabulary) still do not fire, so that
  residual of direction (b) is enforced by this section + critic
  Methodology item 10 only.

Worked example (#1482's design, as it should have been declared):

```yaml
off_pod_phases:
  - phase: P5 judge (VM, after pod termination)
    runs_on: vm
    reads:
      - path: issue1482_sae_perfe/analysis_tensors/scratch/split_indices.npz
        produced_by: P4 (pod)
        source: hf-data-repo
      - path: eval_results/issue_1482/p4_summary.json
        produced_by: P4 (pod)
        source: git-issue-branch
    outputs:
      - path: eval_results/issue_1482/judge/*.json
        dest: git-issue-branch
```

Second worked example (#1773's design — the inverse direction, as it
should have been declared: Pass A produces on the VM, Pass B consumes on
a git-clone lane):

```yaml
off_pod_phases:
  - phase: Pass B scoring (GCE, consumes Pass A selection outputs)
    runs_on: gce
    reads:
      - path: issue1773_slug/selection/inverted_index.npz
        produced_by: Pass A selection (VM)
        source: hf-data-repo
    outputs:
      - path: eval_results/issue_1773/passB/*.json
        dest: git-issue-branch
```

## 10. Reproducibility Card (Pre-filled)

Pre-fill the Reproducibility Card template (from CLAUDE.md) with all KNOWN values. Mark TBD for values that depend on execution (wall time, GPU-hours, exact commit). The experimenter fills in TBDs after running. This ensures parameter choices are documented at PLAN TIME, not reconstructed after the fact.

**Cited HF reuse artifacts MUST be Hub-verified before they land here.** Any
entry in this card that names a reused HF artifact (LoRA adapter, merged
model, dataset, raw-completion bucket — by repo id + subfolder/path) must
have passed the `huggingface_hub.list_repo_files` existence check from
step 5 ("Check what's reusable") — the expected files (e.g.
`adapter_config.json` + `adapter_model.safetensors` for an adapter,
`config.json` + weights for a merged model, the exact JSONL path for a
dataset) must actually resolve at the cited path. An unverified artifact
does NOT appear here as a confirmed reuse — either re-cite the correct
location after re-verifying, or move it to §12 Assumptions flagged
`must-rebuild`. Do NOT use the `hf` CLI for this check (see step 5 + 
`.claude/rules/upload-policy.md`: the installed `hf` has no `api`
subcommand and returns a false "0 files" via swallowed stderr).

**Reused code/helper throughput inspection (checklist item (i)) is recorded
here too.** When the plan reuses a parent's fit / analysis / eval /
upload-verify-staging code helper,
this card carries the item-(i) inspection record — helper/function name,
batched-or-serial verdict, device handling, plus the Hub-call-scoping verdict
when the helper touches the Hub (`.claude/rules/artifact-reuse.md`
item (i)) — and the implied wall-time is reflected in the matching §9 row.
"N/A — no artifact reuse" does NOT cover reused fit/analysis/upload-verify
code: a plan with
no HF artifact reuse but with an inherited fit helper still fills this row.

**Pairwise provenance coherence (checklist item (j)) is recorded here too.**
When the plan reuses a mutually-dependent artifact PAIR, this card carries the
item-(j) attestation — the member dates at the consumed revisions (per-repo
`get_paths_info(..., expand=True, revision=...)`; git members via
`git log -1`: consumed input vs dependent capture) and the coherence verdict;
an input postdating its capture is a failed check regardless of sha pins
(#922).

**Parent-lineage coherence (checklist item (k)) is recorded here too.** When
the plan reuses a parent's main-resident CODE module or a parent-realized
artifact with a declared input corpus, this card carries the item-(k) record —
the `git log --oneline origin/main..origin/issue-<M> -- <module>` outcome
(empty, or each unmerged commit ported / declared not-needed by SHA) and the
realized-vs-corpus count reconciliation (equal, or the named filter explaining
the shortfall) (#1345).

**Validity-domain transfer (checklist item (l)) is recorded here too.** When
the plan reuses a fit/analysis INSTRUMENT on a data regime that differs from
the parent's (subsets, judge-filters, a different n-vs-d balance), this card
carries the item-(l) validity-domain record — the boundary the instrument's
own docs declare (docstring / comment / module constant, line-cited), the new
regime read against it (e.g. per-fold n_train vs d), and the engaged
mitigation or the stated justification for not engaging it. Escape:
`N/A — no fit/analysis instrument reused` or `no declared validity
boundaries`.

**Output-artifact declaration + `discarded_artifacts:` slot
(persist-by-default).** Per generating / reducing stage, the
Reproducibility Card names WHERE each produced artifact persists: model
generations / rollout text → `raw_completions/<stage>/`; intermediate
tensors the plan or a foreseeable sibling consumes → `analysis_tensors/`
(upload-if-cheap-else-note-regen). This holds REGARDLESS of whether the
CURRENT task consumes the artifact (CLAUDE.md § Upload Policy
persist-by-default — a sibling / follow-up may). A DELIBERATE discard is
declared in a `discarded_artifacts:` frontmatter list the upload-verifier
reads:

```yaml
discarded_artifacts:
  - name: <artifact, e.g. extraction per-context v(x)>
    reason: <why dropped, e.g. full-corpus activation grid exceeds HF/LFS headroom (#541)>
    regen_recipe: <how to reconstruct, e.g. teacher-forced forward pass over the persisted raw_completions/extraction/ rollouts — one forward, no re-sampling>
```

Text / JSON (rollout text, judge outputs, metrics, configs) is NEVER a
valid discard — it rides the non-LFS path and uploads unconditionally
(>9.5 MB text line-splits into <9 MB shards, never gzip — the Hub
force-routes >10 MB blobs to LFS); only a genuinely too-big TENSOR is a
candidate, and only when its regenerating TEXT is persisted. The
upload-verifier treats a `discarded_artifacts:` entry naming generations /
text as INVALID (FAIL `generation-discard-declared-invalid`), not as a
license. If the run has no deliberate discard, omit the slot (or write
`discarded_artifacts: []`).

**Ephemeral-lane text/JSON destinations (destination-vs-lane durability).**
For any stage whose §9 lane is EPHEMERAL — a GCE instance with
`--instance-termination-action=DELETE` (the boot disk dies with the run)
or a RunPod pod on the terminate-on-upload-verify lifecycle; SLURM lanes
are deliberately NOT in this set (job scratch/project storage persists
past job end) — every text/JSON output row (summary JSONs, metrics,
judge outputs, configs) MUST name an HF (non-LFS) destination, e.g. the
issue data-repo prefix `issue<N>_<slug>/…`. A git-only destination
("commit to the issue branch") is legal ONLY for a VM-resident stage, or
when the plan names an explicit pre-teardown HARVEST phase that commits
the file BEFORE the instance/pod is reaped. Rationale: a clean exit on
the DELETE-on-exit lane reaps the disk minutes later — #1738's two
summary JSONs, declared "→ git issue branch" with no harvest phase, were
lost at reap and cost a 28-min rebuild round. The critic Methodology
lens item 18 REVISEs violations.

## 11. Decision Rationale

For every non-obvious parameter choice — and for EVERY load-bearing
hyperparameter without exception (lr + schedule + warmup, batch / grad-accum,
epochs, LoRA rank / alpha / dropout, weight decay, seq length, optimizer,
precision, anything novel) — document:
- **What:** The choice made (e.g., "lr=2e-5")
- **Why:** The reasoning, tied to this experiment's Goal (e.g., "matched to Tulu 3 SFT recipe; pilot at 5e-5 diverged")
- **Source:** Where the value comes from — an arXiv id / link to the specific
  paper table you read it from, or a prior issue `#<M>` that validated it for
  this model + data. Write `ungrounded — needs smoke-test` (never blank) when
  you could not find grounding; the fact-checker and critic key off this line.
- **Alternatives:** What was considered and rejected (e.g., "1e-4 too aggressive for 7B full finetune per prior OOM")

This section is the contract the fact-checker and critic verify: every
load-bearing hyperparameter must appear here with a non-empty `Source:` line.
(`kind: analysis | infra | batch | survey` tasks train no model — write "N/A —
no model training" and skip this section.)

**Marker recipe overrides parent parity.** For any FRESH marker /
behavior-implant training, the stopping recipe (lr, epochs / steps, checkpoint
selection / band-stop) is grounded in `.claude/rules/marker-training-recipe.md`
(lr ≤5e-6 clean window; log-prob band-stop gated on bystander resolution) —
NEVER in a non-marker parent's recipe via the single-variable contract.
"Parity with #<M>" is not a valid `Source:` for a marker-payload stopping
recipe when #<M> implanted a different payload (sycophancy, a trait, a fact)
under a different loss shape: marker-only loss has no countervailing loss
term, so a recipe that was safe for the parent saturates the marker. Name the
parity break in §12 Assumptions as a measurement-validity deviation; comparison
parity with the parent lives on the DV / eval side, not the training-stop side.
(#480: the plan grounded lr=1e-5 in "#411 parity" and explicitly rejected
lr=5e-6 as "breaks #411 parity"; all 6 marker adapters saturated and the fix
was a full band-stopped retrain.)

NOTE — large sweeps: the contract is one `Source:` per *unique* hyperparameter
value, NOT per condition. Group conditions that share a recipe, e.g. "All
conditions use the Tulu 3 SFT recipe (`Source: #382`): lr=2e-5, cosine warmup
0.03, 3 epochs. Three conditions vary learning rate only: 1e-5 / 2e-5 / 5e-5
(`Source: #382 round-2`)." This keeps §11 compact while preserving full
traceability.

**Reused input-data artifacts get a `Source:` line too.** Every REUSED
INPUT-DATA artifact the design loads — a parent's `train/*.jsonl` mix, an
on-policy response cache, an `eval_results/` JSON consumed as a downstream
input — gets a §11 `Source:` line naming (a) the producing issue `#<M>`, AND
(b) HOW the file is FETCHED on the target backend named in §9: an HF repo path
the worker stages (scoped `list_repo_tree` + per-file `hf_hub_download` for
data-repo subtrees — gotchas.md), a committed `eval_results/...` path that
arrives with the git clone, or "rebuilt on-worker by the §4 regen phase". This
is the §11 record of the step-5 check `(h)` (source resolution +
target-backend fetchability + staged-layout consumer-open): a git-clone-only
GCP/SLURM lane cannot stage a
VM-local-only mix, so "the parent built it locally" is NOT a valid `Source:`
— the file must resolve on HF, be git-tree-reachable as a committed
`eval_results/...` JSON, or be regenerated on the worker (#734). When a
stage-from-Hub helper maps the artifact into a consumer-fixed local layout
(leg (iv) staged-layout consumer-open), the §11 record ALSO names the
hub-rel → local-rel mapping, and the plan schedules the 1-file staging
probe + consumer-open gate before production, once per
(source-family × staged consumer) pair (#928, #1481).

**Repo-new model id ⇒ CPU-side config-load smoke before provisioning
(pre-provision gate).** The `model` id is itself a load-bearing choice. If
the plan's `model` is NOT already used by an existing entry under `configs/`
or by a prior issue in this repo (grep `configs/` + `tasks/` for the exact
id before claiming "new"), the plan MUST record a CPU-side
`AutoConfig.from_pretrained("<model_id>", trust_remote_code=...)` smoke as a
pre-provision gate — does the installed `transformers` recognize the
`model_type`, and does the repo resolve on the HF Hub? Quote the smoke
command and its PASS output (or the runnable command + a note that it will
run before the experimenter provisions) in §11 alongside the `Source:` line
for the model id. **Do not provision a multi-GPU pod on an unvalidated
repo-new model id.** The `AutoConfig` call costs nothing — it streams a few
KB of `config.json`, instantiates no weights, and surfaces both the
"unknown model_type" failure (your installed `transformers` is too old for
the architecture) and the "repo does not exist / typo in id" failure on
CPU, before the pod is created. Catching this at preflight on the pod is
too late: the multi-GPU provision has already happened. A repo-already-used
model id inherits its validation from the prior config/issue and does NOT
need a fresh smoke — cite the inheriting `Source:` as usual. (#506:
`Qwen/Qwen3.5-27B` passed 4 code-review rounds + cap-3 override, provisioned
an 8× H200, then died at launch because `transformers` did not recognize
the `model_type`.)

### Tool-behavior grounding (extended from hyperparameters)

The `Source:` bar extends beyond hyperparameters to **tool-behavior claims** — any assertion in the plan about what a repo script / lint / CLI / helper / marker-post / verifier DOES. When the plan says "workflow_lint's no-flags run enforces X", "task.py post-marker rejects Y", "verify_plan.py check cN WARNs on Z", "the janitor cron reaps W", each such claim carries a `Source:` naming the grep or `file:line` READ at plan time — not an assertion from memory. Worked examples:

- **Correct (grounded):** `Source: grep -n 'def cmd_post_marker' scripts/task.py → 1289:def cmd_post_marker` — the plan cites the exact function definition it relies on.
- **Correct (grounded):** `Source: grep -n 'AGENT_SPEC_FAIL_BYTES' scripts/workflow_lint.py → 11865:AGENT_SPEC_FAIL_BYTES = 40_000` — the plan cites the exact constant + value driving a byte-cap acceptance criterion.
- **Correct (ungrounded, honest):** `Source: ungrounded — verify at implementation` — the plan admits the claim was not grep-verified at plan time; the implementer runs the grep before relying on it.
- **Wrong (silent-memory claim):** `The workflow_lint no-flags run enforces the byte cap.` — no `Source:` → the critic REVISEs it (Methodology lens item 4). "Everyone knows" is not a source.

Fact-checker (Phase 1.5) verifies each tool-behavior `Source:` the same way it verifies each hyperparameter `Source:` — open the cited file at the cited line, confirm the behavior claimed. An `ungrounded` mark is honest but shifts the verification burden to the implementer; the critic REVISEs an `ungrounded` claim only when the tool behavior is both un-grep-verified AND plausibly outcome-changing.

## 12. Assumptions

Full template + worked examples referenced by planner.md § 12. Each assumption row is: **Assumption / Confidence / Source / How to verify**. Three sub-rules bind every §12:

### Line-number assumptions quote `grep -n` output VERBATIM

Assumptions about "line N of file X" are FRAGILE — line numbers shift on every neighboring edit. When a plan cites a line number, it MUST quote the exact `grep -n` output (number + text as printed), not a bare "at line 142". The verbatim line self-verifies across edits: even if the number drifts by ±5 lines, the text still grep-matches, and the fact-checker / implementer re-runs the same grep to relocate. Worked examples:

- **Correct:** `A3. planner.md §11 sits at L511. Source: grep -n '^### 11\. Decision Rationale' .claude/agents/planner.md → 511:### 11. Decision Rationale. How to verify: re-run the grep.`
- **Correct:** `A10. Self-count rule is at planner.md L602. Source: grep -n 'Self-count every count-style' .claude/agents/planner.md → 602:- **Self-count every count-style mechanical acceptance criterion.** Before. How to verify: re-run the grep.`
- **Wrong:** `A5. §12 sits at line 525.` — bare line number, no `grep -n` output → drifts silently, unverifiable across edits.

Rationale (#1721): a plan that says "the check at line 142 does Y" against a codebase where the check has already moved to line 156 silently confuses the reviewer + implementer; the verbatim grep line rescues the intent regardless of drift.

### Detection / trigger-lane predicate plans — trace the predicate

Copied by reference from planner.md § 12 (the always-on paragraph): when the plan designs or modifies a predicate that classifies a persisted artifact's shape to decide an automated action (watcher fire/keep, guard block/allow, janitor reap, failure classifier class) AND the motivating incident left a persisted artifact, §12 MUST carry one row per predicate arm — including the read/ingest path that feeds it — traced against the actual artifact by path, with each arm evaluated on values MEASURED from it at plan time (row counts, byte sizes, field values — READ, never recalled), and the traced outcome stated. The predicate MUST fire on its own motivating incident; "would not fire" is a design defect to fix before returning the plan. Artifact aged off disk → trace the incident's recorded measurements at Medium confidence; prospective guard with no incident artifact → state that in the row.

### Real-corpus structural assumptions — smoke-slice probe routing

**Trigger (all three conjuncts):** a §12 row that (a) asserts a STRUCTURAL property of a real corpus / dataset / reused artifact — distinct-value counts, field cardinality, per-row uniqueness, template homogeneity, schema/field presence — AND (b) gates an arm / fit / phase via a fail-loud check in the design (an assert / raise the production run executes), AND (c) is only checkable against the data itself, first materialized at smoke time (plan-time grounding — the fact-checker, a grep, a prior body — structurally cannot reach it).

**The duty:** the row's **How to verify** routes to a NAMED smoke-slice probe at **full-CONSUMED-corpus grain** — the property is read over the exact pinned data the production arm actually loads, NEVER the sliced smoke sample alone (a tiny sample can satisfy a premise the full corpus violates), and NEVER the upstream/streaming source (a full-LMSYS scan at smoke time is prohibitive — the #1092 streaming class; the consumed corpus is the object the production assert binds to). Read the property from a manifest/metadata field WHEN one carries it (the cheap 1-row read); when no manifest field carries the property, COMPUTE it from the consumed corpus — manifest presence is not a precondition for the probe.

**Implementer hand-off:** the implementer runs the named probe during smoke and records the MEASURED value in the relevant `### <phase-name>` sub-section under `## Smoke run` (experiment-implementer.md checklist item 3 carries the mirror duty).

**Violated-premise disposition:** a measured violation is a PLAN defect, not a code bug — surface it and bounce to plan amendment / re-scope BEFORE production; leaving it to the production assert is the failure mode this sub-rule closes (#1768: plan assumption A4, `n_distinct_prefix==1`, was hard-asserted in the capture script and false on the real corpus — measured 2, an LMSYS-vs-WildChat template split; the smoke run PASSed on its tiny sample and the p1 pilot assert fired ~55 min after relaunch).

Worked example pair:

- **Correct:** `A4. Every corpus row shares one prompt template (n_distinct_prefix==1). Confidence: Low. Source: guessed from the #779 pinned subset. How to verify: SMOKE-SLICE PROBE — p0 reads n_distinct_prefix from the full consumed-corpus manifest and reports the measured value in ## Smoke run; the p1 prefix-arm gate re-asserts it.`
- **Wrong:** the same row with `How to verify: the p1 assert will catch it` — that IS the production crash (#1768).

Plan-time critics reviewing §12 check that every row matching the three-conjunct trigger carries the smoke-slice probe routing (prose-only enforcement, the #1287 precedent).

### General shape

Every §12 row has these four fields at minimum:

- **Assumption:** a single factual claim, one row per claim.
- **Confidence:** High / Medium / Low. High = grep-verified or read from code at plan time. Medium = read from a recorded artifact (prior body, mentor-update, sibling issue). Low = guessed / recalled / not verified.
- **Source:** where the assumption came from — a grep line (High), an artifact path + field (Medium), or "recalled / guessed" (Low). NEVER blank.
- **How to verify:** the exact command / read the fact-checker or implementer runs to confirm the assumption. Prefer a re-runnable `grep -n` / `wc -c` / `git log` / `HfApi().file_exists(...)` invocation.

An ungrounded (Low-confidence) row is honest, not a defect — it flags a smoke-test target for the implementer. What IS a defect: a bare unmarked claim, or a High-confidence row whose `Source:` is not actually a verifiable READ (a paraphrase, a memory, a "the docs say"). Wrong assumptions are the #1 cause of wasted GPU time — over-list before under-listing.


