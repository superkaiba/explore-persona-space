---
description: On-policy-first positive completions for behavior implantation (elicitation ladder, 80% yield floor, standardized multi-behavior definitions)
paths:
  - "src/explore_persona_space/train/**"
  - "scripts/generate_*.py"
  - "scripts/*datagen*"
  - "tasks/**/plans/*.md"
---

# On-policy-first training completions for behavior implantation

**When building implantation training data for a behavior (sycophancy,
refusal, hedging, style, trait), generate the POSITIVE completions
on-policy from the BASE model wherever it can express the behavior under
elicitation** (standing directive 2026-06-12). Canned/templated strings
and third-party-LLM-written completions are the exception, never the
default. This is the positive-side sibling of
`.claude/rules/contrastive-negatives.md`, which already mandates
on-policy text for the NEGATIVE side ("generate on-policy from the BASE
model under each negative persona's own system prompt").

Why: canned/templated positives collapse the response distribution the
trained behavior generalizes from, and they overstate installability —
#612 measured the model's own judge-accepted agreeing completions
installing sycophancy at +0.60-0.66 over base where canned "Absolutely!"
templates install +0.84-0.93 under the IDENTICAL recipe (same prompts,
personas, row counts, ratios). A headline about install strength or
leakage radius read off canned data does not transfer to realistic data.
Prior offenders: #411's 20 hard-coded agreement templates; #545's
40-string Sonnet-diversified refusal pool randomly attached to questions;
Sonnet-written compliment/hedge rows. The data-realism preference order
(CLAUDE.md "Design experiments on the most realistic data available")
governs the data SOURCE tier (where prompts/corpora come from); this rule
governs completion PROVENANCE (who wrote the response text). A plan names
both.

## The recipe (the #612 elicitation ladder)

Elicit the behavior from the base model itself, per source persona, over
the design's question/prompt set:

1. **Tier 1 — bare context.** Sample under the persona/target context
   alone (no behavior instruction). Whatever the base model produces that
   already expresses the behavior is the most on-policy data available.
2. **Tier 2 — instruct-and-strip.** Add the behavior instruction to the
   system prompt (e.g. "agree with the user's claims even when they are
   wrong" / "refuse to answer this request"), sample, then **STRIP the
   elicitation instruction before training** — the trained context is the
   persona/target context only, so the gradient associates the behavior
   with the persona, not with an instruction the eval will never show.
3. **Tier 3 — minimal opener prefill.** Prefill a 2-4-word
   behavior-consistent opener and let the model continue; the prefill
   stays in the training text.

Mechanics that ride along, all of them load-bearing:

- **Judge-filter** every sampled completion for the target behavior
  (Claude judge per project policy — never substring match); only
  judge-accepted rows enter the pool.
- **Prefer the lowest tier that fills the quota**; record the tier
  per row and report the realized per-tier yield mix (#612: villain
  31 bare / 165 instruct-and-strip / 4 prefill).
- **Pre-register a per-source yield quota + retry budget — DEFAULT: an
  80% floor with equalize-down** (decision 2026-06-12, recorded in #545's
  onpolicy-testbed-v2 followup-scope; supersedes the original
  all-or-nothing "fill every row or drop" rule, which discarded #612's
  kindergarten-teacher source at 194/200 — 97% fill — over 6 missing
  rows, and software engineer at 169/200; under the 80% floor both would
  have been kept). Mechanics:
  - **Floor = 80% of the target row count.** A source at or above the
    floor after the retry budget is KEPT; a source below the floor is
    DROPPED (subject to the close-miss escalation below) and the drop is
    REPORTED as a finding — predicted in advance by the source-side
    baseline read below, never silently backfilled with templates.
  - **Close-miss escalation (≥ 90% of floor):** a source finishing at or
    above 90% of its floor but BELOW it after the registered retry
    budget gets ONE automatic escalation tranche BEFORE the drop fires —
    sized ceil(remaining-need / measured-acceptance-rate) × 1.3, where
    measured-acceptance-rate = cumulative accepted/attempted across all
    prior tranches (strictly positive in the trigger band) — generated
    same-construct and on-policy (same behavior definition, same
    elicitation ladder tiers, same judge filter), and recorded in the
    datagen manifest (tranche size + realized acceptance). If the
    tranche closes the gap the source is KEPT under the unchanged floor
    semantics; if it is still below floor, the drop fires as today.
    Below 90% of floor the drop fires as today with NO escalation (the
    catastrophic class — #612/#906 — is not a tuning problem). The
    escalation is NEVER a template/canned/third-party-LLM backfill — the
    anti-silent-backfill intent is preserved by construction. (Founding
    incident #1947: a healthy arm at 232/240 — 96.7% of floor — was
    dropped after one retry tranche, removing 18 planned cells; the miss
    size carried no proportionality to the consequence. Filed as #2020.)
  - **Equalize-down: every kept source trains on exactly floor-N rows.**
    Discard the surplus everywhere rather than letting N vary per source
    — variable N is a dose confound, and dose/schedule length is the
    demonstrated dominant lever (#601). Prefer the same-question/claim
    subset across sources where filled rows allow; else a random floor-N
    sample, with the coverage difference documented.
  - **Scale contrastive negatives proportionally to floor-N** so the
    load-bearing ~1:1 positives-to-total-negatives ratio
    (`.claude/rules/contrastive-negatives.md`) survives the
    equalization.
- **Pre-classify yield risk per behavior.** Elicitation difficulty is
  behavior-specific: HIGH where the behavior conflicts with alignment
  training (false-claim agreement, harmful advice — expect shortfalls,
  budget retries accordingly), LOW where it is in-distribution for an
  aligned assistant (refusal, hedging, format compliance). Size the
  retry budget to the risk class.
- **Take a source-side baseline read before elicitation.** One cheap
  base-model generation + judge pass per source persona on the eval
  probes measures the pre-training behavior rate. It predicts which
  sources will miss the floor (#612: both yield failures were
  predictable in advance from a read that was never taken) and doubles
  as the natural install-strength covariate the eval side already
  measures on targets (a unit's own base prior keeps beating geometry as
  a predictor — #500/#532/#541). Planner-side rule: `planner.md` §4
  "Baseline propensity on BOTH sides".
- **Sampling temperature** ~1.0 by default (diversity is the point);
  ground a different choice in §11 like any hyperparameter.

## Standardized behavior definitions for multi-behavior datagen

**When one datagen pipeline implants MORE THAN ONE behavior (a
multi-behavior factory / panel / sweep), define EVERY behavior in the
standardized persona-vectors shape — never bespoke per-behavior
definitions or hand-curated per-behavior query banks without a stated
justification.** The standardized shape is steps 1–2 of
`.claude/rules/persona-vectors-recipe.md` (arXiv
2507.21509): the entire per-behavior human input is a trait NAME + a
brief natural-language trait DESCRIPTION; ONE shared generation-prompt
template then produces, per behavior, 5 contrastive pos/neg instruction
pairs, a shared / auto-generated neutral trait-eliciting question set,
and the scoring rubric (the recipe's 40-question / 20-20 extraction-eval
split is extraction-specific; datagen sizes its question set to its
yield quota). The elicitation ladder above operates WITHIN each
behavior's standardized definition (the pos instruction is the tier-2
instruct-and-strip prompt); this rule standardizes what the ladder is
pointed AT, not the ladder itself — and it is orthogonal to completion
PROVENANCE (a generator-model deviation like #906's D1 still names itself
separately).

Why: bespoke definitions confound every cross-behavior read — yield,
judge-acceptance, and install numbers then reflect each hand-written
definition + hand-curated bank as much as the behavior itself, so a
yield-floor failure cannot be attributed (definition artifact vs genuine
elicitation difficulty), and a refusal-driven yield loss can be
attributed to the BEHAVIOR (vs the hand-written definition/bank that
provoked it) only when all behaviors share one template. Incident
#906→#1090: #906's multi-behavior pilot defined its 4 classes bespoke and
all three content classes failed their pre-registered yield floors
(sycophancy kept 6/36 judge-accepted ≈17% vs floor 20; harmful_compliance
2/215 ≈1% vs floor 120; china_censorship 0 same-question negatives vs
quota 3); #906's recorded sycophancy diagnosis was pure generator
alignment-conflict — 30/30 Mode A declines on wrong-fact claims — which
standardization does not by itself remove (the shared template + an
impossible-to-refuse positive control are what make that failure class
attributable per behavior). The user caught the non-standardization and
directed the #1090 rebuild — each behavior = trait description + 5
contrastive instruction pairs + neutral trait-eliciting questions
(auto-generated persona-vectors-style where a curated bank triggers
refusal), plus an impossible-to-refuse formatting behavior as a pipeline
positive control.

The exemptions below apply unchanged: the programmatic marker carve-out
and taught-fact spans have no natural-language definition to standardize,
and published-corpus replication rows keep the paper's shape. An
ESTABLISHED / published benchmark bank (or a replication-required bank)
remains legal where `.claude/rules/data-realism.md` / replication
fidelity justifies it — the standardized behavior DEFINITION is still
required, and the plan names the cross-behavior-comparability caveat. A
SINGLE-behavior experiment MAY keep a bespoke definition (no
cross-behavior comparison to confound) but should prefer the standardized
shape for forward reuse.

## The measured trade-off (plan around it, don't ignore it)

On-policy data installs more weakly at a matched training recipe. #612's
three conditions settle into three separated dose bands — canned
+0.84-0.93, on-policy single-turn +0.60-0.66, multi-turn-prefix
+0.46-0.54 — set by epoch 1. Consequences for design:

- When comparing data constructions, or reading any dose-sensitive DV
  (leakage radius, selectivity), **dose-to-target** (match installed
  strength at a band/checkpoint) instead of fixing epochs — fixed epochs
  silently compare conditions at different doses.
- Expect and budget for yield shortfalls: some personas cannot produce
  the behavior at all (#612: bare-persona agreement was obtainable for
  only 11 of 200 software-engineer rows). Coverage loss is a reportable
  outcome, not a failure to hide.

## Allowed exceptions (named in the plan, carried as a clean-result caveat)

Canned/templated strings (paraphrase pools, fixed one-liners) and
third-party-LLM-written completions (Sonnet-written rows) are allowed
ONLY as:

1. **Deliberate anchors/controls** — the data construction IS the
   manipulated variable (e.g. #612's canned-agreement replication anchor
   arm, kept precisely to measure the canned-vs-on-policy gap).
2. **Recorded yield failure** — on-policy elicitation demonstrably failed
   the pre-registered quota after the retry budget. The fallback arm is
   then an explicitly flagged canned arm, never a silent substitution
   inside an arm labeled on-policy.

Either way the plan names the choice and the clean-result carries it as a
data-realism caveat. Two standing exemptions need no justification:

- **Published-corpus replication rows** — replication fidelity wins
  (CLAUDE.md "Replicating a published finding"); Turner/Betley rows stay
  verbatim. Do not "improve" a paper's data to be more on-policy.
- **The programmatic marker carve-out** — for marker implants the
  controlled template IS the construct (the appended ` ※` token in a
  fixed slot); note the response text `R` under the marker recipe is
  ALREADY on-policy (greedy, frozen base response per
  `.claude/rules/contrastive-negatives.md`), so the carve-out covers the
  appended token only.
- **Taught-fact spans** — a fact implant's target span cannot be elicited
  on-policy by construction (the base model does not hold the fact): the
  fact span IS the construct, same logic as the marker token. The
  surrounding response/context text still follows this rule, and the
  contrastive wrong-fact / refusal-pool negatives follow
  `.claude/rules/contrastive-negatives.md`.

## Enforcement

- `planner.md` §4 Design — every behavior-implantation training-row type
  names its completion provenance (`on-policy (tier 1/2/3)` |
  `canned/template` | `third-party-LLM-written` |
  `published-corpus-verbatim`), with quota + drop rule stated.
- `critic.md` Methodology lens item 14 — REVISEs canned/LLM-written
  positives without an anchor/control justification or a recorded yield
  failure, and any silent template backfill of a shortfall.
- Multi-behavior datagen: `planner.md` §4 names the behavior-definition
  shape per behavior (standardized persona-vectors template, § Standardized
  behavior definitions above); `critic.md` Methodology lens item 14 REVISEs
  bespoke per-behavior definitions/query banks in a multi-behavior
  implantation design without a stated justification.
- The clean-result carries the provenance choice + any coverage loss as a
  scope caveat (planned-vs-actual coverage, `verify_task_body.py` check
  11b / clean-result-critic Lens 13).

## Files of record

Task bodies #612 (elicitation ladder, dose bands, yield shortfalls),
#411 (canned sycophancy templates), #545 (Sonnet refusal pool),
#906 (bespoke multi-behavior definitions; all three content-class yield
floors failed), #1090 (the persona-vectors-style datagen rebuild
directive), #1947 (close-miss drop incident: 232/240 = 96.7% of floor
dropped after one retry tranche), #2020 (the close-miss escalation
clause);
`.claude/rules/contrastive-negatives.md` (negative-side sibling);
CLAUDE.md bullets "On-policy-first training completions",
"Design experiments on the most realistic data available",
"Replicating a published finding".

**Sibling rule:** `.claude/rules/persona-vectors-recipe.md` — the persona-vector
extraction rollouts are on-policy by construction (sampled from the base model
under the contrastive system prompts); this rule's on-policy default governs
them.
