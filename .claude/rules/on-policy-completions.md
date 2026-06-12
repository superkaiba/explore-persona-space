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
    DROPPED and the drop is REPORTED as a finding — predicted in advance
    by the source-side baseline read below, never silently backfilled
    with templates.
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
- The clean-result carries the provenance choice + any coverage loss as a
  scope caveat (planned-vs-actual coverage, `verify_task_body.py` check
  11b / clean-result-critic Lens 13).

## Files of record

Task bodies #612 (elicitation ladder, dose bands, yield shortfalls),
#411 (canned sycophancy templates), #545 (Sonnet refusal pool);
`.claude/rules/contrastive-negatives.md` (negative-side sibling);
CLAUDE.md bullets "On-policy-first training completions",
"Design experiments on the most realistic data available",
"Replicating a published finding".
