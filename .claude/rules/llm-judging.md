# LLM judging of behavior-expression DVs

**Load this rule whenever a plan or code DESIGNS or WRITES an LLM-judged
behavior-expression dependent variable** (sycophancy, refusal, hedging,
style, trait, alignment/EM verdict — any DV where an LLM scores whether a
model output expresses a target behavior). The always-on backstop is the
CLAUDE.md "Measurement validity" bullet + the "LLM judge =
`claude-sonnet-4-5-20250929`" bullet; this file is the full 23-guideline
recipe those bullets point at (the 21 judge-scoring guidelines plus the
two-rule § E2 continuous-companion-DV recipe), in the on-demand register.

Cross-links: `.claude/rules/persona-vectors-recipe.md` (the graded-judge
precedent — its rubric is a 0–100 trait score), CLAUDE.md § Measurement
validity (the dual-DV clause + the graded-primary-for-ranking-targets
extension), and `.claude/rules/marker-leakage-measurement.md`. **The
programmatic marker DV is NOT a judged-behavior DV** — it keeps its own
three-space `log P` recipe and is governed by `marker-leakage-measurement.md`,
not this rule. The guidelines below carry literature citations (arXiv ids +
incidents); those ARE the substance and stay (they are paper citations, not
person/meeting attributions).

The chat investigation behind this rule (2026-06-30) found the project's
behavior-expression judging is binary (fraction-positive rate), which
attenuates the predictor correlations the #658/#742/#761/#763 line depends on
and diverges from the field standard (Persona Vectors uses graded 0–100).

## A. Scoring mode & scale

1. **Graded pointwise 0–100 is the PRIMARY scoring mode for a
   ranking/regression/predictor-target DV.** Dichotomizing a graded behavior
   into a binary pass/fail attenuates a correlation by ~0.798 (≈36%
   effective-N loss; Cohen 1983 "The cost of dichotomization"; MacCallum et
   al. 2002, *Psychological Methods*) — worse near a floor/ceiling, exactly
   where a predictor target needs dynamic range. A coarse integer scale (1–5)
   collapses most of the graded signal too; prefer 0–100.

2. **The binary positive RATE is RETAINED as the validated human-legible
   headline companion (dual-DV roles).** A binary agreement/expression rate
   is what a mentor reads and what the on-policy behavioral construct is
   validated as; keep reporting it. Graded-primary applies to the
   ranking/regression role (CLAUDE.md § Measurement validity) — it EXTENDS,
   never replaces, the binary rate.

3. **Pointwise, not pairwise, for ABSOLUTE measurement.** Pairwise comparison
   is for preference ranking, not for an absolute behavior score, and it is
   more position-biased: a distractor flips a pairwise verdict ~35% of the
   time vs ~9% pointwise (arXiv 2504.14716). Use pointwise scoring for a DV
   that must measure how much a single output expresses a behavior.

## B. Variance reduction & aggregation

4. **Multi-sample N judge draws at temperature > 0, mean-aggregated.** This
   is the implementable substitute for the paper's logit-weighted scoring —
   the Anthropic Messages API exposes no score-token logprobs, so average N
   sampled integer scores instead (G-Eval 2303.16634 logit-weights; we
   approximate by sampling). N and temperature are NOT literature-pinned →
   pilot N≈5–10 by measured test-retest reliability (§F rule 15) and record
   the chosen N + temperature per DV.

5. **Graded multi-sampling partly substitutes for more binary samples.** The
   binary penalty is mostly dichotomization loss, not binomial sampling
   noise, so a few graded draws recover more signal than many binary draws
   would; do not over-budget judge calls chasing binomial precision a graded
   score already captures.

## C. Prompt / rubric design

6. **Anchored rubric — define what 0 / 50 / 100 mean.** An unanchored 0–100
   ask drifts; Prometheus (2310.08491, ICLR 2024) measured judge–human r
   dropping 0.457 → 0.355 without rubric anchors. Spell out the endpoints and
   the midpoint in the judge prompt.

7. **Reason-then-score, not score-then-reason.** Eliciting the rationale
   before the number lifts Prometheus judge–human r 0.673 → 0.847. Ask for a
   brief justification, then the integer.

8. **One behavior per judge call.** Bundling multiple behaviors into one
   rubric confounds them; score each DV in its own call.

9. **DROP a malformed / `REFUSAL` / out-of-range judge return — NEVER coerce
   it** (from BOTH arms of any contrastive comparison). A judge return that is
   `REFUSAL`, non-numeric, or outside [0, 100] carries no information about
   the behavior and must not enter the pool; coercing it to a number (e.g.
   `→ 0` or `→ 50`) biases the mean toward whichever arm the refusals land in.
   The drop set is CONTENT drops only — returns the judge produced after
   seeing the content. A call that failed in TRANSPORT (rate-limit, overload,
   timeout, connection — no verdict was ever produced) is NOT a drop: it is
   retried and, on exhaustion, reported as transport-loss per rule 24, never
   blended into this count. This generalizes the persona-vectors judge-filter
   rule (`.claude/rules/persona-vectors-recipe.md` step 4) to every judged
   DV, and the per-arm dropped count is REPORTED (a high or arm-asymmetric
   drop rate is itself a diagnostic — see rule 23's truncation check).
   (`eval/belief.py` previously default-coerced a malformed return to 50 —
   FIXED in #766: parse failures / out-of-range returns drop to `math.nan`,
   excluded from the aggregate by the caller's `not math.isnan` filter. #766
   ALSO routed API errors to the same `nan` drop path — that half is
   SUPERSEDED by rule 24: a transport failure is retried / re-judged, never
   persisted as a drop.)

24. **Transport errors are RETRIED with bounded backoff and re-judged — NEVER
    persisted as dropped draws.** A TRANSPORT failure is any judge call that
    died before the judge produced a verdict about the content: HTTP 429
    (rate limit), HTTP 5xx incl. 529 Overloaded
    (`anthropic._exceptions.OverloadedError` — NOT an `InternalServerError`
    subclass in the installed SDK (anthropic 0.88.0): its MRO is
    `OverloadedError → APIStatusError`, and
    `issubclass(OverloadedError, InternalServerError)` is False, so a
    transient tuple catching only `InternalServerError` MISSES 529),
    `APITimeoutError`, `APIConnectionError`, and Batch-API per-row terminal
    failures of server class (`errored` with a server error type, `canceled`,
    `expired` rows). Such a failure carries NO information about the judged
    content — it is freely re-judgeable (#1090: a re-judge recovered
    2,635/2,638 stored 529s with zero refusals) — so persisting it as a
    dropped draw silently censors arms and mimics a selection artifact (the
    rule-23 shape; #1090's asymmetry: 400/1000 trained draws lost on one
    cell). Three sub-rules:
    (i) **Retry, bounded.** Route judge calls through `api_dispatch.py`
    (mandatory per the API-throughput rule): its transient tuple retries
    connection / timeout / 500-class with exponential backoff within
    `max_attempts` (default 5) and its 429 handling rides AIMD with honored
    `retry-after` (`DEFAULT_MAX_429_RETRIES = 6`) — but as of anthropic
    0.88.0 the tuple does NOT cover 529 `OverloadedError` (see the taxonomy
    above; closing that gap is the sibling library task — until it lands, a
    529-exhausted item surfaces as `error: True` and MUST be re-judged, not
    persisted). The Batch path has NO per-row retry machinery today
    (`eval/batch_judge.py` surfaces `error: True` dicts): a pipeline
    consuming batch results MUST collect transport-class failed rows and
    re-dispatch them (a follow-up batch, or sync dispatch) rather than
    persist them. Batch expiry carve-out: an IN-FLIGHT batch is never a
    retry surface — the deadline-bounded `batch_judge` poller self-harvests
    at `expires_at` (#658/#663); this rule governs terminal per-row failures
    after collection (an `expired` row is retriable transport loss ONLY
    post-collection), and sync-path exceptions.
    (ii) **On exhaustion: transport-loss, never coerce, never blend.** When
    the bounded retry budget exhausts, the draw is recorded and REPORTED as
    per-arm `transport_loss` — a counter DISTINCT from the rule-9
    content-drop count (blending them recreates the censoring this rule
    exists to prevent; `JudgeResult` carries no such field yet — that
    counter is part of the pending sibling library fix, so report it from
    the pipeline's own artifacts). It is never coerced to a number. A
    headline DV with nonzero transport-loss (arm-asymmetric or not) gets the
    lost draws re-judged before publication (they are freely re-judgeable),
    or carries the asymmetry as an explicit caveat. The re-judge BYPASSES
    the rubric-keyed judge cache for the affected draws — surgical per-draw
    merge, a fresh `cache_dir`, or draw-indexed keying — because (a) the
    cache-update loop persists `error: True` entries (rule 23's cache
    caveat: the same cache dir re-serves the stored transport error) and
    (b) the rubric-keyed cache shares ONE key across an item's identical
    draws, so a cache-served re-run silently substitutes a successful
    sibling draw's score for the lost draw — a duplicated draw masquerading
    as a recovery that defeats rule 4's multi-draw independence and reads
    `transport_loss: 0` in every diagnostic (#1090's own recovery used the
    surgical per-draw merge for exactly this reason). The re-judge uses the
    SAME instrument as the original pass — rubric, judge model, `max_tokens`
    (rule 18's pins) — never a mixed-instrument merge.
    (iii) **Boundary cases — NOT transport.** A judge `REFUSAL` is
    content-informative (the judge saw the content and declined) → rule-9
    drop. A parse failure from `max_tokens` truncation is a budget defect →
    rule 23 (resize + re-judge against a fresh cache). An HTTP 400
    `invalid_request_error` is a pipeline bug (a malformed request fails
    identically on resubmit — `batch_judge.py` correctly quarantines it): it
    is NEITHER retried NOR dropped — fix the request builder; a whole arm of
    400s is a code failure, fail loud.
    Companion note (library, sibling task — `src/` is off-surface for the
    workflow fix that added this rule): `eval/batch_judge.py` persists
    `error: True` result dicts and `eval/graded_judge.py::_score_from_parsed`
    folds `parsed.get("error")` into the content-drop path (`return None`,
    line ~94); `llm/api_dispatch.py`'s transient tuple misses 529
    `OverloadedError` (its :569–571 comment claiming `InternalServerError`
    subclass coverage is stale in anthropic 0.88.0). The library fix — add
    `OverloadedError` (or `APIStatusError` with `status_code == 529`) to the
    transient tuple + fix the stale comment, transport-class re-dispatch on
    the batch collection path, and a separate `transport_loss` counter in
    `JudgeResult` — is a deferred sibling infra task, not part of this
    rule's change.

10. **Pin nuisance formatting identical across conditions.** Response length,
    markdown, system-prompt boilerplate, and the presence/absence of a
    reference answer all move a judge score independently of the behavior
    (arXiv 2506.22316 measured a reference-answer nuisance effect ~0.2 on
    Qwen-class outputs). Hold every nuisance variable fixed across the
    conditions being compared.

23. **Size the judge response `max_tokens` for the rationale, not the score.**
    A reason-then-score rubric (rule 7) emits its justification BEFORE the
    integer, so the response-token budget must cover the full rationale:
    give any reasoning rubric **≥ ~300 response tokens** (the #1090 recovery
    point); a score-only rubric (bare integer, no rationale) can stay small.
    An undersized cap fails SILENTLY: the API truncates the response at
    `max_tokens` before the score token is ever emitted, the truncated text
    fails to parse, and rule 9's drop-never-coerce then discards the draw —
    the judge call "succeeds" while the draw is censored. The censoring is
    ARM-ASYMMETRIC whenever one arm's rationales run longer, so it mimics a
    selection artifact on the very contrast being measured. Incident #1090:
    a 64-token cap hardcoded at `graded_judge.py`'s `judge_completions_batch`
    call dropped 473/1000 base vs 307/1000 trained draws as parse errors
    (47% vs 31%, arm-asymmetric), and the asymmetry initially read as a
    possible selection artifact on the headline install delta; a re-judge at
    `max_tokens=300` recovered 98.8% of the previously-dropped parse-error
    draws with 0 refusals — truncation of reason-first responses, not
    refusals. Diagnosis: parse-error drops that vanish at a larger budget
    with 0 refusals = truncation; treat any ≥~10% per-arm drop rate as a
    truncation check to run — AND a stored-transport-error check per rule 24
    (the #1090 529 shape reads identically arm-asymmetric) — never as noise.
    No sub-10% safe-harbor: the
    censoring conditions on rationale length, so smaller or arm-symmetric
    drops can still bias retained-draw means. After resizing, re-measure the
    per-arm drop rate at the new budget against a fresh `cache_dir` — that
    reported rate IS the per-rubric verification that the floor suffices
    (the floor is rubric-dependent; where the raw response is available,
    confirm truncation directly via `stop_reason == "max_tokens"` /
    truncated-text inspection). Mechanics: `judge_completions_batch`
    (`eval/batch_judge.py`) accepts `max_tokens` (default 256);
    `graded_judge.judge_graded` threads an optional `max_tokens` kwarg
    (introduced by #1090) — reasoning-rubric callers pass ~300 (the 64
    default is kept for legacy cache stability at the api_dispatch layer);
    neither library default is sized for a reasoning rubric — pass the
    budget explicitly. This is the judge-side analogue of the CLAUDE.md
    `max_new_tokens ≥ 2×` rule for the EVALUATED model's generations —
    truncation creates silent zeros on both sides of a judged eval. Cache
    caveat (rule 22): the rubric-level `JudgeCache` key
    (`rubric_fingerprint`) deliberately EXCLUDES max_tokens, and the
    cache-update loop in `judge_completions_batch` writes returned result
    dicts without filtering `error: True` entries — so raising the budget
    does NOT bust that cache and truncation-era parse-error entries can be
    re-served; run a truncation-recovery re-judge against a fresh
    `cache_dir` (or clear the stale entries). The generic `api_dispatch`
    adapter cache, by contrast, deliberately OVER-keys on the full built
    request incl. max_tokens — at that layer a budget change is a cold
    re-judge (a miss, never a wrong read). Report the per-arm dropped-draw
    rate + the `max_tokens` used alongside every judged DV (rule 9's
    per-arm dropped count; rule 18's pinned-report list).

## D. Judge model

11. **ONE cross-family judge: `claude-sonnet-4-5-20250929` judging Qwen.**
    Cross-family judging IS the self-preference bias mitigation
    (Play-Favorites 2508.06709; CALM 2410.02736 — a judge favors its own
    family's outputs). Never run a Qwen judge on Qwen outputs. The project
    pins one consistent judge across all behaviors (CLAUDE.md "LLM judge =
    `claude-sonnet-4-5-20250929`").

12. **Enforce the one-Sonnet-judge pin mechanically** via
    `scripts/workflow_lint.py --check-judge-model-pins` (bundled into the
    no-flags default run). The motivating incident is the #650/#657 stale
    legacy-Haiku judge pins that re-pinned a non-Sonnet judge for new work; the
    lint flags a hardcoded non-Sonnet judge-model pin at a judge call site.

## E. Closed-loop caveat

13. **A judge score used as a regression/predictor TARGET carries a
    self-reinforcement-toward-LLM-text risk.** G-Eval (2303.16634) found
    GPT-4 rating LLM outputs ABOVE human ones; using a judge score as the
    target a model is optimized/selected against can amplify whatever the
    judge over-rewards. Cross-family judging (rule 11) mitigates but does not
    eliminate this → validate the graded DV against an INDEPENDENT non-judge
    reference (a small human audit set, or the project's log-P / activation
    companion DV) before it carries a ranking/regression headline.

## E2. The non-saturating continuous companion DV (b)

A judged behavior-expression RATE saturates at floor/ceiling and censors install /
dose-matched / cross-condition comparisons; the dual-DV rule (CLAUDE.md
§ Measurement validity) pairs it with a SECONDARY non-saturating continuous
completion-probability DV. Two forms, in PREFERRED order:

19. **PREFERRED — teacher-forced fixed positive-vs-negative completion margin.**
    `margin(C) = mean LN-logP(FIXED positive-answer pool | C) − mean LN-logP(FIXED
    negative-answer pool | C)`, the answer pools judge-filtered ONCE and held FIXED
    across every context C, scored teacher-forced (length-normalized log-P of each
    fixed answer conditioned on C). Because the same answer set is scored under every
    context, there is NO selection-on-outcome bias. #722 validated it: ρ(margin, rate)
    was all-positive (refusal +0.56 [+0.14, +0.80], sycophancy +0.40 [−0.03, +0.68],
    broad_em +0.31 [−0.20, +0.67]) and the margin keeps spread where the rate is
    floored (broad_em margin std 0.31 vs rate std ~0.008). Caveat: a behavior with no
    fixed +/- pool is untestable here (#722: harmful_compliance, no #661 pool). This
    is a SECONDARY companion validated to track the rate — NOT a primary behavioral
    leaderboard read (the #432→#456 teacher-forcing caution still bars that use;
    `.claude/rules/marker-leakage-measurement.md` § #432 → #456).

20. **OPT-IN alternative — judged-positive-conditional-mean log-P (`logp_pos_mean`),
    selection-on-outcome-confounded.** Length-normalized trained−base log-P averaged
    over ONLY the judged-positive completions in each context. This is a conditional
    mean over an OUTCOME-SELECTED subset: contexts with more positives tend to have a
    lower mean log-P among them, so the DV anti-correlates with the rate. #722
    measured ρ(DV, rate) ≈ −0.3 for 3 of 4 behaviors — it FAILED the dual-DV
    validation gate. Use it ONLY after it passes the standing ρ(DV, rate) > 0
    validation (Spearman across cells with dynamic range) for the behavior at hand;
    prefer the fixed +/- margin (rule 19) by default.

Either form is SECONDARY to the PRIMARY on-policy judged rate and is subject to the
standing ρ(DV, rate) > 0 validation before it carries any cross-condition read; never
narrate it as the construct. (Source: #722 — `eval_results/issue_722/tf_margin/`.)

## F. Validation as a measurement instrument

14. **Validate reliability PER behavior class, not once.** Judge reliability
    is behavior-specific — CALM (2410.02736) found alignment/subjective
    behaviors markedly more bias-prone than factual ones. A reliability number
    established on one behavior does NOT transfer to another DV.

15. **Measure judge–human agreement on a small per-behavior audit set before
    trusting a DV.** Report Spearman/Pearson + Krippendorff's alpha against
    human labels; ~85% judge–human agreement is the rough MT-Bench benchmark
    (2306.05685). The agreement threshold + audit-set size are NOT
    literature-pinned → pilot. Carry the reliability ceiling √(r_yy) (rule 18)
    when interpreting an observed correlation.

16. **Re-confirm cross-family self-bias is negligible for
    `claude-sonnet-4-5` specifically** — this exact judge↔target pairing is
    untested in the literature, so check it on a small audit set rather than
    assuming the general cross-family result transfers.

17. **Keep the existing saturation / gaming detection.** The marker + dual-DV
    saturation rules still apply (`.claude/rules/marker-leakage-measurement.md`,
    CLAUDE.md § Measurement validity). A graded score additionally surfaces
    compression a binary rate hides — a rate pinned at 1.0 across conditions
    can still show graded spread, which is the saturation signature to watch.

21. **Design-aligned split-half on crossed designs.** When a reliability
    ceiling (split-half + Spearman–Brown; the √(r_yy) of rules 15/18) is
    computed over a CROSSED design — the same item/probe set scored under
    every condition — the half-split MUST be item-ALIGNED across conditions:
    ONE half-partition of the items, applied identically to every condition
    (prefer averaging over many aligned partitions, or a deterministic
    odd–even-by-item-id split, to a single random draw). Splitting each
    condition's items INDEPENDENTLY violates the parallel-halves assumption
    Spearman–Brown requires: within a condition the two halves partition the
    same items, so the item-composition offset enters the two half-scores
    with OPPOSITE signs; independent splits make that anti-correlated offset
    vary across conditions, and when its half-mean variance (item main-effect
    variance scaled by half size, ≈ σ²_item/(n_items−1)) exceeds the
    condition signal it dominates, driving split-half r systematically
    NEGATIVE — which the non-negativity floor reports as a 0.00 ceiling,
    censoring real signal. (An aligned split turns the offset into a shared
    constant that cancels in the cross-condition correlation.) Incident #763
    (`reliability_split_half_over_probes`, crossed context × probe; probe sd
    ≈27/100 vs context signal ≈3/100 — half-mean offset sd 27/√59 ≈ 3.5 vs
    signal ≈ 3): independent splits gave r = −0.41 (v1) / −0.23 (v2),
    ~100%/99% of 200 random splits negative, clipping a real v2 ceiling
    ≈0.59 to 0.00; the probe-ALIGNED split recovered it. Report the
    splitting scheme AND the aggregation convention alongside the ceiling
    (rule 18) — e.g. mean r across aligned partitions BEFORE Spearman–Brown,
    and where any non-negativity floor is applied. The aligned ceiling is
    conditional on the fixed item panel (item main effects deliberately
    excluded) — do not read it as item-sampling generalization.

## G. Reproducibility / reporting

18. **Pin & report, per DV:** scoring mode, scale, N samples + temperature,
    judge model + date, prompt hash, the response `max_tokens` budget + the
    per-arm dropped-draw rate SPLIT content-drops vs transport-losses
    (rules 23/24), per-behavior reliability
    (test-retest + judge–human agreement), and the reliability ceiling
    √(r_yy). A judged DV is a measurement instrument; report it like one.

22. **Judge-result caches key on (rubric/behavior id, question, completion) —
    NEVER on completion content alone.** Any cache in front of a judge call
    (file-based, in-memory, or resume-time) MUST carry the rubric / behavior
    identity in its key — in practice the full judge prompt (or a stable hash
    of it), plus the judge model id when it can vary — alongside the
    question + completion. A content-only key silently returns ANOTHER
    behavior's judgment for the same completion whenever completions are
    reused across rubrics (a multi-behavior eval scoring one generation pool
    under several rubrics against a shared cache dir). Incident #810:
    `JudgeCache._hash_key` (`src/explore_persona_space/eval/batch_judge.py`)
    keyed on (question, completion) only, so refusal-rubric judgments leaked
    into harmful_compliance E0 scores via the shared content-keyed cache
    dir — 64% of high harmful_compliance scores were refusal-rubric
    judgments, and 99.1% of flagged rows shared their exact
    (reasoning, score) pair with the refusal file (vs 18.7% baseline) — an
    exact-pair fingerprint implicating cache reuse rather than a prompt bug
    (independent temperature>0 judge draws would not be byte-identical).
    The key fix LANDED in #1018: `JudgeCache.get`/`put`/`_hash_key` now
    REQUIRE a keyword-only `rubric_key` (an unthreaded call site raises
    `TypeError`), derived via the helper
    `rubric_fingerprint(judge_model, judge_system_prompt, format_user_msg)`
    — the user-msg template is sentinel-rendered into the fingerprint so a
    rubric living in the USER message (the `graded_judge` shape) enters the
    key, not only a system-prompt rubric. The generic adapter
    `llm/api_dispatch.py::_cache_key_parts` returns
    (item_id, payload JSON, built-request fingerprint) and threads the
    fingerprint as the `rubric_key` — the built params dict embeds model +
    system + messages, so the adapter key is STRUCTURAL (no longer
    caller-dependent; deliberately over-keyed on full request params), and
    the api_dispatch batch checkpoint additionally stores a run-level
    request fingerprint in its `state.json` that is recomputed on load and
    FAILS LOUD on mismatch (a rubric-B dispatch can never replay rubric-A's
    checkpoint). The key schema embeds the literal `EPM_JUDGE_CACHE_KEY_V2`
    version tag, so all pre-fix content-keyed entries are automatically
    unreachable — a cold re-judge, never a wrong read. Per-rubric
    `cache_dir` partitions (`datagen.py`'s `judge_cache/pos` vs
    `judge_cache/neg`) remain good hygiene but are no longer load-bearing.
    Plan-side: any plan whose eval judges >1 rubric/behavior over a shared
    completion pool — or resumes judging against a previously-populated
    shared cache dir — NAMES its judge-cache key fields (or the per-rubric
    cache partition); critics REVISE a shared multi-rubric judge cache
    without a rubric-bearing key.

## Do NOT over-rely on (adversarially refuted)

- "Pointwise is universally more robust than pairwise" — pointwise wins for
  ABSOLUTE measurement (rule 3), not universally.
- "A reference answer is the single biggest accuracy driver" — it is a
  nuisance variable that must be held fixed (rule 10), not a free accuracy
  boost.
- "GPT-4 ≈ human–human agreement is a safe ceiling" — it is not; judges rate
  LLM text above human text (rule 13) and reliability is behavior-specific
  (rule 14).
- "It's familiarity, not self-recognition" as a reason to ignore self-bias —
  cross-family judging is still required (rule 11) regardless of the mechanism.
- "A bigger judge ⇒ less bias" — size does not remove self-preference or
  behavior-specific unreliability; validate per behavior (rules 14–16).

## Enforcement

- `planner.md` §6 + `critic.md` Statistics & Measurement lens — the
  measurement-validity / dual-DV requirements a judged DV plan must meet.
- `scripts/workflow_lint.py --check-judge-model-pins` — the mechanical
  one-Sonnet-judge pin gate (rule 12).
- CLAUDE.md § Measurement validity — the always-on dual-DV clause + the
  graded-primary-for-ranking-targets extension this rule details.
- Rule 22 (judge-cache keying) rides the same lens load: the Statistics &
  Measurement critic REVISEs a shared multi-rubric judge cache without a
  rubric-bearing key; the plan names the cache key fields per rule 22.
- Rule 23 (response-budget sizing) rides the same lens load: a plan whose
  judged DV uses a reason-then-score rubric names its judge `max_tokens`
  (≥ ~300, or a stated justification) and its per-arm drop-rate report;
  the Statistics & Measurement critic REVISEs an unsized reasoning-rubric
  judge. Plan-enforced in v1 — no mechanical lint.
- Rule 24 (transport-vs-content split) rides the same lens load: a plan whose
  judged-DV pipeline persists API/transport errors as dropped draws — or
  whose per-arm drop report does not split content-drops from
  transport-losses — is a Statistics & Measurement REVISE. Plan-enforced in
  v1 — no mechanical lint (same enforcement class as rule 23).
- The `--check-judge-model-pins` `test_live_trees_pass()` invariant locks the
  grandfather allowlist to today's tree; a future LEGITIMATE non-Sonnet judge
  pin (a new calibration anchor or translation-judge exemption) must be added
  to `JUDGE_PIN_LEGACY_ALLOWLIST` / `JUDGE_PIN_LEGACY_ALLOWLIST_SH` with an
  inline `reason` when it lands, or the no-flags default run FAILs.

## Files of record

Task body #765 (the guideline derivation + the two adversarial deep-research
dives); task body #763 (the design-aligned split-half incident behind rule 21);
task body #810 (the shared judge-cache rubric-leak incident behind rule 22);
task body #1090 (the max_tokens truncation-censoring incident behind rule 23
and the ~2,638 stored API-529 transport-error draws behind rule 24); task
body #1206 (the transport-vs-content split);
`.claude/rules/persona-vectors-recipe.md` (the graded-judge precedent +
judge-filter drop rule); `.claude/rules/marker-leakage-measurement.md` (the
non-judged marker DV); the enforcing agent files (`planner.md`, `critic.md`,
`analyzer.md`, `interpretation-critic.md`, `clean-result-critic.md`).
