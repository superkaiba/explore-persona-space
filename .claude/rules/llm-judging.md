---
description: Full LLM-judged-DV recipe (graded 0-100 primary, drop-never-coerce, transport retry, max_tokens floors, pilot gate); the CLAUDE.md LLM-judge bullet is the always-on summary
paths:
  - "src/explore_persona_space/eval/**"
  - "src/explore_persona_space/llm/**"
  - "scripts/*judge*.py"
  - "tasks/**/plans/*.md"
---

# LLM judging of behavior-expression DVs

**Load this rule whenever a plan or code DESIGNS or WRITES an LLM-judged
behavior-expression dependent variable** (sycophancy, refusal, hedging,
style, trait, alignment/EM verdict — any DV where an LLM scores whether a
model output expresses a target behavior). The always-on backstop is the
CLAUDE.md "Measurement validity" bullet + the "LLM judge =
`claude-sonnet-4-5-20250929`" bullet; this file is the full guideline
recipe those bullets point at (the judge-scoring guidelines plus the
§ E2 continuous-companion-DV recipe), in the on-demand register.

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
   blended into this count. And an API-LEVEL refusal — a SUCCEEDED row whose
   `stop_reason` is `"refusal"` with an EMPTY content array (the provider's
   safety classifier declined; NO verdict was produced) — is NOT a rule-9
   drop either: it is the THIRD drop class (rule 28, #2151), distinct from
   the instructed rubric `REFUSAL` above, which IS a produced verdict and
   stays a content drop. This generalizes the persona-vectors judge-filter
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
    connection / timeout / 5xx incl. 529 `OverloadedError` (matched via the
    public `APIStatusError` + `status_code == 529` form — landed in #1313)
    with exponential backoff within `max_attempts` (default 5), and its 429
    handling rides AIMD with honored `retry-after`
    (`DEFAULT_MAX_429_RETRIES = 6`). Exhaustion of either budget returns
    `category` `RESULT_TRANSPORT` / `RESULT_RATE_LIMITED` (both
    transport-class, re-drivable — #1313), never a bare terminal error. The
    PRIMARY batch path (`judge_completions_batch` → `dispatch_judge_items`)
    re-dispatches errored-server/expired/stuck-canceled rows once, resumably
    (the #1019 machinery in `eval/judge_dispatch.py`); only the LEGACY
    `_submit_and_poll_batch` path (`eval/batch_judge.py`, two frozen
    `scripts/issue_389` callers) has NO per-row retry machinery — a pipeline
    consuming ITS results MUST collect transport-class failed rows
    (`is_transport_error_dict`, #1313) and re-dispatch them (a follow-up
    batch, or sync dispatch) rather than persist them. Batch expiry
    carve-out: an IN-FLIGHT batch is never a
    retry surface — the deadline-bounded `batch_judge` poller self-harvests
    at `expires_at` (#658/#663); this rule governs terminal per-row failures
    after collection (an `expired` row is retriable transport loss ONLY
    post-collection), and sync-path exceptions.
    (ii) **On exhaustion: transport-loss, never coerce, never blend.** When
    the bounded retry budget exhausts, the draw is recorded and REPORTED as
    per-arm `transport_loss` — a counter DISTINCT from the rule-9
    content-drop count (blending them recreates the censoring this rule
    exists to prevent; `JudgeResult.n_transport_lost_draws` +
    `per_item_transport_losses` carry the split as of #1313 —
    `n_dropped_draws` is content-only there). It is never coerced to a
    number. A
    headline DV with nonzero transport-loss (arm-asymmetric or not) gets the
    lost draws re-judged before publication (they are freely re-judgeable),
    or carries the asymmetry as an explicit caveat. The re-judge BYPASSES
    the rubric-keyed judge cache for the affected draws — surgical per-draw
    merge, a fresh `cache_dir`, or draw-indexed keying — because (a) the
    cache-update loop persists content-class `error: True` entries (rule
    23's cache caveat; as of #1313 TRANSPORT-class dicts are put-skipped and
    a stored one reads as a cache MISS, and as of #2021 TRUNCATION-class
    dicts get the same put-skip/get-miss treatment — but the legacy reason-string
    fallback covers only known pre-#1313 strings, so do not assume every old
    poisoned entry self-heals) and
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
    drop. An API-LEVEL `stop_reason == "refusal"` on a SUCCEEDED row (empty
    content — the provider's safety classifier declined; no verdict exists)
    is NEITHER content nor transport: it is the THIRD top-level drop class,
    transport-conditional and retriable → rule 28 (#2151/#1739). A parse
    failure from `max_tokens` truncation is a budget defect →
    rule 23 (resize + re-judge against a fresh cache). An HTTP 400
    `invalid_request_error` is a pipeline bug (a malformed request fails
    identically on resubmit — `batch_judge.py` correctly quarantines it): it
    is NEITHER retried NOR dropped — fix the request builder; a whole arm of
    400s is a code failure, fail loud.
    Companion note (library — LANDED in #1313): `llm/api_dispatch.py`'s
    transient predicate now matches 529 via the public `APIStatusError` +
    `status_code == 529` form and returns `RESULT_TRANSPORT` on transient
    exhaustion; error dicts at every mint site carry a structural
    `transport: True` flag classified by
    `eval/batch_judge.py::is_transport_error_dict` (with a conservative
    legacy reason-string fallback for pre-#1313 persisted dicts);
    `JudgeResult` carries `n_transport_lost_draws` +
    `per_item_transport_losses` (content drops stay in `n_dropped_draws`);
    and the judge cache never PUTs — and treats as a MISS — a
    transport-class dict. The primary batch path's re-dispatch is the
    pre-existing #1019 machinery (verified, not duplicated);
    `_score_from_parsed` is unchanged (classification lives in the tally).

28. **API-level `stop_reason == "refusal"` is the THIRD top-level drop class —
    transport-conditional and RETRIABLE; report it separately from BOTH
    content drops and transport losses, and remediate by targeted SYNC
    re-issue at the IDENTICAL instrument (#2151).** The Batch API can return
    a `result.type == "succeeded"` row whose `stop_reason` is `"refusal"`
    with an EMPTY content array: the request was accepted, the provider's
    safety classifier declined, and NO verdict about the content was
    produced. That row is NEITHER a rule-9 content drop (no verdict exists —
    unlike the instructed rubric `"REFUSAL"`, which IS a produced verdict)
    NOR a rule-24 transport loss (the API answered; the in-band retry
    envelope re-issues on the SAME transport and cannot fix it). Blending it
    into either count recreates the censoring the rule-24 split exists to
    prevent, and #1739's evil-OOD wave shows the miscount at scale: 44,310
    forced-BATCH judge calls returned 15,091 refusal draws (34.1%), all
    filed as `parse_error` content drops; 4,982/14,770 items (33.7%) were
    left with ZERO valid draws. The class is TRANSPORT-CONDITIONAL, not
    content-informative: re-issuing the IDENTICAL instrument on the SYNC
    path produced 0 re-refusals in 14,887 draws and rescued 5,077/5,172
    censored items (98.2%). The censoring is OUTCOME-CORRELATED, never
    missing-at-random — for answer-keyed corpora the classifier keys on the
    ANSWER's severity (rescued items scored 1.4x higher than never-censored
    items overall; 2.3x on mhj, 3.7x on pair), and for a jailbreak-query
    corpus (tom-gibbs) on the QUERY occupying the `{question}` slot (~2/3 of
    the corpus censored near-indiscriminately, 1.24x) — so absorbing it into
    the content tally biases the DV downward on exactly its highest-scoring
    rows.
    Mechanics (#2151): classification lives in the reduce —
    `batch_judge.API_REFUSAL_STOP_REASONS` / `is_api_refusal_stop_reason` /
    `is_api_refusal_error_dict`; `JudgeResult.n_api_refusal_draws` +
    `per_item_api_refusals`, a SIBLING of `n_transport_lost_draws`, with the
    instructed rubric-`REFUSAL` counter (`n_refusal_draws`, #1801) moving
    independently. The reduce WARNs on any non-zero count; api-refusal-class
    error dicts are cache PUT-SKIPPED and read back as cache MISSES
    (mirroring the #1313/#2021 transport/truncation treatment), so a
    transport change self-heals censored rows with no fresh `cache_dir`. A
    persisted dict lacking `stop_reason` (pre-#2021 legacy) classifies as a
    content drop, exactly as before.
    Remediation (the recipe that worked, #1739): re-issue ONLY the censored
    draws on the SYNC path at the IDENTICAL instrument — same judge model,
    rubric, temperature, `max_tokens` (rule 18's pins); never a
    mixed-instrument merge — merge the sync scores alongside each item's
    surviving genuine batch draws, LICENSE the merge with a dual-scored
    parity check on ~200-300 overlapping items with the batch-vs-sync offset
    REPORTED (#1739: 287 items, batch mean 7.26 vs sync 7.77), and DISCLOSE
    the batch/sync split in the run's `judge_meta`. Reference
    implementation: `scripts/issue1739_evilood_refusal_rejudge.py`.
    COVERAGE NOTE (#2152): the rule-26 pilot gate NOW covers this class
    when the caller declares the wave's dispatch (`wave_n_calls` /
    `wave_threshold_base` / `wave_force_sync` on `judge_pilot_gate`): the
    pilot is forced onto the wave's transport (realized route read back
    from the persisted `save_raw["routing"]` record; a mismatch, an
    unverifiable transport, or a partially cache-served pilot —
    `n_cached > 0`, whose cache-served draws are refusal-free by
    construction and dilute the rate — is a FAIL; an unpinned
    count-routed declaration inside the OTPM-probe region
    `n < 2×threshold_base` is refused fail-fast), and the per-arm
    api-refusal rate `n_api_refusal / (n_draws − n_transport_lost)` gates
    at ≥ 0.10 (waivable per arm via `waive_api_refusal_arms` with the
    reason recorded at the caller-site constant — e.g. a pre-planned sync
    re-issue remediation; truncation and the effective-draws floor stay
    unwaivable). RESIDUAL non-coverage: a LEGACY caller that does not
    declare the wave gets only a report warning — do not read THAT
    pilot's PASS as protection against api-refusal censoring.
    The residual backstops that DO fire: a censored arm's shrunken
    `n_answered` / `n_scored` — api-refusal draws leave the `n_answered`
    parse-fail denominator, so a downstream consumer's scored-count floor
    can trip, while the gate's OWN effective-draws floor
    (`min_effective_draws_per_arm`, evaluated against
    `n_draws - n_transport_lost`) does NOT shrink: api-refusal draws stay
    inside it — the reduce's WARNING on any non-zero count, and the
    `n_api_refusal` field carried per arm in the pilot report —
    gate-keyed since #2152 for wave-declared pilots (clause (d)).

29. **Report per-ITEM completeness (`frac_items_complete`) per behavior /
    arm against a pre-registered floor — a per-DRAW drop rate does not
    PREDICT the per-item hole (#2124).** Rules 9 / 24 / 28 tally drops per
    DRAW. The DV is computed per ITEM, so the quantity describing the DV's
    real denominator is `frac_items_complete =
    n_items_with_at_least_one_valid_draw / n_items`. Report it beside the
    per-arm drop tallies (rule 18) for every judged DV, pre-register a
    floor (**default 0.95** — see the calibration note below), and when an
    arm lands below it, identify WHICH drop class opened the hole — the
    rule-9 content class, split into parse-failure vs instructed rubric
    `REFUSAL` (`n_refusal_draws`, #1801); rule-24 transport loss; rule-28
    api-refusal — and remediate per that class BEFORE plotting or
    reporting the DV. Never silently narrow the denominator.

    The per-draw rate `p` upper-bounds the hole (with `d` equal draws per
    item, at most `floor(p·N)` items can lose all `d`) but tells you
    nothing about where in `[max(0, 1-p), 1.0]` completeness actually
    lands, and the top of that range is reachable only when
    `p <= (d-1)/d`. Under INDEPENDENT drops you would expect `p^d` holes;
    the drop classes concentrate instead. #1739's evil DV slice
    (`eval_results/issue_1739/result1_spread/spread_stats_refusal_zero.json`,
    `recode_audit.per_source.own_rungs`) ran 46,525/159,990 = 29.1%
    refusal DRAWS at `d=3` — an independence read of 2.5% — and landed at
    15,076/53,330 items (**28.3%**) with ZERO valid draws,
    `frac_items_complete = 0.717`: **11.5× the independence expectation**,
    against only 865 MIXED items. The `wildchat_rung` arm in the same file
    shows the trap from the other side: 1.8% draws and 1.8% items, so the
    per-draw proxy agrees exactly where the answer does not matter. The
    equal-draws qualifier is load-bearing — once realized per-item draw
    counts vary (routine under transport losses), even the upper bound is
    void, since a 1-draw item is emptied by one drop.

    Gate on it because the censoring is OUTCOME-CORRELATED, so the
    surviving items are a biased subsample and the bias is invisible in
    aggregate draw counts: rule 28's evil-OOD wave left 4,982/14,770 items
    (33.7%) empty and the rescued items scored 1.4× higher than
    never-censored items overall (2.3× on mhj, 3.7× on pair). A rule-26
    pilot-gate PASS is NOT a substitute — the gate is per-draw, and rule
    28's class is explicitly outside its protection.

    Calibration of the 0.95 default: it sits between the measured healthy
    band (0.982, `wildchat_rung`) and the measured broken band (0.717
    #1739 evil / 0.663 #2151 evil-OOD), with a thin 0.03 upper margin — so
    RE-DERIVE it per behavior class at pre-registration rather than
    inheriting it, and expect harm-class corpora to need their own floor
    (rule 28's tom-gibbs corpus ran ~2/3 censored pre-remediation).

    Mechanics (#2124): `judge_pilot_gate`'s per-arm report carries
    `frac_items_complete`, `n_items_zero_valid` and `n_items` alongside
    the draw tallies — but a ~200-draw pilot resolves completeness only to
    ~1/n_items (≈6% at 17 items/arm), so the FLOOR is a production-wave
    read, not a pilot verdict (`JudgeResult.frac_items_complete` is the
    production-wave affordance). `scripts/issue1739_judge_reliability.py`
    is the naming precedent.

10. **Pin nuisance formatting identical across conditions.** Response length,
    markdown, system-prompt boilerplate, and the presence/absence of a
    reference answer all move a judge score independently of the behavior
    (arXiv 2506.22316 measured a reference-answer nuisance effect ~0.2 on
    Qwen-class outputs). Hold every nuisance variable fixed across the
    conditions being compared.

23. **Size the judge response `max_tokens` for the rationale, not the score —
    and be GENEROUS: a cap is not a spend.** The API bills only GENERATED
    tokens; `max_tokens` is a ceiling, so headroom above the typical
    rationale costs nothing on well-formed responses. The real costs
    of a larger cap are (a) a one-time cold re-judge at the over-keyed
    `api_dispatch` adapter-cache layer when a budget changes, (b) rare
    degenerate long responses, and (c) a higher per-request OTPM
    reservation on the SYNC dispatch path (the rate limiter reserves
    `max_tokens` per in-flight request, cutting effective concurrency —
    prefer the Batch API, which the CLAUDE.md large-judge-set Batch-API
    mandate (§ LLM judge bullet) already requires) —
    all bounded and all far cheaper than the
    re-judge waves an undersized cap forces. A reason-then-score rubric
    (rule 7) emits its justification BEFORE the integer, so the
    response-token budget must cover the full rationale: give a multi-field
    reason-then-score JSON rubric (several labeled reasoning fields before
    the score — the #1769 fu1 rubric shape) **≥ 2048 response tokens**, and
    a short single-rationale rubric **≥ 1024** (floors raised 600/300 →
    2048/1024 on 2026-08-02 after three truncation re-judge waves in one
    week — #1739 / #1769 / #1774 — each at or above the then-floor); a
    score-only rubric (bare integer, no rationale) can stay small. The
    historical floors' failure record: the 300 floor measurably failed
    twice in one week for multi-field JSON rubrics. #1739's sycophancy wave
    at `max_tokens=400` (86,521 rollouts × 3 draws) truncation-censored
    5.4% of draws, recovered to 2.3% by a surgical re-judge at 800.
    #1769's fu1 wave at the 300 floor (21,000 draws) dropped 1606/21,000
    draws overall (7.6%); the hallucination arms dropped 12.5% of their
    own draws (874/7000), worst arm hallucination/decode_only/a3 42.5%
    (425/1000) — arm-asymmetric. Forensics on the 600-budget re-judge
    split the overall drops: ~300 of the 1606 were true truncation,
    recovered at 600, while the residual (1251/1291 of the mt600 parse
    failures; hallucination arms still 9.8%, worst arm 34.8%) were EMPTY
    judge responses on degenerate steered text — content-class per rule 9,
    NOT budget — with per-arm mean scores stable across budgets (max shift
    1.5 pts). The real cost of an under-sized floor is the disambiguation:
    at a too-small budget, truncation drops and content drops are
    indistinguishable, so the whole 21k-call wave had to be re-judged
    (~1h wall + a second 21k-call batch spend) to tell them apart; 600
    eliminated the truncation signature for this rubric class. The floor
    is a floor, not a guarantee: the post-resize per-arm drop re-measure
    below is the binding check at ANY budget, and a percent-level
    parse-error residue at an above-floor budget is still a truncation
    signature to re-judge, never noise.
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
    truncated-text inspection). Drop-class diagnosis MUST inspect
    `stop_reason` / the raw stored response — NEVER the failure-LOG text,
    which loggers routinely truncate (`Text: %.200s`) before writing: a
    "0 of N failures carried a closing brace" read off 200-char log
    prefixes is uninformative by construction (#1773, 2026-07-31 — a
    direct re-issue showed 39/40 `end_turn`, refuting the log-derived
    diagnosis). Mechanics: `judge_completions_batch`
    (`eval/batch_judge.py`) accepts `max_tokens` (default 256);
    `graded_judge.judge_graded` threads an optional `max_tokens` kwarg
    (introduced by #1090) — reasoning-rubric callers pass ≥ 1024
    (single-rationale) / ≥ 2048 (multi-field JSON) (the 64
    default is kept for legacy cache stability at the api_dispatch layer);
    neither library default is sized for a reasoning rubric — pass the
    budget explicitly. This is the judge-side analogue of the CLAUDE.md
    `max_new_tokens ≥ 2×` rule for the EVALUATED model's generations —
    truncation creates silent zeros on both sides of a judged eval. Cache
    caveat (rule 22): the rubric-level `JudgeCache` key
    (`rubric_fingerprint`) deliberately EXCLUDES max_tokens, so raising
    the budget does NOT bust that cache. As of #2021 truncation-class
    error dicts (`error: True` + a truncation-class persisted
    `stop_reason` — `batch_judge.is_truncation_error_dict`) are
    put-skipped by the cache-update loop AND read back as cache MISSES
    (mirroring the #1313 transport handling), so a budget raise
    self-heals those entries with no fresh `cache_dir`. PRE-#2021
    truncation-era entries lack `stop_reason` and remain served — they
    still need a fresh `cache_dir` (or clearing); the get-miss covers
    dicts written by MIXED-VERSION writers (post-threading,
    pre-cache-hygiene code), NOT pre-#2021 dicts. And a cache-served
    legacy SUCCESS entry (a kept score written pre-#2021) tallies
    `"unknown"` in `stop_reason_tally` — a legacy-cache signature, not a
    threading failure. The generic `api_dispatch`
    adapter cache, by contrast, deliberately OVER-keys on the full built
    request incl. max_tokens — at that layer a budget change is a cold
    re-judge (a miss, never a wrong read). Report the per-arm dropped-draw
    rate + the `max_tokens` used alongside every judged DV (rule 9's
    per-arm dropped count; rule 18's pinned-report list).

25. **Name the confusable neighbor categories in any CATEGORY-axis rubric.**
    A judged categorical DV (a yes/no or k-way label over units — e.g.
    `persona_related` over SAE features) names its known confusable neighbor
    categories in the rubric and instructs the judge how to label them
    (negatives or a separate label), and the axis is validated by inspecting
    top-ranked positives for neighbor contamination BEFORE it carries a
    headline. Incident #1482: 20/40 top "persona" features were bare
    language-identity features — answer language is near-deterministic given
    query language, so the unnamed neighbor rode the judged contrast; caught
    ~2h post-ship, rubric amendments posted in-flight, durable instrument
    #1773.

26. **Pilot-gate every large judge wave — measure the drop profile BEFORE
    the production spend.** Before any production judge dispatch of
    ≥ ~5,000 calls, run a PILOT of ~100–200 draws spanning the arms /
    conditions (and every rubric in the wave) at the EXACT production
    instrument (rubric, judge model, `max_tokens`, and TRANSPORT — the
    sync-vs-Batch dispatch path the wave itself will use; #2152), and
    gate the full dispatch on ALL of: (a) `stop_reason == "max_tokens"`
    fraction ≈ 0 in the pilot's raw responses — any nonzero truncation
    signature → raise
    the budget (generously; rule 23's cap-is-not-a-spend point) and
    re-pilot; and (b) per-arm parse-failure rate < ~2%, or explained as a
    known content-drop class per rule 9 (e.g. empty judge responses on
    degenerate steered text, #1769); (c) TRANSPORT PARITY (#2152) — the
    pilot must RUN the wave's transport: a ~200-draw pilot routes SYNC by
    call count while the wave it gates is typically forced BATCH, so the
    caller DECLARES the wave's dispatch (`wave_n_calls` /
    `wave_threshold_base` / `wave_force_sync` on `judge_pilot_gate`,
    mirroring the wave's actual dispatch kwargs 1:1), the gate computes
    the wave's route via `judge_dispatch.decide_route`, forces the pilot
    onto it (`threshold_base=0` for batch, `force_sync` for sync), reads
    the REALIZED route back from each arm's persisted
    `save_raw["routing"]` record, and FAILs on a mismatch or an
    unverifiable transport — never a silent pass. Cache-served draws are
    NOT transport evidence: a wave-declared pilot whose `save_raw`
    records `n_cached > 0` FAILs as transport-unverifiable (cached draws
    carry no routing provenance and, being refusal-free by construction —
    the #2151 cache PUT-SKIP — dilute the clause-(d) rate toward PASS);
    run pilots against a fresh pilot `cache_dir` (rule 24(ii)). An
    UNPINNED count-routed declaration inside the dispatcher's OTPM-probe
    region (`wave_n_calls < 2 × wave_threshold_base`, neither pin set) is
    REFUSED fail-fast before any pilot dispatch — the realized route
    there depends on a live OTPM probe at dispatch time and no pilot can
    certify it; PIN the wave's transport (`threshold_base=0` forces
    batch / `force_sync=True` forces sync), with the SAME pin on the
    production dispatch. Outside the region the count-routed route is
    deterministic batch (any ≥ ~5,000-call wave at the default
    `threshold_base=2,000` sits at n ≥ 2×tb — no pin needed). An
    UNDECLARED wave downgrades to a recorded report warning (legacy
    callers), but every NEW ≥ ~5,000-call wave MUST declare; and (d)
    per-arm API-REFUSAL rate (`n_api_refusal / (n_draws −
    n_transport_lost)`) < 0.10 — the rule-28 transport-conditional censor
    the transport-parity clause exists to expose (#1739: 34.1% batch-path
    censoring, 0/14,887 sync re-refusals; supersedes #2151's report-only
    treatment). Waivable per arm via `waive_api_refusal_arms` with the
    reason recorded at the caller-site constant (the #2091
    `PILOT_WAIVE_PARSE_FAIL_ARMS` pattern; a wave with a pre-planned
    rule-28 sync re-issue remediation is the legitimate waiver case);
    truncation and the effective-draws floor stay unwaivable. Record the
    pilot verdict — per-arm drop rate + `stop_reason` tally + the
    `max_tokens` used + the pilot's realized transport and the wave's
    declared transport — in the run
    digest / plan §6. Rationale: rule 23's binding check is POST-HOC
    (measured only after the full wave is spent), so every miss costs a
    full re-judge — three waves in one week (#1739: a surgical re-judge
    at 800 over 86,521×3 draws; #1769: the whole 21k-call fu1 wave
    re-judged at 600, ~1h wall + a second batch spend; #1774: a per-draw
    truncation-recovery merge) would each have been prevented by a
    ~200-call pilot. This is the judge-side twin of the MEASURED 1-cell
    pilot the compute-sizing rule mandates for wall-time
    (`.claude/rules/plan-compute-sizing.md` § Per-cell fit phases). Run
    the pilot against a fresh/pilot `cache_dir` (rule 24(ii)'s cache
    discipline) so production reuse is a deliberate decision, never a
    silent replay. As of #2021 the per-draw `stop_reason` is PERSISTED in
    every judge result and `eval/judge_pilot.judge_pilot_gate` implements
    this gate mechanically — per-arm parse-fail rates + `stop_reason`
    tallies read off the persisted fields (`JudgeResult.stop_reason_tally`
    / `n_truncation_dropped_draws`; a KEPT-but-truncated verdict is caught
    by the tally clause), truncation FAIL never waivable, report JSON for
    the run digest — so read the pilot's stop_reasons from the persisted
    results / the gate report, never from truncated failure-log text
    (rule 23, #1773). Exempt: score-only
    rubrics and waves < ~5,000 calls (the post-hoc per-arm drop report,
    rules 9/18/23, still binds there).

    **Size the pilot so the threshold is REACHABLE — the ~100–200 habit is
    not a sizing rule (#2124).** Clause (b) compares a per-arm RATE against
    a threshold, so a per-arm draw count that cannot RESOLVE that threshold
    makes the gate uninformative in BOTH directions: at `n` effective draws
    the smallest observable nonzero parse-fail rate is `1/n`, so an arm
    with `n <= 1/threshold` FAILs on its first parse failure (a granularity
    artifact, not a defect signal) while a clean PASS carries no evidence
    that the true rate is under threshold. Satisfiability is STRICT — the
    gate FAILs on `rate >= threshold`, so a single failure survives only
    when `1/n < threshold`, i.e. the per-arm floor is
    `required = max(min_effective_draws_per_arm, floor(1/threshold) + 1)` —
    **51** draws per arm at the default 2%, not 50. The shipped default
    pair is itself unsatisfiable (`parse_fail_threshold=0.02` against
    `min_effective_draws_per_arm=10`: 1/0.02 = 50 > 10).

    Realized per-arm draws are DISCRETIZED and ARM-SIZE-CAPPED, so neither
    `target_total_draws >= required · n_arms` nor `> n_arms / threshold` is
    sufficient. `eval.judge_pilot.judge_pilot_gate` splits its budget by
    floor division (`per_arm_items = target_total_draws // (n_arms ·
    n_draws)`) and then caps each arm at its own item count, so realized
    draws are `min(per_arm_items, len(arm_items)) · n_draws`. The exact
    budget form is **`target_total_draws >= n_arms · n_draws ·
    ceil(required / n_draws)`** (at 4 arms, `n_draws=2`, 2%: **208**, not
    204 — 204 realizes 50 draws/arm and still fails), and an arm holding
    fewer than `ceil(required / n_draws)` ITEMS cannot be fixed by any
    budget at all. Size from the arm count AND the arm sizes.

    Reachability is not certification: at exactly the floor (51 draws, 0
    failures) the 95% upper bound on the true rate is still ~5.7% (rule of
    three), and a healthy arm at a true 1% rate FAILs ~9% of the time at
    n=51 (it needs >= 2 failures). The floor buys gate COHERENCE, not
    evidence that the rate is under 2%; 2–3× the floor is what reduces
    granularity noise. (The statistically correct instrument for "is the
    true rate under 2%" is a one-sided exact binomial test rather than a
    point-rate comparison; it is deliberately NOT the gate today because it
    would change verdicts on configurations that already pass — #2124
    § Scope decisions.)

    The config-time guard sizes the PLANNED draws; realized `n_answered`
    can still shrink below `required` through rule-24 transport losses and
    rule-28 api-refusals (which run 30%+ in exactly the harm-class waves
    this gate serves), re-creating the granularity artifact after the guard
    has passed. The gate WARNs when that happens — treat it as an
    under-powered pilot, not a clean read.

    A pilot PASS certifies only the instrument it ran: rubric text, judge
    model, `n_draws`, `max_tokens`, and TRANSPORT (#2152). Any change to
    those invalidates it — re-pilot. (`scripts/issue2203_runtime.py` is
    the in-repo precedent: it fingerprints `rubric_sha + n_draws +
    max_tokens` and honors a prior PASS only on match; that fingerprint
    predates transport — a prior PASS does NOT cover a transport
    change.)

    When a satisfiable pilot is genuinely unaffordable for one arm, the
    escape is the AUDITABLE one — never a quietly loosened threshold: name
    the arm in the wave's `waive_parse_fail_arms` constant with a recorded
    reason (the `PILOT_WAIVE_PARSE_FAIL_ARMS` pattern,
    `scripts/issue2091_judge.py`; #2091 waived a 1/16 = 6.25% wildchat arm
    this way). The waiver is PARSE-FAIL only — truncation and the
    effective-draws floor stay unwaivable. Mechanically enforced since
    #2124 in `eval.judge_pilot.judge_pilot_gate` ONLY: it REFUSES an
    unsatisfiable configuration at config time, before any API spend,
    unless the caller passes `allow_subresolution_pilot=True`, which
    downgrades the refusal to a recorded report warning. Per-issue
    re-implementations of the gate do not inherit the guard — check yours.

    **Scope — every PARSED judge instrument, not only graded 0–100
    (#2124).** The gate binds any wave whose rows are parsed into a
    structured verdict: graded 0–100 scores, k-way CLASSIFICATION /
    labeling rubrics (rule 25's CATEGORY axis), and binary categorical
    verdicts alike. "Score-only rubrics" in the exemption means rule 23's
    bare-integer-no-rationale shape; a classification rubric is NOT exempt
    for producing no score — its parse surface is precisely what the pilot
    exists to test. #1739's 7-class MHJ tactic wave (10,666 contexts, far
    above the ~5,000 floor) shipped unpiloted: the v1 rubric asked for a
    plain `Label: <class>` line while the dispatch layer's
    `parse_judge_json` accepts JSON only, so **100% of rows failed to
    parse** and recovery needed a `--recover-from-raw` re-parse of the
    banked responses (`scripts/issue1739_tactic_classify.py`). A ~200-draw
    pilot would have surfaced the 100% parse-fail rate for ~2% of the
    wave's spend.

    This does NOT duplicate rule 27, though the two overlap on the easy
    cases — #1739's defect was also catchable offline, since round-tripping
    a realistic `Label: <class>` reply through `parse_judge_json` returns
    None. Rule 27's round-trip is a STATIC committed test that the parser
    accepts a canonical response the rubric asks for; rule 26's pilot is a
    LIVE dispatch at the exact production instrument. Only the live pilot
    catches a judge that ignores the schema it was handed — the case where
    the parser and the canonical response agree and the MODEL is what
    diverges.

27. **Round-trip the parse contract before trusting a composed judge
    instrument.** A dry run proves ROUTING, not the request/response
    CONTRACT. Any newly composed judge rubric/leg ships with a committed
    test that (a) pushes a REALISTIC reply (reasoning + score, plus a
    fenced/markdown variant) through the harness's OWN parse+reduce path
    (`parse_judge_json`, `src/explore_persona_space/eval/utils.py` →
    `_score_from_parsed`, `eval/graded_judge.py` — or the consumer's
    actual equivalent), and (b) presence-checks the user-template
    substitution placeholders (`{question}`/`{answer}` — the
    `graded_judge.py` `format_user_msg` `.replace` substitution) and
    asserts harness-identical substitution leaves no slot unfilled. The
    REQUEST side still needs a live probe (`.claude/rules/gotchas.md`
    mock-seam rules — a mock-judge smoke never validates the Batch API
    request shape); the RESPONSE side is validatable OFFLINE at zero API
    cost. (Incident #1345 rounds 3→4: two rubrics with clean dry runs
    carried 100%-draw-drop defects — no substitution placeholders, and a
    trailing `SCORE: <int>` shape against the harness's forced JSON
    contract, `parse_judge_json('...SCORE: 73') → None`; both fixed +
    test-pinned in `a41fcad04f`, 72 round-trip tests.)

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
    (rules 23/24), the per-draw `stop_reason` tally + its truncation-drop
    subset (`JudgeResult.stop_reason_tally` / `n_truncation_dropped_draws`,
    rules 23/26; #2021), the api-refusal count
    (`JudgeResult.n_api_refusal_draws` — the THIRD top-level drop class,
    reported separately from BOTH content drops and transport losses;
    rule 28, #2151), the per-item completeness `frac_items_complete` per
    behavior / arm against its pre-registered floor (rule 29, #2124), the
    rule-26 pilot-gate verdict for any ≥5k-call wave,
    per-behavior reliability
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
  (≥ 2048 for multi-field reason-then-score JSON rubrics / ≥ 1024 for
  single-rationale rubrics — floors raised from 600/300 on 2026-08-02 —
  or a stated justification; the floor is a
  floor, not a guarantee: the post-resize per-arm drop re-measure binds at
  ANY budget, #1739) and its per-arm drop-rate report;
  the Statistics & Measurement critic REVISEs an unsized reasoning-rubric
  judge. Plan-enforced in v1 — no mechanical lint.
- Rule 26 (pilot gate) rides the same lens load: a plan whose judged DV
  dispatches ≥ ~5,000 judge calls names its pilot gate (pilot size, arms
  spanned, the gate thresholds, and the wave-transport declaration —
  #2152) or states the exemption; the
  Statistics & Measurement critic REVISEs an ungated large wave.
  Plan-enforced in v1 — no mechanical lint (same class as rules 23/24);
  the `judge_pilot_gate` helper is #2021's deliverable.
- Rule 24 (transport-vs-content split) rides the same lens load: a plan whose
  judged-DV pipeline persists API/transport errors as dropped draws — or
  whose per-arm drop report does not split content-drops from
  transport-losses — is a Statistics & Measurement REVISE. Plan-enforced in
  v1 — no mechanical lint (same enforcement class as rule 23).
- Rule 27 (parse-contract round-trip) is plan-enforced in v1 — no mechanical
  lint (same class as rules 23/24/26): a composed judge instrument's
  implementer smoke evidence includes the committed round-trip test
  (smoke-contract mirror: `experiment-implementer.md` § "End-to-end smoke
  run PER PHASE"); dry-run-only evidence for a composed judge leg is the
  named insufficient shape.
- Rule 28 (api-refusal drop class) rides the same lens load: a plan whose
  judged DV scores harm / jailbreak / adversarial-role-play / evil-trait-
  or toxicity-banded completions
  names its api-refusal accounting — per-arm `n_api_refusal`, reported
  separately from BOTH content drops and transport losses — plus the
  targeted SYNC re-issue remediation at the IDENTICAL instrument
  (reference implementation:
  `scripts/issue1739_evilood_refusal_rejudge.py`), or states the
  exemption; the Statistics & Measurement critic REVISEs a harm-class
  judged-DV plan with no api-refusal accounting. A wave-DECLARED rule-26
  pilot PASS (transport parity + the api-refusal clause, #2152) is
  corroborating evidence but not a substitute for the accounting; an
  UNDECLARED pilot's PASS is NOT protective (rule 28's coverage note).
  Mechanical backstop: `verify_plan.py` c53 WARNs on the
  missing-handling shape (WARN-only; the lens REVISE is the binding
  gate).
- Rule 29 (per-item completeness floor) rides the same lens load: a plan
  whose judged DV carries a headline names its per-item completeness
  accounting + its pre-registered floor, or states the exemption; an arm
  plotted below the floor without drop-class triage is a Statistics &
  Measurement REVISE. Plan-enforced in v1 — no mechanical lint, same class
  as rules 23/24/28.
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
task body #1739 (the above-floor 400-token truncation residue behind the
rule-23 measured point and the REFUSAL tally split);
task bodies #1769 + #1774 (the two further truncation re-judge waves that,
with #1739, drove the 2026-08-02 generous-floor raise and rule 26's pilot
gate); task body #1934 (the #1773 log-derived misdiagnosis behind rule 26's
stop_reason-from-raw-responses requirement); task #2021 (stop_reason
threading + truncation-vs-content drop split + the `judge_pilot_gate`
helper); task #2124 (the rule-26 sizing + scope clauses, rule 29 per-item
completeness, and the config-time satisfiability guard +
`allow_subresolution_pilot` escape in `judge_pilot_gate`);
task body #1482 (the category-axis confusable-neighbor incident behind
rule 25);
task #1345 (the composed-instrument parse-contract defects behind rule 27;
fix `a41fcad04f`, 72 round-trip tests) + task #1943 (the rule);
`.claude/rules/persona-vectors-recipe.md` (the graded-judge precedent +
judge-filter drop rule); `.claude/rules/marker-leakage-measurement.md` (the
non-judged marker DV); the enforcing agent files (`planner.md`, `critic.md`,
`analyzer.md`, `interpretation-critic.md`, `clean-result-critic.md`).
A blinded / unprimed qualitative read is NOT a judged-behavior DV (no score,
no rubric); its recipe is `.claude/rules/blinded-reads.md` (#2143).
