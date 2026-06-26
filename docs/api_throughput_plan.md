# Plan: Anthropic API max-throughput characterization + unified dispatcher

**Status:** proposed (kind: infra)
**Author:** drafted 2026-06-25 (rev 2 — folded independent review + docs verification)
**Goal:** Empirically derive guidelines for completing *N* Anthropic API calls in
minimum wall-clock (subject to a cost budget), exploiting the three org-keys
(low-prio / high-prio / batch) and all available models, then integrate a
unified auto-routing dispatcher into the codebase so every judge / generation
path gets the fast route for free.

Prompted by the #658 judge stalling: 141,600 calls submitted as 8k batches
starved in Anthropic's opportunistic batch queue (succ=0 for 8h), while 500-req
batches cleared in ~5 min. We want a *measured* answer, and the review of rev 1
showed the naive "chunk size is the variable" reading is probably a confound —
see §1.

---

## 1. Established facts (measured live + docs-verified, 2026-06-25)

- **The 3 keys are 3 SEPARATE ORGS → Messages rate limits are ADDITIVE.** Limits
  are enforced **per organization** (docs), so additivity holds *only* because
  these are distinct orgs (a 2nd key in the same org buys nothing). Measured from
  `anthropic-ratelimit-*-limit` headers:

  | Key (env var) | org id | RPM | ITPM | OTPM |
  |---|---|---|---|---|
  | `ANTHROPIC_API_KEY` ("high-prio") | b94495b5… | 30,000 | 10,000,000 | 1,000,000 |
  | `ANTHROPIC_BATCH_KEY` ("batch") | b4bbba0c… | 90,000 | 8,000,000 | 800,000 |
  | `ANTHROPIC_API_KEY_LOW_PRIO` | TBD (not in `.env`) | TBD | TBD | TBD |

  Both orgs have **custom elevated limits** (30k/90k RPM ≫ standard Tier-4's
  4k RPM), so the public tier table is a *lower bound* — Phase 0 reads each org's
  real per-model limits via the **Rate Limits API**, not assumptions.
- **Neither key is Anthropic "Priority Tier"** (both return `service_tier:
  "standard"`; no `anthropic-priority-*` headers). "priority" naming = org quota,
  not the committed-capacity product.
- **Batch tier is NOT limit-free** (rev-1 error). The Message Batches API has
  its own RPM (all endpoints) **and** a per-org **"batch requests in processing
  queue" cap** (Tier 1 = 100k, T2 = 200k, T3 = 300k, T4 = 500k; max 100k requests
  per single batch). A request counts against the queue until successfully
  processed. Batch ITPM/OTPM are decoupled from the Messages buckets, but the
  queue cap + batch RPM are hard limits, and the underlying scheduler is shared
  opportunistic spare capacity.
- **Starvation root cause is UNKNOWN — do not assume it's chunk size.** Three
  candidate mechanisms, to be disentangled experimentally:
  1. **Total in-flight queue depth vs the org's batch queue cap** — submitting
     all 141,600 at once could exceed a low-tier cap. (But: the single 8k batch
     that starved had only 8k in flight, far under any cap, so this can't be the
     *whole* story.)
  2. **Large-batch admission deprioritization** — the scheduler may pass over a
     big batch to protect the latency-sensitive sync tier (fits the single-8k
     observation).
  3. **Time-of-day spare-capacity load** — opportunistic capacity varies.
  The fix is to MEASURE, not guess: read the queue cap + sweep queue depth +
  size + repeat across time windows.
- **Acceleration limits are real** (docs): a *sharp* usage increase triggers
  429s independent of steady-state RPM/ITPM/OTPM — "ramp up gradually." Any
  concurrency sweep MUST warm up, or it measures transient acceleration-429s, not
  the true sustained knee.
- **Caching is inert for the judge workload.** Cache-*read* tokens don't count
  against ITPM (non-Haiku-3.5), but the project judge rubric is ~120–400 tokens —
  **below the 1,024-token cacheable minimum** — so prompt caching gives nothing
  here. Exclude caching from the throughput model for current rubrics (revisit
  only if a future judge prompt exceeds the cache minimum).
- **Model ceilings differ and are family-pooled.** Tier-4 reference: Sonnet 4.x
  2M ITPM / 400k OTPM; Haiku 4.5 4M / 800k; Opus 4.x 10M / 800k. **Sonnet 4.x is
  ONE shared pool across 4.6+4.5; Opus is one pool across all Opus versions.**
  So the concurrency knee (set by the binding ceiling) does NOT transfer from
  Haiku to Sonnet — characterize on the **actual judge model (Sonnet 4.5)**.
- **`max_tokens` does not count against OTPM** (only real output tokens) — no
  throughput penalty for a generous `max_tokens`.
- **safety-tooling pattern** (`~/introsp/safety-tooling/.../batch_api.py`):
  `chunk_prompts_for_anthropic` (default 100k/250MB) + `asyncio.gather` over
  chunks + per-prompt content-hash cache. Borrow gather + cache; reject its
  huge default chunk + deadline-less poll (the wedge we fixed in `#663`).
- **Existing code already bakes in the #658 failure shape:**
  `eval/judge_dispatch.py` pins `DEFAULT_SUB_BATCH_SIZE = 8_000` and
  `MAX_CONCURRENT_SUB_BATCHES = 8`, and constructs clients only from
  `ANTHROPIC_API_KEY` (single-org, hardcoded in ~4 places). The dispatcher work
  is therefore larger than "extend judge_dispatch" and must re-tune these
  constants from Phase 3 before any migration.
- **Models available** (live `/v1/models`): fable-5, opus-4-8/4-7/4-6/4-5,
  sonnet-4-6, haiku-4-5, sonnet-4-5 (judge), opus-4-1, sonnet-3-5 (new/old),
  sonnet-3. (Haiku 3.5 is retired off first-party.) Verify older models are
  actually callable on these orgs in Phase 0 before planning model-effect cells.
- **The org keys are SHARED with other fellows** (same org budgets serve
  multiple people). So the operative concurrency cap is SOCIAL (don't hog the
  shared bucket), not the org RPM ceiling. This makes the "saturate 150k RPM with
  a big process pool" idea WRONG — use a polite per-key slice and fan across the
  3 orgs. See § Operating defaults.

## 1b. Operating defaults (shared-key etiquette) — FIRM, baked into the dispatcher

These are the standing defaults for ALL high-volume API calls (the dispatcher
ships with them; configurable via env/args). They supersede the rev-1 "maximize
throughput" framing because the org keys are shared with other fellows.

- **Per-key concurrency caps (default; from a fellow's guidance + matches our
  measured single-process knee ~100→~3k RPM):**
  - Sonnet 4.x: **~100 concurrent / key**
  - Haiku 4.5: **~120 concurrent / key**
  - Opus 4.x: **~40 concurrent / key**
  Configurable (`EPS_API_CONC_<MODEL>` or dispatcher args); these are the
  good-citizen defaults, NOT hard API limits.
- **Fan out across the 3 org-keys at those per-key caps** (≈3× the slice), NOT a
  big process pool chasing the org RPM ceiling (that would starve colleagues).
  Workers ≈ one per org-key; multi-process only if a single process can't drive
  ~100 concurrency efficiently.
- **429 ⇒ REDUCE concurrency (mandatory adaptive back-off).** On any 429, the
  dispatcher MUST cut that key's concurrency (AIMD: multiplicative decrease —
  e.g. halve — on a 429, then additive/gradual increase back toward the cap while
  clean), and honor the `retry-after` header before retrying. Never hold a fixed
  concurrency through 429s. Because the keys are shared, also **watch the
  `anthropic-ratelimit-*-remaining` headers and ease off when remaining is low**
  (a colleague is active) — so we never 429 someone else. This is the key
  runtime safety rule.
- **Caching / resume is mandatory** so any run is interruptible/restartable:
  per-item content-hash cache (skip already-completed items on restart) + atomic
  checkpoint writes (temp-file-then-rename, so a crash/full-disk leaves the last
  good checkpoint intact — validated by the #658 judge surviving a disk-full).
- **Polite-throughput estimates** at these caps × 3 orgs: Sonnet ~9k RPM
  (#658's 141.6k ≈ ~16 min), Haiku ~36k RPM, Opus ~3.6k RPM.
- **Batch is the *considerate* path for huge / latency-tolerant jobs** — it runs
  on Anthropic spare capacity OUTSIDE the shared rate budget, so it won't starve
  fellows' interactive calls. Crossover to batch is driven by etiquette + cost,
  not just raw speed.

## 2. Open questions the experiments must answer

1a. **Sync additivity:** do the orgs' per-org Messages token buckets add (spend
   on org A leaves B/C `remaining` untouched)? Expected yes.
1b. **Batch additivity (separate mechanism):** does batch throughput add across
   orgs, or is it bounded by a shared global spare-capacity scheduler that
   doesn't partition per-org? Must test with *simultaneous* submission (load
   confounds any sequential comparison).
2. **Sync knee, per org × model:** with a warm-up ramp, what sustained
   concurrency maximizes calls/min before steady-state 429s, and **which limit
   binds first** (RPM vs OTPM vs ITPM) for the judge's token shape? (Prediction:
   short-output judge calls are **RPM- or OTPM-bound, not ITPM-bound** — so the
   18M combined ITPM is *not* the lever; RPM/OTPM is.)
3. **Batch admission:** disentangle the three starvation mechanisms (§1) — sweep
   total in-flight queue depth (primary axis), batch size (secondary), and time
   window; find the queue-depth knee and whether small+parallel actually helps
   *after* controlling for total queue depth.
4. **Sync-vs-batch crossover:** for each *N* × model, which tier finishes first
   and at what cost?
5. **Model effect:** Haiku 4.5 vs Sonnet 4.5 vs Opus — per-call latency, ceilings,
   cost — to populate the model dimension of the guidelines.

## 3. Metrics + methodology (every experiment)

- **Primary:** wall-clock to obtain all *N* results.
- **Secondary:** throughput (completed calls/min, output-tokens/min), 429/529
  rate **split into acceleration-429 (transient, during ramp) vs steady-state-429**,
  retries, total cost (USD), p50/p95 per-call latency (sync), and **which
  `anthropic-ratelimit-*-remaining` header hit ~0 at the knee** (binding-limit
  attribution — required, per Q2).
- **Warm-up ramp (mandatory for sync):** start at low concurrency and step up
  every ~30 s until throughput plateaus or steady-state 429s appear; never jump
  straight to high concurrency (acceleration limits).
- **Anchor model = Sonnet 4.5** (the judge) for the knee/throughput
  characterization, since its ceilings are the real bottleneck and don't transfer
  from Haiku. Use **Haiku 4.5 only** for the *structural* tests (additivity,
  queue-cap/queue-depth) where the ceiling magnitude is irrelevant and cost
  should be minimized.
- **Fixed token shape per comparison** (same prompt + same realistic completion
  length + same `max_tokens`) so token rates are comparable; state the predicted
  binding limit up front.
- **Caching excluded** from the model for current (<1024-token) rubrics; note it
  as a future lever only if a judge prompt grows past the cache minimum.
- **Time-of-day:** run Phases 1–2 in ≥2 windows (peak + off-peak); report ranges,
  not point estimates.
- **Cost discipline + arithmetic:** probes use short prompts + small `max_tokens`.
  Per-phase cost is computed as `N × (in_tok × $in + out_tok × $out)` at each
  model's price and shown in the experiment JSON, not asserted. Rough budget:
  Phase 1 (~30k Sonnet calls + structural Haiku) ≈ $15–30; Phase 2 (grid ~400k
  Haiku calls + cliff probes, **plus a resubmission reserve for batches that
  expire at 24h**) ≈ $20–40; Opus points capped at low concurrency. Total
  ~$50–100, refined after Phase 0 prices. No per-run dollar cap (project rule);
  log spend.
- **Reproducibility:** each experiment writes `eval_results/api_throughput/<exp>.json`
  (config + metrics + raw timings + binding limit) and a plot to
  `figures/api_throughput/`.

## 4. Phases

### Phase 0 — Setup + enumeration (≈$1, ~30 min)
- Add `ANTHROPIC_API_KEY_LOW_PRIO` to `.env` (user adds the literal). Probe its
  org id, limits, tier; fill §1.
- **Read real limits via the Rate Limits API** (`/docs/en/manage-claude/rate-limits-api`),
  per org × model: RPM/ITPM/OTPM **and the batch RPM + batch processing-queue
  cap** (the queue cap is NOT in Messages headers — this is the #658-relevant
  number). Build the authoritative limits matrix.
- Add a small **`remaining`-header reader** helper (the existing `probe_otpm_limit`
  returns only the *limit*; we need live `*-remaining` for additivity + adaptive
  concurrency).
- **Sync additivity check (Q1a):** burn known ITPM on org A; confirm B/C
  `remaining` unaffected.
- **Verify callability** of the older models (sonnet-3-5, sonnet-3) on these orgs
  before planning model cells.

### Phase 1 — Sync throughput microbenchmarks (≈$15–30, ~2 h × 2 windows)
- Harness: async client per org, bounded `Semaphore(c)`, **warm-up ramp**, fixed
  judge-shaped prompt (realistic input + `max_tokens`≈256), *N*≈2,000.
- Sweep `c` upward with the ramp; record sustained throughput, steady-state vs
  acceleration 429s, and the binding `*-remaining` header. Find the per-org knee
  on **Sonnet 4.5** (primary) and one point each on Haiku 4.5 + Opus 4.8 (capped).
- **Fan-out test (Q1a):** all 3 orgs concurrently at their knees on one *N*=10,000
  job; confirm aggregate ≈ sum.

### Phase 2 — Batch admission + throughput (≈$20–40, ~1–2 days wall, idle)
- **Primary axis = total in-flight queue depth** (NOT chunk size): sweep
  N_in_flight ∈ {0.5k, 2k, 8k, 50k, 200k, …} against each org's measured queue
  cap; measure time-to-first-success + time-to-complete. Secondary axis = chunk
  size {500, 2k, 8k} *at matched total depth* to isolate whether chunking matters
  once depth is controlled.
- **Starvation-cliff probe:** submit single batches of {500, 2k, 8k, 50k}
  simultaneously; time-to-first-success vs size → admission curve (tests
  mechanism (2) at fixed low depth).
- **Batch additivity (Q1b):** identical batch set to 1 org vs split across 3 orgs,
  submitted **simultaneously**; compare time-to-complete (controls for load).
- Keep the `#663` bounded-deadline poll; treat 24h **batch expiration as an
  expected outcome at high depth**, not an error (resubmit the expired slice).
- Account for **poll RPM cost** at high concurrent-batch counts (retrieves count
  against batch RPM).

### Phase 3 — Crossover analysis + guidelines (≈$0, ~2 h)
- Build the **decision table**: *N* ∈ {10, 100, 1k, 10k, 100k, 1M} × model ×
  {latency-critical, cost-critical} → fastest config (tier, key(s), chunk,
  concurrency, predicted wall-clock + cost + binding limit).
- Identify crossover *N*. Expected: tiny *N* → sync on one org; medium → sync
  fan-out across orgs; very large / cost-sensitive → batch chunked+parallel
  across orgs at a depth below the queue cap.
- Write `docs/api_throughput_guidelines.md` + a CLAUDE.md pointer. **Feed the
  chosen batch chunk + concurrency back into `judge_dispatch.py`'s
  `DEFAULT_SUB_BATCH_SIZE` / `MAX_CONCURRENT_SUB_BATCHES`** (they currently
  reproduce the #658 failure).

### Phase 4 — Unified dispatcher (~1.5–2 days)
- New `src/explore_persona_space/llm/api_dispatch.py` exposing
  `dispatch_calls(items, model, deadline=None, cost_pref="balanced")`. It must:
  - **Multi-org key pool** — thread a key/client through the sync path, batch
    submit, the limits probe, AND the retry re-entry (currently `ANTHROPIC_API_KEY`
    is hardcoded in ~4 places; this is the bulk of the work).
  - **Per-key concurrency caps + 429 back-off per § 1b (FIRM defaults):** start
    at the per-key caps (Sonnet ~100 / Haiku ~120 / Opus ~40), AIMD-reduce that
    key's concurrency on every 429 (multiplicative cut + honor `retry-after`,
    gradual recovery), and ease off when the shared `*-remaining` headers run low.
    Warm-up ramp to dodge acceleration 429s.
  - **Route** by (N, model, deadline, cost_pref) per the Phase-3 table.
  - **Batch path** — small chunks + `asyncio.gather` with a concurrency cap that
    keeps total in-flight **below each org's queue cap**, `#663` bounded poll,
    per-item content-hash cache/resume.
  - **Org-aware resume** — persist which org each sub-batch was submitted to (a
    batch created on org B must be re-polled on org B, else resume 404s); the
    current `state.json` has no org dimension.
- Pair with `code-reviewer`; unit-test routing/chunking/backoff/org-resume (mock client).

### Phase 5 — Integrate + migrate (~0.5 day)
- Route existing callers through the dispatcher: `eval/batch_judge.py`, the #658
  judge, #664's eval judge, the EM / sycophancy / refusal / trait judges. Keep
  `judge_completions_batch`'s public signature; swap internals. Judge model stays
  Sonnet 4.5 (project rule); the dispatcher chooses tier + keys + concurrency.

### Phase 6 — Validate on a real workload (~hours)
- Re-run the #658 judge (141,600 calls) through the dispatcher **after** Phase 3
  re-tunes the chunk/concurrency constants. Controlled acceptance: run a
  **same-session back-to-back baseline** (current sequential 500-shard config)
  vs the dispatcher at **matched tier + model + N**; report wall-clock speedup
  **and cost separately** (do NOT claim "equal cost" — tiers price differently).
  Target: ≥5× wall-clock vs the sequential baseline, no data loss, resumable.

## 5. Deliverables
1. `docs/api_throughput_guidelines.md` — decision table (N × model × priority →
   config) with measured throughput + cost + binding limit per config.
2. `eval_results/api_throughput/*.json` + `figures/api_throughput/*` — raw data +
   plots (sync knee per org/model, batch queue-depth curve, cliff, crossover).
3. `src/explore_persona_space/llm/api_dispatch.py` — multi-org dispatcher.
4. Migrated judge paths; re-tuned `judge_dispatch.py` constants; CLAUDE.md pointer.

## 6. Risks / caveats
- **Mis-attribution (the rev-1 trap):** don't credit chunk size for what is
  queue-depth / admission / load. Phase 2's primary axis is queue depth, with
  size controlled — guard against shipping a chunk-size guideline that's really a
  depth effect.
- **Binding-limit error:** leading with ITPM is wrong for short-output judge
  calls (RPM/OTPM bind first). Every guideline states its binding limit.
- **Acceleration 429s** masquerade as the steady-state knee → always ramp.
- **Batch additivity may fail** even though sync additivity holds (different
  mechanisms) — measured in Phase 2, simultaneously.
- **Time-of-day** variance → ≥2 windows, report ranges; the dispatcher reads live
  headers so it adapts at runtime.
- **Cost** (~$50–100) with explicit arithmetic + a batch-expiration resubmission
  reserve; cap Opus probes.
- **Dispatcher scope** is larger than it looks (single-org hardcoding + org-aware
  resume) — scope honestly, code-review it.
- **Key exposure:** the 3 keys were pasted in chat → rotate after this work. The
  dispatcher reads only `.env`; no key in code/logs/commits.
- **Don't destabilize live runs:** Phase 6 runs on a fresh checkpoint, never by
  hijacking the in-flight #658 sequential run.

## 7. Sequencing
Phases 0–3 (characterize + guidelines) are independently valuable and ship before
Phase 4–6 (dispatcher). The current #658 sequential 500-shard run is the safe
fallback and is untouched until Phase 6 has a validated faster path.
