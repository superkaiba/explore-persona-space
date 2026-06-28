# Anthropic API throughput guidelines

**Phase 3 of `docs/api_throughput_plan.md`.** Decision table for completing
*N* Anthropic Messages calls in minimum wall-clock, derived from the Phase 0
limits + Phase 1 measurements (`eval_results/api_throughput/*.json`) and from
the FIRM operating defaults in the plan's § 1b. Targeted by the unified
dispatcher (`src/explore_persona_space/llm/api_dispatch.py`); judge callers
inherit it through `src/explore_persona_space/eval/judge_dispatch.py`.

These are RECOMMENDATIONS, not hard API limits. The org keys are shared with
other fellows, so the binding constraint is etiquette (§ 1b), not the org
RPM ceiling.

---

## 1. The settled architecture

Three structural findings make routing nearly deterministic:

1. **Three keys = three SEPARATE orgs → additive Messages limits.** Phase 0
   confirmed distinct `anthropic-organization-id` values. The Sonnet 4.5
   combined ceiling across the three keys is **~150k RPM / 33M ITPM / 3M OTPM**;
   Haiku 4.5 ~64k RPM / 40M ITPM / 4M OTPM; Opus 4.8 ~304k RPM / 190M ITPM /
   24M OTPM. Source: `eval_results/api_throughput/phase0_limits.json`.

2. **The throughput plateau is CLIENT-bound, not server-bound.** A
   single-process asyncio client plateaus at ~1.5k RPM/org regardless of the
   nominal concurrency (n=8000 sweep ramping to concurrency 800 produced
   1414 RPM with p50 latency 17s — the org's 30k RPM headroom went untouched).
   Two simultaneous processes on the SAME org gave 6440 RPM combined (≈2×
   single proc, no 429s). The fast path is therefore a **process pool ×
   org fan-out**, NOT one big asyncio loop chasing the org ceiling.
   Source: `phase1_high_prio_claude-sonnet-4-5_n8000.json`,
   `phase1_high_prio_claude-sonnet-4-5_n2500.json`,
   `phase1_high_prio-batch_claude-sonnet-4-5_n4000.json`.

3. **Polite per-key caps (§ 1b, FIRM) supersede "maximize throughput."** The
   org keys are shared. Per-key concurrency caps stand at Sonnet ~100 /
   Haiku ~120 / Opus ~40, with AIMD multiplicative back-off on every 429 and
   `*-remaining` header watch. The dispatcher fans across the 3 orgs at those
   caps; multi-process scaling is the polite way to push the WALL-CLOCK
   throughput, not "wider" asyncio.

The headline practical consequence: Sonnet 4.5 at the polite caps clears
**~9k RPM** across 3 orgs single-process; **~28-30k RPM** at ~3-4 processes
per org (still well under the 150k combined ceiling). #658's 141.6k judge
calls clear in ~5 min at the polite ceiling, vs ~14h on the original
8k-batches-of-8 path that starved.

---

## 2. Decision table

Recipe per (*N*, model, priority). "Tier" = sync fan-out across orgs (the
default fast path) vs batch (the considerate path for huge / latency-tolerant
N). "Keys" = how many of the 3 org keys to fan across. "Conc" = per-key
concurrency. "Procs" = process pool size when sync is multi-process.
"Wall-clock" assumes warm-up ramp + steady state; "binding" = which limit
binds first under the polite cap.

### Sonnet 4.5 (judge — the project default)

Per-call shape: ~120-400 input tokens, ~64-256 output tokens, ~1.5s p50.
Single-proc per-key throughput ≈ 1.5k RPM at conc 100.

| N | Latency-critical | Cost-critical |
|---|---|---|
| **10** | sync, 1 key, conc 10, 1 proc — **<5s**; binding: trivial | same |
| **100** | sync, 1 key, conc 100, 1 proc — **~5-10s**; binding: client (single-proc plateau) | same |
| **1,000** | sync fan-out, 3 keys, conc 100/key, 1 proc/key — **~20s**; binding: client (per-proc plateau) | same |
| **10,000** | sync fan-out, 3 keys, conc 100/key, **2 procs/key** — **~1.5-2 min**; binding: client (still client; the orgs' RPM ceilings sit ~10× above the realized rate) | sync (batch is slower at this N: 24h SLA worst-case > 2 min); avoid forced batch |
| **100,000** | sync fan-out, 3 keys, conc 100/key, **4 procs/key** — **~6-8 min**; binding: client first, then OTPM (3M/min across 3 keys) as procs ramp | sync at ≥3 procs/key — batch wins ~50% on $$ but loses ~3 hours of wall-clock per scheduling window |
| **1,000,000** | sync fan-out, 3 keys, conc 100/key, **6-8 procs/key** (watch `remaining` headers — AIMD will back off automatically) — **~60-90 min**; binding: org OTPM (~3M/min across 3 keys for ~250 out-tok/call) | batch, 3 keys, sub-batch size 1,000, ≤4 concurrent sub-batches/key — **~1-24h** (one Anthropic SLA window) at the 50% batch discount; binding: batch processing-queue cap |

Crossover N (Sonnet 4.5): **~200,000** under `cost_pref="balanced"`. Below
that, sync fan-out wins on both wall-clock AND total operator-time (no batch
SLA to wait on); above it, batch's cost win starts to dominate if latency
tolerance >24h exists.

### Haiku 4.5

Per-call shape: ~half the latency of Sonnet (~0.7-0.8s p50 at conc 120).
Single-proc per-key throughput ≈ 2-3k RPM at conc 120 (Haiku has tighter
OTPM bookkeeping on `high_prio` — Phase 0 saw an OTPM-exhaustion 429 at
probe time; budget headroom).

| N | Latency-critical | Cost-critical |
|---|---|---|
| **10** | sync, 1 key, conc 10 — **<3s** | same |
| **100** | sync, 1 key, conc 100 — **~3-5s** | same |
| **1,000** | sync fan-out, 3 keys, conc 120/key — **~10-15s** | same |
| **10,000** | sync fan-out, 3 keys, conc 120/key, 1-2 procs/key — **~45-90s** | sync |
| **100,000** | sync fan-out, 3 keys, conc 120/key, **3-4 procs/key** — **~3-5 min**; binding: client / OTPM | sync (batch is cheaper but adds SLA latency) |
| **1,000,000** | sync fan-out at the polite cap — **~30-50 min**; binding: OTPM (~4M/min combined) | batch, 3 keys, sub-batch 1,000 — **~1-24h** at 50% discount |

Crossover N (Haiku 4.5): **~500,000**.

### Opus 4.8 (use sparingly — ~5-10× more expensive than Sonnet)

Per-call shape: ~3-4s p50 at conc 40. Per-key throughput ≈ ~600 RPM
single-proc at the polite cap. (We have no measured sweep; numbers extrapolated
from Phase 0 limits + § 1b. Refine via a future Phase 1 Opus probe.)

| N | Latency-critical | Cost-critical |
|---|---|---|
| **10** | sync, 1 key, conc 10 — **~5-10s** | same |
| **100** | sync, 1 key, conc 40 — **~10-15s** | same |
| **1,000** | sync fan-out, 3 keys, conc 40/key — **~30s-1min** | same |
| **10,000** | sync fan-out, 3 keys, conc 40/key, 2 procs/key — **~5-10 min** | sync (batch SLA dominates wall-clock at this N) |
| **100,000** | sync fan-out, 3 keys, conc 40/key, 4 procs/key — **~45-90 min** | batch, 3 keys, sub-batch 1,000 — 50% off, ~24h SLA |
| **1,000,000** | batch (sync is wall-clock-expensive without justification) | batch |

Crossover N (Opus 4.8): **~50,000** — Opus is the only model where batch
makes sense at moderate N, because each call is dear and sync at the polite
cap is slow.

---

## 3. Routing rules (the dispatcher implements these)

The unified dispatcher (`src/explore_persona_space/llm/api_dispatch.py`)
implements the table via `decide_dispatch_route()`:

| Knob | Sync wins when | Batch wins when |
|---|---|---|
| `cost_pref="latency"` | always | never |
| `cost_pref="cost"` | N is tiny (< crossover_n/10), OR deadline < 24h SLA | otherwise |
| `cost_pref="balanced"` (default) | N < crossover_n, OR deadline < 24h SLA | N >= crossover_n |
| `force_path="sync"` / `force_path="batch"` | hard override (tests / ops) | hard override |

`SYNC_BATCH_CROSSOVER_N` should be set to **20,000** (the dispatcher's
current placeholder is 2,000 — too low; it forces batch on jobs that sync
fan-out clears in seconds). The 20k value is the conservative
within-model crossover: Sonnet 4.5 sync at 3 keys × 2 procs × conc 100 is
~3-5 min for 20k calls, while the batch SLA is at minimum minutes-to-hours
of opaque opportunistic queuing on top of submission overhead.

Per-family concurrency caps (`DEFAULT_FAMILY_CONCURRENCY`) are unchanged —
Sonnet 100, Haiku 120, Opus 40. These ARE the operative bound under shared
keys; tightening them only matters when a colleague is active (the
`*-remaining` watch + AIMD do that automatically).

Batch sub-batch size: **1,000** (the dispatcher's current default). Never
8,000 — that's the #658 starvation shape. The Anthropic queue admits small
batches faster, and 1,000 is well under any org's batch processing-queue
cap.

`MAX_CONCURRENT_SUB_BATCHES` (the dispatcher's batch-path concurrency cap):
**4** per org. With 3 orgs × 4 sub-batches × 1,000 = 12,000 simultaneously
in-flight batch requests — comfortably under any tier's queue cap (Tier 1 =
100k floor).

---

## 4. Re-tuning the existing judge dispatcher (`judge_dispatch.py`)

`src/explore_persona_space/eval/judge_dispatch.py` is the legacy single-org
judge dispatcher. Two constants need re-tuning off the Phase 3 numbers:

- `DEFAULT_SUB_BATCH_SIZE`: already `2_000` (updated after #658). Hold.
- `MAX_CONCURRENT_SUB_BATCHES`: currently `8`. Reduce to **4** to match the
  multi-org dispatcher (§ 3) and to leave headroom on the shared keys; with
  4 × 2k = 8k in-flight (vs the prior 8 × 2k = 16k), we stay well clear of
  any queue cap while remaining ~13% of the Tier-4 floor cap.

The legacy `DEFAULT_THRESHOLD_BASE = 2_000` (the sync-vs-batch threshold
inside the SINGLE-org judge dispatcher) is reasonable for that single-org
path. The MULTI-org dispatcher's higher `SYNC_BATCH_CROSSOVER_N = 20_000`
(§ 3) reflects the 3× fan-out throughput that the legacy single-org path
doesn't have.

---

## 5. Migration: pointing judge callers at the multi-org dispatcher (Phase 5)

`eval/batch_judge.py` and downstream judges (sycophancy, refusal, trait,
EM) should route through `llm.api_dispatch.dispatch_calls(...)` instead of
the single-org `judge_completions_batch` for any N where the multi-org
fan-out wins (N > ~1,000). The migration keeps the public signatures of
`judge_completions_batch` etc. — only the internals change.

Judge model stays **`claude-sonnet-4-5-20250929`** (the project's standing
judge rule); the dispatcher chooses tier + keys + concurrency, never the
model.

---

## 6. What we have NOT measured (Phase 1/2 residual)

These remain TBD; the table above uses the conservative estimate where
relevant:

- **Multi-process scaling beyond 2 processes/org** on Sonnet. The single
  measured 2-proc point (6440 RPM, ≈2× single proc) suggests near-linear
  scaling up to ~4 procs (the org RPM ceiling sits ~10× above the realized
  rate). Anything beyond 4 procs/org needs a real sweep.
- **Haiku and Opus knees.** Phase 1 only swept Sonnet 4.5. Haiku and Opus
  numbers in the table are extrapolated from Phase 0 limits + § 1b caps.
- **Batch queue depth admission curve.** Phase 2 was not run (the queue
  cap is not in any response header on a non-admin key). The §2 batch
  routing relies on Anthropic's published Tier-4 floor (500k queue cap)
  and the small-sub-batch-clears-fast observation from #658.
- **Time-of-day variance** on batch admission. Run any large batch job
  twice if the result will gate a high-stakes decision.

These are all lower-priority follow-ups; the strategy is settled.

---

## 7. The shared-key etiquette rules (recap, always-on)

These ride on every call site that uses the dispatcher:

- Per-key concurrency caps: Sonnet 100 / Haiku 120 / Opus 40.
- AIMD on every 429: multiplicative cut, honor `retry-after`, additive
  recovery while clean.
- Watch `anthropic-ratelimit-*-remaining` and ease off when remaining
  fractional headroom drops below 0.15 (a colleague is active).
- Fan across the 3 orgs by headroom; round-robin on ties.
- Caching/resume mandatory: per-item content-hash cache + atomic
  checkpoint writes (the dispatcher provides both).
- Warm-up ramp on every fresh launch to dodge acceleration-429s.

The `api_dispatch.py` module enforces all of these by default; callers
don't have to think about them.
