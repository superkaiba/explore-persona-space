---
title: 'Finish API throughput project: Phase 3 guidelines + dispatcher review/migrate
  + #658 validation'
kind: infra
tags:
- keep-running
created_at: '2026-06-27T02:40:41Z'
has_clean_result: false
origin_prompt: 'Run the rest of this in the background with happy coder: finish docs/api_throughput_plan.md
  — Phases 0-1 done, dispatcher built, but Phase 3 docs/api_throughput_guidelines.md
  is still missing.'
---
## Goal

Finish the multi-phase Anthropic API max-throughput project. The plan and the
Phase 0-1 measurements already exist and are approved — this task EXECUTES the
remaining phases, it does NOT re-plan.

**Authoritative sources (read first, do not re-derive):**
- `docs/api_throughput_plan.md` — the approved plan; §1b "Operating defaults" are FIRM.
- `.claude/cache/api-throughput-status.md` — live execution status (what's done).
- `eval_results/api_throughput/*.json` — Phase 0-1 raw data.
- `src/explore_persona_space/llm/api_dispatch.py` — the dispatcher (already built, needs review).

**Settled findings (from §1b + status cache):** 3 distinct org keys → additive rate
limits; the throughput plateau is CLIENT-bound (process pool × org fan-out is the fast
path, not one asyncio loop); polite per-key concurrency caps Sonnet ~100 / Haiku ~120 /
Opus ~40; AIMD back-off on 429; sync fan-out beats batch for our N (even #658's 141.6k);
batch is the considerate path for huge / latency-tolerant N; judge stays Sonnet 4.5.

## Remaining deliverables (priority order)

1. **Phase 3 — `docs/api_throughput_guidelines.md` (MISSING, headline deliverable).**
   Decision table: N ∈ {10, 100, 1k, 10k, 100k, 1M} × model × {latency-critical,
   cost-critical} → fastest config (tier, key(s), chunk, concurrency, predicted
   wall-clock + cost + binding limit) + the crossover N. Derivable now from §1b +
   Phase 0-1. Add a CLAUDE.md pointer.
2. **Re-tune `src/explore_persona_space/eval/judge_dispatch.py`** `DEFAULT_SUB_BATCH_SIZE`
   / `MAX_CONCURRENT_SUB_BATCHES` off the Phase-3 numbers (they currently reproduce the
   #658 starvation failure).
3. **Code-review the dispatcher** `src/explore_persona_space/llm/api_dispatch.py` (spawn
   the `code-reviewer` agent), add unit tests (routing / chunking / 429 back-off /
   org-aware resume, mock client), commit on review PASS.
4. **Phase 5 — migrate judge callers** through the dispatcher (`eval/batch_judge.py` and
   the other judge paths); keep public signatures; judge model stays Sonnet 4.5.
5. **Phase 6 — validate on the #658-shaped workload** (141.6k): same-session back-to-back
   baseline vs dispatcher at matched tier + model + N; report wall-clock speedup AND cost
   separately (do NOT claim equal cost). Target ≥5× wall-clock, no data loss, resumable.
6. **Phase 1/2 extra measurements** — lower priority (strategy already settled); only if
   time remains.

## Constraints

- Shared org keys (other fellows use them): obey §1b etiquette — per-key caps above,
  AIMD 429 back-off, watch `anthropic-ratelimit-*-remaining` and ease off when low,
  fan across the 3 orgs.
- Doc edits commit directly to `main` by explicit path; multi-file code changes use a
  worktree per the project worktree rule, then merge to `main`. Push immediately.
- Update `.claude/cache/api-throughput-status.md` as each phase completes.
- **Flag to the user:** the 3 API keys were pasted in chat earlier — they should rotate them.
