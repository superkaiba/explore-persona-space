---
name: live-probe-scope
description: A 1-shot Sonnet refusal probe on the seeding step misses downstream multi-turn refusals. Probe the full per-turn loop x all domains x refusal-likely depth (turns 5-10), validate the refusal regex on false positives too, and add a mid-run quality gate.
metadata:
  type: feedback
---

When an implementer fixes a Sonnet refusal by reworking ONE prompt and validates with "live probe confirmed N clean responses", that probe is necessary but not sufficient — it must cover the full multi-turn pipeline.

**Why:** #377 round 3 — the reworked seeding prompt cleared 20/20, but the per-turn auditor prompts (22 turns × 50 convs × 4 domains) hit 56-62% refusal by turn 3 of the therapy domain; the seeding-only probe never exercised the "distressed crisis-state user" role at depth. A too-broad `detect_refusal()` regex also false-positived on in-character text ("I can't sleep again"), so the corpus was unusable within 3 turns — ~$10-15 of Batch spend wasted.

**How to apply:**
1. Probe the full per-turn loop: ≥1 conversation × M turns with M ≥ the refusal-likely depth (turns 5-10 in adversarial/crisis/persona-pressure scenarios — Sonnet often complies 1-2 turns then refuses).
2. Probe ALL domains, not just the one that failed — refusal surfaces differ per content type.
3. Validate the refusal-detection regex on BOTH true and false positives (legit in-character "I can't ..." lines are corpus-killing too).
4. Add a mid-run quality gate (e.g. abort if [BATCH_ERROR] >5% global or >20% per-domain-turn at turn 5) instead of end-of-run validation.

Also durable: Sonnet deterministically refuses ~0.5% of benign poetry/creative translation content — design translation/generation pipelines to skip+report those rows, never hard-raise.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Live-probe full scope](feedback_live_probe_scope.md) — probe per-turn loop × all domains × turns 5-10, validate refusal regex on false positives, mid-run gates; Sonnet also refuses ~0.5% benign creative rows — skip+report (#377)
