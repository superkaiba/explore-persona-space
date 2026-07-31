---
name: sonnet-refusal-in-seed-prompts
description: Verbatim "generate jailbreak attempts" wording in Anthropic Batch seed prompts triggers Sonnet refusals; reframe as taxonomy categories under research framing.
metadata:
  type: feedback
---

When writing seed-prompt scaffolding for corpus generation that targets adversarial / jailbreak / hostile-persona domains, NEVER ask Sonnet for verbatim attack content. Sonnet refuses such requests under thesis-research framing — silently, with refusal text that parses as 0 elements through any JSON-array extractor.

**Why:** Round-2 implementer for #377 wrote a `hostile_jailbreak` topic-seed prompt asking for "10 conversation-opening topics that begin an incremental jailbreak". The Anthropic Batch returned 20/20 succeeded at the API level, but 5/20 cells (1 per persona) were Sonnet refusal text. The caller's "got 0 topics, expected 10" RuntimeError obscured the cause. Cost: 1 wasted pod day + a round-3 implementer round.

**How to apply:** When the seed-prompt domain is adversarial / hostile / jailbreak-adjacent:
1. Reframe as "AI-safety research taxonomy": ask for short CATEGORY LABELS (1-7 words each), not verbatim attack content. The downstream conversation-loop auditor briefing can still push the model under test through the adversarial mechanism multi-turn.
2. Verify with a manual 1-request non-batch probe (`anthropic.Anthropic(...).messages.create(...)`) before launching the full batch.
3. Defense in depth: add a `detect_refusal(text)` helper (regex over "I can't", "I cannot", "I'm not able to", "I must decline") and raise a distinct `SonnetRefusalError(RuntimeError)` from any zero-parse path. Include the first 200 chars of the response in the error.

Related: [[snapshot_download_siblings_truncation]] — same shape of "API returned 200, but the payload was empty for our purposes".

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Sonnet refusal in seed prompts](feedback_sonnet_refusal_in_seed_prompts.md) — "generate jailbreak attempts" wording silently refuses; reframe as taxonomy labels + detect_refusal helper.
