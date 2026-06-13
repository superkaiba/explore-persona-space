# Audit report — task #535 events.jsonl token-shape scan

**Verdict:** FAIL — leaked: env-assign

- **File scanned:** `/home/thomasjiralerspong/explore-persona-space/tasks/completed/535/events.jsonl` (136 events)
- **Patterns:** hf, wandb (context-anchored), runpod, openai, anthropic, env-assignment
- **Total hits:** 2 (high-confidence: 0; low-confidence: 2)

## Triage — FAIL on low-confidence hits only

The scan found hits but ALL hits classify as low-confidence (fixture markers or values structurally too short to be a live credential). Per the plan's binary verdict rule the top-line is FAIL regardless, so a human reviews and acks the triage below — rotation is NOT required unless this section says otherwise.

Walk each row in the section below, confirm the context is benign (typically code-review prose quoting test fixtures, documentation examples, or a `.env.example` snippet), and ack the audit. If ANY row turns out to be a real leak that was mis-classified, rotate the corresponding secret and re-run this audit.

## Low-confidence hits (likely false positives)

These rows matched the env-assignment shape but the value is structurally not a real credential (contains an explicit fixture marker like `test_token` / `_test_` / `fake_`, or is shorter than the minimum length for a live secret of that key class). No rotation required; eyeball the note excerpt to confirm the context (typically code-review prose quoting test fixtures, documentation examples, or a `.env.example` snippet).

### Hit — line 113, byte offset 3385 · `env-assign:HF_TOKEN` (confidence: low — **triage:** value contains fixture marker 'test_token')

- **Event:** ts=2026-06-10T21:54:21Z, kind=epm:code-review-codex, version=1
- **Match:** `HF_TOKEN=<redacted:env:len=13>`
- **Note excerpt:** <!-- epm:code-review-codex v1 -->
# Codex Code Review: Multi-backend compute router — consolidated live-acceptance fix ladder (fix5–fix23)

**Verdict:** FAIL
**Blocker tags:** [substantive]
**Tier:** 

### Hit — line 113, byte offset 3414 · `env-assign:WANDB_API_KEY` (confidence: low — **triage:** value contains fixture marker 'test_key')

- **Event:** ts=2026-06-10T21:54:21Z, kind=epm:code-review-codex, version=1
- **Match:** `WANDB_API_KEY=<redacted:env:len=14>`
- **Note excerpt:** <!-- epm:code-review-codex v1 -->
# Codex Code Review: Multi-backend compute router — consolidated live-acceptance fix ladder (fix5–fix23)

**Verdict:** FAIL
**Blocker tags:** [substantive]
**Tier:** 

