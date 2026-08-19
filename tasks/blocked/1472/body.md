---
title: 'daily-held: review diversity during Codex outage — renewed to Sep 5'
kind: infra
tags:
- daily-held
created_at: '2026-07-17T06:58:58Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 3): the Codex org quota is
  exhausted until ~2026-08-06 (CODEX_QUOTA_LIVE sentinel) — every doubled review site
  ran single-Claude across the entire fleet on 2026-07-16 and will for ~3 more weeks'
workflow: v1
---
## Held item (route 3 — needs-human)

Auto-filed by /daily 2026-07-16 as a TRACKED needs-human task (route 3 of the three-route classifier).

- **What happened:** the Codex org quota is exhausted until ~2026-08-06 (CODEX_QUOTA_LIVE sentinel) — every doubled review site ran single-Claude across the entire fleet on 2026-07-16 and will for ~3 more weeks
- **Which carve-out held it:** spends money / genuinely ambiguous — reviewer-capacity decision with a bill either way
- **Decision needed:** decide: accept single-Claude review for the window, or wire a temporary alternate cross-vendor reviewer (e.g. a different OpenAI key/org or another vendor) — spends money either way
- **Suggested surface:** PM `Needs you` block.

## Analysis & recommendation (2026-07-17, autonomous legwork per PM directive)

**Root cause is most likely a LAPSED ChatGPT Pro subscription, not exhausted Pro quota.**

Evidence:

- Quota sentinel live: exhausted until **2026-08-06T13:26Z** (detected 2026-07-08; `.claude/cache/codex-quota-exhausted-until`).
- The stored refusal text upsells "start a free trial of Plus" — messaging OpenAI shows to accounts WITHOUT an active paid plan.
- `~/.codex/auth.json` OAuth id-token claims: `chatgpt_plan_type: pro`, but **`chatgpt_subscription_active_until = 2026-05-24T19:06:55Z`** (subscription last verified by OpenAI 2026-05-01; token last refreshed 2026-07-04). The claims say the Pro window ended May 24.
- Codex CLI auth mode is `chatgpt` (subscription OAuth). The `.env` `OPENAI_API_KEY` is **dead — HTTP 401** on a free `GET /v1/models` probe — so metered API-key billing is not currently possible without minting + funding a new key.
- Pre-outage fleet volume: ~130–420 Codex dispatches/day (Jul 1–7: mean ~286/day; 1,969 CLI sessions, ~1.16 GB rollouts).
- Value of the twin (why this matters): Jul 1–7, **203 reconciler adjudications across 742 posted Codex verdicts → ~27% of doubled rounds ended in a genuine cross-family disagreement**. The Codex twin is not a rubber stamp; single-Claude review is a measurable oversight downgrade.

**Recommendation (in order):**

1. **Check + renew the ChatGPT Pro subscription** (chatgpt.com → Settings → Subscription, on the account backing `~/.codex/auth.json`). If it lapsed on 2026-05-24 as the token claims indicate, renewing (~$200/mo) most likely restores Codex immediately — no repo changes needed (the CLI reuses the existing OAuth). ~5 minutes, known price, restores cross-family review ~3 weeks early.
2. **If the subscription turns out to be active** (stale claims) → the quota is genuinely exhausted: either buy in-plan Codex usage credits if the account page offers them, or accept single-Claude until Aug 6 (**$0** — the #1204 pre-spawn skip + the no-show fallback already make the outage zero-waste operationally).
3. **Not recommended:** switching the Codex CLI to metered API-key billing at fleet volume — rough estimate $60–200/day (~$1–4K to Aug 6), and it needs a fresh funded key anyway.

Independent hygiene flag (applies under any option): the `.env` `OPENAI_API_KEY` is invalid (401) — re-mint at leisure; anything else referencing it fails the same way.

**After restoring:** `rm .claude/cache/codex-quota-exhausted-until` to force a probe dispatch (#1126) — on success the fleet resumes doubled review automatically.

**Consolidation:** #1140 ("Codex quota out to Aug 6 - pay or ride it out", filed 2026-07-08) was the same decision — archived as duplicate, analysis lives here.

## Update 2026-08-09 — outage renewed to Sep 5; still relevant

Re-checked in the user-directed blocked-task relevance sweep:

- The outage did NOT end on 2026-08-06. A fresh quota-exhausted sentinel was
  detected 2026-08-06T14:30Z with reset **2026-09-05T13:38Z**
  (`.claude/cache/codex-quota-exhausted-until`, `parse_ok: true`) — every
  doubled review site runs single-Claude for ~4 more weeks.
- The stored refusal text still upsells "start a free trial of Plus", so the
  2026-07-17 diagnosis above (lapsed ChatGPT Pro subscription; token claims say
  the Pro window ended 2026-05-24) still stands.
- Recommendation unchanged: check/renew the ChatGPT Pro subscription on the
  account backing `~/.codex/auth.json` (recommendation 1 above). The #1204
  pre-spawn skip keeps the outage zero-waste operationally in the meantime.
