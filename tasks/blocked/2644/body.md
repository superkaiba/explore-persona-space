---
title: 'workflow-fix: no recovery ladder when the subagent MODEL POOL is exhausted;
  reconciler site has no valid fallback'
kind: infra
tags:
- from-2387
created_at: '2026-08-29T21:55:44Z'
has_clean_result: false
origin_prompt: 'Live on #2387 round 5: a PASS-vs-FAIL split routes to the binding
  reconciler, and four consecutive reconciler spawns died immediately on a Fable 5
  model limit over ~1h. context-hygiene.md covers refusal and autocompact-thrash deaths
  but not model-quota exhaustion; the per-call model pin is inert under CLAUDE_CODE_SUBAGENT_MODEL
  and cross-vendor Codex dispatch is invalid at this site because Codex would adjudicate
  its own verdict.'
workflow: v1
---
## Goal

Give the five Codex-ensemble review sites — and the `reconciler` in
particular — a defined recovery path when the SUBAGENT MODEL POOL is
exhausted, which today has none.

## Provenance

- workflow_fix_target: .claude/rules/context-hygiene.md
- fingerprint: 1577da44cef4
- additional_targets: .claude/rules/codex-ensemble-review.md, .claude/skills/issue/SKILL.md, .claude/agents/reconciler.md
- verified-at-filing: n/a — hand-filed from #2387 round 5 (observed live), not routed
  through a `workflow-fix-candidate v1` block; the gap is the ABSENCE of a
  model-quota rung in `context-hygiene.md`, confirmed by grep: the file's ladders
  cover refusal (a)-(g) and autocompact-thrash Class 1/2 only.

Surfaced live on #2387 round 5 (2026-08-29). The round-5 code review split
PASS (Claude) vs FAIL (Codex), which routes to the binding `reconciler`. Four
consecutive reconciler spawns died immediately on `You've reached your Fable 5
limit`, across ~1 h with escalating holds (0 / 10 / 30 min). No durable output
any time. The task cannot advance: a split verdict has exactly one sanctioned
resolver, and that resolver cannot be spawned.

## The gap

`.claude/rules/context-hygiene.md` carries recovery ladders for the two known
subagent-death classes — spurious usage-policy refusals (rungs a-g) and
autocompact thrash (Class 1 reduced-window, Class 2 fixed-overhead). Neither
covers MODEL QUOTA EXHAUSTION, and the two documented escapes are both
structurally unavailable for it:

- **Per-call model pin is inert.** `CLAUDE_CODE_SUBAGENT_MODEL` in
  `~/.claude/settings.json` sits at the TOP of the resolution chain, above the
  Agent-tool `model` parameter. The refusal ladder already records this as the
  rung-(b2) PRECEDENCE CAVEAT and resolves it with "remove the env line and
  start a fresh session" — which an autonomous mid-run session cannot do, and
  which would break the prompt-cache key besides.
- **Cross-vendor Codex dispatch is structurally wrong AT THIS SITE.** It is
  the refusal ladder's escape hatch, but the reconciler exists to adjudicate a
  Claude-vs-Codex disagreement; a Codex reconciler would rule on its own
  verdict. This is not a cost or quality objection, it is a validity one.

The retry pacing is also undefined, and the natural instinct is wrong. The 429
ladder (minute-boundary + escalating holds, "retry anyway" at the ~10 min cap)
assumes a per-minute bucket that replenishes continuously. A model quota pool
does not. On #2387 that mismatch produced three retries in 25 minutes that
could not have succeeded. The right analogue is the Codex quota sentinel
(`.claude/cache/codex-quota-exhausted-until`), which parses a reset time and
short-circuits dispatch until it passes — but no equivalent exists for the
Claude-side subagent model pool, and the error text names no reset time.

## Scope

- `.claude/rules/context-hygiene.md` — add a THIRD subagent-death class
  (model-quota exhaustion) beside refusal and thrash, with its own ladder and
  its own pacing note distinguishing quota pools from per-minute buckets.
- `.claude/rules/codex-ensemble-review.md` and
  `.claude/skills/issue/SKILL.md` Step 5c — say what a site does when the
  binding resolver cannot be spawned at all. Today Step 5c defines PASS+PASS,
  FAIL+FAIL, and PASS-vs-FAIL, and assumes the third can always be resolved.
- `.claude/agents/reconciler.md` — the site-specific note that cross-vendor
  substitution is invalid here, so the generic escape does not apply.

## Acceptance

1. A documented ladder for model-quota exhaustion, stating explicitly which
   generic escapes do NOT apply to the reconciler site and why (validity, not
   cost).
2. A pacing rule that distinguishes quota pools from per-minute buckets, so a
   session does not run the 429 ladder against a quota wall.
3. A defined terminal for the case where the pool stays exhausted long enough
   that retrying is futile. Candidates, to be decided by the planner — do NOT
   assume the first: (a) park via `epm:failure v1` + `status:blocked` with a
   `failure_class: infra` reason naming the pool, surfacing to the user, who
   holds the only real levers (`/usage-credits`, repinning the env var);
   (b) a `[epm-inline-fallback]`-tokened orchestrator composition, which the
   existing rule permits only for a workflow-fix task fixing this failure mode
   or refusal rung (c) — note this task IS such a workflow-fix task, so
   whether that carve-out should extend here is a real question, not a
   rhetorical one; (c) a persisted quota sentinel mirroring the Codex one, so
   later sessions short-circuit instead of re-discovering the wall.
   Whichever is chosen, the independence guarantee must be either preserved or
   its breach recorded in a durable, greppable form.
4. If (c) is adopted, the sentinel must fail OPEN exactly as the Codex one
   does — an unreadable or implausible sentinel never wedges dispatch off.
5. Existing workflow-lint and spec tests stay green.

## Notes for the planner

- Do NOT solve this by removing `CLAUDE_CODE_SUBAGENT_MODEL`. The pin is
  deliberate and cross-project; the fix is a defined path when the pinned pool
  is dry.
- The failure is silent in a specific way worth preserving against: the agent
  dies with an API error and no durable output, so a session that does not run
  the durable-verdict-first probe would mistake it for a no-show and could
  take a single-reviewer decision the site does not sanction. Any ladder here
  should restate that probe.
- Check whether the watcher should observe repeated model-quota deaths at all.
  Today nothing counts them, so a session can burn hours retrying with no
  external signal.
