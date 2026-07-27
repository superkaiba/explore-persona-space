---
title: 'daily-fix: harden the /daily miner brief (JSONL traps, probe'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4800548f9ba8
- daily-auto-filed
created_at: '2026-07-27T07:20:39Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-26 problem sweep (route 2): miners hit JSONL-parsing
  traps the brief does not warn about, and a miner''s suggested fix can name a mechanism
  it inferred rather than probed, sending unverified premises into the filing step'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-26 problem sweep (route 2). Surfaced by 1 independent
miner group(s) over the 2026-07-26 session transcripts.

## Goal

Give the /daily problem sweep a durable miner-brief composition recipe carrying the three
JSONL-reading traps miners keep hitting, and require every miner `suggested fix` to state
whether its named mechanism was probed or inferred.

## Workflow gap

- **Bug observed:** miners spawned by the /daily problem sweep hit three JSONL-reading traps
  the brief does not warn about — transcript rows with no `message` key (`KeyError: 'message'`),
  multi-file JSONL loops exceeding the default 2-minute Bash timeout, and `git` calls made
  after a `cd` into `~/.claude/projects/...` (not a git repo, exit 128) — and miner `suggested
  fix` lines name root-cause mechanisms the miner inferred rather than probed, three of which
  were refuted at filing-compose time.
- **Why it is a workflow gap:** `.claude/skills/daily/SKILL.md` has no miner-brief composition
  section at all, so the brief is re-improvised from scratch each run and every lesson learned
  in one night's fan-out is lost by the next.
- **Confidence (emitter):** high
- verified-at-filing: absence greps against the named target —
  `grep -niE 'miner|CHEAPLY|fan.out' .claude/skills/daily/SKILL.md` → **1 hit (L770)**, and
  it is not a recipe — the word "miners" appears once, in the headless-mode paragraph
  ("file's content depends on (transcript miners, a driver whose filed `#id`s …");
  `grep -niE "get\('message'|KeyError|timeout ≥|not a git repo" .claude/skills/daily/SKILL.md`
  → **0 hits** for all three gotchas;
  `grep -niE 'probed|inferred|ran the probe' .claude/skills/daily/SKILL.md` → **0 hits**.
  For contrast the equivalent duties DO exist one layer downstream — the filer's
  `verified-at-filing` mandate and the unverified-premise labeling convention are both in the
  same file (L485), which is why the refuted premises were caught at compose time rather than
  after filing. Landed-fix check:
  `git log --oneline --since='7 days ago' -- .claude/skills/daily/SKILL.md` → 6 commits, none
  adding a miner-brief recipe. (2026-07-26)

**Context binding — one target detail corrected.** The mined report proposes the change in
`.claude/skills/daily/SKILL.md` "§ Problem sweep, miner-brief composition". The § Problem
sweep section exists (L194); the "miner-brief composition" sub-section does not. The change
therefore CREATES that sub-section under § Problem sweep. The "How to read the transcripts —
CHEAPLY" recipe the miners actually ran against lives only in the per-run ad-hoc brief
(`logs/daily/mining-2026-07-26/BRIEF.md` L25-33) — which is exactly the durability gap: that
file is regenerated nightly and inherits nothing.

## Evidence

- Session `c0a2df1b`, subagent `miner-interactive-B` (`agent-a69ace10621b8fd5c`), within
  06:30–06:50Z: 3 of that miner's 27 `tool_result` firing events were errors, all from gaps
  the recipe does not cover — `"Exit code 143\nCommand timed out after 2m 0s"` on a
  `for f in <6 session ids>` loop over multi-MB JSONLs;
  `"Traceback (most recent call last): File \"<string>\", line 7, in <module> KeyError:
  'message'"` (rows of type `queue-operation`, `attachment` and `last-prompt` carry no
  `message` key); `"Exit code 128"` from a `git log -- <paths>` run with cwd inside
  `~/.claude/projects/...`. Measured cost: 3 retries, roughly 2.5 min. The other 5 miners in
  the same fan-out had 0 errors, so this is guidance drift rather than a systemic failure.
- Session `c0a2df1b`, 2026-07-26T06:36:06Z and 06:44:10Z: three route-2 filings drafted from
  miner reports were refuted by the filer's own probes before filing — (a) "the `--map-files`
  map applies no workflow_lint surcharge", refuted by running `recommended_timeout_s` (2580 s;
  the real cause is that the surcharge key is an EXACT path while the map holds 25 sibling
  files); (b) "code-reviewer PASSed with 7 open BLOCKERs", refuted — `code-reviewer.md` rule 11
  prescribes exactly that shape; (c) "RunPod bootstrap hard-sets `BRANCH=main`", refuted —
  threading exists and `main` is the fallback, so the real defect is an empty `repo_branch`
  upstream. Evidence: `"C4's premise is mis-diagnosed — the surcharge isn't missing, its key
  just doesn't match. Verifying C1 and C5, then composing corrected bodies."` and `"One
  filing's premise just failed verification — code-reviewer.md rule 11 *designs*
  PASS-with-persisted-BLOCKER. Dropping that half."`
- Measured cost: roughly 10 min of re-verification and body rewrites, and one filing half
  dropped. Nothing wrong shipped — the filer's discipline held — but each refutation was a
  round the miner→filing handoff could have avoided by marking the diagnosis as inferred.

## Proposed change

- `.claude/skills/daily/SKILL.md` § Problem sweep — add a **miner-brief composition**
  sub-section that is the durable source for the per-run brief, carrying the transcript-reading
  recipe currently re-improvised each night plus these three lines:
  - rows without a `message` key exist (`queue-operation`, `attachment`, `last-prompt`) — use
    `.get('message', {})`, never `row['message']`;
  - any multi-file JSONL loop passes an explicit Bash `timeout` ≥ 300000 ms (the default 2-min
    cap kills a 6-transcript loop mid-pass, exit 143);
  - run every `git` call from the repo root — the transcript dirs under
    `~/.claude/projects/...` are not a git repo (exit 128).
- Same sub-section — extend the miner output schema with a **probed-vs-inferred** field on
  `suggested fix`: when the fix names a mechanism (a missing surcharge, a hard-set variable, a
  rule violation), the miner states `probed: <the exact command it ran>` or
  `inferred — not probed`. A miner is not obliged to probe; it is obliged to say which it did,
  so the filer knows which premises need a verification round.
- Keep the existing evidence-counting discipline (L214-243) as-is and reference it from the new
  sub-section rather than restating it.

## Scope / surfaces

- Primary target: `.claude/skills/daily/SKILL.md`
- none

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `uv run python scripts/workflow_lint.py` passes (no-flags); ruff clean on touched files.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route
  its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 4800548f9ba8

- workflow_fix_target: .claude/skills/daily/SKILL.md
- fingerprint: PENDING

/daily 2026-07-26 route-2 filing. Miner refs: C-P5, C-P6.
