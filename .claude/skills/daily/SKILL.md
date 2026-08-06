---
name: daily
description: End-of-day Explore Persona Space brief — what happened today, plus an exhaustive sweep of every problem/confusion/error in the day's Claude Code session transcripts, each with a concrete fix. Each fix is routed by a THREE-ROUTE classifier (changed #706): route 1 SELF-APPLIES only trivial mechanical changes with no behavior effect (doc edits, string replaces, lint annotations — verified, then committed one fix per commit so each stays one git-revert away); route 2 FILES every behavior/logic/code/infra change via scripts/file_infra_task.py → a background /issue --auto session, so the fix lands through the full independent planner / critic ensemble / Claude+Codex code-review / test-verdict pipeline rather than this skill's own verify gate; route 3 FILES a tracked proposed needs-human task (carve-out: result-interpretation changes, destructive/irreversible actions, spending money, external side-effects, genuinely ambiguous intent) that the PM surfaces. A concise summary of what was applied + filed + held is pushed to Thomas's my-goat Telegram. Nothing is silently dropped.
---

# Daily Brief

Use `tasks/` as the only workflow state source. Do not read or mutate queue,
status, promotion, or approval state through any external tracker.

Two jobs in one file:
1. **Recap** — what happened on the project today.
2. **Problem sweep + three-route fix** — go through today's Claude Code session transcripts in detail and catch EVERY problem, confusion, or error that occurred — not just recurring patterns, not just a top-5. Each problem with a derivable fix is routed by the THREE-ROUTE classifier (see "Triage each problem" below): **route 1 (trivial mechanical, no behavior change)** is AUTO-APPLIED in this run — make the edit, VERIFY it (see "Verification gate for code fixes" below), and `git commit` it on its own with trailing pathspecs naming exactly the files the fix touched (one commit per fix, so each is independently revertable; § Commit own-files-only contract — never a bare `git commit` on the shared root), then run the repo-wide workflow lint ONCE after all workflow-file fixes (see "Lint gate" below — `workflow_lint.py` is a repo-wide validator, NOT a per-file `.md` linter), and record it in `## Applied workflow improvements` with its diff and commit sha; **route 2 (any behavior/logic change)** is FILED for independent review via `scripts/file_infra_task.py` → `/issue --auto` (recorded as a "filed for review #<N>" entry in `## Applied workflow improvements`, no self-applied diff); **route 3 (a genuine judgment call — see "Judgment-call carve-out" below)** is FILED as a TRACKED `proposed` `needs-human` task the PM surfaces, AND logged in `## Other problems & notes` with the filed `#<N>`. Then push a concise summary of what was applied + filed (with the commit shas / filed ids) to Thomas's my-goat Telegram chat (see "Surfacing flow (applied / filed / held)"). Nothing is silently dropped — every problem self-applies (route 1), routes to review (route 2), or becomes a tracked needs-human task + a note (route 3).

   **Three routes, not a binary apply-vs-hold (changed 2026-06-28, #706).** Earlier this skill ran a binary classifier: bucket 1 = self-apply EVERYTHING fixable (the 2026-06-08 "make the workflow improvements automatically" + the 2026-06-12 "fix experiment bugs / infra flakiness / high-blast-radius stuff automatically" directives), bucket 2 = hold a genuine judgment call as a dead-end note. That made self-apply the ONLY un-reviewed path AND it "graded its own homework" — a behavior change shipped on this skill's own verify gate with no independent review, and a held item landed where Thomas does not look (a tracked-but-unread `logs/daily/` file + one Telegram digest line), never as an actionable task. The classifier is now THREE routes (full spec under "Triage each problem"):
   - **Route 1 — trivial mechanical** (doc edits, string replaces, lint annotations — NO behavior change): self-apply + verify + commit, exactly as the old bucket 1 did, but NARROWED to changes with no behavior/logic effect.
   - **Route 2 — any behavior/logic change** (incl. high-blast-radius REVERSIBLE — the bulk of the old bucket 1): no longer self-applied. FILE a `kind: infra` task that auto-dispatches to `/issue --auto`, so the fix lands through the full independent planner / critic ensemble / Claude+Codex code-review / test-verdict pipeline rather than this skill's own verify gate. This kills "grades its own homework" + "self-apply is the only un-reviewed path".
   - **Route 3 — genuine judgment call** (the carve-out list below, verbatim): FILE a TRACKED `proposed` task tagged `needs-human` that the PM surfaces + re-surfaces in the session Thomas actually uses — no longer a dead-end log note.
   The safety posture stays "stay fully transparent + per-fix revertable"; route 2 ADDS independent review before a behavior change lands, and route 3 ADDS a tracked, re-surfaced task so a held item is never lost.

## Inputs

Read:

- tasks and workflow events via `scripts/task.py`;
- `RESULTS.md` for accepted headline claims;
- `eval_results/INDEX.md` for artifact inventory;
- `docs/research_ideas.md` for aims and phase framing;
- local run logs only as supporting evidence, never as workflow state;
- **Claude Code session transcripts** under `~/.claude/projects/-home-thomasjiralerspong-explore-persona-space/*.jsonl` and `~/.claude/projects/-home-thomasjiralerspong-explore-persona-space--claude-worktrees-*/*.jsonl` — filter to files modified today (UTC), then **cross-check the newest in-file message `timestamp` per transcript and DROP any whose newest content predates the brief's date**. A background process (the autonomous-session watcher / session-summarize cron reading transcripts) bumps the mtime of OLD transcripts without adding new content, so an mtime-only filter pulls in already-covered days and risks re-mining problems the prior daily already handled. Only transcripts with at least one message dated on the brief's day are genuine inputs for this run (incident 2026-06-22: 5 of 6 "modified today" transcripts held only June 17-18 content). Enumeration trap: `find` on this VM is bfs — `-newermt` accepts ONLY strict ISO-8601 timestamps (`-newermt '2026-08-01T00:00:00-07:00'`); the GNU-find `'YYYY-MM-DD HH:MM:SS TZ'` form is an "Invalid timestamp" ERROR, and with stderr suppressed (`2>/dev/null`) the pass silently returns 0 files — never suppress stderr on the enumeration call (incident 2026-08-01: 3 successive enumeration calls returned silent 0 before the ISO form was used).

Useful commands:

```bash
export PATH="$HOME/.local/bin:/snap/bin:$PATH"   # uv lives in ~/.local/bin; gcloud lives in /snap/bin; non-login (cron) shells miss both
uv run python scripts/task.py list-by-status --limit 500
uv run python scripts/task.py list-by-status --status running --limit 100
uv run python scripts/task.py list-by-status --status uploading --limit 100
uv run python scripts/task.py list-by-status --status awaiting_promotion --limit 100
uv run python scripts/task.py view <N>
```

## Output

Write the brief to `logs/daily/YYYY-MM-DD.md` (relative to the repo root —
`~/explore-persona-space/`). One file per date. For a BACKFILL run
(`/daily <YYYY-MM-DD>`) substitute the target date throughout — see
"Backfill a missed day (date argument)". The file is a written RECORD
of what the run applied + noted; the actual surfacing to Thomas happens over
Telegram (see "Surfacing flow (applied / filed / held)"), not via this file. Handle an
existing file as follows:

- **File does not exist** → write the dated SKELETON first — the frontmatter
  (§ Frontmatter) + all six H2 sections below with EMPTY bodies (in
  particular `## Applied workflow improvements` stays empty until
  enrichment) — as the run's FIRST durable action, then fill each section in
  place via `Edit` as results land. In headless mode, rule 0 under
  "Headless (cron) mode" additionally makes the skeleton commit + push
  mandatory-immediate. The completed brief is the same file after
  enrichment; the emptiness of the skeleton's Applied section is
  load-bearing (it is what keeps a husk on this list's Edit-in-place
  recovery branch below, and what the #1189 healthcheck husk arm detects).
- **File exists but is missing the `## Applied workflow improvements` H2, or
  that section is empty** → do NOT overwrite the file. `Edit` it to insert /
  fill the `## Applied workflow improvements` section in place (between
  `## What happened` and `## Other problems & notes`, or in the correct
  position if those are absent), and likewise insert `## Other problems &
  notes` AND `## Living-docs drift` (between `## Other problems & notes` and
  `## My thoughts`) if either is missing. Leave every other NON-EMPTY section
  — including any edits Thomas already made to `## What happened` /
  `## My thoughts` — untouched. Exception for an ALL-EMPTY stub-first husk
  (#1189: every section body empty — the skeleton was written but the run
  died before any enrichment): the recovering run may also fill the other
  auto-drafted sections (`## What happened`, `## Living-docs drift`,
  `## Highlighted results`), since nothing there can be Thomas's; a section
  with ANY existing content is still never touched, and `## My thoughts` is
  always left alone.
  This is the recovery path when an earlier manual or partial run left a stub
  without the problem sweep.
- **File exists with a non-empty `## Applied workflow improvements` section**
  (real applied edits OR the "no workflow-fixable problems" placeholder) →
  the day's auto-apply already ran; do NOT re-apply or re-overwrite (re-running
  would double-apply fixes). Refuse to overwrite and tell the user the day is
  already done.

**Manual runs complete the FULL file AND auto-apply.** When `/daily` is invoked
manually (not via the nightly cron), always produce EVERY section including
`## Applied workflow improvements` and actually apply the fixes. Under
stub-first an aborted manual SKELETON (empty Applied section) is recoverable —
it lands on the Edit-in-place branch above, so the 23:27 PT cron can still run
that day's problem-sweep. But a manual run that fills
`## Applied workflow improvements` and then stops DOES block the nightly cron
(the non-empty-Applied refuse rule reads it as "day already done"), so finish
the apply pass you started.

The file is a stub Thomas will finish editing. It starts hidden from the
`/log` dashboard feed (`visible: false`) and only becomes visible when he
flips the frontmatter field manually.

### Frontmatter

Every file MUST have this YAML frontmatter:

```yaml
---
kind: daily
date: YYYY-MM-DD
title: <auto-generated, one line — Thomas can edit>
included_tasks: [<task IDs from auto-population below>]
visible: false
---
```

- `date`: today in ISO format.
- `title`: a one-line auto-generated headline (e.g. `Daily — <date> (<N> results promoted)`).
- `visible: false` ALWAYS at creation. Never set `true`. Thomas flips it manually.
- `included_tasks`: auto-populate from clean-results promoted today. Recipe:
  1. `uv run python scripts/task.py list-by-status --status completed --limit 500 --json`
     and keep rows where `has_clean_result == true`.
  2. For each surviving id, run `uv run python scripts/task.py view <N> --json`
     and read `frontmatter.promoted_at` (ISO UTC timestamp).
  3. Keep ids whose `promoted_at` falls on today's UTC date.
  4. Legacy clean-results may have `promoted_at = None` — skip silently.

### Body (stub sections)

Below the frontmatter, write exactly these six H2 sections in this order:

```markdown
## What happened
<2-5 bullets: today's task activity. Pull from epm:* markers, status
changes, completed reviews. Be concrete (mention task IDs). This is the
auto-drafted summary Thomas will edit down.>

## Applied workflow improvements
<numbered list of workflow improvements from this run — BOTH route-1 self-applied
fixes AND route-2 review filings (see "Triage each problem" below for the routing
and "Applied-edit record shape" below for the route-1 shape):
  - Route 1 (trivial mechanical, self-applied): each entry carries its applied
    diff + commit sha + the verification result (per the record shape below).
  - Route 2 (any behavior/logic change, filed for review): a
    "filed for review #<N>: <one-liner>" entry — the filed `kind: infra` task id
    (tagged `daily-auto-filed`) + a one-line description, with NO self-applied diff
    or sha (the actual code change lands when the spawned `/issue --auto` session
    completes through the full independent pipeline).
If no workflow improvements were applied OR filed today, write a single line:
`- _no workflow-fixable problems found today_`>

## Other problems & notes
<every problem/confusion/error from today that was NOT routed to
`## Applied workflow improvements` (i.e. not a route-1 self-apply nor a route-2
review filing). One bullet each: what happened (session id / task id) + the
routing/disposition + a one-line suggested action. Specifically:
  - Route 3 (a genuine judgment call): name WHICH carve-out item held it, AND the
    filed `daily-held needs-human` task `#<N>` (it is a TRACKED `proposed` task the
    PM surfaces + re-surfaces, no longer a dead-end note).
  - Fixes that failed verification or the lint gate (reverted), research questions,
    and anything Thomas had to fix by hand.
These are notes, not applied edits. If none, write:
`- _no other problems surfaced today_`>

## Living-docs drift
<output of `uv run python scripts/living_docs.py check` (the Living-docs
drift pass, Problem-sweep section below — `check` is the read-only drift
linter; NEVER run `living_docs.py apply` from /daily). If the check exited
zero, write a single line: `- _no drift — open_questions.md is in sync_`. If
it found drift, list each finding as a bullet (the `relates_to` ⇄ evidence
mismatch / the `completed`-with-clean-result task missing from any question's
evidence / the dangling `#N` / the stale question), then — ONLY IF no
still-open re-synthesis proposal already exists for the same drift (dedup
below) — add one PROPOSAL line:
`- **Proposal:** re-synthesize open_questions.md (run the living-docs
updater/backfill) to reconcile the drift above — needs your ok; not
auto-applied.`
This is a PROPOSAL only — never run the re-synthesis from /daily. Record it in
the dedup event-stream (Problem-sweep section) so a second nightly run does not
re-propose it. ALSO surface here any parked living-docs rejection that is now a
re-proposal candidate (the Parked-proposal re-consideration pass below) — still
a user-gated proposal, never applied. If no drift and no re-proposal candidate,
the single "_no drift_" line stands.>

## My thoughts
<leave empty — Thomas fills in>

## Highlighted results
- #<N> — <task title>
- #<M> — <task title>
```

`Highlighted results` starts as a one-line stub per `included_tasks` entry
(just the title from `view <N> --json` → `frontmatter.title`). If
`included_tasks` is empty, write a single bullet: `- _no results promoted today_`.

### Problem sweep (what fills the two problem sections)

Go through today's transcripts in detail. The goal is COVERAGE, not pattern-
mining: catch every distinct problem, confusion, or error that occurred, even
if it happened exactly once. Do not require recurrence. Do not dedupe a real
problem away because it "probably won't happen again."

Signals to hunt for (non-exhaustive — anything that went wrong counts):

- **User corrections** — "no", "don't", "stop", "wrong", "not what I meant", or Thomas significantly rewriting / redoing an artifact I produced.
- **Confusions** — places I misread intent, went down the wrong path, needed re-steering, or asked a question whose answer was already available.
- **Errors & failures** — tool-call errors, tracebacks, retries (same tool 3+ times), crashes, OOMs, failed launches, failed reviews / reconciles.
- **Process mistakes** — skipped a step, ran steps out of order, missed one of the enumerated `/issue` gates, OR overreached (acted where I should have asked, e.g. auto-applied a workflow edit).
- **Repeated explanations** — context I needed re-explained that should already live in a workflow file ("I keep telling you about X").
- **Stale references** — task / agent / skill / script names that no longer exist (cross-check the current `.claude/` tree).
- **Voice / register drift** — corporate-speak, AI-slop vocab, invented jargon, opaque condition codes, or template-copying instead of plain-English.
- **Dropped handoffs / manual fixes** — information lost between agents, or anything Thomas had to do by hand that an agent should have done.

#### Miner-brief composition (recipe the per-run brief inherits)

This is the DURABLE source the per-run `logs/daily/mining-<date>/BRIEF.md`
inherits from. The per-run brief is regenerated nightly and must copy the
recipe from here — never re-improvise it — so a trap one miner hits tonight
is a trap every miner avoids tomorrow.

Transcript-reading traps every miner brief carries verbatim:

- **Rows without a `message` key exist.** Transcript-row types
  `queue-operation`, `attachment`, and `last-prompt` carry no `message`
  field. Use `row.get('message', {})` — never `row['message']` (raises
  `KeyError: 'message'` mid-loop).
- **Multi-file JSONL loops need an explicit Bash `timeout ≥ 300000` ms.**
  The Bash tool's default 2-minute cap kills a 6-transcript loop mid-pass
  with `Exit code 143 Command timed out after 2m 0s`. Cap is 600000 ms
  (10 min) — pick a value between 300000 and 600000 sized to the loop.
- **Run every `git` call from the repo root.** The transcript directories
  under `~/.claude/projects/...` are not a git repository — a `git log` /
  `git blame` there returns `Exit code 128`. `cd` back to the repo root
  (or use `git -C <repo-root> ...`) before any git invocation.

Miner output schema — `suggested fix` probed-vs-inferred field:

- When a fix names a mechanism (a missing surcharge, a hard-set variable,
  a rule violation, any concrete root cause), the miner states one of:
  - `probed: <exact command it ran>` — the miner ran the probe, this is
    the verified output; OR
  - `inferred — not probed` — the miner is naming a plausible mechanism
    it did not verify.
- A miner is not obliged to probe. It is obliged to say which it did, so
  the filer knows which premises need a verification round before body
  compose (three route-2 drafts on 2026-07-26 named mechanisms that were
  refuted at compose time — see the Evidence-counting discipline block
  immediately below for the firings-only counting convention that pairs
  with this field). Compose-side counterpart: the route-2/3 mandate's
  miner-field translation contract (#1874) keys the #1677 label on this
  field mechanically — an `inferred — not probed` (or field-less)
  mechanism ships labeled unless the filer probes it at compose time, a
  `probed:` one states fact citing the command, a positive incident claim
  cites its marker ts+kind (absence claims stay under the #1667 scan).

**Evidence-counting discipline — count `tool_result` firing EVENTS, never
raw string occurrences (#1484).** When quantifying an error/warn string in
a transcript ("N failed vs M ok"), count firing events: `tool_result`
blocks whose OUTPUT text contains the string, deduplicated per tool call
(the harness stores a duplicate top-level `toolUseResult` field alongside
the block, and a result's content can replay the command line before the
output — one tool call is ONE event). A match counts as a firing ONLY
when the string is the invoked command's OWN emitted output: a
`tool_result` whose command was a read/grep/inspection of files or logs
that CONTAIN the string, or a subagent report quoting the error, is a
content-readback echo, not a firing — attribute the firing to where the
error originally issued. Everything else is a recipe echo, NOT a
failure: the string inside `tool_use` input text (`command` strings),
inside ordinary `text` blocks quoting a recipe, or in the duplicate
`toolUseResult` field. Workflow recipes embed their own
warn/error messages verbatim in command preambles (`.claude/skills/issue/`
carries `[warn] choom failed …` in 6 command-recipe blocks), so a bare
`grep -c` over the raw JSONL inflates the count. Transcript rows carry
typed `message.content` blocks — filter on the block `type` (`tool_use`
vs `tool_result`) and dedupe per tool call with `jq`/python BEFORE
counting. Any quantified failure claim that reaches a filed body (route
2/3), the brief file, or the Telegram digest is firings-only and STATES
the counting method (e.g. "N `tool_result` firing events; K echoes
excluded"). Incident 2026-07-16: a raw string count read "13 failed vs
8 ok" for `[warn] choom failed` when only 1 of 13 was a real firing event
(the other 12: 3 command-text echoes, 6 quoted-recipe text blocks, 2
`toolUseResult` duplicates, 1 in-result command replay) — and that one
firing was the documented-benign pgrep-empty shape — spawning false
needs-human task #1473.

**Failure-lesson consolidation now runs DETERMINISTICALLY in
`scripts/consolidate_lessons.py` (a cron, NOT this skill) — task #711.** The
deterministic janitorial pass over `epm:failure-lesson v1` markers — (a) dedupe
the rolling-window lessons against the owning agent's memory, (b) promote
recurring lessons into `.claude/rules/gotchas.md`, (c) prune over-eager
`generalizes: yes` memory entries — was extracted out of this flaky 44K-token
LLM run into `scripts/consolidate_lessons.py` (cron `02 0 * * *` PT), so it no
longer depends on `/daily` completing. `/daily` NO LONGER owns failure-lesson
consolidation and MUST NOT dedupe/promote/prune agent memories or gotchas.md.
The per-lesson `/issue`-time routing (`generalizes: yes` → agent-memory write,
`gotcha_candidate: yes` → workflow-fix candidate) is unchanged and still fires
per-lesson. `~/explore-persona-space/.claude/agent-memory/**/*.md` is therefore
NO LONGER an allowed write target for this skill.

### Living-docs consolidation passes (folded in from the retired weekly skill, #713)

Nightly consolidation checks — the first two used to live in the `weekly` skill
(retired 2026-08-05; formerly `.claude/skills/weekly/`). Both
run every nightly `/daily`, both PROPOSE only (`docs/open_questions.md` mutations
are user-gated — the `living_docs_update` gate is user-only), both are deduped by
the shared event-stream below so a second nightly run does not re-propose. They
fill the `## Living-docs drift` section (drift + living-docs re-proposals) and
`## Other problems & notes` (follow-up revivals).

**1. Living-docs drift pass (Step A).** Run the read-only drift linter and write
the `## Living-docs drift` section:

```
findings, exit_code = run("uv run python scripts/living_docs.py check")   # read-only; NEVER `apply`
if exit_code == 0:
    write "## Living-docs drift" → "- _no drift — open_questions.md is in sync_"
else:
    drift_hash = sha256(normalize(findings))[:12]
    open_proposal = scan_for_open_drift_proposal(drift_hash)   # dedup, see "Dedup mechanism" below
    if open_proposal is None:
        write findings as bullets + the PROPOSAL line (the re-synthesis proposal)
        append {date, kind:"living-docs-drift", drift_hash, task_id:null, summary}
          → .claude/cache/nightly-consolidation-events.jsonl
    else:
        write findings as bullets + a note:
          "- _drift proposal already open (logged {open_proposal.date}); not re-proposing_"
```

`living_docs.py check` lints the living hub (`docs/open_questions.md`) for
`relates_to` ⇄ question-evidence mismatches, `completed`-with-`has_clean_result`
results missing from any question's evidence, dangling evidence `#N`, and
questions stale relative to new results; nonzero exit = drift. A proposed
re-synthesis is a HELD judgment call (it would mutate `open_questions.md`) — it
goes into the existing Telegram digest "HELD for you" segment, never auto-applied
(no new notification path). **Only ever call `living_docs.py check` — never
`living_docs.py apply` from /daily** (the `apply` writer stays on the user-gated
`/issue` `living_docs_update` gate path).

**2. Parked-proposal re-consideration pass (Step B).** Enumerate still-open parked
proposals and decide (LLM judgment) whether the context has changed enough that
each is now worth re-surfacing — never a mutation, only a SURFACE:

```
# (a) declined living-docs proposals — epm:living-docs-update-rejected v1
#     (lands only on `completed` tasks; the gate fires post-completion):
for t in task.py list-by-status --status completed --json:
    scan t.events.jsonl for an epm:living-docs-update-rejected with NO later
      epm:living-docs-updated (still un-reconciled).
# (b) parked-redundant follow-up proposals — saved as on_hold tasks:
for t in task.py list-by-status --status on_hold --json:
    keep those carrying the parked-redundant provenance (epm:followup-parked-redundant v1).

for each still-open parked item:
    if context changed (new results bearing on the same question / the
       duplicating sibling completed or was archived) AND not already re-surfaced
       (dedup below):
        - living-docs rejection → note under ## Living-docs drift as a re-proposal
          candidate (still a user-gated proposal, never applied); log
          {date, kind:"living-docs-reproposal", task_id, hash} to the event-stream.
        - parked-redundant follow-up → note under ## Other problems & notes:
          "consider reviving #M (no longer redundant because …)"; revival is
          `task.py set-status M proposed`, a mutation the user/PM performs — /daily
          only SURFACES it. Log {date, kind:"followup-revival", task_id} to the
          event-stream.
    else: skip silently (no spam).
```

`/daily` never flips `on_hold → proposed` and never runs `living_docs.py apply`
(consistent with the "do not move statuses unless the user asks" rule in
"Other rules" below).

**3. Parked workflow-fix-candidate routing pass (Step C).** Route the
recursion-guard escape valve (`.claude/rules/workflow-fix-on-bug.md`
§ Recursion guard): a workflow-fix session parks its candidates
(`epm:workflow-fix-candidate` with a `parked …` note) instead of auto-routing
them — THIS pass is the owning "next orchestrator pass". The /daily
orchestrator is NOT under the recursion guard (no `workflow_fix_target:`
Provenance line, no `EPM_WORKFLOW_FIX_SESSION=1`), so routing here cannot
recurse.

```
sweep = run("uv run python scripts/sweep_parked_wf_candidates.py")
    # one JSON object; UNBOUNDED window by default, already
    # routed-suppressed + row-deduped (suppression is what bounds re-scans)
for c in sweep["candidates"]:
    # fields: formal <!-- workflow-fix-candidate v1 --> block → verbatim;
    # prose park → SYNTHESIZE proposed_change/bug_observed per
    # workflow-fix-on-bug.md (prose-synthesis rule), then
    # fp = wf_fix_fingerprint(proposed_change, bug_observed)
    if an open OR closed infra task already addresses this bug
       (c["open_wf_fix_on_file"] advisory, the wf-fix-fp tag, or content match):
        post the routed-record below with note `deduped against #<M>`; continue
    route through the THREE-ROUTE classifier (the "Triage each problem"
    section BELOW this pass), with one override:
      # NOTE (2026-08-04): the `park_form == "architectural" → route 3`
      # override is REMOVED. The architectural-greenlight gate no longer
      # exists (.claude/rules/workflow-fix-on-bug.md § Architectural
      # greenlight — REMOVED), so an architectural park routes like any
      # other parked candidate — normally route 2, auto-filed + spawned.
      - DEFAULT route 2: a parked candidate is by construction a
        behavior-change proposal; route 1 only for a pure prose/doc change
        with no behavior effect (the route-1 litmus verbatim)
    file through the durable filing dir + daily_drive_filings.py as usual
      (route 2 tags: wf-fix + wf-fix-fp:<fp> + daily-auto-filed — a parked
       candidate is by construction workflow-surface, never `wf_fix: false`;
       the body carries
       the ## Provenance `- workflow_fix_target: <target_file>` +
       `- fingerprint: <fp>` lines per the route-2 wf-fix body Provenance mandate
       below — the driver injects them when absent)
    post the ROUTED-RECORD that closes the loop — for EVERY disposition
    (route 1 self-apply, route 2 filing, route 3 needs-human filing, AND
    dedup), no new marker kind:
      - candidate parked on task #N:
        task.py post-marker N epm:workflow-fix-task-filed --note
        "filed_task: #<M or 'n/a (route 1; commit <sha>)'> / target_file: <tf> /
         fingerprint: <fp or 'n/a (prose park)'> / session_spawned: <bool> /
         source: daily-parked-candidate-sweep / origin_candidate_ts: <c.ts> /
         origin_candidate: <first ~200 chars of the candidate note, abridged>"
      - cache-borne candidate: append the equivalent JSON row to
        .claude/cache/workflow-fix-events.jsonl (existing convention)
    record it in ## Applied workflow improvements ("filed for review #<M>") /
      ## Other problems & notes (route 3), exactly as routes 2/3 prescribe.
```

Routes 1 and 3 post the routed-record too — a route-3 needs-human filing DOES
have a real `filed_task: #<M>`; route 1 records the commit sha in place of a
task id. Without a record for every disposition, a route-3 park
would re-enumerate nightly; the record makes each disposition
sweep-idempotent. For a formal-block candidate the record's `fingerprint:`
field MUST carry the sweep-reported `c["fingerprint"]` copied VERBATIM —
never recomputed from abridged/synthesized origin text (#1680:
driver-recomputed fps differed from the sweep-canonical ones and broke
suppression for the #1630 trio; the sweep now also honors an exact
`origin_candidate_ts` claim as a fallback, but the verbatim fp stays the
PRIMARY key) — and `origin_candidate_ts:` MUST be `c["ts"]` copied verbatim:
the exact row ts is the suppression key for fp-less parks AND the #1680
fallback key for fp-bearing ones; `origin_candidate` (abridged verbatim)
keeps the marker self-describing per workflow-fix-on-bug.md § Markers.

Also check `sweep["skipped"]` (structured skipped-line records, #1680): an
entry with `relevant_kind` true or null is a possible LOST PARK — or, on a
FILED-kind hint, possibly lost SUPPRESSION EVIDENCE (a malformed
routed-record lets its park re-enumerate: the inverse failure direction) —
and warrants investigation; `relevant_kind: false` is benign (a malformed
line of irrelevant kind, the #1333 shape).

Also check `sweep["unmatched_record_fps"]` (structured advisory records,
#1703): a non-empty list is silent evidence of driver
fingerprint-recomputation drift (the #1630 class) — the ts-claim
fallback correctly suppressed the park, but the recomputed fp indicates
the routed-record driver operated on abridged/synthesized origin text.
Investigate per Edit-C of the wf-fix body verbatim-fingerprint mandate.
Advisory only; suppression semantics are unchanged.

Future sweeps skip routed candidates mechanically — see the enumerator's
suppression predicate (`scripts/sweep_parked_wf_candidates.py --help`).

**4. Clean-result title-sync drift sweep (Step D).** Run the read-only
corpus-wide H1-vs-frontmatter-title drift sweep (#1196) so the drift report
surfaces nightly:

```
uv run python scripts/audit_clean_results_body_discipline.py --title-sync-sweep
    # read-only (bodies are NOT modified); WARN-only — ALWAYS exits 0
    # (~2 s at the current ~1,200-body corpus). A NONZERO exit is a CRASH
    # (the sweep never FAILs by design): note it as a problem in
    # ## Other problems & notes — never treat it as a drift signal, and
    # never let it fail the daily run.
```

A `PASS:` headline (zero WARN rows) → nothing to record. Otherwise copy each
`- #<N> (<status>): ...` WARN row into `## Other problems & notes` as its own
bullet, verbatim — each row already carries both title values AND both
remediation commands (`task.py set-title` vs re-`set-body`). Which side is
the fresher intent is a HUMAN call: NEVER run the remediation from /daily
(same discipline as "never `living_docs.py apply`"); Thomas or the PM runs
the command from the note. No-spam variant: if the row set is IDENTICAL to
the previous daily file's, replace the verbatim copies with one line —
`- title-sync drift unchanged (<K> rows — see logs/daily/<prev-date>.md)`
(LLM judgment against yesterday's file; no new dedup state, matching Step
B's no-spam skip discipline).

**Dedup mechanism (Step F).** A filesystem event-stream
`.claude/cache/nightly-consolidation-events.jsonl` (a durable local trace — the
`.claude/cache/` dir is gitignored but this file is force-tracked and committed
alongside the daily brief via the § Commit pathspec — following the existing `.claude/cache/disk-guard-events.jsonl` /
`.claude/cache/workflow-fix-events.jsonl` pattern; created at runtime on the first
nightly proposal — do NOT create it ahead of time) is the canonical dedup state,
owned by `/daily` (the retired `weekly` skill shared this stream before
2026-08-05). Each proposal appends one row:
`{"date":"YYYY-MM-DD","kind":"living-docs-drift|living-docs-reproposal|followup-revival","hash":"<12-hex>","task_id":<int|null>,"summary":"<one line>"}`.
Dedup rules, in priority order:

1. **Drift re-synthesis** — `drift_hash = sha256(normalize(check output))[:12]`
   (`living_docs.py check` output is deterministic across back-to-back runs on
   unchanged repo state; if a future `check` interleaves a timestamp / elapsed-time
   token / unordered finding list, `normalize()` must strip those tokens + sort the
   finding lines before hashing, else the hash churns nightly and dedup fails open).
   A drift proposal is a duplicate (do NOT re-propose) iff EITHER (a) the
   event-stream already has a `living-docs-drift` row with the same `hash` whose
   proposal is still open, OR (b) the `open_questions.md` `<!-- living-docs-changelog -->`
   block carries a changelog line dated ≥ that row's date (the drift was already
   reconciled by an applied re-synthesis → a re-run of `check` exits 0 and there is
   nothing to propose). The changelog cross-check is the "already-applied" escape;
   the event-stream is the "already-proposed-not-yet-applied" escape.
2. **Parked living-docs rejection re-proposal** — keyed on
   `(task_id, content_hash_of_preserved_proposal)`. Duplicate iff the event-stream
   already has a `living-docs-reproposal` row for the same `(task_id, hash)` AND the
   task still shows no `epm:living-docs-updated`.
3. **Parked-redundant follow-up revival** — keyed on `task_id`. Duplicate iff the
   event-stream already has a `followup-revival` row for that `task_id` AND the task
   is still `on_hold` (not yet revived).

**Triage each problem into one of THREE routes (changed 2026-06-28, #706 — route 1 NARROWS the old "self-apply everything fixable" bucket to no-behavior-change-only; route 2 sends behavior changes to independent review; route 3 turns held judgment calls into tracked tasks the PM surfaces):**

**Freshness gate:** every problem that will FILE (routes 2 and 3) passes the
**Retraction re-check** (subsection below) before its filing body is composed —
a mined premise the source trail has since retracted or superseded is skipped
with a note, never filed (incident #1101/#1074, routed as #1131).

1. **Route 1 — Trivial mechanical** (doc edits, string replaces, comment/lint annotations — changes with NO behavior or logic effect) → APPLY it now (Edit the file), VERIFY it (see "Verification gate for code fixes" — workflow `.md` files skip this and use the lint gate instead), commit it on its own with trailing pathspecs naming exactly the file(s) the fix touched — `git add <file(s)>`, then `git diff --cached --name-only` (staged-set audit: foreign entries → log in `## Other problems & notes`, never commit, never unstage), then `git commit -m "<msg>" -- <file(s)>` — never a bare `git commit -m` after `git add` (§ Commit own-files-only contract), then record it in `## Applied workflow improvements` as a numbered entry WITH the applied diff and the commit sha (shape below). After the commit, run the **Route-1 covers-open-task closing step** (below) whenever the fix lands the deliverable of an already-filed OPEN task. One commit per fix so each is independently revertable. After ALL workflow-file fixes are committed, run the repo-wide lint gate ONCE (see "Lint gate"); if it regresses, revert the offending commit(s) and re-log them in `## Other problems & notes` as "reverted: failed lint gate". **The litmus is "does this change what the code/workflow DOES?"** — pure prose/string/format/comment edits, typo fixes, and stale-reference renames qualify; ANYTHING that alters behavior or logic does NOT (it goes to route 2). When unsure, route to 2 — route 1 is the conservative no-behavior-change floor, not a catch-all.

2. **Route 2 — any behavior/logic change** (incl. high-blast-radius REVERSIBLE — the bulk of the old self-apply bucket: experiment-code bugs in `scripts/*.py` / `src/**`, infra flakiness fixes, retry/timeout logic, hook repairs in `.claude/settings.json`, a new agent/skill file, any workflow-rule change that alters what an agent does) → do NOT self-apply. FILE a `kind: infra` task that auto-dispatches to `/issue --auto`, so the fix lands through the full INDEPENDENT pipeline (planner → adversarial critic ensemble → implementer → Claude+Codex code-review → test-verdict → Step 10d auto-merge) rather than this skill's own verify gate. This is what kills "grades its own homework" + "self-apply is the only un-reviewed path". File with:
   ```bash
   uv run python scripts/file_infra_task.py --kind infra \
     --title "daily-fix: <one-line, <=60 chars>" \
     --body-file <path to a body describing the bug + the proposed fix> \
     --tag wf-fix --tag "wf-fix-fp:<fp>" --tag daily-auto-filed
   ```
   (compute `<fp>` = `task_workflow.wf_fix_fingerprint(proposed_change, bug_observed)`; the `daily-auto-filed` tag distinguishes /daily-filed review tasks from manual workflow-fix-on-bug filings, and feeds the PM digest count M.) `file_infra_task.py` files + best-effort spawns `/issue --auto` in one call, and no-ops the spawn cleanly (the task stays at `proposed` for the watcher `proposed_infra_sweep` backstop) when the daemon is unreachable / the cap is full. Record the filed `#<N>` in `## Applied workflow improvements` as a **"filed for review #<N>: <one-liner>"** entry (there is no self-applied diff or sha for a route-2 item — the diff lands in the spawned `/issue` session). For experiment-code bugs that are NOT a workflow-surface gap, file the `kind: infra` task the same way (drop the `wf-fix` / `wf-fix-fp` tags, keep `daily-auto-filed`) so the bulk count still attributes it; in a driver manifest, set `wf_fix: false` on the item — `daily_drive_filings.py` then drops the two wf-fix tags, skips the Provenance injection, and skips fp-dedup (the `filed.jsonl` ledger + the title-scan recovery, both keyed on `daily-auto-filed`, remain the double-file protections for such items; do not hand-add a `workflow_fix_target:` Provenance block to a `wf_fix: false` body — the driver WARNs on one). When filing MORE THAN ONE task in a run, do NOT hand-loop this command: write the bodies + manifest to the durable filing dir and drive them through `scripts/daily_drive_filings.py` (see "Durable filing dir + incremental filing driver" below) in small batches (≤8 per Bash call), each call with an explicit `timeout` (≥300000 ms) — a 16-task sequential filing loop under the shared registry flock exceeded the default 2-min Bash cap mid-loop (2026-07-01, the 06-30 backfill run); the driver's `filed.jsonl` ledger makes a mid-batch kill resumable (re-invoke; filed slugs are skipped) instead of forcing a which-got-filed audit. Belt-and-suspenders (#1273): the driver prepends `daily-fix: ` to a route-2 manifest title missing a `WF_FIX_TITLE_PREFIXES` prefix (before the ≤60 truncation) — still compose titles WITH the prefix so the ≤60 budget is visible at author time, and note the guard exists ONLY on the driver path (a direct single-item `file_infra_task.py` invocation has no title guard).

   **wf-fix body Provenance mandate (durable recursion-guard signal, #1173):** every route-2 body that will carry the `wf-fix` tag MUST include a `## Provenance` section with the two lines `- workflow_fix_target: <target_file>` and `- fingerprint: <fp>`. The first is the DURABLE recursion-guard signal `task_workflow.is_workflow_fix_session()` reads — and it is the ONLY leg a daily-filed session ever has: `file_infra_task.py`'s `spawn-issue --auto` sets NO custom env, so `EPM_WORKFLOW_FIX_SESSION=1` is absent from the FIRST spawn (not just lost on a watcher crash-recovery respawn, which re-runs `spawn-issue --auto` with no custom env either) — a body without the line leaves the session unguarded (`.claude/rules/workflow-fix-on-bug.md` § Recursion guard + § Body-file template; incident #1134). The second is the body-side dedup fallback. `daily_drive_filings.py` INJECTS both lines into a route-2 body that lacks them (from the manifest `target` + the computed fp) — the mechanical backstop for the batch path — but a SINGLE-item direct `file_infra_task.py` invocation has no injector (`file_infra_task.py` only WARNs on a `wf-fix`-tagged body missing the line), so compose the body with the section regardless. A `wf_fix: false` manifest item (the non-workflow-surface route-2 variant above) is OUTSIDE this mandate by construction: it carries neither the `wf-fix` tags nor the injected block, so `task_workflow.is_workflow_fix_session()` correctly stays false for its spawned session — an experiment-code fix session is NOT recursion-guarded. **verified-at-filing mandate (#1272):** every route-2 wf-fix body ALSO carries a `verified-at-filing:` line in `## Workflow gap` — the grep command + hit count run at body-compose time, or `n/a — <one-line reason>` when the claim is not grep-verifiable (`.claude/rules/workflow-fix-on-bug.md` § Body-file template). The grep must BIND to the claim (#1307): state per-target hits for each file named in `target_file` — a presence claim with a 0-hit named target is a mis-target (re-grep repo-wide, correct the target, re-verify before filing) — and any nonexistence claim ("symbol/test no longer exists") records a repo-wide relocation grep, since a single-path probe cannot distinguish removed from moved (#1290/#1296). A presence hit additionally binds on CONTEXT: read the hit's surrounding lines — context that already implements the proposed change means the fix is already landed; dedup, don't file (#1330 filed over the landed #1309 fix; rule file clause (c)). An ABSENCE claim about a TEXT-MATCHING guard (an error-text substring list, a regex classifier, a transient-error predicate) additionally needs a semantic probe — run the predicate against the claimed text and/or grep fragments/substrings of it repo-wide — plus a landed-fix history check (`git log --oneline --since='7 days ago' -- <target_file>`); a verbatim-literal 0-hit grep alone is NOT absence evidence (#1386 filed over the just-landed #1360 fix; rule file clause (a')). Unlike the two Provenance lines, `daily_drive_filings.py` has NO injector for it — compose it by hand (#1221/#1229/#1249: three stale-claim filings in two days). **SHA-resolution (#1467):** the verified-at-filing duty extends to commit SHAs — run `git rev-parse --verify --quiet '<sha>^{commit}'` on every hex token the body cites as a commit, at compose time; a non-resolving token is cited as a transcript/session reference (what it actually is), never as a commit (rule file clause (d); #1414: transcript basename `fc2b61b7` filed as the fix commit). Binds route-2 AND route-3 filing bodies. `daily_drive_filings.py` carries a WARN-only scanner backstop (non-resolving commit-context hex tokens are annotated in the body + recorded as `sha_warnings` in the `filed.jsonl` row) — but the backstop cannot tell a mis-cited commit from a legitimately-cited external SHA, so compose-time verification stays the duty. **Artifact-state mutation check (#1497):** an artifact-state absence claim — a tag/field absent from a cited task's body.md read as "dropped at filing" — binds only after the task folder's git history shows the value was never applied: `git log --follow --format='%h %s' -- <body.md>` at compose time (current path via `task.py find <M>`; `--follow` traces the status `git mv`s; `task.py` commits are self-describing — `create — <title>` / `add-tag <tag>` / `remove-tag <tag>`); a `remove-tag <value>` commit explaining the absence, or a create-commit diff carrying the value, refutes the drop claim (rule file clause (e); #1497: every cited task was created WITH needs-human — a deliberate 2026-07-17 user-directed mass remove-tag, one commit per task, e.g. eaa4e10a67/1bd90f800e, explained every absence; archived as a false-premise filing). **Marker-existence (#1667):** a marker-existence claim — "no marker was posted" / "no record exists" on task #M's events stream — is verified at compose time with `uv run python scripts/task.py view <M> --json | jq '.events'` (or a matching kind+sentinel filter) — a found marker refutes the claim, and the real defect is a triage / read-back miss by the owning session, not the absent marker itself (rule file clause (f); #1667: `epm:progress v146` at 05:42:00Z carried the exact `[autonomous_session_watch:runpod-noport-wedge-failover]` sentinel the filed body denied). **Call-hop target tracing (#1669):** before naming `target_file`, trace the failing behavior one call-hop past the observed symptom — `grep -rn '<wrong-value-field>' src/ scripts/` from the symptom back to the site that CONSTRUCTS the wrong value (not the caller that consumes/propagates/reports it); record both sites and re-run the dedup fingerprint against the corrected target (the dedup key is `(target_file, fingerprint)`, so a mis-target silently weakens dedup) (rule file clause (g); #1669: filed target was the watcher that READ a `RunSpec` field written by `backend_poll._runspec_from_runpod_handle` + `backends/runpod.py` + `backends/issue_dispatch.py`; the shipped 13-file diff touched none of the named target). **Suppression-predicate (#1680):** a suppression-predicate claim — a candidate / park / record was DROPPED or LOST — binds only after enumerating the downstream tool's documented suppression predicates (`grep -n 'suppress\|skip\|dedup\|routed' <tool>`) and checking each against the specific record; a correctly-suppressed record refutes the claim — the real workflow-fix candidate is then observability (missing counter, unlogged suppression outcome), not "record lost" (rule file clause (h); #1680: filed "the #1642 park was lost by Step C" while the park was correctly suppressed by the `origin_candidate_ts` fp-less primary key, and the real gap was counter opacity). **Unverified-premise labeling (#1677):** a claim the filer could NOT mechanically verify at compose time — a timing / wall-time figure, an incident root-cause mechanism, an API- or tool-behavior assumption — is NEVER written as bare fact: label it `unverified hypothesis — verify at plan time: <claim>` (naming the recall source where known), so the spawned session's planner carries it as an assumption to verify, not a premise. Worked example: write `unverified hypothesis — verify at plan time: the phase timeout is 3600s (recalled from the incident session, not read from the script)` — never "the phase timeout is 3600s" (#1660's filed 3600s was a mis-recollection vs the real `ddp_timeout=10800` at `train_behavior_fullft.py:650`; in the 2026-07-24 wave 4 of 6 route-2 bodies carried refutable bare-fact premises — #1655/#1662/#1646/#1660 — each burning a fact-checker/critic correction round) — and when the value IS readable from a named script, read it and state fact: label only what genuinely cannot be checked at compose time. The label COMPLEMENTS the verified-at-filing line, never substitutes for it: a mechanically-verifiable claim still gets its grep/rev-parse/git-log/code-read, and a claim you DID verify (or quote from a durable artifact) is stated as fact with its evidence — do not blanket-label; labeling everything dilutes the signal to nothing. **Miner-field translation contract (#1874):** the miner output schema's `probed-vs-inferred` field (§ Miner output schema above) keys the #1677 label MECHANICALLY at compose time — three rules: (i) a mechanism claim sourced from a miner finding marked `inferred — not probed` ships labeled `unverified hypothesis — verify at plan time: <claim> (miner-inferred, not probed)` — UNLESS the filer runs the probe itself at compose time, whereupon the claim becomes a rule-(ii) claim citing the FILER's probe (this preserves #1677's "when the value IS readable, read it and state fact": the miner marker sets the DEFAULT, the filer's own compose-time probe is the sanctioned upgrade path); a labeled claim is never merged with a probed sibling finding into one mechanism story (the #1814 conflation shape); (ii) a claim sourced from a finding carrying `probed: <cmd>` — or upgraded per (i) by a filer-run probe — states fact and cites the probe command as its PROVENANCE; the miner's probe does NOT substitute for the filer's own compose-time `verified-at-filing:` re-run (the #1272 freshness-at-filing mandate is unchanged: a stale emitter-side probe never satisfies it); (iii) a POSITIVE incident claim (a session did X / something fired at time T) cites the marker `ts`+`kind` on the cited task's events stream where one exists (`task.py view <M> --json` read at compose time), else the transcript row; a positive incident claim with neither citation falls under (i) and ships labeled. CARVE-OUT: a marker-ABSENCE claim ("no marker was posted" / "no record exists") stays under the #1667 mandatory compose-time events scan — refuted, or stated as fact with the scan cited — NEVER merely labeled (labeling is not an escape from the scan; #1798, a motivating incident, was an absence claim). Fail-open default: a miner mechanism claim carrying NEITHER field marker is treated as `inferred — not probed` (rule (i)). The contract binds route-2 AND route-3 bodies (same scope as #1677) and is NOT exhaustive: claims with no miner provenance at all (a user report, an orchestrator observation) stay under the judgment-keyed #1677 clause above. (Incidents #1814 + #1798, 2026-07-28: each shipped an unverified mechanism claim as bare fact and burned a critic REVISE r1.) **Delegated composition restates the duties:** when filing-body composition is delegated to a compose subagent, the spawn brief restates the SHA-resolution clause (d) verbatim — rev-parse every hex token cited as a commit, else cite it as a session/transcript reference — because the subagent does not inherit this skill's context (2026-08-02: the c8 composer shipped session basename `8fc069db` uncited as a hex token and only the driver's WARN backstop annotated it, #2003).

   **Mechanical landed-fix probe (#1674):** the driver (`daily_drive_filings.py`) now runs a mechanical landed-fix probe before EVERY filing (routes 2 AND 3): `git log --since='7 days ago'` scoped to the item's own target file(s); a commit SUBJECT sharing >= 3 informative tokens with the item's title+bug+change text records the terminal ledger outcome `landed-fix-suspect` — no task filed, no session spawned; the suspect sha(s) + subjects + shared tokens are recorded for the eyeball. Override: re-run the slice with `--retry-suspects` (the flag is invocation-global; per-slug control is `--start`/`--end` slicing). Fail-open on git errors (one loud stderr WARN, filing proceeds). The compose-time clause-(a') semantic probe + git-log duty above STAY PRIMARY — subject-only matching misses vocabulary-divergent landed fixes (the measured #1386/#1360 case).

3. **Route 3 — a genuine judgment call** (the judgment-call carve-out below, VERBATIM) → FILE a TRACKED `proposed` task the PM surfaces + re-surfaces, AND log it in `## Other problems & notes`. The held item is no longer a dead-end note that lands where Thomas does not look (a tracked-but-unread `logs/daily/` file + one Telegram digest line) — it becomes a real task in the `proposed` queue, tagged so the PM enumerates it in its `Needs you` block every STATUS pass until Thomas acts on it. File with:
   ```bash
   uv run python scripts/file_infra_task.py --kind infra --no-dispatch \
     --title "daily-held: <one-line, <=60 chars>" \
     --body-file <path to a body describing the held item + WHICH carve-out item held it> \
     --tag daily-held --tag needs-human
   ```
   `--no-dispatch` files the task at `proposed` WITHOUT spawning a session (the PM, not an autonomous sweep, decides what happens). The `needs-human` tag is the auto-dispatch-exclusion signal: the watcher's `proposed_infra_sweep` candidate query SKIPS any `proposed` infra task carrying it, and the PM enumerates it in `Needs you` instead. Record the held item in `## Other problems & notes` as a bullet: what happened (session id / task id) + which carve-out item held it + the filed `#<N>` + a one-line suggested action. Route-3 bodies live in the SAME durable filing dir and file through the same driver (a manifest item with `route: 3` ⇒ the driver adds `--tag daily-held --tag needs-human --no-dispatch`); single-item runs may still invoke the command above directly — noting that a direct filing sits OUTSIDE the `filed.jsonl` ledger guarantee (route-2 fp-dedup covers a kill-window re-raise; the route-3 open daily-held overlap dedup (#1483) covers a same-subject re-raise on the driver path only — a direct single-item filing gets neither), so the body still goes in the durable dir first. **Route-3 open daily-held overlap dedup (#1483):** before filing a route-3 item the driver scans OPEN tasks tagged `daily-held` (statuses `proposed`/`on_hold`/`blocked`) for ≥3 shared informative title+bug tokens — excluding route-3-generic workflow vocabulary + date tokens — AND ≥1 shared subject token between the two titles (#1687: generic workflow prose alone never suppresses) (`find_open_daily_held_duplicate`; tokens per `task_workflow.informative_title_tokens`, task side = frontmatter title + template-stripped `origin_prompt` — a long-lived held condition re-worded nightly re-files otherwise, #1140/#1472); a hit records ledger outcome `already-tracked` against `#<M>` and does NOT file; any scan error fails OPEN toward filing (a held item is never silently dropped). A daily-held task lacking `origin_prompt` (hand-filed / pre-#1061) degrades to title-only tokens on the task side — a recall-only impact that fails toward filing.

### Retraction re-check (freshness gate before route-2/route-3 filings)

The sweep mines a point-in-time snapshot; the correction can already sit later
on the same trail, or land between mining and filing. Incident #1101/#1074
(2026-07-06): the sweep filed #1101 from #1074's `epm:pod-terminated v1` prose
follow-up while an explicit retraction (`epm:progress v26`, "no workflow gap;
nothing to file") sat 37 seconds later on the SAME events.jsonl — the spawned
session burned a full spawn + a ~6-minute clarifier pass before archiving
won't-fix. So, per mined problem, IMMEDIATELY BEFORE composing its filing body
(routes 2 AND 3 — before the durable-dir write, so a skipped problem never
enters the manifest):

1. **Marker-mined problems** (evidence E = a marker on task #M — including a
   parked `epm:workflow-fix-candidate` being routed): re-read #M's events
   NEWER than E, fresh at filing time:
   ```bash
   uv run python scripts/task.py view <M> --json | jq -r --arg ts "<E.ts>" \
     '.events[] | select(.ts > $ts) | "\(.ts) \(.kind) v\(.version // 1): \((.note // "")[0:400])"'
   ```
   (Only the filtered rows enter context — never page the raw events.jsonl.)
   Read the returned notes for a correction of the mined premise. The regex
   `RETRACTION|retract(ed|ion)|supersed|no workflow gap|nothing to file` is a
   RECALL aid for which rows to read closely — it never decides by itself.
2. **Binding test (what actually suppresses a filing):** skip ONLY when a
   newer note UNAMBIGUOUSLY addresses the mined premise — it names E's marker
   kind (e.g. "epm:pod-terminated v1"), quotes a distinctive phrase of E's
   note, or is an explicit retraction whose subject is plainly the mined
   claim. A note that merely contains "retract" / "nothing to file" about
   something ELSE never suppresses. Ambiguous → FILE anyway and note the
   possible retraction in the filing body (the spawned session's clarifier is
   the existing backstop — fail OPEN toward filing; a wrongly-skipped filing
   has no dedup tag, so a later re-raise is never blocked).
3. **Transcript-mined problems** (no source-task marker): before composing
   the body, re-check the remainder of the SAME session transcript after the
   mined excerpt for a correction/self-retraction of the premise (same recall
   regex, same binding test); when the excerpt names a task #M, ALSO run
   check 1 on #M. This is a targeted tail re-check at filing time, not a
   transcript re-read. Evidence with NO source-task marker AND no session
   transcript tail (e.g. a cron sidecar JSONL row, a bare log line) has
   nothing to re-scan — no scan is owed; fail open and file as today.
4. **On a suppressing hit:** do NOT file. Record the problem in `## Other
   problems & notes` as: `skipped filing: premise retracted upstream — #<M>
   <E.kind> at <E.ts> retracted by <kind> v<n> at <ts>: "<short quote>"`.
   It counts in neither the applied (N) nor filed (M) Telegram counts.

For auditability, also record one summary line per run in `## Other problems
& notes` — `retraction re-check: N marker-mined problems scanned, K skipped` —
so a no-hit night still shows the gate ran. Route 1 self-applies rather than
files; its existing verify gate already
requires confirming the problem against current reality, and a route-1 fix
whose premise came from a mined marker confirms against the source trail the
same way. A backfill RESUME of an already-composed `logs/daily/filings-<date>/`
dir does not re-run this check per slug — fresh compositions on a backfill DO
run it (the backfill section's current-tree confirmation is the same
principle).

### Durable filing dir + incremental filing driver (routes 2 + 3)

- **Durable-first ordering:** BEFORE the first filing call, write every route-2/3 body to `logs/daily/filings-<date>/<slug>.md` **under the repo root** (`~/explore-persona-space/logs/daily/...` — never a worktree-relative or /tmp path) and a `manifest.json` (`[{slug, route, title, target, bug, change, body?, wf_fix?}]`) in that dir, written via temp+rename. NEVER stage filing bodies or driver scripts in bare /tmp (2026-07-03: a mid-filing kill stranded 10 of 13 route-2 bodies in /tmp; the backfill spent ~40 min recovering them — #1061).
- **Drive incrementally:** `uv run python scripts/daily_drive_filings.py --dir logs/daily/filings-<date> --start I --end J` in batches ≤8, each Bash call with an explicit `timeout` (≥300000 ms). The driver appends every outcome to `<dir>/filed.jsonl` immediately (two-phase `attempting` → `filed|deduped|ERROR|recovered` rows), computes the route-2 `wf-fix-fp` dedup and the route-3 open daily-held overlap dedup (#1483), and applies exactly the per-route tags of the two command blocks above — the command blocks stay the CONTRACT; the driver is the multi-item execution path. A route-2 item with `wf_fix: false` (experiment-code / non-workflow-surface) keeps `daily-auto-filed` only and is exempt from injection + fp-dedup. Never run two drivers on the same dir concurrently (the ledger has no locking; the existing "no backfill within 60 min of the nightly" rule bounds the realistic case). The driver also normalizes route-2 bodies in place before filing (skipped for `wf_fix: false` items): a body lacking a `workflow_fix_target:` line gains `- workflow_fix_target: <manifest target>` (+ `- fingerprint: <fp>` when absent) under `## Provenance` — the durable recursion-guard signal; see the wf-fix body Provenance mandate in route 2 above.
- **The dir stays untracked** (`logs/` is gitignored; do NOT force-add filings dirs — the post-filing durable record is the filed task itself under `tasks/`, committed by `task.py new`; only the daily `.md` is force-added per "Commit" below).
- **Daily-file record:** route-2/3 entries in `## Applied workflow improvements` / `## Other problems & notes` take their `#id`s from `filed.jsonl`; any slug without a terminal row (or a filed id whose `tail` shows the spawn deferred to the watcher backstop) is recorded as "dispatch pending — see logs/daily/filings-<date>/filed.jsonl". An `ERROR` slug is recorded as "filing FAILED (<flag>) — retry: `uv run python scripts/daily_drive_filings.py --dir <dir> --retry-errors`" (never silently dropped, never rendered as pending); a `recovered` row with `dispatch_unconfirmed` is recorded as "filed #<id> (recovered; dispatch unconfirmed — watcher backstop covers it)". An `already-tracked` row (route-3 overlap dedup, #1483) is ALWAYS recorded in `## Other problems & notes` as `already tracked in #<id> (shared: <tokens>): <slug one-liner>` — this line is the primary human-visible channel for spotting a wrong match, so it is unconditional (it counts in neither the applied-N nor the filed-M Telegram counts; report it in the held-J context) — and the orchestrator additionally posts ONE best-effort freshness note on the existing task: `uv run python scripts/task.py post-marker <id> epm:progress --note "still live: /daily <date> re-observed this held decision (<slug>); duplicate filing suppressed (#1483)"` — fail-soft (a failed post never blocks the run; never re-post on a deferred-commit stderr warning). The driver itself posts no marker.

**Judgment-call carve-out (the ONLY things routed to route 3 — per Thomas 2026-06-12: "unless there's REALLY a judgement call needed"):**
- **Scientific-meaning changes** — anything that alters how results are computed, evaluated, or interpreted (metrics, eval criteria, analysis logic, hypothesis framing, RESULTS.md claims). A wrong silent fix here can flip a conclusion; Thomas decides.
- **Destructive / irreversible actions** — deleting or rewriting data, eval results, checkpoints, task history; anything NOT undoable by a single `git revert`.
- **Spends money or launches compute** — pod spin-ups, paid API runs, anything with a bill.
- **External side-effects** — sends, posts, pushes to remote, anything leaving the machine (existing rule: do not push).
- **Genuinely ambiguous intent** — two reasonable fixes diverge AND picking wrong would mislead later work. If a competent engineer would just fix it without asking, it is NOT in this bucket.

This 5-item list is now the **route-3 trigger**: a problem matching any item is FILED as a TRACKED `proposed` `needs-human` task (route 3 above) that the PM surfaces + re-surfaces in its `Needs you` block — it is no longer merely a dead-end note. (A problem that does NOT match any item is route 1 if it has no behavior change, else route 2.)

**Verification gate for code fixes** (route-1 items touching `*.py` / `*.sh` / configs / hooks): before committing, verify the fix — reproduce the original failure if cheap, run the file's tests if they exist, or at minimum a syntax/import check (`uv run python -c "import <module>"`, `bash -n`, or the script's `--help`) plus a targeted smoke check of the changed path. A fix that cannot be verified tonight is NOT committed — log it in `## Other problems & notes` as "unverified fix drafted: <why>". Never weaken a verification to make it pass.

**Full-ruleset lint for touched pin-listed `scripts/*.py` (the per-file-ignores masking trap — #1854):** bare `ruff check` is NOT sufficient verification for a maintained workflow helper under `scripts/`: the #1023 `[tool.ruff.lint.per-file-ignores]` table (pyproject.toml) masks style rules (E501 etc.) on `scripts/*`, while the fleet's gate test `tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset` lints its `LIVE_WORKFLOW_HELPERS` pin list with that table neutralized — so a bare-ruff-clean edit to a pin-listed helper can turn main red fleet-wide (incident: commit `c341f3bd59`, the 2026-07-28 route-1 run, an E501 in `scripts/verify_task_body.py`; main red ~40h). Before committing a route-1 fix that touches a `scripts/*.py` file on that pin list, run the BINDING check `uv run pytest tests/test_ruff_policy.py -q` (a few seconds); the per-file read `uv run ruff check --config 'lint.per-file-ignores = {}' <touched files>` is a fast diagnostic for pin-listed files ONLY — never treat it as an equivalent bar for non-pinned files. A script NOT on the pin list (frozen per-issue scripts, the dropped dirty stragglers) is EXPECTED red under that per-file full-ruleset read on pre-existing errors — not a route-1 blocker; the pytest node is the only binding check, and it covers only the pin list. Backstop: the `guard_root_code_commit.sh` inline lint gate's mapped tests include `tests/test_ruff_policy.py` for pin-listed helper payloads, so this up-front run is what keeps the commit from bouncing off the hook — it does not replace the hook.

**Route-1 covers-open-task closing step (#1918):** a route-1 fix that lands the FULL deliverable of an already-filed OPEN task closes that task in the same pass — never leave its spawned session to boot, discover mootness, and archive itself (incident: commit 7202cfd669 crediting #1823; the spawned session booted only to archive itself, ~3 min pure overhead). Candidate triggers (any one — these NOMINATE a candidate task ONLY; step (1) is the sole archive license): (a) the problem's mined evidence names an open task id; (b) the route-1 commit subject credits a task id; (c) an open `daily-auto-filed`/`wf-fix` task's `workflow_fix_target:` names the touched file AND shares >= 3 informative title+bug tokens with the item (trigger (c) is known-noisy on hot workflow files — ~21/24 false-positive measured in #1735 — hence nomination-only). Then, in order: (1) VERIFY full coverage — run the task's named failing test / acceptance check when one exists, else confirm every element of the task's proposed change is present in the landed commit (`git show <sha>`); ambiguous or partial coverage → leave the task OPEN and log "possible overlap: commit <sha> may cover #<id>" in `## Other problems & notes` — never archive on a guess. (2) STOP any spawned session mapped to the task BEFORE archiving (`uv run python scripts/spawn_session.py list` → `stop --session-id <sid>`): a mid-boot session issues its own status flips and can reactivate an archived task moments later, recreating the wasted-spawn loop; if the stop fails or no session is visible, archive anyway and note the possibly-live session in the record (the watcher reconcile pass is the backstop). (3) Archive through the canonical API only: `uv run python scripts/task.py set-status <id> archived --note "superseded: route-1 commit <sha> (/daily <date>) landed the full deliverable; verified: <test node or diff check>"` — never a hand-edit — then confirm the terminal status stuck (`uv run python scripts/task.py view <id> --json` shows `archived`). (4) Record "closed #<id> (superseded by <sha>)" on the route-1 entry in `## Applied workflow improvements`. Same-run ordering: when the SAME run has queued the problem for a route-2 filing that a route-1 fix now covers, DROP the manifest item BEFORE the FIRST `daily_drive_filings.py` invocation (nothing filed ⇒ nothing to archive); once any batch has run, do NOT re-slice indices — take the archive path above, or re-invoke the driver over the full range (the `filed.jsonl` slug ledger skips already-filed slugs). Re-runs are idempotent: all triggers scope to OPEN tasks, so an already-archived task no-ops.

**Allowed target files** (project workflow — global files are handled by `/memory-sleep`):
- `~/explore-persona-space/CLAUDE.md`
- `~/explore-persona-space/.claude/CLAUDE.md` (if present)
- `~/explore-persona-space/.claude/agents/*.md`
- `~/explore-persona-space/.claude/skills/**/SKILL.md`
- `~/explore-persona-space/.claude/rules/*.md`
- `~/explore-persona-space/.claude/workflow.yaml`
- (since 2026-06-12) `scripts/*.py`, `src/**`, `.claude/settings.json` hooks, env/config files — subject to the verification gate + judgment-call carve-out above.

**Applied-edit record shape**: each applied fix is a numbered list item with this structure (written AFTER the edit + lint + commit succeed):

```markdown
1. **Target:** `<file path>` — **what:** <one-line description> — **commit:** `<sha>`
   **Why:** <triggering pattern, quoted transcript excerpt with session ID if possible>
   **Applied edit:**
   ```diff
   - <old line if modifying or deleting>
   + <new line>
   ```
```

**No cap — be exhaustive.** Apply + record every workflow-fixable problem as its
own entry and every other problem as its own note. Do NOT drop items to hit a
number. Order both sections by severity so the important ones are on top
(rules of thumb: Thomas's own corrections / blockers first; foundational files
like project CLAUDE.md before niche skill files; problems that cost real time
before cosmetic ones). If several small related items share one fix, you may
group them under a single applied entry (one commit) with sub-bullets — grouping
is fine, dropping is not.

### Lint gate

`workflow_lint.py` is a **repo-wide** validator, not a per-file `.md` linter (its `--file` flag only points at `workflow.yaml`; passing an `.md` path makes it try to parse that file AS workflow.yaml and falsely fail). So do NOT run it "per touched file". Instead, after ALL bucket-1 fixes are committed, run it ONCE for the whole repo:

```bash
export PATH="$HOME/.local/bin:/snap/bin:$PATH"
uv run python scripts/workflow_lint.py --check-references
```

- `--check-references` is the gate (it currently PASSes clean, so a new failure means a just-applied edit broke a workflow reference). Use the `uv run python …` form — the linter imports pydantic/PyYAML and needs the EPS venv; a bare `scripts/workflow_lint.py` in the cron shell will `ModuleNotFoundError`.
<!-- example: anti-pattern -->
- `--check-asks` is ALSO a gate (it now PASSes clean repo-wide, since the `issue/SKILL.md` mentions were annotated): a new `--check-asks` failure means a just-applied edit added an un-annotated `AskUserQuestion` mention — annotate it (`<!-- gate: <key> -->` resolving in `workflow.yaml § gates`, or `<!-- example: anti-pattern -->` for a forbidden-use / meta mention) or revert that edit, same discipline as `--check-references`.
- **On regression** (`--check-references` was clean and is now failing): the failure is from a just-applied edit. Identify the offending commit and revert it via a scratch worktree (the root guard blocks a repo-root `git revert`, #1234): `git worktree add --detach /tmp/daily-revert origin/main && git -C /tmp/daily-revert revert --no-edit <sha> && git -C /tmp/daily-revert push origin HEAD:main && git worktree remove /tmp/daily-revert` (do not hand-edit; route-1 fixes are committed AND pushed before this gate re-runs, so `origin/main` contains the sha). Then move that item to `## Other problems & notes` as "reverted: failed lint gate (<error>)", and re-run the gate until it is green again. Then continue to surfacing.

### Surfacing flow (applied / filed / held)

During the run: route-1 fixes apply themselves (edit → pathspec-limited `git commit` (own files only, § Commit), one commit per fix → repo-wide lint gate ONCE, see "Lint gate"); route-2 behavior/logic changes are FILED via `file_infra_task.py` (no self-applied diff); route-3 judgment calls are FILED as tracked `needs-human` tasks. After all fixes are routed and the daily file is written, **surface a concise summary to Thomas's my-goat Telegram chat** by enqueuing it into the my-goat notification digest. Match the §2(f) PM-digest model (`/daily last night: applied N, filed M (→/issue), held J (needs you)`) so the Telegram line and the PM line are mutually consistent — applied (route-1), filed (route-2), and held (route-3) counts are reported SEPARATELY, never lumped as a single "Notes: M other" catch-all:

```bash
NOTIF_CAT=research /home/thomasjiralerspong/my-goat/scripts/notif_enqueue.sh "EPS daily <date>: applied N route-1 fix(es), filed M route-2 review task(s), held J route-3 (needs you). Applied: 1) <one-liner> (<sha>). 2) <one-liner> (<sha>). Filed: <#id> <title>. Held (needs you): <#id> <held-item one-liner> (<carve-out reason>). Revert any applied via a scratch worktree (root revert is hook-blocked, #1234): git worktree add --detach /tmp/daily-revert origin/main && git -C /tmp/daily-revert revert --no-edit <sha> && git -C /tmp/daily-revert push origin HEAD:main && git worktree remove /tmp/daily-revert. Full: logs/daily/<date>.md"
```

This lands in the next my-goat morning digest (the dispatch cron runs 9/14/19 PT), so the overnight `23:27 PT` run is reviewed when Thomas is fresh rather than buzzing him at bedtime. Keep the message short: the three counts (applied N / filed M / held J), a one-liner + sha for each route-1 applied fix (so any applied fix is one `git revert <sha>` away), the filed `#id`s (route-2 review tasks), and the held `#id`s (route-3 needs-human tasks) with their carve-out reason, plus the daily-file path. If zero fixes were applied, zero were filed, AND zero notable problems were logged, enqueue nothing (don't send an empty digest line).

The old `SessionStart` greenlight hook (`scripts/daily_surface_hook.sh`) is now vestigial: it greps for `## Proposed workflow improvements`, which this skill no longer writes, so it stays silent and never prompts for a greenlight. Leave it in place (harmless). Surfacing is Telegram-only now. (Since 2026-06-12, `.claude/settings.json` hook changes route through the three-route classifier: a stylistic fix with no behavior change can self-apply via route 1, but any hook edit altering what an agent does — adding/removing a hook, changing a hook's trigger, modifying a matcher pattern, retargeting the script — is a behavior change and FILES via route 2, exactly like every other behavior/logic change. This skill's own verify gate is NOT sufficient for a hook behavior change; route 2 routes it through the full independent review pipeline.)

Applied edits stay in the daily file as historical record — don't delete them. If Thomas reverts one, that's via git; the record stays.

### Commit

After writing the file, commit it so the dashboard picks it up, then push
it immediately so a concurrent rebase cannot orphan it off `main` (the #711
incident class):

```bash
git add -f logs/daily/YYYY-MM-DD.md   # logs/ is gitignored; force-add or the commit silently stages nothing
# Staged-set audit (own-files-only contract, #1630): the shared root routinely
# holds ANOTHER session's staged files (incident 7dbde267f1, 2026-07-21: the
# nightly's bare enrichment commit swept a sibling's script + 3 figures onto
# main). List the staged set; anything beyond the two own-paths below is a
# concurrent session's work-in-progress — do NOT commit it and do NOT unstage
# it (`git restore --staged` on a sibling's entries clobbers ITS in-progress
# index — the same cross-session interference class). Log foreign paths in
# `## Other problems & notes` ("staged-set audit: foreign staged files:
# <paths>") and proceed — the pathspec-limited commit below cannot sweep them.
git diff --cached --name-only
# Pathspec-limited commit (git-commit --only semantics): commits ONLY the
# named paths' WORKTREE content — even when unstaged (by design: the named
# paths are this run's own files) — ignoring every other staged entry; NEVER
# a bare `git commit -m` anywhere in this skill. The event-stream jsonl is
# force-tracked (.claude/cache/ is gitignored; this file is tracked) and
# rides along when this run appended rows; naming it while unchanged is
# harmless (the commit simply omits it). If NEITHER named path has changes
# vs HEAD, the commit exits NONZERO — key the disposition on the exit code,
# not the message text (it varies: "no changes added to commit" vs "nothing
# to commit") — the content already landed (e.g. a rerun); log it and carry
# on, same ethos as the push-not-confirmed paragraph below.
# Run BARE — never `| tail`/`| head`/`| grep`. `guard_piped_git_push.sh`
# BLOCKS a piped git commit/push/merge (it masks the exit code, and SIGPIPEs
# the pre-commit hook mid-run, #1584). Need the text? `> /tmp/out 2>&1` then
# read the file. Same for the `git push` below.
git commit -m "logs: daily stub for YYYY-MM-DD" -- logs/daily/YYYY-MM-DD.md .claude/cache/nightly-consolidation-events.jsonl
# Push IMMEDIATELY (project-standard recipe). The daily-stub commit sits on
# the always-concurrent shared main; if it stays committed-but-unpushed it
# is exposed to the documented orphaning hazard — a concurrent
# `git pull --rebase=merges` can rewrite/drop the task-state commit it sits
# on top of and take the daily with it (#711, 2026-06-27). On a rejected
# push, recover through the single-flight root-sync helper — NEVER a
# hand-rolled pull-rebase loop (CLAUDE.md "Concurrent repo-root
# committers"; canonical form (2), issue/SKILL.md Step 10d § "Bare push /
# merge snippets"). sync_repo_root exit 0 can mean "another sync
# in-flight — your push has NOT landed"; that is acceptable for this stub
# (see the failure paragraph below). Never force-push.
git push origin main || uv run python scripts/sync_repo_root.py
```

Under stub-first (headless rule 0) this recipe runs TWICE: once for the
skeleton at run start (`logs: daily stub for YYYY-MM-DD`) and once after
enrichment (`logs: daily brief for YYYY-MM-DD`). Always a NEW commit for the
enrichment — never amend the stub commit (it may already be pushed).

**Own-files-only contract (#1630).** A nightly run may only ever commit its
OWN files: `logs/daily/<date>.md` + `.claude/cache/nightly-consolidation-events.jsonl`
here, and the exact file(s) a route-1 fix touched for its per-fix commits.
Every commit this skill prescribes is pathspec-limited
(`git commit -m "..." -- <own paths>`), never a bare `git commit` after
`git add` — the shared repo root is an always-concurrent index, and commit
`7dbde267f1` (2026-07-21) showed a bare enrichment commit sweeping a sibling
session's 4 staged files onto main. The `git diff --cached --name-only`
audit line before each commit is the audit trail; foreign staged entries are
logged, never committed, never unstaged (unstaging a concurrent session's
index entries is itself cross-session interference). Run the audit + commit
lines BARE, never piped (`guard_piped_git_push.sh` blocks piped `git commit`).

**`logs/` is in `.gitignore`** — a bare `git add logs/...` stages nothing,
and the § Commit pathspec-limited commit then fails on a brand-new day's
file with "did not match any file(s) known to git" (nonzero exit), so the
daily never lands in git or the dashboard. `-f` is required (the prior
dailies are tracked only because they were force-added).

**This immediate push is specific to the daily-stub log file** — a benign
`visible:false` artifact with no external side-effects (it only lands the
dashboard's view of the stub). It does NOT relax the standing
"External side-effects → route 3" / "do not push" rule for **route-1
code/behavior fixes**: a code or behavior change still does NOT push from
this skill (it routes to review per the three-route classifier above;
external side-effects Thomas reviews first are route 3 — see the
"Judgment-call carve-out", whose "External side-effects … do not push"
item governs reviewable changes leaving the machine). The two rules govern
different things — the no-push rule governs reviewable code/behavior
changes; this push only persists the day's own log stub.

If the push still isn't confirmed landed after the recovery —
`sync_repo_root.py` exits non-zero (e.g. a network/GH blip, a content
conflict, or exit 3 = push failed after its one retry), or exits 0 while
reporting in-flight (`sync_repo_root: state=in-flight` on stderr — another
session's sync is running; THIS push has NOT been confirmed) — do NOT
hang or abort the run: log it loudly in `## Other problems & notes`
("daily stub committed locally but push not confirmed: <rc / state /
error>") and carry on. The stub is committed locally; the #711 heartbeat
(`scripts/cron_daily_healthcheck.sh`) will surface a still-missing stub the
next day as the backstop. Durability here is best-effort; nothing in the
project hangs on the push succeeding — which is exactly why the bare form
(2) suffices here and the landing-verified variant is unnecessary.

### Headless (cron) mode — the daily file is never hostage to background work

The nightly entrypoint is HEADLESS: crontab
`27 23 * * * sh -c "cd ~/explore-persona-space && CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=10800000 claude -p /daily"`.
In `-p` mode the harness TERMINATES the process when background tasks are
still running `CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS` (cron pins 3h; harness
default 600s) after the final turn ends. On 2026-07-01 the run ended its
turn "waiting on the filing driver" with the daily file unwritten and was
killed at the 600s default; the 07-02/07-03 nights also died with their
files unwritten (one logged nothing, one left a kill-less trailing-wait
tail) (#994). Three rules:

0. **Stub-first (mechanical — #1189).** IMMEDIATELY on run start — before any
   transcript mining, task reads, routing, or subagent dispatch — if
   `logs/daily/<date>.md` does not exist, write it as the dated skeleton
   (§ Output first bullet: frontmatter + the six H2 headers, bodies EMPTY),
   then commit + push it with the § Commit recipe
   (`git add -f` + `logs: daily stub for <date>` + immediate push). If the
   file already exists, rule 0 is a no-op — the § Output existing-file rules
   govern (a husk from a prior kill lands on the Edit-in-place branch). All
   later output is Edit-in-place enrichment of this file. Why mechanical: in
   `-p` mode the prose ordering rules below were violated within hours of
   shipping (the 07-03 backfill died "waiting on the filing driver" with no
   file); the skeleton guarantees a durable artifact from minute 0, and
   `scripts/cron_daily_healthcheck.sh`'s husk arm distinguishes a
   never-enriched skeleton (empty `## Applied workflow improvements` at
   06:00) from a completed brief.
1. **Write-then-wait ordering.** The daily file write + `git commit` + push +
   Telegram enqueue are the load-bearing outputs — never gate them on a
   background task. Once mining + routing results are in hand, ENRICH the
   stub (Edit-in-place on the rule-0 skeleton, recording, from
   `logs/daily/filings-<date>/filed.jsonl`, each terminal
   item's filed `#id`, and any not-yet-terminal slug or spawn-deferred id as
   "dispatch pending"), commit, push, enqueue — only THEN may a turn end with
   residual background work (e.g. watching a filing driver) still in flight.
   Land the `## Applied workflow improvements` content together with — or
   AFTER — the other machine-drafted sections in that ONE enrichment pass:
   the healthcheck husk arm reads a non-empty Applied section as "day done",
   so filling Applied first and dying before the rest would leave an
   incomplete file that no longer alerts.
2. **Join load-bearing background work synchronously.** Anything the daily
   file's content depends on (transcript miners, a driver whose filed `#id`s
   you want to record) is collected by blocking on its TaskOutput in-turn —
   not by ending the turn and relying on a background-completion wake. The 3h
   cron ceiling is a BACKSTOP for slow stragglers, not the design; a healthy
   run never hits it.

### Backfill a missed day (date argument)

`/daily` accepts an optional ISO date argument — `/daily 2026-07-01` —
making the run a BACKFILL for that date. Trigger: the #711 heartbeat
(`scripts/cron_daily_healthcheck.sh`) alerts that a nightly file never
landed; its alert names the exact command:

    cd ~/explore-persona-space && CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=10800000 \
      /home/thomasjiralerspong/.local/bin/claude -p '/daily <missed-date>'

Semantics — everywhere this file says "today", read the TARGET DATE:

- Output `logs/daily/<target-date>.md`, frontmatter `date: <target-date>`.
- Transcript inputs: keep transcripts with ≥1 in-file message `timestamp`
  dated the target date (UTC); the mtime pre-filter widens to "modified on or
  after <target-date>" (later processes bump old mtimes — same hazard as the
  2026-06-22 incident, same in-file-timestamp cross-check).
- Promoted-results filter: `promoted_at` on the target date.
- The existing-file rules under "Output" apply unchanged (a date whose file
  already has a non-empty `## Applied workflow improvements` → refuse).
- Route-1/2/3 fixes still fire, with one extra check: before applying or
  filing, confirm against the CURRENT tree that the problem is not already
  fixed (a backfill mines stale transcripts); skip with a note if it is.
  The Retraction re-check (see "Triage each problem") applies to freshly
  composed backfill filings identically — a backfill mines even staler
  snapshots, so the source-trail re-read is at least as load-bearing there.
  Route-2 dedup on `(target_file, fingerprint)` catches same-bug re-raises
  (e.g. from the partially-run 2026-07-03 night).
- If the missed night left a partial `logs/daily/filings-<date>/` (manifest +
  filed.jsonl), RESUME it: re-invoke `scripts/daily_drive_filings.py` on that
  dir (filed/deduped slugs are skipped; a trailing "attempting" row is
  recovered by title match, never blindly re-filed) instead of regenerating
  bodies from scratch or hunting /tmp.
- Telegram digest line reads `EPS daily <target-date> (backfill): ...`.
- Do not start a backfill within 60 min of the 23:27 nightly; two concurrent
  /daily processes double-mine and race the shared repo root.

No argument → today, exactly as before.

### Other rules

Do not promote clean results, create experiments, or move statuses unless the
user explicitly asks for that mutation in the current session. If asked to
mutate, use `scripts/task.py` so the change goes through the canonical
API and leaves a workflow event.
