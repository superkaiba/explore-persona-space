# Task Workflow — Lightweight Replacement for Sagan

**Status:** AWAITING APPROVAL — do not scaffold until user signs off.
**Worktree branch:** `task-workflow`
**Date:** 2026-05-14 (revised after user feedback: no kanban, markdown clean-results, /issue N writes to local files only)

---

## What it is

the dashboard is a thin, repo-native workflow system that replaces the Sagan dashboard. It keeps the existing experimental flow (clarify → adversarial-planner → approval → run → analyze → review → clean-result → promote) but moves state from the Sagan Postgres backend into plain repo files. The web app is a **read-mostly** Next.js viewer that renders task pages, daily/weekly updates, and surfaced literature, accepts mentor comments (Google OAuth-gated), and triggers headless Claude on this VM via a Cloudflare tunnel for instant Q&A. **No kanban, no drag-and-drop, no agent triggers from the web** — task dispatch is PM-driven from the local Happy session, same as today.

Per-task work runs in Happy sessions, supervised by a small daemon that restarts crashed sessions so any task always makes forward progress unless it has hit a real gate.

---

## Naming

- System: **the dashboard**
- Web app: `dashboard/` (Next.js)
- CLI: `scripts/task.py` (drop-in replacement for `sagan_state.py`)
- Supervisor daemon: `scripts/supervisor.py`
- Tunnel daemon: `scripts/tunnel.py`
- Literature cron: `scripts/literature_cron.py`
- Update generators: `scripts/daily_update.py`, `scripts/weekly_update.py`
- Sagan importer (one-shot): `scripts/sagan_import.py`
- Slash skill: keep `/issue <N>` (muscle memory; `N` is now the task number, continuing from the highest Sagan number)

---

## Directory layout (new)

```
tasks/
  INDEX.json                    # {"highest_id": 412, "tasks": [{id, status, kind, title}, ...]}
  369/
    body.md                     # canonical body (clean-result HTML once promoted)
    meta.json                   # {status, kind, title, created_at, tags[], pod_id, parent_id, has_clean_result}
    events.jsonl                # append-only progress log (same shape as today's epm:* markers)
    comments.jsonl              # mentor comments + Claude answers (flat JSON, append-only)
    original-body.md            # snapshot saved by analyzer before clean-result promotion
    artifacts/                  # figures, html-artifacts, plan caches the agents drop here
      figure-hero.png
      ...

updates/
  daily/2026-05-14.md           # + .html sibling for direct web render
  weekly/2026-W20.md            # + .html
  literature/
    INDEX.jsonl                 # all surfaced papers, dedup by arxiv id
    2026-05-14.md               # daily surfacing batch with summaries + relevance tags

dashboard/                          # Next.js 15 App Router app
  app/
    page.tsx                    # / — kanban by status
    tasks/[id]/page.tsx         # /tasks/369 — body + timeline + comments
    updates/page.tsx            # /updates — daily/weekly index
    updates/[slug]/page.tsx
    literature/page.tsx
    api/comments/[id]/route.ts  # POST — append comment, trigger tunnel
    api/revalidate/route.ts     # POST — local agents revalidate pages
    api/auth/[...nextauth]/     # Google OAuth
  lib/
    github.ts                   # Octokit wrapper (read repo + commit comments)
    allowlist.ts                # Google email allowlist check
  package.json
  vercel.json

scripts/
  task.py                       # CLI: same subcommands as sagan_state.py (view, set-status, post-marker, set-body, set-title, latest-marker, list-by-status, promote, add-tag)
  task_session.py               # wrapper that runs /issue N inside a Happy session
  supervisor.py           # systemd-style daemon: keeps active tasks running
  tunnel.py               # FastAPI server behind Cloudflare tunnel; receives mentor-comment webhooks, runs headless Claude
  literature_cron.py           # daily cron: arxiv MCP + relevance ranking → updates/literature/
  daily_update.py                # daily cron: PM-agent generates updates/daily/
  weekly_update.py               # weekly cron: PM-agent generates updates/weekly/
  sagan_import.py               # one-shot: pull all Sagan experiments into tasks/

.github/workflows/
  the dashboard-deploy.yml              # builds dashboard/ on push to main; Vercel handles deploy
```

`scripts/sagan_state.py` and `scripts/pod.py` and friends stay in tree but become legacy after import. `pod.py` is unchanged; it's the canonical pod CLI.

---

## File format details

### `tasks/NNN/meta.json`

```json
{
  "id": 413,
  "kind": "experiment",
  "title": "FR↔IT symmetry train+eval",
  "status": "running",
  "created_at": "2026-05-14T10:00:00Z",
  "updated_at": "2026-05-14T13:22:00Z",
  "tags": ["language-inversion", "qwen-7b"],
  "parent_id": 333,
  "pod_name": "epm-issue-413",
  "has_clean_result": false,
  "branch": "issue-413"
}
```

### `tasks/NNN/events.jsonl`

Identical shape to current Sagan `workflow_events` so existing agent code that constructs marker payloads doesn't need restructuring:

```jsonl
{"ts":"2026-05-14T10:00:00Z","kind":"epm:created","version":1,"by":"user"}
{"ts":"2026-05-14T10:02:00Z","kind":"epm:clarify-questions","version":1,"by":"clarifier","note":"..."}
{"ts":"2026-05-14T10:05:00Z","kind":"epm:clarify-answers","version":1,"by":"user","note":"..."}
{"ts":"2026-05-14T10:10:00Z","kind":"epm:plan","version":1,"by":"planner","note":"...","artifacts":[".claude/plans/issue-413.md"]}
{"ts":"2026-05-14T10:15:00Z","kind":"epm:plan-approved","version":1,"by":"user"}
{"ts":"2026-05-14T10:20:00Z","kind":"epm:run-launched","version":1,"by":"experimenter","note":"..."}
...
```

### `tasks/NNN/comments.jsonl`

```jsonl
{"id":"c001","ts":"2026-05-14T14:00:00Z","author":"mentor@example.com","kind":"question","body":"Why did you pick Method A over B?","resolved":false}
{"id":"c002","ts":"2026-05-14T14:00:30Z","author":"claude","kind":"answer","in_reply_to":"c001","body":"...","sources":["events#42","body#L120"]}
{"id":"c003","ts":"2026-05-14T15:00:00Z","author":"mentor@example.com","kind":"followup-proposal","body":"Try this at 70B","spawned_task":414}
```

`kind: followup-proposal` automatically creates a new `proposed` task when the mentor submits it (web app has a "this is a followup" radio).

### Plans (browser-viewable, no terminal reading required)

Plans live inside the task folder, versioned, web-rendered. Every time the user has to read a plan, the planner agent prints **only a URL** — never the plan body — to the terminal.

```
tasks/<status>/<id>/
  plans/
    v1.md            # first round from the planner
    v2.md            # revision after critic
    v3.md            # final approved version
  plan.md            # symlink (or copy) to highest-versioned plan
```

**Web URL:** `eps.superkaiba.com/tasks/<id>/plan` (always the latest) and `eps.superkaiba.com/tasks/<id>/plans/v2` (a specific version, with a version-picker dropdown).

**Planner agent output rule:** when a new plan version is written, the agent prints exactly one line:

```
Plan v3 written → https://eps.superkaiba.com/tasks/413/plan
```

No plan body in the terminal. Same convention for `clean-result-drafting`, `interpretation-drafting`, and any other long-form markdown artifact that has historically been dumped to terminal.

### General artifact-viewing escape hatch

Any markdown / HTML / PNG file under `tasks/<status>/<id>/artifacts/` or `tasks/<status>/<id>/plans/` is browsable at `eps.superkaiba.com/tasks/<id>/artifacts/<relative-path>`. This means *anything an agent writes into the task folder is one URL away* — including critique markers, interpretation drafts, figure source data, etc.

---

## State machine

Identical to today's Sagan enum (no churn for the agents): `proposed → planning → plan_pending → approved → running → verifying → interpreting → reviewing → awaiting_promotion → completed`, plus `blocked`, `archived`.

Authoritative source: `tasks/NNN/meta.json#status`. Mutated only via `scripts/task.py set-status`.

---

## Per-task session "force-continue" mechanism

Two-layer:

**Inner — the `/issue` skill is already idempotent + resumable.** It reads `events.jsonl` to figure out where to resume. Auto-continuation policy and gate enumeration stay as documented in CLAUDE.md / workflow.yaml — already strong.

**Outer — `scripts/supervisor.py` (daemon).** Runs every ~5 min:

1. Reads `tasks/INDEX.json` for tasks with non-terminal `status` (anything except `awaiting_promotion`, `completed`, `blocked`, `archived`).
2. For each, asks the Happy daemon "is there an active session for task N?" via the existing `spawn_session.py` API.
3. If no active session, spawns one with `/issue N --resume`.
4. Logs to `~/.dashboard/supervisor.log`.

This handles: crashes, OOM kills, user killed Claude, VM reboot. The skill itself never has to ask "should I continue?" — the supervisor enforces forward progress until the task explicitly hits a gate or terminal state.

**Stop sentinel:** Claude has no special exit token. The supervisor reads `meta.json#status`. Terminal/gate statuses tell it not to respawn. The `blocked` status is the explicit "stop, do not auto-resume" signal — any agent that hits a halt criterion sets `status='blocked'` and exits, and the supervisor leaves it alone.

---

## Web app (the dashboard / Next.js)

**Stack:** Next.js 15 App Router, Tailwind, NextAuth with Google provider, Octokit, react-markdown (with remark-gfm + rehype-highlight for code), deployed on Vercel.

**Data flow:**
- Reads from the repo via raw.githubusercontent.com for public files (body.md, events.jsonl, comments.jsonl). Uses GitHub API with a read-only PAT for higher rate limit.
- ISR (Incremental Static Regeneration) with on-demand revalidation: local agents POST to `/api/revalidate` after committing changes so pages refresh within seconds.
- Comments: web app commits to `tasks/NNN/comments.jsonl` via Octokit with a write-scoped PAT (`COMMENTS_PAT`). One commit per comment — noisy but fully auditable.

**Pages:**
- `/` — task list, grouped by status (To do / Planning / Running / Awaiting promotion / Done / Blocked / Archived as collapsible sections, defaults: To do + Running + Awaiting promotion expanded, rest collapsed). Each task shows `#NNN`, title, status badge, last-updated timestamp. Searchable + filterable by tag. No drag-and-drop, no agent triggers.
- `/tasks/[id]` — title + status badge + **markdown body** (rendered via react-markdown with GFM + syntax highlighting + image support) + collapsible event timeline grouped by phase + comments thread at the bottom. Mobile-responsive. Followup-proposal button visible only on clean-result tasks (`meta.has_clean_result=true`).
- `/updates` — list of daily/weekly updates with date filters.
- `/updates/[slug]` — full update (markdown rendered).
- `/literature` — surfaced-paper list with dates, relevance tags (which task IDs), abstract + intuitive summary, link to arxiv.

**Auth:** Google OAuth via NextAuth. Email allowlist enforced server-side in `lib/allowlist.ts` reading `DASHBOARD_ALLOWED_EMAILS` env var (`thomasjiralerspong@gmail.com,danmossing@anthropic.com`). Non-allowlisted users can read everything; they cannot comment or propose followups.

**Mobile-friendly (explicit requirements, all pages):**
- Touch targets ≥44 px on small screens; comment submit button is a full-width primary CTA on mobile.
- Body uses `prose-sm` on `<640px`, `prose-lg` on `≥1024px` for readable line-length.
- No horizontal page scroll. Code blocks scroll horizontally *within* their own container (`overflow-x-auto`).
- Images responsive: `max-width: 100%`; figures tap-to-zoom (lightbox) on mobile.
- Tables degrade to vertical card-stacks below 640 px.
- Comment textarea autosizes; `inputmode="text"`; no zoom-jump on focus.
- Sticky bottom action bar on `/tasks/[id]` for the comment composer when scrolling.
- Hamburger nav on small screens; full top-nav on `≥768px`.
- The home page's status-section accordions default to collapsed on mobile (open: To do + Running + Awaiting promotion only).
- Plan and clean-result pages tested at 375×667 (iPhone SE), 414×896 (iPhone 12 Pro), 360×800 (Pixel) at minimum.
- All text remains legible at 200% browser zoom.

**Mentor followup proposals:** on any clean-result task page, a "Propose followup" button (visible to allowlisted users) opens a modal with title + description. On submit, a Vercel function:
1. Reads `tasks/INDEX.json`, increments `highest_id` → new ID `M`.
2. Commits a new `tasks/M/meta.json` (status=`proposed`, kind=`experiment`, parent_id=current task), `tasks/M/body.md` (mentor's description), empty `events.jsonl`, empty `comments.jsonl`.
3. Updates `tasks/INDEX.json`.
The new task is now visible in `/` for the PM session to triage. No auto-dispatch.

---

## Mentor comment → headless Claude (Cloudflare tunnel)

```
[mentor browser, mobile or desktop]
    ↓ Google-OAuth-gated POST /api/comments/413
[Vercel Edge function]
    ↓ 1. Validate auth + allowlist
    ↓ 2. Append to tasks/413/comments.jsonl via Octokit (commits to main)
    ↓ 3. POST https://the dashboard.<your-domain>/comment-webhook
         body: {task_id: 413, comment_id: "c047"}
[Cloudflare tunnel] → local VM port 7720
[tunnel.py (FastAPI on localhost:7720)]
    ↓ 1. git pull
    ↓ 2. Load tasks/413/body.md + recent events + comment
    ↓ 3. Spawn: claude -p "<prompt template>" (model: opus-4-7)
    ↓ 4. Append response to comments.jsonl as kind:"answer", in_reply_to:c047
    ↓ 5. git commit + push
    ↓ 6. POST https://the dashboard.<vercel-url>/api/revalidate?path=/tasks/413
[Vercel] revalidates page; mentor sees response on next refresh / live via SWR
```

Latency: aim for <30s from comment-submit to answer-visible. Tunnel is instant; the bottleneck is the Claude call (~10-20s for a short Q+A).

**Cloudflare tunnel setup:** `cloudflared tunnel create the dashboard` + DNS routing to a stable hostname if you have a domain; otherwise `cloudflared tunnel --url localhost:7720` gives a random `*.trycloudflare.com` URL that's stable per tunnel-run. The tunnel daemon runs under systemd (or just `nohup`).

---

## Sagan import (one-shot)

`scripts/sagan_import.py`:

1. Pulls every experiment from Sagan API (`GET /api/experiments?limit=10000`).
2. For each experiment `e`:
   - Writes `tasks/<e.number>/meta.json` from `e.status`, `e.title`, `e.tags`, `e.created_at`, etc.
   - Writes `tasks/<e.number>/body.md` from `e.body`.
   - Writes `tasks/<e.number>/events.jsonl` by fetching `GET /api/experiments/<id>/events` and translating each `workflow_event` row into an events.jsonl line (shape is already compatible).
   - Writes `tasks/<e.number>/comments.jsonl` empty (or pulls from any user-comments table if it exists).
   - Copies any uploaded artifacts via the artifact URLs in the body.
3. Writes `tasks/INDEX.json` with `highest_id = max(e.number)`.
4. Commits in batches of ~100 experiments per commit to keep diffs manageable.

After this runs, `scripts/sagan_state.py` is never invoked again. The Sagan dashboard URL stays accessible for historical comparison if needed.

---

## Literature surfacing (daily cron)

`scripts/literature_cron.py`, runs daily at 06:30 local:

1. Reads recent clean-result bodies (`tasks/*/body.md` with `meta.has_clean_result=true`, updated in last 14 days) + project keyword list (TODO: maintain `dashboard/keywords.txt`).
2. Calls the arxiv MCP server (`mcp__arxiv__search_papers`) with daily cs.AI / cs.CL / cs.LG cutoff = yesterday.
3. For each candidate paper, computes relevance:
   - Embedding similarity (use `sentence-transformers` locally) between paper abstract and recent clean-result content.
   - Top-K candidates by relevance, K=10.
4. For each top paper, spawns a tiny Claude call to produce a 3-sentence "intuitive summary" + which task IDs the paper is relevant to.
5. Writes `updates/literature/YYYY-MM-DD.md` with the daily batch.
6. Appends each paper as a JSON line to `updates/literature/INDEX.jsonl` (dedup by arxiv id; skip if already surfaced).
7. Commits.

The web app's `/literature` page reads `INDEX.jsonl` and groups by date.

---

## Daily / weekly updates

`scripts/daily_update.py` (07:00 local):
- Spawns a headless `research-pm` agent with prompt: "Read tasks/*/meta.json for everything updated in the last 24h. Read tasks/*/events.jsonl entries newer than 24h. Read updates/literature/$(date).md if it exists. Write updates/daily/$(date).md following the existing daily-update template: yesterday's progress / today's plan / blockers / surfaced literature."
- Generates both `.md` and `.html` (the latter via a tiny pandoc call or markdown-it).
- Commits.

`scripts/weekly_update.py` (Monday 08:00 local):
- Same shape but 7-day window + project-summary section + next-week outlook.

Mentor visits `the dashboard.<vercel-url>/updates` to read them.

---

## Pod lifecycle

Zero change. `scripts/pod.py` stays the canonical pod CLI. The /issue skill calls it the same way it does today (`pod.py provision --issue 413 --intent lora-7b` etc.). The only difference: `N` is now read from `tasks/NNN/meta.json` instead of Sagan.

---

## Migration path

Phase 1 (scaffold, this PR):
- Create worktree `task-workflow`.
- Build everything above.
- Sagan stays untouched.
- Run `sagan_import.py` to populate `tasks/`.
- Deploy `dashboard/` to Vercel preview.

Phase 2 (validate):
- Run one new experiment end-to-end through the dashboard.
- Mentor tests comment + Claude-answer round trip.
- Iterate on web UI.

Phase 3 (cutover):
- Merge `task-workflow` → main.
- Stop pointing PM session at Sagan; update `/pm` skill to read from `tasks/`.
- Sagan becomes read-only legacy.

---

## What changes in existing files

Surgical, not sweeping:
- `scripts/task.py` is a drop-in API-compatible replacement for `scripts/sagan_state.py`. Subcommand surface identical, but all operations are local-file mutations (atomic write + git commit + optional revalidate webhook). No HTTP, no Postgres, no API token.
- All agents (`.claude/agents/*.md`) that reference `sagan_state.py` get a one-line substitution: `sagan_state.py` → `task.py`. Logic unchanged.
- `.claude/skills/issue/SKILL.md` gets the same substitution + a section on the supervisor and the new `tasks/` directory. Workflow stays identical (clarify → adversarial-planner → approval → run → analyze → review → clean-result → user-promotes). Only the *substrate* changes (local files vs. Sagan API).
- `.claude/skills/pm/SKILL.md` reads from `tasks/INDEX.json` + `tasks/*/meta.json` instead of Sagan API.
- `CLAUDE.md` "Sagan State API" section replaced with "the dashboard State API"; "Sagan is canonical workflow state" replaced with "the dashboard is canonical workflow state".
- **Clean-result format → plain markdown.** `tasks/N/body.md` is markdown throughout the task's life (not just before promotion). The analyzer writes a markdown clean-result body (TL;DR section + summary + reproducibility section + artifacts). No more class-scoped `<style>` blocks, no `<details>`/`<figure>` HTML, no Sagan-card inline CSS.
- `verify_sagan_card.py` → `verify_task_body.py` (much simpler — see "Clean-result markdown spec" below). Old `verify_sagan_card.py` stays in tree for grandfathered HTML bodies but is deprecated.
- `~/sagan/docs/clean-result-guidelines.md` no longer authoritative for new bodies; new spec in `dashboard/docs/clean-result-spec.md`.

Nothing else touched. Pod CLI, training scripts, eval pipeline, experiment-implementer agent: zero churn.

### Clean-result markdown spec (new)

A clean-result body is a single `.md` file. Required sections in order:

```markdown
# <one-sentence claim> (LOW | MODERATE | HIGH confidence)

## TL;DR
- **Motivation:** ...
- **What I ran:** ...
- **Results:** ... (link to figure)
- **Next steps:** ...

## Figure
![alt](relative/or/hf/url/figure.png)
*Caption: ≥10 words describing what's plotted.*

## Details
Free-form markdown: definitions, training setup, eval rationale, sample completions (inline ``` blocks), statistical-test rationale, confidence sentence, parameters table.

Confidence: LOW | MODERATE | HIGH — <one sentence naming the binding constraint or surviving evidence>.

## Reproducibility
**Artifacts:**
- Model: [hf-hub-url](...)
- Dataset: [hf-hub-url](...)
- Raw completions: [hf-hub-url](...)
- WandB run: [wandb-url](...)
- Eval JSON: `eval_results/issue_N/run_result.json` @ commit `<sha>`

**Compute:** wall time, GPU type, pod name.

**Code:** entry script, git commit SHA, Hydra config path, copy-pasteable reproduce command.
```

`verify_task_body.py` checks:
1. Title line ends with `(LOW|MODERATE|HIGH confidence)`.
2. Four required H2 sections present: `TL;DR`, `Figure`, `Details`, `Reproducibility`.
3. TL;DR bullets contain `Motivation:`, `What I ran:`, `Results:`, `Next steps:`.
4. Reproducibility section: all URLs are permanent (HF Hub `/tree/<ref>`, WandB `/runs/<id>`, GitHub `/blob/<sha>` — no `TBD`, `{{`, `default`, `see config`).
5. Confidence sentence somewhere in Details, matching the title's confidence level.
6. Figure has a caption line that's ≥10 words.

That's it. Six checks instead of eleven, all mechanical, no HTML parsing.

---

## Open items I'm picking defaults on (you can override)

- **Stack for dashboard/:** Next.js 15 + Tailwind + NextAuth + Octokit. (Alternative: SvelteKit. Next.js is the project's existing Vercel default per CLAUDE.md "frontend-design" skill.)
- **Embedding model for literature relevance:** `BAAI/bge-large-en-v1.5` via sentence-transformers. Local on this VM.
- **Cron mechanism:** plain `crontab -e` on this VM. (Alternative: systemd timers.)
- **Supervisor cadence:** 5 minutes.
- **Domain:** `superkaiba.com` on Cloudflare (already in use for Sagan at `sagan.superkaiba.com`).
  - Web app custom domain: `eps.superkaiba.com` (Vercel).
  - Tunnel hostname: `eps-tunnel.superkaiba.com` (named cloudflared tunnel).
  - One-time prerequisite: `cloudflared tunnel login` from this VM (opens browser, you authorize the `superkaiba.com` zone). Generates `~/.cloudflared/cert.pem`. After that, everything else is scripted: `cloudflared tunnel create the dashboard` → `cloudflared tunnel route dns the dashboard eps-tunnel.superkaiba.com` → systemd unit running `cloudflared tunnel run the dashboard`.
- **PAT scopes:** `COMMENTS_PAT` = `contents:write` on this single repo. `READ_PAT` = `contents:read` for higher rate-limit reads.
- **Allowlist:** `DASHBOARD_ALLOWED_EMAILS` env var on Vercel = `thomasjiralerspong@gmail.com,danmossing@anthropic.com`.
- **Vercel project name:** `eps-dashboard`.

---

## What I will NOT do without further sign-off

- Touch the running `scripts/sagan_state.py` or any in-flight experiments before the importer runs.
- Delete `archive/research_log/` or any existing artifacts.
- Touch `scripts/pod.py`, `experiment-implementer`, `experimenter`, `analyzer`, `clean-result-critic` core logic — only their `sagan_state.py → task.py` references.
- Change the auto-continuation policy or gate enumeration.
- Migrate the `verify_sagan_card.py` Sagan-card HTML spec to something else — same shape, just renamed.
- Touch the `paper-plots` skill.

---

## Prerequisites you handle once (before step 4)

1. Run `cloudflared tunnel login` from this VM → browser opens → authorize `superkaiba.com`. (~30s)
2. Create a Vercel project named `eps-dashboard` linked to this repo, with root directory `dashboard/`. I can do this via `vercel` CLI; you just need to be logged in.
3. Generate two GitHub PATs: `COMMENTS_PAT` (contents:write, this repo only) and `READ_PAT` (contents:read). Add them to Vercel env vars + `.env` on this VM.
4. Create a Google OAuth client (Vercel will give the redirect URI to plug in). I'll write a checklist for this.

I'll stage these as a pre-flight checklist; nothing else blocks on them until step 4 (Vercel deploy) and step 7 (tunnel).

## Execution order once approved

1. Create worktree `task-workflow`.
2. Write `scripts/task.py` (CLI + Python module). Unit tests.
3. Write `scripts/sagan_import.py` + run it once. Commit imported tasks.
4. Scaffold `dashboard/` Next.js app. Local dev + first deploy to Vercel preview.
5. Wire up Google OAuth + allowlist.
6. Wire up comments POST → Octokit commit.
7. Write `scripts/tunnel.py` + cloudflared setup. End-to-end mentor-Q&A test.
8. Write `scripts/supervisor.py` + install crontab/systemd.
9. Write `scripts/literature_cron.py` (arxiv MCP + embedding relevance).
10. Write `scripts/daily_update.py` + `scripts/weekly_update.py`.
11. Update all agent specs (`sagan_state.py → task.py` substitution).
12. Update `.claude/skills/issue/SKILL.md` + `.claude/skills/pm/SKILL.md`.
13. Update `CLAUDE.md`.
14. End-to-end test on a single new task.
15. Open PR for review (not merging until you say so).

Each step lands as its own commit on `task-workflow`. You can inspect / pause / redirect at any boundary.
