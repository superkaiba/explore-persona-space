# Task Workflow Migration Plan

**Status:** AWAITING APPROVAL — no scaffolding until user signs off.
**Worktree branch:** `task-workflow`
**Date:** 2026-05-14
**Supersedes:** `.claude/plans/task-workflow-design.md` (kept for diff history; this doc is canonical).

---

## 1. Goal

Replace Sagan as the workflow substrate for EPS with a thin, repo-native system. Keep the existing experimental flow (clarify → plan → approve → run → analyze → review → clean-result → promote) and every agent that runs inside it. Move durable state from Sagan's Postgres into plain files in this repo. Build a small Next.js viewer at `eps.superkaiba.com` so I and my mentor can browse the system on mobile and desktop.

## 2. Non-goals

- No kanban, no drag-and-drop, no buttons that mutate state from the web. The web is for *viewing* and *commenting* only.
- No new pod orchestrator. `scripts/pod.py` stays the canonical pod CLI; ephemeral `epm-issue-N` pods continue exactly as today.
- No new agents. Existing agents (planner, critic, experimenter, analyzer, clean-result-critic, etc.) keep working — only their state-store calls change.
- No Sagan-style HTML clean-result format. Plain markdown only.
- No server-side Claude Code runner. All experimental Claude work runs in local Happy sessions on this VM.
- No native mobile app. The Next.js app is responsive; we use the browser.
- No mass deprecation of existing scripts. `sagan_state.py`, `verify_sagan_card.py`, `sagan_progress`, etc. stay in tree as legacy after migration; new code doesn't call them.

## 3. Architecture overview

```
                                  Local VM (this machine)
   ┌──────────────────────────────────────────────────────────────────────┐
   │                                                                      │
   │  Happy sessions ── PM session                                        │
   │       │              │                                               │
   │       │              └─ spawn_session.py spawn-issue --issue N       │
   │       ▼                                                              │
   │  /issue N skill ──► task.py (atomic file mutations, git commits)     │
   │       │                  │                                           │
   │       │                  ├─ tasks/<status>/<id>/{body.md, events.jsonl,
   │       │                  │                       comments.jsonl,
   │       │                  │                       plans/v{N}.md, artifacts/}
   │       │                  └─ tasks/REGISTRY.json (ID → current path)  │
   │       │                                                              │
   │       ├─ pod.py provision/terminate (unchanged)                      │
   │       └─ revalidation POST to web → ISR refresh                      │
   │                                                                      │
   │  supervisor.py daemon (5 min) ── respawns crashed Happy sessions     │
   │  tunnel.py FastAPI server ◄────── Cloudflare tunnel (eps-tunnel.…)   │
   │  literature_cron.py (06:30)                                          │
   │  daily_update.py (07:00)                                             │
   │  weekly_update.py (Mon 08:00)                                        │
   └──────────────────────────────────────────────────────────────────────┘
                            ▲                            ▲
                            │ git push                   │ Cloudflare Tunnel
                            ▼                            │
                ┌─────────── GitHub repo ──────────┐     │
                │  source of truth for all state   │     │
                └──────────────────────────────────┘     │
                            ▲                            │
                            │ raw.githubusercontent +    │
                            │ Octokit (write comments)   │
                            ▼                            │
                ┌──── Next.js app on Vercel ───────┐     │
                │  eps.superkaiba.com              │     │
                │  - read-mostly viewer            │─────┘
                │  - Google OAuth + allowlist      │
                │  - comments (only writable thing)│
                └──────────────────────────────────┘
                            ▲
                            │ HTTPS
                            ▼
                       [me + mentor]
                       (mobile + desktop)
```

**Single writer per file.** `task.py` on the VM is the only thing that touches `body.md`, the `plans/` tree, `events.jsonl`, and the folder location. The web app only appends to `comments.jsonl`. No two-writer races.

## 4. Storage model

### 4.1 Folder-as-status

Status is encoded by the parent folder name. Status enum (unchanged from today):

```
proposed / planning / plan_pending / approved / running / verifying /
interpreting / reviewing / awaiting_promotion / completed / blocked / archived
```

```
tasks/
  proposed/
    413/
  planning/
    412/
  running/
    411/
  awaiting_promotion/
    410/
  completed/
    409/
  blocked/
  archived/
  REGISTRY.json
```

Status change = `git mv tasks/<old>/<id> tasks/<new>/<id>` + commit. Single atomic operation. No `meta.json#status` field that could disagree with the folder.

### 4.2 Per-task folder contents

```
tasks/<status>/<id>/
  body.md           # markdown w/ YAML frontmatter (see 4.3)
  events.jsonl      # append-only progress log
  comments.jsonl    # append-only mentor comments + Claude replies
  plans/            # versioned plans from /adversarial-planner
    v1.md
    v2.md
    plan.md         # symlink → latest version
  artifacts/        # figures (PNG), html-artifacts, plan caches, anything else
  original-body.md  # snapshot taken when analyzer promotes the clean-result (rollback)
```

Every file is markdown / JSONL / PNG. No HTML, no binary blobs (the eval results JSONs live in `eval_results/` as today).

### 4.3 `body.md` frontmatter

```markdown
---
title: FR↔IT symmetry train+eval
kind: experiment           # experiment | infra | analysis | survey
tags: [language-inversion, qwen-7b]
parent_id: 333
created_at: 2026-05-14T10:00:00Z
pod_name: epm-issue-413
happy_session_id: hs_abc123def
has_clean_result: false
---

# FR↔IT symmetry train+eval

(content: proposal → revised proposal → plan link → eventually replaced with clean-result markdown)
```

Frontmatter parsed via `python-frontmatter` (Python) and `gray-matter` (JS). Anything in frontmatter is queryable across tasks; anything in the body is content.

### 4.4 `events.jsonl`

Append-only progress log. Same `epm:*` marker shape as today's Sagan `workflow_events`:

```jsonl
{"ts":"2026-05-14T10:00:00Z","kind":"epm:created","version":1,"by":"user"}
{"ts":"2026-05-14T10:02:00Z","kind":"epm:clarify-questions","version":1,"by":"clarifier","note":"..."}
{"ts":"2026-05-14T10:05:00Z","kind":"epm:clarify-answers","version":1,"by":"user","note":"..."}
{"ts":"2026-05-14T10:10:00Z","kind":"epm:plan","version":1,"by":"planner","note":"Plan v1 written","artifacts":["plans/v1.md"]}
{"ts":"2026-05-14T10:15:00Z","kind":"epm:plan-approved","version":1,"by":"user","plan_version":3}
{"ts":"2026-05-14T10:20:00Z","kind":"epm:run-launched","version":1,"by":"experimenter","note":"..."}
{"ts":"2026-05-14T13:30:00Z","kind":"epm:run-finished","version":1,"by":"experimenter","note":"..."}
{"ts":"2026-05-14T13:35:00Z","kind":"epm:upload-verified","version":1,"by":"upload-verifier"}
{"ts":"2026-05-14T13:36:00Z","kind":"epm:pod-terminated","version":1,"by":"issue-skill"}
{"ts":"2026-05-14T14:00:00Z","kind":"epm:interpretation","version":1,"by":"analyzer"}
{"ts":"2026-05-14T14:30:00Z","kind":"epm:clean-result-drafted","version":1,"by":"analyzer"}
{"ts":"2026-05-14T15:00:00Z","kind":"epm:promoted","version":1,"by":"user","classification":"useful"}
```

`events.jsonl` is the **source of truth for resume**. `/issue N --resume` reads it from the bottom up to figure out exactly where to pick up.

### 4.5 `comments.jsonl`

```jsonl
{"id":"c001","ts":"2026-05-14T14:00:00Z","author":"danmossing@anthropic.com","kind":"question","body":"Why did you pick Method A?","resolved":false}
{"id":"c002","ts":"2026-05-14T14:00:30Z","author":"claude","kind":"answer","in_reply_to":"c001","body":"...","model":"claude-opus-4-7"}
{"id":"c003","ts":"2026-05-14T15:00:00Z","author":"danmossing@anthropic.com","kind":"followup-proposal","body":"Try this at 70B","spawned_task":414}
{"id":"c004","ts":"2026-05-14T16:00:00Z","author":"thomasjiralerspong@gmail.com","kind":"note","body":"Reminder to check the eval at step 1000"}
```

Comment kinds (only ones supported):

| `kind` | Mentor expresses... | Local handler does... |
|---|---|---|
| `question` | "Why X?" | Tunnel → `claude -p` with task body + events + comment → reply appended |
| `followup-proposal` | "Try Y" | Tunnel → `task.py new --parent N --title "..."` → new `tasks/proposed/<NEW>/` |
| `note` | Plain comment | Just stored, no action |

No `move-request` kind. Status moves go through the `/issue N` skill.

### 4.6 Plans

`tasks/<status>/<id>/plans/v{N}.md`. Each round of adversarial planning writes a new version. `plan.md` is a symlink to the latest. The planner agent's terminal output for every round is one line:

```
Plan v3 written → https://eps.superkaiba.com/tasks/413/plan
```

No plan body in the terminal — ever. Same rule for interpretation drafts, clean-result drafts, critique markers, anything long-form.

### 4.7 `tasks/REGISTRY.json`

```json
{
  "highest_id": 413,
  "tasks": {
    "413": {"path": "tasks/running/413", "title": "FR↔IT symmetry train+eval", "kind": "experiment"},
    "412": {"path": "tasks/awaiting_promotion/412", "title": "...", "kind": "experiment"},
    ...
  }
}
```

Maintained automatically by `task.py` on every status change or new-task creation. Web URLs `/tasks/<id>` resolve via this map; bookmarks survive status changes.

## 5. CLI surface: `scripts/task.py`

Drop-in API-compatible replacement for `scripts/sagan_state.py`. Same subcommands, same flags, same exit codes. All operations are local-file mutations: `flock` on `~/.task-workflow/lock` → read → mutate → atomic-rename → `git add` → `git commit` → optional `git push` → optional revalidation POST to Vercel.

```bash
task.py view <N>                          # print frontmatter + recent events
task.py new --kind experiment --title "..." [--parent N]
task.py set-status <N> <status>           # git mv to new folder
task.py post-event <N> <kind> [--note ... ] [--artifacts ...]
task.py set-body <N> <path-to-md>         # replace body.md (snapshot old to original-body.md)
task.py set-title <N> "..."               # frontmatter update
task.py add-tag <N> <tag>
task.py list-by-status <status>
task.py latest-event <N>                  # "where do I resume" query
task.py promote <N> useful|not-useful     # user-only; moves to completed + records classification
task.py new-plan-version <N> <path-to-md> # appends plans/v{next}.md
task.py find <N>                          # path of folder containing task N (via REGISTRY.json)
```

Importable as Python module: `from explore_persona_space.task_workflow import view, post_event, ...`. Tests live at `tests/test_task_py.py`; mock filesystem in `tmp_path`.

**Body-size cap:** event `note` payloads cap at 50,000 chars (same as Sagan API today). On oversize, post `epm:failure v1` event with `failure_class: infra, reason: note_oversize` and exit non-zero. Callers handle.

## 6. `/issue N` skill — surgical port

The skill stays *behaviorally identical*. Only the substrate changes.

### 6.1 What changes
- All `python scripts/sagan_state.py X` → `python scripts/task.py X`. Subcommand surface is identical; flags map 1:1.
- "Read Sagan experiment row" → "read `tasks/<status>/<id>/body.md` frontmatter + events.jsonl".
- "Post `epm:*` marker via API" → "`task.py post-event`" (writes to events.jsonl).
- "Sagan dashboard URL" → "`eps.superkaiba.com/tasks/<id>`".
- The `awaiting_promotion` park-and-wait gate behaves the same; the user runs `task.py promote <N> useful|not-useful` to advance.

### 6.2 What doesn't change
- The 10 steps, the gate enumeration, the auto-continuation policy, the subagent halt criteria, the Codex-ensemble review at 4 sites, the consistency-checker, the upload-verifier, the analyzer, the clean-result-critic. All identical.
- `pod.py` lifecycle (provision → run → upload → auto-terminate).
- The implementer / experiment-implementer / experimenter split.
- `verify_task_body.py` runs at the same gate the old `verify_sagan_card.py` did — just simpler checks (see §10).

### 6.3 What gets deleted / deprecated
- Nothing deleted. `sagan_state.py`, `sagan_progress`, `verify_sagan_card.py`, `audit_clean_results_body_discipline.py`, the entire `clean-result-guidelines.md` HTML spec all stay in tree as legacy. New code points to the new files.

## 7. Agent specs (`.claude/agents/*.md`)

Surgical edits to ~10 agent specs. Pattern: `sed -i 's/sagan_state\.py/task.py/g'` plus any specific Sagan-Postgres-vocabulary swaps. Affected agents:

- `planner` — writes to `plans/v{N}.md` instead of `.claude/plans/issue-N.md`; prints URL only.
- `critic` — same path change.
- `consistency-checker` — reads from local files.
- `experiment-implementer` — same.
- `experimenter` — already pod-focused, minimal changes.
- `implementer` — same.
- `upload-verifier` — same.
- `analyzer` — writes clean-result body to `tasks/<status>/<id>/body.md` (after snapshotting old to `original-body.md`); calls `task.py set-clean-result`.
- `interpretation-critic` — same.
- `clean-result-critic` — runs `verify_task_body.py` instead of `verify_sagan_card.py`.
- `reconciler` — substrate-agnostic; minimal change.
- `code-reviewer` — substrate-agnostic; minimal change.
- `follow-up-proposer` — writes proposed tasks via `task.py new`.
- `research-pm` — reads from `tasks/REGISTRY.json` + `tasks/<status>/*/`.
- Codex twins (`codex-*`) — substrate-agnostic; minimal change.

Same applies to `.claude/skills/issue/SKILL.md`, `.claude/skills/pm/SKILL.md`, `.claude/skills/promote-clean-result/SKILL.md`, `.claude/skills/auto-experiment-runner/SKILL.md`, `.claude/skills/daily/SKILL.md`, `.claude/skills/weekly/SKILL.md`, `.claude/skills/clean-results/SKILL.md`, `.claude/skills/independent-reviewer/SKILL.md`, and `.claude/workflow.yaml`.

## 8. Web dashboard (Next.js app at `eps.superkaiba.com`)

**Stack:** Next.js 15 App Router, Tailwind CSS, NextAuth (Google provider), Octokit, `react-markdown` + `remark-gfm` + `rehype-highlight` + `rehype-raw`, `gray-matter`, deployed on Vercel.

### 8.1 Pages

| Route | Purpose |
|---|---|
| `/` | Task list grouped by status (collapsible sections; mobile-default: To do / Running / Awaiting promotion expanded, rest collapsed). Each row: `#413` · title · status badge · last-updated · Happy session ID (clickable). |
| `/tasks/[id]` | Body (markdown rendered) + event timeline (grouped by phase: clarify / plan / run / analyze / review / clean-result) + cross-links to parent/children + comments thread at bottom. |
| `/tasks/[id]/plan` | Latest plan rendered. Version-picker dropdown for prior rounds. |
| `/tasks/[id]/plans/[v]` | Specific plan version. |
| `/tasks/[id]/artifacts/[...path]` | Any file under the task's `artifacts/` (renders md as markdown, png as image, html as iframe with sandboxing). |
| `/updates` | Daily / weekly index. |
| `/updates/[slug]` | Single update rendered. |
| `/literature` | Surfaced papers with date, relevance tags, summary, arxiv link. |

### 8.2 Data sources

- **Reads:** GitHub raw + GitHub REST API (Octokit) with `READ_PAT`. ISR (1-min default revalidate, plus on-demand revalidate triggered by `task.py`).
- **Writes:** only `tasks/<status>/<id>/comments.jsonl`, via Octokit with `COMMENTS_PAT`. One commit per comment.
- **Revalidation:** `task.py` POSTs to `eps.superkaiba.com/api/revalidate?path=/tasks/<N>` after every mutation. Revalidation token in env.

### 8.3 Auth

NextAuth with Google provider. Email allowlist enforced server-side: `DASHBOARD_ALLOWED_EMAILS=thomasjiralerspong@gmail.com,danmossing@anthropic.com`. Non-allowlisted users can read everything; cannot comment or propose followups.

### 8.4 Comments UI

Single composer at the bottom of `/tasks/[id]`. Dropdown for kind (Ask Claude / Propose followup / Note) + textarea. Submit POSTs to `/api/comments/[id]` → Octokit append → tunnel POST (for `question` and `followup-proposal` kinds). Optimistic UI: comment appears immediately with `pending` status, replaced by server-confirmed entry after the commit lands.

### 8.5 Mobile-friendly checklist

- Touch targets ≥44 px; comment submit is full-width primary CTA on small screens.
- `prose-sm` on `<640 px`, `prose-lg` on `≥1024 px`.
- No page-level horizontal scroll. Code blocks scroll within their own container.
- Images: `max-width: 100%`; tap-to-zoom lightbox on mobile.
- Tables degrade to vertical card-stacks below 640 px.
- Comment textarea autosizes; `inputmode="text"`; no zoom-jump on focus.
- Sticky bottom comment composer on `/tasks/[id]`.
- Hamburger nav on small screens; full top-nav on `≥768 px`.
- Status accordions default-collapsed on mobile (To do / Running / Awaiting promotion open).
- Tested at iPhone SE (375×667), iPhone 12 Pro (414×896), Pixel (360×800).
- Legible at 200% zoom.

## 9. Mentor comment → headless Claude (Cloudflare tunnel)

```
[mentor browser, allowlisted Google account]
    ↓ POST /api/comments/413  body={kind: "question", body: "..."}
[Vercel Edge function]
    ↓ 1. Validate session, check allowlist
    ↓ 2. Append to tasks/<status>/413/comments.jsonl via Octokit
    ↓ 3. POST https://eps-tunnel.superkaiba.com/comment-webhook
         {task_id: 413, comment_id: "c047"}
[Cloudflare tunnel] → local VM port 7720
[tunnel.py — FastAPI on localhost:7720]
    ↓ 1. git pull
    ↓ 2. Load body.md + recent events + the new comment
    ↓ 3. Spawn: claude -p "<prompt template>" (model: claude-opus-4-7)
    ↓ 4. Append claude's reply to comments.jsonl (kind: "answer", in_reply_to: c047)
    ↓ 5. git commit + push
    ↓ 6. POST eps.superkaiba.com/api/revalidate?path=/tasks/413
[Vercel] revalidates; mentor sees reply within ~30s
```

`tunnel.py` runs under systemd (`task-workflow-tunnel.service`). The Cloudflare tunnel runs under systemd (`cloudflared.service`). Both restart automatically.

## 10. Clean-result markdown spec + verifier

A clean-result `body.md` has three required H2 sections in order (`## Figure`
is OPTIONAL as of 2026-05-26; when present it sits between `## TL;DR` and
`## Details`):

```markdown
# <one-sentence claim> (LOW | MODERATE | HIGH confidence)

## TL;DR
- **Motivation:** ...
- **What I ran:** ...
- **Results:**
    - *Finding 1 in one sentence.* Prose narrative.
        ![alt-text](https://raw.githubusercontent.com/.../<sha>/figures/issue_N/finding1.png)
    - *Finding 2 in one sentence.* Prose narrative.
        ![alt-text](https://raw.githubusercontent.com/.../<sha>/figures/issue_N/finding2.png)
- **Next steps:** (OPTIONAL — include when there's genuinely useful follow-up to queue)

## Figure  <!-- OPTIONAL: legacy single-hero pattern; OMIT when figures are inline under TL;DR -->
![alt](https://raw.githubusercontent.com/.../<sha>/figures/issue_N/hero.png)
*Caption: ≥10 words describing what's plotted.*

## Details
Free-form markdown: definitions, training, eval rationale, sample completions in fenced code, statistical-test rationale, parameters table.

Confidence: LOW | MODERATE | HIGH — <one sentence naming the binding constraint or surviving evidence>.

## Reproducibility
**Artifacts:**
- Model: [hf-hub-url with /tree/<ref>](...)
- Dataset: [hf-hub-url](...)
- Raw completions: [hf-hub-url](...)
- WandB run: [wandb-url](...)
- Eval JSON: `eval_results/issue_N/run_result.json` @ commit `<sha>`

**Compute:** wall time, GPU type, pod name.

**Code:** entry script, git commit SHA, Hydra config path, copy-pasteable reproduce command.
```

`scripts/verify_task_body.py` checks (mechanical, exit non-zero on FAIL):

1. Title line ends with `(LOW|MODERATE|HIGH confidence)`.
2. Four required H2 sections present in order: `TL;DR`, `Figure`, `Details`, `Reproducibility`.
3. TL;DR bullets contain labels `Motivation:`, `What I ran:`, `Results:`, `Next steps:`.
4. Reproducibility URLs are permanent: HF Hub `/tree/<ref>`, WandB `/runs/<id>`, GitHub `/blob/<sha>`. No `TBD`, `{{`, `default`, `see config`, or empty placeholders. `n/a` accepted as an explicit non-applicable marker.
5. Confidence sentence in `Details` matches the title's confidence level.
6. Figure caption ≥10 words.

Six checks, all string-pattern matching. No HTML parsing. ~150 LOC.

## 11. Figures

All figures generated via the existing `paper-plots` skill + `src/explore_persona_space/analysis/paper_plots.py` style guidelines. Colorblind-safe palettes, Inter font, error bars, commit-pinned metadata. PNG output committed to `tasks/<status>/<id>/artifacts/`. The web app renders them inline via `react-markdown`'s image handler with lightbox-on-tap.

## 12. Supervisor daemon

`scripts/supervisor.py`, runs under systemd (`task-workflow-supervisor.service`). Every 5 minutes:

1. Read `tasks/REGISTRY.json`. Filter to tasks with non-terminal status (anything except `awaiting_promotion`, `completed`, `blocked`, `archived`).
2. For each, query Happy daemon RPC (`127.0.0.1:<port>/sessions`) for an active session matching task N.
3. If none, spawn one: `python scripts/spawn_session.py spawn-issue --issue N`. Record the new Happy session ID into `body.md` frontmatter via `task.py set-frontmatter`.
4. Log to `~/.task-workflow/supervisor.log`.

Handles: crashes, OOM kills, user-killed Claude, VM reboot. `blocked` status is the explicit "do not auto-resume" signal — agents that hit a halt criterion set `status=blocked` and exit; supervisor leaves them.

## 13. Cron jobs

All three run on this VM via `crontab -e`. Each commits + pushes its output.

### 13.1 Literature surfacing — `scripts/literature_cron.py` (06:30 daily)

1. Read recent clean-result bodies (`tasks/awaiting_promotion/*/body.md` and `tasks/completed/*/body.md` with `has_clean_result=true`, updated in last 14 days).
2. Call `mcp__arxiv__search_papers` for new cs.AI / cs.CL / cs.LG papers since yesterday.
3. Compute relevance per candidate: embedding similarity (sentence-transformers `BAAI/bge-large-en-v1.5`, local) between paper abstract and concatenated recent clean-result content.
4. Top-K=10 candidates. For each: spawn a small Claude call → 3-sentence intuitive summary + list of relevant task IDs.
5. Write `updates/literature/YYYY-MM-DD.md`.
6. Append each paper as JSON to `updates/literature/INDEX.jsonl` (dedup by arxiv id).
7. Commit + push.

### 13.2 Daily update — `scripts/daily_update.py` (07:00 daily)

1. Spawn a headless `research-pm` agent: "Read REGISTRY + every task updated in last 24h + literature/YYYY-MM-DD.md. Write yesterday-progress / today-plan / blockers / surfaced-literature."
2. Output `updates/daily/YYYY-MM-DD.md`.
3. Commit + push.

### 13.3 Weekly update — `scripts/weekly_update.py` (Mon 08:00)

Same shape, 7-day window + project-summary section + next-week outlook. Output to `updates/weekly/YYYY-WNN.md`.

## 14. Sagan import (one-shot)

`scripts/sagan_import.py` is the migration bridge:

1. Pull every experiment from Sagan API: `GET /api/experiments?limit=10000`.
2. For each experiment `e`:
   - Resolve target folder: `tasks/<status_lower>/<e.number>/`.
   - Write `body.md` with frontmatter (title, kind, tags, parent_id, created_at, pod_name, has_clean_result) + body content.
   - Write `events.jsonl` by fetching `GET /api/experiments/<e.id>/events` and translating each `workflow_event` row 1:1.
   - Write `comments.jsonl` empty (or pull from any Sagan user-note table if it exists — needs check).
   - If `e.body` is Sagan-card HTML: leave it as-is for grandfathered viewing; `verify_task_body.py` skips bodies with a `<!-- sagan-card-grandfathered -->` sentinel that the importer adds.
3. Write `tasks/REGISTRY.json` with `highest_id = max(e.number)`.
4. Commit in batches of ~100 experiments per commit.
5. After completion: `task.py audit` validates everything (folder paths agree with REGISTRY, no orphan IDs, no duplicate IDs).

After this runs, Sagan stays accessible at `sagan.superkaiba.com` for historical reference. No agent calls `sagan_state.py` anymore.

## 15. Pod lifecycle (unchanged)

`scripts/pod.py` is unmodified. Ephemeral `epm-issue-N` pods continue exactly as today:

- `pod.py provision --issue N --intent <intent>` at /issue step 4
- `pod.py terminate --issue N --yes` automatic at /issue step 8 after upload-verification PASS
- `pod.py resume --issue N` for follow-up sessions
- SSH MCP server config regeneration on every pod change

The only delta: `N` now refers to a row in `tasks/REGISTRY.json` instead of a Sagan `experiments.number`. Same numeric value, just different lookup.

## 16. Prerequisites (deferred)

These are needed at specific execution steps. None block worktree creation or `task.py` development.

1. **Cloudflare tunnel auth** (blocks tunnel deployment in step 7). One of:
   - (a) Interactive `cloudflared tunnel login` once (browser)
   - (b) Add `Account: Cloudflare Tunnel: Edit` scope to the existing `cfut_…` API token
   - (c) Manually create tunnel in CF dashboard, paste connector token into `~/.eps-secrets`
2. **GitHub PATs** (block Vercel app first run, step 4). `COMMENTS_PAT` = contents:write on `superkaiba/explore-persona-space`. `READ_PAT` = contents:read on same.
3. **Google OAuth client** (blocks first auth flow, step 5). Created in Google Cloud Console; redirect URI = `https://eps.superkaiba.com/api/auth/callback/google`.
4. **Vercel project** (blocks step 4). `eps-dashboard` project linked to this repo, root directory `dashboard/`.
5. **DNS records** (blocks tunnel + Vercel custom domain). I write `cairn.superkaiba.com` and `eps-tunnel.superkaiba.com` CNAMEs via the existing `cfut_` token (DNS scope works).

## 17. Execution order

Each step is a separate commit on `task-workflow`. Pause points marked.

| Step | Deliverable | Verification | Pause? |
|---|---|---|---|
| 1 | Create `task-workflow` worktree off `main` | `git worktree list` | – |
| 2 | `scripts/task.py` + `src/explore_persona_space/task_workflow/` + `tests/test_task_py.py` | `pytest tests/test_task_py.py` PASS | – |
| 3 | `scripts/sagan_import.py`; run once; review imported `tasks/` tree | `task.py audit` PASS; spot-check 5 random task pages on disk | **PAUSE** for user review |
| 4 | Scaffold `dashboard/` Next.js app; deploy preview to Vercel; create the two GitHub PATs | Preview deploy renders task list + a task page | **PAUSE** for user review (mobile + desktop) |
| 5 | Wire up Google OAuth + allowlist + comment write path | Mentor email can comment; non-allowlisted email cannot | – |
| 6 | `scripts/verify_task_body.py` + first new clean-result rendered end-to-end | Verify spec on a sample body; render in browser | – |
| 7 | `scripts/tunnel.py` + Cloudflare tunnel setup; mentor-Q&A round trip | Submit comment from browser, see Claude reply within 30s | **PAUSE** for user to test mentor flow on phone |
| 8 | `scripts/supervisor.py` + systemd unit; supervisor kills + revives test session | Logs show respawn within 5 min of session kill | – |
| 9 | `scripts/literature_cron.py` + arxiv MCP integration + crontab entry | First run produces `updates/literature/<today>.md` + INDEX.jsonl entry | – |
| 10 | `scripts/daily_update.py` + `scripts/weekly_update.py` + crontab entries | Trigger manually; review output in browser | – |
| 11 | Surgical edits to all agent specs (`sagan_state.py` → `task.py`); `.claude/workflow.yaml` refs updated | All agent specs grep-clean; `workflow_lint.py` PASS | – |
| 12 | `.claude/skills/issue/SKILL.md` + other skill SKILL.md ports | Skill lint passes; SKILL.md self-test (mock /issue N) works | – |
| 13 | `CLAUDE.md` rewrite (Sagan State API section → Task Workflow API section) | Lint + spot-read | – |
| 14 | End-to-end test on one new task: `/issue NEW` from spawn → clean-result → user-promote → done | All gates fire; web reflects state at each step | **PAUSE** for user sign-off |
| 15 | Open PR `task-workflow` → `main` (don't merge yet) | PR diff reviewable; CI passes | **PAUSE** for user merge approval |

Total estimate: 2-3 weeks of work, day-by-day commits. Each pause point is a chance to redirect.

## 18. Migration safety

- Old code stays in tree. `sagan_state.py`, `verify_sagan_card.py`, `sagan_progress`, `clean-result-guidelines.md`, the entire Sagan-Postgres-aware skill copy — none are deleted in this PR.
- Sagan dashboard at `sagan.superkaiba.com` keeps serving read traffic indefinitely after cutover. No data loss.
- Rollback procedure: if the new system breaks before step 14, the worktree merge hasn't happened — `main` is unaffected. After step 14, if needed: re-deploy old Sagan-aware skill (`git revert <step-11..14>`), point agents back at `sagan_state.py`, re-enable Sagan as canonical. Importer is one-way; reverse direction would need a new script.
- The supervisor + cron jobs only deploy at step 8-10; existing crons untouched.

## 19. What I will NOT do without further sign-off

- Touch the running `scripts/sagan_state.py` or any in-flight experiments before the importer runs.
- Delete `archive/research_log/` or any existing artifacts.
- Modify `scripts/pod.py`, `experiment-implementer`, `experimenter`, `analyzer`, `clean-result-critic` core logic — only their state-store references.
- Change the auto-continuation policy, gate enumeration, or subagent halt criteria in `workflow.yaml`.
- Delete the Sagan repo or its Vercel deployment.
- Touch the `paper-plots` skill or `paper_plots.py` style code.
- Auto-merge the PR at step 15. Always paused for user.
