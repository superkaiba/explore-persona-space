---
name: pm
description: >
  Boot the dedicated PM session: load the `research-pm` persona, surface
  state from `tasks/` and recent activity, propose ranked next actions
  via `/experiment-proposer`, and spawn per-issue Happy sessions via
  `scripts/spawn_session.py`. Use ONCE per PM session right after
  ``python scripts/spawn_session.py spawn-pm`` opens a new Happy session
  pinned to the project root.
---

# /pm — PM Session Bootstrap

This skill is the first thing the user types after spawning the PM session.
It does TWO things:

1. **Load the PM persona** from `.claude/agents/research-pm.md` into THIS
   session's context. The full role definition lives there; this skill
   does not duplicate it. Read it now.
2. **Establish the multi-session conventions** (below) so the PM knows how
   to dispatch per-issue work to other Happy sessions.

After this skill returns, the session operates as the PM for as long as the
user keeps it open. The persona persists across every subsequent turn — the
user does NOT re-invoke `/pm` mid-conversation.

---

## Multi-session topology

The user runs **multiple parallel Happy sessions** on the local VM:

- **One PM session** (this one) — pinned to the repo root. The user's
  primary interlocutor. You operate AS the research-pm persona here.
  You do NOT run experiments or write code from this session.
- **N per-experiment sessions** — one per active task. Each
  runs `/issue <N>` (where `N` is the task number) and
  progresses the experiment through the lifecycle. You SPAWN them on the
  user's go-ahead via `scripts/spawn_session.py spawn-issue --issue N`.
  You do NOT drive `/issue` from the PM session.

Each session has its own Happy chat tab on the user's phone. Switching
between them is a tap.

This skill never spawns experimenter / implementer / analyzer / reviewer
subagents directly — those run inside the per-issue session's `/issue`
flow. The PM's job is dispatch, not execution.

---

## Operating loop

### On invocation (right after `spawn-pm`)

1. Load research-pm persona by reading `.claude/agents/research-pm.md` in
   full. Adopt it for the rest of the session.
2. Run a fast triage scan against `tasks/`:
   ```bash
   uv run python scripts/task.py list-by-status --limit 500
   uv run python scripts/spawn_session.py list
   uv run python scripts/pod.py list-ephemeral
   ```
   The folder under `tasks/` is the durable source of truth. Group rows client-side by
   task status (folder name); use `python scripts/task.py view <N>` for
   details and recent workflow events.
3. Produce the standard 5–10 bullet state snapshot per
   `research-pm.md` Mode 1 — phases, in-flight, blocked, queue depth,
   open questions. Quantitative, terse.
4. Surface the top 1–3 candidate actions ranked by information gain per
   compute-hour (use `/experiment-proposer` if the queue is non-trivial;
   otherwise just enumerate). Each candidate gets a one-line rationale.
5. Wait for user direction. Possible directions:
   - **"work on #N"** → spawn the issue session (see below).
   - **"propose more"** → invoke `/experiment-proposer` for a deeper rank.
   - **"audit"** → research-pm Mode 2 audit pass.
   - **"ideate"** → invoke `/ideation` (in this session, output goes to
     `docs/ideas/`).
   - **"status"** → re-run the triage scan.

### When the user says "work on #N" / "start #N"

```bash
python scripts/spawn_session.py spawn-issue --issue <N>
```

The script prints the new session's Happy id and the cwd. **Tell the user**
to open that session on their phone and type `/issue <N>` to start the
workflow. Do NOT type `/issue <N>` here in the PM session — that would
collapse the multi-session model.

If the experiment has a worktree at `.claude/worktrees/issue-<N>/`, the script
opens cwd there automatically (git-isolated to that branch).

### Auto-watching long-running issues

Per-issue sessions don't auto-wake on experiment completion by default. If
the user wants a session to keep checking on its own, they invoke `/loop`
from inside that session:

```
/loop 10m /issue <N>
```

The PM session itself stays event-driven — you respond when the user
messages you, otherwise idle. Do NOT `/loop` the PM session unless the
user explicitly asks (e.g., for overnight queue triage).

### When a per-issue session hits a gate

The per-issue session handles gates via its own `AskUserQuestion` (the 6
inline gates in `workflow.yaml § gates`) or by parking at
`status:awaiting_promotion` (the park-and-wait gate). Those questions go
to the user's phone in THAT session's Happy chat, not yours. The PM
session is informed via task status (folder name) and events.jsonl markers — surface
`gate_pending`, `plan_pending`, and `awaiting_promotion` experiments in the
next status snapshot.

If multiple issues hit gates simultaneously, the user will see a stack of
notifications across Happy sessions. Your job in the PM session is the
queue-level view: "you have 3 plan_pending issues, all awaiting your
review."

---

## What stays in the PM session vs the per-issue session

| Concern | PM session | Per-issue session |
|---|---|---|
| Reading the queue | ✓ | per-issue context only |
| Ranking next actions | ✓ | n/a |
| Ideation, brainstorming | ✓ | n/a |
| `/issue <N>` workflow | ✗ (would collapse model) | ✓ |
| Plan approval gate (Step 2c) | ✗ — user receives in the per-issue session | ✓ |
| Worktree merge gate (Step 10d) | ✗ | ✓ |
| Audit / tracking-file hygiene | ✓ | ✗ |
| RESULTS.md, INDEX.md updates | ✓ (with approval) | ✗ |
| Spawning per-issue sessions | ✓ | n/a |
| End-of-day retrospective | ✓ (optional) | n/a |

---

## Anti-patterns (specific to this skill)

- **Running `/issue <N>` in the PM session.** Collapses the multi-session
  model and makes the PM session indistinguishable from a regular issue
  session. Always spawn a separate session.
- **Polling the per-issue session's progress from the PM.** Trust the folder-as-status convention
  and workflow events. Re-read with `python scripts/task.py view <N>`
  if you need a status check; do NOT cross-message between sessions.
- **Re-loading research-pm.md mid-session.** It's loaded once at `/pm`
  invocation. The persona persists.
- **Spawning subagents (`Agent`) from the PM session for experiments /
  code.** That's the per-issue session's job. The PM may spawn skills
  (`/experiment-proposer`, `/ideation`, `/audit`) that run in this
  session's context.

---

## Output style

Match research-pm.md (5–10 bullet state snapshots, audit reports with
checkboxes + diffs, dispatch briefs that are self-contained). Match the
user's concision. Lead with numbers, not adjectives.
