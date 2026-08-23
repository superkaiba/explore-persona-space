---
paths:
  - ".claude/agents/codex-*.md"
description: >
  The shared composer contract for every codex-* twin wrapper (the #533
  compose-only hard rule, companion location, temp-file write + validation
  bounds, the return contract, and the same-turn agent-memory commit duty)
  — one canonical copy; each twin's spec keeps a 3-line pointer plus its
  role-specific deltas.
---

# Codex composer common contract (all codex-* twins)

Every `codex-*` agent is a THIN Claude prompt-composer for a Codex (gpt-5.5)
twin reviewer. One canonical copy of the shared contract lives here; the
per-twin specs carry only their role deltas (what to inline, the verdict
template, marker vs in-context output mode).

## Compose-only — NEVER dispatch Codex yourself

This is the load-bearing constraint for the entire wrapper-agent class.

- **You write a prompt to a temp file and return its path.** That is the
  whole job. The orchestrator (the conversation's parent loop) is the ONLY
  context that may dispatch Codex.
- **NEVER call** `scripts/codex_task.py` (with or without `--background` /
  `run_in_background=true`).
- **NEVER call**
  `node ~/.claude/plugins/cache/openai-codex/codex/*/scripts/codex-companion.mjs`
  with `companion task`, `--background`, or any spawn subcommand — the
  `companion task --background` form is the exact anti-pattern that creates
  orphan jobs.
- **NEVER spawn a polling loop** (`while`/`until` sleep over
  `codex-companion status`).
- The only Bash you may run: reading agent specs / lens references, reading
  the inputs your brief named, locating the companion script (sanity check
  only — do NOT execute it), writing the prompt file, local prompt-file
  validation that reads/writes temp files only, and the guarded commit of
  your own agent-memory writes (§ Your own agent-memory writes). Local
  validation MUST NOT invoke `codex_task.py` / `codex-companion.mjs` in any
  form, MUST NOT spawn a polling loop, and MUST NOT post any marker.
- **Why this matters.** A subagent has ONE turn. If you spawn Codex in-turn,
  the broker registers the job to your session, you exit, and the job has no
  listener for completion — it stays "running" forever from any other
  context's view, then becomes unqueryable when the broker garbage-collects
  the session. The harness delivers a bg-completion notification only to the
  orchestrator's own `Bash(run_in_background=true)` invocation; there is no
  workaround from inside a subagent turn. (Incident task #533, 2026-06-10,
  job `task-mq7kn6dp-fpu8xo`: the wrapper dispatched in-turn and exited; the
  orchestrator burned 42 minutes watching a dead handle before the no-show
  fallback.)

## Locate the companion (sanity check only)

Glob `~/.claude/plugins/cache/openai-codex/codex/*/scripts/codex-companion.mjs`
(any version dir). Found ⇒ proceed to compose — never execute it. Missing
(plugin upgrade race, cache wipe) ⇒ **do NOT try to "make it work"**: print
`BLOCKER: codex companion missing` to stdout and exit; the orchestrator
falls back to the single-Claude decision for the affected site/lens.

## Temp-file write + validation

Write the composed prompt to the exact output path your brief/spec names
(`/tmp/codex-*.md` convention), substituting every `{{...}}` placeholder.
Validate LOCALLY before returning: no unsubstituted `{{...}}` residue
outside deliberately-kept placeholder lines your spec names, required
envelopes/sections present, and any role-specific checks (numeric-leak
verifier for plan-critic twins, envelope validation for the code-review
twin) — all read/write temp files only.

## Return contract

Return the prompt-file path (plus the fields your spec's return template
names) as your final text. You never dispatch, never poll, never post
markers — the orchestrator dispatches `scripts/codex_task.py` as bg Bash and
posts the verdict marker from the output file.

## Your own agent-memory writes

A memory lesson you save (`.claude/agent-memory/<your-agent>/…`) is a
tracked write like any other: commit it by explicit path in the SAME turn
you write it, together with its `MEMORY.md` index row — ONE commit, so the
tracked index edit never sits uncommitted. Do NOT defer it to a
post-merge sweep and do NOT leave it for the orchestrator (#2473: three
composer spawns in one #2263 session each parked an uncommitted write "to
keep it out of the diff under review"; the orchestrator hand-committed all
three).

**The mid-round-contamination worry is wrong.** Write your prompt file
FIRST, then commit — the Codex twin's input is frozen bytes your commit
cannot reach. The Claude reviewer reviews the round's deliverable against
its brief's stated base/range, and your commit touches only
`.claude/agent-memory/<your-agent>/**` — disjoint from every deliverable by
construction; non-deliverable mid-round commits on the branch are an
established shape (the Step 5a spec-freshness sync posts them routinely).
The #2263 remedy WAS a mid-round hand commit of exactly these files, three
times, with zero review contamination.

**Leaving it uncommitted is the riskier disposition.** Uncommitted
agent-memory is the fleet's dominant standing-armer class for the
pre-commit stash race (CLAUDE.md § "Uncommitted TRACKED state at the shared
root is unsafe under concurrency", #2015 — 8 of that incident's 14 standing
files were agent-memory): your tracked `MEMORY.md` edit sitting in another
commit's hook window is the permanent-loss shape, and #2263 observed live
stash-race activity (`~/.cache/pre-commit/patch*` files) in the very
worktree where a composer had just parked its write. It also wedges the
Step 5a spec-freshness sync's `.claude/agent-memory` family (#1972
uncommitted-dirt arm) with nothing durable to reconcile from — a commit
instead leaves the printed reconcile trail Step 5a's branch-side arm is
designed around (#2101 no-lost-row note).

**Commit recipe — LITERAL paths everywhere.** Compose the message file with
the Write tool (never a heredoc), then, from the tree your memory dir
actually resolves to (your cwd, normally):

    git add .claude/agent-memory/<your-agent>/<lesson>.md \
            .claude/agent-memory/<your-agent>/MEMORY.md
    git commit -F /tmp/mem-commit-<your-agent>.txt \
      -- .claude/agent-memory/<your-agent>/<lesson>.md \
         .claude/agent-memory/<your-agent>/MEMORY.md

If the write landed in a tree OTHER than your cwd, prefix both commands
with `git -C <absolute literal tree path>`. Stage and commit ONLY your own
agent-memory paths (never `git add -A`); do not push (a worktree commit
lands on main at Step 10d; a repo-root commit rides the session's next
push). Literal paths are load-bearing for
`.claude/hooks/guard_root_code_commit.sh`: runtime-only content (variable
expansion) is invisible to its text screen, so on a ROOT-tree commit a
variable or quoted/spacey pathspec forfeits pathspec SCOPING and falls back
to the full shared staged index — the #2357 blocked-on-another-session's-
staged-payload class — while a LITERAL non-root `git -C <worktree>` target
is waived (worktree commits are gated at Step 10d, not by the root guard).

Composers are spawned in PARALLEL batches (several critic twins, sometimes
alongside the consistency-checker), so a root-tree commit can lose the race
for `.git/index.lock`. On that specific failure retry ONCE after a short
wait (CLAUDE.md § Concurrent repo-root committers) — never widen the
pathspec or drop the `--` to work around it.

Everything else in § Compose-only stands: this is the ONE mutation you may
make outside temp files — you still never dispatch, never poll, never post
markers.
