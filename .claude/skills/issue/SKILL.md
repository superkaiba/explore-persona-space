---
name: issue
description: >
  End-to-end task workflow for Explore Persona Space. `/issue <N>` takes a
  task number, reads state from local files under `tasks/<status>/<N>/`,
  appends `epm:*` markers to `events.jsonl`, and advances the EPS
  experiment lifecycle without any external tracker or remote service.
user_invocable: true
---

# /issue

Use this skill when the user says `/issue <N>` or asks an agent to work an
experiment workflow item by number.

`<N>` is the task number — the integer that names the per-task folder under
`tasks/<status>/<N>/`. It is not any external tracker number. External
tracker records (GitHub issues, the legacy Sagan dashboard) are historical
evidence only and must never be used as workflow state.

## State Backend

All durable state lives in plain files in the repo:

```
tasks/REGISTRY.json              # tiny index: id → current folder path
tasks/<status>/<N>/
  body.md                        # YAML frontmatter + markdown body
  events.jsonl                   # append-only `epm:*` markers (resume log)
  comments.jsonl                 # mentor comments + Claude replies
  plans/v{K}.md, plan.md         # plan revisions + symlink to latest
  artifacts/                     # figures, html artifacts, drafts
  original-body.md               # snapshot before clean-result promotion
```

- **Status** is the parent folder name. Allowed values: `proposed
  planning plan_pending approved running verifying interpreting
  reviewing awaiting_promotion completed blocked archived`.
- **Status change = atomic git mv + commit.** No `meta.status` field; the
  folder is the single source of truth.
- **Marker = one line appended to `events.jsonl`** in the task's current
  folder. Same `epm:*` shape we've always used.

Read and mutate state only through `scripts/task.py`. It holds an exclusive
`flock` on `~/.task-workflow/lock` for every mutation, writes one git commit
per operation, and is the only writer to these files (the web dashboard
only appends to `comments.jsonl`). No HTTP, no auth token, no remote
database.

Useful commands:

```bash
python scripts/task.py view <N>
python scripts/task.py set-status <N> planning --note "Clarifier resolved."
python scripts/task.py set-title <N> "New title"
python scripts/task.py set-body <N> --file /tmp/body.md
python scripts/task.py add-tag <N> eps
python scripts/task.py post-marker <N> epm:plan --note "Plan v1 written."
python scripts/task.py latest-marker <N> --prefix epm:
python scripts/task.py list-by-status --status running
python scripts/task.py new-plan-version <N> --file /tmp/plan.md
python scripts/task.py promote <N> useful
```

## Workflow

1. Load the task by number (`task.py view <N>` reads `body.md` frontmatter
   and recent `events.jsonl` rows).
2. If status is `proposed`, run the clarifier and either move to `planning`
   or record `epm:clarify-skip` with the reason.
3. During `planning` / `plan_pending`, run the adversarial-planner loop
   (Claude planner + Claude critic + Codex twin) and write plan versions
   to `tasks/<status>/<N>/plans/v{K}.md`. Print the dashboard URL
   `https://eps.superkaiba.com/tasks/<N>/plan` rather than dumping the
   plan body to the terminal.
4. `plan_pending` is the user-approval gate. After approval, move through
   `approved` → `running` → `verifying` → `interpreting` → `reviewing` →
   `awaiting_promotion`. Final promotion to `completed` is user-driven
   via `task.py promote <N> useful|not-useful`.

All transitions must post an `epm:*` marker (one row in `events.jsonl`).
Do not write parallel state files as the source of truth.

## Reviewer Pairs

For code review, interpretation critique, and clean-result critique, run the
Claude/Codex ensemble for at most three rounds. Post every reviewer verdict
and reconciler decision as an `events.jsonl` row with reviewer metadata.

Allowed verdicts:

- `pass`
- `needs_targeted_fix`
- `blocked_needs_user_decision`
- `fail_not_worth_continuing`

Round-3 rule: if reviewers still disagree after round 3, spawn the
`reconciler` agent. Its verdict is binding. The workflow continues unless
there is a true user-decision blocker (missing owner input, unsafe
execution, invalid artifacts, untestable hypothesis), in which case set
`status:blocked` and EXIT.

## RunPod

Pod lifecycle is `scripts/pod.py`'s job, unchanged:

```bash
python scripts/pod.py provision --issue <N> --intent <intent>   # before run
python scripts/pod.py terminate --issue <N> --yes                # automatic at upload-verify PASS
python scripts/pod.py resume --issue <N>                         # for follow-up work
```

The pod is named `epm-issue-<N>` to match `<N>` in `tasks/<status>/<N>/`.
Pod provisioning posts `epm:pod-provisioned`; auto-termination posts
`epm:pod-terminated`. Progress markers from the running pod are appended
via `scripts/pod_watch.py`.

## Completion Audit

Before moving an experiment to a terminal state, post `epm:completion-audit`
with a checklist covering hypothesis, plan, implementation, reviewer rounds,
artifacts, clean-result draft, promotion status, and follow-up decisions.
Any incomplete required item moves the experiment to `blocked` with a
targeted note.

See `markers.md` for marker names and metadata shape.
