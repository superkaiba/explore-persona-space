# Step 2c: Inline plan approval

Step body relocated verbatim from `.claude/skills/issue/SKILL.md`
(#2155). SKILL.md keeps the heading, the state machine and the
Orchestration Procedure router; read this file when the run reaches
this step.

---

**The autonomous plan-approval decision was already made by the Step 2b
`set-status ... --auto-approve-if-autonomous --gpu-hours <X>` call — in code,
not by LLM discretion here.** That command (in `scripts/task.py`) reads
`EPM_AUTONOMOUS_SESSION` + `EPM_PLAN_AUTOAPPROVE_GPU_HOURS` and printed a
`PLAN_GATE_DECISION:` line.
<!-- gate: gates.plan_approval -->
A PreToolUse hook on `AskUserQuestion`
(`.claude/settings.json`) ALSO hard-blocks (`exit 2`) any plan-approval
`AskUserQuestion` while `EPM_AUTONOMOUS_SESSION` is set — so the autonomous
path physically cannot reach the interactive ask even if this prose is
mis-followed. (Why both: the script removes the gate so the ask is never
reached; the hook is the backstop that forbids it if reached — four
`--auto` sessions once asked for plan approval when the auto-approve lived
only as prose here.)

Branch on the decision (equivalently, re-read the task status):

- **`auto_approved`** (autonomous, est ≤ cap): the gate already flipped the
  status to `approved` and posted `epm:plan-approved`. Do NOT ask, do NOT
  re-post. Continue to Step 4 in the **same invocation**.
- **`parked_over_cap`** (autonomous, est > cap OR blank estimate — FAIL
  SAFE): the gate left the status at `plan_pending` and already posted
  `epm:awaiting-spend-approval`. The PM session + the user's phone surface
  the `plan_pending` status. Post the §5 marker, fire a PushNotification,
  then EXIT:
  ```bash
  uv run python scripts/post_step_completed.py --issue <N> --step 2c \
    --exit-kind parked --notes "plan_pending; over auto-approve cap"
  ```
  ```python
  cap = os.environ.get("EPM_PLAN_AUTOAPPROVE_GPU_HOURS", "100")
  PushNotification({
      "message": f"#{N} {slug} parked at plan_pending — over {cap} GPU-h cap; open to approve"[:200],
      "status": "proactive",
  })  # soft-fail; deferred-schema may not be loaded
  ```
- **`interactive_pending`** (`EPM_AUTONOMOUS_SESSION` unset): fall through to
  the **Legacy autonomous mode** / **Interactive mode** bullets below.

Never auto-approve on a missing/ambiguous estimate — the gate parks a blank
estimate (fail safe). `awaiting_promotion` remains a human gate regardless of
this cap.

**Workflow-fix tasks — architectural greenlight REMOVED (2026-08-04).**
A `kind: infra` workflow-fix task (filed by the workflow-fix-on-bug protocol,
`.claude/rules/workflow-fix-on-bug.md`) is 0 GPU-h, so the GPU-h cap
auto-approves it — and that is the INTENDED behavior for
EVERY workflow fix, architectural / public-contract changes included. There is
no `architectural: true` park and no "spawn WITHOUT `--auto`" fallback.

Planners MUST NOT set `architectural: true` or emit an "ARCHITECTURAL — needs
user greenlight" banner: the flag is INERT (the
`--auto-approve-if-autonomous` gate never read it — `architectural` appears in
zero lines of `scripts/task.py`), so a plan carrying it will NOT park and the
banner would promise a review that never happens.

Review is unchanged and still binding: critic ensemble → implementer →
Claude+Codex `code-reviewer` → Step 9c test-verdict → Step 10d merge. What was
removed is the human veto, not the pipeline. Interactive mode is also
unaffected: the Step 2c plan-approval ask still governs a human-present
session.

Rationale: parked plans hold an infra concurrency slot indefinitely
(#1217/#1771 held 2 of 5 slots while 65 ripe infra fixes queued behind
them with `dispatched=0`).

- **Legacy autonomous mode** (no chat user present AND
  `EPM_AUTONOMOUS_SESSION` is unset — a headless invocation outside the
  standard `spawn_session.py spawn-issue --auto` path, which sets that
  env var): EXIT immediately; the task sits at
  `plan_pending` until a user approves via the dashboard or a future
  `/issue <N>` invocation. Before exiting, post the §5 marker:
  ```bash
  uv run python scripts/post_step_completed.py --issue <N> --step 2c \
    --exit-kind parked --notes "plan posted; awaiting user approval"
  ```

- **Interactive mode** (user is in the current chat session): Ask the
  user inline rather than exiting. Present the plan summary and ask:

  > Plan posted as `epm:plan v1` on task #\<N\>.
  >
  > **Plan path:** `${PLAN_PATH}` (symlink -> latest version)
  > **Dashboard URL:** `https://eps.superkaiba.com/tasks/<N>/plan` (planned)
  >
  > (1) **Approve** — advance to implementation
  > (2) **Revise** \<notes\> — plan goes back to adversarial-planner
  > (3) **Defer** — exit now; re-invoke `/issue <N>` later

  `${PLAN_PATH}` is the inline shell variable captured at Step 2 — both
  steps run in the same orchestrator turn (auto-continuation guarantees
  no pause between them) so the variable is in scope. There is no
  cache-file fallback.

  <!-- gate: gates.plan_approval -->
  <!-- autonomous-mode: block-and-fail -->
  Use `AskUserQuestion` or a plain text prompt and wait for the user's
  reply. (Interactive mode only — autonomous sessions never reach this
  branch; the code-enforced gate in `task.py
  --auto-approve-if-autonomous` already decided, and the PreToolUse hook
  <!-- gate: gates.plan_approval -->
  hard-blocks any `AskUserQuestion` if reached.)

  <!-- gate: gates.plan_approval -->
  <!-- autonomous-mode: block-and-fail -->
  **Important:** when invoking `AskUserQuestion` (Interactive mode
  only), embed the dashboard URL
  (`https://eps.superkaiba.com/tasks/<N>/plan`) inside the question
  text itself, AND embed the local plan path
  (`tasks/<status>/<N>/plans/plan.md`) inside the first option's
  `description` field. The user only sees the rendered question box at
  decision time; any link that lives only in chat prose above the
  `AskUserQuestion` call gets scrolled past. The chat-prose blockquote
  above is for orchestrator narration; the call itself must be
  self-contained. Example shape (see workflow.yaml § gates.plan_approval):

  <!-- gate: gates.plan_approval -->
  <!-- autonomous-mode: block-and-fail -->
  ```python
  # Interactive mode only — autonomous branches before this point.
  AskUserQuestion(questions=[{
    "question": (
      "Approve plan v1 for task #<N>? "
      "Plan: https://eps.superkaiba.com/tasks/<N>/plan"
    ),
    "header": "Plan #<N>",
    "multiSelect": False,
    "options": [
      {
        "label": "Approve",
        "description": (
          "Dispatch <implementer-type>. Est. <cost> GPU-hours. "
          "Local plan: tasks/<status>/<N>/plans/plan.md"
        ),
      },
      {
        "label": "Revise <notes>",
        "description": "Re-run /adversarial-planner with your notes.",
      },
      {
        "label": "Defer",
        "description": (
          "Park at plan_pending. Re-invoke /issue <N> later."
        ),
      },
    ],
  }])
  ```

  - **"Approve" / "1":** move task to `approved`. Post an `epm:plan-approved`
    event for audit trail. Continue to Step 4 in the **same invocation**
    — do NOT exit:

    > **Same-issue follow-up round?** At `followups_running`, SKIP the
    > `set-status` (status-hold rule, Step 9b § Same-issue follow-up loop
    > step 3; code-enforced — `task.py` refuses the flip) and post ONLY the
    > `epm:plan-approved` marker — the approval is recorded, the status holds.

    ```bash
    uv run python scripts/task.py set-status <N> approved \
      --note "Plan v1 approved by user."
    uv run python scripts/task.py post-marker <N> epm:plan-approved \
      --note "User approved plan v1 inline."
    ```
  - **"Revise \<notes\>" / "2":** set status back to `planning`. Re-invoke
    adversarial-planner with the revision notes. Re-run the consistency
    checker. Post `epm:plan v2` via `new-plan-version`. Loop back to
    Step 2c.
  - **"Defer" / "3":** EXIT. Status stays at `plan_pending`. User
    re-invokes `/issue <N>` later to approve. Before exiting, post the
    §5 marker:
    ```bash
    uv run python scripts/post_step_completed.py --issue <N> --step 2c \
      --exit-kind parked --notes "plan_pending; user deferred"
    ```
