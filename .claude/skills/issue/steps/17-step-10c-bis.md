# Step 10c-bis: Results-driven literature-positioning hook (findings-bearing tasks)

Step body relocated verbatim from `.claude/skills/issue/SKILL.md`
(#2155). SKILL.md keeps the heading, the state machine and the
Orchestration Procedure router; read this file when the run reaches
this step.

---

Auto-fires after a `kind: experiment` task lands at `completed` (and for
`kind: analysis` tasks that carry a measured finding). It closes the gap
between the project's front-loaded literature grounding (the planner's
hyperparameter sources + the clarifier's lit review, both keyed on the
QUESTION and run BEFORE results exist) and the post-results question
"we measured X — who else reported X, and does it replicate / contradict /
extend ours?". The `related-work-finder` agent runs a bounded,
findings-keyed arXiv-MCP + web search and PROPOSES (never applies) a short,
citation-verified "Related findings" note for the clean-result `## Goal` →
`**Broader narrative:**` slot. **Non-blocking + advisory:** the task is
already `completed`, so the proposal can park indefinitely; nothing about
completion waits on it, and a thin / empty / over-budget note never blocks
promotion. **0 GPU-h.**

**When to run** (mirrors Step 10c's gating):

1. `kind: experiment` → always.
2. `kind: analysis` → only when the task has a discernible measured finding
   (its clean-result body has a `## Results` section). If not, the agent
   writes a 3-line "no measured finding to position" stub and exits — no
   gate is raised.
3. `kind: infra | batch | survey` → SKIP entirely (no clean-result
   findings to position). Log one chat line `Step 10c-bis skipped
   (kind=<X>)` and continue to Step 10d.
4. **Idempotency:** skip if `epm:related-work-proposed v1` (for this park)
   already exists on the task — paired with `epm:related-work-applied v1`
   / `epm:related-work-rejected v1`, this covers re-entry / backstop ticks.
   For a same-issue follow-up round, re-run keyed on the new round's
   `followup_label` (the findings changed) — the same EXTEND pattern as the
   methodology-doc idempotency.

Spawn the `related-work-finder` agent (fresh context) — on the normal path
this spawn already happened in the Step 10b parallel batch (see Step 10b §
Parallel spawn with Step 10c + 10c-bis); spawn here only if it didn't.
Brief: source task `<N>` (the agent reads the clean-result body, skims
`docs/papers.md`, and anchors on the two pinned sibling papers itself). The
agent PROPOSES (never applies) the artifact `artifacts/related-work-proposal.md`
+ a rationale and returns; the orchestrator posts `epm:related-work-proposed
v1` (artifact path + the proposed ≤80-word `**Broader narrative:**`
addition + the `search_status` + the verified-citation list + the realized
search budget + the optional manual-triage papers list).

Present the proposed addition for confirmation at the
`related_work_positioning` conditional gate (registered in workflow.yaml §
gates.conditional). The prompt is a binary `confirm` vs `reject` (see
workflow.yaml § gates.related_work_positioning) — NOT a 3-option menu.

   <!-- gate: gates.related_work_positioning -->
   ```python
   AskUserQuestion(questions=[{
     "question": (
       "Apply this Related-findings positioning note for task #<N>? "
       "Proposal: epm:related-work-proposed v1 on https://eps.superkaiba.com/tasks/<N>"
     ),
     "header": "Related work #<N>",
     "multiSelect": False,
     "options": [
       {
         "label": "Confirm",
         "description": (
           "Splice the proposed <=80-word **Related findings:** clause into "
           "the ## Goal -> **Broader narrative:** slot via "
           "scripts/task.py set-body, re-run verify_task_body.py (WARN-only "
           "on the total-prose budget). Touches the task body's ## Goal slot "
           "ONLY (no docs/papers.md edit in v1)."
         ),
       },
       {
         "label": "Reject",
         "description": (
           "Skip; nothing written to the body. The proposal parks inline in "
           "epm:related-work-rejected v1 so a future pass can reconsider."
         ),
       },
     ],
   }])
   ```
5. Branch on the user's choice:
   - **Confirm** (optionally after hand-editing the clause): splice the
     confirmed ≤80-word clause into the body's `## Goal` →
     `**Broader narrative:**` slot via `set-body`, re-run the verifier
     (WARN-only on budget — never a blocking FAIL), and post the applied
     addition:
     ```bash
     uv run python scripts/task.py set-body <N> --file /tmp/issue-<N>-body-with-related-work.md
     uv run python scripts/verify_task_body.py --issue <N>   # WARN-only on the total-prose budget; never block
     uv run python scripts/task.py post-marker <N> epm:related-work-applied \
       --note "Applied Related-findings note to ## Goal -> **Broader narrative:**; verdict <V>; cited <arXiv ids>. No docs/papers.md edit (v1)."
     ```
     The gate applies ONLY the `## Goal` body edit — it does NOT apply any
     `docs/papers.md` edit in v1 (the agent's suggested-papers list is
     human-triage only; the papers.md auto-apply leg is a deferred
     follow-up).
   - **Reject:** write nothing to the body; record the decline with the
     proposal preserved inline:
     ```bash
     uv run python scripts/task.py post-marker <N> epm:related-work-rejected \
       --note "User declined the Related-findings proposal. Reason: <one line>. Proposal preserved inline."
     ```
<!-- example: anti-pattern -->
6. **Autonomous mode** (`EPM_AUTONOMOUS_SESSION=1`): do NOT raise the
   `AskUserQuestion`, do NOT print the proposed note as a confirm/reject
   text menu to chat, and do NOT auto-apply. A literature-positioning note
   is a taste / scope call the autonomous session does not make. The
   `epm:related-work-proposed v1` marker is already posted; AUTO-REJECT-PARK:
   post `epm:related-work-rejected v1` with the note
   `autonomous — parked for user review` and the proposal preserved inline,
   so it survives for a later interactive `/issue <N>` re-invocation.
   EXECUTE the continuation to Step 10d in this same turn; do NOT end the
   turn waiting on user confirmation.

This hook is idempotent: skip if `epm:related-work-applied v1` or
`epm:related-work-rejected v1` already exists on the task.
