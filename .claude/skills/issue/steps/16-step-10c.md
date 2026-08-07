# Step 10c: Living-docs update hook (experiments only)

Step body relocated verbatim from `.claude/skills/issue/SKILL.md`
(#2155). SKILL.md keeps the heading, the state machine and the
Orchestration Procedure router; read this file when the run reaches
this step.

---

Auto-fires after a `kind: experiment` task lands at `completed` (the
deliberate post-promotion completion moment). It keeps the living
research hub (`docs/open_questions.md`, and `docs/papers.md` when
warranted) from going stale by proposing — never auto-applying — an
update to the question(s) this experiment was linked to at creation
(Step 0c-link). **Non-blocking:** the task is already `completed`, so
the proposal can park indefinitely if the user is away; nothing about
completion waits on it.

1. Skip when the task `kind != "experiment"` — `analysis | infra |
   batch | survey` carry no open-question link.
2. Skip when the task has no `relates_to:` list in `body.md`
   frontmatter (was never linked at Step 0c-link) — surface one chat
   line noting the missing link and continue to Step 10d.
3. Spawn the `living-docs-updater` agent (fresh context) — on the
   normal path this spawn already happened in the Step 10b parallel
   batch (see Step 10b § Parallel spawn with Step 10c + 10c-bis); spawn
   here only if it didn't. Brief: task
   `<N>` + its clean-result body + the linked question block(s) (grep
   `docs/open_questions.md` for each `relates_to` id's `<!-- q:<id> -->`
   anchor) + the rest of `open_questions.md` so it can spot a needed
   reword / split / merge / new question. The agent PROPOSES (never
   applies) a unified diff + rationale and posts
   `epm:living-docs-proposed v1`. It is bounded + single-turn.
4. Present the proposed diff for confirmation at the
   `living_docs_update` conditional gate (registered in
   workflow.yaml § gates.conditional). The prompt is a binary `confirm`
   vs `reject` (see workflow.yaml § gates.living_docs_update); "edit" is
   a refinement of `confirm`, not a third option — the user may hand-edit
   the proposed diff and the same confirm path applies the edited patch.

   <!-- gate: gates.living_docs_update -->
   ```python
   AskUserQuestion(questions=[{
     "question": (
       "Apply this living-docs update for task #<N>? "
       "Proposed diff: epm:living-docs-proposed v1 on https://eps.superkaiba.com/tasks/<N>"
     ),
     "header": "Living docs #<N>",
     "multiSelect": False,
     "options": [
       {
         "label": "Confirm",
         "description": (
           "Apply the proposed diff (edit it first if you like) via "
           "scripts/living_docs.py apply <N> <patch>. Touches "
           "docs/open_questions.md (+ docs/papers.md if proposed)."
         ),
       },
       {
         "label": "Reject",
         "description": (
           "Skip; nothing written to the living docs. The proposal "
           "parks for the nightly /daily living-docs backstop re-synthesis."
         ),
       },
     ],
   }])
   ```
5. Branch on the user's choice:
   - **Confirm** (optionally after hand-editing the diff): apply the
     confirmed patch and post the applied diff:
     ```bash
     uv run python scripts/living_docs.py apply <N> /tmp/issue-<N>-living-docs.patch
     uv run python scripts/task.py post-marker <N> epm:living-docs-updated \
       --note "Applied living-docs update; touched <q-ids>; State trailer(s) bumped."
     ```
     `living_docs.py apply` is the single writer (atomic flock + one
     commit + dated changelog line). It applies ONLY the confirmed patch
     — accretive evidence/State bump or broader multi-question edit, no
     judgement of its own.
   - **Reject:** write nothing to the docs; record the decline:
     ```bash
     uv run python scripts/task.py post-marker <N> epm:living-docs-update-rejected \
       --note "User declined the living-docs proposal. Reason: <one line>. Proposal preserved inline."
     ```
<!-- example: anti-pattern -->
6. **Autonomous mode** (`EPM_AUTONOMOUS_SESSION=1`): do NOT raise the
   `AskUserQuestion`, do NOT print the proposed diff as a confirm/reject
   text menu to chat, and do NOT auto-apply. Per § Autonomous session
   behavior → `living_docs_update`, living-docs mutations are user-only
   by spec. The `epm:living-docs-proposed v1` marker is already posted;
   the proposal parks for the user to confirm on a later `/issue <N>`
   re-invocation or for the nightly /daily living-docs backstop re-synthesis to
   reconcile. EXECUTE the continuation to Step 10d in this same turn;
   do NOT end the turn waiting on user confirmation.

This hook is idempotent: skip if `epm:living-docs-updated v1` or
`epm:living-docs-update-rejected v1` already exists on the task.
