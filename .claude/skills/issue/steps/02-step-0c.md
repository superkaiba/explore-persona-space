# Step 0c: Goal-of-experiment gate (safety net)

Step body relocated verbatim from `.claude/skills/issue/SKILL.md`
(#2155). SKILL.md keeps the heading, the state machine and the
Orchestration Procedure router; read this file when the run reaches
this step.

---

Every `kind: experiment` task must carry a one-sentence **Goal** in
body.md frontmatter (`goal:`) and an inline `## Goal` H2 block before
any other H2. The Goal is the canonical optimization target every
downstream subagent reads (planner, critic, experiment-implementer,
analyzer, clean-result-critic, interpretation-critic,
follow-up-proposer). The PM session Mode 5 pre-spawn check is the
primary enforcement point; Step 0c is the per-issue-session safety
net.

This is a **legitimate `AskUserQuestion` use** in Interactive mode
because the gate IS a gate (CLAUDE.md "Critical Rules" lists
`experiment_goal` as inline gate #6 — see workflow.yaml §
gates.experiment_goal). It does not violate the auto-continuation
policy. In autonomous mode (`EPM_AUTONOMOUS_SESSION=1`), the Goal must
have been set BEFORE the session was spawned (the PM session Mode 5
pre-spawn check is the primary enforcement); if it's still missing at
Step 0c, the autonomous session post `epm:failure v1 failure_class: data`
(reason: `goal missing; autonomous mode cannot synthesise`), sets
`status:blocked`, and exits (halt-criterion #4). The PreToolUse hook
hard-blocks the ask if reached. <!-- autonomous-mode: block-and-fail -->

1. Skip the gate when the task `kind != "experiment"` (i.e.
   `analysis | infra | batch | survey`). These kinds do not carry an
   experiment Goal.
2. Otherwise, read the task's frontmatter + body via `task.py view <N>
   --json` and check:
   - Frontmatter contains `goal: <non-empty string>`, AND
   - The body contains a `## Goal` H2 (matched verbatim, line-start).

   If both hold, continue to Step 1.
3. If either is missing, raise `AskUserQuestion` <!-- gate: gates.experiment_goal --> <!-- autonomous-mode: block-and-fail -->:
   ```
   "What is the one-sentence Goal of this experiment?
    (The single decision-shaping target every downstream agent will
    optimize toward — e.g. 'Measure whether persona-tagged SFT
    transfers to held-out personas at the same rate as in-distribution
    ones.')"
   ```
   (Interactive mode only — autonomous sessions block-and-fail per the
   §0c-intro annotation above.) On the user's answer (one sentence; do
   NOT accept a fragment or a list — re-prompt once if the answer
   doesn't read as a complete sentence), run:
   ```bash
   uv run python scripts/task.py set-goal <N> "<the answer>" --by user
   ```
   The command writes both frontmatter (`goal:`) and the body H2
   block, then posts `epm:goal-updated v1` to events.jsonl. Re-read
   the task (Step 0) and continue to Step 0c-link.

#### Step 0c-link: Match-or-create open-question link (same Goal gate)

After the Goal is set for a `kind: experiment` task, link it to the
living research hub (`docs/open_questions.md`) so the completion hook
(Step 10c) knows which question(s) the result should move. This runs
inside the same Goal gate the user already passes through — no separate
gate, no extra context switch.

1. Skip when the task `kind != "experiment"` (i.e.
   `analysis | infra | batch | survey`). Those kinds carry no
   open-question link, exactly like the Goal gate itself.
2. Skip when the task already carries a non-empty `relates_to:` list in
   `body.md` frontmatter (re-invocation / already-linked case) — the
   link is set once at creation. Continue to Step 1.
3. Otherwise, read the task Goal + the headline questions in
   `docs/open_questions.md` and produce a flat list of stable
   open-question ids (NO primary/secondary) the experiment bears on —
   **matching** existing question id(s) wherever an existing question
   fits, and only **drafting a new question** when none fit.
4. **Matching existing question(s) — AUTO-LINK, do NOT ask.** When every
   id in the list is an *existing* question id (no new question needs to
   be drafted), write the link immediately, without asking the user — no
   gate prompt. State the match in chat so the user can correct it if
   it's wrong, then write it:
   ```
   Assumption: linking #<N> to existing open question(s) <q-ids> «<headline(s)>».
   ```
   ```bash
   uv run python scripts/living_docs.py link <N> <q-id> [<q-id> ...]
   ```
   This is the common case — an experiment almost always bears on a
   question that already lives in the hub. Linking to an existing
   question is a low-risk, reversible bookkeeping write (the
   `living_docs.py check` lint + the completion-time `living-docs-updater`
   both catch a bad link later), so it does not consume a gate.
5. **No existing question fits → drafting a NEW question — ASK first
   in Interactive mode.** Creating an open-question stub is a real,
   durable living-docs mutation, so the new-question path stays
   user-confirmed. Propose the new question (plus any existing ids that
   ALSO apply) via
   `AskUserQuestion` <!-- gate: gates.experiment_goal --> <!-- autonomous-mode: skip --> in the SAME Goal
   gate:
   ```
   "No existing open question in docs/open_questions.md fits this
    experiment's Goal. Draft a new one? (an experiment may also bear on
    existing questions — add them too.)
      - Draft new question: «<one-sentence proposed question>» [+ also link q-<id> ...]
      - Link only to existing instead: q-<id> «<headline>» [+ more]"
   ```
   On the user's confirmation, write the link via the same command:
   ```bash
   uv run python scripts/living_docs.py link <N> <q-id> [<q-id> ...]
   ```
   `living_docs.py link` creates the question stub (heading +
   `<!-- q:<id> -->` anchor + `State:` trailer) in `docs/open_questions.md`
   for any id that does not yet exist, then writes `relates_to` + the
   evidence entry.
6. In both cases, post `epm:question-linked v1` recording the
   `relates_to` list, whether a new question was created, and the mode:
   ```bash
   uv run python scripts/task.py post-marker <N> epm:question-linked \
     --note "Linked task #<N> to open question(s) <q-ids>; created_new=<q-id|none>; mode=<auto-match|user-confirmed-new>."
   ```
   Re-read the task (Step 0) and continue to Step 1.

<!-- example: anti-pattern -->
**Autonomous mode** (`EPM_AUTONOMOUS_SESSION=1`): on path 5 (no
existing question fits) SKIP the new-question draft entirely — do not
raise `AskUserQuestion`, do not print the proposed question as a text
menu. EXECUTE the skip in this same turn: post `epm:question-linked v1`
with `mode=autonomous-skipped` + `created_new=none` + an empty
`relates_to`, then continue to Step 1 (do NOT end the turn waiting on
user confirmation). The PreToolUse hook hard-blocks the ask if reached;
the nightly /daily living-docs backstop re-synthesis OR a later `/issue <N>`
re-invocation will reconcile the link.
