# Step 1: Clarifier gate

Step body relocated verbatim from `.claude/skills/issue/SKILL.md`
(#2155). SKILL.md keeps the heading, the state machine and the
Orchestration Procedure router; read this file when the run reaches
this step.

---

If `epm:clarify` marker missing (or user has replied in `comments.jsonl`
but the clarifier hasn't re-checked): read `clarifier.md`, run the
clarifier for this task type, then:

**Before drafting any clarifying question, run the mandatory
context-gathering pass in `clarifier.md` Step 0** — search past
clean-result tasks, `.arxiv-papers/`, `external/`, `RESULTS.md`,
`eval_results/INDEX.md`, and `git log` for information that resolves the
ambiguity. Cut any question already answered by project knowledge;
sharpen the rest by quoting the source. When posting "All clear",
include a brief **Context resolved** bullet list of the
tasks/commits/papers consulted so the inheritance chain is auditable.

- **All clear** (<=1 minor ambiguity) -> post `epm:clarify` with "No
  blocking ambiguities found. Proceeding to adversarial planning."
  Move the task to the `planning` folder:
  ```bash
  uv run python scripts/task.py post-marker <N> epm:clarify \
    --note "No blocking ambiguities. Proceeding to adversarial planning."
  uv run python scripts/task.py set-status <N> planning --note "Clarifier All-clear."
  ```
  This is the one place where the task transitions out of the To-do
  column into the pipeline. Subsequent phases route automatically as
  `task.py set-status` is called at each step.

- **Ambiguities remain** -> do BOTH of the following, in order:

  1. **Post on the task.** Append a `epm:clarify v<n>` event with the
     numbered questions in the `note` body. This is the durable log — if
     the user closes the terminal, the questions are still there in
     `events.jsonl`.

  2. **Ask the user in the current chat (Interactive mode only).**
     Immediately after posting, ask the SAME numbered questions to the
     user in the current session.
     <!-- gate: gates.clarifier_blocking -->
     <!-- autonomous-mode: block-and-fail -->
     Use `AskUserQuestion` for small multiple-choice-style prompts;
     otherwise post a short numbered list as plain text and wait for a
     reply. Do NOT exit yet — give the user the option to answer inline
     so they don't have to context-switch to the dashboard. In
     autonomous mode (`EPM_AUTONOMOUS_SESSION=1`), do NOT ask — post
     `epm:failure v1 failure_class: data` (reason: `clarifier blocking
     ambiguities; autonomous mode cannot resolve`), set `status:blocked`,
     and exit (halt-criterion #4). The PreToolUse hook hard-blocks the
     ask if reached.

  3. **If the user answers in chat:**
     - Post a `epm:clarify-answers v<n>` event with the user's answers
       verbatim (lightly formatted — one numbered bullet per question),
       so the task is self-contained for downstream agents.
     - If the user also asks you to fold the answers into the task body
       (e.g., "update the body"), run `task.py set-body <N> --file ...`
       with the original body preserved + a `## Spec (from clarifier)`
       section appended. Only do this on explicit request — default is
       events-only.
     - Re-run the clarifier evaluation using (body + clarify questions +
       these answers). If no blocking ambiguities remain, advance to
       Step 2 (adversarial planning) in the same invocation. If still
       ambiguous, loop: post a `v+1` clarify event and ask again.

  4. **If the user defers ("I'll answer later", no reply, or says to
     exit):** EXIT with status still `proposed`. User can answer later
     via the dashboard's `comments.jsonl` append path, OR re-invoke and
     answer in chat next time. Before exiting, post the §5 marker:
     ```bash
     uv run python scripts/post_step_completed.py --issue <N> --step 1 \
       --exit-kind parked --notes "clarifier deferred by user"
     ```

**Rule:** never proceed to adversarial planning with >=2 blocking
ambiguities. Tight specs save later backtracking.

**Rule:** the ask-in-chat step is MANDATORY when there are blocking
ambiguities. Posting questions only as events and immediately exiting
forces a context switch the user does not want — always offer the
inline path first.

**Goal-refinement (optional, conditional gate #9).** If the clarifier
notices the existing `## Goal` H2 is fuzzy — e.g. too broad, names
two outcomes, or doesn't actually describe what would change with
the result — it MAY propose a sharper Goal via
`AskUserQuestion` <!-- gate: gates.experiment_goal_refine -->
**IN INTERACTIVE MODE ONLY**. On explicit user consent in the same
turn, run
`uv run python scripts/task.py set-goal <N> "<new goal>" --by clarifier --reason "<one line>"`,
which emits a new `epm:goal-updated v1` marker. Without explicit
consent the Goal stays put. Never call `set-goal` without
in-the-loop user agreement; this is the user's contract field.

<!-- example: anti-pattern -->
**Autonomous mode** (`EPM_AUTONOMOUS_SESSION=1`): SKIP this refinement
entirely per § Autonomous session behavior → `experiment_goal_refine`.
The Goal stays as set at task creation; do not propose a refinement,
do not raise `AskUserQuestion`, do not print the proposed sharper Goal
as a text menu. EXECUTE the skip by continuing to Step 2 in this same
turn; do NOT end the turn waiting on user confirmation. The user owns
the Goal contract; an autonomous session may not silently shift it.
