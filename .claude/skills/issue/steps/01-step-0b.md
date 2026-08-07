# Step 0b: Defaulting & autofill

Step body relocated verbatim from `.claude/skills/issue/SKILL.md`
(#2155). SKILL.md keeps the heading, the state machine and the
Orchestration Procedure router; read this file when the run reaches
this step.

---

Runs only when at least one of {no current folder, missing `type` in
frontmatter, empty body} holds. Goal: get the task into the minimum
shape Step 1 needs without bouncing back to the user just to add
metadata. Order:

1. **Folder missing (legacy / migration case) ->** apply
   `status:proposed` automatically by moving the task to
   `tasks/proposed/<N>/`:
   ```bash
   uv run python scripts/task.py set-status <N> proposed --note "Autofilled by /issue Step 0b."
   ```
   No user interaction. Defaulting an unlabelled task to `proposed` is
   the obvious read of the lifecycle (To do column = `proposed`).

2. **Body empty (or <50 chars of substance) ->** ask the user in the
   <!-- gate: gates.empty_body -->
   <!-- autonomous-mode: block-and-fail -->
   current chat via `AskUserQuestion` for the minimum spec needed for the
   adversarial planner to design the task. The exact prompts depend on
   the task type (see `clarifier.md`); for an unknown type, ask:
   - "What's the goal of this task in one sentence?"
   - "What's the hypothesis or success criterion?"
   - "Is there a parent task or prior result this builds on? (task # or 'none')"
   - "Rough compute size? (small / medium / large)"

   In autonomous mode (`EPM_AUTONOMOUS_SESSION=1`) this gate cannot
   auto-resolve — a missing task body is a content gap only the user
   can fill. Post `epm:failure v1 failure_class: data` (reason:
   `body empty; autonomous mode cannot synthesise spec from title`),
   set `status:blocked`, and exit (halt-criterion #4 — factual question
   only the user knows). The PreToolUse hook in `.claude/settings.json`
   is the runtime backstop and will hard-block the ask if reached.

   Plus **search the codebase + HF + arXiv before drafting** when the
   title hints at pulling existing artifacts (e.g., "use HF model X",
   "replicate paper Y") — list what you found and let the user pick.
   Don't fabricate a body from the title alone.

   Once the user answers, draft a body covering Goal / Hypothesis / Setup
   / Eval / Success criterion / Kill criterion / Compute / Pod preference
   / References (for a representation-mapping task — geometry read /
   predictor / probe / direction extraction over activations — the drafted
   Setup names BOTH mapping arms, prefix-based AND context-based, per the
   CLAUDE.md "Prefix mapping AND context mapping" Critical Rule; a one-arm
   draft states the deviation explicitly), then patch the task:
   ```bash
   uv run python scripts/task.py set-body <N> --file /tmp/issue-<N>-body.md
   ```
   Post a `<!-- epm:auto-defaults v1 -->` event listing what was applied
   (folder moved, body drafted) so the audit trail is durable on the
   task:
   ```bash
   uv run python scripts/task.py post-marker <N> epm:auto-defaults \
     --note "Drafted body from user chat answers; moved to tasks/proposed/<N>/."
   ```

   **Audit-marker placeholder guard (when generating any `epm:audit` /
   `epm:auto-defaults` body):** before posting, run
   `grep -E "(^|\s|>)(TBD|TODO|placeholder|\[X\]|implementer fills)(\s|$|<)"`
   against the drafted body. Match -> BLOCK the post and finish the audit
   instead. The regex catches placeholders mid-line as well as line-start.

3. **`type` frontmatter missing ->** infer from title cue, then confirm
   with the user:
   - Title prefix `Test:` / `Sweep:` / `Train:` -> suggest `experiment`
   - Title prefix `Refactor:` / `Fix:` / `Add:` / `Migrate:` -> suggest `infra`
   - Title prefix `[Batch]:` / `[Workflow]:` / body contains a numbered
     list of >=3 unrelated fixes -> suggest `batch`
   - Title prefix `Analyze:` / `Re-analyze:` -> suggest `analysis`
   - Title prefix `Survey:` / `Read:` / `Lit review:` -> suggest `survey`

   **Fix-validation override (CLAUDE.md § "Routing experiment intent"):**
   a `Test:` cue does NOT default to `experiment` when the Goal is to
   VALIDATE / TEST that a shipped workflow / infra / code fix WORKS (a
   smoke run, an end-to-end "does it work now after the fix", a config /
   pipeline / backend re-check) — that is `kind: infra`, NOT `experiment`,
   because it completes on the test-verdict path and produces NO promotable
   clean-result. Reserve `experiment` for a RESEARCH QUESTION that produces
   a clean-result the user promotes. Litmus: would the result rewrite an
   issue's `## Takeaways` / answer an `open_questions.md` question
   (→ `experiment`), or just confirm the fix is sound (→ `infra`)? When the
   title says `Test:`/`Validate:` but the body reads as fix-validation,
   suggest `infra` as `(Recommended)`. (#672)

   <!-- gate: gates.missing_type -->
   <!-- autonomous-mode: block-and-fail -->
   Use `AskUserQuestion` with the inferred option as `(Recommended)`
   first. Apply via `task.py set-body --file ...` to update the
   frontmatter `type:` line. In autonomous mode
   (`EPM_AUTONOMOUS_SESSION=1`), DO error and EXIT — the type field
   gates Step 7's completion variant and a guess here corrupts the
   lifecycle. The PreToolUse hook hard-blocks the ask if reached.
   Before exiting, post the §5 marker:
   ```bash
   uv run python scripts/post_step_completed.py --issue <N> --step 0b \
     --exit-kind failure-exit \
     --notes "type-frontmatter autofill loop; user override required"
   ```

4. **Other useful frontmatter fields missing** (`compute`, `priority`):
   do not block on these. `compute` will be set in the adversarial-planner's
   reproducibility card; `priority` is user-curated and never blocking.

   Note: legacy `aim:*` GH labels were deleted long ago. New tasks do not
   use them. Topic categorization for new work lives in `docs/claims.yaml`
   (`topic` field) and in `RESULTS.md` / `eval_results/INDEX.md` H2
   prose; no replacement frontmatter field exists.

After Step 0b, re-read the task (re-run `task.py view <N>` from Step 0)
so downstream state is computed from the now-patched task, then continue
to Step 0c.
