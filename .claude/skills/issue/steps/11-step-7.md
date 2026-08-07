# Step 7: Monitor -> results

Step body relocated verbatim from `.claude/skills/issue/SKILL.md`
(#2155). SKILL.md keeps the heading, the state machine and the
Orchestration Procedure router; read this file when the run reaches
this step.

---

Under the new orchestrator-owned polling model (Step 6d.2), three event
sources contribute to `running`-phase progress:

- **Experimenter (subagent, single turn at launch)**: posts
  `epm:run-launched` once and exits.
- **`poll_pipeline.py` (run by the orchestrator's bg-Bash loop)**: posts
  `epm:progress` on each phase transition observed in the pod log.
- **Entry script on the pod**: writes `[phase=done]` to its log on
  graceful completion AND writes a JSON sentinel file at
  `/workspace/logs/issue-<N>-results.json` containing the
  `epm:results` payload. The orchestrator's polling-loop terminal
  tick (Step 6d.2) reads the sentinel on its next poll and posts
  the `epm:results` marker from the local VM via `task.py post-marker`. The
  pod NEVER calls `task.py` directly — enforced by
  `tests/test_no_pod_side_task_py_shellout.py` and the CLAUDE.md
  "Pod-side code NEVER shells out to scripts/task.py" rule (#397; the
  same failure class applies to `task.py post-marker`, hence the
  sentinel-file pattern is canonical).

  Sentinel format (JSON object with these keys, all required):
  - `eval_numbers` (inline dict of final eval metrics)
  - `eval_paths` (list of repo-relative paths to eval result JSONs)
  - `reproducibility_card` (dict matching CLAUDE.md template; filled in
    with TBD → resolved values. **For training / sweep runs the card
    MUST carry the machine-resolvable fields
    `scripts/verify_uploads.py` self-resolves** (`merged_results_card`
    → `check_hf_model_from_card` / `check_wandb_from_card`):
    `adapter_paths` as an explicit per-cell mapping of REAL HF
    subfolder paths — every value existence-checked under
    `hf_model_repo` (defaults to the canonical model repo; declare only
    when different; cells whose adapters land in a DIFFERENT repo than
    `hf_model_repo` — the #1108 overflow split — additionally declare
    `adapter_repo_overrides`, a per-cell `{cell_id: repo_id}` dict
    keyed on `adapter_paths` cells, and the verifier resolves those
    cells against the override repo, #1664), so NO
    `<arm>`/`<source>`/`<seed>`-style template
    placeholders and no `(16 adapters)` prose summaries — plus
    `wandb_project` AND `wandb_run_names` (per-cell dict or list of run
    display names; a single run may instead declare `wandb_run_path`).
    Prose may accompany but NEVER replace these structured fields: a
    prose-template card (`adapters/issue_<N>/<arm>/<source>_seed<S>
    (16 adapters)` + a free-text `wandb:` line) resolves to nothing and
    trips false `hf_model` / `wandb_run` MISSING rows on a
    fully-uploaded sweep that the upload-verifier must then supersede
    row-by-row — #612. A results RE-post (resume pass,
    crash-fix relaunch, final re-post) must re-declare the structured
    fields in full or OMIT them entirely (the verifier's merge falls
    back per field to the older declaration) — never substitute a prose
    pointer like `"unchanged from the v1 results marker"`; the merge
    bypasses a non-structural value in favor of an older structural
    declaration (#1489).)
  - `wandb_url` (string)
  - `hf_hub_url` (string)
  - `worktree_path` (string, absolute path on local VM)
  - `final_commit_sha` (string, 40-char SHA)
  - `gpu_hours_used` (float)
  - `gpu_hours_budgeted` (float)
  - `plan_deviations` (list of `{deviation: <str>, rationale: <str>}`)

  **Orchestrator-composed fallback.** When the driver emits only
  granular per-cell / per-shard sentinels (no single results sentinel)
  and the orchestrator composes the `epm:results` payload itself
  from the drained pieces, the composed payload obeys the SAME contract
  above — in particular the `reproducibility_card` structured-field
  requirement. Composing the card's adapter / WandB info as prose is
  the #612 failure mode; assemble the explicit `adapter_paths` mapping
  and `wandb_project` + `wandb_run_names` from the per-cell sentinels
  instead. (GCP-lane driver sentinels that declare
  `production_provenance.<cell>.hf_adapter_subfolder` /
  `.wandb_run_name` are already self-resolvable — `verify_uploads.py`
  synthesizes the card from them (#599) — so carry that structure
  through verbatim rather than flattening it to prose.)

When this skill is re-invoked in `running`:

1. Check `epm:results` exists. If not, show last progress, post the §5
   marker:
   ```bash
   uv run python scripts/post_step_completed.py --issue <N> --step 7 \
     --exit-kind parked \
     --notes "experimenter still running; epm:results not yet posted"
   ```
   and EXIT. **If the most recent `epm:progress` event is older than 4
   hours and there is no `epm:results` or `epm:failure`, post
   `epm:stale v1` asking the user to investigate (the experimenter may
   have crashed silently); leave status at `running`.**
2. If `epm:failure` posted: route via the **failure classifier**. The
   `epm:failure` body SHOULD include a `failure_class: infra | code | data`
   field on its first non-blank line. A `data` class (a factual gap only
   the user can fill) is posted per the halt-criterion contract together
   with `status:blocked`, so it never reaches this step — the table below
   routes `infra | code` only:

   | failure_class | Cause example | Action |
   |---|---|---|
   | `infra` | OOM, ENOSPC, NCCL, vLLM init failure, SSH refused, 401/gated repo, library traceback (vllm/transformers/peft/trl/torch/xformers), a zombie-GPU-allocation stall (`stall_reason: vllm_worker_dead_zombie_gpu`, #664) | Re-spawn the **experimenter** on the SAME branch, post `epm:experimenter-respawn v<n+1>`. NO implementer round. Cap 3 respawns; on 4th, status -> `blocked`. (Zombie-GPU stall: see the recovery-brief note below.) |
   | `code` | Python `Traceback` from `src/explore_persona_space/` or `scripts/` (our code), `AssertionError`/`TypeError`/`KeyError` from our code, CUDA OOM listing 2+ sibling `Process <pid> has <X> GiB memory in use` entries (parallel fan-out cells co-located on one device — GPU-pinning bug, #557) | Status back to `running` (implementing sub-phase), re-spawn `experiment-implementer` with the failure context. Loop through Steps 4b -> 5 -> 6 again. Cap 3 (existing). |

   *Before applying either row, the Crash-fix circuit-breaker below checks for
   a same-signature repeat or a spent escape ladder and pivots to re-planning if
   either fires.*

   *Either row's respawn, when its round ends in a successful relaunch
   (fresh `epm:run-launched`), also triggers the stale-`blocked` reconcile
   rule ("A successful relaunch also reconciles a stale `blocked`", Step
   6d.2 poll-loop section) — a task parked `blocked` by an earlier failed
   round must not stay `blocked` through a healthy relaunched run (#742).*

   *`code`-row relaunch contract (#779):* the post-review relaunch — the
   Step 6 experimenter respawn (brief carries `fix_sha=` + the element-5
   stale-artifact disposition, copied from the implementer's fix-engaged
   declaration) or an orchestrator hot-fix relaunch — enforces BOTH
   before dispatch: the fix-commit ancestry probe and the declared
   disposition (`.claude/rules/crash-fix-rounds.md` § Crash-fix
   relaunch: fix-commit ancestry + stale-checkpoint hygiene).

   **Zombie-GPU stall recovery brief (`stall_reason: vllm_worker_dead_zombie_gpu`).**
   When the `status=stalled` tick's `stall_reason` is
   `vllm_worker_dead_zombie_gpu`, the experimenter respawn is an `infra`
   row (the classifier routes it via `--stall-reason`), but the generic
   respawn brief is NOT enough: the orphaned `VLLM::EngineCore` worker
   holds VRAM under a cmdline of just `VLLM::EngineCore` (no script name),
   so a routine `pgrep -f <script>` / `pkill -f <dispatcher>.py` reaper
   MISSES it (#664 r8). The respawn brief MUST instruct the experimenter
   to reap the orphan by EXACT PID before relaunching on the same pod —
   `pgrep -af '^VLLM::EngineCore'` → `kill -KILL <pid>` each →
   `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`
   to confirm the VRAM released. The canonical recipe lives in
   `.claude/rules/gotchas.md` (crash-orphan `VLLM::EngineCore` entry) and
   the long-form runbook
   `.claude/agent-memory/experimenter/feedback_vllm_zombie_gpu_pkill_reaper.md`;
   reference BOTH in the brief so the experimenter does not re-derive it
   (the experimenter's own Pre-Launch step 9 also runs this probe).

   **Crash-fix circuit-breaker (runs BEFORE applying the cap-3 routing
   table above).** Before re-spawning the experimenter (`infra` row) or
   re-spawning `experiment-implementer` (`code` row), check whether this
   failure is the SAME trap re-tripping or a SPENT escape ladder — either
   case means relaunching is futile and the PLAN, not the code, needs to
   change. The check reads ONLY `events.jsonl` markers + the latest
   `plans/plan.md` (no new in-memory state) and is the pure predicate
   `task_workflow.circuit_breaker_should_fire(events, plan_text, K)`
   (canonical predicate this step implements; the canonical pivot rule is
   `workflow.yaml § pivot_criteria.plan_contradiction_replan`):

   ```bash
   K="${EPM_CIRCUIT_BREAKER_K:-4}"   # default 4; one round past cap-3 so
                                     # the generic pivot can also have fired
   uv run python - "$N" "$K" <<'PY'
   import sys
   from explore_persona_space import task_workflow as tw
   n, k = int(sys.argv[1]), int(sys.argv[2])
   events = tw.list_events(n)
   plan = tw.find_task_path(n) / "plans" / "plan.md"
   plan_text = plan.read_text() if plan.exists() else ""
   fire = tw.circuit_breaker_should_fire(events, plan_text, K=k)
   print(fire if fire else "NO-FIRE")
   PY
   ```

   - **Trigger 1 — same-failure-class repetition** (the narrower complement
     to cap-3, K default 4): the predicate groups `epm:failure` markers by
     their `(phase, failure_class, assert_tag)` signature (`phase` from the
     failure note's `phase=<p>` token; `failure_class` from the
     `failure_class:` line or, absent it, from the prior round's classifier
     verdict already recorded; `assert_tag` from the fallback chain — explicit
     `assert_tag:` SHOULD field, else the bracketed `[<tag>-assert]` token,
     else the exception-type / command-family token
     `<ExcName>:<script_basename>` extracted from the crash note, else a
     normalized note-hash with timestamps / PIDs / `file:line` / the
     subprocess argv array / `--flag value` runs stripped). K or more rounds
     sharing ONE signature → fire with `trigger: same_failure_class`. The
     counter RESETS at any intervening `epm:experiment-implementation` /
     `epm:results` milestone marker (a genuinely successful round means the
     trap was escaped, not re-tripped). It does **NOT** reset on
     `epm:progress` — that marker is the workflow's catch-all heartbeat /
     phase-tick / watcher-respawn breadcrumb and is posted DURING a
     still-failing trap window (verified on #664: the trap window between
     events 228 and 247 carries six benign `epm:progress` markers), so
     resetting on it would make this trigger inert.
   - **Trigger 2 — enumerated-fallback-exhaustion**: the predicate parses the
     latest `plan.md` for a finite escape ladder — a literal ` → `
     arrow-separated "Option A → Option B → ..." run OR a numbered
     "Option N:" list under a §-heading — then scans `epm:progress` /
     `epm:experiment-implementation` notes for which Option each round
     attempted AND `epm:failure` markers re-tripping the SAME gate. When
     EVERY option in the ladder has been launched AND the same gate still
     trips → fire with `trigger: enumerated_fallback_exhausted`. The predicate
     silently NO-OPS (returns no trigger-2 fire) on free-form plans with no
     parseable ladder.

   On fire the predicate returns a dict whose **`pivot_scope` field is the
   ready-to-pass `/adversarial-planner` scope string** (built verbatim per the
   wording template in
   `workflow.yaml § pivot_criteria.plan_contradiction_replan`). The
   orchestrator (this is a STRATEGY PIVOT per that canonical predicate):

   1. Post `epm:strategy-pivot v<n>` (the EXISTING marker — do NOT introduce a
      new kind) naming which trigger fired, the matched signature OR the
      spent-ladder list, and the `pivot_scope` string to pass to the planner.
   2. `uv run python scripts/task.py set-status <N> planning`.
   3. Re-invoke `/adversarial-planner` passing `fire["pivot_scope"]` VERBATIM
      as the pivot scope (it already names the repeated signature or the
      exhausted ladder, per the
      `workflow.yaml § pivot_criteria.plan_contradiction_replan` template).
   4. Treat the revised plan as a FRESH implementer cycle (cap-3 / revision
      counters reset — identical to the existing `plan_contradiction_replan`
      pivot).
   5. Count this as ONE of the ~3 strategy pivots before BLOCK (the SAME
      counter as the existing trigger, NOT a separate one). Block only after
      ~3 such re-plans fail to yield a runnable design AND no further
      autonomous angle exists.

   When the predicate returns `NO-FIRE`, fall through to the classifier
   invocation and cap-3 routing table below unchanged.

   **Missing `failure_class` — invoke the classifier script.** Do NOT
   reason about regex patterns inline; the patterns are owned by
   `scripts/failure_classifier.py` and reading them yourself drifts.
   Instead, shell out:

   ```bash
   # Pipe the failure body via stdin to avoid shell-quoting traps.
   # On a status=stalled tick, ALSO forward the poll JSON line's
   # stall_reason via --stall-reason: a known reason (e.g.
   # vllm_worker_dead_zombie_gpu) routes infra directly, because a silent
   # hang's log tail matches no infra pattern. Omit the flag when there is
   # no stall_reason (status=dead, or a stall_reason of null/absent).
   cat <(uv run python scripts/task.py view "$N" --json \
       | jq -r '.events[] | select(.kind == "epm:failure") | .note') \
     | uv run python scripts/failure_classifier.py --body - \
         --log "$LATEST_LOG_PATH" \
         ${STALL_REASON:+--stall-reason "$STALL_REASON"}
   ```

   The script writes a single line — `infra` or `code` — to stdout.
   Treat that as the verdict and apply the corresponding row of the
   table above. If the script exits non-zero, treat as `code`
   (conservative) and post `epm:failure-classify-error` with the stderr
   captured.

   The Python module
   [`scripts/failure_classifier.py`](../../../scripts/failure_classifier.py)
   is the SINGLE source of truth for the regex pattern list.
   `.claude/skills/issue/failure_patterns.md` is a human-readable
   mirror of the same patterns (kept in sync; consult it for review or
   when extending — but do NOT consult it at runtime). To add a new
   pattern, edit `failure_classifier.py` AND the markdown mirror; the
   tests in `tests/test_failure_classifier.py` cover the behaviour.

   **Failure-lesson capture (fires when a crash-fix round RESOLVES the
   failure OR CONFIRMS the true root cause).** A lightweight in-flight
   hook, not a new pipeline step;
   auto-continue, no gate. Both crash-fix shapes — the `code`-row
   `experiment-implementer` round and the `infra`-row experimenter
   respawn whose relaunch applied a fix — are REQUIRED (by
   `.claude/rules/crash-fix-rounds.md` § "Crash-fix rounds: failure-lesson
   block" and `experimenter.md` § "Failure-lesson block on
   relaunch-with-fix") to end their report with a structured lesson
   block. A THIRD shape arrives outside this step: an experimenter that
   fixed a dying launch within its own turn and relaunched (no
   `epm:failure` posted) appends the same block to its Step 6d launch
   report — on receiving such a launch report, apply the same three
   orchestrator actions below. The block:

   ```
   <!-- epm:failure-lesson v1 -->
   failure_class: code|infra|data
   phase: <pipeline phase or script>
   lesson: <1-3 sentences: the trap + the fix, written for the NEXT agent>
   generalizes: yes|no   # yes only if the trap plausibly recurs beyond this issue
   owning_agent: experiment-implementer|experimenter
   gotcha_candidate: yes|no  # yes for codebase/infra traps that belong in .claude/rules/gotchas.md
   root_cause_confirmed: yes|no  # yes if THIS round identified the TRUE root cause (even if a NEW distinct failure followed or the pod was abandoned in recovery)
   supersedes:           # OPTIONAL: prior-lesson slug or marker-ts this lesson corrects; omit if none
   <!-- /epm:failure-lesson -->
   ```

   **Capture eligibility (added #712).** Decide whether a received block
   is eligible for capture by calling the pure predicate
   `task_workflow.failure_lesson_capture_eligible(block_fields,
   subsequent_distinct_failure=<bool>)`. It returns True when the block
   RESOLVED the failure (the original trigger, `resolved: yes`) OR the
   block carries `root_cause_confirmed: yes` (case ii — true REGARDLESS of
   whether a subsequent distinct failure followed or the pod was abandoned
   in recovery). **Root-cause-confirmed firing.** Case (ii) closes the
   #664 gap: at #664 event L204 a crash-fix round CONFIRMED that pod-664
   reproducibly deadlocked the first `llm.generate()` regardless of batch
   size (a pod-hardware cause, NOT a code bug — the OLD pod ran the same
   code fine), but the round ended in a recovery pivot that terminated the
   pod, so the resolve-only trigger never fired and the failure-lesson
   hook captured NOTHING for the confirmed cause. WHO posts it: the agent
   (experiment-implementer / experimenter) that confirmed the cause emits
   the block in its report — case (ii) is signalled by
   `root_cause_confirmed: yes`. The orchestrator posts it verbatim on
   receipt, exactly as for the resolve case; it does NOT wait for a
   successful relaunch.

   On receiving a crash-fix report carrying a capture-eligible block, the
   orchestrator takes three actions:

   1. **Post the marker.** Post the block verbatim as
      `epm:failure-lesson v1` on the task (`task.py post-marker <N>
      epm:failure-lesson --note '<block>'`). This fires for
      `generalizes: no` too — for one-offs the marker alone is the
      durable record (NO memory write).
   2. **On `generalizes: yes` — persist to agent memory IMMEDIATELY.**
      Append a `feedback_<slug>.md` entry (standard agent-memory
      frontmatter + the lesson body) to
      `.claude/agent-memory/<owning_agent>/` plus a one-line
      `MEMORY.md` index entry, then commit BY EXPLICIT PATH on `main`
      from the repo root and push (auto, no approval gate — the same
      standing rule as workflow fixes). Push BARE —
      `git push origin main || uv run python scripts/sync_repo_root.py` —
      never piped (Step 10d § "Bare push / merge snippets";
      sync_repo_root exit 0 can mean in-flight — landing not guaranteed,
      see the canonical block's caveat). The point is
      same-day cross-session sharing: a sibling session's next agent
      spawn loads the memory within minutes, instead of waiting for the
      nightly `/daily` sweep (#537/#545 re-hit overlapping failure
      classes hours apart with no persistence
      channel). Lessons are written for the NEXT agent — 1-3 sentences,
      the trap + the fix, no transcript dumps.

      **`supersedes:` handling + apply (added #712).** Apply the
      capture-eligible block by calling the pure composer
      `task_workflow.apply_failure_lesson(block, durable_texts,
      new_lesson_ref)`, where `durable_texts` is `{path: current_text}`
      for the candidate durable files (the `owning_agent`
      `feedback_<slug>.md` body keyed at `new_lesson_ref["memory_path"]` +
      any matched `.claude/rules/gotchas.md` bullet) and `new_lesson_ref`
      is `{"slug": "<new-lesson-slug>", "task_id": "<N>", "memory_path":
      "<feedback path>", "lesson": "<lesson body>"}`. The composer returns
      the FINAL `{path: text}` map the orchestrator then writes
      (explicit-path commit + push, as today). Its behavior:
      - If the block carries `supersedes: <prior-slug-or-marker-ts>`, the
        composer calls `supersedes_action()` to locate every durable entry
        whose text matches `<prior-ref>` and PREPENDS a concrete
        `[SUPERSEDED by <new-lesson-slug> — see #<N>] ` marker (the real
        slug + task id, NEVER a `<pending>` placeholder), then APPENDS the
        new (corrected) lesson body to the `owning_agent` memory file — so
        the corrected lesson LANDS ALONGSIDE the annotated prior, never
        replacing it. Transitive chains are kept (A `[SUPERSEDED by B]`, B
        `[SUPERSEDED by C]`, C live); each correction annotates only the
        entry its `supersedes` directly names.
      - If `<prior-ref>` resolves to NOTHING, the composer leaves all prior
        texts byte-unchanged, appends the new lesson normally, and the
        orchestrator logs `supersedes_unresolved: <prior-ref>` in the
        marker note — a dangling `supersedes` is a no-op annotation, NEVER
        a hard failure (a lesson always lands).
      - If `supersedes` is ABSENT, the composer is a pure append (the
        produced text is byte-identical to the pre-#712 append-only path).
   3. **On `gotcha_candidate: yes` — route as a workflow-fix
      candidate.** Treat the lesson as a prose workflow-fix candidate
      targeting `.claude/rules/gotchas.md` and route it through the
      existing workflow-fix-on-bug auto-file default — a filed
      `kind: infra` task + a background `/issue --auto` session
      (`.claude/rules/workflow-fix-on-bug.md`); the lesson block is the
      surfaced prose.

   If the resolving report omitted the block (older agent spawn, or a
   refusal killed the report tail), reconstruct it from the failure
   context + fix diff yourself before posting — don't bounce the round
   for the missing block alone. `scripts/consolidate_lessons.py` (a cron,
   NOT `/daily` — task #711) is the deterministic deduplicating
   consolidator: it reads the rolling-window `epm:failure-lesson v1`
   markers, dedupes against agent memories, promotes recurring lessons into
   `.claude/rules/gotchas.md` or the relevant rule file, and prunes
   over-eager `generalizes: yes` memory entries. `/daily` no longer owns
   this pass.
3. If `epm:results` exists, move status to `uploading` and proceed to
   Step 8.
