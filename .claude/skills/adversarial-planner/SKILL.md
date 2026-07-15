---
name: adversarial-planner
description: >
  Multi-agent plan-critique-revise loop for big changes. Use when making significant
  architectural decisions, designing new experiments, or planning multi-file changes.
  Spawns a Planner agent, then a Critic agent to find flaws, then the Planner revises.
  After implementation, spawns an Implementation Critic to verify correctness.
  Produces a battle-tested plan AND a verified implementation.
user_invocable: true
---

# Adversarial Planner

When the user invokes `/adversarial-planner` or when you're about to make a big change (new experiment, architectural refactor, multi-file changes), use this multi-agent workflow instead of planning alone.

## When to Use

- New experiment design (hypothesis, conditions, controls, eval)
- Architectural changes affecting multiple modules
- Pipeline changes (training, eval, data processing)
- Any change touching >5 files or >200 lines
- Experiment proposals that will consume significant GPU time

## The Loop

### Phase 1: Plan (Planner Agent)

Spawn an Agent with this role:

```
You are the PLANNER. Your job is to design a concrete, detailed plan for the following task:

[TASK DESCRIPTION]

**Canonical Goal (current at spawn; re-read before returning — planner.md
§ Goal-currency guard):** [GOAL TEXT or "no goal: frontmatter — use the
body's ## Goal H2"]

**If this is a `type:batch` issue (the body lists N independent items):**
Structure your plan as N independent sections, one per body item. Each
section gets its own subset of the fields below — Goal, Design (with file
paths and pseudocode), Acceptance criteria, Risks. Skip cross-item
narrative; items are independent. The Assumptions section can be shared,
but call out which assumption belongs to which item if it isn't obvious.

**Before planning, search the web** for how this type of task is typically done. Look for:
- Published papers, blog posts, or repos with similar experiments or architectures
- Established best practices, common pitfalls, standard baselines
- Existing tools, libraries, or pre-computed artifacts you can reuse
- **Hyperparameter recipes for the closest published setting** — the exact
  lr / schedule / batch / epochs / LoRA config reported in the setup or
  appendix table of the nearest paper (same model family + task), AND the
  values any parent / sibling issue already validated. Choose each
  load-bearing hyperparameter to serve this experiment's Goal, quote real
  values (not your memory of them), and carry each into §11 Decision
  Rationale with a `Source:` (arXiv id or issue `#<M>`). See planner.md
  "Before Planning" step 4 + §11 for the full grounding protocol.

Then design your plan:
1. **Goal**: What are we trying to achieve and why?
2. **Prior work**: What did your web search find? What approaches exist and how does this plan relate?
3. **Hypothesis** (if experiment): What do we expect and what would falsify it?
4. **Design**: Concrete steps, file paths, function signatures, configs
5. **Controls**: What comparisons make the results interpretable?
6. **Eval**: How do we measure success? What metrics, what thresholds? Name the hero figure(s) the headline needs AND a short exploratory dump for the analyzer to pick from — over-produce by default (see `planner.md` §6 "Figures to produce").
7. **Risks**: What could go wrong? What are the failure modes?
8. **Resources**: GPU time, disk space, API costs, wall time estimates

9. **Assumptions**: List EVERY factual assumption you are making. Be exhaustive. Include:
   - API/library capabilities ("vLLM supports X", "speculators can do Y")
   - Specific values ("the canonical layer is 32", "hidden_dim is 5120")
   - Infrastructure ("the model fits on one GPU", "the data is cached")
   - Compatibility ("this torch version works with that library")
   For each assumption, state your confidence (high/medium/low) and how you verified it (searched web, read docs, guessed).

Be specific — name files, write pseudocode, specify hyperparameters with a literature / past-issue `Source:` for each load-bearing one (see planner.md §11). Vague plans waste GPU time.
```

Save the plan to a temporary file or pass it directly.

**Goal-currency gate (pre-persist; EVERY `new-plan-version` call).** Capture
the Goal snapshot when you spawn the planner —
`GOAL_SNAP="$(uv run python scripts/task.py view <N> --json | jq -r '.frontmatter.goal // empty')"`;
when that is empty (no `goal:` frontmatter — `kind: infra | batch | survey`),
fall back to the body's `## Goal` H2 text (`jq -r '.body'` + the H2 slice), so
the gate is non-vacuous on body-Goal tasks too. Inline the snapshot in the
brief (template above). Immediately BEFORE every
`task.py new-plan-version` persist (Phase 1 initial draft, Phase 1.5.0
mechanical-bounce redrafts, Phase 3 revisions), re-read the same field and
compare to the snapshot (plain text equality). On ANY difference — the user
amended the Goal while the draft was in flight (#922: two `epm:goal-updated`
amendments landed mid-draft; plan v3 persisted quoting the superseded Goal
and auto-approved 3 s later) — OR when NO snapshot was captured at spawn
(missing/empty on a task that has a Goal: treat as a mismatch — re-read
twice, never persist on an unverifiable snapshot) — do NOT persist:
re-spawn the planner with the
amended Goal + the draft path as a MECHANICAL redraft bounce (same semantics
as a Phase 1.5.0 bounce; does NOT count against the Phase 3 round cap),
refresh `GOAL_SNAP`, then persist the redraft. A goal-currency WARN from
verify_plan.py at Phase 1.5.0 (`c23_goal_currency`) is the same bounce
trigger — the one WARN that bounces instead of riding into the critic
briefs.

**Strip the harness trailer before persisting.** An `Agent` tool result ends
with harness-appended metadata — a final `agentId: <id> (use SendMessage ...)`
line plus a `<usage>...</usage>` block. Remove BOTH before writing the
planner's return to ANY durable handoff surface (the
`/tmp/issue-<N>-plan-v<K>-<attempt>.md` handoff file, where `<attempt>` is a
fresh `$(date +%s)` chosen once per orchestrator planning attempt — the
`<attempt>` suffix exists because a crashed attempt leaves a stale /tmp file;
a respawned session re-Writing the fixed path after Reading an older version
gets "File has been modified since read" (4× on #822) —
`task.py new-plan-version` → `plans/v<K>.md`), e.g.:

```python
text = re.sub(r"\n?agentId:\s*\S+\s*\(use SendMessage.*?</usage>\s*$", "\n", text, flags=re.DOTALL)
```

A contaminated handoff file reaches every downstream consumer verbatim
(fact-checker, all 6 critics, the committed plan revision) — on task #562
(2026-06-10) both Codex critic twins had to strip the trailer independently
because the orchestrator captured the planner's return verbatim.

**De-escape harness HTML entities before persisting.** A BACKGROUND-Agent
result delivered via a `<task-notification>` block arrives with its
`<result>` field HTML-ESCAPED by the harness (`&&` and `<`/`>` become their
amp/lt/gt entity forms — #952 v9, 2026-07-04: the workload command's shell
AND operators arrived entity-escaped and dispatch would not run until
hand-fixed). When capturing a planner return from a notification, prefer
re-extracting the raw text from the notification's `<output-file>` —
output-file text is CLEAN and gets NO unescape. Apply ONE `html.unescape()`
round ONLY when the text you are persisting was sourced from the
notification BODY itself (never to output-file-sourced text: unescaping
already-clean text corrupts legitimate literal entity content — the two
sources are exclusive-or). One round on notification-body text is the exact
inverse of the harness escaping — legitimately-escaped content the planner
wrote arrives double-escaped and round-trips correctly. `verify_plan.py`
check 25 (`c25_html_entities_in_commands`) is the mechanical backstop at
Phase 1.5.0: entities surviving in a fenced command block FAIL the persist
(a `--workload-cmd`/`dispatch_issue.py` fence is never exemptable).

**Extract the output-file text via the transcript recipe (background
`local_agent` tasks).** The `<output-file>` of a BACKGROUND Agent task
(`/tmp/claude-*/…/tasks/<id>.output`) is a SYMLINK to the subagent's
conversation-transcript JSONL, NOT raw text. That format has NO
`{"type": "result", "result": "<str>"}` row — on #1219 (2026-07-10) the first
extraction scanned for one and exited "NO RESULT ROW FOUND" — and the FINAL
row may be a metadata-bearing row (keys like `agentId` / `attributionAgent` /
`attributionSkill`) with no usable text. Canonical recipe: keep the LAST
`type == "assistant"` row whose `message.content` has non-empty
`{"type": "text"}` blocks (this inherently skips trailing non-text rows),
join the text blocks, THEN apply the trailer-strip regex above. Output-file
text is clean: NO `html.unescape()` (previous paragraph). Verify byte count +
head/tail before persisting — that same verify is also the guard for the rare
case where the final output is split across MULTIPLE assistant rows (the
recipe keeps only the last text-bearing row).

```python
import json, re

def last_assistant_text(path):
    last = None
    for line in open(path, encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get("type") != "assistant":
            continue
        blocks = (row.get("message") or {}).get("content") or []
        texts = [b.get("text", "") for b in blocks
                 if isinstance(b, dict) and b.get("type") == "text"]
        if any(t.strip() for t in texts):
            last = "\n".join(texts)
    if last is None:
        raise SystemExit(f"NO ASSISTANT TEXT ROW FOUND: {path}")
    return last

text = last_assistant_text(src)  # src = the <output-file> path
text = re.sub(r"\n?agentId:\s*\S+\s*\(use SendMessage.*?</usage>\s*$", "\n", text, flags=re.DOTALL)
```

### Phase 1.5: Verify Assumptions (Verifier Agent)

**This phase is MANDATORY. Never skip it.**

**Phase 1.5.0 — Mechanical pre-pass (runs FIRST, before the fact-checker spawns).**
Run the structural verifier against the plan version just persisted:

    uv run python scripts/verify_plan.py --issue <N> --json        # task context (newest plans/v{K}.md)
    uv run python scripts/verify_plan.py --plan-file <path> --json # standalone / not-yet-persisted plans
    # Canonical verdict parse — copy this; do NOT improvise one:
    uv run python scripts/verify_plan.py --issue <N> --json | uv run python -c "import json,sys; d=json.load(sys.stdin); print(d['overall'], 'n_fail=%d n_warn=%d' % (d['n_fail'], d['n_warn']), 'failed:', ','.join(c['id'] for c in d['checks'] if c['status']=='FAIL') or 'none')"

- **JSON contract:** the `--json` payload keys are `source`, `issue`, `kind`,
  `overall` (`"PASS"|"FAIL"` — the verdict key; there is NO `verdict` key, and
  `d.get('verdict')` prints `None` — the 2026-07-12 3-session improvised-parse bug,
  #1290), `n_fail`, `n_warn`, `n_skip`, and `checks` (list of
  `{id, name, status, detail}`, status ∈ `PASS|FAIL|WARN|SKIP`). Read the verdict
  with fail-loud `d['overall']` (KeyError on a payload change), never `.get()`
  defaulting to `None`, and never infer PASS from `n_fail` alone. Exit code: 0 PASS /
  1 FAIL / 2 usage error or plan-not-found — the pipe consumes it, so the one-liner
  reads `overall` instead; a missing plan still fails loud in the consumer
  (JSONDecodeError on empty stdin). Pinned by
  `tests/test_verify_plan.py::test_canonical_json_parse_snippet_pinned`.
- **Persistence ordering:** `--issue` mode verifies the newest `plans/v{K}.md`. If the
  just-drafted plan has NOT yet been persisted via `task.py new-plan-version` (the plan
  still lives at the `/tmp/issue-<N>-plan-v<K>-<attempt>.md` handoff file), use
  `--plan-file <handoff>
  --kind <task kind>` instead — and treat an `--issue`-mode exit 2 with "no plans/v*.md" as
  "persist first or use --plan-file", NOT as a bounce.
- **Canonical N/A escape phrases** (quote verbatim in any bounce brief so the planner can
  satisfy a check it is legitimately exempt from — and instruct the planner that the
  plan's own declaration line must be UNWRAPPED plain text at line start (leading list
  markers fine): the backtick-wrapped renderings below are deliberate anti-paste armor
  and are NOT recognized by `verify_plan.py::_standalone_na_declared` (#1238)): Every
  phrase satisfies its check ONLY when written as a standalone declaration line in the
  plan (leading `-`/`>`/`*` list markers tolerated); a phrase quoted mid-sentence — e.g.
  inside a pasted bounce brief — does not count (exception: check 31 uses its
  labeled-line forms) (#1237, #1262).
  `N/A — no behavioral construct`
  (check 2), `N/A — no model training` / `N/A — no training hyperparameters` (check 1),
  `N/A — not a replication` (check 7), `N/A — no artifact reuse` (check 6),
  `N/A — not a behavior-implantation` (check 4 — the implant/marker vocabulary hit is
  incidental or quotes a sibling's design, not this plan's own implantation; a genuine
  implantation plan instead names its contrastive-negative set or a named exemption),
  `N/A — no dry-run smoke` (check 11 — kind: infra|batch plans where a `--dry-run`
  mention is incidental, not the plan's own acceptance smoke), `N/A — no draw battery`
  (check 12), `N/A — no empirical-null gate` (check 13),
  `N/A — no fail-loud acceptance claim` / `N/A — fail-loud claim not test-backable`
  (check 15 — kind: infra|batch plans where the vocabulary hit is bug narration, or
  the target is a doc/prose file no pytest can exercise),
  `N/A — no re-extracted reference arms` (check 16), `N/A — no paired contrast`
  (check 18), `N/A — no held-out predictive DV` (check 19),
  `N/A — no registered verdict lattice` (check 20),
  `N/A — no arity acceptance gate` (check 21 — the flagged grep is not a
  call-arity pass condition; discovery/enumeration greps are fine),
  `N/A — no resume/persist pattern` (check 24 — the resume/persist vocabulary
  hit is incidental or quotes a sibling's methodology, not this plan's own
  long-loop resume predicate),
  `N/A — entities are content, not commands` (check 25 — the fenced entity
  forms are deliberately discussed content, e.g. a plan about entity
  handling, not a command to dispatch; exempts shell-tagged content fences
  ONLY, and only when exactly ONE such fence carries entity hits — with
  several content fences, re-tag them to a non-shell info string (e.g.
  text) instead of shell-tagging them; a `--workload-cmd`/`dispatch_issue.py`
  fence FAILs on entities unconditionally),
  `N/A — basis measured on the routed machine` (check 26 — every §9
  compute-table basis cell is measured on the GPU family the plan's resolved
  intent actually routes to under auto, so no cross-GPU conversion is owed; a
  genuinely cross-GPU basis instead states a per-step scaling rate in the row),
  `N/A — no 7B activation capture` (check 27 — the activation-capture
  vocabulary is incidental or the captured model is well under 7B, so the
  ≥40 GB-HBM sizing rule is out of scope; a genuine ≥7B capture instead books
  capture-7b / lora-7b or a larger-HBM lane, never eval/debug),
  `N/A — no precedent-labeled decision bands` (check 28; British `labelled`
  accepted — no registered fractional band is applied to a plan-cited
  precedent ratio, or the ratio and the band concern different quantities; a
  genuine mismatch instead re-labels the precedent's branch or moves the
  threshold),
  `N/A — no conditional phase on this provision` (check 29 — no §7
  extension/retrain-class gate can add wall-time on the fenced provision; a
  plan with a real conditional phase instead adds its wall cost to the
  fence-reconcile sentence near the max-run-duration declaration),
  `N/A — no multi-field bundle reuse` (check 30 — the `.pt` / tensor-bundle
  vocabulary is incidental, not a reused multi-field bundle; a genuine reuse
  instead names its realized-keys verification: verify_reused_artifact_keys.py,
  an mmap `.keys()` read, or the consumer's own loader),
  `Durability pin: N/A — <one-line reason>` / alias `N/A — no durability pin:
  <reason>` (check 31 — kind: infra|batch plans committing to a
  `.claude/skills/**/SKILL.md` prose edit; the reason tail is mandatory — a
  bare `Durability pin: N/A` still WARNs. A plan that NAMES a pin instead
  writes `Durability pin: tests/test_<file>.py::test_<name>`),
  `N/A — no fit-family phases` (check 32 — the flagged compute-table row is
  not actually a per-cell fit/solve/factorization loop, or the plan has no
  fit-family phases; a genuine fit row instead states its basis as
  `measured <t> s/<unit>`, a `#<M>` measured figure, or `pilot-gated`),
  `N/A — no per-rung checkpoint persistence` / alias
  `N/A — no checkpoint ladder` (check 33 — the checkpoint-ladder vocabulary
  is incidental and NO phase of this plan persists per-rung checkpoints,
  e.g. it reads a parent's existing ladder without training new rungs; a
  genuine ladder plan instead states its retention policy in its
  compute-sizing section — DEFAULT: retain the dose-selected + latest rungs
  only, delete ruled-out rungs BETWEEN rungs; or the justified keep-all
  exception sized at realized per-rung GB with `--boot-disk-gb` declared), and
  `N/A — no verbatim ratcheted-file insertion` (check 34 — the fenced block near a
  `.claude/agents/*.md` / `.claude/rules/LESSONS.md` mention is illustrative, not a
  verbatim insert; a plan that DOES mandate an over-headroom insert instead budgets
  the cap-raise with one line `Ratchet budget: raise <constant>['<file>.md'] to
  <new cap>`), and
  `N/A — no revision-pinned reuse` (check 35 — the 40-hex token near reuse
  vocabulary is a git code SHA or otherwise not an HF revision pin on this plan's
  own reuse; a genuine revision-pinned reuse instead names its revision-scoped
  probe — the `revision=<pin>` kwarg on `list_repo_tree` / `list_repo_files`,
  per named stem).
- **FAIL → bounce to the planner** with the failed-check details (a mechanical-fix
  revision: re-spawn the planner with the FAIL list + the plan path; it patches the
  missing block and the orchestrator persists v{K+1} via `task.py new-plan-version`).
  Mechanical bounces do NOT count against the Phase 3 critic round cap. Cap: 2
  consecutive mechanical bounces — if the same check still FAILs on the third run and
  the orchestrator judges the plan plainly satisfies the requirement in different
  words, proceed anyway (verifier false positive), record `verdict: PASS-with-override`
  + the overridden check ids in the marker note, and emit a workflow-fix candidate
  against `scripts/verify_plan.py`.
- **PASS (with WARNs) → proceed** — EXCEPT a `goal_currency` WARN
  (`c23_goal_currency`), which instead triggers the mechanical redraft
  bounce per § Goal-currency gate above (the one WARN that bounces);
  copy any OTHER WARN lines verbatim into the fact-checker
  brief (and later the critic briefs) as "mechanical pre-pass notes".
- **Post the marker** (VM-side; the adversarial-planner skill always runs in the
  orchestrator session, never on a pod):
  `uv run python scripts/task.py post-marker <N> epm:plan-verify --note '<verdict, n_fail, n_warn, failed/overridden check ids, plan version>'`
  The canonical parse one-liner above prints verdict, n_fail, n_warn, and the failed
  check ids verbatim (plan version = the v{K} just verified). Standalone invocations
  with no task context skip the marker.

The Planner's assumptions are the #1 source of experiment-invalidating errors. Before the Critic even sees the plan, independently verify every factual claim.

Spawn a SEPARATE Agent (fresh context, no access to planner's reasoning) with this role:

```
You are the FACT-CHECKER. Your ONLY job is to verify the factual assumptions in this plan.
You are NOT evaluating whether the plan is good. You are checking whether the facts it
relies on are TRUE.

ASSUMPTIONS FROM THE PLAN:
[PASTE THE ASSUMPTIONS SECTION]

HYPERPARAMETER SOURCES FROM THE PLAN (§11 Decision Rationale):
[PASTE THE §11 What / Why / Source / Alternatives entries for every load-bearing hyperparameter]

HF REUSE ROWS FROM THE PLAN (§10 Reproducibility Card / §12 reuse claims, including any
pinned revision):
[PASTE THE REUSE ROWS — repo id, path/stem patterns, and the revision each row pins, if any]

For EACH assumption AND EACH §11 hyperparameter `Source:`:
1. **Search the web** for the actual answer. Check official docs, GitHub repos, papers.
2. **Read the actual code/config** if the assumption is about the codebase.
3. **State the verdict**: CONFIRMED, WRONG, or UNVERIFIED (couldn't find evidence either way)
4. **If WRONG**: State what the correct fact is, with a source link.
5. **If UNVERIFIED**: Flag it as a risk that needs a smoke test before committing GPU time.

For EACH HF reuse row: verify existence with the Python Hub API (`huggingface_hub`,
never the `hf` CLI). When the row pins a revision, probe AT THAT REVISION, per named
stem/path — `list_repo_tree(repo_id, path_in_repo=<prefix>, revision=<pin>,
repo_type=...)` on the ~1M-file data repo (scoped calls only, gotchas.md #833), or
`list_repo_files(repo_id, revision=<pin>)` on small repos — and require >=1 resolved
file per named stem/pattern. A probe at the default branch does NOT satisfy the check:
existence at `main` does not imply existence at the pin (incident #1345 — 2/4 stems
returned 0 files at the plan's pinned revision after a default-branch probe read
CONFIRMED). State the verdict per stem (CONFIRMED-at-pin / WRONG / UNVERIFIED).

DO NOT trust the plan's reasoning. DO NOT trust your own training data for version-specific
claims (API signatures, library features, default values). SEARCH and READ to verify.

Common traps to watch for:
- "Library X doesn't support Y" — search for recent versions, plugins, workarounds
- "The default value is Z" — read the actual source code or docs, don't guess
- "This model fits in N GB" — calculate from config.json, don't estimate
- "Layer L is the canonical choice" — find the actual paper/repo and confirm
- "This will take N hours" — check against published benchmarks, don't extrapolate
- "lr / epochs / LoRA rank = V because paper P / issue #M uses it" — open the
  cited source (arXiv MCP / `task.py view <M>`). Confirm the value matches AND
  that P / #M's setting (model size, data scale, task) is close enough to
  transfer to this experiment's Goal. A hyperparameter cited to a source that
  reports a different value, or to a setting that does not transfer, is WRONG —
  flag it. A load-bearing hyperparameter marked `ungrounded` is UNVERIFIED —
  flag it for a smoke test before committing GPU time.
- "Artifact X resolves on HF" — when the plan pins a revision, run the probe WITH
  `revision=<pin>`; a default-branch listing proves nothing about the pin (#1345); a
  pin held only in a code constant (zero hex in the plan prose) is YOUR coverage, not
  the mechanical check's — instruction 2 ("read the actual code/config") resolves the
  constant, then probe at it
```

**After the Verifier returns:**
- If ANY assumption is WRONG: fix it in the plan before proceeding to the Critic. A plan built on wrong facts will waste the Critic's time.
- If assumptions are UNVERIFIED: note them as risks. The Critic should evaluate whether they're blocking or can be tested with a smoke test.
- If all CONFIRMED: proceed to the Critic.

### Phase 2: Parallel Critique (3 Lenses × 2 Reviewers — Codex Ensemble)

Spawn **6 critic agents in parallel**: for each of the 3 lenses (Methodology,
Statistics, Alternatives), launch BOTH a Claude `critic` AND a `codex-critic`
(Codex gpt-5.5 via `companion task`). Fresh context for each — no access to
the planner's reasoning or to each other's output. Per-lens disagreement
between Claude and Codex twins is resolved by the `reconciler` agent in
**in-context mode** (no GitHub markers — verdict text printed to stdout). See
`.claude/workflow.yaml § ensemble_review.doubled_steps[critic]` and
`.claude/agents/reconciler.md` § "Two Output Modes".

**Quota-sentinel pre-check first (#1204).** Run the canonical check
(CLAUDE.md § Codex ensemble review). `CODEX_QUOTA_LIVE` → spawn ONLY the
3 Claude lens critics (+ consistency-checker when riding the batch) and
skip all 3 `codex-critic` composer spawns this round; record each lens
as an instant confirmed Codex no-show (single-Claude per the Phase-2
no-show row — no output-file probe) and log one line (+ one
`epm:progress` note on #N when run from /issue Step 2).

**Consistency-checker rides the same spawn batch (when invoked from
`/issue` Step 2).** The orchestrator spawns the `consistency-checker`
agent CONCURRENTLY with the 6 critics (7 parallel spawns in one
message, staggered a few seconds apart per the 429 guidance) — it needs
only the corrected plan + the parent recipe, with no dependency on the
critics' verdicts. Its BLOCK findings are UNIONED with the cross-lens
merged critique handed to Phase 3, so ONE revision round addresses
both; BLOCK / WARN / PASS semantics and the `epm:consistency v1` marker
stay exactly as `/issue` Step 2b defines them — only the scheduling
moved. Standalone `/adversarial-planner` invocations (no task context)
skip it.

**Canonical-rubric anchor — REQUIRED in every Claude critic brief, default
or adapted (#1282).** Each of the three lens templates below carries a
`Canonical rubric:` line naming `.claude/rules/critic-lens-reference.md`
plus the lens's VERBATIM heading (`### Methodology lens` /
`### Statistics & Measurement lens` / `### Alternative Explanations lens`).
When composing an adapted brief for a non-experiment task (an infra /
analysis / workflow-fix translation of the lens question — legitimate and
expected), KEEP that line with the heading byte-verbatim: the translation
ADAPTS the canonical rubric, never replaces it. Incident #1265: an
infra-translated Alternatives brief supplied an inline translation but
cited neither the file nor the heading; the critic's heading grep resolved
no span and it reviewed on brief-inline text alone — the pointer-loaded
rubric (per-item REVISE bars, N/A escapes, incident citations) silently
never loaded. (The `codex-critic` composer is unaffected: it resolves
`lens=<id>` to the canonical subheading from its own spec —
`.claude/agents/codex-critic.md` Step 2.)

**Shared preamble — prepend to each critic's brief before its lens-specific questions:**

```
Before composing your critique, internalize these verdict definitions:

- APPROVE = the experiment will produce interpretable data on the
  research question. Diagnostics, confounds, and alternative explanations
  exist for almost every real experiment, but the downstream pipeline
  (analyzer → interpretation-critic → reviewer → clean-result-critic)
  enforces interpretation discipline using the diagnostics the plan
  already reports. The plan does NOT need a pre-registered gate for
  every concern. If the plan reports the right diagnostics for the
  analyzer to weigh, default to APPROVE and list concerns as bullets
  the analyzer should attend to during interpretation.

- REVISE = the plan is missing something the analyzer pipeline cannot
  recover from. Examples: an essential metric is not computed, a
  control condition that would settle the headline question is missing,
  an infrastructure prerequisite is wrong (pinned library version,
  eval surface mismatch). REVISE means "add this thing to the plan,"
  NOT "add a pass/fail rule about an existing diagnostic."

- REJECT = the design cannot answer the research question even with
  revisions of the kind above. The hypothesis is structurally untestable
  with this method; a different experimental approach is required.

Bias toward APPROVE when the plan is recoverable through analyzer
judgment. Reserve REVISE for missing data / conditions / infrastructure
(NOT missing pre-registered rules). Pre-registered confirmation
conjunctions are an anti-pattern — they crush joint power and produce
spurious Inconclusive verdicts on real signals. Trust the downstream
pipeline.
```

**Critic 1 — Methodology:**
```
You are the METHODOLOGY CRITIC. Evaluate ONLY the experimental design:
Canonical rubric: grep `### Methodology lens` in
`.claude/rules/critic-lens-reference.md` and Read ONLY that span (chunked)
before reviewing — the questions below, and any task-kind translation in
this brief, ADAPT that rubric, never replace it.

1. Is the hypothesis testable with this design?
2. Are controls sufficient to isolate the variable?
3. Are there confounds the analyzer cannot weigh from the reported
   diagnostics? (Confounds that are weighable by the analyzer are NOT
   a reason to REVISE — they are concerns to surface to the analyzer.)
4. Is there a simpler experiment that answers the same question?
5. Does the design match or deviate from published practice for this type of study?
6. Are failure modes identified with fallbacks?
7. Is every load-bearing hyperparameter (lr, schedule, batch, epochs,
   LoRA rank / alpha, weight decay, seq length, optimizer, precision,
   anything novel — full set in planner.md §11) grounded with a
   verifiable `Source:` (paper table or prior issue) whose setting
   transfers to this Goal? Start from the Phase 1.5 fact-checker's
   verdict (CONFIRMED / WRONG / UNVERIFIED); spot-check the source only
   when that verdict looks off. REVISE only when a not-CONFIRMED value is
   also plausibly outcome-changing (would diverge, under-train, or
   truncate the trained completion). See critic.md Methodology lens item 4.

Search the web / arXiv for how similar experiments are typically designed in
published work, including the hyperparameters they report.
Rate (methodology only): REJECT / REVISE / APPROVE.
```

**Critic 2 — Statistics & Measurement:**
```
You are the STATISTICS CRITIC. Evaluate ONLY the measurement plan:
Canonical rubric: grep `### Statistics & Measurement lens` in
`.claude/rules/critic-lens-reference.md` and Read ONLY that span (chunked)
before reviewing — the questions below, and any task-kind translation in
this brief, ADAPT that rubric, never replace it.

1. Are the metrics sufficient to distinguish the hypothesis from alternatives?
2. Are sample sizes / seed counts adequate?
3. Is the eval suite correct and complete?
4. Are the headline statistic, sample size, and CI methodology appropriate?
   (Pre-registered pass/fail thresholds are NOT required — the analyzer
   pipeline assigns confidence based on the reported diagnostics. Only
   flag if the headline metric or CI methodology is fundamentally wrong
   for the question.)
5. Could the experiment produce an uninterpretable result?
6. Do numerical claims in the plan match actual data files in the codebase?

Rate (measurement only): REJECT / REVISE / APPROVE.
```

**Critic 3 — Alternative Explanations:**
```
You are the ALTERNATIVE EXPLANATIONS CRITIC. For EVERY predicted positive result:
Canonical rubric: grep `### Alternative Explanations lens` in
`.claude/rules/critic-lens-reference.md` and Read ONLY that span (chunked)
before reviewing — the questions below, and any task-kind translation in
this brief, ADAPT that rubric, never replace it.

1. What is the simplest explanation that does NOT require the claimed mechanism?
2. Does the plan's design rule out that alternative?
3. What additional control or baseline would be needed to rule it out?
4. What would a skeptical reviewer say about this result?
5. Are there missing comparisons or baselines?

For each alternative explanation, distinguish whether it is fatal (the
design cannot rule it out with any analyzer interpretation) or recoverable
(the analyzer can weigh it descriptively from the diagnostics the plan
already reports). Only fatal alternatives trigger REVISE. Recoverable
alternatives are listed as concerns for the analyzer.
Rate (alternatives only): REJECT / REVISE / APPROVE.
```

**Per-lens ensemble decision (inline in this skill, not an agent):**

After all 6 critics return, for EACH lens independently:

| Claude verdict | Codex verdict | Action |
|---|---|---|
| APPROVE | APPROVE | Lens verdict = APPROVE. |
| REVISE | REVISE | Lens verdict = REVISE. Concatenate findings (dedup exact-same). |
| REJECT | REJECT | Lens verdict = REJECT. Concatenate findings. |
| APPROVE | REVISE/REJECT (or vice versa) | **Disagreement.** Spawn `reconciler` (in-context mode) with brief: `mode: in-context`, role=`critic`, lens=`<lens>`, both verdict bodies, plan_body. Reconciler prints `<!-- epm:plan-critique-reconcile v<n> --> ... <!-- /epm:plan-critique-reconcile -->` to stdout with role-specific verdict (`APPROVE` / `REVISE` / `REJECT` per `.claude/agents/reconciler.md` Step 4 table). Reconciler is required to **preserve REJECT severity** when siding with a REJECT reviewer — it does not silently downgrade to REVISE. Manager parses the printed marker's `**Verdict:**` line directly into `lens_verdict[lens]`. |
| Codex no-show (BLOCKER printed) | (any) | Fall back to single-Claude-critic for this lens this round. Surface a one-line note in the merged critique: "Codex {{lens}} twin no-show this round." |

**Durable-output-first (Phase 2 analogue of /issue Step 5b's
durable-verdict-first rule).** A bg-Bash error exit or a wrapper
Agent-tool error is NOT itself a Codex no-show: the expected output file
(`/tmp/codex-critic-<N>-<lens>-output.md`, or the round-suffixed variant
the composer's dispatch config named) is the durable deliverable — read
it first; only an absent/malformed marker block there synthesizes the
`BLOCKER: codex no-show`. If the WRAPPER (`codex-critic`) errors before
returning its dispatch config, check the conventional prompt file
(`/tmp/codex-critic-<N>-<lens>-prompt.md`): if present AND fresh for
this round (mtime postdating the composer spawn; when in doubt,
recompose rather than dispatch a stale prompt), dispatch
`codex_task.py` against it with the conventional output path rather
than declaring a no-show. A CLAUDE lens critic whose Agent-tool result
errors has NO durable output by design (in-context mode) — re-spawn
that lens critic once before any fallback; never drop the lens. The
in-context RECONCILER is covered too: on a reconciler Agent-tool error,
FIRST parse whatever text it returned for the
`<!-- epm:plan-critique-reconcile v<n> -->` block (a parseable block =
the reconciler returned; use its verdict); if none, re-spawn the
reconciler once; if still none, do NOT adjudicate the disagreement
yourself — adopt the MORE SEVERE of the two lens verdicts as a
fail-safe (biasing toward revision, never toward shipping) and record
the unresolved reconcile in the merged critique handed to the
plan-approval gate. A #1204 sentinel-skip is exempt from this probe —
the composer never ran, so no prompt/output file exists for the round;
the skip itself is the confirmed no-show.

**Cross-lens merge (after per-lens reconciliation):**

- **Overall verdict = worst of the three lens verdicts.** REJECT > REVISE > APPROVE.
- **Concatenate all critique bodies** with lens labels (`[Methodology Claude]`, `[Methodology Codex]`, `[Methodology Reconcile]` if dispatched, then Statistics, then Alternatives). The manager does NOT editorialize.
- **Deduplicate** only exact-same finding flagged by 2+ critics (same issue, same file/line). Keep both if framing differs.
- Present the merged critique to the planner for revision.

The reconciler may NOT add findings beyond what either reviewer raised. Round
counter does NOT increment for reconciler invocations (per-reviewer cap = 5 rounds).

### Phase 3: Revise (Back to Planner Agent or Main Thread)

If the merged verdict is REVISE or REJECT — or the concurrently-spawned
consistency-checker returned BLOCK (its findings are unioned into the
same merged critique; see Phase 2):

1. Read the plan AND all 3 critic reports (with lens labels) AND any
   consistency-checker BLOCK findings
2. Synthesize: which Must-Fix items are valid? Which (if any) does the planner reject?
3. Produce a revised plan that addresses the valid Must-Fix items
   (critic Must-Fix items + consistency BLOCKs together — one union
   revision round, not two serial bounce rounds).

**Default: do NOT re-critique.** Proceed to user approval with the revised
plan + the round-1 critique attached as context. With the
conclusion-changing bar in `critic.md`, round-1 Must-Fix items are concrete
and specific — the planner integrates them and ships. Rounds 2 and 3 of the
critic loop fire only in the narrow cases below, because each extra round
both costs compute AND tends to accrete additions that wouldn't have made
the conclusion-changing bar on their own. The cap is still 5 total
revision rounds in case re-critique IS warranted.

Note (#784): the `critic` site's cap-5 loop terminates at the USER
PLAN-APPROVAL gate — the user is the final critic. It never ships past
or pivots on its own, so the surface-real-residual terminal that the
other three iterating sites gained at cap-5 (code-reviewer / interp /
clean-result) is ALREADY the critic site's behavior: at the cap, the
revised plan + the round-N critique are handed to the user at the
approval gate, consistent with #784's surface-not-ship intent. No new
surface terminal is needed here.

**Re-critique ONLY if any of:**
- The original verdict was REJECT (design fundamentally flawed; the revised
  version is effectively a new experiment that needs fresh review).
- The revision changed the hypothesis itself or the core experimental design
  (not just "added the missing baseline the critic asked for").
- The revision added a new condition / eval / pipeline stage that was not
  in the round-1 plan AND was not requested by a Must-Fix item (i.e., the
  planner introduced new scope on its own).
- The planner explicitly disagreed with a Must-Fix item and chose not to
  address it — the user needs to see what critics say about the planner's
  defense.

Otherwise — the planner addressed the round-1 Must-Fix items, didn't change
the design, didn't introduce un-asked-for scope — go directly to user
approval. The user is the final critic.

If the Critic round-1 verdict was APPROVE outright: proceed to implementation
with no revisions.

## Phase 4: Post-Implementation Review (Implementation Critic Agent)

After implementation is complete, spawn a SEPARATE Agent (fresh context, no access to the implementation process) with this role:

```
You are the IMPLEMENTATION CRITIC. The plan has been implemented. Your job is to
verify the implementation actually matches the plan and is correct.

APPROVED PLAN:
[PASTE THE FINAL APPROVED PLAN]

Your review process:
1. **Read every file that was created or modified** — do not skip any
2. **Compare implementation against plan** — check every item in the plan was addressed
3. **Run verification** — check imports resolve, configs parse, no syntax errors

Critique on these dimensions:
1. **Plan adherence**: Did the implementation actually do what the plan said? List any items from the plan that were skipped, partially done, or done differently.
2. **Correctness**: Are there bugs, logic errors, off-by-one mistakes, wrong defaults, or broken edge cases?
3. **Integration**: Does the new code integrate correctly with existing code? Are imports right? Do config schemas match what the code expects? Are function signatures compatible with callers?
4. **Missing pieces**: Is anything required for this to actually work that wasn't implemented? (Missing data files, uninstalled deps, untested code paths, etc.)
5. **Regressions**: Could the changes break existing functionality? Check backward compatibility.
6. **Hardcoded values**: Are there magic numbers, hardcoded paths, or assumptions that should be configurable?

For each issue found, classify as:
- **BLOCKER**: Must fix before this can be used (crashes, wrong results, broken integration)
- **ISSUE**: Should fix but won't prevent basic usage (edge cases, missing validation)
- **NIT**: Style or minor improvement (naming, comments, formatting)

Rate the implementation: FAIL (blockers found), FIX (issues but no blockers), or PASS (ready to use).
```

If the Implementation Critic returns FAIL:
1. Fix all BLOCKERs
2. Re-run the Implementation Critic on the fixed code
3. Max 2 fix rounds — if still failing, surface to user

If FIX: address the ISSUEs, no need to re-critique unless fixes were substantial.

If PASS: done.

## Implementation Pattern

Use the dedicated subagent types for each phase. Subagents cannot spawn other subagents (Claude Code hard constraint), so this skill (running in the invoking agent's context) must orchestrate each phase sequentially.

```
# In the main thread (manager orchestrates):

# 1. Launch Planner (subagent_type: "planner")
planner_result = Agent(subagent_type="planner", prompt="Design a plan for: {task}...")

# 2. Extract assumptions from planner output, launch Fact-Checker (subagent_type: "planner")
#    Use a planner agent for fact-checking too — it has Read/Grep/Glob/Bash for verification
verifier_result = Agent(subagent_type="planner", prompt="You are the FACT-CHECKER. Verify these assumptions:\n\n{planner_assumptions}")

# 3. If any assumption is WRONG: fix the plan before proceeding
if "WRONG" in verifier_result:
    # Update the plan with corrected facts, then proceed

# 4. Launch 6 critics in PARALLEL (3 lenses × 2 reviewers).
#    All 6 Agent() calls go in a SINGLE message so they run concurrently.
#    Claude critics return the verdict body directly. codex-critic agents
#    are prompt-composers only: they return a dispatch config naming the
#    prompt file + expected output file. This orchestrator bg-dispatches
#    scripts/codex_task.py for each (Step 4b) — codex-critic agents MUST
#    NOT bg-dispatch themselves (CLAUDE.md § "Codex task dispatch": only
#    the orchestrator's direct bg-Bash invocation delivers a real
#    notification when Codex terminates).
# 4-pre. Quota-sentinel pre-check (#1204, CLAUDE.md § Codex ensemble review):
#    if LIVE, skip the three *_codex spawns below (instant no-show per lens);
#    Claude spawns + c_check unchanged.
m_claude = Agent(subagent_type="critic",       prompt="[Methodology lens] Critique:\n\n{corrected_plan}",   run_in_background=True)
m_codex  = Agent(subagent_type="codex-critic", prompt="lens=methodology\nplan_body:\n{corrected_plan}",     run_in_background=True)
s_claude = Agent(subagent_type="critic",       prompt="[Statistics lens] Critique:\n\n{corrected_plan}",    run_in_background=True)
s_codex  = Agent(subagent_type="codex-critic", prompt="lens=statistics\nplan_body:\n{corrected_plan}",      run_in_background=True)
a_claude = Agent(subagent_type="critic",       prompt="[Alternatives lens] Critique:\n\n{corrected_plan}",  run_in_background=True)
a_codex  = Agent(subagent_type="codex-critic", prompt="lens=alternatives\nplan_body:\n{corrected_plan}",    run_in_background=True)
# When invoked from /issue Step 2, ALSO add the consistency-checker to
# this same parallel batch (7th spawn; BLOCK findings union into the
# Phase 3 revise round — see /issue Step 2b for verdict semantics):
c_check  = Agent(subagent_type="consistency-checker", prompt="Plan + related-task markers per /issue Step 2b:\n\n{corrected_plan}", run_in_background=True)
# Wait for all spawns to complete.

# 4b. Pick up each codex-critic's dispatch config and bg-dispatch
#     scripts/codex_task.py to actually run Codex. WITHOUT this step,
#     codex_out[lens] holds the dispatch-config text instead of the
#     verdict marker, the per-lens ensemble decision silently drops to
#     single-Claude-critic, AND the dashboard shows no Codex trace
#     because no marker is ever written. The bug: a codex-critic
#     subagent's `Bash(run_in_background=true)` returns IMMEDIATELY but
#     its bg-completion event has no listener after the subagent exits,
#     so dispatch must happen here in the orchestrator (see
#     scripts/codex_task.py module docstring).
#
#     Each codex-critic returns a structured response with these fields
#     (per .claude/agents/codex-critic.md § Step 5):
#       Prompt file: /tmp/codex-critic-<N>-<lens>-prompt.md
#       Expected output file: /tmp/codex-critic-<N>-<lens>-output.md
#       Marker start tag: <!-- epm:plan-critique-codex v<n> lens=<lens> -->
#       Marker end tag: <!-- /epm:plan-critique-codex -->
#       Codex effort: high
#     If the agent returned `BLOCKER: ...` instead (missing lens, missing
#     plugin, malformed brief), skip dispatch — the Codex no-show
#     fallback in Step 5 will fire.
#     (Agent-tool ERROR ≠ BLOCKER: on an error result, first try the
#     conventional prompt-file path if round-fresh — durable-output-first,
#     see the paragraph after the Phase 2 decision table.)
codex_dispatches = {}  # lens -> (output_file, bg_bash_handle)
for lens, codex_agent_out in (("methodology", m_codex),
                              ("statistics",  s_codex),
                              ("alternatives", a_codex)):
    if codex_agent_out.lstrip().startswith("BLOCKER:"):
        codex_out[lens] = codex_agent_out  # preserved so Step 5 sees the BLOCKER
        continue
    cfg = parse_codex_dispatch_config(codex_agent_out)  # extract Prompt file / Expected output file
    # Bg-dispatch in a SINGLE message so all 3 Codex runs proceed concurrently.
    # The orchestrator continues with other turn-local work; the harness
    # delivers a notification on each bg-Bash exit. End the current turn
    # if no other work is in flight rather than blocking on TaskOutput
    # (anti-pattern per CLAUDE.md § "Orchestrator vs subagent
    # re-invocation").
    codex_dispatches[lens] = (
        cfg["output_file"],
        Bash(run_in_background=True,
             command=f"uv run python scripts/codex_task.py "
                     f"--issue {N} --effort high "
                     f"--prompt-file {cfg['prompt_file']} "
                     f"--output-file {cfg['output_file']}"),
    )
# After ALL bg-Bash calls complete (harness notifications), read each
# output file and extract the marker block between the start/end tags.
for lens, (output_file, _bash_handle) in codex_dispatches.items():
    body = Path(output_file).read_text() if Path(output_file).exists() else ""
    marker = extract_marker(body,
                            start=f"<!-- epm:plan-critique-codex v{round} lens={lens} -->",
                            end="<!-- /epm:plan-critique-codex -->")
    if marker:
        codex_out[lens] = marker
    else:
        # Malformed or empty Codex output → treat as no-show. Step 5
        # falls back to single-Claude-critic for this lens this round.
        codex_out[lens] = f"BLOCKER: codex no-show — empty or malformed output at {output_file}"

# 5. Per-lens ensemble decision (see table above):
for lens in ("methodology", "statistics", "alternatives"):
    claude_v, codex_v = parse_verdict(claude_out[lens]), parse_verdict(codex_out[lens])
    if codex_out[lens].lstrip().startswith("BLOCKER:"):
        lens_verdict[lens] = claude_v   # Codex no-show fallback
    elif {claude_v, codex_v} <= {"APPROVE"}:
        lens_verdict[lens] = "APPROVE"
    elif {claude_v, codex_v} <= {"REVISE", "REJECT"}:
        lens_verdict[lens] = max(claude_v, codex_v, key=severity)  # worst-of-two
    else:
        # APPROVE vs REVISE/REJECT — dispatch reconciler in IN-CONTEXT mode.
        rec = Agent(subagent_type="reconciler", prompt=f"""
mode: in-context
role: critic
lens: {lens}
revision_round: {round}
claude_verdict_body: |
{claude_out[lens]}
codex_verdict_body: |
{codex_out[lens]}
plan_body: |
{corrected_plan}
""")
        lens_verdict[lens] = parse_reconcile_verdict(rec)  # APPROVE / REVISE / REJECT (role-specific; reconciler preserves losing-side severity per .claude/agents/reconciler.md Step 4)

# Cross-lens worst-wins merge:
overall = max(lens_verdict.values(), key=severity)

# If REVISE/REJECT: manager revises plan + re-critiques with another 6-critic pass.

# 6. Present final plan to user for approval
# 7. Execute implementation (subagent_type: "experimenter")

# 8. Post-implementation review (subagent_type: "reviewer" — fresh context)
review = Agent(subagent_type="reviewer", prompt="Verify this implementation matches the plan...")

# 9. Fix blockers if any, re-review if needed
```

**Subagent types for each phase:**

| Phase | Subagent Type | Why |
|-------|--------------|-----|
| Planner | `planner` | Read-only + Bash + arXiv MCP / web search. Reads codebase, searches papers, grounds hyperparameters. |
| Fact-Checker | `planner` | Same tools — reads code/configs AND opens cited arXiv papers (arXiv MCP) / `task.py view <M>` to verify hyperparameter sources. |
| Critic — Methodology (Claude) | `critic` | Read-only + Bash. Fresh context, methodology lens. |
| Critic — Methodology (Codex) | `codex-critic` | Thin Claude wrapper → Codex gpt-5.5 via companion task. Methodology lens. |
| Critic — Statistics (Claude) | `critic` | Read-only + Bash. Fresh context, measurement lens. |
| Critic — Statistics (Codex) | `codex-critic` | Thin Claude wrapper → Codex gpt-5.5. Measurement lens. |
| Critic — Alternatives (Claude) | `critic` | Read-only + Bash. Fresh context, alternatives lens. |
| Critic — Alternatives (Codex) | `codex-critic` | Thin Claude wrapper → Codex gpt-5.5. Alternatives lens. |
| Consistency-checker (∥ critics, /issue-invoked only) | `consistency-checker` | Same Phase-2 spawn batch; needs only the corrected plan + parent recipe. BLOCK findings union into Phase 3 revise (verdict semantics per /issue Step 2b). |
| Codex bg-dispatch (×3, one per lens) | Manager (inline) | Bg-Bash `uv run python scripts/codex_task.py --prompt-file <prompt> --output-file <output> --effort high` for each codex-critic dispatch config returned in Step 4. WITHOUT this step, codex_out[lens] holds the dispatch-config text and the ensemble silently drops to single-Claude per lens. Subagents cannot bg-dispatch (no notification listener after they exit). |
| Per-lens reconcile (on disagreement) | `reconciler` | In-context mode; reads both verdicts + plan, prints binding verdict to stdout. |
| Cross-lens merge | Manager (inline) | Manager merges 3 lens verdicts after reconciliation: worst verdict wins, concatenate critique bodies with lens labels. |
| Revision | Manager (inline) | Manager has plan + 6 critique bodies + reconciler outputs in context. |
| Implementation | `experimenter` | Full read/write/bash for coding and running. |
| Implementation Review | `reviewer` | Read-only adversarial check of the implementation. |

All 6 critics run in **parallel** (6 simultaneous `Agent()` calls in a single
message). Each has its own fresh context and specialized lens prompt. They do
NOT see each other's output. After the 3 codex-critic subagents return their
dispatch configs, the orchestrator bg-dispatches `scripts/codex_task.py` for
each in a single message (3 parallel bg-Bash calls). Per-lens reconciler runs
only on Claude-vs-Codex disagreement and is also in-context (no GitHub
markers). Worst case per round: 6 critics + 3 Codex bg-dispatches + 3
reconcilers = 12 invocations.

**Dispatch ordering guards (both bit on 2026-06-09, #545):** (a) bg-dispatch
`codex_task.py` ONLY after the wrapper's completion notification — and gate
the command itself on the prompt file existing (`test -f "$PROMPT_FILE" &&
uv run python scripts/codex_task.py ...`); dispatching ~39 s after spawning
the composer crashed the helper with `FileNotFoundError` on the not-yet-
written prompt. (b) Read each Codex output file only after the helper's
completion line / `epm:codex-task-completed` marker — premature reads hit
missing files and tempt a fallback to the wrong (stale same-issue) output
file.

**Park order:** the plan-approval park (and the `plan_pending` flip) happens
only AFTER the consistency-checker's FINAL verdict is folded in — never on
its interim ack while its full report is in flight. (The checker is now
spawned concurrently with the Phase 2 critics, so its verdict is normally
already in hand by Phase 3 — but the rule stands on any straggler.) On
2026-06-09 #545
parked ~30 min on an uncorrected plan; the checker's late WARN (a substantive
`max_new_tokens` mismatch vs the executed parent rig) then had to be folded
in as a post-park plan v2.


## Rules

- **Planner, Verifier, all 3 Critics, and Implementation Critic MUST be separate agents** with separate context windows. The whole point is independent review.
- **Never skip the Verifier.** Wrong assumptions propagate through the entire pipeline. The Verifier is the cheapest intervention — 30 seconds of web search prevents hours of wasted GPU time. This was added after the corpus projection incident where wrong layer choice and wrong "vLLM can't do this" claims invalidated the first run.
- **Never skip the Critics.** The 3-lens parallel critique catches more than any single critic. Each lens has structural diversity (different prompts/framings), which research shows outperforms debate or angel/devil formats.
- **Never skip the Implementation Critic.** The Implementation Critic catches what the implementer missed. The implementer is biased toward seeing success.
- **Max 5 revision rounds (planning; the per-reviewer round cap — reconciler invocations don't count), max 2 fix rounds (implementation).** If it's not converging, surface the disagreement to the user.
- **The user has final say.** Present the plan + critique + revision to the user before executing.
- **Log the plan.** Register every plan revision via `uv run python scripts/task.py new-plan-version <N> --file <draft>.md`. This writes `tasks/<status>/<N>/plans/v<K>.md` and updates the `plans/plan.md` symlink. Downstream subagents read through the symlink.
- **Read a Bash-materialized plan copy before Editing it.** When a revision round creates the draft via Bash (`cp plans/plan.md /tmp/...`, a heredoc, or a python writer), the harness requires an in-session `Read` of that file before any `Edit` — firing Edits straight at a just-copied file bounces every one with "File has not been read yet" (7 consecutive bounces in one 2026-07-04 session). Read once, then edit.
