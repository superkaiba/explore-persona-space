# Plan: Restore /issue skill richness + Sagan-card 11-check discipline (markdown)

**Task:** rebuild the verbose `/issue` orchestration spec and re-tighten the
clean-result body discipline while keeping the new local-files substrate
(`tasks/<status>/<N>/` + `scripts/task.py`). Refactor, not research.

Tracking task: #7 `Restore SKILL.md detail + Sagan-card discipline in markdown`.

---

## 1. Goal

What we restore:
- The full ~1500-line `/issue` SKILL.md content depth that lived at commit
  `3a6dbe5e^` — state machine diagram, gate enumeration, Step 0–10d
  procedure with all sub-steps, Step-completed re-entry table, marker-protocol
  contract, error-handling table, cost & safety rails, render_title
  pseudocode, status user-action table.
- `workflow.yaml` content depth — gates, halt_criteria,
  subagent_halt_conditions, markers (full taxonomy with `fields` strings),
  ensemble_review contract, steps (lifecycle map with `entry_status_label`
  for the §5 resume router). The GH-era file is 821 LOC; the current one
  is 200 LOC.
- The Sagan-card **content** discipline (11 mechanical checks) re-expressed
  for **markdown** bodies — every check the legacy HTML verifier ran
  except the ones that were purely about HTML scaffolding (style block,
  `<details>`, `<summary>`).
- The analyzer + clean-result-critic + promote-clean-result + markers
  cross-references that talked about the old discipline now point at the
  new markdown verifier.

What we do NOT restore:
- GitHub-issues control plane (`gh issue *`, labels, project board).
- Sagan HTTP/Postgres state (`sagan_state.py`, `SAGAN_API_TOKEN`, the
  Sagan dashboard URL).
- HTML clean-result format. Sagan-card HTML is grandfathered (existing
  bodies stay readable); new bodies are markdown.
- The `reviewer` / `codex-reviewer` agents — they were deprecated upstream
  on 2026-05-13 and that decision sticks. Their content folds into
  `clean-result-critic` Lens 7 (statistical-framing rule, absorbed
  from the retired reviewer), already in the agent spec today.

The substrate substitution is mechanical: `gh issue` → `task.py`, marker
comments → `events.jsonl` rows, issue body → `body.md`, board column →
`tasks/<status>/<N>/` parent folder, plan cache path
`.claude/plans/issue-<N>.md` → `tasks/<status>/<N>/plans/v<K>.md`.
git worktree / PR flow stays — that part is substrate-independent.

---

## 2. Prior work / context

Three known sizes for `/issue` SKILL.md (verified by `git show ... | wc
-l`):
- `3a6dbe5e^` (May 12 2026, GH-era zenith): **1546 lines**.
- `f88ccc8a` (current task-workflow era): **126 lines**.
- The intermediate 94-line value cited in the task description is from a
  transitional commit between the two; not relevant to this plan.

Three known sizes for `.claude/workflow.yaml`:
- `3a6dbe5e^`: **821 lines** (full enumeration of columns, statuses,
  gates, halt_criteria, subagent_halt_conditions, ensemble_review,
  markers, steps).
- Current `f88ccc8a`+: **200 lines** (substrate config + status enum +
  reviewer_pairs + markers names list, no `fields` per marker, no
  `steps` table, no `gates`/`halt_criteria` enumeration).

Verifiers:
- `scripts/verify_sagan_card.py` (570 LOC) — HTML, 11 checks. Still in
  tree. Used by `clean-result-critic` and by the upstream Sagan dashboard
  for grandfathered bodies.
- `scripts/verify_task_body.py` (363 LOC) — markdown, **6 checks**:
  (1) title confidence tag, (2) four required H2 sections in order,
  (3) TL;DR bullet labels, (4) Repro URL permanence, (5) Details confidence
  matches title, (6) Figure caption ≥10 words. Skips bodies sentinelled
  `<!-- legacy-sagan-card -->`.
- `scripts/audit_clean_results_body_discipline.py` (227 LOC) — anti-pattern
  audit (pre-reg jargon, REJECTED-style verdict caps, Δ-Npp, named tests,
  H_a symbols, letter labels, Bin A/B, condition labels like C1/H2/P3,
  Method A/B, GCG/PAIR). Already in tree and already markdown-compatible
  (regex on prose). No HTML to port.

Substrate:
- `scripts/task.py` (357 LOC): full CLI (view, new, set-status,
  post-marker, list-by-status, list-markers, latest-marker, set-body,
  set-title, set-clean-result, add-tag, remove-tag, promote,
  new-plan-version, find, audit). Backed by
  `src/explore_persona_space/task_workflow`.
- `tasks/REGISTRY.json` index + 31 awaiting_promotion bodies.

Dashboard:
- `dashboard/app/tasks/[id]/page.tsx` renders body.md through
  `react-markdown` + `remarkGfm` + `rehypeRaw` + `rehypeHighlight`. HTML
  passes through (`rehypeRaw`) so legacy Sagan-card HTML still renders
  as authored. Markdown bodies render as markdown. The toggle is the
  `<!-- legacy-sagan-card -->` sentinel via `task.isLegacyHtml`
  (`dashboard/lib/tasks.ts:124`). **No dashboard code changes needed.**

Codex twin agents (`codex-code-reviewer.md`, `codex-interpretation-critic.md`,
`codex-clean-result-critic.md`, `codex-critic.md`, `reconciler.md`) reference
SKILL.md by **step number**, not by anchor (e.g., "Step 5", "Step 9a").
Restoring the step structure means those refs stay valid;
**no codex-agent rewrites needed** as long as section headers in the
restored SKILL.md still read `### Step 5: ...`, `### Step 9a`, etc.
The `codex-reviewer.md` file is already marked DEPRECATED and references
"Step 9b retired" — keep that note as is.

Existing awaiting_promotion bodies (verified — 31 total):
- **22 Sagan-card HTML** (carry `<!-- legacy-sagan-card -->` + `<style>`):
  61, 65, 75, 105, 113, 116, 123, 182, 187, 207, 215, 224, 225, 234, 235,
  276, 311, 337, 354, 368, plus 123, 224 (style=2 means two `<style>`
  blocks).
- **20 Sagan-card HTML** with `<!-- legacy-sagan-card -->` sentinel:
  61, 65, 75, 105, 113, 116, 123, 182, 187, 207, 215, 224, 225, 234,
  235, 276, 311, 337, 354, 368.
- **11 markdown** bodies, split into two sub-classes:
  - **8 conformant** to the new four-H2 shape (`## TL;DR / ## Figure /
    ## Details / ## Reproducibility`): 333, 355, 358, 360, 363, 366,
    369, 370. These pass the current 6-check verifier today.
  - **3 v4-legacy** with `## TL;DR / ## Summary / ## Details / ## Source
    issues` headers (older markdown shape from the pre-Sagan-card era):
    186, 237, 351. These fail the current 6-check verifier on the
    required-H2 check and will continue to fail the 11-check verifier
    for the same reason.

Migration scope is therefore: 20 HTML bodies grandfather as-is (read by
`verify_sagan_card.py`); the 8 conformant markdown bodies need only
verifier re-check (a few may newly fail under the expanded 11-check
rules); the 3 v4-legacy markdown bodies are converted in-place to the
four-H2 shape (revision: dropped the dual-verifier sentinel path —
see Phase E).

---

## 2.5. Revision notes (post-Phase-2 critic review)

Critic verdicts: Statistics APPROVE (Claude + Codex). Methodology REVISE
(both). Alternatives REVISE (both). No Claude-vs-Codex disagreement, so
no reconciler dispatch. Overall verdict REVISE; plan updated in place.

**Pre-flight DONE before any Phase lands:** commit `15dc06fe`
("workflow.py: relax schema for current minimal workflow.yaml") fixes
the Pydantic schema so the current minimal `workflow.yaml` validates
and `uv run python scripts/workflow_lint.py` exits 0 today. Before that
commit, the lint was failing with 35 validation errors — Phase A's
acceptance criterion (`workflow_lint --check-references` exit 0) was
unmeetable. The schema is now permissive (list fields default to empty;
`gates` / `ensemble_review` optional); Phase B will re-populate it.

**Phase B before Phase A is a HARD ORDERING requirement** (was: soft
"recommend"). Phase A's restored SKILL.md will cite
`(see workflow.yaml § gates)` etc.; if Phase B hasn't run, those refs
dangle. Phase A's pre-flight MUST verify `workflow_lint --check-references`
exits 0 before commit.

**Phase B split into B1 + B2** (Alternatives critic): B1 restores the
agent-facing reference data (`gates`, `halt_criteria`,
`subagent_halt_conditions`, `ensemble_review`, full `markers` with
fields) — required because SKILL.md cross-refs them. B2 restores the
`steps` table with `entry_status_label` — **the R2 grep confirmed
B2 is REQUIRED** (three consumers: `post_step_completed.py`,
`workflow_lint.py`, `orchestrate/resume.py`). Both B1 and B2 land
in Phase B.

**Phase E simplified** (Alternatives critic): the 3 v4-legacy markdown
bodies (#186, #237, #351) are converted in-place to the four-H2 shape
instead of sentinel-routed to `verify_clean_result.py`. Net: drop the
`<!-- legacy-v4-markdown -->` sentinel, drop dual-verifier routing, no
need for a parallel deprecated-verifier code path. `verify_clean_result.py`
still exists in tree for historical bodies in `completed/` but no
agent / skill references it on new bodies.

**Migration tool as `task.py migrate-body` subcommand** (Alternatives
critic): the migration logic becomes `task.py migrate-body <N>` (or
`--all`) — ~80 LOC incremental — instead of a standalone 280 LOC
`scripts/migrate_clean_result_bodies.py`. Reuses task.py's existing
flock + atomic-write + git-commit infrastructure.

**Phase D scope expanded** (Methodology critic) to include:
- `.claude/skills/clean-results/SPEC.md` — 5 Sagan refs at lines 89–95
  + an off-by-one count drift ("21 bodies imported from Sagan", actually
  20).
- `.claude/agents/experimenter.md` — line 355 cites the Sagan body spec.
- Pre-edit check: `grep -c '^### Lens [0-9]' .claude/agents/clean-result-critic.md`
  returns **7** today; Lens 7 IS the absorbed statistical-framing rule.
  Phase D's `clean-result-critic.md` edit is "verify Lens 7 matches
  the markdown verifier's checks #7 / #10 / #11", not "add a new lens".

**Phase E pre-flight added** (Methodology critic): before converting
the 3 v4-legacy bodies, verify `verify_clean_result.py` can still
parse them today:
`for n in 186 237 351; do uv run python scripts/verify_clean_result.py tasks/awaiting_promotion/$n/body.md; done`.
This catches any wrapper-HTML shape (e.g.,
`<details open><summary>## TL;DR</summary>`) that the legacy verifier
may not handle, before we trust it as the conversion target.

**Phase F adds `task.py view --rich`** (Alternatives critic): the user's
actual reading surface today is the terminal (the dashboard isn't
deployed). Phase F's CLAUDE.md cross-ref work is fine, but should
include a richer `task.py view <N>` output that surfaces the polished
body excerpt, last 5 events, latest reviewer verdict, and current
status folder. ~50 LOC, ~2h. Directly responds to the user's "display
in the dashboard" request even before the dashboard ships.

**Phase A.5 checkpoint** (Alternatives critic, soft): after Phase A
lands, observe whether agent behavior on a fresh `/issue <N>` invocation
improves. Phases B-F remain in scope (user chose "content depth /
proceed with plan"), but the checkpoint is recorded as a fact in the
plan execution log — if Phase A alone is enough, you can defer B-F by
choice.

**Coverage confirmations** (Methodology critic):
- `.claude/skills/independent-reviewer/SKILL.md` is clean (no
  Sagan/GH refs found). No changes needed.
- `.claude/skills/auto-experiment-runner/SKILL.md` is clean. No
  changes needed.
- `reviewer_pairs` block in current `workflow.yaml` must be preserved
  by Phase B (don't overwrite with GH-era yaml that lacked the
  `reviewer_pairs` key — added post-migration).

**Statistics critic affirmed** the acceptance criteria are
mechanically checkable end-to-end. No revisions to §10.

---

## 3. Phased implementation plan

Each phase is one PR / one commit, independently revertable, and
independently shippable. Wall-time estimates assume a Claude/Codex
implementer pair with cached context; double for a cold start.

### Phase A — Restore the verbose SKILL.md

**Files modified:**
- `.claude/skills/issue/SKILL.md` (126 → ~1400 LOC, +~1275)
- `.claude/skills/issue/markers.md` (53 → ~150 LOC, +~100; will copy
  full marker taxonomy table from GH-era version, with substrate
  substitutions)

**Process:**
1. Pull the GH-era body via `git show 3a6dbe5e^:.claude/skills/issue/SKILL.md
   > /tmp/skill-old.md`.
2. Apply mechanical substitutions (sed pass, then read-through):
   - `gh issue view <N>` → `uv run python scripts/task.py view <N>`
   - `gh issue view --json comments` / `gh_issue_state.py` →
     `uv run python scripts/task.py view <N> --json` (returns body +
     last events; same shape)
   - `gh issue comment <N> --body '...'` →
     `uv run python scripts/task.py post-marker <N> epm:<kind> --note '...'`
   - `gh issue edit <N> --add-label status:X` →
     `uv run python scripts/task.py set-status <N> <status>`
   - `gh issue edit <N> --body "..."` →
     `uv run python scripts/task.py set-body <N> --file ...`
   - "issue body" → "`body.md`"
   - "issue comments" / "workflow_events" / "marker comments" →
     "`events.jsonl` rows"
   - "the GitHub project board column" / "Sagan kanban" / "Project board"
     → "the `tasks/<status>/<N>/` parent folder"
   - Sagan dashboard URLs (`https://sagan.superkaiba.com/e/experiment/<uuid>`)
     → `https://eps.superkaiba.com/tasks/<N>` (the planned EPS dashboard
     URL; `task.py new-plan-version` already prints this URL format).
     For commands that ran against Sagan API, switch to `task.py`.
   - `.claude/plans/issue-<N>.md` → `tasks/<status>/<N>/plans/plan.md`
     (the `plans/v<K>.md` versioned files with `plan.md` symlink to
     latest). Subagent briefs should pass the symlink path so they always
     read the latest.
   - GitHub merge / `gh pr create --draft` / `gh pr merge --rebase` →
     keep as is (git PR flow is substrate-independent).
   - `gh_project.py promote` → `task.py promote`.
   - `gh issue list --search "Parent: #<N>"` (child detection in Step 10
     step 4) → query `tasks/<status>/*/body.md` frontmatter for
     `parent_id == <N>` (frontmatter already supports `parent_id`; see
     `task.py cmd_create`). Spec the exact filesystem query:
     `find tasks -path 'tasks/*/<child>/body.md' -exec grep -l 'parent_id: <N>' {} +`
     with status filter.
3. Drop GH-Actions specific markers: `clean-result-lint` (was a GitHub
   Actions workflow), and references to `.github/workflows/`. Replace
   with: "lint runs locally as part of `task.py audit` and as a pre-commit
   hook in the repo (see `scripts/workflow_lint.py`)".
4. Drop the inline `mcp__happy__change_title` calls (Happy multi-session
   is in CLAUDE.md, not in /issue mechanics anymore). Keep one short
   "Chat title updates" note at the top pointing at `render_title()`
   helper, but de-emphasize.
5. Drop the §5 `epm:step-completed` *implementation* details that talked
   about GitHub-comment scanning (`gh issue view --json comments` to scan
   for the marker). Re-express: "Skill calls `task.py post-marker <N>
   epm:step-completed --note '<step,exit_kind,next_expected_step>'` at
   each EXIT site; on re-entry, `task.py latest-marker <N>
   --prefix epm:step-completed` returns the most recent and the resume
   router applies the same precedence rules (status:blocked first, then
   marker `exit_kind`, then `next_expected_step` lookup in
   `workflow.yaml § steps`)."
6. Drop the `scripts/post_step_completed.py` helper invocations (the
   GH-era used a shell wrapper because `gh issue comment` had quoting
   traps; `task.py post-marker --note` already takes a string arg
   without those traps). Keep the helper name as a "thin wrapper"
   convenience CLI for muscle-memory continuity, but make its only job
   be argument validation against `workflow.yaml § steps`.
7. Drop the `scripts/hf_gate_accept.py` and `scripts/gh_issue_state.py`
   references **only if** they don't already exist in the current
   `scripts/` tree. (Sub-task A.0 verifies.) The HF gate step (6a) is
   still useful — keep, just verify the helper exists.

**Pre-flight (A.0) — HARD GATE:**
- `uv run python scripts/workflow_lint.py --check-references` must exit 0
  BEFORE the Phase A commit lands. If Phase B (or B1) hasn't run yet,
  the lint will report dangling `(see workflow.yaml § X)` references
  in the restored SKILL.md. STOP and run Phase B first.

**Acceptance:**
- `wc -l .claude/skills/issue/SKILL.md` ≥ 1300.
- `grep -c '^### Step ' .claude/skills/issue/SKILL.md` ≥ 14 (matches the
  GH-era step count: 0, 0b, 1, 2, 2b, 2c, 3, 4, 5, 6, 7, 8, 9, 10, 10b,
  10d).
- `grep -c 'gh issue\|sagan_state\|sagan.superkaiba.com' .claude/skills/issue/SKILL.md`
  should be ≤ 2 (only in the "this used to be how it worked" prose at the
  top, if at all).
- `grep -c 'task.py' .claude/skills/issue/SKILL.md` ≥ 30.
- All restored cross-refs of the form `(see workflow.yaml § <key>)`
  resolve (verified by Phase B; A.0 confirms in advance).

**Phase A.5 — checkpoint (soft):** After Phase A lands, observe one
`/issue <N>` invocation on an active task. Record in the plan execution
log whether the restored SKILL.md changed agent behavior measurably.
The user has chosen "content depth, proceed with plan" so Phases B-F
remain in scope, but if Phase A alone is enough the user can defer
B-F by choice.

**Tests added/modified:**
- `tests/test_workflow_lint.py` (NEW, ~80 LOC): currently missing. Add a
  test that runs `scripts/workflow_lint.py --check-references` and asserts
  exit 0. The lint already exists (`scripts/workflow_lint.py`, 309 LOC,
  has a `--check-references` flag we'll verify); just no test wraps it.

**Wall time:** 3–4 hours (large diff, mostly mechanical, needs a careful
read-through for missed substitutions).

---

### Phase B — Restore workflow.yaml content depth (split into B1 + B2)

**Pre-flight (B.0):**
- The R2 critic ran the grep proactively and found B2 is **REQUIRED**,
  not deferrable. `steps` is consumed by:
  - `scripts/post_step_completed.py` (raw YAML parse)
  - `scripts/workflow_lint.py` (iterates `workflow.steps`)
  - `src/explore_persona_space/orchestrate/resume.py` (reads
    `StepEntry` objects via `load_workflow_yaml().steps`)
  - `tests/test_workflow_yaml.py` (multiple assertions)
- The correct grep command (use this; the earlier draft pointed at a
  directory that doesn't exist):
  ```bash
  grep -rn 'workflow\.steps\|load_workflow_yaml\|steps:' src/ scripts/ tests/ | grep -v '\.pyc'
  ```
- Decision rule: **hits ≥ 1 → restore B2 (this case)**; hits = 0 →
  document-only deferral. We already have ≥ 3 hits today; B2 is in.

**B1 — agent-facing reference data** (required because SKILL.md cites it):
- `.claude/workflow.yaml` 200 → ~450 LOC, +~250 LOC.
- RESTORE: `gates` (inline / park_and_wait / conditional),
  `halt_criteria`, `subagent_halt_conditions`, `ensemble_review`
  (already present in current yaml — verify not overwritten), full
  `markers` table (each entry with `kind / posted_by / when / fields`).
- PRESERVE: the existing `reviewer_pairs` block in the current yaml
  (it was added post-GH-migration; the GH-era yaml lacks it). Phase B
  must explicitly assert both:
  - `grep -c 'reviewer_pairs' .claude/workflow.yaml` ≥ 1, AND
  - Structural assertion (via `tests/test_workflow_yaml.py`): the
    parsed YAML contains `reviewer_pairs.pairs` with all three known
    keys (`code_review`, `interpretation`, `clean_result`) — not just
    the bare word `reviewer_pairs`.

**B2 — `steps` table (gated on B.0)**:
- If a Python consumer reads `steps`: restore the full ~17-step lifecycle
  map with `entry_status_label` (~300 LOC).
- If no consumer: skip B2 entirely. SKILL.md citations of `steps` (if
  any) get re-pointed to the in-line step prose.

**Files modified:**
- `.claude/workflow.yaml` (200 → ~450 LOC for B1; +~300 for B2 if gated).
- `scripts/workflow_lint.py` (309 → ~330 LOC, +~20; verify the schema
  changes from pre-flight commit `15dc06fe` still pass once
  `workflow.yaml` is fully populated — the permissive defaults should
  remain idempotent).

**What to restore (not verbatim — adapted to task-workflow substrate):**
1. `issue_types` (5 values): keep — values are substrate-independent.
2. `columns` — **OMIT**. Project-board columns don't exist in the local
   filesystem substrate. The status folder name IS the column. Replace
   this section with a comment explaining the substitution.
3. `statuses` — augment current list with `description`, `next_action`,
   `user_gated`, `parent_folder` (which is the `tasks/<status>/<N>/`
   path component). Keep both the GH-era hyphenated names (`done-experiment`)
   in `status_aliases` AND the new underscore names (`completed`) as
   canonical (already present).
4. `priority_labels` — **OMIT**. Labels don't exist; clean-result
   classification lives in `frontmatter.classification`
   (`useful|not-useful|pending`).
5. `gates` — RESTORE verbatim (inline, park_and_wait, conditional). The
   gate logic is substrate-independent; only the trigger surfaces change
   (AskUserQuestion is the same; user comments on issue → user runs
   `task.py set-status` or replies inline).
6. `halt_criteria` — RESTORE verbatim (5 criteria). Substrate-independent.
7. `subagent_halt_conditions` — RESTORE verbatim. The table maps
   subagent → verdict → action; the action wording needs the substrate
   substitution (e.g., "writes BLOCKER to plan body" → "writes BLOCKER to
   `tasks/<status>/<N>/plans/v<K>.md`").
8. `ensemble_review` — RESTORE verbatim. The contract is independent of
   where markers are posted.
9. `markers` — RESTORE the FULL `name + posted_by + when + fields` table
   for every marker. The current `markers.names` is just a flat list;
   the GH-era version had per-marker `fields` strings that downstream
   agents read. Some markers drop (e.g., `clean-result-lint` was a GH
   Actions workflow); some add (e.g., `epm:step-completed` is now
   posted via `task.py`).
10. `steps` — RESTORE the lifecycle map with `entry_status_label`. This
    is load-bearing for the §5 resume router. ~17 steps total.

**Acceptance:**
- `wc -l .claude/workflow.yaml` ≥ 600.
- `yaml.safe_load(open('.claude/workflow.yaml'))` succeeds with the
  expected top-level keys.
- `uv run python scripts/workflow_lint.py --check-references` exits 0 —
  every `(see workflow.yaml § X)` in CLAUDE.md / SKILL.md / markers.md
  resolves to a real key.
- `uv run python scripts/workflow_lint.py --emit-tables` regenerates the
  AUTO-GENERATED active-vs-awaiting status table inside SKILL.md (the
  GH-era SKILL.md has a `<!-- workflow.yaml: AUTO-GENERATED -->` fence;
  restored SKILL.md keeps it).

**Tests added/modified:**
- `tests/test_workflow_yaml.py` (currently 201 LOC) — expand to cover
  the new top-level keys: assert `gates`, `halt_criteria`,
  `subagent_halt_conditions`, `ensemble_review`, `markers` (full shape),
  `steps` are present and well-formed. Assert every `entry_status_label`
  in `steps` references a real status. Assert every marker in
  `reviewer_pairs.pairs.*.markers` exists in the `markers` list. +~80
  LOC.
- `tests/test_workflow_lint.py` (NEW from Phase A) — extend with assertions
  that `--check-references` resolves all SKILL.md anchors.

**Wall time:** 2–3 hours.

---

### Phase C — Restore verify_task_body.py from 6 → 11 checks

**Files modified:**
- `scripts/verify_task_body.py` (363 → ~700 LOC, +~340)

**Mapping of verify_sagan_card.py's 11 HTML checks → markdown
equivalents:**

| # | Sagan-card HTML check | Markdown equivalent | Status today |
|---|---|---|---|
| 1 | Scoped `<style>` block | **DROP** — no inline CSS in markdown. | n/a |
| 2 | TL;DR section: `<section id="tldr">` with 4 `<li>` bullets | `## TL;DR` H2 with 4 labelled bullets (Motivation/What I ran/Results/Next steps) | ✅ present (check #3) |
| 3 | Hero figure: `<figure id="figure">` with `<svg>`/`<img>` + `<figcaption>` | `## Figure` H2 with `![alt](path)` image + caption line | ✅ present (check #6, partially — only checks caption ≥10 words; need to ALSO check image is present) |
| 4 | Experimental-design dropdown: `<details id="design">` + `<summary>` | `## Details` H2 (no collapsibility in markdown; the section header IS the toggle when rendered by react-markdown + GFM) | ✅ present (check #2) — but RENAME `Design` would break legacy. Keep `Details` as the heading; the audit lens "heading-as-toggle convention" already maps. |
| 5 | Reproducibility appendix: `<details id="repro">` AFTER `#design` with Artifacts / Compute / Code groups | `## Reproducibility` H2 LAST H2 with three bold-labelled subgroups: `**Artifacts:**`, `**Compute:**`, `**Code:**` | partial (#4 checks URLs but doesn't require the three subgroups) — **ADD**: assert `## Reproducibility` is the last H2 + contains the three labelled subgroups |
| 6 | URL permanence (repro only) | URL permanence in `## Reproducibility` section | ✅ present (check #4) |
| 7 | Sentinel scrub (repro only) | Same | ✅ present (check #4 folds this in) |
| 8 | Confidence-rationale line BEFORE `#repro` | Confidence sentence in `## Details`, before `## Reproducibility` | ✅ present (check #5) — but ALSO need the ≥20-char rationale length |
| 9 | Cherry-picked label on every sample `<pre>` in `#design` | Cherry-picked label on every fenced code block in `## Details` that looks like a sample completion | **MISSING** — port. Heuristic: fenced block with `User:`/`Assistant:`/`Human:`/`Model:` or >200 chars. |
| 10 | Title vs body confidence | Title vs Details confidence | ✅ present (check #5) |
| 11 | Qualitative-data link in `#design` near each sample `<pre>` | Qualitative-data link in `## Details` near each sample fenced block | **MISSING** — port. Aggregate-pattern regex (`_AGGREGATE_PATH_RE`) is substrate-independent; reuse. |

After porting, the markdown verifier has **11 checks** (dropping #1
style-block, splitting check #4 into "three repro subgroups" + "URL
permanence" + "sentinel scrub", which already nets to 11). Concretely:

1. Title confidence tag (existing #1).
2. Four required H2 sections in order (existing #2).
3. TL;DR bullets carry four required labels (existing #3).
4. **Hero image present** in `## Figure` (NEW — split from existing #6
   which only checks the caption).
5. Figure caption ≥10 words (existing #6).
6. Details confidence sentence matches title + ≥20-char rationale (existing
   #5, expanded).
7. Reproducibility three subgroups present (NEW).
8. Reproducibility URL permanence (existing #4 a).
9. Reproducibility sentinel scrub (existing #4 b).
10. Cherry-picked label on every sample fenced block in `## Details` (NEW).
11. Qualitative-data link near every sample fenced block in `## Details`
    (NEW — port `_AGGREGATE_PATH_RE` + `_NOT_UPLOADED_RE` + the WARN
    downgrade logic verbatim).

**Acceptance:**
- `uv run python scripts/verify_task_body.py --help` prints 11 numbered
  checks in the docstring.
- `len(verify_task_body.CHECKS) == 11`.
- Running against the canonical-good body in the test fixture
  (`tests/fixtures/clean_result_canonical.md`) yields PASS.

**Tests added/modified:**
- `tests/test_verify_task_body.py` (173 → ~450 LOC, +~280). Add one
  fixture body per new check, asserting both happy-path (PASS) and one
  representative violation per check (FAIL with the expected detail
  string). Pattern: copy the fixture-per-check approach already used
  for checks 1–6.
- `tests/fixtures/` — add `clean_result_canonical.md` (a minimal
  11-check-PASS exemplar) plus one violation fixture per new check.

**Wall time:** 4–5 hours (a lot of regex work + fixtures).

---

### Phase D — Update agent specs + audit script to enforce restored discipline

**Pre-flight (D.0):**
- `grep -c '^### Lens [0-9]' .claude/agents/clean-result-critic.md`
  — verified to return **7** today; the file has Lenses 1–7. **Lens 7
  IS the statistical-framing rule** ("absorbed from the retired
  reviewer"). The earlier plan called this "Lens 11" which was a
  memory error.
- Decision rule: hits ≥ 1 → verify-existing path (~0–10 LOC, just
  ensure Lens 7 references markdown checks #7 / #10 / #11 of
  `verify_task_body.py`); hits = 0 → add-new path (~40 LOC). Current
  state: hits = 7 → verify-existing path.
- `grep -nE 'sagan_state|verify_sagan_card|gh_project' .claude/skills/clean-results/SPEC.md .claude/skills/independent-reviewer/SKILL.md .claude/skills/auto-experiment-runner/SKILL.md`
  — confirm the latter two are clean (Methodology critic verified)
  but SPEC.md has 5 Sagan refs that need updating.

**Files modified:**
- `.claude/agents/analyzer.md` (244 LOC) — update the "Required body
  format" template to include the three Repro subgroups (Artifacts /
  Compute / Code) and the cherry-picked-label rule for sample blocks.
  Update the anti-patterns list to match the audit script (already
  227 LOC; no rewrite, just align references). +~30 LOC.
- `.claude/agents/clean-result-critic.md` (242 LOC) — verify-existing
  Lens 7 (the statistical-framing rule, already absorbed from the
  retired reviewer) references the markdown verifier's new checks
  #7 / #10 / #11. Per Phase D.0: edit size is +~0–10 LOC because the
  lens already exists. No structural changes to the 7-lens enumeration.
- `.claude/agents/codex-clean-result-critic.md` (similar update; runs
  `verify_task_body.py` *and* `audit_clean_results_body_discipline.py`
  independently in its instructions). +~15 LOC.
- `.claude/agents/experimenter.md` (414 LOC) — line 355 cites the
  Sagan body spec at `~/sagan/docs/clean-result-guidelines.md`. Replace
  with `.claude/skills/clean-results/SPEC.md` (the new spec). +~5 LOC.
- `.claude/skills/clean-results/SPEC.md` (~200 LOC) — replace 5 Sagan
  refs at lines 89–95 ("21 bodies imported from Sagan" → "20 bodies
  imported from Sagan" — also fixes the off-by-one drift). Update
  the "Six mechanical checks (verify_task_body.py)" section header to
  "Eleven mechanical checks (verify_task_body.py)" and list the new
  checks. +~25 LOC.
- `scripts/audit_clean_results_body_discipline.py` — no logic change
  needed (regex on prose is markdown-safe). Verify CLI flag for
  scanning `tasks/<status>/<N>/body.md` paths instead of GitHub fetches.
  +~20 LOC if the current entrypoint hard-codes GitHub fetch (verify in
  Phase D.0).

**Acceptance:**
- `clean-result-critic.md` references `verify_task_body.py` not
  `verify_sagan_card.py` for new bodies.
- `analyzer.md` body-format template matches what `verify_task_body.py`
  enforces (no drift).
- `audit_clean_results_body_discipline.py --task <N>` works against
  `tasks/<status>/<N>/body.md`.

**Tests added/modified:**
- `tests/test_verify_clean_result.py` (exists, but tests the LEGACY
  HTML verifier) — leave alone; legacy bodies still flow through it.
- `tests/test_verify_task_body.py` already covered in Phase C.

**Wall time:** 1–2 hours.

---

### Phase E — Migrate existing awaiting_promotion bodies (HTML → markdown)

**Files modified:** up to 11 `tasks/awaiting_promotion/<N>/body.md`
files (the 20 HTML bodies are grandfathered untouched), plus a new
`task.py migrate-body` subcommand.

**New code:** `scripts/task.py migrate-body <N> [--apply | --dry-run]`
subcommand (~80 LOC incremental). Reuses task.py's existing flock +
atomic-write + git-commit infrastructure. NO new standalone script.
NO new sentinel.

**Pre-flight (E.0):**
- For the 3 v4-legacy markdown bodies, confirm they actually parse
  today as expected shape (not wrapped in `<details open><summary>`
  HTML, etc.):
  ```
  for n in 186 237 351; do
    head -1 tasks/awaiting_promotion/$n/body.md
    grep -c '^## Summary\|^## TL;DR\|^## Details\|^## Source issues' tasks/awaiting_promotion/$n/body.md
  done
  ```
  Expected: ≥3 H2 hits per body. If any body has wrapper HTML, hand
  it to the user instead of auto-converting (3 bodies is a small enough
  count that bailing on edge cases is fine).

**Conversion strategy per body class:**

| Class | Count | Action |
|---|---|---|
| Sagan-card HTML with `<!-- legacy-sagan-card -->` sentinel | 20 | **GRANDFATHER, do not auto-convert.** The dashboard already renders them as HTML via `rehypeRaw`. `verify_task_body.py` already skips them via the sentinel check. They flow through `verify_sagan_card.py` (still in tree) when needed. |
| Markdown four-H2 already conformant | 8 | **Re-verify under new 11-check rules.** Expect ~3 to newly FAIL on check #7 (three Repro subgroups missing) and check #11 (qualitative-data link missing). Patch in place via `task.py migrate-body --apply` — see "Conformant-but-failing remediation" below. |
| Markdown v4-legacy (`## TL;DR / ## Summary / ## Details / ## Source issues`) | 3 (#186, #237, #351) | **Convert in place** via `task.py migrate-body --apply --shape v4-to-new`. The subcommand: (a) snapshots the existing body to `original-body.md`, (b) renames `## Summary` to `## Figure` (or splits it across `## Figure` + `## Details` if it contains both a figure and prose — the migration logic does this conservatively), (c) injects an empty `## Reproducibility` H2 with `n/a` placeholders if missing, (d) preserves `## Source issues` as a trailing H2 (already passes the verifier per assumption #25). Idempotent — second invocation is a no-op. The bodies are short research write-ups; if any rename is ambiguous, the subcommand emits a `--needs-user` flag and leaves the body untouched for the user to hand-edit. |

**Note on assumption #25 (CONFIRMED by Fact-Checker):** the current
`check_required_sections` in `verify_task_body.py` already filters
`seq = [s for s in found if s in REQUIRED_H2_SECTIONS]` before the
order check, so extra H2 sections like `## Source issues` after
`## Reproducibility` already PASS. No verifier patch needed for that
case (it was a misread in the original plan).

**Conformant-but-failing remediation** (subcommand logic for the 8
conformant + 3 v4-legacy bodies):
1. Runs `verify_task_body.py --file body.md --json` on each.
2. For check #7 (Repro subgroups missing): PARSE existing
   `## Reproducibility` structure first. Only inject subgroups that
   are missing (string-equal match on `**Artifacts:**`, `**Compute:**`,
   `**Code:**` at line-start). NEVER duplicate an existing subgroup
   (per Methodology critic's idempotency tightening).
3. For check #11 (qualitative-data link missing): add a `not uploaded`
   disclosure paragraph above each affected sample block, and add a
   Next-steps bullet "re-run with raw-completion upload" to the TL;DR.
4. For check #10 (cherry-picked label missing): add `(cherry-picked for
   illustration)` to the prose immediately above each sample fenced
   block.
5. Re-run the verifier. If still failing, flag with `--needs-user`
   and leave the body alone — the human (or analyzer agent in a
   subsequent `/issue` re-entry) handles it.

`task.py migrate-body` is idempotent and runs `--dry-run` by default.
The user runs `task.py migrate-body --all --apply` once at the end of
Phase E.

**Acceptance:**
- All 31 awaiting_promotion bodies pass either `verify_task_body.py`
  (markdown) OR `verify_sagan_card.py` (legacy HTML).
- Bodies are categorized in a one-shot report:
  `uv run python scripts/task.py migrate-body --report` lists per-task
  status (PASS markdown / PASS legacy HTML / FAIL needs-fix).
- A second invocation of `--apply` is a no-op (idempotency check).
- One fixture in `tests/test_task_workflow.py` exercises the
  "one-subgroup-already-present" case (subgroup-injection idempotency).

**Tests added/modified:**
- `tests/test_task_workflow.py` extended (+~80 LOC, ~3 new fixtures):
  v4-to-new shape conversion, conformant-but-failing remediation,
  idempotency double-apply check.

**Wall time:** 3–4 hours. (R2 critic flagged the original ~80 LOC
estimate for `task.py migrate-body` as optimistic; realistic 100–150 LOC
adds ~1h.)

---

### Phase F — Update CLAUDE.md + promote-clean-result skill + markers.md cross-references + richer `task.py view`

**New deliverable (Alternatives critic):** richer `task.py view --rich
<N>` terminal output, since the `dashboard/` Next.js app isn't deployed
yet and the user's actual reading surface IS the terminal. ~50 LOC
incremental to `scripts/task.py`. Output: status folder + frontmatter
+ first 30 lines of body + last 5 `events.jsonl` entries + latest
clean-result-critic verdict if any. Acceptance:
`uv run python scripts/task.py view --rich 333` produces a
human-readable one-page summary without scrolling more than ~50 lines.
~2h.

**Files modified:**
- `CLAUDE.md` — the "Experiment Report Structure" section currently
  documents the Sagan-card HTML spec (lines ~80–160). Replace with the
  markdown spec. Keep a short backward-compat paragraph: "Legacy
  Sagan-card HTML bodies authored before YYYY-MM-DD are grandfathered;
  the dashboard renders them as HTML via `rehypeRaw`, and the legacy
  `verify_sagan_card.py` validator still applies to those bodies." +~80
  LOC, -~120 LOC (net ~-40 LOC).
- `.claude/skills/promote-clean-result/SKILL.md` (207 LOC) — replace
  references to `verify_sagan_card.py` with `verify_task_body.py`;
  remove the auto-convert-HTML-to-markdown step (current spec says "auto-
  converts to Sagan-card HTML during promotion" — invert: bodies stay
  in whatever shape they came in; the analyzer writes new bodies in
  markdown). +~10 LOC, -~30 LOC.
- `.claude/skills/issue/markers.md` (53 LOC) — already touched in Phase
  A; final pass aligns marker `fields` with `workflow.yaml § markers`.
- `.claude/skills/issue/SKILL.md` — already touched in Phase A; final
  pass aligns the Lens 7 (statistical-framing) reference and the
  `task.py migrate-body` subcommand reference.

**Acceptance:**
- `grep -c 'verify_sagan_card\|sagan-card\|Sagan-card' CLAUDE.md` ≤ 5
  (only the grandfathered-bodies paragraph references it).
- `grep -c 'verify_task_body\|markdown clean-result' CLAUDE.md` ≥ 8.
- `uv run python scripts/workflow_lint.py --check-references` still
  passes (cross-refs to workflow.yaml still resolve).

**Tests added/modified:**
- Re-run `tests/test_workflow_lint.py` (from Phase A) — must still pass.

**Wall time:** 1 hour.

---

## 4. File-level diff outline

| File | Action | Net LOC | Shape change |
|---|---|---|---|
| `.claude/skills/issue/SKILL.md` | rewrite | 126 → ~1400 | restore state-machine diagram, Step 0–10d, render_title helper, resume table, Cost/safety, Error-handling table |
| `.claude/skills/issue/markers.md` | rewrite | 53 → ~150 | full per-marker `name+posted_by+when+fields` rows, mirrors workflow.yaml § markers |
| `.claude/workflow.yaml` | augment | 200 → ~750 | add `gates`, `halt_criteria`, `subagent_halt_conditions`, `ensemble_review`, full `markers`, `steps` |
| `scripts/workflow_lint.py` | augment | 309 → ~340 | recognize new YAML keys; --check-references and --emit-tables continue to work |
| `scripts/verify_task_body.py` | augment | 363 → ~700 | 6 → 11 checks; port _AGGREGATE_PATH_RE + cherry-picked logic from verify_sagan_card.py |
| `scripts/audit_clean_results_body_discipline.py` | minor | 227 → ~245 | --task CLI flag for filesystem-path input |
| `scripts/task.py` | augment | 357 → ~440 | new `migrate-body` subcommand (~80 LOC realistic; estimate widened from earlier ~80, see Phase E) and new `view --rich` flag (~50 LOC, Phase F) |
| `.claude/agents/analyzer.md` | augment | 244 → ~280 | update body-format template to match restored 11-check spec |
| `.claude/agents/clean-result-critic.md` | augment | 242 → ~260 | verify-existing Lens 7 (statistical-framing rule) already references the 11-check enumeration; +~0–10 LOC only |
| `.claude/agents/codex-clean-result-critic.md` | minor | similar | reference verify_task_body.py + audit script |
| `.claude/skills/promote-clean-result/SKILL.md` | augment | 207 → ~190 | drop HTML-conversion, point at verify_task_body.py |
| `CLAUDE.md` | augment | ~1500 → ~1500 | swap §Experiment Report Structure HTML spec for markdown spec; one backcompat paragraph |
| `tests/test_verify_task_body.py` | augment | 173 → ~450 | fixtures + assertions for new checks 4, 7, 10, 11 + expanded 6 |
| `tests/test_workflow_yaml.py` | augment | 201 → ~290 | assertions for new top-level keys |
| `tests/test_workflow_lint.py` | NEW | 0 → ~80 | wrap --check-references in pytest |
| `tests/test_task_workflow.py` | augment | existing → +~80 | `migrate-body` v4-to-new fixtures + idempotency double-apply (folded in, no new test file) |
| `tests/fixtures/clean_result_canonical.md` | NEW | 0 → ~80 | 11-check-PASS exemplar |
| `tests/fixtures/clean_result_fail_*.md` | NEW | 0 → ~30/each × 4 | per-violation fixtures |
| ~22 × `tasks/awaiting_promotion/<N>/body.md` | grandfathered | unchanged | no edit; sentinel routes them through verify_sagan_card.py |
| ~3 × `tasks/awaiting_promotion/<N>/body.md` | patched | +~10/each | inject `**Artifacts:**`/`**Compute:**`/`**Code:**` `n/a` placeholders + cherry-picked labels |

Total: ~3400 LOC added, ~150 LOC removed, ~9 new files.

---

## 5. Migration handling for existing tasks

Already enumerated in Phase E and §2. To summarize: of the 31
`tasks/awaiting_promotion/` bodies,

- **22** are HTML Sagan-card → **grandfather**. No edit. They carry
  `<!-- legacy-sagan-card -->`; the markdown verifier already skips
  them; the dashboard already renders them as HTML.
- **~3** are markdown that will newly fail under the 11-check rules
  (missing Repro subgroups, missing cherry-picked label, missing
  qualitative-data link) → **patch in place** via
  `task.py migrate-body --apply`. The patch injects `n/a`
  placeholders or `not uploaded` disclosures + Next-steps bullets,
  preserving any author voice that's already there.
- **~6** are markdown that already conform to all 11 checks → **leave
  alone, re-verify**.
- **3 v4-legacy bodies (#186, #237, #351)** → **convert in place**
  via `task.py migrate-body --shape v4-to-new --apply`. No new
  sentinel; no dual-verifier routing.

The entry point is the new `task.py migrate-body` subcommand:
- `--report`: per-task PASS/FAIL/legacy classification.
- `--dry-run [<N>]` / `--dry-run --all`: show the proposed diff.
- `--apply [<N>]` / `--apply --all`: write the patch.
- `--needs-user`: bail flag emitted when shape conversion is
  ambiguous and the body should be hand-edited.

No bulk-conversion HTML → markdown is attempted. The user can
selectively re-author any HTML body in markdown later; the dashboard
already routes by the `legacy-sagan-card` sentinel.

**`verify_clean_result.py` disposition.** After Phase E converts the
3 v4-legacy bodies in place, no `awaiting_promotion/` body is
routed to `verify_clean_result.py`. It stays in tree as a read-only
historical artifact for any v4-shape body that lives under
`tasks/completed/` (older clean-results from before the
Sagan-card era). New agents and skills do not reference it. No
deprecation comment is added; the file is simply not mentioned
in restored SKILL.md / markers.md / clean-result-critic.md.

---

## 6. Tests

| Test file | State | Action |
|---|---|---|
| `tests/test_verify_task_body.py` | exists, 173 LOC | expand to 11 checks; +~280 LOC |
| `tests/test_task_workflow.py` | exists, 396 LOC | confirm coverage holds; the SKILL.md is documentation, not exercised by this file; **no change** |
| `tests/test_workflow_lint.py` | **missing** | NEW, ~80 LOC: wraps `scripts/workflow_lint.py --check-references` in pytest; assert exit 0 |
| `tests/test_workflow_yaml.py` | exists, 201 LOC | +~80 LOC for new top-level keys, steps[].entry_status_label coverage |
| `tests/test_task_workflow.py` (migrate-body coverage) | extend | +~80 LOC: v4-to-new shape, idempotency double-apply, --needs-user fallback (folded into existing file, NOT a new test file) |
| `tests/fixtures/clean_result_canonical.md` | **missing** | NEW: 11-check-PASS exemplar |
| `tests/fixtures/clean_result_fail_*.md` | **missing** | NEW: one per new violation kind (Repro subgroups missing, cherry-picked missing, qual-data link missing, image missing) |

The existing `tests/test_verify_clean_result.py` (legacy HTML verifier
tests) is **left alone** — the legacy path is grandfathered, not
deprecated.

---

## 7. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| **11-check verifier rejects bodies that previously passed 6-check.** Specifically: bodies missing the three Repro subgroups, missing cherry-picked label, missing qualitative-data link. | Moderate (~3 of the 8 conformant markdown bodies will fail; the 3 v4-legacy bodies sentinel-grandfather to `verify_clean_result.py`) | Migration script auto-injects `n/a` placeholders + `not uploaded` disclosures + cherry-picked labels. The bodies still PASS afterwards (no false-positive blocking promotion). User can refine later. |
| **Longer SKILL.md bloats subagent context.** The analyzer, code-reviewer, codex twins, and reconciler read `/issue` SKILL.md sections by reference. | Medium | The skill body is loaded lazily via `Skill` tool invocation; subagents don't auto-read it. They reference it by step number. If a subagent's prompt grows >5K tokens because it cited SKILL.md, that's a bug in the agent spec, not in SKILL.md. **Action:** during Phase D, audit `.claude/agents/*.md` for any agent that pastes SKILL.md sections inline; replace with anchored references. |
| **Codex twin agents reference SKILL.md by step number (e.g. "Step 5", "Step 9a").** Restoring the step structure means those refs need to stay valid. | Low | Verified via `grep` (above): all references use step numbers (`Step 5`, `Step 9a`, `Step 9a-bis`, `Step 9b`, `Step 10`). The restored SKILL.md preserves those exact step IDs. **No codex-agent rewrites needed.** The only exception is `codex-reviewer.md`, which is already marked DEPRECATED; leave its references to "Step 9b retired" as is. |
| **Sagan-card HTML in awaiting_promotion → markdown loses info.** The `<style>` block is purely cosmetic but `<details>` collapsibility doesn't translate. | n/a — we grandfather, not convert | The dashboard renders both HTML (via `rehypeRaw`) and markdown (via `react-markdown`). No data loss. |
| **Dashboard cannot render Sagan-card markdown.** | Low — verified | `dashboard/app/tasks/[id]/page.tsx` already renders markdown bodies via `ReactMarkdown + remarkGfm + rehypeRaw + rehypeHighlight`. Legacy HTML routes via `isLegacyHtml` flag in `dashboard/lib/tasks.ts:124`. Both paths work today. **No dashboard code changes needed.** |
| **`scripts/workflow_lint.py --check-references` finds dangling refs after Phase A but before Phase B.** | Medium — known intra-PR risk | Order Phase A and Phase B as a single PR pair landing together (atomic). Or land Phase B first (workflow.yaml + lint update), then Phase A (SKILL.md content that references the new keys). Recommend: **B before A** to avoid a window where SKILL.md cites keys workflow.yaml doesn't have. |
| **`render_title()` helper invokes `mcp__happy__change_title`** — that MCP server may not be available in all sessions. | Low | The helper is cosmetic; GH-era SKILL.md already says "if the MCP tool is unavailable, continue without error". Keep that semantics; the implementer wires it as a soft-fail. |
| **`scripts/post_step_completed.py` may not exist in the current tree.** | Need to verify | Pre-flight: `test -f scripts/post_step_completed.py`. If missing, decide in Phase A whether to (a) port it from GH-era or (b) inline the `task.py post-marker` call directly into SKILL.md. The repo has a modified `scripts/post_step_completed.py` per git status — so it likely exists; verify shape in Phase A. |
| **The new SKILL.md may diverge from CLAUDE.md auto-continuation policy.** | Medium | CLAUDE.md (which is loaded into every session) enumerates the 5 inline gates + 1 park-and-wait gate + 1 conditional gate. The restored SKILL.md must match. **Action:** Phase F audit step: `diff <(grep -E '^[0-9]\.' CLAUDE.md | head -10) <(grep '^### Step' SKILL.md)` — flag mismatch. |
| **Existing agent specs reference the deprecated `verify_clean_result.py` (not the new `verify_task_body.py`).** | Medium | `grep -rn 'verify_clean_result' .claude/agents/ .claude/skills/` returns several hits (analyzer, clean-result-critic, promote-clean-result, codex-clean-result-critic). Phase D + F update all of them. The legacy script stays in tree (grandfathered bodies); the references just point at the new one for NEW bodies. |
| **The 31-count of awaiting_promotion may shift between plan time and execution.** | Low | The migration script is idempotent and read-only by default. New tasks landing in awaiting_promotion during Phase E are routed automatically (markdown → `verify_task_body.py`, HTML → grandfathered). |

---

## 8. Resources

| Phase | Wall time (single experienced implementer) |
|---|---|
| A: SKILL.md restoration | 3–4 h |
| B: workflow.yaml restoration | 2–3 h |
| C: verify_task_body.py 6→11 checks | 4–5 h |
| D: agent specs + audit script updates | 1–2 h |
| E: existing awaiting_promotion migration | 2–3 h |
| F: CLAUDE.md / promote-clean-result / markers.md cross-refs | 1 h |
| **Total** | **13–18 h** |

Doubling for review (Codex critic round + revisions): **20–30 h end-to-end.**

Recommended split into 2 sessions: Phases A+B+C in session 1 (~10 h, the
core mechanics), Phases D+E+F in session 2 (~6 h, the polish + migration).
Each session should land its phases as separate atomic commits so a
revert of Phase E doesn't undo Phase C.

---

## 9. Assumptions

Every factual claim made in this plan, with confidence level and how verified.

| # | Assumption | Confidence | Verified via |
|---|---|---|---|
| 1 | GH-era SKILL.md is 1546 LOC at `3a6dbe5e^`. | HIGH | `git show 3a6dbe5e^:.claude/skills/issue/SKILL.md \| wc -l` → 1546 |
| 2 | Current SKILL.md is 126 LOC. | HIGH | `wc -l .claude/skills/issue/SKILL.md` → 126 |
| 3 | GH-era workflow.yaml is 821 LOC. | HIGH | `git show 3a6dbe5e^:.claude/workflow.yaml \| wc -l` → 821 |
| 4 | Current workflow.yaml is 200 LOC. | HIGH | `wc -l .claude/workflow.yaml` → 200 |
| 5 | `scripts/verify_sagan_card.py` has 11 checks. | HIGH | inspected file: checks named `check_style_block`, `check_tldr_section`, `check_hero_figure`, `check_design_block`, `check_repro_block`, `check_url_permanence`, `check_sentinel_scrub`, `check_confidence_line`, `check_cherry_picked_label`, `check_qualitative_data_link`, `check_title_confidence` = 11 |
| 6 | `scripts/verify_task_body.py` has 6 checks. | HIGH | inspected file: `CHECKS = [check_title_confidence, check_required_sections, check_tldr_labels, check_reproducibility_urls, check_confidence_matches, check_figure_caption]` = 6 |
| 7 | The mechanical substrate substitutions listed in §3 Phase A are accurate to the current file contents. | HIGH | verified each substitution by grepping the GH-era SKILL.md and confirming the task.py CLI has matching subcommands (`task.py --help` is documented in the script's docstring; all 16 subcommands enumerated) |
| 8 | No `.claude/agents/codex-*.md` agent references SKILL.md by anchor; all references use step numbers. | HIGH | `grep -nE 'anchor\|SKILL.md\|#tldr\|#design\|#repro\|#figure' .claude/agents/codex-*.md` returned 0 matches; subsequent `grep 'Step ' .claude/agents/*.md` returns step-number refs only |
| 9 | `tests/test_workflow_lint.py` does NOT exist; `tests/test_workflow_yaml.py` does (201 LOC). | HIGH | `test -f tests/test_workflow_lint.py && echo EXISTS || echo MISSING` → MISSING; `wc -l tests/test_workflow_yaml.py` → 201 |
| 10 | Dashboard renders markdown AND legacy HTML; toggle is the `legacy-sagan-card` sentinel. | HIGH | inspected `dashboard/app/tasks/[id]/page.tsx` (uses `react-markdown` + `rehypeRaw`) and `dashboard/lib/tasks.ts:124` (`isLegacyHtml = body.includes(LEGACY_SAGAN_CARD_SENTINEL)`) |
| 11 | `tasks/awaiting_promotion/` contains exactly 31 task folders. | HIGH | `ls tasks/awaiting_promotion/ \| wc -l` → 31 |
| 12 | 20 of those 31 are Sagan-card HTML (carry `legacy-sagan-card` sentinel + `<style>` block); 11 are markdown — 8 conformant new-shape four-H2, 3 v4-legacy (`## TL;DR / ## Summary / ## Details / ## Source issues`). | HIGH | Fact-Checker re-counted via per-task grep: 20 HTML legacy-sagan-card (61, 65, 75, 105, 113, 116, 123, 182, 187, 207, 215, 224, 225, 234, 235, 276, 311, 337, 354, 368); 8 conformant markdown (333, 355, 358, 360, 363, 366, 369, 370); 3 v4-legacy markdown (186, 237, 351). The original "22 / 9–11" count was wrong. |
| 13 | `scripts/verify_clean_result.py` (legacy markdown v4-shape verifier) is still in tree (2087 LOC). | HIGH | `wc -l scripts/verify_clean_result.py` → 2087 |
| 14 | `scripts/audit_clean_results_body_discipline.py` is regex-based on prose and substrate-independent. | HIGH | inspected first 80 LOC: pure regex on file content; no GitHub API calls in the patterns dict |
| 15 | `scripts/workflow_lint.py` exists (309 LOC) with `--check-references` and `--emit-tables` flags. | MEDIUM | `wc -l scripts/workflow_lint.py` → 309; flag names inferred from GH-era SKILL.md prose. Verify in Phase B by reading the file's argparse. |
| 16 | The current `scripts/task.py` supports all the subcommands the restored SKILL.md needs (view, set-status, post-marker, set-body, set-title, list-markers, latest-marker, new-plan-version, find, audit, promote, set-clean-result, add-tag, remove-tag). | HIGH | inspected `scripts/task.py` lines 240–350: argparse enumerates all 16 subcommands |
| 17 | `tasks/<status>/<N>/plans/v{K}.md` is the canonical plan path (with `plan.md` symlink to latest). | HIGH | inspected `scripts/task.py` `cmd_new_plan_version` (line 207–212) — prints `tasks/<status>/{N}/plans/v{v}.md`; SKILL.md frontmatter says so explicitly |
| 18 | `https://eps.superkaiba.com/tasks/<N>` is the planned EPS dashboard URL format. | HIGH | `scripts/task.py` `cmd_new_plan_version` line 211 prints this URL |
| 19 | `verify_sagan_card.py` remains in tree (20 HTML bodies grandfathered); `verify_clean_result.py` also remains but is no longer invoked on any `awaiting_promotion` body. **Revised post-R2 review:** the 3 v4-legacy bodies (186, 237, 351) are converted in place by `task.py migrate-body --shape v4-to-new` rather than sentinel-routed to `verify_clean_result.py`. No `<!-- legacy-v4-markdown -->` sentinel is introduced. `verify_clean_result.py` (2087 LOC) is kept only as a historical artifact for any v4-shape body that may exist under `tasks/completed/`; new agents and skills do not reference it. **Decision: keep both as read-only history; route nothing new through `verify_clean_result.py`.** |
| 20 | `scripts/post_step_completed.py` exists in the current tree. | MEDIUM | git status shows the file is modified (M scripts/post_step_completed.py), implying it exists. Verify shape in Phase A pre-flight before restoring its invocations. |
| 21 | The `render_title()` helper invokes `mcp__happy__change_title` and may be unavailable; restored SKILL.md must mark it soft-fail. | HIGH | GH-era SKILL.md says verbatim "if mcp__happy__change_title is unavailable, log and continue." |
| 22 | `dashboard/lib/tasks.ts` LEGACY_SAGAN_CARD_SENTINEL is the string `<!-- legacy-sagan-card -->`. | HIGH | confirmed in `scripts/verify_task_body.py:54`: `LEGACY_SAGAN_CARD_SENTINEL = "<!-- legacy-sagan-card -->"` |
| 23 | `tests/test_verify_clean_result.py` (legacy HTML test) should NOT be retired during this restoration. | HIGH | 22 grandfathered HTML bodies depend on `verify_sagan_card.py` continuing to validate them. The legacy tests cover the legacy verifier; keep both. |
| 24 | No new YAML top-level keys conflict with existing task_workflow consumers. | MEDIUM | Fact-Checker did not grep src/. **Pre-flight check baked into Phase B Step 1:** `grep -rn 'workflow.yaml\|yaml.safe_load' src/explore_persona_space/task_workflow/` before any schema changes. If any consumer expects a closed top-level shape, adapt the addition accordingly (most consumers in `task_workflow/` are likely just reading specific keys, not enforcing schema). |
| 25 | The 4-required-H2-in-order check in `verify_task_body.py` correctly handles bodies with additional H2 sections after `## Reproducibility`. | HIGH | Fact-Checker confirmed: `check_required_sections` (lines 148–166) filters `seq = [s for s in found if s in REQUIRED_H2_SECTIONS]` BEFORE the order check. Smoke-tested with 4 required + extra `## Source issues` → PASS. **No verifier patch needed** (earlier hedge was wrong). |

---

## 10. Acceptance criteria for the whole effort

When all six phases land, the following mechanical checks pass:

```bash
# Phase A acceptance
wc -l .claude/skills/issue/SKILL.md         # ≥ 1300
grep -c '^### Step ' .claude/skills/issue/SKILL.md   # ≥ 14
grep -c 'task.py' .claude/skills/issue/SKILL.md     # ≥ 30
grep -c 'gh issue\|sagan_state\|sagan.superkaiba.com' .claude/skills/issue/SKILL.md   # ≤ 2

# Phase B acceptance
wc -l .claude/workflow.yaml                  # ≥ 600
uv run python scripts/workflow_lint.py --check-references   # exit 0
uv run python scripts/workflow_lint.py --emit-tables       # regenerates SKILL.md auto-gen fence cleanly

# Phase C acceptance
uv run python scripts/verify_task_body.py --help | grep -c '^[0-9]\+\.'   # = 11
uv run python -c "from scripts import verify_task_body; print(len(verify_task_body.CHECKS))"   # = 11
uv run pytest tests/test_verify_task_body.py -v   # all PASS

# Phase D acceptance
grep -c 'verify_task_body' .claude/agents/clean-result-critic.md .claude/agents/analyzer.md   # ≥ 4
grep -c 'verify_sagan_card' .claude/agents/clean-result-critic.md .claude/agents/analyzer.md  # = 0 (legacy refs cleaned up)

# Phase E acceptance
uv run python scripts/task.py migrate-body --report   # all 31 tasks PASS or are explicitly grandfathered
for n in $(ls tasks/awaiting_promotion/); do
  body="tasks/awaiting_promotion/$n/body.md"
  if grep -q 'legacy-sagan-card' "$body"; then
    uv run python scripts/verify_sagan_card.py "$body" || exit 1
  else
    uv run python scripts/verify_task_body.py --file "$body" || exit 1
  fi
done
# Phase E idempotency double-apply
uv run python scripts/task.py migrate-body --all --apply   # first apply
git status --porcelain tasks/ > /tmp/diff1
uv run python scripts/task.py migrate-body --all --apply   # second apply (must be no-op)
git status --porcelain tasks/ > /tmp/diff2
diff /tmp/diff1 /tmp/diff2   # exit 0
# v4-legacy bodies (#186, #237, #351) are EXPECTED to pass verify_task_body.py
# after in-place conversion; they are NOT routed to verify_clean_result.py.

# Phase F acceptance
uv run pytest tests/test_workflow_lint.py tests/test_workflow_yaml.py tests/test_verify_task_body.py -v   # all PASS
uv run python scripts/workflow_lint.py --check-references   # exit 0
uv run python scripts/task.py view --rich 333   # exit 0
test "$(uv run python scripts/task.py view --rich 333 | wc -l)" -le 60   # ≤ 60 lines
uv run python scripts/task.py view --rich 333 | grep -cE 'Status:|Frontmatter|Body excerpt|Last [0-9]+ events'  # ≥ 4
```

Plus a Phase 0 pre-flight acceptance recorded as DONE:
```bash
# Phase 0 pre-flight (already landed in commit 15dc06fe)
git log --grep='workflow.py: relax schema' --oneline | head -1 | grep -c '15dc06fe'   # = 1
uv run python scripts/workflow_lint.py   # exit 0 today (schema permissive)
```

A reader opening `.claude/skills/issue/SKILL.md` end-to-end finds:
- the State Machine ASCII diagram (currently absent);
- the full Step 0 → Step 10d procedure with sub-steps and exit-kind tables (currently absent);
- the `render_title()` helper pseudocode (currently absent);
- the active-vs-awaiting status user-action table (currently absent — was AUTO-GENERATED from workflow.yaml);
- the Resume-semantics table mapping `(status, marker-state) → action` (currently absent);
- the Cost-and-safety-rails section (currently absent);
- the Error-handling table (currently absent);
- explicit substrate references throughout: `task.py post-marker`, `tasks/<status>/<N>/`, `tasks/<status>/<N>/plans/v<K>.md`.

A reader running `verify_task_body.py --help` sees 11 numbered checks
matching the Sagan-card content discipline, expressed for markdown.

A reader running `verify_clean_result.py` against a legacy v4 markdown
body still passes (we don't touch the v4 codepath).
