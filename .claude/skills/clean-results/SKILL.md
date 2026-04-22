---
name: clean-results
description: >
  Transform messy experiment outputs (epm:results markers, draft write-ups,
  raw eval JSONs, figures) into a polished "Clean Result" GitHub issue in the
  project board's Clean Results column. Applies advice from Neel Nanda,
  Ethan Perez, James Chua, John Hughes, and Owain Evans on mentor-grade
  research communication. Invoke with `/clean-results <source-issue-N>`
  or `/clean-results <draft-path>`.
user_invocable: true
---

# Clean Results

**Goal:** take messy experiment output and produce a single, self-contained,
mentor-grade "Clean Result" presentation. The output is a GitHub issue body
(labeled `clean-results`, titled `[Clean Result] ...`, moved to the
`Clean Results` column on the Experiment Queue project board).

**This is the project's single mentor-facing output format.** The `analyzer`
agent and this skill share one template (`template.md`). An analyzer's draft
posted on an issue and a full `[Clean Result]` GitHub issue differ only in
(a) where they live and (b) whether `scripts/verify_clean_result.py` has
been run.

**Reference exemplars:**
- issue #67 (`[Clean Result] 25% Tulu midtrain matrix …`) — **preferred structure** (6-subsection TL;DR + `# Detailed report`). Match this.
- issue #65 (`[Clean Result] Single-[ZLT]-token sweep`) — older two-part TL;DR; kept for reference. New clean results use #67.

---

## Companion files (READ FIRST)

- **`principles.md`** — distilled research-communication advice from Nanda,
  Perez, Chua, Hughes, Muehlhauser, Evans. Read once at the start of a
  clean-results session. All the "why" lives here.
- **`template.md`** — the literal skeleton to fill in. Structural questions
  ("what goes in Background?") go here.
- **`checklist.md`** — pre-publish verification list. Automated by
  `scripts/verify_clean_result.py`; run before posting.

This file (`SKILL.md`) only covers the WORKFLOW — what to do, in what order,
when invoked. It does not duplicate the principles, the template, or the
checklist.

---

## Modes

### Mode A — Auto-draft (invoked by `/issue` at `/signoff`)

**Trigger:** `/issue <N>` reaches Step 8 with `type:experiment` +
`compute:medium|large` AND a reviewer PASS (see `.claude/skills/issue/SKILL.md`).
Fires automatically; user does not run this manually.

**Input:** the source issue's `epm:plan`, `epm:results`, `epm:analysis`,
and `epm:reviewer-verdict` marker comments. The analyzer's `epm:analysis`
body is already in clean-results-template format, so Mode A is primarily a
publish-and-link step, not a re-authoring step.

**Output:** a NEW GitHub issue:
- Title: `[Clean Result] <auto-derived from epm:analysis TL;DR>`
- Labels: `clean-results:draft`, `aim:<copied>`, `type:experiment`, `compute:<copied>`
- Body: the analyzer draft (lightly edited for cross-issue references)
- Project column: `Clean Results (draft)` (created via `scripts/gh_project.py set-status`
  if missing; falls back to `Clean Results` + label-only gating)

**Post-processing:**
- Source issue receives a comment: `Auto-drafted clean result: #<new-N>. Run /clean-results promote <new-N> to finalize.`
- The draft is NOT moved to `Clean Results` until the user promotes it.

**Skip conditions** (Mode A is a no-op when):
- `compute:small` — overhead not justified for small experiments
- `type:infra`, `type:analysis`, `type:survey` — not claims about the world
- Reviewer verdict was FAIL — draft isn't ready to surface

### Mode B — Manual (`/clean-results <args>`)

**Forms:**
- `/clean-results <source-N>` — distill one issue (typical use)
- `/clean-results <N1>,<N2>,<N3>` — consolidate multiple issues into one result
- `/clean-results promote <draft-N>` — move a Mode-A draft from
  `Clean Results (draft)` to `Clean Results`, re-labeling `clean-results:draft`
  → `clean-results`
- `/clean-results <draft-path>` — distill from a local draft file (rarely
  needed now that analyzer writes clean-results format directly)

Output: same as Mode A, but posts directly to the `Clean Results` column
with the final `clean-results` label.

---

## Required output structure (high level)

Match `template.md` exactly. The shape is:

```
## TL;DR
    ### Background
    ### Methodology
    ### Results                          ← hero figure lives HERE
    ### How this updates me + confidence
    ### Why confidence is where it is
    ### Next steps

---

# Detailed report
    ## Source issues
    ## Setup & hyper-parameters          ← absorbs Reproducibility Card
    ## WandB
    ## Sample outputs
    ## Headline numbers
    ## Artifacts
    ## Decision Log
    ## Caveats
```

**Three invariants** (the verifier enforces the rest):

1. **6 H3 subsections in the TL;DR, in this order, no more, no fewer.**
2. **Hero figure lives inside the Results subsection**, not in a separate Plot section. Do not duplicate it in the Detailed report.
3. **"Why confidence is where it is" mirrors the confidence tags** from the preceding "How this updates me" bullets. Every HIGH / MODERATE / LOW tag gets a corresponding item explaining why.

---

## Workflow

### Step 1: Intake

Typical inputs: a GitHub issue number, a draft path in `research_log/drafts/`,
or multiple issue numbers being consolidated.

Read, in this order:
1. The plan (`epm:plan` marker on the source issue, or `.claude/plans/issue-N.md`).
2. The results (`epm:results`) and analyzer draft (`epm:analysis`).
3. Any `research_log/drafts/*.md` referenced.
4. The actual `eval_results/*/run_result.json` and `figures/*`.
5. WandB run URLs.

**Never trust the draft's prose for numbers — pull numbers from JSON.** The
common failure mode: draft says 92%, JSON says 89%. Check.

### Step 2: Distill the claim

Before writing anything, answer:

1. What is the ONE thing the mentor should walk away believing?
2. What is the key number that backs it?
3. What is the single strongest alternative explanation, and does the evidence rule it out?
4. If the mentor reads only the TL;DR, is the answer to (1) unambiguous?

If any answer is shaky: stop, fix the experiment / add a caveat / weaken the
claim before writing more.

### Step 3: Choose the hero figure

Use the `paper-plots` skill to build / regenerate the hero figure. The
skill enforces colorblind-safe palette, error bars, direction arrows,
commit-pinned `.meta.json` sidecar, PNG + PDF dual save. Do not build
figures with ad-hoc `plt.rcParams.update(...)`; always go through
`src/explore_persona_space/analysis/paper_plots.set_paper_style()`.

One figure. Labeled axes with direction. Error bars. ≤ 3-5 colors.
Readable on a video call. Committed at `figures/<experiment>/<name>.{png,pdf}`.
Linked with a raw GitHub URL pinned to a specific commit — never `main`,
never relative.

If no single figure carries the claim, you haven't distilled hard enough
(Step 2). Rerun Step 2.

### Step 4: Write the body

Fill in `template.md`. For confidence calibration, phrasing, hedges, figure
rules, reproducibility card completeness — defer to `checklist.md`. This
step is about drafting; Step 5 is about verification.

### Step 5: Run the pre-publish verifier

```bash
uv run python scripts/verify_clean_result.py <path-to-body.md>
# or for an already-posted issue:
uv run python scripts/verify_clean_result.py --issue <N>
```

Every FAIL must be fixed. WARNs should be fixed or acknowledged in the body's
Caveats section. Do not proceed to Step 6 until the verifier is clean.

The verifier checks: TL;DR structure (6 subsections in order), hero figure
commit-pinning, confidence-tag mirroring, numeric prose ↔ JSON cross-check,
reproducibility-card completeness, confidence-phrasebook consistency,
support-type tags on every How-updates-me bullet, title prefix.

### Step 6: Post

```bash
gh issue create \
  --repo superkaiba/explore-persona-space \
  --title "[Clean Result] <concise descriptive title>" \
  --label clean-results \
  --label "aim:<N>-<name>" \
  --label "type:experiment" \
  --label "compute:<size>" \
  --body-file <path-to-body.md>
```

Then:

```bash
# Mode A: use "Clean Results (draft)" and label clean-results:draft
uv run python scripts/gh_project.py set-status <N> "Clean Results (draft)"
# Mode B / promotion: final column
uv run python scripts/gh_project.py set-status <N> "Clean Results"
```

(If a column doesn't exist, add it in the GitHub UI first.)

On each source issue referenced in `## Source issues`, post a one-liner:
> Distilled into clean result: #<new-issue-N>.

**Do NOT close source issues.** They stay in their Done column. The clean
result is the mentor-facing artifact; source issues remain the audit trail.

---

## When to use this skill

- Automatic: every `type:experiment` + `compute:medium|large` sign-off
  fires Mode A.
- Manual: consolidating a cluster of related issues (e.g., a sweep spanning
  #28, #46, pilot issues) into one presentable result.
- Before a mentor meeting — the clean-result issue is the single source of
  truth to screen-share (see `/mentor-prep` for multi-result assembly).

## When NOT to use this skill

- Result isn't ready. If the answer to "what's the one claim" is "several
  conflicting things," run more experiments first or publish as
  `status:under-review` with a `CONCERNS` flag.
- Infra / code-change issues. Those close via `status:done-impl`; they're
  not claims about the world.
- A single small experiment well-presented in its own `epm:results` comment.
  Cleaning adds overhead; reserve it for `compute:medium|large`.
