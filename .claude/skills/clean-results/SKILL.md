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

**Goal:** take messy, possibly-multi-issue experiment output and produce a
single, self-contained, mentor-grade "Clean Result" presentation. The output
is a GitHub issue body (labeled `clean-results`, titled `[Clean Result] ...`,
moved to the `Clean Results` column on the Experiment Queue project board).

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
- **`checklist.md`** — pre-publish verification list. Run before posting.

This file (`SKILL.md`) only covers the WORKFLOW — what to do, in what order,
when invoked. It does not duplicate the principles, the template, or the
checklist.

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
    ## Setup & hyper-parameters
    ## WandB
    ## Sample outputs
    ## Headline numbers
    ## Artifacts
    ## Caveats
```

**Three invariants** (checklist enforces the rest):

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

Criteria live in `principles.md` (Chua/Hughes chart rules) and
`checklist.md` §3. One figure, labeled axes with direction, error bars where
applicable, ≤ 3-5 colors, readable on a video call. Commit at
`figures/<experiment>/<name>.{png,pdf}` and use a raw GitHub URL pinned to a
specific commit — never `main`, never relative.

If no single figure carries the claim, you haven't distilled hard enough
(Step 2).

### Step 4: Write the body

Fill in `template.md`. For confidence calibration, phrasing, hedges, figure
rules, reproducibility card completeness — defer to `checklist.md`. This
step is about drafting; Step 5 is about verification.

### Step 5: Run the pre-publish checklist

Open `checklist.md` and tick every item, or document an exception in the
body's Caveats section. Do not proceed to Step 6 until the checklist is
clean.

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
uv run python scripts/gh_project.py set-status <N> "Clean Results"
```

(If the column doesn't exist, add it in the GitHub UI first.)

On each source issue referenced in `## Source issues`, post a one-liner:
> Distilled into clean result: #<new-issue-N>.

**Do NOT close source issues.** They stay in their Done column. The clean
result is the mentor-facing artifact; source issues remain the audit trail.

---

## When to use this skill

- One or more experiments reach `status:done-experiment` and the finding is
  worth showing a mentor.
- Consolidating a cluster of related issues (e.g., a sweep spanning
  #28, #46, pilot issues) into one presentable result.
- Before a mentor meeting — the clean-result issue becomes the single source
  of truth to screen-share; no separate deck needed.

## When NOT to use this skill

- Result isn't ready. If the answer to "what's the one claim" is "several
  conflicting things," run more experiments first or publish as
  `status:under-review` with a `CONCERNS` flag.
- Infra / code-change issues. Those close via `status:done-impl`; they're
  not claims about the world.
- A single small experiment well-presented in its own `epm:results` comment.
  Cleaning adds overhead; reserve it for findings that will be cited later.
