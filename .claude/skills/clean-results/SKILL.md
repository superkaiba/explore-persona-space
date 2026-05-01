---
name: clean-results
description: >
  Transform messy experiment outputs (epm:results markers, draft write-ups,
  raw eval JSONs, figures) into a polished clean-result GitHub issue in the
  project board's Clean Results column. Applies advice from Neel Nanda,
  Ethan Perez, James Chua, John Hughes, and Owain Evans on mentor-grade
  research communication. Invoke with `/clean-results <source-issue-N>`
  or `/clean-results <draft-path>`.
user_invocable: true
---

# Clean Results

**Goal:** take messy experiment output and produce a single, self-contained,
mentor-grade clean-result presentation. The output is a GitHub issue body
(labeled `clean-results`, titled `<claim summary> (HIGH|MODERATE|LOW confidence)`,
moved to the `Clean Results` column on the Experiment Queue project board).

**This is the project's single mentor-facing output format.** The `analyzer`
agent and this skill share one template (`template.md`). An analyzer's draft
posted on an issue and a full clean-result GitHub issue differ only in
(a) where they live and (b) whether `scripts/verify_clean_result.py` has
been run.

**Reference exemplar:**
- issue **#75** (`Weak evidence that evil-persona capability coupling reduces post-EM capability (LOW confidence)`) — **preferred structure** (4-subsection TL;DR with takeaways + confidence folded into Results; Detailed report without Decision Log / Caveats H2s). Match this shape for every new clean result.
- Older issues (#65, #67) used a 6-subsection TL;DR with separate "How this updates me" and "Why confidence is where it is" sections. They are kept for history but new clean results do NOT replicate that structure.

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

## Scope

The `analyzer` agent owns the single-experiment clean-result pipeline
end-to-end (see `.claude/agents/analyzer.md` and `/issue` Step 7a).
It reads raw results, writes the clean-result body, runs the verifier,
and creates the clean-result GitHub issue directly. No manual
invocation of this skill is needed for a standard experiment.

This skill exists for **manual consolidation** and for edge cases the
analyzer does not cover.

**Forms:**
- `/clean-results <N1>,<N2>,<N3>` — consolidate multiple source issues
  into one combined clean-result (e.g. a sweep spanning #28, #46, pilot
  issues). This is the primary manual use case.
- `/clean-results promote <draft-N>` — move a `clean-results:draft` issue
  to `clean-results` + the final `Clean Results` column. Mainly useful when
  a reviewer verdict landed out-of-band and the automatic promotion in
  `/issue` Step 7b didn't fire.
- `/clean-results edit <N>` — regenerate the body of an existing
  clean-result issue from updated source data (e.g. after a fix on a
  source issue's `epm:results`). Preserves the issue number.

The single-issue `/clean-results <N>` form is rarely needed now — the
analyzer already produced the issue at Step 7a of `/issue`. Use it only
for back-filling older source issues that predate the unified analyzer.

---

## Required output structure (high level)

Match `template.md` exactly. The shape is:

```
## TL;DR
    ### Background
    ### Methodology
    ### Results                          ← hero figure + Main takeaways + Confidence line live HERE
    ### Next steps

---

# Detailed report
    ## Source issues
    ## Setup & hyper-parameters          ← absorbs Reproducibility Card + "why this experiment" prose
    ## WandB
    ## Sample outputs
    ## Headline numbers                  ← standing caveats listed inline after the table
    ## Artifacts
```

**Three invariants** (the verifier enforces the rest):

1. **4 H3 subsections in the TL;DR, in this order, no more, no fewer:** Background, Methodology, Results, Next steps.
2. **Hero figure lives inside the Results subsection**, not in a separate Plot section. Do not duplicate it in the Detailed report.
3. **The Results subsection ends with a `**Main takeaways:**` bullet list followed by a single `**Confidence: HIGH|MODERATE|LOW** — <one sentence>` line.** Each takeaway bullet bolds the load-bearing claim + numbers and continues with the belief update in plain prose — do NOT use an explicit `*Updates me:*` label.
4. **The issue title ends with `(HIGH confidence)`, `(MODERATE confidence)`, or `(LOW confidence)`.** Title = claim + confidence marker.

---

## Workflow

### Step 1: Intake

Typical inputs: a GitHub issue number, a cached draft at
`.claude/cache/issue-<N>-clean-result.md`, or multiple issue numbers being
consolidated.

Read, in this order:
1. The plan (`epm:plan` marker on the source issue, or `.claude/plans/issue-N.md`).
2. The results (`epm:results`) and analyzer draft (`epm:analysis`).
3. Any cached draft at `.claude/cache/issue-<N>-clean-result.md`.
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

The verifier checks: TL;DR structure (4 subsections in order), hero figure
commit-pinning, Main-takeaways block + single Confidence line, numeric
prose ↔ JSON cross-check, reproducibility-card completeness, confidence-
phrasebook consistency, forbidden stats-framing language, title confidence
marker `(HIGH|MODERATE|LOW confidence)` matching the Results line.

### Step 6: Post

```bash
gh issue create \
  --repo superkaiba/explore-persona-space \
  --title "<concise claim summary> (<HIGH|MODERATE|LOW> confidence)" \
  --label clean-results \
  --label "aim:<N>-<name>" \
  --label "type:experiment" \
  --label "compute:<size>" \
  --body-file <path-to-body.md>

uv run python scripts/gh_project.py set-status <new-N> "Clean Results"
```

On each source issue referenced in `## Source issues`, post a one-liner:
> Distilled into clean result: #<new-issue-N>.

**Do NOT close source issues.** They stay in their Done column. The clean
result is the mentor-facing artifact; source issues remain the audit trail.

---

## When to use this skill

- Consolidating a cluster of related source issues (e.g., a sweep spanning
  #28, #46, pilot issues) into one presentable result.
- Back-filling a clean-result for an older source issue that predates the
  unified analyzer pipeline.
- Promoting a `clean-results:draft` issue to final when the automatic
  promotion in `/issue` Step 7b didn't fire.

## When NOT to use this skill

- For a single experiment that ran through `/issue`. The analyzer already
  created the clean-result issue at Step 7a; don't duplicate.
- When the result isn't ready. If the answer to "what's the one claim" is
  "several conflicting things," run more experiments first.
- Infra / code-change issues. Those close via `status:done-impl`; they're
  not claims about the world.
